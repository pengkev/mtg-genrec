"""Mtgtop8 scraping support."""

from __future__ import annotations

import argparse
import logging
import re
import requests
import sys
import time
from bs4 import BeautifulSoup
from datetime import date, datetime
from mtgdeck.data import BOARD_NAMES, clean_board, decklist_card_names, make_decklist_record, normalize_deck_cards, positive_quantity
from scrape.http import PaginationError, ScrapeHTTPError, request_with_retries, validate_page_fingerprint
from scrape.sources import EXPORT_SECTION_HEADERS, MTGTOP8, MTGTOP8_FORMATS
from scrape.state import Checkpoint, CorpusOutputs, finish_page
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse


def make_mtgtop8_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = (
        "mtg-deck-evaluator/2.0 (+https://github.com/pengkev/mtg-deck-evaluator)"
    )
    return session


def mtgtop8_search_records(html: str) -> list[dict[str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    records: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for row in soup.select(".hover_tr"):
        for tag in row.find_all("a", href=True):
            href = str(tag["href"])
            if parse_qs(urlparse(href).query).get("d") and href not in seen:
                seen.add(href)
                date_match = re.search(r"\b\d{2}/\d{2}/\d{2}\b", row.get_text(" ", strip=True))
                records.append(
                    {
                        "url": urljoin(MTGTOP8, href),
                        "date": date_match.group(0) if date_match else None,
                    }
                )
                break
    return records


def mtgtop8_export_url(page_html: str) -> str | None:
    """Extract the current /dec?... export URL from an MTGTop8 event page."""

    soup = BeautifulSoup(page_html, "html.parser")
    fallback: str | None = None
    for tag in soup.find_all("a", href=True):
        href = str(tag["href"])
        path = "/" + urlparse(href).path.lstrip("/")
        if path == "/dec":
            return urljoin(MTGTOP8, href)
        if path == "/mtgo":
            fallback = urljoin(MTGTOP8, href)
    return fallback


def parse_mtgo_decklist(text: str, format_name: str) -> dict[str, list[dict[str, Any]]]:
    """Parse quantities and zones from MTGTop8's MTGO text export."""

    boards: dict[str, list[dict[str, Any]]] = {zone: [] for zone in BOARD_NAMES}
    current_zone = "mainboard"
    commander_format = format_name in {
        "commander",
        "cedh",
        "duel-commander",
        "mtgo-commander",
        "pauper-commander",
    }
    for raw_line in text.replace("\r\n", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        sideboard_prefix = re.match(r"^SB:\s*(.+)$", line, re.IGNORECASE)
        if sideboard_prefix:
            current_zone = "commanders" if commander_format else "sideboard"
            line = sideboard_prefix.group(1).strip()
        header = line.casefold().rstrip(":")
        if header in EXPORT_SECTION_HEADERS:
            if header in {"deck", "mainboard"}:
                current_zone = "mainboard"
            elif header == "sideboard":
                # MTGTop8 historically emits Commander cards after this marker.
                current_zone = "commanders" if commander_format else "sideboard"
            elif header in {"commander", "commanders"}:
                current_zone = "commanders"
            elif header in {"companion", "companions"}:
                current_zone = "companions"
            continue
        match = re.match(r"^\s*(\d+)x?\s+(?:\[[^\]]*\]\s*)?(.+?)\s*$", line)
        if match:
            boards[current_zone].append(
                {"name": match.group(2), "quantity": positive_quantity(match.group(1))}
            )
    return boards


def mtgtop8_seed_searches(date_start: str) -> list[dict[str, Any]]:
    start = datetime.strptime(date_start, "%d/%m/%Y").date()
    return [
        {"key": f"year:{year}:{max(start, date(year, 1, 1)).isoformat()}",
         "params": {"date_start": max(start, date(year, 1, 1)).strftime("%d/%m/%Y"),
                    "date_end": date(year, 12, 31).strftime("%d/%m/%Y")}}
        for year in range(date.today().year, start.year - 1, -1)
    ]


def validate_mtgtop8_search(html: str, format_code: str, page: int) -> int:
    soup = BeautifulSoup(html, "html.parser")
    selected = soup.select_one("select[name=format] option[selected]")
    current = soup.select_one("input[name=current_page]")
    total = re.search(r"([\d,]+)\s+decks matching", soup.get_text(" ", strip=True))
    if selected is None or selected.get("value") != format_code:
        raise PaginationError("MTGTop8 search did not honor the format filter")
    if current is None or int(current.get("value") or 1) != page:
        raise PaginationError("MTGTop8 search returned an unexpected page")
    if total is None:
        raise PaginationError("MTGTop8 search is missing its result count")
    return int(total.group(1).replace(",", ""))


def parse_event_metadata(html: str, deck_id: str) -> tuple[int | None, int | None]:
    soup = BeautifulSoup(html, "html.parser")
    text = " ".join(soup.stripped_strings)
    players_match = re.search(r"(\d+)\s+players?\b", text, re.IGNORECASE)
    placement = None
    for row in soup.select(".chosen_tr, .hover_tr"):
        link = row.find("a", href=True)
        if link and parse_qs(urlparse(link["href"]).query).get("d") == [deck_id]:
            field = row.find(class_="S14")
            match = re.match(r"\s*(\d+)", field.get_text() if field else "")
            placement = int(match.group(1)) if match else None
            break
    return placement, int(players_match.group(1)) if players_match else None



def collect_mtgtop8_format(
    format_name: str, outputs: CorpusOutputs, checkpoint: Checkpoint,
    valid_names: set[str], args: argparse.Namespace, session: requests.Session | None = None,
) -> tuple[int, int, int]:
    checkpoint.add_searches("mtgtop8", format_name, mtgtop8_seed_searches(args.date_start))
    session = session or make_mtgtop8_session()
    fetched = combined_appended = format_appended = known = 0
    for _ in range(args.max_pages_per_format or sys.maxsize):
        job = checkpoint.current_search("mtgtop8", format_name)
        if job is None or (args.limit_per_format is not None and format_appended >= args.limit_per_format):
            break
        page = int(job["next_page"])
        response = request_with_retries(
            session, "POST", f"{MTGTOP8}/search",
            data={**job["params"], "current_page": str(page), "format": MTGTOP8_FORMATS[format_name],
                  "compet_check[P]": "1", "compet_check[M]": "1", "compet_check[C]": "1",
                  "compet_check[R]": "1", "MD_check": "1"},
            timeout=args.timeout, retries=args.retries,
        )
        total = validate_mtgtop8_search(response.text, MTGTOP8_FORMATS[format_name], page)
        records = mtgtop8_search_records(response.text)
        if not records:
            if total > (page - 1) * 25:
                raise PaginationError("MTGTop8 reported matches but no deck rows could be parsed")
            finish_page(outputs, checkpoint, "mtgtop8", format_name, job, complete=True)
            time.sleep(args.mtgtop8_delay)
            continue
        fingerprint = validate_page_fingerprint(job, (str(record["url"]) for record in records))
        page_complete = True
        for index, record in enumerate(records):
            deck_url = str(record["url"])
            source_id = parse_qs(urlparse(deck_url).query)["d"][0]
            if outputs.has_source(format_name, "mtgtop8", source_id):
                known += 1
                continue
            # The public export accepts a deck ID alone; f only names the file.
            # This saves an event-page request for every deck.
            export_url = f"{MTGTOP8}/dec?d={source_id}"
            event = None
            try:
                time.sleep(args.mtgtop8_delay)
                export = request_with_retries(session, "GET", export_url,
                                              timeout=args.timeout, retries=args.retries)
                parsed = parse_mtgo_decklist(export.text, format_name)
                if not parsed["mainboard"]:
                    time.sleep(args.mtgtop8_delay)
                    event = request_with_retries(session, "GET", deck_url,
                                                 timeout=args.timeout, retries=args.retries)
                    fallback = mtgtop8_export_url(event.text)
                    if not fallback or fallback == export_url:
                        raise ValueError(f"No valid MTGTop8 export for {source_id}")
                    export_url = fallback
                    time.sleep(args.mtgtop8_delay)
                    export = request_with_retries(session, "GET", export_url,
                                                  timeout=args.timeout, retries=args.retries)
                    parsed = parse_mtgo_decklist(export.text, format_name)
                    if not parsed["mainboard"]:
                        raise ValueError(f"Invalid MTGTop8 export for {source_id}")
            except ScrapeHTTPError as exc:
                if exc.status not in (404, 410):
                    raise
                logging.warning("[mtgtop8/%s] unavailable deck %s; continuing", format_name, source_id)
                outputs.mark_source(format_name, "mtgtop8", source_id)
                continue
            except ValueError as exc:
                logging.warning(
                    "[mtgtop8/%s] permanently skipping malformed deck %s: %s",
                    format_name,
                    source_id,
                    exc,
                )
                outputs.mark_source(format_name, "mtgtop8", source_id)
                continue
            event_metadata = {}
            if getattr(args, "mtgtop8_event_metadata", False) and event is None:
                time.sleep(args.mtgtop8_delay)
                event = request_with_retries(session, "GET", deck_url,
                                             timeout=args.timeout, retries=args.retries)
            if event is not None:
                placement, players = parse_event_metadata(event.text, source_id)
                event_metadata = {"placement": placement, "players": players}
            fetched += 1
            boards = {zone: clean_board(parsed[zone], valid_names) for zone in BOARD_NAMES}
            decklist = make_decklist_record(
                source="mtgtop8", source_id=source_id, format_name=format_name,
                url=deck_url, deck_date=record.get("date"), boards=boards,
                metadata={"export_url": export_url, **event_metadata},
            )
            cards = normalize_deck_cards(decklist_card_names(decklist), valid_names, args.min_cards)
            if cards:
                combined_added, format_added = outputs.append(format_name, cards, decklist)
                combined_appended += int(combined_added)
                format_appended += int(format_added)
            else:
                outputs.mark_source(format_name, "mtgtop8", source_id)
            if args.limit_per_format is not None and format_appended >= args.limit_per_format:
                page_complete = index == len(records) - 1
                break
        if page_complete:
            job["next_page"] = page + 1
            job["last_fingerprint"] = fingerprint
        finish_page(outputs, checkpoint, "mtgtop8", format_name, job,
                    complete=page_complete and page * 25 >= total)
        logging.info("[mtgtop8/%s] %s page %s; fetched %s; known %s; combined +%s; format +%s",
                     format_name, job["key"], page, fetched, known, combined_appended, format_appended)
        time.sleep(args.mtgtop8_delay)
    return fetched, combined_appended, format_appended
