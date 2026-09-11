"""Deckbox scraping support."""

from __future__ import annotations

import argparse
import json
import logging
import re
import requests
import sys
import time
from bs4 import BeautifulSoup
from collections.abc import Iterable, Iterator, Mapping
from itertools import combinations
from mtgdeck.data import BOARD_NAMES, clean_board, decklist_card_names, make_decklist_record, normalize_card_name, normalize_deck_cards, positive_quantity
from scrape.http import PaginationError, ScrapeHTTPError, request_with_retries, validate_page_fingerprint
from scrape.sources import DECKBOX, DECKBOX_CORPUS, DECKBOX_FORMATS, DECKBOX_FORMAT_NAMES
from scrape.state import Checkpoint, CorpusOutputs, finish_page
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse


def make_deckbox_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "mtg-deck-evaluator/2.0 "
                "(+https://github.com/pengkev/mtg-deck-evaluator)"
            ),
            "Referer": f"{DECKBOX}/decks/mtg",
        }
    )
    return session


def deckbox_search_records(
    html: str, expected_format: str | None = None
) -> list[dict[str, Any]]:
    """Parse public constructed-deck rows from Deckbox's deck database."""

    soup = BeautifulSoup(html, "html.parser")
    canonical_formats = {
        label.casefold(): format_name
        for format_name, label in DECKBOX_FORMAT_NAMES.items()
    }
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in soup.select("#users_list_container table.simple_table tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 4:
            continue
        link = cells[0].find("a", href=True)
        if link is None:
            continue
        path = urlparse(str(link["href"])).path
        match = re.fullmatch(r"/sets/(\d+)", path.rstrip("/"))
        if not match:
            continue
        source_id = match.group(1)
        if source_id in seen:
            continue
        format_label = cells[3].get_text(" ", strip=True)
        format_name = canonical_formats.get(format_label.casefold())
        if format_name is None or (
            expected_format is not None and format_name != expected_format
        ):
            continue
        seen.add(source_id)
        date_tag = cells[-1].find("span")
        user_link = cells[1].find("a") if len(cells) > 1 else None
        icon = link.find("img")
        icon_classes = set(icon.get("class", [])) if icon is not None else set()
        records.append(
            {
                "source_id": source_id,
                "url": urljoin(DECKBOX, path),
                "name": link.get_text(" ", strip=True),
                "format": format_name,
                "date": (
                    date_tag.get_text(" ", strip=True)
                    if date_tag is not None
                    else cells[-1].get_text(" ", strip=True)
                ),
                "user": user_link.get_text(" ", strip=True) if user_link else None,
                "built": "s_brick" in icon_classes,
            }
        )
    return records


def deckbox_next_page_url(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    controls = soup.select_one("#users_list_container .pagination_controls")
    if controls is None:
        return None
    for tag in controls.find_all("a", href=True):
        url = urljoin(DECKBOX, str(tag["href"]))
        parsed = urlparse(url)
        if (tag.get_text(" ", strip=True).casefold() == "next"
                and parsed.netloc == urlparse(DECKBOX).netloc
                and parsed.path.rstrip("/") == "/decks/mtg"):
            return url
    return None


def deckbox_boards(
    html: str, valid_names: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """Parse mainboard, commander, and sideboard rows from a Deckbox deck page."""

    soup = BeautifulSoup(html, "html.parser")
    raw: dict[str, list[dict[str, Any]]] = {zone: [] for zone in BOARD_NAMES}
    for table_zone, selector in (
        ("mainboard", "table.set_cards.main"),
        ("sideboard", "table.set_cards.sideboard"),
    ):
        table = soup.select_one(selector)
        if table is None:
            continue
        for row in table.find_all("tr"):
            count = row.find("td", class_="card_count")
            card = row.find("td", class_="card_name")
            link = card.find("a") if card is not None else None
            if count is None or link is None:
                continue
            zone = (
                "commanders"
                if table_zone == "mainboard" and row.get("data-is-commander") == "1"
                else table_zone
            )
            raw[zone].append(
                {
                    "name": link.get_text(" ", strip=True),
                    "quantity": positive_quantity(count.get_text(" ", strip=True)),
                }
            )
    return {zone: clean_board(raw[zone], valid_names) for zone in BOARD_NAMES}


def deckbox_seed_searches(format_name: str) -> Iterator[dict[str, Any]]:
    base = f"33{DECKBOX_FORMATS[format_name]}"
    # The public client encodes exact colors as 2a<IDs joined by dots>.
    colored = [".".join(group) for size in range(1, 6)
               for group in combinations("12345", size)]
    # Put W/U/B/R/G and the first multicolor slice ahead of colorless. With
    # the default six-group threshold, card selection therefore cannot begin
    # from monocolor/colorless decks alone.
    colors = [*colored[:6], "6", *colored[6:]]
    # Interleave colors within each sort so an early interrupted run still
    # samples colorless, monocolor, and multicolor decks.
    for sort_name, sort_code in (
        ("updated", "c"),
        ("views", "b"),
        ("stars", "a"),
        ("name", "n"),
    ):
        for order in ("d", "a"):
            for color in (None, *colors):
                yield {
                    "key": f"colors:{color or 'all'}:{sort_name}:{order}",
                    "params": {
                        "f": base + (f"!2a{color}" if color else ""),
                        "s": sort_code,
                        "o": order,
                    },
                }


def deckbox_sample_color(job: Mapping[str, Any]) -> str:
    encoded = str(job["key"]).split(":", 2)[1]
    return {
        "1": "W", "2": "U", "3": "B", "4": "R", "5": "G", "6": "colorless",
    }.get(encoded, "multicolor" if "." in encoded else "all")


def deckbox_card_candidates(
    html: str, oracle_profiles: Mapping[str, Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Extract canonical nonland card IDs and their balancing metadata."""

    # Parse the two JSON arguments; never execute JavaScript from a deck page.
    soup = BeautifulSoup(html, "html.parser")
    cards = {}
    for script in soup.find_all("script"):
        match = re.search(r"new\s+Tcg\.MtgDeck\(", script.get_text())
        if match is None:
            continue
        args = script.get_text()[match.end():].lstrip()
        decoder = json.JSONDecoder()
        try:
            _, offset = decoder.raw_decode(args)
            cards, _ = decoder.raw_decode(args[offset:].lstrip().removeprefix(",").lstrip())
        except ValueError:
            continue
        break
    if not isinstance(cards, Mapping):
        return
    for row in soup.select("table.set_cards.main tr[data-id], table.set_cards.sideboard tr[data-id]"):
        card = cards.get(row["data-id"], {})
        if not isinstance(card, Mapping) or not str(card.get("id", "")).isdigit():
            continue
        cells = row.find_all("td", recursive=False)
        if len(cells) > 4 and re.search(r"\bland\b", cells[4].get_text(), re.I):
            continue
        name = str(card.get("name") or "").strip()
        profile = oracle_profiles.get(normalize_card_name(name))
        if not profile or profile.get("is_land"):
            continue
        yield {
            "card_id": str(card["id"]),
            "name": name,
            "color": profile["color"],
            "roles": tuple(profile["roles"]),
        }


def deckbox_card_searches(
    cards: Iterable[Mapping[str, Any]], format_name: str,
) -> Iterator[dict[str, Any]]:
    for card in cards:
        card_id = str(card["card_id"])
        for order in ("d", "a"):
            yield {
                "key": f"card:{card_id}:updated:{order}",
                "label": card.get("name"),
                "sample_popularity": card.get("deck_count"),
                "color": card.get("color"),
                "roles": list(card.get("roles", ())),
                "params": {
                    "f": f"33{DECKBOX_FORMATS[format_name]}!51{card_id}",
                    "s": "c",
                    "o": order,
                },
            }


def expand_capped_deckbox_search(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Add alternate views/stars/name windows for a capped card search."""
    if job.get("expanded") or not job["key"].startswith("card:"):
        return
    job["expanded"] = True
    prefix = job["key"].rsplit(":", 2)[0]
    base_params = {key: value for key, value in job["params"].items() if key not in ("s", "o")}
    for sort_name, sort_code in (("views", "b"), ("stars", "a"), ("name", "n")):
        for order in ("d", "a"):
            yield {
                "key": f"{prefix}:{sort_name}:{order}",
                "label": job.get("label"),
                "params": {**base_params, "s": sort_code, "o": order},
                "expanded": True,
            }


def add_balanced_deckbox_searches(
    format_name: str, outputs: CorpusOutputs, checkpoint: Checkpoint, args: argparse.Namespace,
) -> int:
    if getattr(args, "deckbox_discovery", "balanced") == "off":
        return 0
    selected = outputs.select_balanced_deckbox_cards(
        format_name,
        limit=getattr(args, "deckbox_card_count", 56),
        min_samples=getattr(args, "deckbox_card_sample_size", 300),
        min_color_buckets=getattr(args, "deckbox_card_color_buckets", 6),
        min_decks=getattr(args, "deckbox_card_min_decks", 2),
    )
    before = len(checkpoint.state[checkpoint.key("deckbox", format_name)]["searches"])
    checkpoint.add_searches("deckbox", format_name, deckbox_card_searches(selected, format_name))
    return len(checkpoint.state[checkpoint.key("deckbox", format_name)]["searches"]) - before


def validate_deckbox_search(response: requests.Response, page: int, params: Mapping[str, Any]) -> None:
    soup = BeautifulSoup(response.text, "html.parser")
    container = soup.select_one("#users_list_container")
    controls = soup.select_one("#users_list_container .pagination_controls")
    current = re.search(r"\bPage\s+(\d+)", controls.get_text(" ", strip=True)) if controls else None
    if container is None or current is None:
        raise PaginationError("Deckbox did not return its deck listing and pagination controls")
    if int(current.group(1)) != page:
        raise PaginationError(f"Deckbox returned page {current.group(1)} for page {page}")
    url = getattr(response, "url", None)
    if url and parse_qs(urlparse(url).query).get("f") != [params["f"]]:
        raise PaginationError("Deckbox redirected away from the requested filters")


def collect_deckbox_format(
    format_name: str, outputs: CorpusOutputs, checkpoint: Checkpoint,
    valid_names: set[str], args: argparse.Namespace, session: requests.Session | None = None,
) -> tuple[int, int, int]:
    checkpoint.add_searches("deckbox", format_name, deckbox_seed_searches(format_name))
    add_balanced_deckbox_searches(format_name, outputs, checkpoint, args)
    session = session or make_deckbox_session()
    fetched = combined_appended = format_appended = known = 0
    for _ in range(args.max_pages_per_format or sys.maxsize):
        job = checkpoint.current_search("deckbox", format_name)
        if job is None or (args.limit_per_format is not None and format_appended >= args.limit_per_format):
            break
        page = int(job["next_page"])
        try:
            # Follow the actual Next link, preserving all site-selected filters.
            request_args = {} if job.get("next_url") else {
                "params": {**job["params"], **({"p": page} if page > 1 else {})}
            }
            response = request_with_retries(
                session, "GET", job.get("next_url") or f"{DECKBOX}/decks/mtg",
                timeout=args.timeout, retries=args.retries, **request_args,
            )
            validate_deckbox_search(response, page, job["params"])
        except (ScrapeHTTPError, PaginationError) as exc:
            if page == 1 or (isinstance(exc, ScrapeHTTPError) and exc.status not in (404, 410)):
                raise
            logging.warning("[deckbox/%s] %s stopped at page %s: %s; moving to another search",
                            format_name, job["key"], page, exc)
            checkpoint.add_searches(
                "deckbox", format_name, expand_capped_deckbox_search(job)
            )
            finish_page(outputs, checkpoint, "deckbox", format_name, job, complete=True,
                        reason="pagination unavailable")
            time.sleep(args.deckbox_delay)
            continue
        records = deckbox_search_records(response.text, expected_format=format_name)
        next_url = deckbox_next_page_url(response.text)
        if not records:
            # Missing or changed markup must not silently mark a search complete.
            text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
            if not re.search(r"\b0 total results\b", text):
                raise PaginationError("Deckbox listing has no matching rows and no explicit zero-result count")
            finish_page(outputs, checkpoint, "deckbox", format_name, job, complete=True)
            time.sleep(args.deckbox_delay)
            continue
        try:
            fingerprint = validate_page_fingerprint(job, (str(row["source_id"]) for row in records))
        except PaginationError as exc:
            logging.warning("[deckbox/%s] %s; moving to another search", format_name, exc)
            finish_page(outputs, checkpoint, "deckbox", format_name, job, complete=True, reason="repeated page")
            time.sleep(args.deckbox_delay)
            continue
        if next_url:
            query = parse_qs(urlparse(next_url).query)
            if query.get("p") != [str(page + 1)] or query.get("f") != [job["params"]["f"]]:
                raise PaginationError("Deckbox Next link does not advance the same filtered search")
        page_complete = True
        for index, record in enumerate(records):
            source_id = str(record["source_id"])
            if outputs.has_source(DECKBOX_CORPUS, "deckbox", source_id):
                known += 1
                continue
            time.sleep(args.deckbox_delay)
            try:
                detail = request_with_retries(session, "GET", str(record["url"]),
                                              timeout=args.timeout, retries=args.retries)
            except ScrapeHTTPError as exc:
                if exc.status not in (404, 410):
                    raise
                logging.warning("[deckbox/%s] unavailable deck %s; continuing", format_name, source_id)
                outputs.mark_source(DECKBOX_CORPUS, "deckbox", source_id)
                continue
            if BeautifulSoup(detail.text, "html.parser").select_one("table.set_cards.main") is None:
                raise ValueError(f"Deckbox deck {source_id} is missing its mainboard table")
            fetched += 1
            boards = deckbox_boards(detail.text, valid_names)
            decklist = make_decklist_record(
                source="deckbox", source_id=source_id, format_name=format_name,
                url=str(record["url"]), name=str(record["name"]), deck_date=record.get("date"), boards=boards,
                metadata={"user": record.get("user"), "built": record.get("built")},
            )
            cards = normalize_deck_cards(decklist_card_names(decklist), valid_names, args.min_cards)
            if cards:
                if (getattr(args, "deckbox_discovery", "balanced") != "off"
                        and job["key"].startswith("colors:")):
                    outputs.record_deckbox_card_sample(
                        format_name,
                        source_id,
                        deckbox_sample_color(job),
                        deckbox_card_candidates(
                            detail.text, getattr(args, "oracle_profiles", {}),
                        ),
                    )
                combined_added, format_added = outputs.append(DECKBOX_CORPUS, cards, decklist)
                combined_appended += int(combined_added)
                format_appended += int(format_added)
            else:
                outputs.mark_source(DECKBOX_CORPUS, "deckbox", source_id)
            if args.limit_per_format is not None and format_appended >= args.limit_per_format:
                page_complete = index == len(records) - 1
                break
        added_searches = add_balanced_deckbox_searches(format_name, outputs, checkpoint, args)
        if added_searches:
            selected_count = len(outputs.selected_deckbox_cards(format_name))
            logging.info(
                "[deckbox/%s] selected %s balanced card IDs and opened %s card windows",
                format_name, selected_count, added_searches,
            )
        if page_complete:
            job["next_page"] = page + 1
            job["last_fingerprint"] = fingerprint
            job["next_url"] = next_url
        finish_page(outputs, checkpoint, "deckbox", format_name, job, complete=page_complete and not next_url)
        logging.info("[deckbox/%s] %s page %s; fetched %s; known %s; combined +%s; deckbox corpus +%s",
                     format_name, job["key"], page, fetched, known, combined_appended, format_appended)
        time.sleep(args.deckbox_delay)
    return fetched, combined_appended, format_appended
