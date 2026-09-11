"""Moxfield scraping support."""

from __future__ import annotations

import argparse
import logging
import os
import requests
import sys
import time
from collections.abc import Iterator, Mapping
from mtgdeck.data import BOARD_NAMES, clean_board, decklist_card_names, make_decklist_record, normalize_card_name, normalize_deck_cards, positive_quantity
from scrape import ROOT
from scrape.http import PaginationError, ScrapeHTTPError, request_with_retries, validate_page_fingerprint
from scrape.sources import MOXFIELD_API, MOXFIELD_FORMATS
from scrape.state import Checkpoint, CorpusOutputs, finish_page
from typing import Any


def make_moxfield_session() -> requests.Session:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    user_agent = os.getenv("MOXFIELD_USER_AGENT") or os.getenv("user-agent")
    if not user_agent:
        raise RuntimeError(
            "Moxfield requires MOXFIELD_USER_AGENT (or legacy user-agent) in .env. "
            "Use --sources mtgtop8 deckbox to run without it."
        )
    try:
        import cloudscraper
    except ImportError as exc:
        raise RuntimeError("Moxfield scraping requires cloudscraper") from exc
    session = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    session.headers.update({"User-Agent": user_agent, "Referer": "https://www.moxfield.com/"})
    return session


def moxfield_boards(
    payload: Mapping[str, Any], valid_names: set[str]
) -> dict[str, list[dict[str, Any]]]:
    boards = payload.get("boards", {})
    if not isinstance(boards, Mapping):
        boards = {}
    result: dict[str, list[dict[str, Any]]] = {}
    for board_name in BOARD_NAMES:
        board = boards.get(board_name, {})
        records = board.get("cards", {}) if isinstance(board, Mapping) else {}
        values = records.values() if isinstance(records, Mapping) else records if isinstance(records, list) else []
        entries: list[dict[str, Any]] = []
        for entry in values:
            if not isinstance(entry, Mapping):
                continue
            card = entry.get("card", {})
            name = card.get("name") if isinstance(card, Mapping) else entry.get("name")
            if name:
                entries.append(
                    {"name": str(name), "quantity": positive_quantity(entry.get("quantity", 1))}
                )
        result[board_name] = clean_board(entries, valid_names)
    return result


def moxfield_seed_searches() -> list[dict[str, Any]]:
    return [
        {"key": f"{sort}:{direction}", "params": {"sortType": sort, "sortDirection": direction}}
        for sort, direction in (
            ("views", "descending"), ("created", "descending"),
            ("created", "ascending"), ("views", "ascending"),
        )
    ]


def moxfield_card_searches(payload: Mapping[str, Any], mode: str) -> Iterator[dict[str, Any]]:
    if mode == "off":
        return
    boards = payload.get("boards", {})
    for zone in ("commanders", "mainboard", "sideboard"):
        if mode == "commanders" and zone != "commanders":
            continue
        records = boards.get(zone, {}).get("cards", {})
        for entry in records.values() if isinstance(records, Mapping) else records:
            card = entry.get("card", {})
            if not card.get("id") or not card.get("name"):
                continue
            # Lands are ubiquitous and yield heavily overlapping searches.
            if zone != "commanders" and "land" in card.get("type_line", "").casefold():
                continue
            field = "commanderCardId" if zone == "commanders" else "cardId"
            name = normalize_card_name(card["name"])
            yield {
                "key": f"{field}:{name}", "label": card["name"],
                "params": {field: card["id"], "sortType": "created", "sortDirection": "descending"},
            }


def expand_capped_moxfield_search(job: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if job.get("expanded"):
        return
    job["expanded"] = True
    params = job["params"]
    if not any(key in params for key in ("cardId", "commanderCardId")):
        return
    for direction in ("ascending",):
        yield {"key": job["key"] + ":" + direction,
               "params": {**params, "sortDirection": direction}, "expanded": True}
    if "commanderCardId" in params and "minBracket" not in params:
        for bracket in range(1, 6):
            yield {"key": job["key"] + f":bracket:{bracket}",
                   "params": {**params, "minBracket": bracket, "maxBracket": bracket}}


def collect_moxfield_format(
    format_name: str,
    outputs: CorpusOutputs,
    checkpoint: Checkpoint,
    valid_names: set[str],
    args: argparse.Namespace,
    session: requests.Session | None = None,
) -> tuple[int, int, int]:
    checkpoint.add_searches("moxfield", format_name, moxfield_seed_searches())
    session = session or make_moxfield_session()
    fetched = combined_appended = format_appended = known = missing = 0
    for _ in range(args.max_pages_per_format or sys.maxsize):
        job = checkpoint.current_search("moxfield", format_name)
        if job is None or (args.limit_per_format is not None and format_appended >= args.limit_per_format):
            break
        page = int(job["next_page"])
        response = request_with_retries(
            session, "GET", f"{MOXFIELD_API}/v2/decks/search",
            params={**job["params"], "pageNumber": page, "pageSize": 100,
                    "fmt": MOXFIELD_FORMATS[format_name]},
            timeout=args.timeout, retries=args.retries,
        )
        payload = response.json()
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
            raise PaginationError("Moxfield search did not return a deck result list")
        if payload.get("pageNumber") != page:
            raise PaginationError(f"Moxfield returned page {payload.get('pageNumber')} for page {page}")
        if any(not isinstance(payload.get(key), int) or payload[key] < 0
               for key in ("totalResults", "totalPages")):
            raise PaginationError("Moxfield search is missing valid result counts")
        summaries = payload["data"]
        if summaries and payload["totalPages"] < page:
            raise PaginationError("Moxfield search returned inconsistent page counts")
        if any(not isinstance(row, Mapping) or not row.get("publicId") for row in summaries):
            raise PaginationError("Moxfield search returned malformed deck identities")
        if any(row.get("format", MOXFIELD_FORMATS[format_name]) != MOXFIELD_FORMATS[format_name] for row in summaries):
            raise PaginationError("Moxfield search did not honor the format filter")
        if not summaries:
            finish_page(outputs, checkpoint, "moxfield", format_name, job, complete=True, reason="empty search")
            time.sleep(args.moxfield_delay)
            continue
        fingerprint = validate_page_fingerprint(job, (str(row["publicId"]) for row in summaries))
        capped = int(payload.get("totalResults", 0)) >= 10_000
        if capped:
            checkpoint.add_searches("moxfield", format_name, expand_capped_moxfield_search(job))
        page_complete = True
        for index, summary in enumerate(summaries):
            source_id = str(summary["publicId"])
            if outputs.has_source(format_name, "moxfield", source_id):
                known += 1
                continue
            time.sleep(args.moxfield_delay)
            try:
                detail = request_with_retries(
                    session, "GET", f"{MOXFIELD_API}/v3/decks/all/{source_id}",
                    timeout=args.timeout, retries=args.retries,
                )
            except ScrapeHTTPError as exc:
                if exc.status not in (404, 410):
                    raise
                missing += 1
                logging.warning("[moxfield/%s] unavailable deck %s; continuing", format_name, source_id)
                outputs.mark_source(format_name, "moxfield", source_id)
                continue
            raw = detail.json()
            if not isinstance(raw, Mapping) or not isinstance(raw.get("boards"), Mapping):
                raise ValueError(f"Invalid Moxfield deck response for {source_id}")
            fetched += 1
            if raw.get("format", MOXFIELD_FORMATS[format_name]) != MOXFIELD_FORMATS[format_name]:
                logging.warning("[moxfield/%s] deck %s changed format; skipping", format_name, source_id)
                outputs.mark_source(format_name, "moxfield", source_id)
                continue
            checkpoint.add_searches(
                "moxfield", format_name,
                moxfield_card_searches(raw, getattr(args, "moxfield_discovery", "cards")),
            )
            boards = moxfield_boards(raw, valid_names)
            decklist = make_decklist_record(
                source="moxfield", source_id=source_id, format_name=format_name,
                url=f"https://www.moxfield.com/decks/{source_id}", name=raw.get("name"),
                deck_date=raw.get("createdAtUtc") or raw.get("createdAt"), boards=boards,
                metadata={"user_bracket": raw.get("userBracket", raw.get("bracket")),
                          "auto_bracket": raw.get("autoBracket"),
                          "updated_at": raw.get("lastUpdatedAtUtc"),
                          "hubs": raw.get("hubNames", []) or []},
            )
            cards = normalize_deck_cards(decklist_card_names(decklist), valid_names, args.min_cards)
            if cards:
                combined_added, format_added = outputs.append(format_name, cards, decklist)
                combined_appended += int(combined_added)
                format_appended += int(format_added)
            else:
                outputs.mark_source(format_name, "moxfield", source_id)
            if args.limit_per_format is not None and format_appended >= args.limit_per_format:
                page_complete = index == len(summaries) - 1
                break
        total_pages = int(payload.get("totalPages", page + 1))
        complete = page_complete and page >= total_pages
        if page_complete:
            job["next_page"] = page + 1
            job["last_fingerprint"] = fingerprint
        finish_page(outputs, checkpoint, "moxfield", format_name, job, complete=complete,
                    reason="result window cap" if complete and capped else None)
        logging.info(
            "[moxfield/%s] %s page %s; fetched %s; known %s; unavailable %s; combined +%s; format +%s; searches pending %s",
            format_name, job["key"], page, fetched, known, missing, combined_appended, format_appended,
            len(checkpoint.state[checkpoint.key("moxfield", format_name)]["queue"]),
        )
        time.sleep(args.moxfield_delay)
    return fetched, combined_appended, format_appended
