#!/usr/bin/env python3
"""Refresh Scryfall metadata used by collection and offline curation."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import requests
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BULK_ENDPOINT = "https://api.scryfall.com/bulk-data"
SEARCH_ENDPOINT = "https://api.scryfall.com/cards/search"
USER_AGENT = "mtg-deck-evaluator/2.0 (+https://github.com/pengkev/mtg-deck-evaluator)"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Scryfall Oracle cards and explicit commander eligibility metadata.")
    parser.add_argument("--oracle-output", type=Path, default=ROOT / "data" / "oracle_cards.jsonl.gz")
    parser.add_argument("--eligibility-output", type=Path, default=ROOT / "data" / "commander_eligible_oracle_ids.json")
    parser.add_argument("--skip-oracle", action="store_true", help="Keep the existing Oracle bulk file and refresh eligibility only.")
    parser.add_argument("--skip-eligibility", action="store_true", help="Refresh Oracle cards without querying commander eligibility.")
    parser.add_argument("--force-oracle", action="store_true", help="Download Oracle cards even when the local snapshot is current.")
    return parser.parse_args(argv)


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return session


def _get_json_with_backoff(session: requests.Session, url: str, *, params=None, timeout: int = 60) -> tuple[requests.Response, dict]:
    for attempt in range(6):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code != 429:
            response.raise_for_status()
            return response, response.json()
        retry_after = float(response.headers.get("Retry-After", 1.0))
        time.sleep(max(retry_after, 0.5 * (attempt + 1)))
    response.raise_for_status()
    raise RuntimeError("unreachable")


def select_bulk_item(payload: dict, bulk_type: str = "oracle_cards") -> dict:
    """Select a named bulk item from the response returned by GET /bulk-data."""

    items = payload.get("data", [])
    if not isinstance(items, list):
        raise ValueError("Scryfall bulk-data response did not contain a data list")
    for item in items:
        if isinstance(item, dict) and item.get("type") == bulk_type:
            return item
    raise ValueError(f"Scryfall bulk-data response did not include type={bulk_type!r}")


def oracle_metadata_path(output: Path) -> Path:
    return output.with_name(output.name + ".metadata.json")


def refresh_oracle_cards(
    session: requests.Session,
    output: Path,
    *,
    force: bool = False,
    metadata_output: Path | None = None,
) -> dict:
    """Refresh Oracle JSONL when Scryfall advertises a different snapshot."""

    _response, payload = _get_json_with_backoff(session, BULK_ENDPOINT)
    metadata = select_bulk_item(payload)
    download_uri = metadata.get("jsonl_download_uri") or metadata.get("download_uri")
    if not download_uri:
        raise ValueError("Scryfall bulk metadata supplied no supported download URI")
    metadata_output = metadata_output or oracle_metadata_path(output)
    local_metadata: dict = {}
    if metadata_output.exists():
        try:
            candidate = json.loads(metadata_output.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                local_metadata = candidate
        except (OSError, json.JSONDecodeError):
            pass
    is_current = (
        output.exists()
        and output.stat().st_size > 0
        and local_metadata.get("bulk_updated_at") == metadata.get("updated_at")
        and local_metadata.get("download_uri") == download_uri
    )
    if is_current and not force:
        return {
            "bulk_updated_at": metadata.get("updated_at"),
            "cards": int(local_metadata.get("cards", 0)),
            "download_uri": download_uri,
            "skipped": True,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    metadata_temporary = metadata_output.with_name(metadata_output.name + ".tmp")
    try:
        with session.get(download_uri, stream=True, timeout=180) as download:
            download.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if metadata.get("jsonl_download_uri"):
            count = 0
            with gzip.open(temporary, "rt", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        card = json.loads(line)
                        if not isinstance(card, dict) or card.get("object") != "card":
                            raise ValueError("Scryfall compressed Oracle bulk file contained a non-card row")
                        count += 1
        else:
            with temporary.open(encoding="utf-8") as handle:
                cards = json.load(handle)
            if not isinstance(cards, list):
                raise ValueError("Scryfall Oracle bulk download was not a card list")
            count = len(cards)
        if not count:
            raise ValueError("Scryfall Oracle bulk download was empty")
        os.replace(temporary, output)
        local_metadata = {
            "source": BULK_ENDPOINT,
            "type": metadata.get("type"),
            "bulk_updated_at": metadata.get("updated_at"),
            "download_uri": download_uri,
            "compressed_size": metadata.get("compressed_size"),
            "cards": count,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_temporary.write_text(json.dumps(local_metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(metadata_temporary, metadata_output)
    except Exception:
        temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)
        raise
    return {**local_metadata, "skipped": False}


def refresh_commander_eligibility(session: requests.Session, output: Path) -> dict:
    oracle_ids: set[str] = set()
    page_url = SEARCH_ENDPOINT
    params = {"q": "is:commander", "unique": "cards", "order": "name", "dir": "asc"}
    pages = 0
    while page_url:
        _response, payload = _get_json_with_backoff(session, page_url, params=params if pages == 0 else None)
        pages += 1
        oracle_ids.update(str(card["oracle_id"]) for card in payload.get("data", []) if card.get("oracle_id"))
        page_url = payload.get("next_page") if payload.get("has_more") else None
        time.sleep(0.2)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://api.scryfall.com/cards/search?q=is%3Acommander",
        "query": "is:commander",
        "pages": pages,
        "oracle_ids": sorted(oracle_ids),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return {"cards": len(oracle_ids), "pages": pages}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    session = create_session()
    if not args.skip_oracle:
        oracle_result = refresh_oracle_cards(session, args.oracle_output, force=args.force_oracle)
        action = "Already current:" if oracle_result["skipped"] else "Downloaded"
        print(f"{action} {oracle_result['cards']:,} Oracle cards at {args.oracle_output}")
    if not args.skip_eligibility:
        eligibility_result = refresh_commander_eligibility(session, args.eligibility_output)
        print(f"Downloaded {eligibility_result['cards']:,} commander-eligible Oracle IDs to {args.eligibility_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
