"""Collect the curated creator allowlist plus MTGTop8 into an isolated corpus.

Run: python -m scrape.premium --max-pages 1 --limit 2
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup

from scrape import ROOT
from scrape.http import PaginationError, request_with_retries
from scrape.operations import atomic_json, check_stop, scraper_entry
from scrape.scraper import main as scrape_main, positive_int, nonnegative_float
from scrape.state import CorpusOutputs
from mtgdeck.data import (BOARD_NAMES, clean_board, decklist_card_names,
                          make_decklist_record, normalize_deck_cards)
from mtgdeck.metadata import load_oracle_names

# Multiplayer EDH uses commander on creator sites and cedh on MTGTop8.
PREMIUM_FORMATS = ("legacy", "modern", "commander", "cedh")

TCG_API = "https://infinite-api.tcgplayer.com"
ARCH_FORMATS = {1: "standard", 2: "modern", 3: "commander", 4: "legacy", 5: "vintage", 6: "pauper"}


def folder_payload(html, folder_id):
    script = BeautifulSoup(html, "html.parser").find("script", id="__NEXT_DATA__")
    if script is None:
        raise PaginationError("Archidekt folder page has no embedded data")
    folder = json.loads(script.string)["props"]["pageProps"]["redux"]["folders"]["rootFolder"]
    if folder.get("id") != folder_id or not isinstance(folder.get("decks"), list):
        raise PaginationError("Archidekt returned the wrong folder or malformed decks")
    return folder


def archidekt_record(raw, root, folder_path):
    fmt = ARCH_FORMATS.get(raw["deckFormat"])
    if fmt is None:
        raise ValueError(f"Unsupported Archidekt format {raw['deckFormat']}")
    boards = {zone: [] for zone in BOARD_NAMES}
    included = {c["name"] for c in raw["categories"] if c.get("includedInDeck")}
    for entry in raw["cards"]:
        if entry.get("deletedAt"):
            continue
        cats = set(entry["categories"])
        if "Commander" in cats:
            zone = "commanders"
        elif entry.get("companion"):
            zone = "companions"
        elif "Sideboard" in cats:
            zone = "sideboard"
        elif not cats.intersection(included):
            continue
        else:
            zone = "mainboard"
        boards[zone].append({"name": entry["card"]["oracleCard"]["name"], "quantity": entry["quantity"]})
    return make_decklist_record(
        source="archidekt", source_id=str(raw["id"]), format_name=fmt,
        url=f"https://archidekt.com/decks/{raw['id']}", name=raw["name"],
        deck_date=raw.get("createdAt"), boards=boards,
        metadata={"quality_tier": "curated_folder", "curation_url": f"https://archidekt.com/folders/{root}",
                  "creator": raw["owner"]["username"], "folder_path": folder_path,
                  "theorycrafted": raw.get("theorycrafted"), "updated_at": raw.get("updatedAt")})


def tcgplayer_record(raw, author, author_id):
    deck = raw["deck"]
    if deck.get("playerName", "").casefold() != author.casefold() or deck.get("game") != "magic":
        raise ValueError("TCGplayer deck does not match the selected Magic creator")
    boards = {}
    for upstream, zone in (("maindeck", "mainboard"), ("sideboard", "sideboard"),
                           ("commanders", "commanders"), ("commander", "commanders"), ("companion", "companions")):
        for entry in deck["subDecks"].get(upstream, []):
            boards.setdefault(zone, []).append({"name": raw["cards"][str(entry["cardID"])]["name"],
                                               "quantity": entry["quantity"]})
    if not boards.get("mainboard"):
        raise ValueError("TCGplayer deck has no mainboard")
    return make_decklist_record(
        source="tcgplayer", source_id=str(raw["id"]), format_name=deck["format"].lower(),
        url=f"https://www.tcgplayer.com/content/magic-the-gathering/deck/{quote(deck['name'], safe='')}/{raw['id']}",
        name=deck["name"], deck_date=deck.get("created"), boards=boards,
        metadata={"quality_tier": "curated_creator", "creator": author, "author_id": author_id,
                  "curation_url": f"https://www.tcgplayer.com/content/author/{quote(author)}/decks",
                  "updated_at": deck.get("updated"), "is_test_deck": deck.get("isTestDeck"),
                  **{key: deck.get(key) for key in ("eventWins", "eventLosses", "eventDraws", "eventPlacementMin", "eventPlacementMax")}})


class Collector:
    def __init__(self, args, outputs, names):
        self.args, self.outputs, self.names = args, outputs, names
        self.session = requests.Session()
        self.state_path = args.output_dir / "creators.checkpoint.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        if args.refresh:
            self.state = {}
        self.added = 0

    def get(self, url, **kwargs):
        check_stop()
        time.sleep(self.args.delay)
        return request_with_retries(self.session, "GET", url, timeout=30, retries=3, **kwargs)

    def save(self):
        self.outputs.flush()
        atomic_json(self.state_path, self.state)

    def append(self, record):
        if record["format"] not in self.args.formats:
            return
        # Reject unknown names rather than silently train on an incomplete list.
        for zone in BOARD_NAMES:
            entries = record[zone]
            cleaned = clean_board(entries, self.names)
            if sum(e["quantity"] for e in cleaned) != sum(e["quantity"] for e in entries):
                raise ValueError(f"Unrecognized cards in {record['url']}; refresh Oracle data")
            record[zone] = cleaned
        cards = normalize_deck_cards(decklist_card_names(record), self.names, 10)
        if cards is None:
            raise ValueError(f"Too few recognized cards in {record['url']}")
        self.added += int(self.outputs.append(record["format"], cards, record)[1])

    def at_limit(self):
        return self.args.limit is not None and self.added >= self.args.limit

    def tcgplayer(self, author):
        meta = self.get(f"{TCG_API}/c/author/{quote(author)}/").json()["result"]
        if meta["name"].casefold() != author.casefold():
            raise ValueError("TCGplayer returned a different author")
        job = self.state.setdefault(f"tcgplayer:{author}:{','.join(sorted(self.args.formats))}", {"offset": 0})
        for _ in range(self.args.max_pages or 1_000_000):
            if job.get("complete") or self.at_limit():
                break
            payload = self.get(f"{TCG_API}/content/decks/magic", params={
                "authorID": meta["uuid"], "rows": 50, "offset": job["offset"],
                "sort": "created", "order": "desc"}).json()
            rows = payload["result"]
            if (payload["offset"] != job["offset"] or not isinstance(rows, list)
                    or payload["count"] != len(rows) or (not rows and job["offset"] < payload["total"])):
                raise PaginationError("TCGplayer returned inconsistent pagination")
            ids = [str(row["deckID"]) for row in rows]
            if ids and ids == job.get("last_ids"):
                raise PaginationError("TCGplayer repeated the preceding page")
            for row in rows:
                fmt = row["deckData"]["format"].lower()
                if fmt not in self.args.formats or self.outputs.has_source(fmt, "tcgplayer", str(row["deckID"])):
                    continue
                raw = self.get(f"{TCG_API}/deck/magic/{row['deckID']}/", params={
                    "subDecks": "true", "cards": "true", "stats": "true",
                    **({"external": "true"} if row.get("isExternalID") else {})}).json()["result"]
                self.append(tcgplayer_record(raw, author, meta["uuid"]))
                if self.at_limit():
                    self.save()  # Replay partial page, using persisted source IDs.
                    return
            job.update(offset=job["offset"] + len(rows), last_ids=ids,
                       complete=job["offset"] + len(rows) >= payload["total"])
            self.save()

    def archidekt(self, root, recursive=True):
        job = self.state.setdefault(f"archidekt:{root}:{recursive}:{','.join(sorted(self.args.formats))}", {"queue": [[root, [], 1]], "done": []})
        for _ in range(self.args.max_pages or 1_000_000):
            if not job["queue"] or self.at_limit():
                break
            folder_id, ancestors, page = job["queue"][0]
            folder = folder_payload(self.get(f"https://archidekt.com/folders/{folder_id}",
                                             params={"page": page}).text, folder_id)
            path = ancestors + [folder["name"]]
            ids = [str(row["id"]) for row in folder["decks"]]
            if page > 1 and ids and ids == job.get("last_ids"):
                raise PaginationError("Archidekt repeated the preceding page")
            for row in folder["decks"]:
                fmt = ARCH_FORMATS.get(row["deckFormat"])
                if fmt is None:
                    raise ValueError(f"Unsupported Archidekt format {row['deckFormat']} in {row['id']}")
                if fmt not in self.args.formats or self.outputs.has_source(fmt, "archidekt", str(row["id"])):
                    continue
                raw = self.get(f"https://archidekt.com/api/decks/{row['id']}/").json()
                if raw["id"] != row["id"] or raw["parentFolder"] != folder_id:
                    raise ValueError("Archidekt deck no longer belongs to the selected folder")
                self.append(archidekt_record(raw, root, path))
                if self.at_limit():
                    self.save()
                    return
            if folder.get("next"):
                if not ids:
                    raise PaginationError("Archidekt returned an empty page with a next page")
                job["queue"][0][2] += 1
                job["last_ids"] = ids
            else:
                job["queue"].pop(0)
                job["done"].append(folder_id)
                job.pop("last_ids", None)
                if recursive:
                    known = set(job["done"]) | {item[0] for item in job["queue"]}
                    job["queue"].extend([child["id"], path, 1] for child in folder["subfolders"]
                                        if child["id"] not in known and not child.get("private"))
            self.save()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/premium_sources.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/premium")
    parser.add_argument("--oracle-cards", type=Path, default=ROOT / "data/oracle_cards.jsonl.gz")
    parser.add_argument("--sources", nargs="+", choices=["moxfield", "mtgtop8", "tcgplayer", "archidekt"],
                        default=["moxfield", "mtgtop8", "tcgplayer", "archidekt"])
    parser.add_argument("--formats", nargs="+", choices=PREMIUM_FORMATS, default=list(PREMIUM_FORMATS),
                        help="Premium formats: Legacy, Modern, and multiplayer EDH (commander/cedh).")
    parser.add_argument("--max-pages", type=positive_int, help="Page cap per existing source/format or new creator source")
    parser.add_argument("--limit", type=positive_int, help="New-deck cap per source/format; per source for TCGplayer/Archidekt")
    parser.add_argument("--delay", type=nonnegative_float, default=2.0)
    parser.add_argument("--refresh", action="store_true", help="Revisit completed listings, retaining deck deduplication")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = json.loads(args.config.read_text())
    if config["tournament_source"] != "mtgtop8" or not config["moxfield_users"]:
        raise ValueError("Premium configuration requires MTGTop8 and an explicit Moxfield allowlist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "sources.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Source configuration changed; use a new output directory to avoid mixing selections")
    atomic_json(manifest, config)
    status = 0
    # Use the existing resilient scheduler for Moxfield and MTGTop8.
    base = [source for source in args.sources if source in ("moxfield", "mtgtop8")]
    if base:
        options = ["--sources", *base, "--formats", *args.formats,
                   "--moxfield-users", *config["moxfield_users"], "--moxfield-discovery", "off",
                   "--mtgtop8-event-metadata", "--output", str(args.output_dir / "embedding_corpus.jsonl"),
                   "--format-output-dir", str(args.output_dir / "formats"),
                   "--checkpoint", str(args.output_dir / "scraper.checkpoint.json"),
                   "--oracle-cards", str(args.oracle_cards), "--oracle-refresh", "never"]
        if args.max_pages:
            options += ["--max-pages-per-format", str(args.max_pages)]
        if args.limit:
            options += ["--limit-per-format", str(args.limit)]
        if args.refresh:
            options += ["--refresh-searches"]
        status = scrape_main(options)
    extras = [s for s in args.sources if s in ("tcgplayer", "archidekt")]
    if not extras:
        return status
    outputs = CorpusOutputs(args.output_dir / "embedding_corpus.jsonl", args.output_dir / "formats",
                            flush_every=1, seen_path=args.output_dir / ".creators.seen.sqlite3")
    collector = None
    try:
        outputs.load(args.formats)
        collector = Collector(args, outputs, load_oracle_names(args.oracle_cards))
        for source in extras:
            collector.added = 0
            try:
                if source == "tcgplayer":
                    collector.tcgplayer(config["tcgplayer_author"])
                else:
                    collector.archidekt(config["archidekt_folder"], config["archidekt_recursive"])
            except Exception:
                logging.exception("%s paused; rerun to resume", source)
                status = 1
            finally:
                collector.save()
                logging.info("%s: appended %s decks", source, collector.added)
    finally:
        outputs.close()
        if collector:
            collector.session.close()
    return status


if __name__ == "__main__":
    raise SystemExit(scraper_entry(main))
