"""Single-writer scheduler and CLI; existing corpus and checkpoint paths are preserved."""

from __future__ import annotations

import argparse
import copy
import logging
import random
import requests
from collections.abc import Iterable
from datetime import date, datetime
from mtgdeck.metadata import load_oracle_names, load_oracle_profiles
from pathlib import Path
from scrape import ROOT
from scrape.http import ScrapeHTTPError, TransientRequestError
from scrape.sources import DECKBOX_CORPUS, DECKBOX_FORMATS, DEFAULT_FORMATS, MOXFIELD_FORMATS, MTGTOP8_FORMATS
from scrape.sources.deckbox import collect_deckbox_format, deckbox_seed_searches, make_deckbox_session
from scrape.sources.moxfield import collect_moxfield_format, make_moxfield_session, moxfield_seed_searches
from scrape.sources.mtgtop8 import collect_mtgtop8_format, make_mtgtop8_session, mtgtop8_seed_searches
from scrape.state import Checkpoint, CorpusOutputs, migrate_deckbox_records, migrate_mtgtop8_formats
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append deduplicated constructed/EDH decks to embedding_corpus.jsonl."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("moxfield", "mtgtop8", "deckbox"),
        default=("moxfield", "mtgtop8", "deckbox"),
        help="Constructed-deck sources to collect (default: all three).",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=DEFAULT_FORMATS,
        default=DEFAULT_FORMATS,
        help="Canonical format names (default: every configured format).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "cooccurence" / "embedding_corpus.jsonl",
        help="Embedding JSONL to append to.",
    )
    parser.add_argument(
        "--format-output-dir",
        type=Path,
        default=ROOT / "data" / "format_corpora",
        help="Directory for full, quantity-preserving <format>.jsonl decklists.",
    )
    parser.add_argument(
        "--oracle-cards",
        type=Path,
        default=ROOT / "data" / "oracle_cards.jsonl.gz",
        help="Local Scryfall Oracle bulk file used to reject non-card response text.",
    )
    parser.add_argument(
        "--oracle-refresh",
        choices=("if-stale", "always", "never"),
        default="if-stale",
        help="Check GET /bulk-data and refresh Oracle cards before scraping (default: if-stale).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data" / "diverse_scraper.checkpoint.json",
        help="Per-source/per-format resume state.",
    )
    parser.add_argument(
        "--max-pages-per-format",
        type=positive_int,
        default=None,
        help="Optional cap on round-robin page cycles per site/format (default: unlimited).",
    )
    parser.add_argument(
        "--limit-per-format",
        type=positive_int,
        default=None,
        help="Optional cap on newly appended decks per site/format.",
    )
    parser.add_argument(
        "--min-cards",
        type=positive_int,
        default=10,
        help="Minimum unique recognized cards required for a corpus row (default: 10).",
    )
    parser.add_argument(
        "--date-start",
        type=scrape_date,
        default="01/01/1993",
        help="MTGTop8 archive start in DD/MM/YYYY form (default: 01/01/1993).",
    )
    parser.add_argument(
        "--moxfield-discovery", choices=("cards", "commanders", "off"), default="cards",
        help="Expand Moxfield searches using discovered nonland cards and commanders (default: cards).",
    )
    parser.add_argument(
        "--deckbox-discovery", choices=("balanced", "cards", "off"), default="balanced",
        help=("Expand Deckbox with popular nonland cards selected across colors and strategic roles "
              "(default: balanced; cards is a compatibility alias)."),
    )
    parser.add_argument(
        "--deckbox-card-sample-size", type=positive_int, default=300,
        help="Distinct Deckbox decks to sample before selecting card searches (default: 300).",
    )
    parser.add_argument(
        "--deckbox-card-color-buckets", type=positive_int, default=6,
        help="Distinct deck-color groups required in the card sample (default: 6 of 7).",
    )
    parser.add_argument(
        "--deckbox-card-count", "--deckbox-card-windows", dest="deckbox_card_count",
        type=positive_int, default=56,
        help="Maximum selected card IDs per Deckbox format (default: 56).",
    )
    parser.add_argument(
        "--deckbox-card-min-decks", type=positive_int, default=2,
        help="Minimum sampled decks containing a card before selection (default: 2).",
    )
    parser.add_argument(
        "--refresh-searches", action="store_true",
        help="Reopen completed searches at page one, retaining unfinished searches and known deck IDs.",
    )
    parser.add_argument(
        "--moxfield-delay",
        type=nonnegative_float,
        default=1.25,
        help="Seconds between Moxfield requests (default: 1.25).",
    )
    parser.add_argument(
        "--mtgtop8-delay",
        type=nonnegative_float,
        default=2.0,
        help="Seconds between MTGTop8 requests (default: 2.0).",
    )
    parser.add_argument(
        "--deckbox-delay",
        type=nonnegative_float,
        default=1.0,
        help="Seconds between Deckbox requests (default: 1.0).",
    )
    parser.add_argument("--timeout", type=positive_float, default=30.0)
    parser.add_argument("--retries", type=positive_int, default=3)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint pages; corpus content is still deduplicated.",
    )
    parser.add_argument(
        "--shuffle-formats",
        action="store_true",
        help="Visit formats in a deterministic shuffled order instead of listed order.",
    )
    parser.add_argument("--mtgtop8-event-metadata", action="store_true",
                        help="Fetch event pages for tournament placement/player counts (one extra request per deck).")
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def scrape_date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected DD/MM/YYYY") from exc
    if parsed > date.today():
        raise argparse.ArgumentTypeError("start date must not be in the future")
    return parsed.strftime("%d/%m/%Y")


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def refresh_oracle_cards(path: Path, mode: str) -> dict[str, Any] | None:
    """Use the shared Scryfall updater, retaining a local snapshot on transient failure."""

    if mode == "never":
        return None
    from scrape.metadata import create_session, refresh_oracle_cards as refresh

    try:
        return refresh(create_session(), path, force=mode == "always")
    except Exception:
        if mode == "if-stale" and path.exists() and path.stat().st_size > 0:
            logging.exception("Could not check Scryfall; continuing with local Oracle snapshot %s", path)
            return None
        raise


def formats_for_source(source: str, requested: Iterable[str]) -> list[str]:
    if source == "moxfield":
        supported = MOXFIELD_FORMATS
    elif source == "mtgtop8":
        supported = MTGTOP8_FORMATS
    elif source == "deckbox":
        supported = DECKBOX_FORMATS
    else:
        return []
    return [format_name for format_name in requested if format_name in supported]


def build_scrape_buckets(sources: Iterable[str], formats: Iterable[str]) -> list[tuple[str, str]]:
    """Interleave sites within each format for balanced round-robin scheduling."""

    buckets: list[tuple[str, str]] = []
    unique_sources = list(dict.fromkeys(sources))
    for format_name in dict.fromkeys(formats):
        for source in unique_sources:
            if format_name in formats_for_source(source, [format_name]):
                buckets.append((source, format_name))
    return buckets


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.info("Loading Oracle names from %s", args.oracle_cards)
    oracle_result = refresh_oracle_cards(args.oracle_cards, args.oracle_refresh)
    if oracle_result:
        action = "already current" if oracle_result["skipped"] else "updated"
        logging.info(
            "Oracle snapshot %s: %s cards (%s)",
            action,
            f"{oracle_result['cards']:,}",
            oracle_result.get("bulk_updated_at"),
        )
    valid_names = load_oracle_names(args.oracle_cards)
    logging.info("Loaded %s recognized card names", f"{len(valid_names):,}")
    args.oracle_profiles = (
        load_oracle_profiles(args.oracle_cards) if "deckbox" in args.sources else {}
    )
    if args.oracle_profiles:
        logging.info("Loaded color/role profiles for %s cards", f"{len(args.oracle_profiles):,}")

    formats = list(dict.fromkeys(args.formats))
    if args.shuffle_formats:
        random.Random(42).shuffle(formats)

    moved, migrated, migrated_files = migrate_deckbox_records(args.format_output_dir)
    if moved:
        logging.info(
            "Migrated %s Deckbox rows (%s unique) out of %s format corpora into %s",
            f"{moved:,}",
            f"{migrated:,}",
            migrated_files,
            args.format_output_dir / f"{DECKBOX_CORPUS}.jsonl",
        )
    corrected = migrate_mtgtop8_formats(args.format_output_dir)
    if corrected:
        logging.info("Corrected %s historical MTGTop8 format labels using their source URLs", f"{corrected:,}")

    outputs = CorpusOutputs(
        args.output,
        args.format_output_dir,
        flush_every=500,
        seen_path=args.format_output_dir / ".diverse_scraper.seen.sqlite3",
    )
    output_corpora = [
        *formats,
        *([DECKBOX_CORPUS] if "deckbox" in args.sources else []),
    ]
    logging.info("Indexing corpora with temporary on-disk deduplication (2 MiB shared page cache)")
    existing, existing_by_format = outputs.load(output_corpora)
    logging.info(
        "Indexed %s existing corpus rows (%s distinct fingerprints)",
        f"{existing:,}",
        f"{len(outputs.combined.fingerprints):,}",
    )
    logging.info(
        "Indexed %s existing rows across %s format corpora in %s",
        f"{sum(existing_by_format.values()):,}",
        len(existing_by_format),
        args.format_output_dir,
    )
    logging.info("Loading search checkpoint from %s", args.checkpoint)
    checkpoint = Checkpoint(args.checkpoint, fresh=args.fresh)
    for source, format_name in build_scrape_buckets(args.sources, formats):
        if source == "moxfield":
            checkpoint.add_searches(source, format_name, moxfield_seed_searches())
        elif source == "deckbox":
            checkpoint.add_searches(source, format_name, deckbox_seed_searches(format_name))
        else:
            checkpoint.add_searches(source, format_name, mtgtop8_seed_searches(args.date_start))
    if args.refresh_searches:
        checkpoint.refresh_completed()
    for source, format_name in build_scrape_buckets(args.sources, formats):
        jobs = checkpoint.state[checkpoint.key(source, format_name)]["searches"]
        if source == "mtgtop8":
            keys = [job["key"] for job in mtgtop8_seed_searches(args.date_start)]
        elif source == "moxfield":
            keys = [key for key, job in jobs.items()
                    if (args.moxfield_discovery == "cards" or "cardId" not in job["params"])
                    and (args.moxfield_discovery != "off" or "commanderCardId" not in job["params"])]
        else:
            selected_ids = {
                card["card_id"] for card in outputs.selected_deckbox_cards(format_name)
            } if args.deckbox_discovery != "off" else set()
            removed = checkpoint.remove_searches(
                source,
                format_name,
                (key for key in jobs if key.startswith("card:")
                 and key.split(":", 2)[1] not in selected_ids),
            )
            if removed:
                logging.info(
                    "[deckbox/%s] removed %s legacy unranked card windows from the checkpoint",
                    format_name, f"{removed:,}",
                )
            jobs = checkpoint.state[checkpoint.key(source, format_name)]["searches"]
            keys = [
                key for key in jobs
                if not key.startswith("card:") or key.split(":", 2)[1] in selected_ids
            ]
        checkpoint.select_searches(source, format_name, keys)

    total_fetched = total_combined = total_by_format = 0
    collectors = {
        "moxfield": collect_moxfield_format,
        "mtgtop8": collect_mtgtop8_format,
        "deckbox": collect_deckbox_format,
    }
    sessions: dict[str, requests.Session] = {}
    unavailable_sources: set[str] = set()
    live_sources = list(dict.fromkeys(args.sources))
    for source in live_sources:
        source_formats = formats_for_source(source, formats)
        skipped = [format_name for format_name in formats if format_name not in source_formats]
        if skipped:
            logging.info("[%s] unsupported formats skipped: %s", source, ", ".join(skipped))
        try:
            session_factories = {
                "moxfield": make_moxfield_session,
                "mtgtop8": make_mtgtop8_session,
                "deckbox": make_deckbox_session,
            }
            sessions[source] = session_factories[source]()
        except Exception:
            logging.exception("Could not initialize %s; its format buckets will be skipped", source)
            unavailable_sources.add(source)

    active = [
        bucket
        for bucket in build_scrape_buckets(live_sources, formats)
        if bucket[0] not in unavailable_sources and not checkpoint.is_complete(*bucket)
    ]
    bucket_format_additions: dict[tuple[str, str], int] = {bucket: 0 for bucket in active}

    try:
        failures = 0
        cycle = 0
        while active and (args.max_pages_per_format is None or cycle < args.max_pages_per_format):
            cycle += 1
            logging.info(
                "Round-robin cycle %s/%s: visiting one page from %s active format buckets",
                cycle,
                args.max_pages_per_format if args.max_pages_per_format is not None else "unlimited",
                len(active),
            )
            next_active: list[tuple[str, str]] = []
            for source, format_name in active:
                if source in unavailable_sources:
                    continue
                current_total = bucket_format_additions[(source, format_name)]
                remaining = (
                    None
                    if args.limit_per_format is None
                    else args.limit_per_format - current_total
                )
                if remaining is not None and remaining <= 0:
                    continue

                page_args = copy.copy(args)
                page_args.max_pages_per_format = 1
                page_args.limit_per_format = remaining
                logging.info("Visiting next page for %s/%s", source, format_name)
                try:
                    fetched, combined_appended, format_appended = collectors[source](
                        format_name,
                        outputs,
                        checkpoint,
                        valid_names,
                        page_args,
                        sessions[source],
                    )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    failures += 1
                    source_failure = (
                        isinstance(exc, TransientRequestError)
                        or isinstance(exc, ScrapeHTTPError) and exc.status in (401, 403)
                    )
                    if source_failure:
                        unavailable_sources.add(source)
                        detail = (
                            f"HTTP {exc.status}"
                            if getattr(exc, "status", None) is not None
                            else type(getattr(exc, "cause", exc)).__name__
                        )
                        logging.error(
                            "Circuit open for %s after exhausted %s; all of its searches remain resumable",
                            source,
                            detail,
                        )
                    logging.exception("Paused %s/%s; checkpoint retained for the next run", source, format_name)
                    outputs.flush()
                    checkpoint.write()
                    continue
                total_fetched += fetched
                total_combined += combined_appended
                total_by_format += format_appended
                bucket_format_additions[(source, format_name)] += format_appended
                reached_limit = (
                    args.limit_per_format is not None
                    and bucket_format_additions[(source, format_name)] >= args.limit_per_format
                )
                if not checkpoint.is_complete(source, format_name) and not reached_limit:
                    next_active.append((source, format_name))
            active = next_active

        logging.info(
            "Done: live exports %s; constructed embedding +%s at %s; "
            "format rows +%s at %s",
            f"{total_fetched:,}",
            f"{outputs.combined.appended:,}",
            args.output,
            f"{sum(writer.appended for writer in outputs.by_format.values()):,}",
            args.format_output_dir,
        )
    finally:
        outputs.close()
        checkpoint.write()
        for session in sessions.values():
            close = getattr(session, "close", None)
            if close is not None:
                close()
    return 1 if failures or unavailable_sources else 0


if __name__ == "__main__":
    raise SystemExit(main())
