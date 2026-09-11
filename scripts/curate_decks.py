#!/usr/bin/env python3
"""Create a clean, Oracle-resolved, legal Commander corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mtgdeck.data import canonical_validation_errors, iter_jsonl, normalize_deck_record
from mtgdeck.metadata import default_oracle_path
from mtgdeck.legality import OracleCatalog, oracle_deck_fingerprint, validate_commander_deck


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter canonical JSONL into an Oracle-backed legal Commander corpus.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", type=Path, help="One canonical or older source JSONL (never modified).")
    inputs.add_argument("--manifest", type=Path, help="JSON subset manifest; defaults to configs/commander.json.")
    parser.add_argument("--oracle-cards", type=Path, default=default_oracle_path(ROOT / "data"), help="Scryfall Oracle bulk JSON or compressed JSONL snapshot.")
    parser.add_argument(
        "--commander-eligibility",
        type=Path,
        default=ROOT / "data" / "commander_eligible_oracle_ids.json",
        help="Versioned Scryfall is:commander Oracle-ID snapshot; used when present.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "decks_clean.jsonl", help="Accepted, clean deck JSONL.")
    parser.add_argument("--rejected-output", type=Path, default=None, help="Compact rejection manifest; defaults beside --output.")
    parser.add_argument("--report", type=Path, default=None, help="JSON audit report; defaults beside --output.")
    parser.add_argument("--allow-sideboard", action="store_true", help="Do not reject nonempty sideboards. Companion legality is not inferred.")
    parser.add_argument("--no-oracle-ids", action="store_true", help="Omit Oracle IDs from accepted card entries to reduce output size.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many records (for diagnostics).")
    parser.add_argument("--progress-every", type=int, default=10_000)
    return parser.parse_args(argv)


def _default_neighbor(output: Path, suffix: str) -> Path:
    return output.with_name(output.stem + suffix)


def load_manifest(path: Path) -> tuple[list[Path], set[str]]:
    """Resolve input paths relative to the manifest; only explicit optional files may be absent."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    inputs = []
    for entry in manifest["inputs"]:
        candidate = (path.parent / entry["path"]).resolve()
        if not candidate.is_file():
            if entry.get("optional", False):
                print(f"Skipping absent optional input: {candidate}")
                continue
            raise FileNotFoundError(candidate)
        if candidate not in inputs:
            inputs.append(candidate)
    if not inputs:
        raise ValueError(f"No input files exist for {path}; run the scraper first")
    return inputs, {str(value).lower() for value in manifest["formats"]}


def iter_input_decks(paths: list[Path], formats: set[str] | None = None):
    """Stream selected records, adapting older harvests into schema v1."""
    for path in paths:
        for record in iter_jsonl(path):
            if formats and str(record.get("format", "commander")).lower() not in formats:
                continue
            # Validate existing canonical records without silently repairing corrupt fields.
            yield record if "schema_version" in record else normalize_deck_record(record)


def curate_file(
    input_path: Path | list[Path],
    oracle_path: Path,
    output_path: Path,
    rejected_path: Path,
    report_path: Path,
    *,
    formats: set[str] | None = None,
    commander_eligibility_path: Path | None = None,
    annotate_oracle_ids: bool = True,
    allow_sideboard: bool = False,
    limit: int | None = None,
    progress_every: int = 10_000,
) -> dict[str, Any]:
    input_paths = [input_path] if isinstance(input_path, Path) else list(input_path)
    if not input_paths:
        raise ValueError("At least one input is required")
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    protected = {path.resolve() for path in [*input_paths, oracle_path]}
    if commander_eligibility_path:
        protected.add(commander_eligibility_path.resolve())
    destinations = [output_path, rejected_path, report_path,
                    output_path.with_name(output_path.name + ".tmp"),
                    rejected_path.with_name(rejected_path.name + ".tmp")]
    resolved = [path.resolve() for path in destinations]
    if len(set(resolved)) != len(resolved) or protected.intersection(resolved):
        raise ValueError("Outputs and temporary paths must be distinct and cannot overwrite inputs or metadata")
    eligibility_path = commander_eligibility_path if commander_eligibility_path and commander_eligibility_path.exists() else None
    catalog = OracleCatalog.from_path(oracle_path, eligibility_path)
    for path in (output_path, rejected_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(output_path.name + ".tmp")
    rejected_tmp = rejected_path.with_name(rejected_path.name + ".tmp")

    processed = accepted = rejected = duplicate_count = 0
    reason_decks: Counter[str] = Counter()
    reason_cards: dict[str, Counter[str]] = defaultdict(Counter)
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    fingerprints: set[tuple] = set()
    started = datetime.now(timezone.utc)
    try:
        with output_tmp.open("w", encoding="utf-8") as accepted_handle, rejected_tmp.open("w", encoding="utf-8") as rejected_handle:
            for deck in iter_input_decks(input_paths, formats):
                if limit is not None and processed >= limit:
                    break
                processed += 1
                source = str(deck.get("source", "unknown"))
                source_stats[source]["processed"] += 1
                schema_errors = canonical_validation_errors(deck)
                if schema_errors:
                    issues = [{"code": "invalid_schema", "message": message} for message in schema_errors]
                    cleaned = None
                    codes = {"invalid_schema"}
                else:
                    result = validate_commander_deck(
                        deck,
                        catalog,
                        annotate_oracle_ids=annotate_oracle_ids,
                        allow_sideboard=allow_sideboard,
                    )
                    cleaned = result.cleaned_deck
                    issues = [issue.to_dict() for issue in result.issues]
                    codes = result.reason_codes
                    if result.legal:
                        fingerprint = oracle_deck_fingerprint(result)
                        if fingerprint in fingerprints:
                            issues = [{"code": "duplicate_deck", "message": "Oracle-resolved deck contents duplicate an earlier accepted record"}]
                            codes = {"duplicate_deck"}
                            duplicate_count += 1
                        else:
                            fingerprints.add(fingerprint)

                if not issues and cleaned is not None:
                    accepted_handle.write(json.dumps(cleaned, ensure_ascii=False, separators=(",", ":")) + "\n")
                    accepted += 1
                    source_stats[source]["accepted"] += 1
                else:
                    rejection = {
                        "deck_id": deck.get("deck_id"),
                        "source": source,
                        "source_id": deck.get("source_id"),
                        "issues": issues,
                    }
                    rejected_handle.write(json.dumps(rejection, ensure_ascii=False, separators=(",", ":")) + "\n")
                    rejected += 1
                    source_stats[source]["rejected"] += 1
                    for code in codes:
                        reason_decks[code] += 1
                    for issue in issues:
                        if issue.get("card"):
                            reason_cards[str(issue["code"])][str(issue["card"])] += 1

                if progress_every and processed % progress_every == 0:
                    rate = accepted / processed if processed else 0.0
                    print(f"Processed {processed:,}; accepted {accepted:,} ({rate:.1%}); rejected {rejected:,}", flush=True)

        os.replace(output_tmp, output_path)
        os.replace(rejected_tmp, rejected_path)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        rejected_tmp.unlink(missing_ok=True)
        raise

    oracle_stat = oracle_path.stat()
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        "inputs": [str(path) for path in input_paths],
        "output": str(output_path),
        "rejected_output": str(rejected_path),
        "oracle_snapshot": {
            "path": str(oracle_path),
            "modified_at": datetime.fromtimestamp(oracle_stat.st_mtime, timezone.utc).isoformat(),
            "cards": len(catalog),
            "release_range": list(catalog.release_range),
            "commander_eligibility_path": str(eligibility_path) if eligibility_path else None,
            "commander_eligible_cards": (
                len(catalog.commander_eligible_oracle_ids)
                if catalog.commander_eligible_oracle_ids is not None
                else None
            ),
        },
        "settings": {
            "annotate_oracle_ids": annotate_oracle_ids,
            "allow_sideboard": allow_sideboard,
            "formats": sorted(formats) if formats else None,
            "limit": limit,
        },
        "processed": processed,
        "accepted": accepted,
        "rejected": rejected,
        "acceptance_rate": accepted / processed if processed else 0.0,
        "oracle_resolved_duplicates": duplicate_count,
        "reason_deck_counts": dict(reason_decks.most_common()),
        "reason_card_examples": {code: counter.most_common(25) for code, counter in sorted(reason_cards.items())},
        "sources": {source: dict(counts) for source, counts in sorted(source_stats.items())},
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rejected_path = args.rejected_output or _default_neighbor(args.output, "_rejected.jsonl")
    report_path = args.report or _default_neighbor(args.output, "_report.json")
    input_paths, formats = ([args.input], None) if args.input else load_manifest(args.manifest or ROOT / "configs" / "commander.json")
    report = curate_file(
        input_paths,
        args.oracle_cards,
        args.output,
        rejected_path,
        report_path,
        formats=formats,
        commander_eligibility_path=args.commander_eligibility,
        annotate_oracle_ids=not args.no_oracle_ids,
        allow_sideboard=args.allow_sideboard,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(
        f"Accepted {report['accepted']:,}/{report['processed']:,} decks ({report['acceptance_rate']:.1%}) into {args.output}; "
        f"rejections: {rejected_path}; report: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
