"""Durable JSONL outputs, disk-backed deduplication, and restartable progress."""

from __future__ import annotations

import atexit
import json
import logging
import os
import sqlite3
import tempfile
import time
from collections.abc import Iterable, Mapping
from mtgdeck.data import cards_fingerprint, decklist_fingerprint
from pathlib import Path
from scrape.sources import DECKBOX_CARD_COLORS, DECKBOX_CARD_ROLES, DECKBOX_CORPUS
from typing import Any
from urllib.parse import parse_qs, urlparse


class DedupIndex:
    """Disposable on-disk indexes rebuilt from the authoritative JSONL files.

    All writers in a run share one bounded SQLite page cache. Keeping this
    separate from the persistent seen-ID cache prevents stale fingerprints
    after interrupted writes, migrations, or manual corpus replacements.
    """

    def __init__(self):
        self.scopes: dict[str, int] = {}
        self.directory = tempfile.TemporaryDirectory(prefix="mtg-scraper-index-")
        self.connection: sqlite3.Connection | None = sqlite3.connect(
            Path(self.directory.name) / "dedup.sqlite3"
        )
        self.connection.execute("PRAGMA cache_size=-2048")
        self.connection.execute("PRAGMA temp_store=FILE")
        # This database is disposable; only the JSONL outputs need durability.
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute(
            "CREATE TABLE keys (scope INTEGER, value TEXT, PRIMARY KEY (scope, value)) WITHOUT ROWID"
        )
        atexit.register(self.close)

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        self.directory.cleanup()
        atexit.unregister(self.close)


class DiskKeySet:
    """The membership/add/count operations needed by corpus deduplication."""

    def __init__(self, index: DedupIndex, scope: str):
        self.index = index
        self.scope = index.scopes.setdefault(scope, len(index.scopes))

    @staticmethod
    def encode(value: str | tuple[str, str]) -> str:
        return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))

    def add(self, value: str | tuple[str, str]) -> None:
        self.index.connection.execute(
            "INSERT OR IGNORE INTO keys VALUES (?, ?)", (self.scope, self.encode(value))
        )

    def __contains__(self, value: str | tuple[str, str]) -> bool:
        return self.index.connection.execute(
            "SELECT 1 FROM keys WHERE scope=? AND value=?", (self.scope, self.encode(value))
        ).fetchone() is not None

    def __len__(self) -> int:
        return self.index.connection.execute(
            "SELECT COUNT(*) FROM keys WHERE scope=?", (self.scope,)
        ).fetchone()[0]


class CorpusWriter:
    """Append count-free embedding rows with disk-backed deduplication."""

    def __init__(self, path: Path, flush_every: int = 1, *, index: DedupIndex | None = None):
        self.path = path
        self._owns_index = index is None
        self._index = index if index is not None else DedupIndex()
        self.fingerprints = DiskKeySet(self._index, f"cards:{path.resolve()}")
        self.appended = 0
        self.flush_every = max(flush_every, 1)
        self._pending = 0
        self._handle: Any = None
        atexit.register(self.close)

    def load(self) -> int:
        if not self.path.exists():
            return 0
        loaded = 0
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{line_number}: invalid JSON") from exc
                cards = row.get("cards")
                if not isinstance(cards, list) or not all(isinstance(card, str) for card in cards):
                    raise ValueError(f"{self.path}:{line_number}: expected {{'cards': [str, ...]}}")
                self.fingerprints.add(cards_fingerprint(sorted(set(cards))))
                loaded += 1
        return loaded

    def append(self, cards: list[str]) -> bool:
        fingerprint = cards_fingerprint(cards)
        if fingerprint in self.fingerprints:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps({"cards": cards}, ensure_ascii=False) + "\n")
        self._pending += 1
        if self._pending >= self.flush_every:
            self._handle.flush()
            self._pending = 0
        self.fingerprints.add(fingerprint)
        self.appended += 1
        return True

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._pending = 0
        if self._owns_index:
            self._index.close()
        atexit.unregister(self.close)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._pending = 0


class DecklistWriter:
    """Append full deck records, deduplicating source identities and zone contents."""

    def __init__(self, path: Path, flush_every: int = 1, *, index: DedupIndex | None = None):
        self.path = path
        self._owns_index = index is None
        self._index = index if index is not None else DedupIndex()
        self.source_keys = DiskKeySet(self._index, f"sources:{path.resolve()}")
        self.fingerprints = DiskKeySet(self._index, f"decks:{path.resolve()}")
        self.appended = 0
        self.flush_every = max(flush_every, 1)
        self._pending = 0
        self._handle: Any = None
        atexit.register(self.close)

    def load(self) -> int:
        if not self.path.exists():
            return 0
        loaded = 0
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{self.path}:{line_number}: invalid JSON") from exc
                if not isinstance(record, Mapping) or not isinstance(record.get("mainboard"), list):
                    raise ValueError(
                        f"{self.path}:{line_number}: expected a full decklist with mainboard; "
                        "choose a new --format-output-dir for old embedding-only format files"
                    )
                source_key = (str(record.get("source", "")), str(record.get("source_id", "")))
                self.source_keys.add(source_key)
                self.fingerprints.add(decklist_fingerprint(record))
                loaded += 1
        return loaded

    def append(self, record: Mapping[str, Any]) -> bool:
        source_key = (str(record.get("source", "")), str(record.get("source_id", "")))
        fingerprint = decklist_fingerprint(record)
        if source_key in self.source_keys or fingerprint in self.fingerprints:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._handle is None:
            self._handle = self.path.open("a", encoding="utf-8")
        self._handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        self._pending += 1
        if self._pending >= self.flush_every:
            self._handle.flush()
            self._pending = 0
        self.source_keys.add(source_key)
        self.fingerprints.add(fingerprint)
        self.appended += 1
        return True

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._pending = 0
        if self._owns_index:
            self._index.close()
        atexit.unregister(self.close)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._pending = 0


def migrate_deckbox_records(format_directory: Path) -> tuple[int, int, int]:
    """Move legacy Deckbox rows out of format corpora into deckbox.jsonl.

    The target corpus is populated before source files are replaced. If a run is
    interrupted, rerunning is therefore lossless and target deduplication
    removes any repeated rows.
    """

    if not format_directory.exists():
        return 0, 0, 0
    target_path = format_directory / f"{DECKBOX_CORPUS}.jsonl"
    target = DecklistWriter(target_path, flush_every=500)
    target_loaded = False
    moved = added = migrated_files = 0
    try:
        for path in sorted(format_directory.glob("*.jsonl")):
            if path == target_path:
                continue
            file_moved = 0
            # Copy one record at a time; even a large legacy corpus must not
            # become a list of every parsed deck and card in memory.
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip() or '"deckbox"' not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                    if not isinstance(record, Mapping) or record.get("source") != "deckbox":
                        continue
                    if not target_loaded:
                        target.load()
                        target_loaded = True
                    added += int(target.append(record))
                    file_moved += 1
            if not file_moved:
                continue
            # Destination must be durable before removing any source rows.
            target.flush()
            temporary = path.with_name(path.name + ".deckbox-migration.tmp")
            with path.open(encoding="utf-8") as source, temporary.open(
                "w", encoding="utf-8"
            ) as destination:
                for line in source:
                    record = json.loads(line) if '"deckbox"' in line else None
                    if isinstance(record, Mapping) and record.get("source") == "deckbox":
                        continue
                    destination.write(line)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, path)
            moved += file_moved
            migrated_files += 1
    finally:
        target.close()
    return moved, added, migrated_files


def migrate_mtgtop8_formats(format_directory: Path) -> int:
    """Repair old mislabeled records only when their source URL proves the format.

    Write and sync the destination before atomically replacing each source file.
    Content and identity deduplication make interrupted migrations restartable.
    """
    corrections = {"commander": ("EDH", "duel-commander"), "explorer": ("EX", "extended")}
    moved = 0
    for old_format, (code, new_format) in corrections.items():
        source_path = format_directory / f"{old_format}.jsonl"
        if not source_path.exists():
            continue
        file_moved = 0
        target = DecklistWriter(format_directory / f"{new_format}.jsonl", flush_every=500)
        try:
            with source_path.open(encoding="utf-8") as source:
                for line in source:
                    if '"mtgtop8"' not in line:
                        continue
                    record = json.loads(line)
                    parsed = urlparse(str(record.get("url", "")))
                    if (record.get("source") != "mtgtop8"
                            or parsed.hostname not in ("mtgtop8.com", "www.mtgtop8.com")
                            or parse_qs(parsed.query).get("f") != [code]):
                        continue
                    if not file_moved:
                        target.load()
                    target.append({**record, "format": new_format})
                    file_moved += 1
            if not file_moved:
                continue
            target.flush()
            temporary = source_path.with_name(source_path.name + ".format-migration.tmp")
            with source_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as destination:
                for line in source:
                    record = json.loads(line) if '"mtgtop8"' in line else {}
                    parsed = urlparse(str(record.get("url", "")))
                    if (record.get("source") == "mtgtop8"
                            and parsed.hostname in ("mtgtop8.com", "www.mtgtop8.com")
                            and parse_qs(parsed.query).get("f") == [code]):
                        continue
                    destination.write(line)
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, source_path)
            moved += file_moved
        finally:
            target.close()
    return moved


class CorpusOutputs:
    """Write constructed embeddings and full decklist corpora."""

    def __init__(
        self,
        combined_path: Path,
        format_directory: Path,
        flush_every: int = 1,
        seen_path: Path | None = None,
    ):
        self.flush_every = flush_every
        self._index = DedupIndex()
        self.combined = CorpusWriter(combined_path, flush_every=flush_every, index=self._index)
        self.format_directory = format_directory
        self.by_format: dict[str, DecklistWriter] = {}
        self.seen_db: sqlite3.Connection | None = None
        if seen_path is not None:
            seen_path.parent.mkdir(parents=True, exist_ok=True)
            self.seen_db = sqlite3.connect(seen_path)
            self.seen_db.execute(
                "CREATE TABLE IF NOT EXISTS seen (corpus TEXT, source TEXT, source_id TEXT, "
                "PRIMARY KEY (corpus, source, source_id)) WITHOUT ROWID"
            )
            self.seen_db.execute(
                "CREATE TABLE IF NOT EXISTS deckbox_card_samples ("
                "format TEXT, source_id TEXT, deck_color TEXT, "
                "PRIMARY KEY (format, source_id)) WITHOUT ROWID"
            )
            self.seen_db.execute(
                "CREATE TABLE IF NOT EXISTS deckbox_card_candidates ("
                "format TEXT, card_id TEXT, name TEXT, color TEXT, roles TEXT, deck_count INTEGER, "
                "PRIMARY KEY (format, card_id)) WITHOUT ROWID"
            )
            self.seen_db.execute(
                "CREATE TABLE IF NOT EXISTS deckbox_card_selections ("
                "format TEXT, card_id TEXT, name TEXT, color TEXT, roles TEXT, deck_count INTEGER, "
                "selection_order INTEGER, PRIMARY KEY (format, card_id)) WITHOUT ROWID"
            )
        atexit.register(self.close)

    def has_source(self, corpus: str, source: str, source_id: str) -> bool:
        writer = self.by_format.get(corpus)
        if writer is not None and (source, source_id) in writer.source_keys:
            return True
        return self.seen_db is not None and self.seen_db.execute(
            "SELECT 1 FROM seen WHERE corpus=? AND source=? AND source_id=?",
            (corpus, source, source_id),
        ).fetchone() is not None

    def mark_source(self, corpus: str, source: str, source_id: str) -> None:
        """Remember a fully evaluated or permanently unavailable source ID."""
        writer = self.by_format.get(corpus)
        if writer is not None:
            writer.source_keys.add((source, source_id))
        if self.seen_db is not None:
            self.seen_db.execute(
                "INSERT OR IGNORE INTO seen VALUES (?, ?, ?)",
                (corpus, source, source_id),
            )

    def record_deckbox_card_sample(
        self, format_name: str, source_id: str, deck_color: str,
        candidates: Iterable[Mapping[str, Any]],
    ) -> bool:
        """Count each card once per newly sampled Deckbox deck."""

        if self.seen_db is None:
            return False
        cursor = self.seen_db.execute(
            "INSERT OR IGNORE INTO deckbox_card_samples VALUES (?, ?, ?)",
            (format_name, source_id, deck_color),
        )
        if not cursor.rowcount:
            return False
        unique = {str(candidate["card_id"]): candidate for candidate in candidates}
        for card_id, candidate in unique.items():
            self.seen_db.execute(
                "INSERT INTO deckbox_card_candidates VALUES (?, ?, ?, ?, ?, 1) "
                "ON CONFLICT(format, card_id) DO UPDATE SET deck_count=deck_count+1, "
                "name=excluded.name, color=excluded.color, roles=excluded.roles",
                (
                    format_name,
                    card_id,
                    str(candidate["name"]),
                    str(candidate["color"]),
                    json.dumps(candidate["roles"], separators=(",", ":")),
                ),
            )
        return True

    def selected_deckbox_cards(self, format_name: str) -> list[dict[str, Any]]:
        if self.seen_db is None:
            return []
        rows = self.seen_db.execute(
            "SELECT card_id, name, color, roles, deck_count FROM deckbox_card_selections "
            "WHERE format=? ORDER BY selection_order", (format_name,),
        )
        return [
            {"card_id": row[0], "name": row[1], "color": row[2],
             "roles": tuple(json.loads(row[3])), "deck_count": row[4]}
            for row in rows
        ]

    def select_balanced_deckbox_cards(
        self, format_name: str, *, limit: int, min_samples: int,
        min_color_buckets: int, min_decks: int,
    ) -> list[dict[str, Any]]:
        """Persist popular cards chosen round-robin across color/role strata."""

        selected = self.selected_deckbox_cards(format_name)
        if self.seen_db is None or len(selected) >= limit:
            return selected[:limit]
        sample_count = self.seen_db.execute(
            "SELECT COUNT(*) FROM deckbox_card_samples WHERE format=?", (format_name,),
        ).fetchone()[0]
        color_count = self.seen_db.execute(
            "SELECT COUNT(DISTINCT deck_color) FROM deckbox_card_samples "
            "WHERE format=? AND deck_color!='all'", (format_name,),
        ).fetchone()[0]
        if sample_count < min_samples or color_count < min_color_buckets:
            return selected

        rows = self.seen_db.execute(
            "SELECT card_id, name, color, roles, deck_count FROM deckbox_card_candidates "
            "WHERE format=? AND deck_count>=? ORDER BY deck_count DESC, name, card_id",
            (format_name, min_decks),
        )
        candidates = [
            {"card_id": row[0], "name": row[1], "color": row[2],
             "roles": tuple(json.loads(row[3])), "deck_count": row[4]}
            for row in rows
        ]
        chosen_ids = {card["card_id"] for card in selected}
        while len(selected) < limit:
            added = False
            for role in DECKBOX_CARD_ROLES:
                for color in DECKBOX_CARD_COLORS:
                    match = next((card for card in candidates
                                  if card["card_id"] not in chosen_ids
                                  and card["color"] == color and role in card["roles"]), None)
                    if match is None:
                        continue
                    selected.append(match)
                    chosen_ids.add(match["card_id"])
                    added = True
                    if len(selected) >= limit:
                        break
                if len(selected) >= limit:
                    break
            if not added:
                break
        for candidate in candidates:
            if len(selected) >= limit:
                break
            if candidate["card_id"] not in chosen_ids:
                selected.append(candidate)
                chosen_ids.add(candidate["card_id"])
        existing_count = self.seen_db.execute(
            "SELECT COUNT(*) FROM deckbox_card_selections WHERE format=?", (format_name,),
        ).fetchone()[0]
        for offset, card in enumerate(selected[existing_count:], existing_count + 1):
            self.seen_db.execute(
                "INSERT OR IGNORE INTO deckbox_card_selections VALUES (?, ?, ?, ?, ?, ?, ?)",
                (format_name, card["card_id"], card["name"], card["color"],
                 json.dumps(card["roles"], separators=(",", ":")), card["deck_count"], offset),
            )
        return self.selected_deckbox_cards(format_name)

    def load(self, formats: Iterable[str]) -> tuple[int, dict[str, int]]:
        combined_count = self.combined.load()
        format_counts: dict[str, int] = {}
        for format_name in dict.fromkeys(formats):
            writer = DecklistWriter(
                self.format_directory / f"{format_name}.jsonl",
                flush_every=self.flush_every,
                index=self._index,
            )
            format_counts[format_name] = writer.load()
            self.by_format[format_name] = writer
        return combined_count, format_counts

    def append(
        self,
        format_name: str,
        cards: list[str],
        decklist: Mapping[str, Any],
    ) -> tuple[bool, bool]:
        """Return (added_to_embedding_corpus, added_to_format)."""

        if format_name not in self.by_format:
            writer = DecklistWriter(
                self.format_directory / f"{format_name}.jsonl",
                flush_every=self.flush_every,
                index=self._index,
            )
            writer.load()
            self.by_format[format_name] = writer
        # These calls are intentionally independent. An existing global deck
        # must still be written when it is new to this format-specific corpus.
        combined_added = self.combined.append(cards)
        format_added = self.by_format[format_name].append(decklist)
        # Remember even IDs whose contents duplicate another deck. Otherwise
        # overlapping card searches repeatedly download the same full lists.
        source_key = (str(decklist["source"]), str(decklist["source_id"]))
        self.mark_source(format_name, *source_key)
        return combined_added, format_added

    def close(self) -> None:
        self.flush()
        self.combined.close()
        for writer in self.by_format.values():
            writer.close()
        if self.seen_db is not None:
            self.seen_db.close()
            self.seen_db = None
        self._index.close()
        atexit.unregister(self.close)

    def flush(self) -> None:
        self.combined.flush()
        for writer in self.by_format.values():
            writer.flush()
        if self.seen_db is not None:
            self.seen_db.commit()


class Checkpoint:
    VERSION = 4

    def __init__(self, path: Path, fresh: bool = False):
        self.path = path
        self.state: dict[str, dict[str, Any]] = {}
        self.legacy: dict[str, dict[str, Any]] = {}
        if not fresh and path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("schema_version") == self.VERSION:
                buckets = raw.get("buckets", {})
                if isinstance(buckets, dict):
                    self.state = buckets
                self.legacy = raw.get("legacy_buckets", {})
            elif isinstance(raw, dict) and raw.get("schema_version") == 3:
                self.legacy = raw.get("buckets", {})
                logging.info("Upgrading v3 checkpoints to resumable search partitions")
            else:
                raise ValueError(f"Unsupported checkpoint schema in {path}; use a new --checkpoint")

    @staticmethod
    def key(source: str, format_name: str) -> str:
        return f"{source}:{format_name}"

    def next_page(self, source: str, format_name: str, default: int) -> int:
        value = self.state.get(self.key(source, format_name), {}).get("next_page", default)
        return int(value)

    def is_complete(self, source: str, format_name: str) -> bool:
        return bool(self.state.get(self.key(source, format_name), {}).get("complete", False))

    def save(self, source: str, format_name: str, next_page: int, complete: bool = False) -> None:
        self.state.setdefault(self.key(source, format_name), {}).update({
            "next_page": next_page,
            "complete": complete,
        })
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        payload = {"schema_version": self.VERSION, "buckets": self.state, "legacy_buckets": self.legacy}
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(3):
            try:
                os.replace(temporary, self.path)
                break
            except PermissionError:
                # Windows can briefly lock a recently written checkpoint.
                # Retry the atomic replacement, keeping the previous file intact.
                if attempt == 2:
                    raise
                time.sleep(0.1 * (attempt + 1))

    def add_searches(self, source: str, format_name: str, searches: Iterable[dict[str, Any]]) -> None:
        bucket = self.state.setdefault(self.key(source, format_name), {})
        jobs = bucket.setdefault("searches", {})
        queue = bucket.setdefault("queue", [])
        for search in searches:
            key = search["key"]
            if key in jobs:
                continue
            job = {"next_page": 1, "complete": False, **search}
            # The original views query can continue where v3 left off. New
            # sort orders and card searches reopen previously exhausted formats.
            legacy = self.legacy.get(self.key(source, format_name), {})
            if source == "moxfield" and key == "views:descending" and legacy:
                job.update(next_page=legacy.get("next_page", 1), complete=legacy.get("complete", False))
            jobs[key] = job
            if not job["complete"]:
                queue.append(key)
        bucket["complete"] = not queue

    def current_search(self, source: str, format_name: str) -> dict[str, Any] | None:
        bucket = self.state[self.key(source, format_name)]
        return bucket["searches"][bucket["queue"][0]] if bucket["queue"] else None

    def select_searches(self, source: str, format_name: str, keys: Iterable[str]) -> None:
        """Suspend out-of-scope searches without discarding their saved progress."""
        bucket = self.state[self.key(source, format_name)]
        allowed = set(keys)
        queue = [key for key in bucket["queue"] if key in allowed]
        queued = set(queue)
        queue.extend(key for key, job in bucket["searches"].items()
                     if key in allowed and key not in queued and not job["complete"])
        bucket["queue"] = queue
        bucket["complete"] = not queue

    def remove_searches(self, source: str, format_name: str, keys: Iterable[str]) -> int:
        """Discard obsolete generated searches while retaining seed progress."""

        bucket = self.state[self.key(source, format_name)]
        removed = set(keys) & set(bucket["searches"])
        if not removed:
            return 0
        bucket["queue"] = [key for key in bucket["queue"] if key not in removed]
        for key in removed:
            del bucket["searches"][key]
        bucket["complete"] = not bucket["queue"]
        return len(removed)

    def finish_search_page(
        self, source: str, format_name: str, job: dict[str, Any],
        *, complete: bool = False, reason: str | None = None,
    ) -> None:
        bucket = self.state[self.key(source, format_name)]
        queue = bucket["queue"]
        assert queue[0] == job["key"]
        queue.pop(0)
        job["complete"] = complete
        if reason:
            job["stop_reason"] = reason
        if not complete:
            queue.append(job["key"])
        self.save(source, format_name, int(job["next_page"]), complete=not queue)

    def refresh_completed(self) -> None:
        for bucket in self.state.values():
            for key, job in bucket.get("searches", {}).items():
                if job.get("complete"):
                    job.update(next_page=1, complete=False)
                    for field in ("last_fingerprint", "next_url", "stop_reason"):
                        job.pop(field, None)
                    bucket["queue"].append(key)
            if bucket.get("queue"):
                bucket["complete"] = False


def finish_page(
    outputs: CorpusOutputs, checkpoint: Checkpoint, source: str, format_name: str,
    job: dict[str, Any], *, complete: bool = False, reason: str | None = None,
) -> None:
    # Corpus data and the identity cache must reach disk before the resume cursor.
    outputs.flush()
    checkpoint.finish_search_page(source, format_name, job, complete=complete, reason=reason)
