"""Canonical deck records and small, reproducible data preparation helpers.

The canonical files retain display names and quantities.  Model-facing helpers use
``normalize_card_name`` as the one comparison key used throughout the project.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = 1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
ORACLE_TOKEN_PREFIX = "oid:"
ROLE_MAINBOARD = 0
ROLE_COMMANDER = 1


def normalize_card_name(name: str) -> str:
    """Return a stable key for case, Unicode, whitespace, and multi-face variants.

    Scryfall and the old collectors variously emitted ``A // B``, ``A/B``, and
    accented Unicode.  The front-face key matches the convention in the previous
    Card2Vec work while the original full display name stays in canonical records.
    """

    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = re.sub(r"^a-", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.split(r"\s*//?\s*", text, maxsplit=1)[0].strip()




def _card_items(value: Any) -> list[dict[str, Any]]:
    """Convert common old board shapes to ``[{name, quantity}]``."""

    items: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for name, raw in value.items():
            quantity = raw.get("quantity", raw.get("qty", raw.get("q", 1))) if isinstance(raw, Mapping) else raw
            if name:
                items.append({"name": str(name).strip(), "quantity": positive_quantity(quantity)})
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, str):
                name, quantity = item, 1
            elif isinstance(item, Mapping):
                card = item.get("card")
                name = item.get("name", item.get("n"))
                if not name and isinstance(card, Mapping):
                    name = card.get("name")
                quantity = item.get("quantity", item.get("qty", item.get("q", 1)))
            else:
                continue
            if name:
                entry: dict[str, Any] = {"name": str(name).strip(), "quantity": positive_quantity(quantity)}
                oracle_id = item.get("oracle_id") if isinstance(item, Mapping) else None
                if oracle_id:
                    entry["oracle_id"] = str(oracle_id)
                items.append(entry)

    collapsed: dict[str, dict[str, Any]] = {}
    for item in items:
        key = normalize_card_name(item["name"])
        if not key:
            continue
        if key not in collapsed:
            collapsed[key] = item.copy()
        else:
            collapsed[key]["quantity"] += item["quantity"]
    return list(collapsed.values())




def infer_source(record: Mapping[str, Any], source_hint: str | None = None) -> str:
    if source_hint:
        return source_hint.lower().replace("-", "")
    source = str(record.get("source", "")).lower()
    url = str(record.get("url", record.get("deck_url", ""))).lower()
    if "moxfield" in source or "moxfield" in url or "user_bracket" in record or "auto_bracket" in record:
        return "moxfield"
    if "mtgtop8" in source or "mtgtop8" in url or "mtgo_url" in record or "cmds" in record:
        return "mtgtop8"
    return source or "local"


def normalize_deck_record(record: Mapping[str, Any], source_hint: str | None = None) -> dict[str, Any]:
    """Convert canonical, old Moxfield, old MTGTop8, and generic records."""

    source = infer_source(record, source_hint)
    source_id = record.get("source_id")
    if source_id is None:
        source_id = record.get("id") if source == "moxfield" else record.get("deck_id")
    source_id = str(source_id) if source_id not in (None, "") else None
    if source_id is None and record.get("schema_version") == SCHEMA_VERSION and record.get("deck_id"):
        source_id = str(record["deck_id"]).split(":", 1)[-1]

    if record.get("schema_version") == SCHEMA_VERSION:
        mainboard = _card_items(record.get("mainboard", []))
        commanders = _card_items(record.get("commanders", []))
        sideboard = _card_items(record.get("sideboard", []))
    elif source == "moxfield":
        mainboard = _card_items(record.get("mainboard", record.get("main", [])))
        commanders = _card_items(record.get("commanders", record.get("cmds", [])))
        sideboard = _card_items(record.get("sideboard", []))
    elif source == "mtgtop8":
        mainboard = _card_items(record.get("main", record.get("mainboard", [])))
        commanders = _card_items(record.get("cmds", record.get("commanders", record.get("commander", []))))
        sideboard = _card_items(record.get("sideboard", []))
    else:
        mainboard = _card_items(record.get("mainboard", record.get("main", record.get("cards", []))))
        commanders = _card_items(record.get("commanders", record.get("cmds", record.get("commander", []))))
        sideboard = _card_items(record.get("sideboard", []))

    metadata_in = record.get("metadata", {}) if isinstance(record.get("metadata"), Mapping) else {}
    metadata = {
        "user_bracket": metadata_in.get("user_bracket", record.get("user_bracket", record.get("userBracket"))),
        "auto_bracket": metadata_in.get("auto_bracket", record.get("auto_bracket", record.get("autoBracket"))),
        "placement": metadata_in.get("placement", record.get("placement")),
        "players": metadata_in.get("players", record.get("players")),
        "hubs": metadata_in.get("hubs", record.get("hubs", record.get("hubNames", []))) or [],
    }
    for key, value in metadata_in.items():
        metadata.setdefault(key, value)
    for key in ("bracket", "is_autobracket", "placement_of", "mtgo_url"):
        if key in record:
            metadata.setdefault(key, record[key])

    url = record.get("url", record.get("deck_url"))
    if not url and source == "moxfield" and source_id:
        url = f"https://www.moxfield.com/decks/{source_id}"
    provisional = {
        "schema_version": SCHEMA_VERSION,
        "deck_id": str(record.get("deck_id")) if record.get("schema_version") == SCHEMA_VERSION and record.get("deck_id") else (f"{source}:{source_id}" if source_id else ""),
        "source": source,
        "source_id": source_id,
        "url": url,
        "name": record.get("name"),
        "format": str(record.get("format", "commander")).lower(),
        "date": normal_date(record.get("date", record.get("createdAt"))),
        "commanders": commanders,
        "mainboard": mainboard,
        "sideboard": sideboard,
        "companions": _card_items(record.get("companions", [])),
        "metadata": metadata,
    }
    if not provisional["deck_id"]:
        provisional["deck_id"] = f"{source}:content-{deck_fingerprint(provisional)[:16]}"
    validate_canonical_deck(provisional)
    return provisional


def canonical_validation_errors(record: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ("schema_version", "deck_id", "source", "commanders", "mainboard", "sideboard", "metadata")
    for key in required:
        if key not in record:
            errors.append(f"missing {key}")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not record.get("deck_id") or not record.get("source"):
        errors.append("deck_id and source must be non-empty")
    for zone in ("commanders", "mainboard", "sideboard", "companions"):
        value = record.get(zone, []) if zone == "companions" else record.get(zone)
        if not isinstance(value, list):
            errors.append(f"{zone} must be a list")
            continue
        for index, item in enumerate(value):
            if not isinstance(item, Mapping) or not normalize_card_name(item.get("name", "")):
                errors.append(f"{zone}[{index}] must have a name")
            elif not isinstance(item.get("quantity"), int) or item["quantity"] < 1:
                errors.append(f"{zone}[{index}].quantity must be a positive integer")
    if not isinstance(record.get("metadata"), Mapping):
        errors.append("metadata must be an object")
    return errors


def validate_canonical_deck(record: Mapping[str, Any]) -> Mapping[str, Any]:
    errors = canonical_validation_errors(record)
    if errors:
        raise ValueError("invalid canonical deck: " + "; ".join(errors))
    return record


def deck_fingerprint(record: Mapping[str, Any]) -> str:
    commanders = sorted((normalize_card_name(x.get("name", "")), int(x.get("quantity", 1))) for x in record.get("commanders", []))
    cards: Counter[str] = Counter()
    for zone in ("mainboard", "sideboard"):
        for item in record.get(zone, []):
            cards[normalize_card_name(item.get("name", ""))] += int(item.get("quantity", 1))
    payload = {"commanders": commanders, "cards": sorted(cards.items())}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def deduplicate_decks(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically keep the first source identity and first exact decklist."""

    kept: list[dict[str, Any]] = []
    source_keys: set[tuple[str, str]] = set()
    fingerprints: set[str] = set()
    for raw in records:
        record = normalize_deck_record(raw) if raw.get("schema_version") != SCHEMA_VERSION else copy.deepcopy(dict(raw))
        source_key = (str(record["source"]), str(record.get("source_id") or record["deck_id"]))
        fingerprint = deck_fingerprint(record)
        if source_key in source_keys or fingerprint in fingerprints:
            continue
        source_keys.add(source_key)
        fingerprints.add(fingerprint)
        kept.append(record)
    return kept


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def load_decks(path: str | Path, *, normalize: bool = False, source_hint: str | None = None) -> list[dict[str, Any]]:
    records = list(iter_jsonl(path))
    if normalize:
        return [normalize_deck_record(record, source_hint) for record in records]
    for record in records:
        validate_canonical_deck(record)
    return records


def write_decks(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_canonical_deck(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_vocabulary(records: Iterable[Mapping[str, Any]], min_count: int = 1) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for deck in records:
        seen = {normalize_card_name(item["name"]) for zone in ("commanders", "mainboard") for item in deck.get(zone, [])}
        counts.update(name for name in seen if name)
    names = sorted(name for name, count in counts.items() if count >= min_count)
    return {name: index for index, name in enumerate([PAD_TOKEN, UNK_TOKEN, *names])}


def deck_to_tokens(record: Mapping[str, Any], vocab: Mapping[str, int]) -> tuple[list[int], list[int], list[int]]:
    """Return unique card IDs, quantities, and roles without repeating basic lands."""

    merged: dict[tuple[str, int], int] = defaultdict(int)
    for zone, role in (("commanders", ROLE_COMMANDER), ("mainboard", ROLE_MAINBOARD)):
        for item in record.get(zone, []):
            merged[(normalize_card_name(item["name"]), role)] += int(item.get("quantity", 1))
    rows = sorted(merged.items(), key=lambda row: (row[0][1], row[0][0]))
    return (
        [vocab.get(name, vocab.get(UNK_TOKEN, 1)) for (name, _), _quantity in rows],
        [quantity for (_key, quantity) in rows],
        [role for ((_name, role), _quantity) in rows],
    )


def split_decks(records: Sequence[Mapping[str, Any]], seed: int = 42, ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> tuple[list, list, list]:
    """Split fingerprint groups so exact duplicates can never leak across splits."""

    if len(ratios) != 3 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios must contain three values summing to 1")
    groups: dict[str, list] = defaultdict(list)
    for deck in records:
        groups[deck_fingerprint(deck)].append(deck)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])
    partitions = (keys[:train_end], keys[train_end:val_end], keys[val_end:])
    return tuple([deck for key in part for deck in groups[key]] for part in partitions)  # type: ignore[return-value]


def mask_deck(record: Mapping[str, Any], mask_ratio: float = 0.2, seed: int | None = None, keep_commander: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Hide unique mainboard card tokens for the discrete completion target."""

    if not 0 < mask_ratio < 1:
        raise ValueError("mask_ratio must be between 0 and 1")
    visible = copy.deepcopy(dict(record))
    candidates = list(range(len(visible.get("mainboard", []))))
    if not candidates:
        return visible, []
    hide_count = min(len(candidates), max(1, round(len(candidates) * mask_ratio)))
    hidden_indices = set(random.Random(seed).sample(candidates, hide_count))
    hidden = [item for index, item in enumerate(visible["mainboard"]) if index in hidden_indices]
    visible["mainboard"] = [item for index, item in enumerate(visible["mainboard"]) if index not in hidden_indices]
    if not keep_commander:
        hidden.extend(visible.get("commanders", []))
        visible["commanders"] = []
    return visible, hidden


def candidate_mask(present_cards: Iterable[str], vocab: Mapping[str, int]) -> np.ndarray:
    """Boolean array where True denotes a recommendable, not-already-present card."""

    allowed = np.ones(len(vocab), dtype=bool)
    for special in (PAD_TOKEN, UNK_TOKEN):
        if special in vocab:
            allowed[vocab[special]] = False
    for name in present_cards:
        normalized = normalize_card_name(name)
        if normalized in vocab:
            allowed[vocab[normalized]] = False
    return allowed


BOARD_NAMES = ("mainboard", "commanders", "sideboard", "companions")


def normalize_deck_cards(card_names: Iterable[str], valid_names: set[str], min_cards: int) -> list[str] | None:
    """Return the exact model-facing deck representation or reject a fragment."""

    cards = sorted(
        {
            normalized
            for raw_name in card_names
            for normalized in (normalize_card_name(raw_name),)
            if normalized and normalized in valid_names
        }
    )
    return cards if len(cards) >= min_cards else None



def cards_fingerprint(cards: Iterable[str]) -> str:
    canonical = json.dumps(list(cards), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()



def positive_quantity(value: Any) -> int:
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 1



def clean_board(entries: Iterable[Mapping[str, Any]], valid_names: set[str]) -> list[dict[str, Any]]:
    """Retain display names and quantities while filtering against Oracle names."""

    collapsed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = str(entry.get("name") or "").strip()
        key = normalize_card_name(name)
        if not key or key not in valid_names:
            continue
        quantity = positive_quantity(entry.get("quantity", 1))
        if key in collapsed:
            collapsed[key]["quantity"] += quantity
        else:
            collapsed[key] = {"name": name, "quantity": quantity}
            if entry.get("oracle_id"):
                collapsed[key]["oracle_id"] = str(entry["oracle_id"])
    return list(collapsed.values())



def normal_date(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}[T ]", text):
        return text[:10]
    for date_format in (
        "%Y-%m-%d",
        "%d/%m/%y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%b-%Y %H:%M",
        "%d-%b-%Y",
    ):
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            pass
    return text



def make_decklist_record(
    *,
    source: str,
    source_id: str,
    format_name: str,
    url: str,
    boards: Mapping[str, list[dict[str, Any]]],
    name: str | None = None,
    deck_date: Any = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the lossless-enough schema used by each format corpus."""

    return {
        "schema_version": 1,
        "deck_id": f"{source}:{source_id}",
        "source": source,
        "source_id": str(source_id),
        "url": url,
        "name": name,
        "format": format_name,
        "date": normal_date(deck_date),
        "mainboard": list(boards.get("mainboard", [])),
        "sideboard": list(boards.get("sideboard", [])),
        "commanders": list(boards.get("commanders", [])),
        "companions": list(boards.get("companions", [])),
        "metadata": dict(metadata or {}),
    }



def decklist_card_names(record: Mapping[str, Any]) -> Iterator[str]:
    for zone in BOARD_NAMES:
        entries = record.get(zone, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, Mapping) and entry.get("name"):
                yield str(entry["name"])



def decklist_fingerprint(record: Mapping[str, Any]) -> str:
    zones: dict[str, list[tuple[str, int]]] = {}
    for zone in BOARD_NAMES:
        entries = record.get(zone, [])
        zones[zone] = sorted(
            (
                normalize_card_name(str(entry.get("name") or "")),
                positive_quantity(entry.get("quantity", 1)),
            )
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("name")
        )
    canonical = json.dumps(zones, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
