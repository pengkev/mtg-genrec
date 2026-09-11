"""Oracle snapshot readers and card discovery metadata."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterator, Mapping
from mtgdeck.data import normalize_card_name
from pathlib import Path
from typing import Any


def iter_oracle_cards(path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream compressed JSONL or read the legacy JSON-array bulk format."""

    if not path.exists():
        raise FileNotFoundError(
            f"Oracle card file not found: {path}. Run python -m scrape.metadata first."
        )
    with path.open("rb") as raw_handle:
        compressed = raw_handle.read(2) == b"\x1f\x8b"
    if compressed:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    card = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid Oracle JSONL") from exc
                if isinstance(card, Mapping):
                    yield card
        return

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cards = payload.get("data", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(cards, list):
        raise ValueError(f"Unexpected Oracle card structure in {path}")
    yield from (card for card in cards if isinstance(card, Mapping))


def load_oracle_names(path: Path) -> set[str]:
    """Load normalized card and face names from a local Scryfall bulk file."""

    names: set[str] = set()
    for card in iter_oracle_cards(path):
        candidates = [card.get("name")]
        faces = card.get("card_faces")
        if isinstance(faces, list):
            candidates.extend(face.get("name") for face in faces if isinstance(face, Mapping))
        for candidate in candidates:
            normalized = normalize_card_name(str(candidate or ""))
            if normalized:
                names.add(normalized)
    return names


def oracle_card_profile(card: Mapping[str, Any]) -> dict[str, Any]:
    """Return stable color and strategic-role buckets for discovery sampling."""

    colors = tuple(color for color in "WUBRG" if color in set(card.get("color_identity") or ()))
    color = colors[0] if len(colors) == 1 else "multicolor" if colors else "colorless"
    faces = card.get("card_faces") if isinstance(card.get("card_faces"), list) else []
    type_line = " // ".join(
        str(value) for value in (card.get("type_line"), *(face.get("type_line") for face in faces
                                                          if isinstance(face, Mapping))) if value
    ).lower()
    oracle_text = "\n".join(
        str(value) for value in (card.get("oracle_text"), *(face.get("oracle_text") for face in faces
                                                            if isinstance(face, Mapping))) if value
    ).lower()
    roles: list[str] = []
    if re.search(r"counter target|destroy target|exile target|deals? \w+ damage|"
                 r"return target .+ to (?:its|their) owner", oracle_text):
        roles.append("interaction")
    if re.search(r"draw (?:a|one|two|three|x|that many|cards?)|"
                 r"search your library(?! for (?:a|up to).*land)", oracle_text):
        roles.append("card-advantage")
    if (card.get("produced_mana") or "treasure token" in oracle_text
            or re.search(r"search your library for (?:a|up to).*land", oracle_text)):
        roles.append("mana")
    if "graveyard" in oracle_text or re.search(r"\bmill\b|from a graveyard", oracle_text):
        roles.append("graveyard")
    if "creature" in type_line:
        roles.append("creature")
    if re.search(r"artifact|enchantment|planeswalker|battle", type_line):
        roles.append("engine")
    return {
        "color": color,
        "roles": tuple(dict.fromkeys(roles or ["other"])),
        "is_land": "land" in type_line,
    }


def load_oracle_profiles(path: Path) -> dict[str, dict[str, Any]]:
    """Index discovery metadata by normalized card and face name."""

    profiles: dict[str, dict[str, Any]] = {}
    for card in iter_oracle_cards(path):
        profile = oracle_card_profile(card)
        candidates = [card.get("name")]
        faces = card.get("card_faces")
        if isinstance(faces, list):
            candidates.extend(face.get("name") for face in faces if isinstance(face, Mapping))
        for candidate in candidates:
            normalized = normalize_card_name(str(candidate or ""))
            if normalized:
                profiles[normalized] = profile
    return profiles


def default_oracle_path(data_dir: Path) -> Path:
    """Prefer the updater's stable filename, then the newest dated snapshot."""
    current = data_dir / "oracle_cards.jsonl.gz"
    if current.exists():
        return current
    snapshots = list(data_dir.glob("oracle_cards_*.jsonl.gz"))
    return max(snapshots, key=lambda path: path.stat().st_mtime) if snapshots else data_dir / "oracle_cards.json"
