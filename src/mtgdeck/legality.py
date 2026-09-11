"""Offline, Oracle-backed Commander deck legality and curation helpers.

The canonical schema in :mod:`mtgdeck.data` answers "can this record be
parsed?".  This module deliberately answers the stricter question "is this a
currently legal paper Commander deck according to this Oracle snapshot?".
Questionable records are rejected with stable reason codes; deck contents are
never guessed or silently repaired.
"""

from __future__ import annotations

import copy
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .metadata import iter_oracle_cards
from .data import ORACLE_TOKEN_PREFIX, PAD_TOKEN, UNK_TOKEN


SUPPORTED_COMMANDER_FORMATS = {"commander", "edh", "cedh"}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def oracle_name_key(name: str) -> str:
    """Normalize a card name for lookup without discarding additional faces."""

    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"’": "'", "‘": "'", "‐": "-", "‑": "-", "–": "-", "—": "-", "…": "..."}))
    text = re.sub(r"\s*/{1,2}\s*", " // ", text)
    text = re.sub(r"\s*\.\s*\.\s*\.\s*", "...", text)
    text = re.sub(r"_{3,}", "_____", text)
    return re.sub(r"\s+", " ", text).casefold().strip()


def _front_value(card: Mapping[str, Any], key: str, default: Any = "") -> Any:
    faces = card.get("card_faces")
    if isinstance(faces, Sequence) and faces and isinstance(faces[0], Mapping):
        return faces[0].get(key, card.get(key, default))
    return card.get(key, default)


def _front_type_line(card: Mapping[str, Any]) -> str:
    return str(_front_value(card, "type_line", ""))


def _front_oracle_text(card: Mapping[str, Any]) -> str:
    return str(_front_value(card, "oracle_text", ""))


def _type_words(card: Mapping[str, Any]) -> set[str]:
    card_types = re.split(r"\s+", _front_type_line(card).split("—", 1)[0].strip())
    return {word.casefold() for word in card_types}


def _subtype_words(card: Mapping[str, Any]) -> set[str]:
    type_line = _front_type_line(card)
    subtype = type_line.split("—", 1)[1] if "—" in type_line else ""
    return {word.casefold() for word in re.split(r"\s+", subtype.strip()) if word}


def is_individual_commander(card: Mapping[str, Any], eligible_oracle_ids: set[str] | None = None) -> bool:
    """Return whether a card can be the deck's sole commander."""

    if eligible_oracle_ids is not None:
        return str(card.get("oracle_id", "")) in eligible_oracle_ids
    types = _type_words(card)
    oracle_text = _front_oracle_text(card).casefold()
    if "legendary" in types and "creature" in types:
        return True
    if "can be your commander" in oracle_text:
        return True
    # Grist's characteristic-defining ability makes it a creature before play.
    return "legendary" in types and "isn't on the battlefield, it's a 1/1 insect creature" in oracle_text


def _partner_mode(card: Mapping[str, Any]) -> tuple[str | None, str | None]:
    text = _front_oracle_text(card)
    folded = text.casefold().replace("—", "-").replace("–", "-")
    match = re.search(r"(?:^|\n)partner with ([^(\n]+)", text, re.IGNORECASE)
    if match:
        return "partner_with", oracle_name_key(match.group(1))
    labeled = re.search(r"(?:^|\n)partner-([^\n(]+)", folded)
    if labeled:
        return "partner_label", oracle_name_key(labeled.group(1))
    if re.search(r"(?:^|\n)partner(?: \([^\n]*|\s*)$", folded):
        return "partner", None
    return None, None


def _chooses_background(card: Mapping[str, Any]) -> bool:
    return "choose a background" in _front_oracle_text(card).casefold()


def _is_background(card: Mapping[str, Any]) -> bool:
    return "legendary" in _type_words(card) and "background" in _subtype_words(card)


def _is_doctors_companion(card: Mapping[str, Any]) -> bool:
    return "doctor's companion" in _front_oracle_text(card).casefold().replace("’", "'")


def _is_doctor(card: Mapping[str, Any]) -> bool:
    types = _type_words(card)
    subtypes = _subtype_words(card)
    return "legendary" in types and "creature" in types and {"time", "lord", "doctor"}.issubset(subtypes)


def is_valid_commander_pair(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    eligible_oracle_ids: set[str] | None = None,
) -> bool:
    """Validate the supported two-commander mechanics from Oracle text."""

    first_mode, first_target = _partner_mode(first)
    second_mode, second_target = _partner_mode(second)
    individually_eligible = is_individual_commander(first, eligible_oracle_ids) and is_individual_commander(second, eligible_oracle_ids)
    if first_mode == second_mode == "partner" and individually_eligible:
        return True
    if first_mode == second_mode == "partner_label" and individually_eligible:
        return first_target == second_target
    if first_mode == second_mode == "partner_with" and individually_eligible:
        return first_target == oracle_name_key(second.get("name", "")) and second_target == oracle_name_key(first.get("name", ""))
    if (_chooses_background(first) and _is_background(second)) or (_chooses_background(second) and _is_background(first)):
        return True
    if (_is_doctors_companion(first) and _is_doctor(second)) or (_is_doctors_companion(second) and _is_doctor(first)):
        return True
    return False


def allowed_copy_count(card: Mapping[str, Any]) -> int | None:
    """Return the Commander copy limit, or ``None`` for unlimited copies."""

    if "basic" in _type_words(card):
        return None
    text = _front_oracle_text(card).casefold()
    if re.search(r"\ba deck can have any number of cards named\b", text):
        return None
    match = re.search(r"\ba deck can have up to ([a-z]+|\d+) cards named\b", text)
    if match:
        raw = match.group(1)
        return int(raw) if raw.isdigit() else _NUMBER_WORDS.get(raw, 1)
    return 1


def _card_preference(card: Mapping[str, Any]) -> tuple:
    legality = card.get("legalities", {}).get("commander")
    layout = str(card.get("layout", ""))
    return (
        legality != "legal",
        "paper" not in card.get("games", []),
        layout in {"token", "emblem", "art_series", "double_faced_token"},
        card.get("lang") != "en",
        str(card.get("oracle_id", "")),
    )


class OracleCatalog:
    """Compact name/Oracle-ID resolver built from Scryfall Oracle bulk data."""

    def __init__(
        self,
        cards: Iterable[Mapping[str, Any]],
        source: str | Path | None = None,
        commander_eligible_oracle_ids: Iterable[str] | None = None,
    ) -> None:
        exact: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        face_aliases: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_oracle_id: dict[str, Mapping[str, Any]] = {}
        release_dates: list[str] = []
        for raw in cards:
            card = dict(raw)
            name = card.get("name")
            oracle_id = card.get("oracle_id")
            if not name or not oracle_id:
                continue
            exact[oracle_name_key(name)].append(card)
            by_oracle_id[str(oracle_id)] = card
            if card.get("released_at"):
                release_dates.append(str(card["released_at"]))
            for face in card.get("card_faces") or []:
                if isinstance(face, Mapping) and face.get("name"):
                    face_aliases[oracle_name_key(face["name"])].append(card)
        self._exact = {key: sorted(values, key=_card_preference) for key, values in exact.items()}
        self._face_aliases = {key: sorted(values, key=_card_preference) for key, values in face_aliases.items()}
        self._by_oracle_id = by_oracle_id
        self.source = str(source) if source is not None else None
        self.release_range = (min(release_dates), max(release_dates)) if release_dates else (None, None)
        self.commander_eligible_oracle_ids = (
            {str(oracle_id) for oracle_id in commander_eligible_oracle_ids}
            if commander_eligible_oracle_ids is not None
            else None
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        commander_eligibility_path: str | Path | None = None,
    ) -> "OracleCatalog":
        source = Path(path)
        cards = iter_oracle_cards(source)
        eligible_ids = None
        if commander_eligibility_path is not None:
            eligibility_source = Path(commander_eligibility_path)
            with eligibility_source.open(encoding="utf-8") as handle:
                eligibility_payload = json.load(handle)
            eligible_ids = eligibility_payload.get("oracle_ids", eligibility_payload) if isinstance(eligibility_payload, Mapping) else eligibility_payload
            if not isinstance(eligible_ids, list):
                raise ValueError(f"{eligibility_source}: expected an Oracle-ID list or an object containing oracle_ids")
        return cls(cards, source, eligible_ids)

    def __len__(self) -> int:
        return len(self._by_oracle_id)

    def resolve(self, name: str, oracle_id: str | None = None) -> Mapping[str, Any] | None:
        if oracle_id and str(oracle_id) in self._by_oracle_id:
            return self._by_oracle_id[str(oracle_id)]
        key = oracle_name_key(name)
        candidates = self._exact.get(key) or self._face_aliases.get(key) or []
        return candidates[0] if candidates else None

    def name_matches(self, card: Mapping[str, Any], name: str) -> bool:
        key = oracle_name_key(name)
        if key == oracle_name_key(card.get("name", "")):
            return True
        return any(key == oracle_name_key(face.get("name", "")) for face in card.get("card_faces") or [] if isinstance(face, Mapping))

    def can_be_sole_commander(self, card: Mapping[str, Any]) -> bool:
        return is_individual_commander(card, self.commander_eligible_oracle_ids)

    def can_be_commander_pair(self, first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        return is_valid_commander_pair(first, second, self.commander_eligible_oracle_ids)


_COLOR_BITS = {"W": 1, "U": 2, "B": 4, "R": 8, "G": 16}


def _color_mask(colors: Iterable[str]) -> int:
    mask = 0
    for color in colors:
        mask |= _COLOR_BITS.get(str(color).upper(), 0)
    return mask


class CommanderCandidateIndex:
    """Vectorized legal-mainboard masks for a fixed model vocabulary."""

    def __init__(self, catalog: OracleCatalog, vocab: Mapping[str, int]) -> None:
        self.catalog = catalog
        self.vocab = dict(vocab)
        self.card_color_masks = np.zeros(len(vocab), dtype=np.uint8)
        self.commander_legal = np.zeros(len(vocab), dtype=bool)
        self.oracle_id_to_indices: dict[str, list[int]] = defaultdict(list)
        self.token_cards: dict[str, Mapping[str, Any]] = {}

        for token, index in vocab.items():
            if token in {PAD_TOKEN, UNK_TOKEN}:
                continue
            oracle_id = (
                token[len(ORACLE_TOKEN_PREFIX) :]
                if token.startswith(ORACLE_TOKEN_PREFIX)
                else None
            )
            card = catalog.resolve(token, oracle_id)
            if card is None:
                if oracle_id is not None:
                    raise ValueError(
                        f"Vocabulary references an unknown Oracle ID: {oracle_id}"
                    )
                continue
            self.token_cards[token] = card
            self.card_color_masks[index] = _color_mask(card.get("color_identity") or [])
            self.commander_legal[index] = card.get("legalities", {}).get("commander") == "legal"
            self.oracle_id_to_indices[str(card.get("oracle_id", ""))].append(index)

    def allowed_mask(self, commanders: Mapping | Sequence[Mapping | str]) -> np.ndarray:
        """Return candidates legal for the supplied one- or two-card command zone."""

        raw_commanders = commanders.get("commanders", []) if isinstance(commanders, Mapping) else commanders
        resolved: list[Mapping[str, Any]] = []
        for item in raw_commanders:
            if isinstance(item, Mapping):
                name = str(item.get("name", ""))
                oracle_id = item.get("oracle_id")
            else:
                name = str(item)
                oracle_id = None
            card = self.catalog.resolve(name, str(oracle_id) if oracle_id else None)
            if card is None:
                raise ValueError(f"commander was not found in Oracle catalog: {name}")
            resolved.append(card)

        valid_setup = (
            len(resolved) == 1 and self.catalog.can_be_sole_commander(resolved[0])
        ) or (
            len(resolved) == 2 and self.catalog.can_be_commander_pair(resolved[0], resolved[1])
        )
        if not valid_setup:
            names = ", ".join(str(card.get("name", "")) for card in resolved)
            raise ValueError(f"invalid Commander command zone: {names or '<empty>'}")

        identity = _color_mask(color for card in resolved for color in card.get("color_identity") or [])
        outside_identity = np.uint8(31 ^ identity)
        allowed = self.commander_legal & ((self.card_color_masks & outside_identity) == 0)
        allowed = allowed.copy()
        for commander in resolved:
            for index in self.oracle_id_to_indices.get(str(commander.get("oracle_id", "")), []):
                allowed[index] = False
        for special in (PAD_TOKEN, UNK_TOKEN):
            if special in self.vocab:
                allowed[self.vocab[special]] = False
        return allowed


@dataclass(frozen=True)
class LegalityIssue:
    code: str
    message: str
    zone: str | None = None
    card: str | None = None

    def to_dict(self) -> dict[str, str]:
        result = {"code": self.code, "message": self.message}
        if self.zone is not None:
            result["zone"] = self.zone
        if self.card is not None:
            result["card"] = self.card
        return result


@dataclass
class DeckLegalityResult:
    cleaned_deck: dict[str, Any]
    issues: list[LegalityIssue]

    @property
    def legal(self) -> bool:
        return not self.issues

    @property
    def reason_codes(self) -> set[str]:
        return {issue.code for issue in self.issues}


def _valid_iso_date(value: Any) -> bool:
    if value in (None, ""):
        return True
    try:
        return date.fromisoformat(str(value)).isoformat() == str(value)
    except ValueError:
        return False


def validate_commander_deck(
    deck: Mapping[str, Any],
    catalog: OracleCatalog,
    *,
    annotate_oracle_ids: bool = True,
    allow_sideboard: bool = False,
) -> DeckLegalityResult:
    """Resolve and validate one canonical deck without altering the input."""

    cleaned = copy.deepcopy(dict(deck))
    issues: list[LegalityIssue] = []
    seen_issues: set[tuple[str, str | None, str | None]] = set()

    def add(code: str, message: str, zone: str | None = None, card: str | None = None) -> None:
        identity = (code, zone, card)
        if identity not in seen_issues:
            issues.append(LegalityIssue(code, message, zone, card))
            seen_issues.add(identity)

    if str(deck.get("format", "commander")).casefold() not in SUPPORTED_COMMANDER_FORMATS:
        add("wrong_format", f"format is {deck.get('format')!r}, not Commander")
    if not _valid_iso_date(deck.get("date")):
        add("invalid_date", "date must be null or ISO YYYY-MM-DD")

    resolved: dict[str, list[tuple[dict[str, Any], Mapping[str, Any] | None]]] = {}
    for zone in ("commanders", "mainboard", "sideboard"):
        zone_rows: list[tuple[dict[str, Any], Mapping[str, Any] | None]] = []
        cleaned_rows: list[dict[str, Any]] = []
        for raw_item in deck.get(zone, []) if isinstance(deck.get(zone, []), list) else []:
            item = dict(raw_item)
            name = str(item.get("name", ""))
            oracle_id = item.get("oracle_id")
            card = catalog.resolve(name, str(oracle_id) if oracle_id else None)
            if card is None:
                add("unknown_card", "card name or Oracle ID was not found in the snapshot", zone, name)
                cleaned_rows.append(item)
            else:
                if oracle_id and not catalog.name_matches(card, name):
                    add("oracle_name_mismatch", "stored Oracle ID does not match the card name", zone, name)
                clean_item = {"name": str(card["name"]), "quantity": int(item.get("quantity", 1))}
                if annotate_oracle_ids:
                    clean_item["oracle_id"] = str(card["oracle_id"])
                cleaned_rows.append(clean_item)
            zone_rows.append((item, card))
        cleaned[zone] = cleaned_rows
        resolved[zone] = zone_rows

    commander_quantity = sum(int(item.get("quantity", 1)) for item, _card in resolved["commanders"])
    mainboard_quantity = sum(int(item.get("quantity", 1)) for item, _card in resolved["mainboard"])
    sideboard_quantity = sum(int(item.get("quantity", 1)) for item, _card in resolved["sideboard"])
    if commander_quantity + mainboard_quantity != 100:
        add("invalid_deck_size", f"commander plus mainboard quantity is {commander_quantity + mainboard_quantity}, expected 100")
    if commander_quantity not in (1, 2) or len(resolved["commanders"]) not in (1, 2):
        add("invalid_commander_count", f"found {len(resolved['commanders'])} commander entries with total quantity {commander_quantity}")
    if any(int(item.get("quantity", 1)) != 1 for item, _card in resolved["commanders"]):
        add("invalid_commander_quantity", "each designated commander must have quantity one")
    if sideboard_quantity and not allow_sideboard:
        add("sideboard_not_supported", f"found {sideboard_quantity} sideboard cards")

    for zone in ("commanders", "mainboard", "sideboard"):
        for item, card in resolved[zone]:
            if card is None:
                continue
            name = str(item.get("name", ""))
            legality = card.get("legalities", {}).get("commander")
            if legality == "banned":
                add("banned_card", "card is banned in Commander in this snapshot", zone, name)
            elif legality != "legal":
                add("card_not_commander_legal", f"Commander legality is {legality or 'missing'}", zone, name)
    resolved_commanders = [card for _item, card in resolved["commanders"] if card is not None]
    commander_setup_valid = False
    if len(resolved_commanders) == len(resolved["commanders"]) == 1 and commander_quantity == 1:
        commander_setup_valid = catalog.can_be_sole_commander(resolved_commanders[0])
        if not commander_setup_valid:
            add("invalid_commander", f"{resolved_commanders[0].get('name')} cannot be a sole commander", "commanders", str(resolved_commanders[0].get("name")))
    elif len(resolved_commanders) == len(resolved["commanders"]) == 2 and commander_quantity == 2:
        commander_setup_valid = catalog.can_be_commander_pair(resolved_commanders[0], resolved_commanders[1])
        if not commander_setup_valid:
            add("invalid_commander_pair", "the two designated cards do not form a legal commander pair")

    commander_ids = {str(card.get("oracle_id")) for card in resolved_commanders}
    mainboard_ids = {str(card.get("oracle_id")) for _item, card in resolved["mainboard"] if card is not None}
    for duplicate_id in sorted(commander_ids & mainboard_ids):
        duplicate = next(card for card in resolved_commanders if str(card.get("oracle_id")) == duplicate_id)
        add("commander_in_mainboard", "a designated commander also appears in the mainboard", "mainboard", str(duplicate.get("name")))

    copies: Counter[str] = Counter()
    cards_by_id: dict[str, Mapping[str, Any]] = {}
    for zone in ("commanders", "mainboard"):
        for item, card in resolved[zone]:
            if card is None:
                continue
            oracle_id = str(card.get("oracle_id"))
            copies[oracle_id] += int(item.get("quantity", 1))
            cards_by_id[oracle_id] = card
    for oracle_id, quantity in copies.items():
        card = cards_by_id[oracle_id]
        maximum = allowed_copy_count(card)
        if maximum is not None and quantity > maximum:
            add("singleton_violation", f"found {quantity} copies; maximum is {maximum}", card=str(card.get("name")))

    if commander_setup_valid:
        commander_identity = set().union(*(set(card.get("color_identity") or []) for card in resolved_commanders))
        for item, card in resolved["mainboard"]:
            if card is None:
                continue
            outside = set(card.get("color_identity") or []) - commander_identity
            if outside:
                add(
                    "color_identity_violation",
                    f"card uses {sorted(outside)} outside commander identity {sorted(commander_identity)}",
                    "mainboard",
                    str(item.get("name", "")),
                )

    return DeckLegalityResult(cleaned, issues)


def oracle_deck_fingerprint(result: DeckLegalityResult) -> tuple:
    """Return a print-independent identity for an already resolved clean deck."""

    def zone_rows(zone: str) -> tuple:
        return tuple(sorted((str(item.get("oracle_id") or oracle_name_key(item.get("name", ""))), int(item.get("quantity", 1))) for item in result.cleaned_deck.get(zone, [])))

    return zone_rows("commanders"), zone_rows("mainboard")
