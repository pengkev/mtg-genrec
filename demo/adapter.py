"""Gradio output formatting; all card resolution and scoring live in mtgdeck."""

from __future__ import annotations

import csv
import html
import tempfile
from typing import Callable, Mapping, Any

from mtgdeck.inference import (
    COMMANDER_SECTIONS, IGNORED_SECTIONS, MAINBOARD_SECTIONS,
    RECOMMENDATION_COLUMNS, SET_SUFFIX, parse_deck_text, prepare_request,
)
from mtgdeck.legality import OracleCatalog


def card_image(card: Mapping[str, Any] | None) -> str | None:
    """Use the shipped image metadata, never a live card lookup."""
    card = card or {}
    sources = [card, *(card.get("card_faces") or [])]
    for source in sources:
        images = source.get("image_uris") or {}
        for size in ("normal", "large", "small", "png"):
            if images.get(size):
                return images[size]
    return None


def recommendation_gallery(rows, catalog: OracleCatalog):
    """Keep every rank selectable, even when the snapshot lacks its image."""
    import numpy as np

    records = [dict(zip(RECOMMENDATION_COLUMNS, row)) for row in rows]
    gallery = []
    for row in records:
        uri = card_image(catalog.resolve(row["Card"]))
        caption = f"#{row['Rank']} · {row['Card']}"
        # A neutral card-shaped placeholder requires no asset or network call.
        gallery.append((uri if uri else np.full((336, 240, 3), 48, dtype=np.uint8),
                        caption if uri else caption + " · Art unavailable"))
    return records, gallery


def select_recommendation(records, index):
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(records):
        return None, "Select a card to inspect it."
    card = dict(records[index])
    # HTML escaping keeps card names and Oracle type lines literal.
    detail = "<br>".join(
        f"<strong>{label}:</strong> {html.escape(str(card[key]))}"
        for label, key in (("Card", "Card"), ("Rank", "Rank"), ("Model score", "Score"),
                           ("Color identity", "Color identity"), ("Type", "Type"))
    )
    return card, detail


def add_to_deck(deck: str, selected, catalog: OracleCatalog) -> tuple[str, str]:
    """Append one mainboard card; preserve text and avoid duplicate identities."""
    if not selected:
        return deck, "Select a recommendation first."
    card = catalog.resolve(selected["Card"])
    if card is None:
        return deck, "The selected card is missing from the Oracle catalog."
    parsed = parse_deck_text(deck)
    for zone in parsed.values():
        for name in zone:
            existing = catalog.resolve(name) or catalog.resolve(SET_SUFFIX.sub("", name).strip())
            if existing and existing["oracle_id"] == card["oracle_id"]:
                return deck, f"{card['name']} is already in the deck text; no copy added."
    # A trailing sideboard/commander section must not swallow the new card.
    zone = "mainboard"
    for line in deck.splitlines():
        heading = line.strip().casefold()
        if heading in MAINBOARD_SECTIONS:
            zone = "mainboard"
        elif heading in COMMANDER_SECTIONS or heading in IGNORED_SECTIONS:
            zone = "other"
    separator = "" if not deck or deck.endswith("\n") else "\n"
    prefix = "Deck\n" if zone != "mainboard" else ""
    return deck + separator + prefix + f"1 {card['name']}", f"Added {card['name']}."


def resolve_device_mode(environment: Mapping[str, str]) -> str:
    """Match serving mode to the host without requesting a GPU locally."""
    zero_gpu = environment.get("SPACES_ZERO_GPU", "").lower() in {"1", "t", "true"}
    mode = environment.get("MTG_DEVICE", "zerogpu" if zero_gpu else "cpu")
    if mode not in {"cpu", "cuda", "zerogpu"}:
        raise ValueError("MTG_DEVICE must be cpu, cuda, or zerogpu")
    if zero_gpu and mode != "zerogpu":
        raise ValueError("ZeroGPU hardware requires MTG_DEVICE=zerogpu. CPU mode requires CPU hardware.")
    return mode


def generate(
    bundles: Mapping[str, Any], score: Callable, checkpoint: str,
    commander: str, deck: str, count: int, sample: bool, draws: int, seed: int,
) -> tuple[list[list[Any]], list[list[Any]], str, str | None]:
    """One queued request, including a unique per-request downloadable CSV."""
    try:
        if checkpoint not in bundles:
            raise ValueError("Select an available checkpoint.")
        request = prepare_request(bundles[checkpoint], commander, deck, count, sample, draws, seed)
        rows = score(checkpoint, request.partial, request.count, request.sample_latent, request.draws, request.seed)
    except (ValueError, AssertionError) as exc:
        # Returning empty outputs also clears any previous request's CSV/results.
        return [], [], str(exc), None
    # The UI moves this producer file into its expiring cache, then removes it.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix="mtg_recommendations_", newline="", encoding="utf-8", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=RECOMMENDATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        output_path = handle.name
    status = "\n".join([*request.notices, f"Generated {len(rows)} recommendations."])
    return [[row[column] for column in RECOMMENDATION_COLUMNS] for row in rows], request.visible, status, output_path
