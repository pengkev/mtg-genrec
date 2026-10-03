"""Gradio output formatting; all card resolution and scoring live in mtgdeck."""

from __future__ import annotations

import csv
import html
import re
import tempfile
from typing import Callable, Mapping, Any

from mtgdeck.inference import (
    COMMANDER_SECTIONS, IGNORED_SECTIONS, MAINBOARD_SECTIONS,
    RECOMMENDATION_COLUMNS, SET_SUFFIX, parse_deck_text, parse_deck_zones, prepare_request, resolve_partial_deck,
)
from mtgdeck.legality import OracleCatalog
from mtgdeck.deck_rules import copy_limit, companion_errors, card_types


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


def remaining_copies(deck, card, catalog, format_name="commander", companion=None):
    zones = parse_deck_zones(deck)
    total = 0
    mainboard_total = 0
    for zone in ("mainboard", "sideboard", "commanders"):
        for name, quantity in zones[zone].items():
            existing = catalog.resolve(name) or catalog.resolve(SET_SUFFIX.sub("", name).strip())
            if existing and existing["oracle_id"] == card["oracle_id"]:
                total += quantity
                if zone == "mainboard":
                    mainboard_total += quantity
    # A selected companion occupies one sideboard slot; an explicit sideboard
    # copy already represents that slot and must not be counted twice.
    if companion and companion['oracle_id'] == card['oracle_id']:
        if not any((catalog.resolve(name) or catalog.resolve(SET_SUFFIX.sub('', name).strip()) or {}).get('oracle_id') == card['oracle_id'] for name in zones['sideboard']):
            total += 1
    limit = copy_limit(card, format_name)
    remaining = max(0, limit - total) if limit is not None else 100
    if companion and companion['name'] == 'Lutri, the Spellchaser' and 'land' not in card_types(card):
        remaining = min(remaining, max(0, 1 - mainboard_total))
    return remaining


def add_to_deck(deck: str, selected, catalog: OracleCatalog, format_name="commander",
                quantity=1, companion_text="", commander_text="") -> tuple[str, str]:
    """Add copies while preserving pasted sections and enforcing construction rules."""
    if not selected:
        return deck, "Select a recommendation first."
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) or int(quantity) != quantity or not 1 <= quantity <= 100:
        return deck, "Choose a whole number of copies between 1 and 100."
    quantity = int(quantity)
    card = catalog.resolve(selected["Card"])
    if card is None:
        return deck, "The selected card is missing from the Oracle catalog."
    try:
        partial, unresolved, _ = resolve_partial_deck(catalog, {}, commander_text, deck, format_name, companion_text)
    except ValueError as exc:
        return deck, str(exc)
    companion = partial.get('companion')
    remaining = remaining_copies(deck, card, catalog, format_name, companion)
    if quantity > remaining:
        if format_name == 'commander' and remaining == 0:
            return deck, f"{card['name']} is already in the deck text; no copy added."
        return deck, f"Only {remaining} more copies of {card['name']} can be added under the deck's copy limit."
    if card.get('legalities', {}).get(format_name) != 'legal':
        return deck, f"{card['name']} is not legal in {format_name.title()} in this snapshot."
    if format_name == 'commander' and partial['commanders']:
        from mtgdeck.legality import CommanderCandidateIndex
        from mtgdeck.data import PAD_TOKEN, UNK_TOKEN, ORACLE_TOKEN_PREFIX
        token = ORACLE_TOKEN_PREFIX + card['oracle_id'].casefold()
        candidates = CommanderCandidateIndex(catalog, {PAD_TOKEN: 0, UNK_TOKEN: 1, token: 2})
        try:
            if not candidates.allowed_mask(partial)[2]:
                return deck, "The card cannot be added under the commanders' color identity or command-zone rules."
        except ValueError as exc:
            return deck, str(exc)
    if companion:
        if unresolved:
            return deck, "Resolve unknown cards before checking companion constraints: " + ', '.join(unresolved)
        starting = [(catalog.resolve('', item['oracle_id']), item['quantity'])
                    for zone in ('commanders', 'mainboard') for item in partial[zone]]
        # Include the addition when determining Umori's common type.
        existing = next((i for i, (c, _) in enumerate(starting) if c['oracle_id'] == card['oracle_id']), None)
        if existing is None:
            starting.append((card, quantity))
        else:
            starting[existing] = (card, starting[existing][1] + quantity)
        errors = companion_errors(companion, starting)
        if errors:
            return deck, f"This would violate {companion['name']}'s companion requirement: " + ', '.join(errors)
    zone = "mainboard"
    lines = deck.splitlines(keepends=True)
    matching_lines = []
    existing_quantity = 0
    for index, line in enumerate(lines):
        heading = line.strip().casefold().rstrip(":")
        if heading in MAINBOARD_SECTIONS:
            zone = "mainboard"
        elif heading in COMMANDER_SECTIONS or heading in IGNORED_SECTIONS:
            zone = "other"
        elif zone == "mainboard":
            for name, count in parse_deck_zones(line)["mainboard"].items():
                existing = catalog.resolve(name) or catalog.resolve(SET_SUFFIX.sub("", name).strip())
                if existing and existing["oracle_id"] == card["oracle_id"]:
                    matching_lines.append(index)
                    existing_quantity += count
    if matching_lines:
        first = matching_lines[0]
        ending = "\r\n" if lines[first].endswith("\r\n") else "\n" if lines[first].endswith("\n") else ""
        lines[first] = f"{existing_quantity + quantity} {card['name']}" + ending
        for index in reversed(matching_lines[1:]):
            del lines[index]
        updated = "".join(lines)
        if not deck.endswith("\n"):
            updated = updated.rstrip("\r\n")
        return updated, f"Added {quantity} × {card['name']}."
    separator = "" if not deck or deck.endswith("\n") else "\n"
    prefix = "Deck\n" if zone != "mainboard" else ""
    return deck + separator + prefix + f"{quantity} {card['name']}", f"Added {quantity} × {card['name']}."


def render_symbols(text):
    """Escape Oracle text and replace brace notation with Scryfall SVG symbols."""
    return re.sub(r"\{([A-Z0-9/]+)\}", lambda match:
                  f'<img class="mana-symbol" src="https://svgs.scryfall.io/card-symbols/{match[1].replace("/", "")}.svg" '
                  f'alt="{match[0]}" title="{match[0]}">', html.escape(str(text), quote=True)).replace('\n', '<br>')


def card_modal(selected, catalog, deck, format_name, companion=None, message=""):
    """Escaped card details and a single-copy add action inside the browser's dialog."""
    if not selected:
        return ""
    card = catalog.resolve(selected['Card']) or {}
    escape = lambda value: html.escape(str(value), quote=True)
    title = escape(card.get('name', selected['Card']))
    faces = card.get('card_faces') or [card]
    details = []
    for face_index, face in enumerate(faces):
        image = card_image(face) or card_image(card)
        picture = f'<img src="{escape(image)}" alt="{escape(face.get("name", title))}" loading="lazy">' if image else ''
        text = render_symbols(face.get('oracle_text', card.get('oracle_text', '')))
        stats = ' / '.join(str(face[k]) for k in ('power', 'toughness') if k in face)
        if face.get('loyalty') is not None:
            stats = 'Loyalty: ' + str(face['loyalty'])
        heading = f'<h3>{escape(face.get("name", ""))}</h3>' if len(faces) > 1 else ''
        details.append(f'<section class="card-face{" no-art" if not picture else ""}">{picture}<div>{heading}'
                       f'<p class="mana-cost">{render_symbols(face.get("mana_cost", card.get("mana_cost", "")))}</p>'
                       f'<p>{escape(face.get("type_line", card.get("type_line", "")))}</p>'
                       f'<p class="oracle-text" data-face="{face_index}">{text}</p><p>{escape(stats)}</p></div></section>')
    if companion is None:
        entries = parse_deck_zones(deck)['companion']
        if len(entries) == 1:
            companion = catalog.resolve(next(iter(entries)))
    remaining = remaining_copies(deck, card, catalog, format_name, companion)
    message = message or ('Already in deck' if remaining == 0 else '')
    return (f'<header data-card-name="{title}"><h2 id="card-dialog-title">{title}</h2><button type="button" data-close aria-label="Close card details">✕</button></header>'
            + ''.join(details)
            + f'<footer><button type="button" data-add>Add to deck</button>'
            + f'<p class="modal-message" role="status">{escape(message)}</p></footer>')


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
    companion: str = "",
) -> tuple[list[list[Any]], list[list[Any]], str, str | None]:
    """One queued request, including a unique per-request downloadable CSV."""
    try:
        if checkpoint not in bundles:
            raise ValueError("Select an available checkpoint.")
        request = prepare_request(bundles[checkpoint], commander, deck, count, sample, draws, seed, companion_text=companion)
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
