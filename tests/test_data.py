from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from mtgdeck.data import (
    build_vocabulary,
    candidate_mask,
    canonical_validation_errors,
    deck_fingerprint,
    deduplicate_decks,
    mask_deck,
    normalize_card_name,
    normalize_deck_record,
    split_decks,
)


def moxfield_record(deck_id="abc"):
    return {
        "id": deck_id,
        "name": "Example",
        "user_bracket": 3,
        "auto_bracket": 4,
        "hubs": ["Primer"],
        "mainboard": [{"n": "Sol Ring", "q": 1}, {"n": "Island", "q": 12}],
        "commanders": [{"n": "Éowyn, Shieldmaiden", "q": 1}],
    }


def mtgtop8_record(deck_id="123"):
    return {
        "deck_id": deck_id,
        "deck_url": f"https://www.mtgtop8.com/event?e=9&d={deck_id}&f=cEDH",
        "date": "02/03/26",
        "placement": 2,
        "players": 30,
        "main": [{"name": "Fire // Ice", "qty": 1}],
        "cmds": [{"name": "Tymna the Weaver", "qty": 1}],
    }


def test_card_normalization_handles_case_unicode_whitespace_and_faces():
    assert normalize_card_name("  ÉOWYN,   Shieldmaiden ") == "eowyn, shieldmaiden"
    assert normalize_card_name("Fire // Ice") == "fire"
    assert normalize_card_name("Fire / Ice") == "fire"
    assert normalize_card_name("Fire/Ice") == "fire"


def test_old_moxfield_schema_conversion_preserves_quantities_and_metadata():
    deck = normalize_deck_record(moxfield_record())
    assert canonical_validation_errors(deck) == []
    assert deck["deck_id"] == "moxfield:abc"
    assert deck["mainboard"][1]["quantity"] == 12
    assert deck["metadata"]["user_bracket"] == 3


def test_old_mtgtop8_schema_conversion_preserves_event_fields():
    deck = normalize_deck_record(mtgtop8_record())
    assert deck["deck_id"] == "mtgtop8:123"
    assert deck["date"] == "2026-03-02"
    assert deck["metadata"]["placement"] == 2
    assert deck["commanders"][0]["name"] == "Tymna the Weaver"


def test_fingerprint_is_deterministic_and_deduplication_uses_content():
    first = normalize_deck_record(moxfield_record())
    reordered = deepcopy(first)
    reordered["mainboard"].reverse()
    cross_source = deepcopy(reordered)
    cross_source.update(deck_id="local:different", source="local", source_id="different")
    assert deck_fingerprint(first) == deck_fingerprint(reordered)
    assert len(deduplicate_decks([first, cross_source])) == 1


def test_vocabulary_split_mask_and_candidate_mask_are_reproducible():
    decks = [normalize_deck_record(moxfield_record(str(index))) for index in range(10)]
    for index, deck in enumerate(decks):
        deck["mainboard"].append({"name": f"Unique Card {index}", "quantity": 1})
    vocab = build_vocabulary(decks)
    assert "sol ring" in vocab and "eowyn, shieldmaiden" in vocab
    first = split_decks(decks, seed=7)
    second = split_decks(decks, seed=7)
    assert [[d["deck_id"] for d in part] for part in first] == [[d["deck_id"] for d in part] for part in second]

    visible, hidden = mask_deck(decks[0], mask_ratio=0.5, seed=3)
    assert hidden
    assert visible["commanders"] == decks[0]["commanders"]
    assert decks[0]["mainboard"][1]["quantity"] == 12
    allowed = candidate_mask(["Sol Ring"], vocab)
    assert isinstance(allowed, np.ndarray) and not allowed[vocab["sol ring"]]


def test_mask_deck_supports_fixed_variable_ratio_schedule_values():
    deck = normalize_deck_record({
        "id": "ratios",
        "commanders": [{"n": "Commander", "q": 1}],
        "mainboard": [{"n": f"Card {index}", "q": 1} for index in range(10)],
    }, "local")
    expected = {0.1: 1, 0.2: 2, 0.3: 3, 0.4: 4, 0.5: 5}
    for ratio, hidden_count in expected.items():
        first = mask_deck(deck, mask_ratio=ratio, seed=17)
        second = mask_deck(deck, mask_ratio=ratio, seed=17)
        assert first == second
        assert len(first[1]) == hidden_count
        assert first[0]["commanders"] == deck["commanders"]


def test_preparation_imports_both_old_schemas(tmp_path):
    from scripts.curate_decks import iter_input_decks
    paths = [tmp_path / "moxfield.jsonl", tmp_path / "mtgtop8.jsonl"]
    for path, row in zip(paths, [moxfield_record(), mtgtop8_record()]):
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    rows = list(iter_input_decks(paths))
    assert {row["source"] for row in rows} == {"moxfield", "mtgtop8"}
    assert rows == [normalize_deck_record(moxfield_record()), normalize_deck_record(mtgtop8_record())]


def test_normalization_preserves_companions():
    original = normalize_deck_record(moxfield_record())
    original["companions"] = [{"name": "Gyruda, Doom of Depths", "quantity": 1}]
    assert normalize_deck_record(original)["companions"] == original["companions"]
