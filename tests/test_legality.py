import json
import gzip
from pathlib import Path

from mtgdeck.data import ORACLE_TOKEN_PREFIX
from mtgdeck.legality import (
    CommanderCandidateIndex,
    OracleCatalog,
    allowed_copy_count,
    is_valid_commander_pair,
    oracle_name_key,
    validate_commander_deck,
)

from scrape import metadata as FETCH


def oracle_card(
    name,
    *,
    oracle_id=None,
    type_line="Creature — Test",
    oracle_text="",
    color_identity=(),
    legality="legal",
    games=("paper",),
    card_faces=None,
):
    card = {
        "name": name,
        "oracle_id": oracle_id or f"id:{name}",
        "type_line": type_line,
        "oracle_text": oracle_text,
        "color_identity": list(color_identity),
        "legalities": {"commander": legality},
        "games": list(games),
        "lang": "en",
        "layout": "normal",
        "released_at": "2025-01-01",
    }
    if card_faces is not None:
        card["card_faces"] = card_faces
    return card


ATRAxa = oracle_card(
    "Atraxa, Praetors' Voice",
    type_line="Legendary Creature — Phyrexian Angel Horror",
    color_identity="WUBG",
)
PLAINS = oracle_card("Plains", type_line="Basic Land — Plains", color_identity="W")
ISLAND = oracle_card("Island", type_line="Basic Land — Island", color_identity="U")
MOUNTAIN = oracle_card("Mountain", type_line="Basic Land — Mountain", color_identity="R")
SOL_RING = oracle_card("Sol Ring", type_line="Artifact")
ANCESTRAL = oracle_card("Ancestral Recall", type_line="Instant", color_identity="U", legality="banned")


def deck(commanders, mainboard, **updates):
    result = {
        "schema_version": 1,
        "deck_id": "local:test",
        "source": "local",
        "source_id": "test",
        "url": None,
        "name": "Test",
        "format": "commander",
        "date": "2025-01-02",
        "commanders": [{"name": name, "quantity": quantity} for name, quantity in commanders],
        "mainboard": [{"name": name, "quantity": quantity} for name, quantity in mainboard],
        "sideboard": [],
        "metadata": {},
    }
    result.update(updates)
    return result


def reason_codes(result):
    return {issue.code for issue in result.issues}


def test_name_resolution_preserves_faces_and_prefers_legal_card_over_token():
    fire_ice = oracle_card(
        "Fire // Ice",
        oracle_id="fire-ice",
        type_line="Instant // Instant",
        color_identity="UR",
        card_faces=[{"name": "Fire", "type_line": "Instant", "oracle_text": ""}, {"name": "Ice", "type_line": "Instant", "oracle_text": ""}],
    )
    token = oracle_card("Sol Ring", oracle_id="token-sol-ring", type_line="Token Artifact", legality="not_legal")
    catalog = OracleCatalog([fire_ice, token, SOL_RING])
    assert oracle_name_key("  Fire/Ice ") == oracle_name_key("Fire // Ice")
    assert oracle_name_key("With Great Power...") == oracle_name_key("With Great Power . . .")
    assert oracle_name_key("________ Goblin") == oracle_name_key("_____ Goblin")
    assert catalog.resolve("Fire / Ice")["oracle_id"] == "fire-ice"
    assert catalog.resolve("Ice")["oracle_id"] == "fire-ice"
    assert catalog.resolve("Sol Ring")["oracle_id"] == SOL_RING["oracle_id"]


def test_catalog_streams_current_scryfall_compressed_jsonl(tmp_path):
    path = tmp_path / "oracle.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(ATRAxa) + "\n")
        handle.write(json.dumps(PLAINS) + "\n")
    catalog = OracleCatalog.from_path(path)
    assert len(catalog) == 2
    assert catalog.resolve("Atraxa, Praetors' Voice")["oracle_id"] == ATRAxa["oracle_id"]


def test_scryfall_bulk_list_selects_oracle_cards_by_type():
    payload = {
        "object": "list",
        "data": [
            {"type": "default_cards", "jsonl_download_uri": "https://example/default.gz"},
            {
                "type": "oracle_cards",
                "updated_at": "2026-08-31T21:01:56Z",
                "jsonl_download_uri": "https://example/oracle.gz",
            },
        ],
    }
    selected = FETCH.select_bulk_item(payload)
    assert selected["type"] == "oracle_cards"
    assert selected["jsonl_download_uri"] == "https://example/oracle.gz"


def test_valid_deck_is_resolved_and_annotated_without_mutating_input():
    catalog = OracleCatalog([ATRAxa, PLAINS])
    original = deck([("Atraxa, Praetors' Voice", 1)], [("Plains", 99)])
    result = validate_commander_deck(original, catalog)
    assert result.legal
    assert result.cleaned_deck["commanders"][0]["oracle_id"] == ATRAxa["oracle_id"]
    assert result.cleaned_deck["mainboard"][0]["quantity"] == 99
    assert "oracle_id" not in original["commanders"][0]


def test_hard_legality_failures_have_stable_reason_codes():
    catalog = OracleCatalog([ATRAxa, PLAINS, MOUNTAIN, SOL_RING, ANCESTRAL])
    invalid = deck(
        [("Atraxa, Praetors' Voice", 1)],
        [("Plains", 95), ("Mountain", 1), ("Sol Ring", 2), ("Ancestral Recall", 1)],
        date="not-a-date",
        sideboard=[{"name": "Plains", "quantity": 1}],
    )
    result = validate_commander_deck(invalid, catalog)
    assert {
        "invalid_date",
        "sideboard_not_supported",
        "banned_card",
        "singleton_violation",
        "color_identity_violation",
    }.issubset(reason_codes(result))


def test_commander_candidate_index_masks_color_identity_bans_and_commanders():
    catalog = OracleCatalog([ATRAxa, PLAINS, MOUNTAIN, SOL_RING, ANCESTRAL])
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        "atraxa, praetors' voice": 2,
        "plains": 3,
        "mountain": 4,
        "sol ring": 5,
        "ancestral recall": 6,
    }
    candidates = CommanderCandidateIndex(catalog, vocab)
    allowed = candidates.allowed_mask([{"name": "Atraxa, Praetors' Voice", "quantity": 1}])

    assert allowed[vocab["plains"]]
    assert allowed[vocab["sol ring"]]
    assert not allowed[vocab["mountain"]]
    assert not allowed[vocab["ancestral recall"]]
    assert not allowed[vocab["atraxa, praetors' voice"]]
    assert not allowed[vocab["<PAD>"]] and not allowed[vocab["<UNK>"]]


def test_commander_candidate_index_supports_oracle_id_vocabulary():
    catalog = OracleCatalog([ATRAxa, PLAINS, MOUNTAIN, SOL_RING, ANCESTRAL])
    token = lambda card: ORACLE_TOKEN_PREFIX + card["oracle_id"]
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
        token(ATRAxa): 2,
        token(PLAINS): 3,
        token(MOUNTAIN): 4,
        token(SOL_RING): 5,
        token(ANCESTRAL): 6,
    }
    candidates = CommanderCandidateIndex(catalog, vocab)
    allowed = candidates.allowed_mask(
        [{"name": token(ATRAxa), "oracle_id": ATRAxa["oracle_id"], "quantity": 1}]
    )

    assert allowed[vocab[token(PLAINS)]]
    assert allowed[vocab[token(SOL_RING)]]
    assert not allowed[vocab[token(MOUNTAIN)]]
    assert not allowed[vocab[token(ANCESTRAL)]]
    assert not allowed[vocab[token(ATRAxa)]]
    assert candidates.token_cards[token(SOL_RING)]["name"] == "Sol Ring"


def test_unknown_card_size_and_commander_overlap_are_rejected():
    catalog = OracleCatalog([ATRAxa, PLAINS])
    invalid = deck(
        [("Atraxa, Praetors' Voice", 1)],
        [("Atraxa, Praetors' Voice", 1), ("Plains", 97), ("Imaginary Card", 1)],
    )
    result = validate_commander_deck(invalid, catalog)
    assert {"unknown_card", "commander_in_mainboard", "singleton_violation"}.issubset(reason_codes(result))


def test_copy_limit_exceptions_are_derived_from_oracle_text():
    rats = oracle_card(
        "Relentless Rats",
        oracle_text="A deck can have any number of cards named Relentless Rats.",
        color_identity="B",
    )
    nazgul = oracle_card(
        "Nazgûl",
        oracle_text="A deck can have up to nine cards named Nazgûl.",
        color_identity="B",
    )
    assert allowed_copy_count(PLAINS) is None
    assert allowed_copy_count(rats) is None
    assert allowed_copy_count(nazgul) == 9
    assert allowed_copy_count(SOL_RING) == 1


def test_supported_two_commander_mechanics():
    tymna = oracle_card(
        "Tymna the Weaver",
        type_line="Legendary Creature — Human Cleric",
        oracle_text="Lifelink\nPartner (You can have two commanders if both have partner.)",
    )
    thrasios = oracle_card(
        "Thrasios, Triton Hero",
        type_line="Legendary Creature — Merfolk Wizard",
        oracle_text="Partner (You can have two commanders if both have partner.)",
    )
    halisin = oracle_card(
        "Halsin, Emerald Archdruid",
        type_line="Legendary Creature — Elf Druid",
        oracle_text="Choose a Background (You can have a Background as a second commander.)",
    )
    background = oracle_card("Agent of the Iron Throne", type_line="Legendary Enchantment — Background")
    doctor = oracle_card("The Tenth Doctor", type_line="Legendary Creature — Time Lord Doctor")
    rose = oracle_card(
        "Rose Tyler",
        type_line="Legendary Creature — Human",
        oracle_text="Doctor's companion (You can have two commanders if the other is the Doctor.)",
    )
    lore = oracle_card("Lore Weaver", oracle_text="Partner with Ley Weaver (When this creature enters, search.)")
    ley = oracle_card("Ley Weaver", oracle_text="Partner with Lore Weaver (When this creature enters, search.)")
    trynn = oracle_card(
        "Trynn, Champion of Freedom",
        type_line="Legendary Creature — Human Soldier",
        oracle_text="Partner with Silvar, Devourer of the Free (When this creature enters, search.)",
    )
    silvar = oracle_card(
        "Silvar, Devourer of the Free",
        type_line="Legendary Creature — Cat Nightmare",
        oracle_text="Partner with Trynn, Champion of Freedom (When this creature enters, search.)",
    )
    friend = oracle_card(
        "Sophina",
        type_line="Legendary Creature — Human",
        oracle_text="Partner—Friends forever (You can have two commanders if both have this ability.)",
    )
    other_friend = oracle_card(
        "Wernog",
        type_line="Legendary Creature — Human",
        oracle_text="Partner—Friends forever (You can have two commanders if both have this ability.)",
    )
    assert is_valid_commander_pair(tymna, thrasios)
    assert is_valid_commander_pair(halisin, background)
    assert is_valid_commander_pair(doctor, rose)
    assert is_valid_commander_pair(trynn, silvar)
    assert is_valid_commander_pair(friend, other_friend)
    assert not is_valid_commander_pair(lore, ley)
    assert not is_valid_commander_pair(tymna, friend)
    assert not is_valid_commander_pair(lore, thrasios)


def test_curation_cli_writes_accepted_rejected_and_report(tmp_path):
    from scripts.curate_decks import curate_file

    oracle_path = tmp_path / "oracle.json"
    input_path = tmp_path / "decks.jsonl"
    output_path = tmp_path / "clean.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"
    report_path = tmp_path / "report.json"
    oracle_path.write_text(json.dumps([ATRAxa, PLAINS]), encoding="utf-8")
    valid = deck([("Atraxa, Praetors' Voice", 1)], [("Plains", 99)])
    invalid = deck([], [("Plains", 99)])
    invalid["deck_id"] = "local:invalid"
    input_path.write_text("\n".join(json.dumps(row) for row in (valid, invalid)) + "\n", encoding="utf-8")

    report = curate_file(input_path, oracle_path, output_path, rejected_path, report_path, progress_every=0)
    accepted = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    rejected = [json.loads(line) for line in rejected_path.read_text(encoding="utf-8").splitlines()]
    assert len(accepted) == len(rejected) == 1
    assert report["accepted"] == report["rejected"] == 1
    assert report["reason_deck_counts"]["invalid_commander_count"] == 1
    assert rejected[0]["deck_id"] == "local:invalid"


def test_manifest_curation_filters_and_deduplicates_sources(tmp_path):
    from scripts.curate_decks import main
    oracle = tmp_path / 'oracle.json'
    oracle.write_text(json.dumps([ATRAxa, PLAINS]), encoding='utf-8')
    valid = deck([("Atraxa, Praetors' Voice", 1)], [("Plains", 99)])
    other_format = {**valid, 'format': 'modern', 'deck_id': 'local:modern'}
    first, second = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl'
    first.write_text(json.dumps(valid) + '\n' + json.dumps(other_format) + '\n', encoding='utf-8')
    second.write_text(json.dumps({**valid, 'deck_id': 'moxfield:duplicate', 'source': 'moxfield'}) + '\n', encoding='utf-8')
    manifest = tmp_path / 'subset.json'
    manifest.write_text(json.dumps({'formats': ['commander'], 'inputs': [
        {'path': 'a.jsonl'}, {'path': 'b.jsonl'}, {'path': 'missing.jsonl', 'optional': True}]}), encoding='utf-8')
    before = [p.read_bytes() for p in (first, second)]
    out = tmp_path / 'clean.jsonl'
    assert main(['--manifest', str(manifest), '--oracle-cards', str(oracle),
                 '--commander-eligibility', str(tmp_path / 'absent.json'), '--output', str(out), '--progress-every', '0']) == 0
    report = json.loads((tmp_path / 'clean_report.json').read_text())
    assert report['processed'] == 2
    assert report['accepted'] == report['oracle_resolved_duplicates'] == 1
    assert [p.read_bytes() for p in (first, second)] == before


def test_manifest_requires_inputs_and_explicit_optional_files(tmp_path):
    import pytest
    from scripts.curate_decks import load_manifest
    path = tmp_path / 'subset.json'
    path.write_text(json.dumps({'formats': ['commander'], 'inputs': [{'path': 'absent.jsonl'}]}))
    with pytest.raises(FileNotFoundError):
        load_manifest(path)
    path.write_text(json.dumps({'formats': ['commander'], 'inputs': [{'path': 'absent.jsonl', 'optional': True}]}))
    with pytest.raises(ValueError, match='No input files'):
        load_manifest(path)


def test_curation_refuses_to_overwrite_source_or_metadata(tmp_path):
    import pytest
    from scripts.curate_decks import curate_file
    source = tmp_path / 'source.jsonl'
    source.write_text('{}\n')
    oracle = tmp_path / 'oracle.json'
    oracle.write_text('[]')
    for output in (source, oracle):
        with pytest.raises(ValueError, match='cannot overwrite'):
            curate_file(source, oracle, output, tmp_path / 'rejected.jsonl', tmp_path / 'report.json')
    assert source.read_text() == '{}\n'
    assert oracle.read_text() == '[]'


def test_default_oracle_snapshot_includes_refreshed_stable_filename(tmp_path):
    from mtgdeck.metadata import default_oracle_path
    dated = tmp_path / 'oracle_cards_2026-08-27.jsonl.gz'
    dated.touch()
    assert default_oracle_path(tmp_path) == dated
    current = tmp_path / 'oracle_cards.jsonl.gz'
    current.touch()
    assert default_oracle_path(tmp_path) == current
