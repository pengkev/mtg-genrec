import csv
from pathlib import Path

import pytest
import torch

from demo.adapter import generate, resolve_device_mode
from mtgdeck.data import PAD_TOKEN, UNK_TOKEN
from mtgdeck.inference import prepare_request, recommend, resolve_partial_deck
from mtgdeck.legality import CommanderCandidateIndex, OracleCatalog
from mtgdeck.vae import Card2VecAttentionVAE


@pytest.fixture
def bundle():
    def card(name, colors=(), legal="legal", commander=False, text=""):
        return {"name": name, "oracle_id": name.lower(), "color_identity": list(colors),
                "legalities": {"commander": legal}, "oracle_text": text,
                "type_line": "Legendary Creature" if commander else "Artifact"}
    cards = [card("Leader", "U", commander=True), card("Partner", "U", commander=True),
             card("Present"), card("Allowed", "U"), card("Red", "R"),
             card("Banned", legal="banned"), card("Untrained", "U")]
    catalog = OracleCatalog(cards)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, **{"oid:" + c["oracle_id"]: i + 2 for i, c in enumerate(cards[:-1])}}
    torch.manual_seed(4)
    model = Card2VecAttentionVAE(torch.randn(len(vocab), 12), model_dim=16, num_heads=4,
                               num_layers=1, latent_dim=4, num_pool_queries=2, num_decoder_queries=2).eval()
    index = CommanderCandidateIndex(catalog, vocab)
    return {"model": model, "vocab": vocab, "inverse_vocab": {v: k for k, v in vocab.items()},
            "catalog": catalog, "candidate_index": index, "token_cards": index.token_cards,
            "device": torch.device("cpu")}


def test_parser_preserves_sections_aliases_quantities_and_warnings(bundle):
    partial, unresolved, oov = resolve_partial_deck(
        bundle["catalog"], bundle["vocab"], "2 Leader",
        "Commander\nLeader\nDeck\n2x Present (SET) 12\nPresent\nLeader\nUntrained\nUnknown\nSideboard\nRed",
    )
    assert [(c["display_name"], c["quantity"]) for c in partial["commanders"]] == [("Leader", 1)]
    assert [(c["display_name"], c["quantity"]) for c in partial["mainboard"]] == [("Present", 3), ("Untrained", 1)]
    assert unresolved == ["Unknown"]
    assert oov == ["Untrained"]


def test_adapter_agrees_with_direct_inference_and_csv(bundle):
    score = lambda name, *args: recommend(bundle, *args)
    tables, visible, status, path = generate({"model": bundle}, score, "model", "Leader", "Present\nUnknown\nUntrained", 25, False, 1, 42)
    request = prepare_request(bundle, "Leader", "Present\nUnknown\nUntrained")
    direct = recommend(bundle, request.partial, 25, False, 1, 42)
    assert tables == [list(row.values()) for row in direct]
    assert {r[1] for r in tables} == {"Allowed", "Partner"}
    assert visible[0] == ["Commander", 1, "Leader"]
    assert "Could not resolve: Unknown" in status
    assert "absent from the training vocabulary: Untrained" in status
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            exported = list(csv.DictReader(handle))
        assert [r["Card"] for r in exported] == [r["Card"] for r in direct]
        assert [float(r["Score"]) for r in exported] == [r["Score"] for r in direct]
    finally:
        Path(path).unlink()


@pytest.mark.parametrize("commander", ["Unknown", "Present", "Leader\nPartner"])
def test_invalid_commanders_clear_outputs(bundle, commander):
    result = generate({"model": bundle}, lambda name, *args: recommend(bundle, *args), "model", commander, "Present", 25, False, 1, 42)
    assert result[0] == result[1] == []
    assert result[2]
    assert result[3] is None


def test_partner_pair_and_sampling_are_preserved(bundle):
    for name in ("Leader", "Partner"):
        bundle["catalog"].resolve(name)["oracle_text"] = "Partner"
    request = prepare_request(bundle, "Leader\nPartner", "Present", sample_latent=True, draws=4)
    first = recommend(bundle, request.partial, 25, True, 4, 42)
    second = recommend(bundle, request.partial, 25, True, 4, 42)
    assert first == second
    assert [row["Card"] for row in first] == ["Allowed"]
    bundle["model"].variational = False
    assert recommend(bundle, request.partial, 25, True, 16, 12) == recommend(bundle, request.partial, 25, False, 1, 42)


@pytest.mark.parametrize("kwargs", [{"draws": 17}, {"count": 10000}, {"seed": -1}, {"seed": 1.5}])
def test_public_api_validates_slider_limits(bundle, kwargs):
    with pytest.raises(ValueError):
        prepare_request(bundle, "Leader", "Present", **kwargs)


@pytest.mark.parametrize("environment,expected", [
    ({}, "cpu"), ({"MTG_DEVICE": "cuda"}, "cuda"),
    ({"SPACES_ZERO_GPU": "true"}, "zerogpu"),
    ({"SPACES_ZERO_GPU": "1"}, "zerogpu"),
])
def test_device_mode_matches_host(environment, expected):
    assert resolve_device_mode(environment) == expected


@pytest.mark.parametrize("environment", [{"MTG_DEVICE": "invalid"}, {"SPACES_ZERO_GPU": "true", "MTG_DEVICE": "cpu"}])
def test_device_mode_rejects_incompatible_configuration(environment):
    with pytest.raises(ValueError):
        resolve_device_mode(environment)
