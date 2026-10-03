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


def test_commander_choices_include_backgrounds_and_exclude_banned(bundle):
    cards = [dict(bundle['catalog'].resolve(name)) for name in ('Leader', 'Partner', 'Present')]
    cards += [dict(cards[0], name='Banned leader', oracle_id='banned', legalities={'commander': 'banned'}),
              dict(cards[2], name='A Background', oracle_id='background', type_line='Legendary Enchantment — Background')]
    catalog = OracleCatalog(cards, commander_eligible_oracle_ids=['leader'])
    assert catalog.commander_choices() == ['A Background', 'Leader']
    choices = catalog.commander_choices()
    choices.clear()
    assert catalog.commander_choices() == ['A Background', 'Leader']


def test_card_image_uses_normal_faces_and_missing_fallback():
    from demo.adapter import card_image
    assert card_image({'image_uris': {'normal': 'normal.jpg', 'small': 'small.jpg'}}) == 'normal.jpg'
    assert card_image({'card_faces': [{'image_uris': {'normal': 'front.jpg'}},
                                       {'image_uris': {'normal': 'back.jpg'}}]}) == 'front.jpg'
    assert card_image({'image_uris': {'small': 'small.jpg'}}) == 'small.jpg'
    assert card_image({}) is None
    assert card_image(None) is None


def test_gallery_and_selection_preserve_rank_and_details(bundle):
    from demo.adapter import recommendation_gallery, select_recommendation
    bundle['catalog'].resolve('Allowed')['image_uris'] = {'normal': 'allowed.jpg'}
    rows = [[1, 'Allowed', 0.4, 'U', 'Artifact'], [2, 'Present', -0.2, 'Colorless', 'Artifact']]
    records, gallery = recommendation_gallery(rows, bundle['catalog'])
    assert gallery[0] == ('allowed.jpg', '#1 · Allowed')
    assert gallery[1][0].shape == (336, 240, 3)
    assert gallery[1][1] == '#2 · Present · Art unavailable'
    selected, details = select_recommendation(records, 1)
    assert selected['Card'] == 'Present'
    for value in ('Present', '-0.2', 'Colorless', 'Artifact', 'Rank'):
        assert value in details
    for index in (-1, 2, None, True, (0, 1)):
        assert select_recommendation(records, index)[0] is None
    assert recommendation_gallery([], bundle['catalog']) == ([], [])


@pytest.mark.parametrize('text', ['Allowed', '2x Allowed', '1 Allowed (SET) 12', 'aLLoWeD', 'Commander\nAllowed'])
def test_add_does_not_duplicate_resolved_card(bundle, text):
    from demo.adapter import add_to_deck
    updated, message = add_to_deck(text, {'Card': 'Allowed'}, bundle['catalog'])
    assert updated == text
    assert 'already' in message


@pytest.mark.parametrize('text', ['', 'Present', 'Deck\nPresent\nSideboard\nRed', 'Commander\nLeader'])
def test_add_appends_in_mainboard_and_is_idempotent(bundle, text):
    from demo.adapter import add_to_deck
    from mtgdeck.inference import parse_deck_text
    selected = {'Card': 'Allowed'}
    updated, _ = add_to_deck(text, selected, bundle['catalog'])
    assert updated.startswith(text)
    assert parse_deck_text(updated)['mainboard']['Allowed'] == 1
    assert add_to_deck(updated, selected, bundle['catalog'])[0] == updated
    assert add_to_deck(text, None, bundle['catalog'])[0] == text


def test_add_recognizes_face_alias():
    from demo.adapter import add_to_deck
    catalog = OracleCatalog([{'name': 'Front // Back', 'oracle_id': 'dfc',
                             'card_faces': [{'name': 'Front'}, {'name': 'Back'}]}])
    assert add_to_deck('1 Front (SET) 123', {'Card': 'Front // Back'}, catalog)[0] == '1 Front (SET) 123'


@pytest.mark.parametrize('text', [
    '1 Present', 'Present', '1 Present\n1 Present',
    '2x pReSeNt (SET) 12',
])
def test_add_consolidates_copies_and_keeps_adding(bundle, text):
    from demo.adapter import add_to_deck
    bundle['catalog'].resolve('Present')['legalities']['modern'] = 'legal'
    initial_count = 2 if text.startswith(('1 Present\n', '2x')) else 1
    updated, message = add_to_deck(text, {'Card': 'Present'}, bundle['catalog'], 'modern')
    assert updated == f'{initial_count + 1} Present'
    assert 'Added' in message
    updated, _ = add_to_deck(updated, {'Card': 'Present'}, bundle['catalog'], 'modern')
    assert updated == f'{initial_count + 2} Present'
    assert add_to_deck('4 Present', {'Card': 'Present'}, bundle['catalog'], 'modern')[0] == '4 Present'


def test_add_consolidates_only_mainboard_and_preserves_sections(bundle):
    from demo.adapter import add_to_deck
    bundle['catalog'].resolve('Present')['legalities']['modern'] = 'legal'
    text = 'Deck:\n1 Present\n# keep this\n1 Present\nSideboard:\n1 Present\nSB: 1 Red'
    updated, _ = add_to_deck(text, {'Card': 'Present'}, bundle['catalog'], 'modern')
    assert updated == 'Deck:\n3 Present\n# keep this\nSideboard:\n1 Present\nSB: 1 Red'


def test_asset_export_preserves_card_art():
    from scripts.export_space_assets import ORACLE_FIELDS
    assert {'image_uris', 'card_faces'} <= ORACLE_FIELDS


def test_gradio_app_serializes_recommend_and_add_and_preserves_api(bundle, monkeypatch):
    import runpy
    import gradio as gr
    import mtgdeck.inference as inference
    from mtgdeck.legality import OracleCatalog

    bundle['checkpoint'] = {}
    monkeypatch.setattr(inference, 'available_checkpoints', lambda _: [Path('fixture.pt')])
    monkeypatch.setattr(inference, 'load_bundle', lambda *args, **kwargs: bundle)
    monkeypatch.setattr(inference, 'DEFAULT_COMMANDER', 'Leader')
    monkeypatch.setattr(OracleCatalog, 'from_path', lambda *args: bundle['catalog'])
    monkeypatch.setenv('MTG_DEVICE', 'cpu')
    monkeypatch.delenv('SPACES_ZERO_GPU', raising=False)
    namespace = runpy.run_path(str(Path(__file__).parents[1] / 'demo' / 'app.py'))
    app = namespace['demo']
    functions = {event.fn.__name__: event for event in app.fns.values() if event.fn}
    for name in ('submit_request', 'visual_request', 'select_card', 'add_request'):
        assert functions[name].concurrency_id == 'inference'
        assert functions[name].concurrency_limit == 1
    public = functions['submit_request']
    assert public.api_name == 'recommend'
    assert len(public.inputs) == 7 and len(public.outputs) == 4
    assert isinstance(public.inputs[1], gr.Textbox)
    output = functions['visual_request'].fn('fixture.pt', ['Leader'], 'Present', 25, False, 1, 42)
    records = output[4]
    assert output[6] is None
    event = gr.SelectData(None, {'index': 0, 'value': None})
    selected, detail = functions['select_card'].fn(records, 'fixture.pt', 'Present', None, event)
    assert selected == records[0] and 'data-add' in detail
    # Adding must neither score nor replace the current gallery/selection/CSV.
    def unexpected_scoring(*args, **kwargs):
        pytest.fail('Add to deck must not run inference')

    with monkeypatch.context() as guard:
        guard.setitem(functions['add_request'].fn.__globals__, 'SCORE', unexpected_scoring)
        updated, status = functions['add_request'].fn(selected, 'Present', 'fixture.pt', None, ['Leader'], gr.EventData(None, {'quantity': 1}))
        duplicate, duplicate_status = functions['add_request'].fn(selected, updated, 'fixture.pt', None, ['Leader'], gr.EventData(None, {'quantity': 1}))
    assert functions['add_request'].outputs == [functions['visual_request'].inputs[2], functions['visual_request'].outputs[2]]
    assert f"1 {selected['Card']}" in updated
    assert 'Added' in status
    assert duplicate == updated and 'already' in duplicate_status
    next_output = functions['visual_request'].fn('fixture.pt', ['Leader'], updated, 25, False, 1, 42)
    assert selected['Card'] not in [row['Card'] for row in next_output[4]]
    assert next_output[6] is None
    invalid = functions['visual_request'].fn('fixture.pt', [], 'Present', 25, False, 1, 42)
    assert invalid[4] == [] and invalid[3] is None and invalid[6] is None
    app.close()


@pytest.mark.parametrize("format_name", ["modern", "legacy"])
def test_constructed_inference_uses_format_legality_without_commander(bundle, format_name):
    from mtgdeck.inference import ConstructedCandidateIndex
    for card in bundle["token_cards"].values():
        card["legalities"][format_name] = "legal" if card["name"] in {"Present", "Red"} else "banned"
    index = ConstructedCandidateIndex(bundle["catalog"], bundle["vocab"], format_name)
    bundle.update(format=format_name, candidate_index=index, token_cards=index.token_cards)
    # A stale hidden Commander value must not enter the constructed encoder.
    request = prepare_request(bundle, "Leader", "4 Present")
    assert request.partial["format"] == format_name
    assert request.partial["commanders"] == []
    assert request.visible == [["Mainboard", 4, "Present"]]
    rows = recommend(bundle, request.partial, 25, False, 1, 42)
    assert [row["Card"] for row in rows] == ["Red"]
    with pytest.raises(ValueError, match="no command zone"):
        prepare_request(bundle, "", "Commander\nLeader\nDeck\nPresent")


def test_companion_request_filters_recommendations_and_rejects_invalid_partial(bundle):
    from mtgdeck.inference import ConstructedCandidateIndex
    companion = {'name': 'Keruga, the Macrosage', 'oracle_id': 'keruga', 'cmc': 5,
                 'type_line': 'Legendary Creature', 'legalities': {'modern': 'legal'}}
    cards = [dict(card) for card in bundle['token_cards'].values()] + [companion]
    for card in cards:
        card['legalities']['modern'] = 'legal'
        card['cmc'] = 3 if card['name'] == 'Red' else 1
    companion['cmc'] = 5
    catalog = OracleCatalog(cards)
    index = ConstructedCandidateIndex(catalog, bundle['vocab'], 'modern')
    bundle.update(format='modern', catalog=catalog, candidate_index=index, token_cards=index.token_cards)
    request = prepare_request(bundle, '', 'Red', companion_text=companion['name'])
    assert recommend(bundle, request.partial, 25, False, 1, 42) == []
    with pytest.raises(ValueError, match='violates'):
        prepare_request(bundle, '', 'Present', companion_text=companion['name'])
    with pytest.raises(ValueError, match='unknown cards'):
        prepare_request(bundle, '', 'Red\nUnknown', companion_text=companion['name'])


def test_card_popup_renders_symbols_and_simple_add(bundle):
    from demo.adapter import card_modal
    card = bundle['catalog'].resolve('Allowed')
    card.update(mana_cost='{2}{U/B}', oracle_text='Pay {T}: Draw a card.\n<script>literal</script>')
    modal = card_modal({'Card': 'Allowed', 'Rank': 1, 'Score': .4}, bundle['catalog'], '', 'commander')
    assert 'card-symbols/UB.svg' in modal and 'card-symbols/T.svg' in modal
    assert 'alt="{U/B}"' in modal
    assert '&lt;script&gt;literal&lt;/script&gt;' in modal
    assert 'data-card-name="Allowed"' in modal
    assert 'data-add' in modal
    assert 'name="copies"' not in modal
    assert 'copies available' not in modal
    assert 'Mana value:' not in modal and 'Color identity:' not in modal


def test_add_button_stays_enabled_at_copy_limit(bundle):
    from demo.adapter import card_modal
    from demo.card_dialog import DIALOG_JS
    modal = card_modal({'Card': 'Allowed'}, bundle['catalog'], '1 Allowed', 'commander')
    assert '<button type="button" data-add>Add to deck</button>' in modal
    assert 'button.disabled = true' not in DIALOG_JS
