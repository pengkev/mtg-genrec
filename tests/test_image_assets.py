"""The deployed Oracle export must carry art, not just inference metadata."""

import json
from pathlib import Path

from demo.adapter import card_image, recommendation_gallery
from mtgdeck.legality import OracleCatalog
from mtgdeck.metadata import iter_oracle_cards
from scripts import export_space_assets


def test_exported_catalog_renders_single_and_double_faced_cards(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    (root / 'data').mkdir(parents=True)
    (root / 'data/commander_eligible_oracle_ids.json').write_text('[]')
    cards = [
        {'oracle_id': 'single', 'name': 'Single',
         'image_uris': {'normal': 'https://cards.scryfall.io/single.jpg'}},
        {'oracle_id': 'double', 'name': 'Front // Back', 'card_faces': [
            {'name': 'Front', 'image_uris': {'normal': 'https://cards.scryfall.io/front.jpg'}},
            {'name': 'Back', 'image_uris': {'normal': 'https://cards.scryfall.io/back.jpg'}},
        ]},
    ]
    (root / 'data/oracle_cards.json').write_text(json.dumps(cards))
    monkeypatch.setattr(export_space_assets, 'ROOT', root)
    monkeypatch.setattr(export_space_assets, 'available_checkpoints', lambda _: [Path('fixture.pt')])
    monkeypatch.setattr(export_space_assets.torch, 'load', lambda *args, **kwargs: {'config': {}})
    output = tmp_path / 'assets'
    manifest = export_space_assets.export_assets(output, tmp_path / 'manifest.json')
    exported = list(iter_oracle_cards(output / manifest['oracle']['path']))
    assert exported == cards
    rows = [[rank, card['name'], 0.0, 'Colorless', ''] for rank, card in enumerate(cards, 1)]
    _, gallery = recommendation_gallery(rows, OracleCatalog(exported))
    assert [image for image, caption in gallery] == [card_image(card) for card in cards]
    assert all('Art unavailable' not in caption for image, caption in gallery)
