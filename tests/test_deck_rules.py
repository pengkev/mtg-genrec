import pytest

from demo.adapter import add_to_deck, card_modal, remaining_copies
from mtgdeck.deck_rules import companion_card_allowed, companion_errors, copy_limit
from mtgdeck.inference import parse_deck_zones, serving_checkpoints, prepare_request
from mtgdeck.legality import OracleCatalog


def card(name, type_line='Creature', cmc=2, **extra):
    return dict(name=name, oracle_id=name.casefold(), type_line=type_line, cmc=cmc,
                legalities={'modern': 'legal', 'legacy': 'legal', 'commander': 'legal'}, **extra)


@pytest.mark.parametrize('companion,allowed,rejected', [
    ('Gyruda, Doom of Depths', card('Even', cmc=2), card('Odd', cmc=3)),
    ('Jegantha, the Wellspring', card('Unique', mana_cost='{2}{R}{G}'), card('Repeated', mana_cost='{X}{X}{R}')),
    ('Kaheera, the Orphanguard', card('Cat', 'Creature — Cat'), card('Human', 'Creature — Human')),
    ('Keruga, the Macrosage', card('Land', 'Land', 0), card('Small', cmc=2)),
    ('Lurrus of the Dream-Den', card('Spell', 'Sorcery', 8), card('Permanent', 'Enchantment', 3)),
    ('Obosh, the Preypiercer', card('Odd', cmc=3), card('Even', cmc=2)),
    ('Umori, the Collector', card('Creature'), card('Spell', 'Sorcery')),
    ('Zirda, the Dawnwaker', card('Cycler', keywords=['Cycling']), card('Vanilla')),
])
def test_companion_card_constraints(companion, allowed, rejected):
    chosen = card(companion)
    assert companion_card_allowed(chosen, allowed, {'creature'})
    assert not companion_card_allowed(chosen, rejected, {'creature'})


def test_companion_special_cases_and_combined_types():
    assert companion_card_allowed(card('Kaheera, the Orphanguard'), card('Shifter', keywords=['Changeling']))
    assert companion_card_allowed(card('Zirda, the Dawnwaker'), card('Island', 'Land — Island', 0))
    assert not companion_card_allowed(card('Zirda, the Dawnwaker'), card('Grant', oracle_text='Creatures you control have "{T}: Add {G}."'))
    assert companion_errors(card('Lutri, the Spellchaser'), [(card('Spell'), 2), (card('Land', 'Land'), 6)]) == ['Spell']
    # Umori needs one common type across the whole deck, not pairwise matches.
    cards = [(card('AC', 'Artifact Creature'), 1), (card('A', 'Artifact'), 1), (card('C', 'Creature'), 1)]
    assert companion_errors(card('Umori, the Collector'), cards)
    split = card('Split', 'Instant', 5, layout='split', mana_cost='{1}{R} // {2}{R}',
                 card_faces=[{'cmc': 2, 'mana_cost': '{1}{R}'}, {'cmc': 3, 'mana_cost': '{2}{R}'}])
    assert not companion_card_allowed(card('Jegantha, the Wellspring'), split)
    transform = card('Transform', 'Creature', 2, layout='transform', card_faces=[{'cmc': 2}, {'cmc': 5}])
    assert companion_card_allowed(card('Lurrus of the Dream-Den'), transform)


@pytest.mark.parametrize('fmt', ['modern', 'legacy'])
def test_add_copies_and_sideboard_limits(fmt):
    bolt = card('Lightning Bolt', 'Instant', 1)
    catalog = OracleCatalog([bolt, card('Mountain', 'Basic Land — Mountain', 0)])
    selected = {'Card': bolt['name']}
    text, message = add_to_deck('', selected, catalog, fmt, 3)
    assert parse_deck_zones(text)['mainboard']['Lightning Bolt'] == 3
    updated, _ = add_to_deck(text, selected, catalog, fmt, 1)
    assert parse_deck_zones(updated)['mainboard']['Lightning Bolt'] == 4
    assert add_to_deck(updated, selected, catalog, fmt, 1)[0] == updated
    sideboard = 'Deck\n2 Lightning Bolt\nSideboard\n1 Lightning Bolt'
    assert add_to_deck(sideboard, selected, catalog, fmt, 2)[0] == sideboard
    allowed, _ = add_to_deck(sideboard, selected, catalog, fmt, 1)
    assert parse_deck_zones(allowed)['mainboard']['Lightning Bolt'] == 3
    assert add_to_deck('', {'Card': 'Mountain'}, catalog, fmt, 12)[0] == '12 Mountain'


def test_explicit_companion_limits_only_apply_when_selected():
    lutri = card('Lutri, the Spellchaser', cmc=3)
    bolt = card('Bolt', 'Instant', 1)
    catalog = OracleCatalog([lutri, bolt])
    selected = {'Card': 'Bolt'}
    assert add_to_deck('1 Bolt', selected, catalog, 'modern', 1)[0] != '1 Bolt'
    assert add_to_deck('1 Bolt', selected, catalog, 'modern', 1, lutri['name'])[0] == '1 Bolt'
    pasted = 'Companion\n1 Lutri, the Spellchaser\nDeck\n1 Bolt'
    assert add_to_deck(pasted, selected, catalog, 'modern', 1)[0] == pasted
    assert parse_deck_zones(pasted)['mainboard'] == {'Bolt': 1}
    # One card can be a normal mainboard copy without companion restrictions.
    assert add_to_deck('1 Lutri, the Spellchaser\n1 Bolt', selected, catalog, 'modern', 1)[0].endswith('2 Bolt')


def test_add_enforces_companion_candidate_and_partial_deck_rules():
    keruga = card('Keruga, the Macrosage', cmc=5)
    catalog = OracleCatalog([keruga, card('Small', cmc=1), card('Big', cmc=4)])
    assert add_to_deck('', {'Card': 'Small'}, catalog, 'legacy', 1, keruga['name'])[0] == ''
    assert add_to_deck('1 Small', {'Card': 'Big'}, catalog, 'legacy', 1, keruga['name'])[0] == '1 Small'
    assert add_to_deck('', {'Card': 'Big'}, catalog, 'legacy', 3, keruga['name'])[0] == '3 Big'
    keruga['legalities']['modern'] = 'banned'
    assert 'not legal' in add_to_deck('', {'Card': 'Big'}, catalog, 'modern', 1, keruga['name'])[1]


@pytest.mark.parametrize('quantity', [True, 0, -1, 1.2, 101, '2'])
def test_add_rejects_invalid_quantities(quantity):
    catalog = OracleCatalog([card('Card')])
    assert add_to_deck('', {'Card': 'Card'}, catalog, 'modern', quantity)[0] == ''


def test_modal_escapes_rules_text_and_has_copy_controls():
    catalog = OracleCatalog([card('Card', oracle_text='<script>alert(1)</script>\nDraw a card.', mana_cost='{1}{U}')])
    html = card_modal({'Card': 'Card', 'Rank': 1, 'Score': .5}, catalog, '', 'modern')
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert 'Draw a card.' in html and 'data-add' in html


def test_serving_selects_one_premium_model_per_format(tmp_path):
    files = []
    for fmt in ('commander', 'modern', 'legacy'):
        for stage in ('base', 'premium'):
            path = tmp_path / f'attention_oracleid_v2_{fmt}_static_{stage}_finetuned_384.pt'
            path.touch()
            files.append(path)
    selected = serving_checkpoints(files)
    assert len(selected) == 3 and all('_premium_' in path.name for path in selected)


def test_explicit_one_copy_exception_and_basic_lands():
    assert copy_limit(card('Normal'), 'modern') == 4
    assert copy_limit(card('Normal'), 'commander') == 1
    assert copy_limit(card('Basic', 'Basic Land'), 'modern') is None
    assert copy_limit(card('Seven', oracle_text='A deck can have up to seven cards named Seven.'), 'legacy') == 7
    assert copy_limit(card('One', oracle_text='A deck can have up to one card named One.'), 'modern') == 1


def test_lutri_does_not_restrict_sideboard_duplicates():
    lutri, spell = card('Lutri, the Spellchaser', cmc=3), card('Spell')
    catalog = OracleCatalog([lutri, spell])
    pasted = 'Deck\nSideboard\n2 Spell'
    updated, _ = add_to_deck(pasted, {'Card': 'Spell'}, catalog, 'legacy', 1, lutri['name'])
    assert parse_deck_zones(updated)['mainboard']['Spell'] == 1
    assert parse_deck_zones(updated)['sideboard']['Spell'] == 2


def test_pasted_companion_controls_modal_copy_count():
    lutri, spell = card('Lutri, the Spellchaser', cmc=3), card('Spell')
    catalog = OracleCatalog([lutri, spell])
    modal = card_modal({'Card': 'Spell', 'Rank': 1, 'Score': 1}, catalog,
                       'Companion\nLutri, the Spellchaser\nDeck', 'legacy')
    assert 'data-add' in modal
    assert remaining_copies('Companion\nLutri, the Spellchaser\nDeck', spell, catalog, 'legacy', lutri) == 1


def test_commander_companion_ban_is_separate_from_card_legality():
    from mtgdeck.deck_rules import companion_choices
    from mtgdeck.inference import resolve_partial_deck
    lutri = card('Lutri, the Spellchaser', 'Legendary Creature', 3)
    catalog = OracleCatalog([lutri])
    assert 'Lutri, the Spellchaser' not in companion_choices(catalog, 'commander')
    assert 'Lutri, the Spellchaser' in companion_choices(catalog, 'modern')
    with pytest.raises(ValueError, match='banned as a Commander companion'):
        resolve_partial_deck(catalog, {}, '', '', 'commander', lutri['name'])
    partial, _, _ = resolve_partial_deck(catalog, {}, lutri['name'], '', 'commander')
    assert partial['commanders'] and 'companion' not in partial
