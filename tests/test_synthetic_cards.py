"""Hypothetical designs preserve color access and stay distinct from Oracle cards."""
import copy
from pathlib import Path

import pytest

from mtgdeck import static_v4 as v4
from mtgdeck import static_semantic as semantic
from mtgdeck import static_synthetic_cards as designs


@pytest.fixture
def policy():
    return v4.read_json(designs.DEFAULT_POLICY)


def test_cost_effect_extremes_and_color_restrictions(policy):
    cards = designs.generate_cards(policy)
    assert any(c['mana_cost'] == '{U}' and c['oracle_text'] == 'Draw five cards.' for c in cards)
    assert all('oracle_id' not in c and c['synthetic_id'].startswith('synthetic:') for c in cards)
    for c in cards:
        effects = c['parameters']['effect_ids']
        if 'black_reanimation' in effects:
            assert 'B' in c['colors'] and '{B}' in c['mana_cost']
        if 'red_burn' in effects:
            assert 'R' in c['colors'] and '{R}' in c['mana_cost']
        if c['colors'] == ['G'] and 'damage' in c['oracle_text']:
            assert ('with flying' in c['oracle_text'] or 'equal to its power' in c['oracle_text'])
        if 'white_small_reanimation' in effects:
            assert 'mana value' in c['oracle_text']
            assert c['parameters']['amounts'][effects.index('white_small_reanimation')] <= 3
        assert c['cmc'] >= len(c['colors'])
    assert any(set(c['colors']) == {'U', 'B'} and 'black_reanimation' in c['parameters']['effect_ids'] for c in cards)
    assert any('up to one basic land card' in c['oracle_text'] for c in cards)


def test_deterministic_family_partition_and_unique_serializations(policy):
    first = designs.generate_cards(policy)
    second = designs.generate_cards(copy.deepcopy(policy))
    assert first == second
    assert len({semantic.serialize(c) for c in first}) == len(first)
    families = {}
    for card in first:
        families.setdefault(card['family'], set()).add(card['split'])
    assert all(len(splits) == 1 for splits in families.values())
    assert {c['split'] for c in first} == {'train', 'validation'}


def test_mlm_mixing_is_explicit_structured_and_auditable():
    oracle = [{'oracle_id': str(i), 'canonical': str(i), 'split': 'train' if i < 4 else 'validation',
               'views': {'rules_only': 'Draw a card.', 'structured_no_name': '[CARD]\n[MANA] {U}\n[TEXT] Draw a card.'}}
              for i in range(6)]
    synthetic = [{'synthetic_id': 'synthetic:' + str(i), 'canonical': 'synthetic:' + str(i),
                  'family': 'train_family' if i < 2 else 'validation_family',
                  'split': 'train' if i < 2 else 'validation',
                  'views': {'structured_no_name': '[CARD]\n[MANA] {U}\n[TEXT] Draw five cards.'}}
                 for i in range(4)]
    baseline = semantic.mlm_documents(oracle)
    assert len(baseline['train']) == 8
    mixed = semantic.mlm_documents(oracle, synthetic, fraction=.5)
    assert mixed == semantic.mlm_documents(oracle, synthetic, fraction=.5)
    assert sum(r['source'] == 'synthetic_design' for r in mixed['train']) == 8
    assert all(r['view'] == 'structured_no_name' for r in mixed['train'] if r['source'] == 'synthetic_design')
    assert {r['id'] for r in mixed['train']}.isdisjoint({r['id'] for r in mixed['validation']})
    assert len(mixed['validation']) == 6  # no oversampling held-out sources
    with pytest.raises(ValueError, match='fraction'):
        semantic.mlm_documents(oracle, synthetic, fraction=1)


def test_prepare_excludes_existing_cards_and_cross_source_rules(tmp_path, policy):
    oracle = tmp_path / 'oracle'
    existing = {'name': 'Real Draw', 'oracle_text': 'Draw a card.', 'mana_cost': '{U}', 'type_line': 'Instant'}
    with v4.stage(oracle, {'stage': 'oracle'}) as path:
        v4.write_json(path / 'corpus.json', [
            {'oracle_id': 'real1', 'canonical': 'real1', 'split': 'train',
             'views': {'rules_only': 'Draw a card.', 'structured_no_name': semantic.serialize(existing)}},
            {'oracle_id': 'real2', 'canonical': 'real2', 'split': 'validation',
             'views': {'rules_only': 'Draw two cards.', 'structured_no_name': '[CARD] real2'}}])
    config = tmp_path / 'policy.json'
    v4.write_json(config, policy)
    output = tmp_path / 'designs'
    designs.prepare(oracle, output, config)
    cards = v4.read_json(output / 'cards.json')
    assert not any(semantic.serialize(c) == semantic.serialize(existing) for c in cards)
    assert not any(c['split'] == 'train' and c['oracle_text'] == 'Draw two cards.' for c in cards)
    assert v4.read_json(output / 'coverage.json')['source_oracle_corpus_hash'] == v4.digest(v4.read_json(oracle / 'corpus.json'))
    v4.validate_stage(output)
    with pytest.raises(FileExistsError):
        designs.prepare(oracle, output, config)
