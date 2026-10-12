"""Hypothetical MLM cards from an explicit, restricted color-pie whitelist.

These are design records, not Oracle identities, deck tokens or relation labels.
Only structured records enter MLM so color-bearing mana costs stay in the text.
"""
from __future__ import annotations

from collections import Counter
from itertools import product
from pathlib import Path

from . import static_v4 as v4
from .static_semantic import discover_tokens, serialize

DEFAULT_POLICY = v4.ROOT / 'configs/card2vec_synthetic_designs_v1.json'
DEFAULT_OUTPUT = v4.DEFAULT_ROOT / 'synthetic_cards_v2'
NUMBER_WORDS = {1: 'a', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 8: 'eight'}
COLOR_ORDER = 'WUBRG'


def render_effect(effect, amount):
    words = 'one' if effect['id'] == 'green_ramp' and amount == 1 else NUMBER_WORDS.get(amount, str(amount))
    return effect['text'].format(amount=amount, amount_words=words,
                                 cards='card' if amount == 1 else 'cards',
                                 them='it' if amount == 1 else 'them', red_mana='{R}' * (amount or 0))


def generate_cards(policy, seed=42):
    """Vary cost, amount and speed independently; preserve every effect restriction."""
    effects = {e['id']: e for e in policy['effects']}
    if len(effects) != len(policy['effects']):
        raise ValueError('Effect IDs must be unique')
    combinations = [(ident,) for ident in effects] + [tuple(ids) for ids in policy['compositions']]
    records = []
    seen = set()
    for effect_ids in combinations:
        selected = [effects[ident] for ident in effect_ids]
        family = 'synthetic_design:' + '+'.join(effect_ids)
        split = 'train' if v4.partition(family, seed) == 'train' else 'validation'
        for assigned_colors in product(*(e['colors'] for e in selected)):
            colors = [c for c in COLOR_ORDER if c in assigned_colors]
            if len(set(assigned_colors) - set(COLOR_ORDER)):
                raise ValueError('Synthetic policy requires explicit W/U/B/R/G assignments')
            for mana_value, type_line, amounts in product(policy['mana_values'], policy['types'],
                                                          product(*(e.get('amounts', [None]) for e in selected))):
                if mana_value < len(colors):
                    continue  # every color that supplies an effect gets a colored pip
                generic = mana_value - len(colors)
                cost = (f'{{{generic}}}' if generic else '') + ''.join(f'{{{c}}}' for c in colors)
                text = '\n'.join(render_effect(e, n) for e, n in zip(selected, amounts))
                parameters = {'effect_ids': list(effect_ids), 'colors': colors, 'effect_colors': list(assigned_colors),
                              'mana_value': mana_value, 'type_line': type_line, 'amounts': list(amounts)}
                ident = 'synthetic:' + v4.digest(parameters)[:24]
                card = {'id': ident, 'canonical': ident, 'name': ident, 'synthetic_id': ident,
                        'mana_cost': cost, 'cmc': mana_value, 'type_line': type_line, 'oracle_text': text,
                        'colors': colors, 'source_kind': 'hypothetical_design', 'family': family,
                        'split': split, 'parameters': parameters,
                        'color_pie': {'status': 'consistent_with_declared_whitelist', 'source': policy['source'],
                                      'rules': [e['color_pie_rule'] for e in selected]},
                        'balance': 'not_assessed; cost/effect extremes are intentional'}
                structured = serialize(card)
                if structured in seen:
                    continue
                seen.add(structured)
                records.append(card)
    return sorted(records, key=lambda c: c['synthetic_id'])


def prepare(oracle, output=DEFAULT_OUTPUT, policy_path=DEFAULT_POLICY, seed=42):
    v4.validate_stage(oracle)
    policy = v4.read_json(policy_path)
    with v4.stage(output, {'stage': 'synthetic_cards', 'seed': seed, 'policy_version': policy['version'],
                          'mlm_view': 'structured_no_name', 'color_pie': 'consistent only'}, [oracle, policy_path]) as dest:
        generated = generate_cards(policy, seed)
        real = v4.read_json(Path(oracle) / 'corpus.json')
        real_records = {r['views']['structured_no_name'] for r in real}
        # Unlabelled rules-only Oracle text is already used in MLM. A synthetic
        # validation design must not have identical rules text in Oracle TRAIN;
        # synthetic TRAIN must not copy Oracle validation rules. This conservative
        # filter still permits cost novelty against same-split real wording.
        real_rules = {split: {r['views']['rules_only'] for r in real if r['split'] == split}
                      for split in ('train', 'validation')}
        cards, exclusions = [], []
        for card in generated:
            reason = None
            if serialize(card) in real_records:
                reason = 'exact_existing_structured_record'
            other = 'validation' if card['split'] == 'train' else 'train'
            if serialize(card, 'rules_only') in real_rules[other]:
                reason = 'opposite_oracle_split_rules_overlap'
            if reason:
                exclusions.append({'synthetic_id': card['synthetic_id'], 'family': card['family'], 'reason': reason})
            else:
                cards.append(card)
        corpus = [{'synthetic_id': c['synthetic_id'], 'canonical': c['canonical'], 'source_kind': c['source_kind'],
                   'family': c['family'], 'split': c['split'],
                   'views': {'structured_no_name': serialize(c)}} for c in cards]
        v4.write_json(dest / 'policy.json', policy)
        v4.write_json(dest / 'cards.json', cards)
        v4.write_json(dest / 'corpus.json', corpus)
        v4.write_json(dest / 'tokens.json', discover_tokens(cards))
        v4.write_json(dest / 'exclusions.json', exclusions)
        v4.write_json(dest / 'coverage.json', {'generated': len(generated), 'retained': len(cards),
            'splits': dict(Counter(c['split'] for c in cards)),
            'families': dict(Counter(c['family'] for c in cards)),
            'exclusions': dict(Counter(e['reason'] for e in exclusions)), 'corpus_hash': v4.digest(corpus),
            'source_oracle_corpus_hash': v4.digest(real),
            'policy': 'hypothetical, color-pie-consistent only; no balance or printability claim',
            'split_policy': 'all parameter variations of a template/composition share a split; cross-source rules overlap excluded',
            'evaluation_scope': 'template split with shared underlying effects; not unseen-mechanic or functional-embedding proof'})
    return Path(output)
