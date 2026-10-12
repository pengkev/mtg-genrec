"""Cached Commander Spellbook hyperedges and separate mechanical relation heads."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import gzip
import json
import os

import numpy as np

from .data import normalize_card_name
from .static_export import unit_rows
from . import static_v4 as v4

BULK_ENDPOINT = 'https://json.commanderspellbook.com/variants.json.gz'
SCHEMA_ENDPOINT = 'https://backend.commanderspellbook.com/schema/?format=json'


def snapshot(output):
    """The only Spellbook network stage. Preserve bytes and official OpenAPI."""
    import requests
    with v4.stage(output, {'stage': 'spellbook_snapshot', 'endpoint': BULK_ENDPOINT}) as dest:
        headers = {'User-Agent': 'mtg-deck-evaluator-static-v4/1.0'}
        response = requests.get(BULK_ENDPOINT, headers=headers, timeout=180)
        response.raise_for_status()
        raw = response.content
        (dest / 'variants.raw').write_bytes(raw)
        payload = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
        schema = requests.get(SCHEMA_ENDPOINT, headers=headers, timeout=60)
        schema.raise_for_status()
        (dest / 'schema.raw.json').write_bytes(schema.content)
        rows = variant_rows(payload)
        v4.write_json(dest / 'snapshot.json', {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
            'endpoint': BULK_ENDPOINT, 'schema_endpoint': SCHEMA_ENDPOINT, 'api_info': schema.json().get('info'),
            'bulk_version': payload.get('version'), 'bulk_timestamp': payload.get('timestamp'),
            'variants': len(rows), 'aliases': len(payload.get('aliases', [])),
            'raw_sha256': v4.sha256_file(dest / 'variants.raw'),
            'fields_present': dict(Counter(k for row in rows for k in row)),
            'variant_schema': schema.json().get('components', {}).get('schemas', {}).get('Variant'),
            'temporal_policy': 'bulk build timestamp is not variant creation time; no temporal split'})


def variant_rows(payload):
    if not isinstance(payload, dict):
        raise ValueError('Expected official bulk object or REST page')
    rows = payload.get('variants', payload.get('results'))
    if not isinstance(rows, list):
        raise ValueError('Missing variants/results array; unsupported Spellbook schema')
    if payload.get('next'):
        raise ValueError('Incomplete REST page; use the official bulk snapshot')
    return rows


def parse_variants(payload, cards):
    by_name = defaultdict(set)
    by_id = {}
    for card in cards:
        canonical = card.get('canonical', normalize_card_name(card['name']))
        by_name[canonical].add(card['oracle_id'])
        by_id[card['oracle_id']] = canonical
    rows, stats = [], Counter()
    seen = set()
    for raw in variant_rows(payload):
        if 'id' not in raw or not isinstance(raw.get('uses'), list):
            raise ValueError('Variant missing id/uses; unsupported schema')
        ident = str(raw['id'])
        if ident in seen:
            raise ValueError(f'Duplicate variant ID: {ident}')
        seen.add(ident)
        members, unresolved = [], []
        for use in raw['uses']:
            card = use['card']
            name = normalize_card_name(card['name'])
            oid = card.get('oracleId')
            # Even known names cannot silently substitute a different Oracle ID.
            valid = len(by_name.get(name, set())) == 1 and (not oid or oid in by_name[name])
            if not valid:
                unresolved.append({'name': card['name'], 'oracle_id': oid, 'reason':
                                   'ambiguous' if len(by_name.get(name, set())) > 1 else 'unknown_or_identity_mismatch'})
            else:
                members.append(name)
        outcomes = [str(p['feature']['id']) for p in raw.get('produces', [])]
        reasons = []
        if unresolved:
            reasons.append('unresolved_card')
        if len(set(members)) < 2:
            reasons.append('fewer_than_two_real_cards')
        if raw.get('requires'):
            reasons.append('template_context_not_encoded_by_primary_baseline')
        if any(u.get('quantity', 1) != 1 for u in raw['uses']):
            reasons.append('multiplicity_not_encoded_by_set_baseline')
        if raw.get('status') != 'OK':
            reasons.append('not_verified_OK')
        if raw.get('legalities', {}).get('commander') is not True:
            reasons.append('not_commander_legal')
        row = dict(id=ident, cards=sorted(set(members)), family_ids=sorted(str(c['id']) for c in raw.get('of', [])),
                   included_combo_ids=sorted(str(c['id']) for c in raw.get('includes', [])),
                   templates=raw.get('requires', []), outcomes=sorted(set(outcomes)), raw=raw,
                   unresolved=unresolved, eligible=not reasons, exclusion_reasons=reasons)
        rows.append(row)
        stats.update(reasons)
        stats['variants'] += 1
        stats['eligible'] += int(row['eligible'])
        stats['template_variants'] += bool(row['templates'])
    return rows, dict(stats)


def split_hyperedges(rows, seed=42):
    """Union shared official generator/included combo IDs and exact card sets.

    Components may be large: report their sizes and fail training on empty splits.
    Near variants crossing different official components are audited, not hidden.
    """
    families = v4.Families()
    for row in rows:
        tokens = ['variant:' + row['id'], 'set:' + v4.digest(row['cards'])]
        tokens += ['combo:' + ident for ident in row['family_ids'] + row['included_combo_ids']]
        families.join(tokens)
    return [{**row, 'split_group': families.find('variant:' + row['id']),
             'split': v4.partition(families.find('variant:' + row['id']), seed)} for row in rows]


def overlap_audit(rows, threshold=.8):
    """Exact overlap counts and bounded-memory intersection audit, with examples."""
    rows = [r for r in rows if r['eligible']]
    index = defaultdict(list)
    summary = Counter()
    examples = defaultdict(list)
    card_counts = defaultdict(Counter)
    for i, row in enumerate(rows):
        cards = set(row['cards'])
        card_counts[row['split']].update(cards)
        candidates = Counter(j for card in cards for j in index[card])
        for j, intersection in candidates.items():
            other = rows[j]
            if other['split'] == row['split']:
                continue
            pair = ':'.join(sorted([row['split'], other['split']]))
            union = len(cards) + len(other['cards']) - intersection
            categories = []
            if cards == set(other['cards']):
                categories.append('exact_duplicate')
            elif intersection == min(len(cards), len(other['cards'])):
                categories.append('subset_superset')
            if intersection / union >= threshold:
                categories.append('high_jaccard')
            for category in categories:
                summary[f'{pair}:{category}'] += 1
                if len(examples[f'{pair}:{category}']) < 20:
                    examples[f'{pair}:{category}'].append([row['id'], other['id']])
        for card in cards:
            index[card].append(i)
    shared = {card: {split: counts[card] for split, counts in card_counts.items()}
              for card in sorted(index)}
    return {'threshold': threshold, 'cross_split_overlap_counts': dict(summary), 'examples': dict(examples),
            'shared_card_frequency': shared, 'eligible_split_counts': dict(Counter(r['split'] for r in rows)),
            'component_sizes': dict(Counter(r['split_group'] for r in rows)),
            'claim': 'transductive card retrieval; cross-family overlap must be inspected before unseen-combo claims'}


def leave_one_out(rows):
    return [{'variant_id': row['id'], 'context': [c for c in row['cards'] if c != target],
             'target': target, 'split': row['split'], 'split_group': row['split_group'],
             'outcomes': row['outcomes']}
            for row in rows if row['eligible'] for target in row['cards']]


def replacement_slots(rows):
    slots = defaultdict(list)
    for row in rows:
        if not row['eligible']:
            continue
        for target in row['cards']:
            context = tuple(c for c in row['cards'] if c != target)
            # Same surrounding members, official split component and outcomes.
            # Preserve relevant visible-card state; target state is separately auditable.
            states = [u for u in row['raw']['uses'] if normalize_card_name(u['card']['name']) in context]
            states = [{k: v for k, v in u.items() if k != 'card'} | {'card': normalize_card_name(u['card']['name'])} for u in states]
            key = (row['split'], row['split_group'], context, tuple(row['outcomes']),
                   v4.digest(sorted(states, key=lambda u: u['card'])))
            slots[key].append((target, row['id']))
    result = []
    for (split, group, context, outcomes, states), values in sorted(slots.items()):
        alternatives = sorted({c for c, _ in values})
        if len(alternatives) > 1:
            result.append({'shared_context': list(context), 'alternatives': alternatives,
                'source_variant_ids': sorted({i for _, i in values}), 'split': split, 'split_group': group,
                'outcomes': list(outcomes), 'visible_states_hash': states, 'confidence': 'conditional_slot_evidence',
                'derivation_rule': 'same_context_official_component_visible_states_and_outcomes_one_card_difference',
                'limitation': 'conditional replacement, not unconditional functional equivalence'})
    return result


def prepare(snapshot_path, oracle, output, seed=42):
    v4.validate_stage(snapshot_path)
    v4.validate_stage(oracle)
    with v4.stage(output, {'stage': 'spellbook_prepare', 'seed': seed,
                          'candidate_policy': 'full unambiguous Oracle vocabulary; exclude visible real cards'},
                  [snapshot_path, oracle]) as dest:
        raw = (Path(snapshot_path) / 'variants.raw').read_bytes()
        payload = json.loads(gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw)
        cards = v4.read_json(Path(oracle) / 'cards.json')
        rows, stats = parse_variants(payload, cards)
        rows = split_hyperedges(rows, seed)
        vocab = [c['canonical'] for c in cards]
        (dest / 'vocab.txt').write_text('\n'.join(vocab) + '\n', encoding='utf-8')
        v4.write_json(dest / 'variants.json', rows)
        v4.write_json(dest / 'completion.json', leave_one_out(rows))
        v4.write_json(dest / 'replacements.json', replacement_slots(rows))
        v4.write_json(dest / 'overlap.json', overlap_audit(rows))
        frequencies = Counter(outcome for row in rows if row['eligible'] and row['split'] == 'train' for outcome in row['outcomes'])
        v4.write_json(dest / 'coverage.json', {'parsing': stats, 'fields_present': dict(Counter(k for row in rows for k in row['raw'])),
            'outcome_train_frequency': dict(frequencies), 'candidate_cards': len(vocab),
            'outcome_imbalance': 'raw train frequencies; optional BCE auxiliary, no evaluation-label fitting',
            'timestamps': 'bulk build timestamp available; variant creation timestamps absent in inspected 7.1.4 schema'})


def make_model(vocabulary_size, dimension=128, outcomes=0):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SetCompletion(nn.Module):
        def __init__(self):
            super().__init__()
            self.cards = nn.Embedding(vocabulary_size, dimension)
            self.query = nn.ModuleDict({r: nn.Linear(dimension, dimension, bias=False)
                                       for r in ('complementarity', 'replacement')})
            self.key = nn.ModuleDict({r: nn.Linear(dimension, dimension, bias=False)
                                     for r in ('complementarity', 'replacement')})
            self.outcome = nn.Linear(dimension, outcomes) if outcomes else None

        def aggregate(self, ids, mask):
            return (self.cards(ids) * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

        def score(self, ids, mask, relation='complementarity', slot_anchor=None):
            visible = self.aggregate(ids, mask)
            if slot_anchor is not None:
                # Replacement is conditioned on both surrounding cards and the
                # current slot occupant, distinct from missing-component scoring.
                visible = (visible + self.cards(slot_anchor)) / 2
            query = F.normalize(self.query[relation](visible), dim=1)
            keys = F.normalize(self.key[relation](self.cards.weight), dim=1)
            return query @ keys.T / .07

    return SetCompletion()


def batch_context(rows, lookup, device):
    import torch
    size = max(len(r['context']) for r in rows)
    ids = torch.zeros((len(rows), size), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), size), device=device)
    for i, row in enumerate(rows):
        values = [lookup[c] for c in row['context']]
        ids[i, :len(values)] = torch.tensor(values, device=device)
        mask[i, :len(values)] = 1
    return ids, mask


def train(prepared, output, seed=42, dimension=128, epochs=3, batch_size=64, auxiliary=False):
    import torch
    from torch.nn import functional as F
    v4.validate_stage(prepared)
    config = dict(stage='spellbook_train', seed=seed, dimension=dimension, epochs=epochs,
                  batch_size=batch_size, auxiliary=auxiliary, learning_rate=.001, candidate_filter='exclude visible cards',
                  replacement_weight=.25, outcome_weight=.1)
    with v4.stage(output, config, [prepared]) as dest:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        names = (Path(prepared) / 'vocab.txt').read_text(encoding='utf-8').splitlines()
        lookup = {n: i for i, n in enumerate(names)}
        rows = v4.read_json(Path(prepared) / 'completion.json')
        training = [r for r in rows if r['split'] == 'train']
        validation = [r for r in rows if r['split'] == 'validation']
        if not training or not validation:
            raise ValueError('Empty training/validation partition; inspect connected component sizes')
        slots = [r for r in v4.read_json(Path(prepared) / 'replacements.json') if r['split'] == 'train']
        replacements = [dict(context=r['shared_context'], anchor=a, alternatives=[b for b in r['alternatives'] if b != a],
                             source_variant_ids=r['source_variant_ids']) for r in slots for a in r['alternatives']]
        outcomes = sorted({o for r in training for o in r['outcomes']}) if auxiliary else []
        outcome_lookup = {o: i for i, o in enumerate(outcomes)}
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = make_model(len(names), dimension, len(outcomes)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        rng = np.random.default_rng(seed)
        seen = np.zeros(len(names), dtype=bool)
        for row in training:
            seen[[lookup[c] for c in row['context'] + [row['target']]]] = True
        history = []
        def completion_loss(batch):
            ids, mask = batch_context(batch, lookup, device)
            scores = model.score(ids, mask)
            for i, row in enumerate(batch):
                scores[i, [lookup[c] for c in row['context']]] = -torch.inf
            targets = torch.tensor([lookup[r['target']] for r in batch], device=device)
            loss = F.cross_entropy(scores, targets)
            if outcomes:
                labels = torch.zeros((len(batch), len(outcomes)), device=device)
                # Auxiliary sees the full hyperedge, including the held-out target.
                complete = [{**r, 'context': r['context'] + [r['target']]} for r in batch]
                full_ids, full_mask = batch_context(complete, lookup, device)
                for i, row in enumerate(batch):
                    for outcome in row['outcomes']:
                        if outcome in outcome_lookup:
                            labels[i, outcome_lookup[outcome]] = 1
                loss += .1 * F.binary_cross_entropy_with_logits(model.outcome(model.aggregate(full_ids, full_mask)), labels)
            return loss
        for epoch in range(epochs):
            model.train()
            losses = []
            order = rng.permutation(len(training))
            for start in range(0, len(training), batch_size):
                batch = [training[i] for i in order[start:start + batch_size]]
                optimizer.zero_grad()
                loss = completion_loss(batch)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            replacement_losses = []
            if replacements:
                for start in range(0, len(replacements), batch_size):
                    batch = replacements[start:start + batch_size]
                    ids, mask = batch_context(batch, lookup, device)
                    anchors = torch.tensor([lookup[r['anchor']] for r in batch], device=device)
                    scores = model.score(ids, mask, 'replacement', anchors)
                    for i, row in enumerate(batch):
                        scores[i, [lookup[c] for c in row['context'] + [row['anchor']]]] = -torch.inf
                    losses_for_batch = [torch.logsumexp(scores[i], 0) - torch.logsumexp(
                        scores[i, [lookup[c] for c in r['alternatives']]], 0) for i, r in enumerate(batch)]
                    loss = .25 * torch.stack(losses_for_batch).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    replacement_losses.append(float(loss.detach()))
            model.eval()
            with torch.inference_mode():
                valid_loss = [float(completion_loss(validation[i:i + batch_size])) for i in range(0, len(validation), batch_size)]
            history.append(dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation_loss=float(np.mean(valid_loss)),
                                replacement_loss=float(np.mean(replacement_losses)) if replacement_losses else None))
        torch.save(model.state_dict(), dest / 'model.pt')
        raw = model.cards.weight.detach().cpu().numpy().astype(np.float32)
        raw[~seen] = 0  # never export random untrained rows as mechanical evidence
        np.save(dest / 'vectors.npy', raw, allow_pickle=False)
        np.save(dest / 'mask.npy', seen, allow_pickle=False)
        (dest / 'vocab.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
        v4.write_json(dest / 'outcomes.json', outcomes)
        v4.write_json(dest / 'losses.json', history)
        v4.write_json(dest / 'training_completion.json', training)
        v4.write_json(dest / 'training_replacements.json', replacements)
        v4.write_json(dest / 'heads.json', {'complementarity': 'trained leave-one-out',
            'replacement': 'trained conditional slots' if replacements else 'unavailable: no training slots',
            'zero_policy': 'mechanically train-unseen cards export zeros with false mask'})


def incidence_baselines(rows, names):
    """Training hyperedge incidence; pair scores are explicit comparators only."""
    from scipy import sparse
    lookup = {n: i for i, n in enumerate(names)}
    training = [r for r in rows if r['eligible'] and r['split'] == 'train']
    rr, cc = [], []
    for i, row in enumerate(training):
        rr += [i] * len(row['cards'])
        cc += [lookup[c] for c in row['cards']]
    incidence = sparse.csr_matrix((np.ones(len(rr), dtype=np.float32), (rr, cc)), shape=(len(training), len(names)))
    counts = np.asarray(incidence.sum(0)).ravel()
    def score(context, mode):
        if mode == 'popularity':
            return counts.copy()
        query = np.asarray(incidence[:, context].sum(1)).ravel()
        co = np.asarray(query @ incidence).ravel().astype(float)
        if mode == 'incidence_cosine':
            denominator = np.sqrt(counts * float(query @ query))
            return np.divide(co, denominator, out=np.zeros_like(co), where=denominator > 0)
        return co
    return score


def evaluate(prepared, trained, deck_export, output, split='test', seed=42):
    import torch
    import pandas as pd
    from .static_export import load_static_card_embeddings
    from .static_multiview import align_view, retrieval_metrics
    v4.validate_stage(prepared)
    provenance = v4.validate_stage(trained)
    with v4.stage(output, dict(stage='spellbook_evaluation', split=split, deck_seed=seed,
                              candidates='full Oracle vocabulary minus visible cards; unseen deck/mechanical rows zero'),
                  [prepared, trained, deck_export]) as dest:
        names = (Path(prepared) / 'vocab.txt').read_text(encoding='utf-8').splitlines()
        lookup = {n: i for i, n in enumerate(names)}
        cfg = provenance['config']
        outcomes = v4.read_json(Path(trained) / 'outcomes.json')
        model = make_model(len(names), cfg['dimension'], len(outcomes))
        model.load_state_dict(torch.load(Path(trained) / 'model.pt', weights_only=True, map_location='cpu'))
        seen = np.load(Path(trained) / 'mask.npy', allow_pickle=False)
        with torch.no_grad():
            model.cards.weight[torch.tensor(~seen)] = 0
        model.eval()
        rows = v4.read_json(Path(prepared) / 'completion.json')
        variants = v4.read_json(Path(prepared) / 'variants.json')
        baseline = incidence_baselines(variants, names)
        decks = {}
        coverage = {}
        for label in ('sgns_input_128', 'ppmi_svd_128', 'concat_input_context_svd_384'):
            bundle = load_static_card_embeddings(label, seed, deck_export)
            matrix, mask = align_view(names, bundle.vocab, bundle.embedding_matrix)
            decks[label] = unit_rows(matrix)
            coverage[label] = int(mask.sum())
        results = []
        with torch.inference_mode():
            for row in rows:
                if row['split'] != split:
                    continue
                context = [lookup[c] for c in row['context']]
                target = lookup[row['target']]
                ids, mask = batch_context([row], lookup, 'cpu')
                scores = {'set_complementarity': model.score(ids, mask)[0].numpy()}
                scores.update({mode: baseline(context, mode) for mode in ('popularity', 'coincidence', 'incidence_cosine')})
                scores.update({label: matrix @ matrix[context].mean(0) for label, matrix in decks.items()})
                for label, values in scores.items():
                    metrics, _ = retrieval_metrics(values, context, [target], names)
                    results.append({'variant_id': row['variant_id'], 'target': row['target'], 'model': label,
                                    'task': 'partner_retrieval' if len(context) == 1 else 'set_completion',
                                    'train_seen_target': bool(seen[target]), **metrics})
        replacement_results = []
        if v4.read_json(Path(trained) / 'heads.json')['replacement'].startswith('trained'):
            with torch.inference_mode():
                for slot in v4.read_json(Path(prepared) / 'replacements.json'):
                    if slot['split'] != split:
                        continue
                    for anchor in slot['alternatives']:
                        row = {'context': slot['shared_context']}
                        ids, mask = batch_context([row], lookup, 'cpu')
                        a = lookup[anchor]
                        values = model.score(ids, mask, 'replacement', torch.tensor([a]))[0].numpy()
                        metrics, _ = retrieval_metrics(values, [a, *[lookup[c] for c in row['context']]],
                                                       [lookup[c] for c in slot['alternatives'] if c != anchor], names)
                        replacement_results.append({'anchor': anchor, 'shared_context': row['context'],
                                                    'variant_ids': slot['source_variant_ids'], **metrics})
        v4.write_json(dest / 'retrieval.json', results)
        if results:
            pd.DataFrame(results).groupby(['model', 'task', 'train_seen_target']).agg(
                mrr=('mrr', 'mean'), recall10=('recall@10', 'mean'),
                recall50=('recall@50', 'mean'), queries=('variant_id', 'count')).reset_index().to_csv(
                    dest / 'retrieval_summary.csv', index=False)
        v4.write_json(dest / 'replacement.json', replacement_results)
        v4.write_json(dest / 'coverage.json', {'eligible_queries': sum(r['split'] == split for r in rows),
            'vocabulary': len(names), 'mechanical_train_seen': int(seen.sum()), 'deck_overlap': coverage})
        v4.write_json(dest / 'unavailable.json', [{'task': 'replacement', 'reason': 'no held-out slots or no trained replacement head'}]
                      if not replacement_results else [])
