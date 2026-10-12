"""Aligned independent view export and frozen v4 evaluation; never fit fusion weights."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from .static_export import StaticCardEmbeddings, load_static_card_embeddings, unit_rows
from . import static_v4 as v4

FUSIONS = {'deck_text': ('deck', 'text'), 'deck_mechanical': ('deck', 'mechanical'),
           'text_mechanical': ('text', 'mechanical'), 'deck_text_mechanical': ('deck', 'text', 'mechanical'),
           'sgns_ppmi': ('sgns_input_128', 'ppmi_svd_128'),
           'sgns_ppmi_difference': ('sgns_input_128', 'ppmi_svd_128', 'difference'),
           'sgns_ppmi_difference_text': ('sgns_input_128', 'ppmi_svd_128', 'difference', 'text'),
           'sgns_ppmi_difference_mechanical': ('sgns_input_128', 'ppmi_svd_128', 'difference', 'mechanical'),
           'sgns_ppmi_difference_text_mechanical': ('sgns_input_128', 'ppmi_svd_128', 'difference', 'text', 'mechanical')}
ABLATIONS = ('vanilla', 'mlm', 'semantic_mlm', 'tier1', 'tier12', 'tier12_description')


def align_view(target_names, source_names, matrix, source_mask=None):
    matrix = np.asarray(matrix)
    if (len(set(source_names)) != len(source_names) or len(set(target_names)) != len(target_names)
            or matrix.ndim != 2 or len(matrix) != len(source_names) or not np.isfinite(matrix).all()):
        raise ValueError('Expected unique vocabularies and a finite source-aligned matrix')
    if source_mask is None:
        source_mask = np.ones(len(source_names), dtype=bool)
    if np.asarray(source_mask).shape != (len(source_names),):
        raise ValueError('Source mask shape disagrees with vocabulary')
    result = np.zeros((len(target_names), matrix.shape[1]), dtype=np.float32)
    mask = np.zeros(len(target_names), dtype=bool)
    lookup = {n: i for i, n in enumerate(source_names)}
    for i, name in enumerate(target_names):
        if name in lookup and source_mask[lookup[name]]:
            result[i] = matrix[lookup[name]]
            mask[i] = True
    return result, mask


def concatenate(blocks):
    if not blocks or any(x.ndim != 2 or len(x) != len(blocks[0]) or not np.isfinite(x).all() for x in blocks):
        raise ValueError('Blocks must be finite and row-aligned')
    return np.concatenate([unit_rows(x) for x in blocks], axis=1).astype(np.float32)


def aligned_difference(sgns, ppmi):
    """Transductive orthogonal Procrustes, without labels or evaluation scores."""
    if sgns.shape != ppmi.shape or sgns.ndim != 2 or not len(sgns):
        raise ValueError('Difference requires equal, nonempty row-aligned source shapes')
    a, b = (unit_rows(x).astype(np.float64) for x in (sgns, ppmi))
    left, _, right = np.linalg.svd(b.T @ a, full_matrices=False)
    rotation = left @ right
    difference = (a - b @ rotation).astype(np.float32)
    difference[np.linalg.norm(difference, axis=1) <= 1e-6] = 0
    return difference, rotation


def export(deck_export, semantics, mechanical, output, seed=42, primary='mlm'):
    """One explicit seed per portable checkpoint; no coordinate averaging."""
    if primary not in semantics:
        raise ValueError('Primary semantic ablation must be explicitly supplied')
    for path in [*semantics.values(), mechanical]:
        v4.validate_stage(path)
    with v4.stage(output, {'stage': 'multiview_export', 'deck_seed': seed, 'primary_semantic': primary,
                          'fusion': 'L2 each source block, no final normalization, no learned weights',
                          'missing': 'zero block and boolean source mask'},
                  [deck_export, *semantics.values(), mechanical]) as dest:
        deck = load_static_card_embeddings(seed=seed, artifact_root=deck_export)
        primary_names = (Path(semantics[primary]) / 'vocab.txt').read_text(encoding='utf-8').splitlines()
        names = deck.vocab + sorted(set(primary_names) - set(deck.vocab))
        views = {'deck': align_view(names, deck.vocab, deck.embedding_matrix)}
        deck_sources = {}
        for label in ('sgns_input_128', 'sgns_context_128', 'ppmi_svd_128'):
            bundle = load_static_card_embeddings(label, seed, deck_export)
            deck_sources[label] = bundle.embedding_matrix
            views[label] = align_view(names, bundle.vocab, bundle.embedding_matrix)
        difference, rotation = aligned_difference(deck_sources['sgns_input_128'], deck_sources['ppmi_svd_128'])
        views['difference'] = align_view(names, deck.vocab, difference)
        np.save(dest / 'ppmi_to_sgns_rotation.npy', rotation, allow_pickle=False)
        for label, path in semantics.items():
            if label not in ABLATIONS:
                raise ValueError(f'Unknown semantic ablation: {label}')
            source_names = (Path(path) / 'vocab.txt').read_text(encoding='utf-8').splitlines()
            views['semantic_' + label] = align_view(names, source_names, np.load(Path(path) / 'mean.npy', allow_pickle=False))
        mech_names = (Path(mechanical) / 'vocab.txt').read_text(encoding='utf-8').splitlines()
        views['mechanical'] = align_view(names, mech_names, np.load(Path(mechanical) / 'vectors.npy', allow_pickle=False),
                                        np.load(Path(mechanical) / 'mask.npy', allow_pickle=False))
        (dest / 'vocab.txt').write_text('\n'.join(names) + '\n', encoding='utf-8')
        specs = {}
        for label, (matrix, mask) in views.items():
            np.save(dest / f'{label}.npy', matrix, allow_pickle=False)
            np.save(dest / f'{label}_mask.npy', mask, allow_pickle=False)
            specs[label] = {'matrix': f'{label}.npy', 'mask': f'{label}_mask.npy',
                            'dimension': matrix.shape[1], 'covered': int(mask.sum()), 'normalization': 'raw source'}
            if label.startswith('semantic_'):
                origin = Path(semantics[label.removeprefix('semantic_')]) / 'model_lineage.json'
                specs[label]['training_lineage'] = v4.read_json(origin) if origin.exists() else 'unavailable: external vector fixture'
            elif label == 'mechanical':
                specs[label]['training_configuration'] = v4.read_json(Path(mechanical) / 'provenance.json')['config']
        v4.write_json(dest / 'manifest.json', {'version': 'static_export_v2', 'deck_seed': seed,
            'vocabulary_size': len(names), 'vocab_sha256': v4.sha256_file(dest / 'vocab.txt'),
            'sources': specs, 'aliases': {'text': 'semantic_' + primary}, 'fusions': FUSIONS,
            'normalization': 'independent source-block L2; zeros stay zero; no final normalization',
            'source_coverage_masks': 'boolean .npy per source; missing entries have zero block',
            'difference': {'definition': 'unit(SGNS input) - unit(PPMI) @ orthogonal rotation',
                           'alignment_population': 'all frozen deck-vocabulary rows; transductive, no labels',
                           'rotation': 'ppmi_to_sgns_rotation.npy', 'numerical_zero_tolerance': 1e-6,
                           'interpretation': 'discrepancy feature; useful additional signal is not assumed'},
            'ablation_status': {name: 'available' if name in semantics else 'unavailable/pending' for name in ABLATIONS}})


def load_multiview_embeddings(root=v4.DEFAULT_EXPORT, representation='deck_text_mechanical', verify=True):
    """NumPy-only portable loading; raw views and normalized fusions are selectable."""
    root = Path(root)
    if verify:
        v4.validate_stage(root)
    manifest = v4.read_json(root / 'manifest.json')
    if manifest['version'] != 'static_export_v2':
        raise ValueError('Unsupported multiview export version')
    names = (root / 'vocab.txt').read_text(encoding='utf-8').splitlines()
    if len(names) != manifest['vocabulary_size'] or len(set(names)) != len(names):
        raise ValueError('Invalid multiview vocabulary')
    blocks = manifest['fusions'].get(representation, [representation])
    matrices, masks = [], {}
    for label in blocks:
        source = manifest['aliases'].get(label, label)
        spec = manifest['sources'][source]
        paths = [(root / spec[field]).resolve() for field in ('matrix', 'mask')]
        if any(root.resolve() not in p.parents for p in paths):
            raise ValueError('Source path escapes export root')
        matrix, mask = [np.load(p, allow_pickle=False) for p in paths]
        if (matrix.shape != (len(names), spec['dimension']) or matrix.dtype != np.float32
                or not np.isfinite(matrix).all() or mask.shape != (len(names),) or mask.dtype != bool
                or np.any(matrix[~mask] != 0)):
            raise ValueError('Invalid multiview source matrix/mask')
        matrices.append(matrix)
        masks[source] = mask
    matrix = concatenate(matrices) if representation in manifest['fusions'] else matrices[0]
    return StaticCardEmbeddings(names, {n: i for i, n in enumerate(names)}, matrix,
                                {**manifest, 'representation': representation, 'coverage_masks': masks})


def retrieval_metrics(scores, excluded, targets, names):
    """Canonical deterministic tie order, full candidate pool, single/multiple targets."""
    scores = np.asarray(scores, dtype=float)
    if (scores.shape != (len(names),) or not np.isfinite(scores).all() or not targets
            or set(targets) & set(excluded) or len(set(targets)) != len(targets)):
        raise ValueError('Invalid retrieval scores/targets')
    candidates = np.setdiff1d(np.arange(len(names)), excluded)
    order = candidates[np.lexsort((np.asarray(names)[candidates], -scores[candidates]))]
    rank = np.zeros(len(names), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    ranks = rank[targets]
    result = {'mrr': float(1 / ranks.min()), 'mean_target_rank': float(ranks.mean()),
              'median_target_rank': float(np.median(ranks)), 'candidate_count': len(candidates)}
    for k in (1, 5, 10, 20, 50):
        hits = np.isin(order[:k], targets)
        ideal = min(k, len(targets))
        result[f'recall@{k}'] = float(hits.sum() / len(targets))
        result[f'ndcg@{k}'] = float(np.sum(hits / np.log2(np.arange(len(hits)) + 2)) /
                                   np.sum(1 / np.log2(np.arange(ideal) + 2)))
    return result, order


def geometry(x, y, seed=4204, sample_size=2000):
    if len(x) != len(y):
        raise ValueError('Geometry inputs must be row-aligned')
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(x), min(sample_size, len(x)), replace=False)
    if len(ids) < 2:
        return {'linear_cka': None, 'cosine_correlation': None, 'cards': len(ids), 'undefined_reason': 'insufficient covered cards'}
    a, b = np.asarray(x[ids], float), np.asarray(y[ids], float)
    a -= a.mean(0)
    b -= b.mean(0)
    denominator = np.linalg.norm(a.T @ a, 'fro') * np.linalg.norm(b.T @ b, 'fro')
    cka = float(np.linalg.norm(a.T @ b, 'fro')**2 / denominator) if denominator else None
    if len(ids) < 2:
        return {'linear_cka': cka, 'cosine_correlation': None, 'cards': len(ids)}
    left = rng.integers(0, len(ids), 20000)
    right = (left + rng.integers(1, len(ids), len(left))) % len(ids)
    cosines = [np.sum(unit_rows(m[ids])[left] * unit_rows(m[ids])[right], axis=1) for m in (x, y)]
    correlation = float(np.corrcoef(cosines)[0, 1]) if all(np.std(c) > 0 for c in cosines) else None
    return {'linear_cka': cka, 'cosine_correlation': correlation, 'cards': len(ids), 'sample_seed': seed,
            'undefined_reason': 'constant/insufficient covered geometry' if cka is None or correlation is None else None}


def cold_start(deck, selected):
    masked = np.asarray(deck).copy()
    masked[selected] = 0
    return masked


def evaluate(root, checkpoint, oracle, output, probes=False, spellbook=None):
    """Reuse frozen labels and probe splits; trained relation families are excluded."""
    import joblib
    import pandas as pd
    from . import static_relations as relations
    from . import static_experiment as experiment
    from . import static_geometry as existing
    from .static_semantic import ANCHORS, effect_signature
    inputs_path = Path(root) / 'artifacts/card2vec/static_v2/evaluation_inputs.joblib'
    relations_path = Path(root) / 'configs/card2vec_relations.json'
    v4.validate_stage(checkpoint)
    v4.validate_stage(oracle)
    sources = [checkpoint, oracle, inputs_path, relations_path] + ([spellbook] if spellbook else [])
    with v4.stage(output, dict(stage='multiview_evaluation', frozen_split_seeds=[101, 102, 103],
                              probes=probes, cold_start='simulation: hide selected existing deck rows'), sources) as dest:
        inputs = joblib.load(inputs_path)
        names, counts = list(inputs['names']), inputs['counts']
        lookup = {n: i for i, n in enumerate(names)}
        manifest = v4.read_json(Path(checkpoint) / 'manifest.json')
        primary = manifest['aliases']['text']
        representations = ['text', *[n for n in manifest['sources'] if n != primary], *manifest['fusions']]
        matrices, masks = {}, {}
        for label in representations:
            bundle = load_multiview_embeddings(checkpoint, label, verify=False)
            matrices[label], present = align_view(names, bundle.vocab, bundle.embedding_matrix)
            source_masks = bundle.metadata['coverage_masks']
            union_mask = np.logical_or.reduce(list(source_masks.values()))
            _, masks[label] = align_view(names, bundle.vocab, bundle.embedding_matrix, union_mask)
        records = relations.build_benchmark(inputs, v4.read_json(relations_path))
        synthetic = v4.read_json(Path(oracle) / 'synthetic.json')
        trained_pairs = set()
        if spellbook:
            v4.validate_stage(spellbook)
            for variant in v4.read_json(Path(spellbook) / 'variants.json'):
                if variant['eligible'] and variant['split'] == 'train':
                    trained_pairs.update(combinations(sorted(variant['cards']), 2))
        # MLM uses text, not the synthetic relation pairs. Exclude relation
        # families only when a supplied semantic checkpoint trained on them.
        contrastive_tiers = set()
        for spec in manifest['sources'].values():
            lineage = spec.get('training_lineage', {})
            config = lineage.get('config', {}) if isinstance(lineage, dict) else {}
            if config.get('stage') == 'contrastive':
                contrastive_tiers.update(config['tiers'])
        train_families = {r['family'] for r in synthetic if r['split'] == 'train'
                          and r['task'] == 'card_to_card' and r['tier'] in contrastive_tiers}
        supervised = {n for r in synthetic if r['family'] in train_families for n in (r['anchor'], r['positive'])}
        heldout, exclusions = [], []
        for row in records:
            if row['relation'] in ('similarity', 'substitutability') and set(row['query'] + row['targets']) & supervised:
                exclusions.append({'id': row['id'], 'reason': 'synthetic_training_family_overlap'})
            else:
                if row['relation'] == 'complementarity':
                    exposed = [n for n in row['targets'] if any(tuple(sorted((q, n))) in trained_pairs for q in row['query'])]
                    if exposed:
                        exclusions.append({'id': row['id'], 'targets': exposed, 'reason': 'spellbook_train_pair_overlap'})
                        row = {**row, 'targets': [n for n in row['targets'] if n not in exposed]}
                        if not row['targets']:
                            continue
                heldout.append(row)
        v4.write_json(dest / 'queries.json', heldout)
        v4.write_json(dest / 'leakage_exclusions.json', exclusions)
        unavailable = []
        retrieval, probe_frames, qualitative, geometry_rows = [], [], [], []
        for label, matrix in matrices.items():
            scorer = existing.GeometryScorer(matrix)
            covered = [r for r in heldout if all(masks[label][lookup[n]] for n in r['query'] + r['targets'] if n in lookup)]
            frame, _ = relations.evaluate_records(scorer, covered, names, counts)
            if not frame.empty:
                frame['representation'] = label
                retrieval.append(frame)
            if probes:
                for task in ('color', 'type', 'format', 'mana', 'archetype'):
                    frame, missing = experiment.probe_long(matrix, counts, names, inputs['metadata'],
                                                          inputs['targets'][task], task)
                    if not frame.empty:
                        frame = frame[frame.baseline != 'random']  # existing random control caps at 256d
                        frame['representation'] = label
                        probe_frames.append(frame)
                    # Different unavailable reasons have different optional
                    # fields; omit Pandas' missing placeholders from strict JSON.
                    unavailable.extend({'representation': label, **{k: value for k, value in r.items()
                                        if pd.notna(value)}} for r in missing.to_dict('records'))
            else:
                unavailable.append({'representation': label, 'task': 'probes', 'reason': 'explicit --probes stage not requested'})
        cards = {c['canonical']: c for c in v4.read_json(Path(oracle) / 'cards.json')}
        for label, matrix in matrices.items():
            normalized = unit_rows(matrix)
            for raw in ANCHORS:
                from .data import normalize_card_name
                anchor = normalize_card_name(raw)
                if anchor not in lookup or not masks[label][lookup[anchor]]:
                    continue
                a = lookup[anchor]
                values = normalized @ normalized[a]
                _, order = retrieval_metrics(values, [a], [(a + 1) % len(names)], names)
                for neighbor in order[:10]:
                    neighbor_sig = effect_signature(cards[names[neighbor]]) if names[neighbor] in cards else None
                    anchor_sig = effect_signature(cards[anchor]) if anchor in cards else None
                    qualitative.append({'representation': label, 'anchor': anchor, 'neighbor': names[neighbor],
                        'cosine': float(values[neighbor]), 'neighbor_signature': neighbor_sig,
                        'functional_differences': {field: {'anchor': anchor_sig[field], 'neighbor': neighbor_sig[field]}
                            for field in ('actions', 'targets', 'speed', 'secondary', 'restrictions', 'replacement', 'mana_value')
                            if anchor_sig[field] != neighbor_sig[field]} if anchor_sig and neighbor_sig else None})
        for a, b in [*combinations(('deck', 'text', 'mechanical'), 2), ('sgns_input_128', 'ppmi_svd_128')]:
            if a not in matrices or b not in matrices:
                continue  # older immutable exports lack the separate deck sources
            keep = masks[a] & masks[b]
            geometry_rows.append({'views': [a, b], **geometry(matrices[a][keep], matrices[b][keep])})
        rng = np.random.default_rng(4204)
        buckets = existing.se.frequency_buckets(counts)
        functional_queries = [r for r in heldout if r['status'] == 'covered' and r['relation'] in ('similarity', 'substitutability')]
        for row in synthetic:
            if row['split'] == 'test' and row['task'] == 'card_to_card' and row['anchor'] in lookup and row['positive'] in lookup:
                functional_queries.append({'id': v4.digest(row), 'query': [row['anchor']], 'targets': [row['positive']],
                    'status': 'covered', 'relation': 'synthetic_similarity', 'confidence': row['confidence']})
        eligible_anchors = {lookup[n] for r in functional_queries for n in r['query'] if masks['text'][lookup[n]]}
        selected = []
        for bucket in sorted(set(buckets)):
            ids = np.flatnonzero((buckets == bucket) & masks['text'])
            anchors = sorted(eligible_anchors & set(ids))[:50]
            rest = np.array(sorted(set(ids) - set(anchors)), dtype=int)
            selected.extend(anchors + rng.choice(rest, min(50 - len(anchors), len(rest)), replace=False).tolist())
        hidden_deck = cold_start(matrices['deck'], selected)
        simulated = concatenate([hidden_deck, matrices['text']])
        simulation_results = []
        for row in functional_queries:
            query = [lookup[n] for n in row['query']]
            if not set(query) & set(selected):
                continue
            for label, matrix in [('intact_deck_text', matrices['deck_text']), ('cold_start_deck_text', simulated), ('text_only', matrices['text'])]:
                scores = unit_rows(matrix) @ unit_rows(matrix)[query].mean(0)
                metrics, _ = retrieval_metrics(scores, query, [lookup[n] for n in row['targets']], names)
                simulation_results.append({'representation': label, 'id': row['id'],
                                           'confidence': row.get('confidence', 'curated_research'),
                                           'frequency_bucket': str(buckets[query[0]]), **metrics})
        if not simulation_results:
            unavailable.append({'task': 'cold_start_functional_retrieval', 'reason': 'no covered held-out functional query anchors'})
        if retrieval:
            frame = pd.concat(retrieval)
            frame.to_csv(dest / 'relations.csv', index=False)
            frame.groupby(['representation', 'relation', 'source']).agg(
                mrr=('mrr', 'mean'), recall10=('recall@10', 'mean'),
                recall50=('recall@50', 'mean'), queries=('id', 'count')).reset_index().to_csv(
                    dest / 'relation_summary.csv', index=False)
        if probe_frames:
            frame = pd.concat(probe_frames)
            frame.to_csv(dest / 'probes.csv', index=False)
            primary_scores = frame[(frame.group == 'all') & (frame.label == 'all')
                                   & frame.metric.isin(['macro_f1', 'mae'])]
            primary_scores.groupby(['representation', 'task', 'baseline', 'metric']).value.agg(
                ['mean', 'std', 'count']).reset_index().to_csv(dest / 'probe_summary.csv', index=False)
        if spellbook:
            v4.validate_stage(spellbook)
            full = load_multiview_embeddings(checkpoint, 'deck_text_mechanical', verify=False)
            full_lookup = full.card_to_idx
            mechanical_results = []
            for label in representations:
                bundle = load_multiview_embeddings(checkpoint, label, verify=False)
                norm = unit_rows(bundle.embedding_matrix)
                for row in v4.read_json(Path(spellbook) / 'completion.json'):
                    if row['split'] != 'test' or any(n not in full_lookup for n in row['context'] + [row['target']]):
                        continue
                    query = [full_lookup[n] for n in row['context']]
                    metrics, _ = retrieval_metrics(norm @ norm[query].mean(0), query, [full_lookup[row['target']]], full.vocab)
                    mechanical_results.append({'representation': label, 'variant_id': row['variant_id'],
                        'task': 'partner_retrieval' if len(query) == 1 else 'set_completion', **metrics})
            v4.write_json(dest / 'spellbook_cosine.json', mechanical_results)
            if mechanical_results:
                pd.DataFrame(mechanical_results).groupby(['representation', 'task']).agg(
                    mrr=('mrr', 'mean'), recall10=('recall@10', 'mean'),
                    recall50=('recall@50', 'mean'), queries=('variant_id', 'count')).reset_index().to_csv(
                        dest / 'spellbook_cosine_summary.csv', index=False)
        else:
            unavailable.append({'task': 'heldout_spellbook', 'reason': 'pass --spellbook prepared snapshot'})
        v4.write_json(dest / 'qualitative.json', qualitative)
        v4.write_json(dest / 'geometry.json', geometry_rows)
        v4.write_json(dest / 'cold_start.json', {'label': 'simulation, not future-card evidence',
            'seed': 4204, 'hidden_cards': [names[i] for i in selected], 'retrieval': simulation_results,
            'selection': 'up to 50 per frequency bin; held-out functional anchors first, remainder seeded random',
            'source_dimensions': {n: m.shape[1] for n, m in matrices.items()},
            'deck_block_zero_verified': bool(np.all(hidden_deck[selected] == 0))})
        v4.write_json(dest / 'coverage.json', {label: {'cards': int(mask.sum()), 'population': len(names),
            'probe_policy': 'all frozen label IDs/splits; unavailable sources explicitly zero; includes missing-view effect',
            'retrieval_policy': 'queries/targets must be source-covered; candidate population unchanged'} for label, mask in masks.items()})
        v4.write_json(dest / 'unavailable.json', unavailable)
        v4.write_json(dest / 'ablation_status.json', manifest['ablation_status'])
        (dest / 'report.md').write_text('''# Static v4 saved evaluation

These tables describe the explicit checkpoint and seeds in provenance.json.
No test-selected weights or pooling are fitted. Probe split variability is not
independent training variability. A single seed has no model-seed SD. The
historical probe helper's random control caps at 256d and is excluded here.
Coverage differs between views; inspect coverage.json before comparing scores.
relations.csv separates relation types and records training-family exclusions.
Cold-start results are a simulation. Qualitative examples are not quantitative
evidence. Inspect Spellbook overlap.json before any unseen-combo claim.
The established 384d deck representation remains the downstream baseline until
the new views demonstrate improvements in frozen evaluations and deck tasks.
''', encoding='utf-8')
