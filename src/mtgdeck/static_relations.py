"""Versioned relation-aware extension of static_geometry; no work on import.

Historical code and results remain immutable. All scorers share query sets,
full-vocabulary candidates, and deterministic name-order ties. O(V) scores/query.
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy.stats import t

from . import static_geometry as g

VERSION = 'static_v3_relations_v1'
POLICY = dict(version=VERSION, seed=202, metadata_queries=1500, concept_subsets=24,
              ks=[10, 20, 50], ties='score descending, canonical card name ascending',
              coverage='exclude whole query for missing vocabulary; metadata required only for metadata-derived labels; report missing metadata separately',
              query_sets='arithmetic mean of per-seed scores; PMI requires observation with every seed',
              relevance='binary unless positive finite grades supplied; unjudged candidates treated as nonrelevant',
              negatives='unobserved is not proven incompatible; endpoint-disjoint association diagnostic',
              primary_dimension=128)
MODES = ('input_cosine', 'context_cosine', 'combined_cosine', 'normalized_sum_cosine',
         'cross_dot', 'reverse_cross_dot', 'cross_cosine', 'symmetric_cross_dot', 'symmetric_cross_cosine')
QUALITATIVE = ['Lightning Bolt', "Thassa's Oracle", 'Demonic Consultation', 'Young Pyromancer',
               'Soulherder', 'Stoneforge Mystic', 'Reanimate', 'Brainstorm', 'Llanowar Elves',
               'Arcbound Ravager', 'Golgari Grave-Troll', 'Niv-Mizzet, Parun']


def canonical(values):
    return sorted({g.normalize_card_name(v) for v in values})


def signature(meta):
    """Structural proxy only: lacks Oracle text and must not be called a role label."""
    try:
        mana = float(meta['cmc'])
        colors = tuple(sorted(meta['color_identity']))
        types = tuple(sorted(set(meta['type_line'].split('—')[0].lower().split()) - {'legendary', 'snow', 'basic'}))
        if not types or not np.isfinite(mana) or not 0 <= mana <= 20:
            return None
        return colors, types, int(np.floor(mana))
    except (KeyError, TypeError, ValueError):
        return None


def build_benchmark(inputs, config, policy=None):
    policy = policy or POLICY
    names, metadata = inputs['names'], inputs['metadata']
    vocabulary = set(names)
    rows = []

    def add(query, targets, relation, source, concept='', curated=False, **extra):
        query, targets = canonical(query), canonical(targets)
        if set(query) & set(targets):
            raise ValueError('Query and targets must be disjoint')
        missing_vocab = sorted((set(query) | set(targets)) - vocabulary)
        missing_meta = sorted((set(query) | set(targets)) - set(metadata))
        metadata_required = relation in ('structural_proxy', 'cross_format_proxy')
        reason = ('empty_query_or_targets' if not query or not targets else
                  'missing_vocabulary' if missing_vocab else 'missing_metadata' if missing_meta and metadata_required else '')
        identity = json.dumps([relation, source, concept, query, targets, extra], sort_keys=True)
        rows.append(dict(id=f'{g.se.stable_hash(identity):016x}', query=query, targets=targets,
                         relation=relation, source=source, curated=curated, concept=concept,
                         format=None, metadata_available=not missing_meta, metadata_required=metadata_required,
                         missing_metadata=missing_meta, missing_vocabulary=missing_vocab,
                         exclusion_reason=reason, status='excluded' if reason else 'covered', **extra))

    for key, relation in [('roles', 'similarity'), ('slots', 'substitutability')]:
        for concept, members in config[key].items():
            for card in canonical(members):
                add([card], set(canonical(members)) - {card}, relation, 'curated_role_or_slot', concept, True)
    for query, targets in config['complements']:
        add([query], targets, 'complementarity', 'curated_mechanical', curated=True)
        for target in targets:
            add([target], [query], 'complementarity', 'curated_mechanical_reverse', curated=True)
    for record in config.get('cross_format', []):
        add(record['query'], record['targets'], 'cross_format', record['source'], record['concept'], True,
            source_format=record['source_format'], target_format=record['target_format'])

    rng = np.random.default_rng(policy['seed'])
    for concept, raw in inputs['concepts'].items():
        members = canonical(raw)
        seen = set()
        if len(members) < 3:
            continue
        for _ in range(policy['concept_subsets'] * 5):
            query = tuple(sorted(rng.choice(members, min(3, len(members) // 2), replace=False)))
            if query in seen:
                continue
            seen.add(query)
            add(query, set(members) - set(query), 'archetype', 'frozen_curated_concept_subsets', concept, True)
            if len(seen) >= policy['concept_subsets']:
                break
    evidence = inputs['evidence']
    for anchor, group in evidence[evidence.kind == 'corpus_supported'].groupby('anchor'):
        add([anchor], group.partner, 'association', 'frozen_corpus_incidence_selected')

    groups = defaultdict(list)
    for name in names:
        key = signature(metadata.get(name, {}))
        if key is not None:
            groups[key].append(name)
    eligible = sorted(n for members in groups.values() if len(members) >= 3 for n in members)
    anchors = rng.choice(eligible, min(len(eligible), policy['metadata_queries']), replace=False)
    for name in sorted(anchors):
        members = groups[signature(metadata[name])]
        add([name], set(members) - {name}, 'structural_proxy', 'frozen_metadata_color_type_mana',
            concept=str(signature(metadata[name])))
    # Cross-format proxy uses observed usage labels, NEVER legality. Curated transfer remains separate.
    target = inputs.get('targets', {}).get('format')
    if target is not None:
        ids, values, labels = target
        usage = {names[int(i)]: {str(labels[j]) for j in np.flatnonzero(y)} for i, y in zip(ids, values)}
        for name in sorted(anchors):
            formats = usage.get(name, set())
            others = [n for n in groups[signature(metadata[name])] if usage.get(n) and formats
                      and formats.isdisjoint(usage[n])]
            if others:
                add([name], others, 'cross_format_proxy', 'structural_match_disjoint_observed_usage',
                    source_format=sorted(formats), target_format=sorted(set().union(*(usage[n] for n in others))))
    unique = {row['id']: row for row in rows}
    return list(unique.values())


def coverage(records):
    frame = pd.DataFrame(records)
    return (frame.groupby(['relation', 'source', 'status', 'exclusion_reason', 'metadata_available'], dropna=False)
            .size().rename('queries').reset_index())


def rank_metrics(scores, query_ids, target_ids, names, grades=None, ks=(10, 20, 50)):
    """Standard first-hit MRR; AP@K denominator min(total relevant,K).

    -inf scores are unranked (absent PMI), assigned worst candidate rank for rank
    summaries but never awarded hits. Zero PPMI ties are explicitly deterministic.
    """
    scores = np.asarray(scores, dtype=float)
    targets = np.asarray(target_ids, dtype=int)
    if not len(targets) or len(set(targets)) != len(targets) or set(targets) & set(query_ids):
        raise ValueError('Invalid target set')
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('Nonfinite score')
    grades = np.ones(len(targets)) if grades is None else np.asarray(grades, dtype=float)
    if grades.shape != targets.shape or not np.isfinite(grades).all() or (grades <= 0).any():
        raise ValueError('Grades must be finite positive target-aligned values')
    mask = np.ones(len(scores), dtype=bool)
    mask[list(query_ids)] = False
    valid = np.flatnonzero(mask & np.isfinite(scores))
    order = valid[np.lexsort((np.asarray(names)[valid], -scores[valid]))]
    rank = np.full(len(scores), int(mask.sum()), dtype=int)
    rank[order] = np.arange(1, len(order) + 1)
    retrieved = np.isfinite(scores[targets])
    target_ranks = rank[targets]
    rr = 1 / target_ranks[retrieved].min() if retrieved.any() else 0.
    result = dict(mrr=float(rr), mean_target_rr=float(np.mean(np.where(retrieved, 1 / target_ranks, 0))),
                  mean_target_rank=float(target_ranks.mean()), median_target_rank=float(np.median(target_ranks)),
                  target_count=len(targets), candidate_count=int(mask.sum()),
                  zero_score_candidates=int(np.sum(scores[mask] == 0)))
    relevance = dict(zip(targets.tolist(), grades.tolist()))
    for k in ks:
        top = order[:k]
        hits = np.array([i in relevance for i in top], dtype=float)
        gain = np.array([np.expm1(np.log(2) * relevance.get(i, 0)) for i in top])
        ideal = np.sort(grades)[::-1][:k]
        dcg = np.sum(gain / np.log2(np.arange(len(top)) + 2))
        idcg = np.sum(np.expm1(np.log(2) * ideal) / np.log2(np.arange(len(ideal)) + 2))
        result.update({f'recall@{k}': float(hits.sum() / len(targets)),
                       f'precision@{k}': float(hits.sum() / min(k, mask.sum())),
                       f'ap@{k}': float(np.sum(np.cumsum(hits) / (np.arange(len(hits)) + 1) * hits) / min(k, len(targets))),
                       f'ndcg@{k}': float(dcg / idcg),
                       f'boundary_ties@{k}': int(np.sum(scores[valid] == scores[top[-1]])) if len(top) else 0})
    return result, order


def evaluate_records(scorer, records, names, counts):
    lookup = {n: i for i, n in enumerate(names)}
    rows, snapshots = [], []
    for record in records:
        if record['status'] != 'covered':
            continue
        query = [lookup[n] for n in record['query']]
        targets = [lookup[n] for n in record['targets']]
        scores = np.zeros(len(names), dtype=float)
        for anchor in query:
            scores += scorer(anchor) / len(query)
        metrics, order = rank_metrics(scores, query, targets, names, record.get('grades'))
        top = order[:20]
        row = {k: record[k] for k in ('id', 'relation', 'source', 'concept')}
        row.update(metrics)
        row['frequency_bucket'] = str(g.se.frequency_buckets(np.array([np.mean(counts[query])]))[0])
        row['neighbor_log_frequency'] = float(np.log1p(counts[top]).mean()) if len(top) else np.nan
        row['neighbor_popularity_excess'] = (row['neighbor_log_frequency'] - float(np.log1p(counts).mean()))
        # Inverse-log-frequency recall is descriptive, not a debiased causal estimate.
        weights = 1 / np.log2(2 + counts[targets])
        row['rarity_weighted_recall@50'] = float(np.sum(weights * np.isin(targets, order[:50])) / weights.sum())
        rows.append(row)
        snapshots.append({**{k: row[k] for k in ('id', 'relation', 'source', 'frequency_bucket')},
                          'neighbors': [names[i] for i in top], 'neighbor_log_frequency': row['neighbor_log_frequency']})
    return pd.DataFrame(rows), snapshots


def make_scorer(u, v, mode):
    if mode == 'reverse_cross_dot':
        return g.GeometryScorer(v, u, 'cross_dot')
    if mode == 'normalized_sum_cosine':
        return g.GeometryScorer(g.se.unit_vectors(u) + g.se.unit_vectors(v))
    return g.GeometryScorer(u, v, mode)


def hard_negative_candidates(anchor, positive, incidence, names, metadata, allowed):
    """Match positive partner's color/type/MV and log-frequency bucket, exclude observed pairs.

    Missing metadata yields no candidates, never an easy-negative fallback.
    Format and curated role matches are optional refinements for future experiments.
    """
    key = signature(metadata.get(names[positive], {}))
    if key is None:
        return np.array([], dtype=int)
    joint = incidence.row(anchor)
    observed = set(joint.indices[joint.data > 0]) | {anchor, positive}
    bucket = int(np.floor(np.log10(max(1, incidence.frequencies[positive]))))
    return np.array([int(i) for i in allowed if i not in observed
                     and signature(metadata.get(names[i], {})) == key
                     and int(np.floor(np.log10(max(1, incidence.frequencies[i])))) == bucket], dtype=int)


def compatibility_pairs(incidence, inputs, records, max_pairs=6000, seed=202):
    """Card-disjoint train/validation/test and canonical pair deduplication.

    Curated benchmark endpoints are additionally excluded from fitting. Corpus and
    structural proxies remain transductive retrieval diagnostics, not a held-out test.
    """
    names, metadata = inputs['names'], inputs['metadata']
    excluded = {n for r in records if r['curated'] for n in r['query'] + r['targets']}
    pools = defaultdict(list)
    for i, name in enumerate(names):
        if name not in excluded and signature(metadata.get(name, {})) is not None:
            h = g.se.stable_hash(f'{seed}:{name}') % 10
            pools['train' if h < 6 else 'validation' if h < 8 else 'test'].append(i)
    rng = np.random.default_rng(seed)
    rows, diagnostics, seen = [], [], set()
    for split in ('train', 'validation', 'test'):
        allowed = np.array(pools[split], dtype=int)
        allowed_set = set(allowed)
        budget = max_pairs * (3 if split == 'train' else 1) // 5
        accepted = 0
        for a in rng.permutation(allowed):
            co = incidence.row(int(a))
            positives = [int(b) for b, count in zip(co.indices, co.data) if b in allowed_set and b != a
                         and count >= 100 and count * incidence.contexts / (float(incidence.frequencies[a]) * incidence.frequencies[b]) >= 2]
            observed = set(co.indices)
            easy = np.array([i for i in allowed if i != a and i not in observed], dtype=int)
            for b in rng.permutation(positives)[:8]:
                pair = tuple(sorted((int(a), int(b))))
                if pair in seen:
                    continue
                hard = hard_negative_candidates(int(a), int(b), incidence, names, metadata, allowed)
                diagnostics.append(dict(split=split, anchor=int(a), positive=int(b), hard_candidates=len(hard),
                                        easy_candidates=len(easy), positive_frequency=int(incidence.frequencies[b])))
                hard = [i for i in hard if tuple(sorted((int(a), int(i)))) not in seen]
                if not hard:
                    continue
                negative = int(rng.choice(hard))
                seen.update([pair, tuple(sorted((int(a), negative)))])
                for target, label in ((int(b), 1), (negative, 0)):
                    left, right = sorted((int(a), target))
                    rows.append(dict(a=left, b=right, label=label, split=split))
                if len(easy):
                    diagnostics[-1]['easy_example'] = int(rng.choice(easy))
                    diagnostics[-1]['hard_example'] = negative
                accepted += 2
                if accepted >= budget:
                    break
            if accepted >= budget:
                break
    return pd.DataFrame(rows, columns=['a', 'b', 'label', 'split']), pd.DataFrame(diagnostics)


def fit_compatibility(u, v, pairs, incidence):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    for split in ('train', 'validation', 'test'):
        counts = pairs[pairs.split == split].label.value_counts()
        if len(counts) < 2 or counts.min() < 20:
            return None, dict(status='unavailable', reason=f'{split}: fewer than 20 pairs/class')
    unit = g.se.unit_vectors(u)
    features = unit[pairs.a.to_numpy()] * unit[pairs.b.to_numpy()]
    train, valid, test = [(pairs.split == s).to_numpy() for s in ('train', 'validation', 'test')]
    candidates = []
    for c in (.01, .1, 1.):
        model = LogisticRegression(C=c, max_iter=2000, random_state=202).fit(features[train], pairs.label[train])
        auc = roc_auc_score(pairs.label[valid], model.decision_function(features[valid]))
        candidates.append((auc, c, model))
    _, chosen, model = max(candidates, key=lambda item: item[0])
    methods = {'learned_diagonal': lambda a, b: float(unit[a] @ (model.coef_[0] * unit[b])),
               'input_cosine': lambda a, b: float(unit[a] @ unit[b]),
               'cross_dot': lambda a, b: float((u[a] @ v[b] + u[b] @ v[a]) / 2),
               'ppmi': lambda a, b: float(incidence(a)[b])}
    results = []
    for split, mask in [('validation', valid), ('test', test)]:
        subset = pairs[mask]
        for method, score in methods.items():
            pred = [score(int(a), int(b)) for a, b in zip(subset.a, subset.b)]
            results.append(dict(split=split, scorer=method, auc=float(roc_auc_score(subset.label, pred)),
                                average_precision=float(average_precision_score(subset.label, pred)), pairs=len(subset)))
    return model.coef_[0].astype(np.float32), dict(status='complete', parameter_count=u.shape[1] + 1,
            selected_C=chosen, intercept=float(model.intercept_[0]), metrics=results,
            note='Frozen vectors; card-disjoint supervised splits; corpus used in embedding pretraining. Not external generalization.')


def validate(root, output):
    root, source, output = g.validate(root, output)
    manifest = g.read_json(output / 'relations_manifest.json')
    expected = {'policy': POLICY, 'implementation': g.se.sha256_file(__file__),
                'config': g.se.sha256_file(root / 'configs/card2vec_relations.json')}
    if manifest != expected:
        raise ValueError('Relation implementation/config changed; choose a fresh output directory')
    if not (output / 'benchmark/complete.json').exists():
        raise ValueError('Incomplete benchmark stage')
    return root, source, output


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    historical = root / 'artifacts/card2vec/static_v3'
    if output == historical or historical in output.parents or output in historical.parents:
        raise ValueError('Relation output must be disjoint from historical v3')
    if (output / 'relations_manifest.json').exists():
        return validate(root, output)
    g.audit(root, output)
    source = root / 'artifacts/card2vec/static_v2'
    inputs = joblib.load(source / 'evaluation_inputs.joblib')
    config = g.read_json(root / 'configs/card2vec_relations.json')
    with g.stage_directory(output / 'benchmark') as dest:
        records = build_benchmark(inputs, config)
        g.write_json(dest / 'queries.json', records)
        coverage(records).to_csv(dest / 'coverage.csv', index=False)
        g.write_json(dest / 'availability.json', {'curated_cross_format': len(config['cross_format']),
                     'note': 'No curated cross-format assertions supplied; proxies are not ground truth.'})
    g.write_json(output / 'relations_manifest.json', {'policy': POLICY,
                 'implementation': g.se.sha256_file(__file__), 'config': g.se.sha256_file(root / 'configs/card2vec_relations.json')})
    return root, source, output


def save_run(dest, scorer, records, inputs, info):
    with g.stage_directory(dest):
        g.write_json(dest / 'model.json', info)
        metrics, snapshots = evaluate_records(scorer, records, inputs['names'], inputs['counts'])
        metrics.to_csv(dest / 'metrics.csv', index=False)
        g.write_json(dest / 'neighbors.json', snapshots)
        names, counts = inputs['names'], inputs['counts']
        lookup = {n: i for i, n in enumerate(names)}
        anchors = canonical(QUALITATIVE) + [names[int(np.argmin(counts))]]
        ids = [lookup[n] for n in anchors if n in lookup]
        g.neighborhoods(scorer, names, counts, ids, k=10).to_csv(dest / 'qualitative.csv', index=False)
        g.write_json(dest / 'qualitative_missing.json', [n for n in anchors if n not in lookup])


def evaluate(root, output, family='sgns'):
    _, source, output = validate(root, output)
    inputs = joblib.load(source / 'evaluation_inputs.joblib')
    records = g.read_json(output / 'benchmark/queries.json')
    if family in ('baselines', 'compatibility') and not (output / 'matrices/complete.json').exists():
        raise ValueError('Run matrices first')
    if family == 'baselines':
        for mode in ('raw_incidence', 'incidence_cosine', 'pmi', 'ppmi', 'popularity'):
            dest = output / 'relations' / mode
            if (dest / 'complete.json').exists():
                continue
            scorer = ((lambda a: inputs['counts'].astype(float)) if mode == 'popularity' else
                      g.IncidenceScorer(output / 'matrices', inputs['counts'], mode))
            save_run(dest, scorer, records, inputs, dict(scorer=mode, dimension=0, seed='deterministic'))
        return
    if family == 'compatibility':
        incidence = g.IncidenceScorer(output / 'matrices', inputs['counts'], 'ppmi')
        dest = output / 'supervision'
        if not (dest / 'complete.json').exists():
            with g.stage_directory(dest):
                pairs, diagnostics = compatibility_pairs(incidence, inputs, records)
                pairs.to_csv(dest / 'pairs.csv', index=False)
                diagnostics.to_csv(dest / 'negative_diagnostics.csv', index=False)
        pairs = pd.read_csv(dest / 'pairs.csv')
    for dimension in (128, 256):
        for seed in (42, 43, 44):
            modes = MODES if family == 'sgns' else ('ppmi_svd_cosine', 'ppmi_svd_dot') if family == 'svd' else ('compatibility',)
            pending = [mode for mode in modes if not (output / f'relations/{mode}_d{dimension}_s{seed}/complete.json').exists()]
            if not pending:
                continue
            if family == 'svd':
                svd = output / f'svd/d{dimension}_s{seed}'
                if not (svd / 'complete.json').exists():
                    raise ValueError(f'SVD incomplete: {svd}')
                u, v = np.load(svd / 'vectors.npy'), None
            else:
                u, v = g.aligned_model(source / f'seed_{seed}/{dimension}/sample_0/model.gensim', inputs['names'])
                if v is None:
                    raise ValueError('Context vectors missing; do not retrain silently')
            for mode in pending:
                dest = output / f'relations/{mode}_d{dimension}_s{seed}'
                if family == 'compatibility':
                    fit_dir = output / f'compatibility_v1/d{dimension}_s{seed}'
                    if not (fit_dir / 'complete.json').exists():
                        with g.stage_directory(fit_dir):
                            weights, fit = fit_compatibility(u, v, pairs, incidence)
                            g.write_json(fit_dir / 'fit.json', fit)
                            if weights is not None:
                                np.save(fit_dir / 'weights.npy', weights)
                    if not (fit_dir / 'weights.npy').exists():
                        continue
                    scorer = g.GeometryScorer(u, mode='compatibility', weights=np.load(fit_dir / 'weights.npy'))
                elif family == 'svd':
                    scorer = g.GeometryScorer(u, mode='input_cosine' if mode.endswith('cosine') else 'svd_dot')
                else:
                    scorer = make_scorer(u, v, mode)
                print(f'Evaluate relation benchmark: {dest.name}', flush=True)
                save_run(dest, scorer, records, inputs, dict(scorer=mode, dimension=dimension, seed=seed))


def probes(root, output):
    _, source, output = validate(root, output)
    inputs = joblib.load(source / 'evaluation_inputs.joblib')
    for dimension in (128, 256):
        for seed in (42, 43, 44):
            u, v = g.aligned_model(source / f'seed_{seed}/{dimension}/sample_0/model.gensim', inputs['names'])
            representations = {'input': u, 'context': v, 'combined': u + v,
                               'input_unit': g.se.unit_vectors(u), 'context_unit': g.se.unit_vectors(v),
                               'normalized_sum': g.se.unit_vectors(u) + g.se.unit_vectors(v)}
            svd = output / f'svd/d{dimension}_s{seed}'
            if (svd / 'complete.json').exists():
                representations['ppmi_svd'] = np.load(svd / 'vectors.npy')
            for representation, vectors in representations.items():
                dest = output / f'relation_probes/{representation}_d{dimension}_s{seed}'
                if (dest / 'complete.json').exists():
                    continue
                with g.stage_directory(dest):
                    frames, unavailable = [], []
                    for task in ('color', 'type', 'format', 'archetype', 'mana', 'mana_spells'):
                        if task not in inputs['targets']:
                            unavailable.append(dict(task=task, reason='missing frozen target'))
                            continue
                        frame, missing = g.v2.probe_long(vectors, inputs['counts'], inputs['names'],
                                    inputs['metadata'], inputs['targets'][task], task)
                        frames.append(frame)
                        unavailable.extend(missing.to_dict('records'))
                    if frames:
                        pd.concat(frames).to_csv(dest / 'probes.csv', index=False)
                    g.write_json(dest / 'unavailable.json', unavailable)


def report(root, output):
    """Append-only report snapshots keyed by completed stage content."""
    _, _, output = validate(root, output)
    frames, neighborhoods, qualitative = [], defaultdict(dict), []
    completed = []
    for path in sorted((output / 'relations').glob('*/complete.json')):
        dest = path.parent
        info = g.read_json(dest / 'model.json')
        frame = pd.read_csv(dest / 'metrics.csv')
        for key, value in info.items():
            frame[key] = value
        frames.append(frame)
        completed.append(dest.name)
        neighborhoods[(info['scorer'], info['dimension'])][str(info['seed'])] = g.read_json(dest / 'neighbors.json')
        q = pd.read_csv(dest / 'qualitative.csv')
        for key, value in info.items():
            q[key] = value
        qualitative.append(q)
    expected = [f'{m}_d{d}_s{s}' for m in (*MODES, 'ppmi_svd_cosine', 'ppmi_svd_dot', 'compatibility')
                for d in (128, 256) for s in (42, 43, 44)] + ['raw_incidence', 'incidence_cosine', 'pmi', 'ppmi', 'popularity']
    state = {'completed': completed, 'pending': sorted(set(expected) - set(completed)),
             'completed_probes': sorted(p.parent.name for p in (output / 'relation_probes').glob('*/complete.json')),
             'compatibility_fits': {p.parent.name: g.read_json(p) for p in sorted((output / 'compatibility_v1').glob('*/fit.json'))}}
    state['pending_probes'] = sorted({f'{m}_d{d}_s{s}' for m in ('input', 'context', 'combined', 'input_unit', 'context_unit', 'normalized_sum', 'ppmi_svd')
                                     for d in (128, 256) for s in (42, 43, 44)} - set(state['completed_probes']))
    state['status'] = 'partial' if state['pending'] or state['pending_probes'] else 'complete'
    fingerprint = f'{g.se.stable_hash(json.dumps(state, sort_keys=True)):016x}'
    dest = output / 'reports' / fingerprint
    if (dest / 'complete.json').exists():
        return dest
    with g.stage_directory(dest):
        g.write_json(dest / 'status.json', state)
        coverage(g.read_json(output / 'benchmark/queries.json')).to_csv(dest / 'coverage.csv', index=False)
        if not frames:
            return dest
        frame = pd.concat(frames, ignore_index=True)
        group = ['scorer', 'dimension', 'seed', 'relation', 'source']
        metrics = ['mrr', 'mean_target_rr', 'mean_target_rank', 'median_target_rank', 'neighbor_log_frequency',
                   'neighbor_popularity_excess', 'rarity_weighted_recall@50'] + [f'{m}@{k}' for m in ('recall', 'precision', 'ap', 'ndcg') for k in POLICY['ks']]
        per_seed = frame.groupby(group)[metrics].mean().reset_index()
        long = per_seed.melt(group, var_name='metric', value_name='value')
        summary = long.groupby(['scorer', 'dimension', 'relation', 'source', 'metric']).value.agg(['mean', 'std', 'count']).reset_index()
        summary['ci95_halfwidth'] = [float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
                                    for sd, n in zip(summary['std'], summary['count'])]
        summary.to_csv(dest / 'summary.csv', index=False)
        per_seed.to_csv(dest / 'per_seed.csv', index=False)
        frame.groupby(group)[['mrr', 'median_target_rank']].median().to_csv(dest / 'query_medians.csv')
        frame.groupby(group + ['frequency_bucket'])[metrics].mean().to_csv(dest / 'by_frequency.csv')
        summary[summary.metric == 'mrr'].pivot_table(index=['scorer', 'dimension'], columns=['relation', 'source'], values='mean').to_csv(dest / 'relation_mrr.csv')
        stability = []
        for (scorer, dimension), seeds in neighborhoods.items():
            for a, b in combinations(sorted(seeds), 2):
                left, right = [{r['id']: r for r in seeds[s]} for s in (a, b)]
                for key in sorted(left.keys() & right.keys()):
                    x, y = set(left[key]['neighbors']), set(right[key]['neighbors'])
                    if not x or not y:
                        continue
                    stability.append(dict(scorer=scorer, dimension=dimension, seed_a=a, seed_b=b,
                        **{k: left[key][k] for k in ('id', 'relation', 'source', 'frequency_bucket')},
                        jaccard=len(x & y) / len(x | y), neighbor_log_frequency=(left[key]['neighbor_log_frequency'] + right[key]['neighbor_log_frequency']) / 2))
        if stability:
            st = pd.DataFrame(stability)
            st.to_csv(dest / 'stability.csv', index=False)
            st.groupby(['scorer', 'dimension', 'relation', 'source', 'frequency_bucket'])[['jaccard', 'neighbor_log_frequency']].mean().to_csv(dest / 'stability_summary.csv')
        pd.concat(qualitative).to_csv(dest / 'qualitative.csv', index=False)
    return dest
