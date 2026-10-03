"""Isolated static_v3 geometry diagnostics. Historical artifacts are read-only.

No work on import. CLI stages are explicit; the notebook only reviews saved files.
Counts/PPMI are row-sharded sparse matrices, including during randomized SVD.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
import importlib.metadata
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from . import static_embeddings as se
from . import static_experiment as v2
from .data import normalize_card_name

SEMANTIC = [
    ('Lightning Bolt', 'Chain Lightning', 'one-mana three-damage burn spells'),
    ('Young Pyromancer', 'Third Path Iconoclast', 'noncreature-spell token engines'),
    ('Soulherder', 'Teleportation Circle', 'repeatable end-step blink engines'),
]
COMPLEMENTARY = [
    ("Thassa's Oracle", 'Demonic Consultation'),
    ('Niv-Mizzet, Parun', 'Curiosity'),
    ('Stuffy Doll', 'Guilty Conscience'),
    ("Painter's Servant", 'Grindstone'),
    ('Dark Depths', "Thespian's Stage"),
]
POLICY = {
    'version': 'static_v3', 'dimensions': [128, 256], 'seeds': [42, 43, 44],
    'split_seeds': [101, 102, 103], 'row_block': 128,
    'counts': 'C_ij = number of frozen contexts containing both i and j; symmetric; diagonal zero; quantities ignored',
    'context_weight': 'one per containing context for each pair; large contexts contribute more total pairs than small contexts',
    'sgns_weight_difference': 'SGNS samples 32 pairs per context; full incidence does not match its per-pair exposure weighting. Objective effects cannot be isolated causally.',
    'normalized_incidence': 'C_ij / sqrt(f_i f_j), same formula as v2 incidence baseline',
    'pmi': 'log2(C_ij * N / (f_i f_j)); binary-context marginals, no smoothing; absent pairs score -inf',
    'ppmi': 'max(PMI, 0); absent pairs and diagonal zero',
    'ties': 'pessimistic rank: all non-self candidates with score >= partner score; -inf absent PMI pairs rank last',
    'svd': {'algorithm': 'streamed randomized SVD of sparse PPMI; QR-normalized power iteration',
            'oversamples': 10, 'power_iterations': 3, 'embedding': 'U sqrt(S)',
            'replicates': 'randomized factorization seeds on one fixed matrix, not independent corpora'},
    'compatibility': {'model': 'symmetric diagonal bilinear e_i^T diag(w) e_j with intercept',
                      'labels': 'balanced sampled pairs: positives C>=100 and lift>=2; negatives C=0; ambiguous pairs excluded',
                      'holdout': 'all covered benchmark endpoints excluded from scorer fitting and pair validation; separate pair-hash holdout inside remaining population',
                      'max_pairs': 20000, 'proposal_budget': 2000000, 'seed': 202,
                      'regularization_C': 0.1,
                      'scope': 'transductive corpus-association diagnostic, not ground-truth mechanical synergy'},
    'semantic_pairs': SEMANTIC, 'complementary_pairs': COMPLEMENTARY,
    'historical_mechanical_pairs': se.COMBOS,
    'stability': 'top-20 Jaccard across seed pairs, fixed frequency-stratified anchors; correlated seed pairs are descriptive, not independent replicates',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temp.replace(path)


def locations(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    source = root / 'artifacts/card2vec/static_v2'
    for historical in (source, root / 'artifacts/card2vec/static_v1'):
        if output == historical or historical in output.parents or output in historical.parents:
            raise ValueError('V3 output must be disjoint from historical artifacts')
    return root, source, output


def implementation_hashes():
    return {Path(p).name: se.sha256_file(p) for p in (__file__, se.__file__, v2.__file__)}


def validate(root, output):
    root, source, output = locations(root, output)
    manifest = read_json(output / 'provenance.json')
    if read_json(output / 'policy.json') != json.loads(json.dumps(POLICY)):
        raise ValueError('Policy changed; choose a fresh v3 output directory')
    if manifest['implementation'] != implementation_hashes():
        raise ValueError('Implementation changed; choose a fresh v3 output directory')
    for relative, expected in manifest['source_hashes'].items():
        if se.sha256_file(source / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    return root, source, output


@contextmanager
def stage_directory(path):
    """Completed stages are immutable; incomplete stages require explicit relocation."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Stage already exists: {path}; use saved output or move an incomplete stage aside')
    path.mkdir(parents=True)
    started = time.perf_counter()
    yield path
    write_json(path / 'complete.json', {'seconds': time.perf_counter() - started})


def aligned_model(path, names):
    from gensim.models import Word2Vec
    model = Word2Vec.load(str(path))
    order = np.array([model.wv.key_to_index[name] for name in names])
    u = np.asarray(model.wv.vectors[order], dtype=np.float32)
    v = getattr(model, 'syn1neg', None)
    return u, None if v is None else np.asarray(v[order], dtype=np.float32)


def audit(root, output):
    """Read exact v2 packed inputs and models; never invoke v2 preparation/training."""
    root, source, output = locations(root, output)
    if (output / 'provenance.json').exists():
        validate(root, output)
        return read_json(output / 'model_inventory.json')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Nonempty output without a completed audit; choose a fresh directory')
    output.mkdir(parents=True, exist_ok=True)
    names, tokens, offsets, counts = se.load_prepared(source / 'prepared')
    if len(counts) != len(names) or offsets[-1] != len(tokens) or (counts <= 0).any():
        raise ValueError('Invalid packed corpus')
    inputs = joblib.load(source / 'evaluation_inputs.joblib')
    if inputs['names'] != names or not np.array_equal(inputs['counts'], counts):
        raise ValueError('Evaluation vocabulary/counts differ from packed corpus')
    preparation = read_json(source / 'prepared/preparation.json')
    if preparation['contexts'] != len(offsets) - 1:
        raise ValueError('Packed context count differs from preparation manifest')
    files = [source / 'evaluation_inputs.joblib', source / 'evaluation_policy.json',
             source / 'experiment.json', source / 'quality_exclusions.json',
             source / 'quality_source_manifest.json', source / 'provenance_summary.json']
    files += [source / 'prepared' / name for name in
              ('names.json', 'tokens.i32', 'offsets.npy', 'counts.npy', 'preparation.json')]
    models = []
    for seed in POLICY['seeds']:
        for dimension in POLICY['dimensions']:
            path = source / f'seed_{seed}/{dimension}/sample_0'
            config = read_json(path / 'config.json')
            if (config['preparation'] != preparation or config['config']['seed'] != seed
                    or config['dimension'] != dimension
                    or config['metadata_sha256'] != read_json(source / 'evaluation_policy.json')['metadata_sha256']):
                raise ValueError(f'Model provenance mismatch: {path}')
            expected = json.loads(json.dumps(v2.experiment_config(seed).__dict__))
            if config['config'] != expected or config['subsampling'] != 0:
                raise ValueError('Model hyperparameters differ from the frozen v2 policy')
            u, v = aligned_model(path / 'model.gensim', names)
            if u.shape != (len(names), dimension) or not np.array_equal(u, np.load(path / 'vectors.npy')):
                raise ValueError('Saved model input vectors differ from exported vectors')
            if not np.isfinite(u).all() or (v is not None and (v.shape != u.shape or not np.isfinite(v).all())):
                raise ValueError('Invalid model vectors')
            models.append({'seed': seed, 'dimension': dimension, 'path': str(path.relative_to(source)),
                           'origin': 'frozen_static_v2', 'context_vectors': v is not None,
                           'retraining_required': v is None, 'vocabulary': len(names)})
            files += [path / 'config.json', path / 'vectors.npy', *path.glob('model.gensim*')]
    source_hashes = {str(p.relative_to(source)).replace('\\', '/'): se.sha256_file(p) for p in files}
    write_json(output / 'policy.json', POLICY)
    write_json(output / 'model_inventory.json', models)
    write_json(output / 'provenance.json', {
        'source': 'artifacts/card2vec/static_v2', 'source_hashes': source_hashes,
        'implementation': implementation_hashes(), 'training_contexts': len(offsets) - 1,
        'vocabulary': len(names), 'preparation': preparation,
        'evaluation_policy': read_json(source / 'evaluation_policy.json'),
        'packages': {p: importlib.metadata.version(p) for p in
                     ('numpy', 'scipy', 'pandas', 'scikit-learn', 'gensim', 'joblib')},
        'python': platform.python_version(),
        'note': 'Use frozen metadata/targets from the evaluation cache; do not join a newer live metadata snapshot.'})
    evidence = benchmarks(inputs)
    evidence.to_csv(output / 'benchmark_coverage.csv', index=False)
    write_json(output / 'concepts.json', inputs['concepts'])
    return models


def benchmarks(inputs):
    names, metadata = inputs['names'], inputs['metadata']
    vocabulary = set(names)
    rows = []
    groups = [('semantic_similarity', [(a, b, reason) for a, b, reason in SEMANTIC]),
              ('complementary_mechanical', [(a, b, 'explicit mechanical relationship') for a, b in COMPLEMENTARY]),
              ('mechanical_v2', [(a, b, 'unchanged historical benchmark') for a, b in se.COMBOS])]
    for kind, pairs in groups:
        for pair_id, (a, b, reason) in enumerate(pairs):
            for a, b in ((a, b), (b, a)):
                a, b = normalize_card_name(a), normalize_card_name(b)
                status = ('missing_vocabulary' if a not in vocabulary or b not in vocabulary else
                          'missing_or_ambiguous_metadata' if a not in metadata or b not in metadata else 'covered')
                # Historical coverage remains vocabulary-only, exactly as v2.
                if kind == 'mechanical_v2' and a in vocabulary and b in vocabulary:
                    status = 'covered'
                rows.append(dict(kind=kind, pair_id=f'{kind}:{pair_id}', anchor=a, partner=b,
                                 reason=reason, status=status))
    for i, row in inputs['evidence'].query("kind == 'corpus_supported'").iterrows():
        rows.append(dict(kind='corpus_supported', pair_id=f'corpus:{i}', anchor=row.anchor,
                         partner=row.partner, reason='frozen v2 transductive incidence-selected partner', status=row.status))
    return pd.DataFrame(rows)


def ppmi_block(counts_block, start, frequencies, contexts):
    coo = counts_block.tocoo()
    values = np.log2(coo.data.astype(np.float64) * contexts /
                     (frequencies[start + coo.row].astype(np.float64) * frequencies[coo.col]))
    values = np.maximum(values, 0).astype(np.float32)
    result = sparse.csr_matrix((values, (coo.row, coo.col)), shape=coo.shape)
    result.eliminate_zeros()
    return result


def build_matrices(root, output, max_disk_gib=12):
    _, source, output = validate(root, output)
    names, tokens, offsets, frequencies = se.load_prepared(source / 'prepared')
    with stage_directory(output / 'matrices') as dest:
        # CSR is context x card; binary packed contexts already have unique IDs.
        x = sparse.csr_matrix((np.ones(len(tokens), dtype=np.float32), tokens, offsets),
                              shape=(len(offsets) - 1, len(names)))
        x.sort_indices()
        observed = np.asarray(x.sum(axis=0)).ravel()
        if not np.array_equal(observed, frequencies):
            raise ValueError('Packed incidence disagrees with frozen frequency counts')
        xt = x.T.tocsr()
        shards, stored = [], 0
        for start in range(0, len(names), POLICY['row_block']):
            stop = min(start + POLICY['row_block'], len(names))
            c = (xt[start:stop] @ x).tocsr()
            # Eliminate diagonal without converting even one block to dense.
            coo = c.tocoo()
            keep = coo.col != start + coo.row
            c = sparse.csr_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=c.shape)
            p = ppmi_block(c, start, frequencies, len(offsets) - 1)
            size = sum(a.nbytes for m in (c, p) for a in (m.data, m.indices, m.indptr))
            stored += size
            if stored > max_disk_gib * 1024**3:
                raise MemoryError('Sparse shard budget exceeded; partial stage retained, no completion marker')
            sparse.save_npz(dest / f'counts_{start}.npz', c)
            sparse.save_npz(dest / f'ppmi_{start}.npz', p)
            shards.append(dict(start=start, stop=stop, count_nnz=c.nnz, ppmi_nnz=p.nnz))
            print(f'Incidence/PPMI rows {stop}/{len(names)}', flush=True)
        write_json(dest / 'index.json', {'shape': [len(names), len(names)], 'contexts': len(offsets) - 1,
                                        'shards': shards, 'uncompressed_bytes': stored,
                                        'incidence_storage_bytes': sum(a.nbytes for m in (x, xt) for a in (m.data, m.indices, m.indptr))})


class ShardedPPMI:
    def __init__(self, directory):
        self.directory = Path(directory)
        if not (self.directory / 'complete.json').exists():
            raise ValueError('Matrix stage is incomplete')
        self.index = read_json(self.directory / 'index.json')
        self.shape = tuple(self.index['shape'])

    def matmat(self, right, transpose=False):
        result = np.zeros((self.shape[0], right.shape[1]), dtype=np.float32)
        for shard in self.index['shards']:
            start, stop = shard['start'], shard['stop']
            block = sparse.load_npz(self.directory / f'ppmi_{start}.npz')
            if transpose:
                result += block.T @ right[start:stop]
            else:
                result[start:stop] = block @ right
        return result


def randomized_svd(operator, dimension, seed, iterations=3, oversamples=10):
    """Bounded dense V x (d+10) work arrays; never V x V."""
    width = min(dimension + oversamples, min(operator.shape))
    if dimension >= min(operator.shape):
        raise ValueError('SVD rank must be smaller than the vocabulary')
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((operator.shape[1], width), dtype=np.float32)
    q, _ = np.linalg.qr(operator.matmat(omega), mode='reduced')
    for _ in range(iterations):
        z, _ = np.linalg.qr(operator.matmat(q, transpose=True), mode='reduced')
        q, _ = np.linalg.qr(operator.matmat(z), mode='reduced')
    small = operator.matmat(q, transpose=True).T
    left, singular, _ = np.linalg.svd(small, full_matrices=False)
    embeddings = (q @ left[:, :dimension]) * np.sqrt(singular[:dimension])
    return embeddings.astype(np.float32), singular[:dimension]


def build_svd(root, output):
    _, _, output = validate(root, output)
    operator = ShardedPPMI(output / 'matrices')
    for dimension in POLICY['dimensions']:
        for seed in POLICY['seeds']:
            dest = output / f'svd/d{dimension}_s{seed}'
            if (dest / 'complete.json').exists():
                continue
            with stage_directory(dest):
                vectors, singular = randomized_svd(operator, dimension, seed)
                np.save(dest / 'vectors.npy', vectors)
                np.save(dest / 'singular_values.npy', singular)
                write_json(dest / 'model.json', dict(dimension=dimension, seed=seed, origin='static_v3_ppmi_svd',
                                                      **POLICY['svd']))


class GeometryScorer:
    """One O(Vd) query at a time; no all-pairs dense score matrix."""
    def __init__(self, u, v=None, mode='input_cosine', weights=None):
        self.mode = mode
        self.u, self.v = u, v
        if mode in ('input_cosine', 'context_cosine', 'cross_cosine', 'symmetric_cross_cosine', 'compatibility'):
            self.u = se.unit_vectors(u)
            self.v = None if v is None else se.unit_vectors(v)
        if mode == 'combined_cosine':
            self.u = se.unit_vectors(u + v)
        self.weights = weights

    def __call__(self, anchor):
        if self.mode in ('input_cosine', 'combined_cosine'):
            return self.u @ self.u[anchor]
        if self.mode == 'context_cosine':
            return self.v @ self.v[anchor]
        if self.mode == 'cross_dot':
            return self.v @ self.u[anchor]
        if self.mode == 'cross_cosine':
            return self.v @ self.u[anchor]
        if self.mode in ('symmetric_cross_dot', 'symmetric_cross_cosine'):
            return (self.v @ self.u[anchor] + self.u @ self.v[anchor]) / 2
        if self.mode == 'svd_dot':
            return self.u @ self.u[anchor]
        if self.mode == 'compatibility':
            return self.u @ (self.weights * self.u[anchor])
        raise ValueError(self.mode)


class IncidenceScorer:
    def __init__(self, directory, frequencies, mode):
        self.directory, self.frequencies, self.mode = Path(directory), frequencies, mode
        index = read_json(self.directory / 'index.json')
        self.contexts = index['contexts']
        self.shards = index['shards']
        self.cached_start, self.cached = None, None

    def row(self, anchor):
        shard = next(s for s in self.shards if s['start'] <= anchor < s['stop'])
        start = shard['start']
        if self.cached_start != start:
            self.cached = sparse.load_npz(self.directory / f'counts_{start}.npz')
            self.cached_start = start
        return self.cached.getrow(anchor - start)

    def __call__(self, anchor):
        row = self.row(anchor)
        scores = np.zeros(len(self.frequencies), dtype=np.float64)
        if self.mode == 'pmi':
            scores.fill(-np.inf)
        if self.mode == 'raw_incidence':
            values = row.data
        elif self.mode == 'incidence_cosine':
            values = row.data / np.sqrt(float(self.frequencies[anchor]) * self.frequencies[row.indices])
        else:
            values = np.log2(row.data.astype(np.float64) * self.contexts /
                             (float(self.frequencies[anchor]) * self.frequencies[row.indices]))
            if self.mode == 'ppmi':
                values = np.maximum(values, 0)
        scores[row.indices] = values
        return scores


def pessimistic_rank(scores, anchor, partner):
    # Explicit candidate mask handles -inf targets without accidentally counting self.
    candidates = np.arange(len(scores)) != anchor
    return int(np.count_nonzero(scores[candidates] >= scores[partner]))


def evaluate_pairs(scorer, names, evidence):
    lookup = {name: i for i, name in enumerate(names)}
    rows = []
    for anchor, group in evidence[evidence.status == 'covered'].groupby('anchor'):
        ia = lookup[anchor]
        scores = scorer(ia)
        if np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError('Invalid pair scores')
        for record in group.to_dict('records'):
            ib = lookup[record['partner']]
            rank = pessimistic_rank(scores, ia, ib)
            score = float(scores[ib])
            rows.append({**record, 'score': score if np.isfinite(score) else 'absent_pair_negative_infinity',
                         'rank': rank, 'rr': 1 / rank,
                         **{f'recall{k}': float(rank <= k) for k in (10, 20, 50)}})
    return pd.DataFrame(rows)


def neighborhoods(scorer, names, counts, ids, k=20):
    rows = []
    for i in ids:
        scores = scorer(i).copy()
        candidates = np.arange(len(names)) != i
        valid = np.flatnonzero(candidates & np.isfinite(scores))
        # Stable index tie-breaking for snapshots; retrieval uses conservative ties.
        top = valid[np.lexsort((valid, -scores[valid]))[:k]]
        rows.append(dict(card=names[i], frequency=int(counts[i]),
                         bucket=str(se.frequency_buckets(counts[i:i + 1])[0]),
                         neighbors=json.dumps([names[j] for j in top]),
                         zero_score_candidates=int(np.count_nonzero(scores[valid] == 0)),
                         boundary_ties=int(np.count_nonzero(scores[valid] == scores[top[-1]])) if len(top) else 0))
    return pd.DataFrame(rows)


def compatibility_pairs(incidence, names, evidence, policy=None):
    """Exclude all benchmark cards. Labels are associations, never combo labels."""
    policy = policy or POLICY['compatibility']
    excluded = set(evidence.anchor) | set(evidence.partner)
    eligible = np.array([i for i, n in enumerate(names) if n not in excluded])
    rng = np.random.default_rng(policy['seed'])
    wanted = policy['max_pairs'] // 2
    positive, negative, seen = [], [], set()
    if len(eligible) < 2:
        return pd.DataFrame(columns=['a', 'b', 'label', 'split'])
    # Batch proposals by anchor to avoid loading a sparse shard for every pair.
    proposed = 0
    while proposed < policy['proposal_budget'] and (len(positive) < wanted or len(negative) < wanted):
        a = int(rng.choice(eligible))
        js = rng.choice(eligible, min(256, policy['proposal_budget'] - proposed), replace=True)
        row = incidence.row(a)
        joints = dict(zip(row.indices, row.data))
        for b in js:
            proposed += 1
            b = int(b)
            if a == b:
                continue
            pair = tuple(sorted((a, b)))
            if pair in seen:
                continue
            seen.add(pair)
            joint = float(joints.get(b, 0))
            lift = joint * incidence.contexts / (float(incidence.frequencies[a]) * incidence.frequencies[b])
            if joint >= 100 and lift >= 2 and len(positive) < wanted:
                positive.append(pair)
            elif joint == 0 and len(negative) < wanted:
                negative.append(pair)
    # Balance labels before deterministic, unordered-pair-level holdout.
    size = min(len(positive), len(negative))
    records = []
    for label, pairs in ((1, positive[:size]), (0, negative[:size])):
        for a, b in pairs:
            split = 'validation' if se.stable_hash(f'{a}:{b}') % 5 == 0 else 'train'
            records.append(dict(a=a, b=b, label=label, split=split))
    return pd.DataFrame(records, columns=['a', 'b', 'label', 'split'])


def fit_compatibility(vectors, pairs):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    if pairs.empty:
        return None, {'status': 'unavailable', 'reason': 'no eligible supervision'}
    for split in ('train', 'validation'):
        counts = pairs[pairs.split == split].label.value_counts()
        if len(counts) != 2 or counts.min() < 100:
            return None, {'status': 'unavailable', 'reason': 'fewer than 100 examples per class per split'}
    u = se.unit_vectors(vectors)
    x = u[pairs.a.to_numpy()] * u[pairs.b.to_numpy()]
    train = (pairs.split == 'train').to_numpy()
    model = LogisticRegression(C=POLICY['compatibility']['regularization_C'], max_iter=2000,
                               random_state=POLICY['compatibility']['seed'])
    model.fit(x[train], pairs.label.to_numpy()[train])
    score = model.decision_function(x[~train])
    return model.coef_[0].astype(np.float32), {
        'status': 'diagnostic_only', 'train_pairs': int(train.sum()), 'validation_pairs': int((~train).sum()),
        'association_auc': float(roc_auc_score(pairs.label.to_numpy()[~train], score)),
        'association_average_precision': float(average_precision_score(pairs.label.to_numpy()[~train], score)),
        'intercept': float(model.intercept_[0]),
        'note': 'Intercept does not affect ranking. No claim of statistically meaningful mechanical generalization.'}


def save_evaluation(dest, scorer, inputs, evidence, vectors=None, probe=True):
    names, counts = inputs['names'], inputs['counts']
    evaluate_pairs(scorer, names, evidence).to_csv(dest / 'pairs.csv', index=False)
    neighborhoods(scorer, names, counts, inputs['stability_ids']).to_csv(dest / 'neighborhoods.csv', index=False)
    unavailable = []
    if vectors is None:
        unavailable += [dict(task=task, reason='not defined for this score; corresponding card-vector evaluations are reported separately where available')
                        for task in ('linear_probes', 'heldout_centroid')]
    else:
        concepts = {label: [c for c in members if normalize_card_name(c) in inputs['metadata']]
                    for label, members in inputs['concepts'].items()}
        centroids = []
        for split_seed in POLICY['split_seeds']:
            frame, _ = v2.expanded_centroids(vectors, names, concepts, seed=split_seed)
            if not frame.empty:
                frame['recall20'] = (frame['rank'] <= 20).astype(float)
                frame['split_seed'] = split_seed
                centroids.append(frame)
        if centroids:
            pd.concat(centroids).to_csv(dest / 'centroids.csv', index=False)
        else:
            unavailable.append(dict(task='heldout_centroid', reason='insufficient concept coverage'))
        if probe:
            frames = []
            for task in ('color', 'type', 'format', 'mana', 'archetype'):
                if task not in inputs['targets']:
                    unavailable.append(dict(task=task, reason='target unavailable in frozen v2 cache'))
                    continue
                print(f'  probe {task}: {dest.name}', flush=True)
                frame, missing = v2.probe_long(vectors, counts, names, inputs['metadata'], inputs['targets'][task], task)
                if not frame.empty:
                    frames.append(frame)
                unavailable += missing.to_dict('records')
            if frames:
                pd.concat(frames).to_csv(dest / 'probes.csv', index=False)
        else:
            unavailable.append(dict(task='linear_probes', reason='not run in retrieval-only stage; run probes stage'))
    write_json(dest / 'unavailable.json', unavailable)


def evaluate(root, output, family='sgns', probes=False):
    _, source, output = validate(root, output)
    inputs = joblib.load(source / 'evaluation_inputs.joblib')
    evidence = benchmarks(inputs)
    if family in ('baselines', 'compatibility') and not (output / 'matrices/complete.json').exists():
        raise ValueError('Run matrices stage first')
    if family == 'baselines':
        for method in ('raw_incidence', 'incidence_cosine', 'pmi', 'ppmi'):
            dest = output / 'evaluations' / method
            if (dest / 'complete.json').exists():
                continue
            with stage_directory(dest):
                write_json(dest / 'model.json', dict(method=method, dimension=0, seed='deterministic', family=family))
                scorer = IncidenceScorer(output / 'matrices', inputs['counts'], method)
                save_evaluation(dest, scorer, inputs, evidence)
        return
    supervision = None
    if family == 'compatibility':
        pair_path = output / 'compatibility_pairs.csv'
        if pair_path.exists():
            supervision = pd.read_csv(pair_path)
        else:
            incidence = IncidenceScorer(output / 'matrices', inputs['counts'], 'raw_incidence')
            supervision = compatibility_pairs(incidence, inputs['names'], evidence)
            supervision.to_csv(pair_path, index=False)
    for seed in POLICY['seeds']:
        for dimension in POLICY['dimensions']:
            if family == 'svd':
                path = output / f'svd/d{dimension}_s{seed}'
                if not (path / 'complete.json').exists():
                    raise ValueError(f'Incomplete SVD: {path}')
                u, v = np.load(path / 'vectors.npy'), None
                variants = [('ppmi_svd_cosine', GeometryScorer(u, mode='input_cosine'), u),
                            ('ppmi_svd_dot', GeometryScorer(u, mode='svd_dot'), None)]
            else:
                path = source / f'seed_{seed}/{dimension}/sample_0'
                u, v = aligned_model(path / 'model.gensim', inputs['names'])
                if family == 'compatibility':
                    fit_path = output / f'compatibility/d{dimension}_s{seed}'
                    if (fit_path / 'complete.json').exists():
                        fit = read_json(fit_path / 'fit.json')
                        weights = np.load(fit_path / 'weights.npy') if (fit_path / 'weights.npy').exists() else None
                    else:
                        weights, fit = fit_compatibility(u, supervision)
                        with stage_directory(fit_path):
                            write_json(fit_path / 'fit.json', fit)
                            if weights is not None:
                                np.save(fit_path / 'weights.npy', weights)
                    if weights is None:
                        continue
                    variants = [('compatibility_diagonal', GeometryScorer(u, mode='compatibility', weights=weights), None)]
                else:
                    variants = [('input_cosine', GeometryScorer(u), u)]
                    if v is not None:
                        variants += [(mode, GeometryScorer(u, v, mode),
                                      v if mode == 'context_cosine' else u + v if mode == 'combined_cosine' else None)
                                     for mode in ('context_cosine', 'cross_dot', 'cross_cosine', 'symmetric_cross_dot',
                                                  'symmetric_cross_cosine', 'combined_cosine')]
                    else:
                        raise ValueError('Context vectors unavailable: audit requires a separately versioned v3 retraining stage')
            for method, scorer, vectors in variants:
                dest = output / 'evaluations' / f'{method}_d{dimension}_s{seed}'
                if (dest / 'complete.json').exists():
                    continue
                print(f'Evaluate {dest.name}', flush=True)
                with stage_directory(dest):
                    write_json(dest / 'model.json', dict(method=method, dimension=dimension, seed=seed, family=family,
                                                         origin='static_v3' if family == 'svd' else 'frozen_static_v2'))
                    save_evaluation(dest, scorer, inputs, evidence, vectors, probes)


def run_probes(root, output):
    """Separate expensive stage, attach to representation evaluations without rewriting them."""
    _, source, output = validate(root, output)
    inputs = joblib.load(source / 'evaluation_inputs.joblib')
    for model_file in sorted((output / 'evaluations').glob('*/model.json')):
        parent = model_file.parent
        if not (parent / 'complete.json').exists():
            continue
        info = read_json(model_file)
        method, dimension, seed = info['method'], info['dimension'], info['seed']
        if method not in ('input_cosine', 'context_cosine', 'combined_cosine', 'ppmi_svd_cosine'):
            continue
        dest = output / 'probes' / parent.name
        if (dest / 'complete.json').exists() or (parent / 'probes.csv').exists():
            continue
        if method == 'ppmi_svd_cosine':
            vectors = np.load(output / f'svd/d{dimension}_s{seed}/vectors.npy')
        else:
            u, v = aligned_model(source / f'seed_{seed}/{dimension}/sample_0/model.gensim', inputs['names'])
            vectors = u if method == 'input_cosine' else v if method == 'context_cosine' else u + v
        with stage_directory(dest):
            frames, missing = [], []
            for task in ('color', 'type', 'format', 'mana', 'archetype'):
                if task not in inputs['targets']:
                    missing.append(dict(task=task, reason='target unavailable'))
                    continue
                print(f'Probe {parent.name}: {task}', flush=True)
                frame, absent = v2.probe_long(vectors, inputs['counts'], inputs['names'], inputs['metadata'],
                                             inputs['targets'][task], task)
                if not frame.empty:
                    frames.append(frame)
                missing += absent.to_dict('records')
            if frames:
                pd.concat(frames).to_csv(dest / 'probes.csv', index=False)
            write_json(dest / 'unavailable.json', missing)


def aggregate_metrics(per_seed):
    """Seed is the unit; deterministic baselines have no invented uncertainty."""
    from scipy.stats import t
    keys = ['method', 'dimension', 'task', 'metric']
    aggregates, paired = [], []
    for key, group in per_seed.groupby(keys):
        row = dict(zip(keys, key))
        row.update(mean=float(group.value.mean()), replicates=len(group),
                   replicate_kind='fixed_matrix_factorization_seed' if key[0].startswith('ppmi_svd') else
                   'deterministic' if key[1] == 0 else 'training_seed')
        if len(group) > 1:
            row.update(sd=float(group.value.std(ddof=1)), sd_status='available')
        else:
            row.update(sd='not_applicable', sd_status='one deterministic result or incomplete seed set')
        aggregates.append(row)
    for key, group in per_seed[per_seed.dimension > 0].groupby(['method', 'task', 'metric']):
        pivot = group.pivot(index='seed', columns='dimension', values='value')
        if not {128, 256}.issubset(pivot.columns):
            continue
        diff = (pivot[256] - pivot[128]).dropna()
        row = dict(zip(['method', 'task', 'metric'], key))
        row.update(pairs=len(diff))
        if len(diff) < 3:
            row.update(status='unavailable_fewer_than_three_matched_seeds')
        else:
            mean, sd = float(diff.mean()), float(diff.std(ddof=1))
            margin = float(t.ppf(.975, len(diff) - 1) * sd / np.sqrt(len(diff)))
            row.update(status='descriptive_unadjusted_interval', delta_256_minus_128=mean,
                       sd=sd, ci95_low=mean - margin, ci95_high=mean + margin)
        paired.append(row)
    return pd.DataFrame(aggregates, columns=['method', 'dimension', 'task', 'metric', 'mean', 'replicates', 'replicate_kind', 'sd', 'sd_status']), paired


def stability_table(snapshots):
    rows = []
    for (method, dimension), group in snapshots.groupby(['method', 'dimension']):
        if dimension == 0:
            continue  # Fixed matrix has no training-seed variability.
        for seed_a, seed_b in combinations(sorted(group.seed.unique()), 2):
            left = group[group.seed == seed_a].set_index('card')
            right = group[group.seed == seed_b].set_index('card')
            for card in left.index.intersection(right.index):
                a, b = set(json.loads(left.loc[card, 'neighbors'])), set(json.loads(right.loc[card, 'neighbors']))
                if not a or not b:
                    continue
                rows.append(dict(method=method, dimension=dimension, seed_a=seed_a, seed_b=seed_b,
                                 card=card, bucket=left.loc[card, 'bucket'], jaccard=len(a & b) / len(a | b)))
    return pd.DataFrame(rows)


def markdown_table(frame):
    """Small report tables without pandas' optional tabulate dependency."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.5g}'
        return str(value).replace('|', '&#124;').replace('\n', ' ')
    header = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    divider = '| ' + ' | '.join('---' for _ in frame.columns) + ' |'
    return '\n'.join([header, divider] + ['| ' + ' | '.join(map(cell, row)) + ' |'
                                        for row in frame.itertuples(index=False, name=None)])


def build_report(root, output):
    """Regenerable review exports; completed model/evaluation stages stay immutable."""
    _, _, output = validate(root, output)
    rows, pairs, snapshots, unavailable, inventories = [], [], [], [], []
    for model_file in sorted((output / 'evaluations').glob('*/model.json')):
        dest = model_file.parent
        if not (dest / 'complete.json').exists():
            continue
        info = read_json(model_file)
        inventories.append(info)
        tags = {key: info[key] for key in ('method', 'dimension', 'seed')}
        retrieval = pd.read_csv(dest / 'pairs.csv')
        pairs.append(retrieval.assign(**tags))
        for kind, group in retrieval.groupby('kind'):
            for metric in ('rr', 'recall10', 'recall20', 'recall50'):
                rows.append({**tags, 'task': kind, 'metric': metric, 'value': float(group[metric].mean())})
        centroid_file = dest / 'centroids.csv'
        if centroid_file.exists():
            frame = pd.read_csv(centroid_file)
            for metric in ('rr', 'recall10', 'recall20', 'recall50'):
                # Card splits averaged inside each trained model.
                value = frame.groupby('split_seed')[metric].mean().mean()
                rows.append({**tags, 'task': 'heldout_centroid', 'metric': metric, 'value': float(value)})
        probe_file = output / 'probes' / dest.name / 'probes.csv'
        if not (probe_file.parent / 'complete.json').exists():
            probe_file = dest / 'probes.csv'
        if probe_file.exists():
            frame = pd.read_csv(probe_file)
            headline = frame[(frame.baseline == 'learned') & (frame.group == 'all') & (frame.label == 'all')]
            for (task, metric), group in headline.groupby(['task', 'metric']):
                rows.append({**tags, 'task': task, 'metric': metric, 'value': float(group.value.mean())})
        snapshots.append(pd.read_csv(dest / 'neighborhoods.csv').assign(**tags))
        for item in read_json(dest / 'unavailable.json'):
            if item.get('task') == 'linear_probes' and probe_file.exists():
                continue
            unavailable.append({**tags, **item})
        if probe_file.exists() and (probe_file.parent / 'unavailable.json').exists():
            unavailable += [{**tags, **item} for item in read_json(probe_file.parent / 'unavailable.json')]
    per_seed = pd.DataFrame(rows, columns=['method', 'dimension', 'seed', 'task', 'metric', 'value'])
    if len(per_seed) and not np.isfinite(per_seed.value).all():
        raise ValueError('Nonfinite headline metric')
    per_seed.to_csv(output / 'per_seed_metrics.csv', index=False)
    summary, paired = aggregate_metrics(per_seed)
    summary.to_csv(output / 'aggregate_metrics.csv', index=False)
    pd.DataFrame(paired).fillna('not_applicable').to_csv(output / 'paired_dimension_differences.csv', index=False)
    write_json(output / 'unavailable_metrics.json', unavailable)
    write_json(output / 'scorer_inventory.json', inventories)
    if pairs:
        pair_frame = pd.concat(pairs)
        pair_frame.to_csv(output / 'pair_results.csv', index=False)
        pair_frame[pair_frame.kind.isin(['complementary_mechanical', 'mechanical_v2'])].to_csv(output / 'mechanical_pairs.csv', index=False)
        pair_frame[pair_frame.kind == 'semantic_similarity'].to_csv(output / 'semantic_pairs.csv', index=False)
    stability = stability_table(pd.concat(snapshots)) if snapshots else pd.DataFrame()
    if not stability.empty:
        stability.to_csv(output / 'stability_pairs.csv', index=False)
        # Pairwise comparisons reuse models; no inferential SD or CI on these rows.
        stability_summary = stability.groupby(['method', 'dimension', 'bucket']).agg(
            mean_jaccard=('jaccard', 'mean'), anchors=('card', 'nunique'), seed_pairs=('jaccard', 'size')).reset_index()
        stability_summary.rename(columns={'seed_pairs': 'anchor_seed_pair_observations'}, inplace=True)
        stability_summary.to_csv(output / 'stability_by_frequency.csv', index=False)
    coverage = pd.read_csv(output / 'benchmark_coverage.csv')
    coverage_summary = coverage.groupby(['kind', 'status']).agg(
        directed_queries=('anchor', 'size'), unique_pairs=('pair_id', 'nunique')).reset_index()
    coverage_summary.to_csv(output / 'coverage_summary.csv', index=False)
    completed_methods = {f"{i['method']}:{i['dimension']}:{i['seed']}" for i in inventories}
    expected_methods = {f'{method}:{d}:{s}' for method in (
        'input_cosine', 'context_cosine', 'cross_dot', 'cross_cosine', 'symmetric_cross_dot',
        'symmetric_cross_cosine', 'combined_cosine', 'ppmi_svd_cosine', 'ppmi_svd_dot', 'compatibility_diagonal')
        for d in POLICY['dimensions'] for s in POLICY['seeds']}
    expected_methods |= {f'{m}:0:deterministic' for m in ('raw_incidence', 'incidence_cosine', 'pmi', 'ppmi')}
    unavailable_scorers = []
    for fit_path in sorted((output / 'compatibility').glob('*/fit.json')):
        if not (fit_path.parent / 'complete.json').exists():
            continue
        fit = read_json(fit_path)
        if fit['status'] == 'unavailable':
            dimension, seed = fit_path.parent.name.split('_')
            identifier = f'compatibility_diagonal:{int(dimension[1:])}:{int(seed[1:])}'
            unavailable_scorers.append({'identifier': identifier, 'reason': fit['reason']})
    pending = sorted(expected_methods - completed_methods - {r['identifier'] for r in unavailable_scorers})
    missing_probes = []
    for method in ('input_cosine', 'context_cosine', 'combined_cosine', 'ppmi_svd_cosine'):
        for d in POLICY['dimensions']:
            for s in POLICY['seeds']:
                key = f'{method}_d{d}_s{s}'
                if not any((output / folder / key / 'probes.csv').exists() and
                           (output / folder / key / 'complete.json').exists() for folder in ('probes', 'evaluations')):
                    missing_probes.append(key)
    status = ('complete_with_unavailable' if unavailable_scorers else 'complete') if not pending and not missing_probes else 'partial'
    result = {'status': status, 'completed_scorers': len(inventories), 'pending_scorers': pending,
              'pending_probes': missing_probes, 'unavailable_scorers': unavailable_scorers, 'coverage': coverage_summary.to_dict('records'),
              'metrics': summary.to_dict('records'), 'paired_dimension_differences': paired,
              'retraining': 'none; all six frozen v2 models retain context vectors' if
              all(m['context_vectors'] for m in read_json(output / 'model_inventory.json')) else 'required_for_missing_context_vectors'}
    write_json(output / 'summary.json', result)
    lines = ['# Static v3: SGNS geometry, compression and compatibility', '',
             f'Status: **{status}**. {len(inventories)} scorer runs complete; {len(pending)} pending; '
             f'{len(missing_probes)} representation probe runs pending; {len(unavailable_scorers)} scorers unavailable with recorded reasons.', '',
             '## Provenance and scope', '',
             'All training contexts, exclusions, metadata targets and historical models are read from frozen static_v2. '
             'Exact hashes are in provenance.json. No historical result is overwritten. '
             'model_inventory.json records context-vector availability and whether retraining is necessary.', '',
             '## Coverage', '', markdown_table(coverage_summary), '',
             'The three semantic relationships and five requested complementary relationships are tiny curated sets. '
             'Both directions are scored, but they are not independent relationships. mechanical_v2 separately preserves '
             'the original five pairs (including Tainted Pact and Ophidian Eye). Corpus-supported partners were selected '
             'by incidence on this same training corpus; that comparison is transductive and favors incidence by construction.', '',
             '## Results', '']
    if not summary.empty:
        headline = summary[summary.metric.isin(['rr', 'recall50', 'macro_f1', 'mae'])]
        lines += [markdown_table(headline), '',
                  'Full Recall@10/20/50 and MRR appear in aggregate_metrics.csv. Probe per-label support, '
                  'frequency groups and baselines remain in each probes.csv; absent statistics have reasons in unavailable_metrics.json.', '']
    else:
        lines += ['No scorer results yet. Run the explicitly enabled stages; this is an audit, not an experimental conclusion.', '']
    lines += ['## Research questions and interpretation', '',
              '1. **SGNS scoring:** compare input cosine with cross dot/cosine, symmetric cross scores, context cosine '
              'and combined-vector cosine on identical directed queries. A scoring improvement demonstrates exposed '
              'association information, not a new model or a proof of causal mechanism.',
              '2. **Compression/objective:** compare incidence, PMI, PPMI and PPMI-SVD at both ranks. '
              'Full-incidence weighting differs from 32 sampled pairs per context, so gaps cannot uniquely isolate '
              'the SGNS objective. SVD cosine is compared with its own dot-product geometry too. U sqrt(S) '
              'is a semantic embedding; its Gram matrix is not an exact reconstruction of an indefinite PPMI matrix.',
              '3. **Similar role versus complementary mechanics:** use separate semantic_similarity and '
              'complementary_mechanical rows. Inspect pair-level ranks and coverage; do not infer a fundamental '
              'distinction from three versus five relationships. Relationships overlap with the historical benchmark.',
              '4. **Compatibility diagnostic:** a regularized diagonal bilinear scorer learns corpus associations '
              'from frozen vectors. Every benchmark endpoint is excluded from fitting and pair validation. '
              'The embeddings themselves still saw the corpus: this is transductive. Zero-cooccurrence negatives '
              'can include real but unobserved synergies. Balanced case-control AUC/AP are not deployment accuracy. '
              'Insufficient supervision is recorded as unavailable, never as a fabricated result.', '',
              '**Semantic representation**, **static similarity retrieval**, **pairwise compatibility**, and '
              '**format-specific downstream recommendation** are distinct outcomes. Good probes do not establish '
              'combo retrieval; poor cosine combo retrieval does not make embeddings useless or require every combo '
              'to be nearest neighbors. These experiments do not evaluate the later format-specific deck model.', '',
              '## Uncertainty and rare cards', '',
              'Card splits are averaged inside each model before seed means and sample SD. Three SGNS seeds are '
              'a small sample. SVD seeds measure algorithmic factorization variability on one fixed matrix. '
              'Paired 128/256 intervals are descriptive and unadjusted for multiple comparisons. Correlated recall '
              'cutoffs are not independent evidence. Deterministic baselines have no sample SD.', '',
              'Frequency-stratified top-20 Jaccard uses fixed anchors and three pairwise seed comparisons. '
              'Those comparisons share models and are not independent replicates. Deterministic incidence has '
              'no seed-stability experiment. Stable index tie-breaking can inflate stability for tied scores; '
              'pessimistic ties are used for retrieval ranks. No claim about rare-card improvement is made until '
              'the corresponding methods and all seeds finish.', '',
              '## Remaining stages', '',
              f'Pending scorer runs: {len(pending)}. Pending probe runs: {len(missing_probes)}. '
              'See summary.json for identifiers and docs/card2vec-static-v3.md for commands.', '']
    if not stability.empty:
        lines += ['## Descriptive neighborhood stability', '', markdown_table(stability_summary), '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    if not summary.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        selected = summary[summary.task.isin(['semantic_similarity', 'complementary_mechanical', 'mechanical_v2']) & (summary.metric == 'rr')].copy()
        if not selected.empty:
            selected['label'] = selected.method + ' / ' + selected.dimension.astype(str)
            pivot = selected.pivot(index='label', columns='task', values='mean')
            ax = pivot.plot.barh(figsize=(10, max(4, len(pivot) * .38)))
            ax.set_xlabel('Mean reciprocal rank (tiny curated benchmark; descriptive)')
            ax.figure.tight_layout()
            ax.figure.savefig(output / 'retrieval_mrr.png', dpi=140)
            plt.close(ax.figure)
    if not stability.empty:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6))
        order = ['very_rare', 'rare', 'medium', 'common', 'very_common']
        for (method, dimension), group in stability_summary.groupby(['method', 'dimension']):
            series = group.set_index('bucket').mean_jaccard.reindex(order)
            ax.plot(order, series, marker='.', label=f'{method}/{dimension}')
        ax.set_ylabel('Top-20 neighbor Jaccard across seeds (descriptive)')
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(output / 'frequency_stability.png', dpi=140)
        plt.close(fig)
    return result


def render_notebook(root, output):
    """Execute only explicitly tagged v3 review cells and save a NEW archive."""
    import nbformat
    from nbclient import NotebookClient
    root, _, output = validate(root, output)
    archive = output / 'notebook_executed.ipynb'
    if archive.exists():
        raise FileExistsError('Executed archive exists; move it aside explicitly before rendering a newer review')
    original = nbformat.read(root / 'notebooks/card2vec_static_embeddings.ipynb', as_version=4)
    review = nbformat.v4.new_notebook(cells=[c for c in original.cells
                                          if 'static-v3-review' in c.metadata.get('tags', [])])
    if not review.cells:
        raise ValueError('No v3 review cells')
    # Explicit output is injected into the review-only execution namespace.
    config = nbformat.v4.new_code_cell(f'from pathlib import Path\nV3_OUTPUT_OVERRIDE = Path({str(output)!r})')
    review.cells.insert(0, config)
    NotebookClient(review, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(root)}}).execute()
    nbformat.write(review, archive)
    return archive
