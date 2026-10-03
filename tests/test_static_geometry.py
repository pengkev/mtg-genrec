"""V3 numerical identities, leakage boundaries, provenance and stage safety."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from mtgdeck import static_geometry as g


def test_cross_scores_use_output_vectors_and_correct_direction():
    u = np.array([[1., 2.], [3., 4.], [2., -1.]], dtype=np.float32)
    v = np.array([[4., 1.], [1., -2.], [-2., 3.]], dtype=np.float32)
    assert np.allclose(g.GeometryScorer(u, v, 'cross_dot')(0), v @ u[0])
    expected = (v @ u[0] + u @ v[0]) / 2
    assert np.allclose(g.GeometryScorer(u, v, 'symmetric_cross_dot')(0), expected)
    combined = g.se.unit_vectors(u + v)
    assert np.allclose(g.GeometryScorer(u, v, 'combined_cosine')(0), combined @ combined[0])
    normalized_u, normalized_v = g.se.unit_vectors(u), g.se.unit_vectors(v)
    assert np.allclose(g.GeometryScorer(u, v, 'cross_cosine')(0), normalized_v @ normalized_u[0])
    assert not np.allclose(g.GeometryScorer(u)(0), g.GeometryScorer(u, v, 'cross_cosine')(0))


def test_pmi_ppmi_and_absent_pair_ranks(tmp_path):
    counts = sparse.csr_matrix([[0., 1., 0.], [1., 0., 2.], [0., 2., 0.]])
    freq = np.array([1, 3, 2])
    p = g.ppmi_block(counts, 0, freq, 4)
    assert p[0, 1] == pytest.approx(np.log2(4 / 3))
    assert p[0, 2] == 0
    sparse.save_npz(tmp_path / 'counts_0.npz', counts)
    g.write_json(tmp_path / 'index.json', {'contexts': 4, 'shards': [{'start': 0, 'stop': 3}]})
    pmi = g.IncidenceScorer(tmp_path, freq, 'pmi')
    scores = pmi(0)
    assert scores[1] == pytest.approx(np.log2(4 / 3))
    assert scores[2] == -np.inf
    assert g.pessimistic_rank(scores, 0, 2) == 2  # never counts self
    assert g.pessimistic_rank(np.zeros(3), 0, 1) == 2
    negative = sparse.csr_matrix([[0., 1., 0.]])
    assert g.ppmi_block(negative, 0, np.array([3, 3, 1]), 4).nnz == 0


def test_sparse_matrix_stage_counts_contexts_and_removes_diagonal(tmp_path, monkeypatch):
    source, output = tmp_path / 'v2', tmp_path / 'v3'
    prepared = source / 'prepared'
    prepared.mkdir(parents=True)
    (prepared / 'names.json').write_text(json.dumps(['a', 'b', 'c']))
    np.array([0, 1, 1, 2, 0, 1, 2], dtype=np.int32).tofile(prepared / 'tokens.i32')
    np.save(prepared / 'offsets.npy', [0, 2, 4, 7])
    np.save(prepared / 'counts.npy', [2, 3, 2])
    monkeypatch.setattr(g, 'validate', lambda *args: (tmp_path, source, output))
    g.build_matrices(tmp_path, output)
    matrix = sparse.load_npz(output / 'matrices/counts_0.npz').toarray()
    assert np.array_equal(matrix, [[0, 2, 1], [2, 0, 2], [1, 2, 0]])
    with pytest.raises(FileExistsError):
        g.build_matrices(tmp_path, output)
    op = g.ShardedPPMI(output / 'matrices')
    dense = sparse.load_npz(output / 'matrices/ppmi_0.npz').toarray()
    right = np.arange(6).reshape(3, 2).astype(np.float32)
    assert np.allclose(op.matmat(right), dense @ right)
    assert np.allclose(op.matmat(right, transpose=True), dense.T @ right)


def test_randomized_svd_recovers_low_rank_geometry():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(30, 3)).astype(np.float32)
    matrix = x @ x.T
    class Operator:
        shape = matrix.shape
        def matmat(self, right, transpose=False):
            return (matrix.T if transpose else matrix) @ right
    vectors, values = g.randomized_svd(Operator(), 3, 42)
    assert np.allclose(vectors @ vectors.T, matrix, atol=2e-5)
    assert np.allclose(values, np.linalg.svd(matrix, compute_uv=False)[:3], atol=2e-5)


def test_benchmarks_preserve_old_pairs_and_report_missing_metadata():
    names = sorted({g.normalize_card_name(c) for pair in g.se.COMBOS + g.COMPLEMENTARY for c in pair}
                   | {g.normalize_card_name(c) for a, b, _ in g.SEMANTIC for c in (a, b)})
    inputs = dict(names=names, metadata={n: {} for n in names if n != 'chain lightning'},
                  evidence=pd.DataFrame(columns=['kind', 'anchor', 'partner', 'status']))
    evidence = g.benchmarks(inputs)
    assert len(evidence.query("kind == 'mechanical_v2'")) == 10
    assert len(evidence.query("kind == 'complementary_mechanical'")) == 10
    assert len(evidence.query("kind == 'semantic_similarity' and status != 'covered'")) == 2
    assert 'tainted pact' in set(evidence.query("kind == 'mechanical_v2'").partner)


def test_supervision_holds_out_endpoints_and_unordered_pairs():
    names = [f'card{i}' for i in range(40)]
    class Incidence:
        contexts = 10000
        frequencies = np.full(40, 100)
        def row(self, anchor):
            row = np.array([100 if (anchor + j) % 2 == 0 and j != anchor else 0 for j in range(40)])
            return sparse.csr_matrix(row[None, :])
    evidence = pd.DataFrame([{'anchor': names[0], 'partner': names[1]}])
    pairs = g.compatibility_pairs(Incidence(), names, evidence,
                                  {'seed': 202, 'max_pairs': 100, 'proposal_budget': 10000})
    assert not ({0, 1} & (set(pairs.a) | set(pairs.b)))
    assert all(pairs.a < pairs.b)
    assert not pairs.duplicated(['a', 'b']).any()
    assert set(pairs.split) == {'train', 'validation'}
    assert pairs.label.value_counts().nunique() == 1
    weights, fit = g.fit_compatibility(np.eye(40), pairs)
    assert weights is None and fit['status'] == 'unavailable'


def test_paired_stats_are_across_model_seeds():
    rows = [dict(method='input_cosine', dimension=d, seed=s, task='color', metric='macro_f1',
                 value=.5 + (d == 256) * .02 + (s - 42) * .1)
            for d in (128, 256) for s in (42, 43, 44)]
    rows += [dict(method='ppmi', dimension=0, seed='deterministic', task='color', metric='macro_f1', value=.2)]
    summary, paired = g.aggregate_metrics(pd.DataFrame(rows))
    assert summary.query('dimension == 128').iloc[0].sd == pytest.approx(.1)
    assert summary.query('dimension == 0').iloc[0].sd == 'not_applicable'
    assert paired[0]['delta_256_minus_128'] == pytest.approx(.02)
    assert paired[0]['pairs'] == 3


def test_historical_destinations_rejected(tmp_path):
    for path in ('static_v1', 'static_v2', 'static_v2/nested', ''):
        with pytest.raises(ValueError, match='disjoint'):
            g.locations(tmp_path, tmp_path / 'artifacts/card2vec' / path)


def test_partial_stage_not_marked_complete(tmp_path):
    with pytest.raises(RuntimeError):
        with g.stage_directory(tmp_path / 'stage'):
            raise RuntimeError('interrupted')
    assert not (tmp_path / 'stage/complete.json').exists()
    with pytest.raises(FileExistsError):
        with g.stage_directory(tmp_path / 'stage'):
            pass


def test_changed_source_is_rejected(tmp_path, monkeypatch):
    root, source, output = g.locations(tmp_path, tmp_path / 'artifacts/card2vec/static_v3')
    source.mkdir(parents=True)
    output.mkdir(parents=True)
    (source / 'fixture').write_text('old')
    g.write_json(output / 'policy.json', g.POLICY)
    g.write_json(output / 'provenance.json', {'implementation': g.implementation_hashes(),
                                             'source_hashes': {'fixture': g.se.sha256_file(source / 'fixture')}})
    g.validate(root, output)
    (source / 'fixture').write_text('new')
    with pytest.raises(ValueError, match='Frozen source changed'):
        g.validate(root, output)


def test_report_marks_unexecuted_experiments_partial(tmp_path, monkeypatch):
    output = tmp_path / 'v3'
    output.mkdir()
    monkeypatch.setattr(g, 'validate', lambda *a: (tmp_path, tmp_path / 'v2', output))
    g.write_json(output / 'model_inventory.json', [{'context_vectors': True}])
    pd.DataFrame([dict(kind='semantic_similarity', status='covered', anchor='a', pair_id='one')]).to_csv(
        output / 'benchmark_coverage.csv', index=False)
    result = g.build_report(tmp_path, output)
    assert result['status'] == 'partial'
    assert result['completed_scorers'] == 0
    assert len(result['pending_scorers']) == 64
    assert 'No scorer results yet' in (output / 'report.md').read_text()


def test_saved_geometry_to_report_pipeline(tmp_path, monkeypatch):
    import joblib
    source, output = tmp_path / 'v2', tmp_path / 'v3'
    source.mkdir()
    output.mkdir()
    names = sorted({g.normalize_card_name(c) for pair in g.se.COMBOS + g.COMPLEMENTARY for c in pair}
                   | {g.normalize_card_name(c) for a, b, _ in g.SEMANTIC for c in (a, b)})
    inputs = dict(names=names, metadata={n: {} for n in names}, counts=np.arange(len(names)) + 5,
                  evidence=pd.DataFrame([dict(kind='corpus_supported', anchor=names[0], partner=names[1], status='covered')]),
                  stability_ids=np.arange(4), concepts={'toy': names[:12]}, targets={})
    joblib.dump(inputs, source / 'evaluation_inputs.joblib')
    monkeypatch.setattr(g, 'validate', lambda *a: (tmp_path, source, output))
    def aligned(path, vocabulary):
        seed = int(next(p for p in path.parts if p.startswith('seed_')).split('_')[1])
        rng = np.random.default_rng(seed)
        return rng.normal(size=(len(names), 6)), rng.normal(size=(len(names), 6))
    monkeypatch.setattr(g, 'aligned_model', aligned)
    g.write_json(output / 'model_inventory.json', [{'context_vectors': True}])
    g.benchmarks(inputs).to_csv(output / 'benchmark_coverage.csv', index=False)
    g.evaluate(tmp_path, output, 'sgns')
    result = g.build_report(tmp_path, output)
    assert result['completed_scorers'] == 42
    assert result['status'] == 'partial'  # matrix/SVD/probes have not run
    summary = pd.read_csv(output / 'aggregate_metrics.csv')
    assert set(summary.replicates) == {3}
    assert np.isfinite(summary['mean']).all()
    pair_results = pd.read_csv(output / 'mechanical_pairs.csv')
    assert set(pair_results.kind) == {'mechanical_v2', 'complementary_mechanical'}
    assert (output / 'retrieval_mrr.png').exists()
    assert (output / 'frequency_stability.png').exists()
    # Completed evaluations are reused, not rewritten.
    complete = output / 'evaluations/input_cosine_d128_s42/complete.json'
    before = complete.read_bytes()
    g.evaluate(tmp_path, output, 'sgns')
    assert complete.read_bytes() == before


def test_compatibility_fit_learns_symmetric_pair_rule_without_test_pair_fit():
    from itertools import combinations
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(80, 4)).astype(np.float32)
    pairs = pd.DataFrame([
        dict(a=a, b=b, label=int(vectors[a, 0] * vectors[b, 0] > 0),
             split='validation' if g.se.stable_hash(f'{a}:{b}') % 5 == 0 else 'train')
        for a, b in combinations(range(80), 2)])
    weights, fit = g.fit_compatibility(vectors, pairs)
    assert weights.shape == (4,)
    assert fit['association_auc'] > .95
    scorer = g.GeometryScorer(vectors, mode='compatibility', weights=weights)
    assert scorer(3)[7] == pytest.approx(scorer(7)[3])
