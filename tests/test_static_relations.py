"""Independent metric examples, coverage, frozen splitting, and saved-run integration."""
import json
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from mtgdeck import static_relations as r


def test_multi_target_metrics_and_grades():
    # Query removed; candidates b,c,d,e; targets c,e at ranks 2,4.
    scores = np.array([100, 4, 3, 2, 1.])
    result, order = r.rank_metrics(scores, [0], [2, 4], list('abcde'), grades=[2, 1], ks=(2, 4))
    assert order.tolist() == [1, 2, 3, 4]
    assert result['mrr'] == .5
    assert result['mean_target_rr'] == .375
    assert result['recall@2'] == .5
    assert result['precision@2'] == .5
    assert result['ap@4'] == .5
    assert result['median_target_rank'] == 3
    assert result['ndcg@2'] == pytest.approx((3 / np.log2(3)) / (3 + 1 / np.log2(3)))


def test_ties_missing_pmi_and_empty_candidates():
    result, order = r.rank_metrics([0, 0, 0, -np.inf], [0], [3], ['query', 'z', 'a', 'b'], ks=(10,))
    assert order.tolist() == [2, 1]
    assert result['mrr'] == result['ap@10'] == result['recall@10'] == 0
    assert result['median_target_rank'] == 3
    with pytest.raises(ValueError):
        r.rank_metrics([0, 1], [0], [0], ['a', 'b'])
    with pytest.raises(ValueError):
        r.rank_metrics([0, np.nan], [0], [1], ['a', 'b'])


def fixture_inputs():
    names = [f'card {i:04}' for i in range(80)]
    metadata = {n: dict(cmc=1, type_line='Creature — Elf', color_identity=['G']) for n in names}
    return dict(names=names, counts=np.full(80, 100), metadata=metadata,
                concepts={'concept': names[:12]}, targets={},
                evidence=pd.DataFrame([dict(kind='corpus_supported', anchor=names[0], partner=names[1])]))


def fixture_config(inputs):
    names = inputs['names']
    return dict(roles={'role': names[:4]}, slots={'slot': names[4:7]},
                complements=[[names[10], [names[11], 'missing card']]], cross_format=[])


def test_benchmark_reproducibility_provenance_and_strict_coverage():
    inputs = fixture_inputs()
    config = fixture_config(inputs)
    rows = r.build_benchmark(inputs, config)
    assert rows == r.build_benchmark(inputs, config)
    assert len({x['id'] for x in rows}) == len(rows)
    assert sum(x['relation'] == 'archetype' for x in rows) == 24
    assert sum(x['relation'] == 'structural_proxy' for x in rows) == 80
    assert all(not set(x['query']) & set(x['targets']) for x in rows)
    missing = next(x for x in rows if 'missing card' in x['targets'])
    assert missing['status'] == 'excluded'
    assert len(missing['targets']) == 2  # not silently truncated
    inputs['metadata'].pop(inputs['names'][1])
    updated = r.build_benchmark(inputs, config)
    assert any(not x['metadata_available'] and x['status'] == 'covered' for x in updated if x['curated'])
    assert all(x['metadata_available'] for x in updated if x['relation'] == 'structural_proxy')


def test_reverse_cross_and_normalized_sum():
    u = np.array([[1., 2], [3, 4], [-1, 2]])
    v = np.array([[4., 1], [-2, 3], [2, -1]])
    assert np.allclose(r.make_scorer(u, v, 'reverse_cross_dot')(0), u @ v[0])
    z = r.g.se.unit_vectors(r.g.se.unit_vectors(u) + r.g.se.unit_vectors(v))
    assert np.allclose(r.make_scorer(u, v, 'normalized_sum_cosine')(1), z @ z[1])


def test_endpoint_disjoint_balanced_hard_pairs():
    inputs = fixture_inputs()
    class Incidence:
        frequencies = inputs['counts']
        contexts = 10000
        def row(self, anchor):
            return sparse.csr_matrix([[100 if (anchor + j) % 3 == 0 and j != anchor else 0 for j in range(80)]])
    records = [dict(curated=True, query=inputs['names'][:2], targets=inputs['names'][2:4])]
    pairs, diag = r.compatibility_pairs(Incidence(), inputs, records, max_pairs=150)
    assert set(pairs.split) == {'train', 'validation', 'test'}
    endpoints = {s: set(g.a) | set(g.b) for s, g in pairs.groupby('split')}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not endpoints[a] & endpoints[b]
    assert not set(range(4)) & set.union(*endpoints.values())
    assert not pairs.duplicated(['a', 'b']).any()
    for split, group in pairs.groupby('split'):
        assert group.label.value_counts().nunique() == 1
    for a, b in pairs[pairs.label == 0][['a', 'b']].itertuples(index=False):
        assert Incidence().row(a)[0, b] == 0
    assert not diag.empty


def test_compatibility_test_is_separate_and_low_capacity():
    rng = np.random.default_rng(9)
    u = rng.normal(size=(90, 4))
    records = []
    for split, start in [('train', 0), ('validation', 30), ('test', 60)]:
        for a in range(start, start + 30):
            for b in range(a + 1, start + 30):
                records.append(dict(a=a, b=b, split=split, label=int(u[a, 0] * u[b, 0] > 0)))
    pairs = pd.DataFrame(records)
    weights, fit = r.fit_compatibility(u, u, pairs, lambda a: np.maximum(0, u @ u[a]))
    assert weights.shape == (4,)
    assert fit['parameter_count'] == 5
    assert next(x['auc'] for x in fit['metrics'] if x['split'] == 'test' and x['scorer'] == 'learned_diagonal') > .95
    flipped = pairs.copy()
    flipped.loc[flipped.split == 'test', 'label'] = 1 - flipped.loc[flipped.split == 'test', 'label']
    other, _ = r.fit_compatibility(u, u, flipped, lambda a: np.maximum(0, u @ u[a]))
    assert np.array_equal(weights, other)


def test_saved_reports_are_append_only_and_relation_specific(tmp_path, monkeypatch):
    inputs = fixture_inputs()
    records = r.build_benchmark(inputs, fixture_config(inputs))
    r.g.write_json(tmp_path / 'benchmark/queries.json', records)
    monkeypatch.setattr(r, 'validate', lambda *a: (tmp_path, tmp_path, tmp_path))
    rng = np.random.default_rng(1)
    for seed in (42, 43):
        u = rng.normal(size=(80, 4))
        r.save_run(tmp_path / f'relations/input_cosine_d128_s{seed}', r.g.GeometryScorer(u), records, inputs,
                   dict(scorer='input_cosine', dimension=128, seed=seed))
    path = r.report(tmp_path, tmp_path)
    assert (path / 'complete.json').exists()
    status = json.loads((path / 'status.json').read_text())
    assert status['status'] == 'partial'
    summary = pd.read_csv(path / 'summary.csv')
    assert set(summary['count']) == {2}
    assert 'substitutability' in set(summary.relation)
    assert (path / 'stability_summary.csv').exists()
    before = (path / 'complete.json').read_bytes()
    assert r.report(tmp_path, tmp_path) == path
    assert before == (path / 'complete.json').read_bytes()
