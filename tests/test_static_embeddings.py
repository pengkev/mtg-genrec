"""Offline invariants for static experiments; these tests do not train models."""
from dataclasses import replace
import json

import numpy as np
import pytest

from mtgdeck.static_embeddings import (
    PairCorpus, StaticConfig, centroid_retrieval, load_prepared,
    metadata_index, prepare_corpus, recover_provenance, retrieval, usage_targets,
)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return path


def test_preparation_retains_unknown_provenance_and_audits_bad_contexts(tmp_path):
    corpus = write_rows(tmp_path / 'raw.jsonl', [
        {'cards': ['B', 'A', 'A']}, {'cards': ['a', 'b']},
        {'cards': ['a', 'b', 'c', 'd']}, {'cards': 'bad'},
        {'cards': ['C', 'B']},
    ])
    cfg = StaticConfig(min_count=1, max_context_size=3)
    prepared = prepare_corpus(corpus, tmp_path / 'prepared', cfg)
    names, tokens, offsets, counts = load_prepared(prepared)
    assert names == ['a', 'b', 'c']
    assert counts.tolist() == [1, 2, 1]
    assert offsets.tolist() == [0, 2, 4]
    assert tokens.tolist() == [0, 1, 1, 2]
    audit = (prepared / 'context_audit.csv').read_text()
    assert 'duplicate' in audit and 'size_quarantine' in audit and 'malformed' in audit
    with pytest.raises(FileExistsError):
        prepare_corpus(corpus, prepared, cfg)


def test_pair_budget_order_invariance_and_no_self_pairs(tmp_path):
    rows = [{'cards': ['a', 'b']}, {'cards': list('abcdefghij')}]
    config = StaticConfig(min_count=1, pairs_per_context=9)
    a = prepare_corpus(write_rows(tmp_path/'a.jsonl', rows), tmp_path/'a', config)
    reversed_rows = [{'cards': row['cards'][::-1]} for row in rows]
    b = prepare_corpus(write_rows(tmp_path/'b.jsonl', reversed_rows), tmp_path/'b', config)
    first = list(PairCorpus(a, config, 0, 0))
    assert len(first) == 18  # Same bounded budget for 2-card and 10-card sets.
    assert all(left != right for left, right in first)
    assert first == list(PairCorpus(b, config, 0, 0))
    assert first == list(PairCorpus(a, config, 0, 0))
    assert first != list(PairCorpus(a, config, 0, 1))
    assert len(list(PairCorpus(a, config, 1e-3, 0))) <= len(first)
    small = replace(config, pair_strategy='sqrt')
    assert len(list(PairCorpus(a, small, 0))) == 1 + 3


def test_min_count_does_not_leave_never_trainable_vectors(tmp_path):
    rows = [{'cards':cards} for cards in [
        ['a','b','x'], ['a','b','y'], ['c','u'], ['c','v'],
    ]]
    prepared = prepare_corpus(write_rows(tmp_path/'raw.jsonl',rows), tmp_path/'p',
                              StaticConfig(min_count=2))
    names, tokens, offsets, counts = load_prepared(prepared)
    assert names == ['a','b']
    assert counts.tolist() == [2,2]
    assert offsets.tolist() == [0,2,4]
    assert tokens.tolist() == [0,1,0,1]


def test_provenance_requires_exact_matches_and_deduplicates_labels(tmp_path):
    config = StaticConfig(min_count=1)
    prepared = prepare_corpus(write_rows(tmp_path/'corpus.jsonl', [
        {'cards':['a','b']}, {'cards':['b','c']}, {'cards':['c','d']}
    ]), tmp_path/'prepared', config)
    record = {'format':'modern', 'source':'moxfield', 'mainboard':[{'name':'a'},{'name':'b'}]}
    rows = [record, record, {**record, 'format':'legacy', 'source':'mtgtop8'},
            {'mainboard':[{'name':'b'},{'name':'c'}]},
            {**record, 'mainboard':[{'name':'a'}]}]
    provenance = recover_provenance([write_rows(tmp_path/'formats.jsonl',rows)], prepared)
    assert provenance['formats'] == {'modern':{0}, 'legacy':{0}}
    assert provenance['sources'] == {'moxfield':{0}, 'mtgtop8':{0}}
    assert provenance['matched'] == 2
    ids, labels, formats = usage_targets(prepared, provenance, min_contexts=1, min_observations=1)
    assert ids.tolist() == [0, 1]  # Unknown-format cards aren't treated as negative examples.
    assert formats == ['modern','legacy']


def test_metadata_ambiguous_aliases_are_not_arbitrarily_labelled(tmp_path):
    path = tmp_path/'oracle.json'
    path.write_text(json.dumps([
        {'name':'Thing // Back', 'oracle_id':'one'},
        {'name':'A-Thing // Back', 'oracle_id':'two'},
        {'name':'Other', 'oracle_id':'three'},
    ]))
    metadata, collisions = metadata_index(path)
    assert set(metadata) == {'other'}
    assert 'thing' in collisions


def test_missing_retrieval_partners_count_as_failures():
    names = ["thassa's oracle", 'demonic consultation', 'other']
    vectors = np.array([[1,0],[1,.1],[0,1]], dtype=np.float32)
    table = retrieval(vectors,names)
    partner = table[(table.anchor=="Thassa's Oracle") & (table.partner=='Demonic Consultation')].iloc[0]
    assert partner['rank'] == 1 and partner['recall@10'] == 1
    missing = table[~table.covered]
    assert (missing['rr']==0).all() and (missing['recall@50']==0).all()


def test_centroids_exclude_seeds_and_only_evaluate_heldout_cards():
    from mtgdeck.static_embeddings import CONCEPTS
    from mtgdeck.data import normalize_card_name
    names = [normalize_card_name(n) for n in CONCEPTS['Burn']]
    vectors = np.eye(len(names), dtype=np.float32)
    heldout, nearby = centroid_retrieval(vectors,names)
    evaluated = set(heldout.loc[heldout.status=='held-out','card'])
    retrieved = set(nearby.card)
    assert len(evaluated) == 4
    assert retrieved == evaluated  # Only non-seed vocabulary cards may be returned.


def test_mana_audit_excludes_outliers_without_clipping_or_affecting_other_targets():
    from mtgdeck.static_embeddings import mana_value_audit, metadata_targets
    names = ['ordinary','gleemax','zero','fractional','missing','nan','inf','negative','bad','boolean','edge']
    values = [3,1_000_000,0,0.5,None,float('nan'),float('inf'),-1,'bad',True,20]
    metadata = {n:{'cmc':v,'color_identity':[],'type_line':'Artifact'} for n,v in zip(names,values)}
    audit = {r['card']:r for r in mana_value_audit(names,metadata)}
    assert audit['gleemax']['status'] == 'above_primary_range'
    assert audit['gleemax']['mana_value'] == 1_000_000
    assert audit['negative']['status'] == 'negative'
    for name in ['missing','nan','inf','bad','boolean']:
        assert audit[name]['status'] == 'missing_or_invalid'
    targets = metadata_targets(names,metadata)
    assert targets['mana'][0].tolist() == [0,2,3,10]
    assert targets['mana'][1].tolist() == [3,0,0.5,20]
    assert len(targets['color'][0]) == len(names)
    assert metadata['gleemax']['cmc'] == 1_000_000
    with pytest.raises(ValueError):
        mana_value_audit(names,metadata,float('inf'))


def test_reviewed_quarantine_is_exact_and_optional(tmp_path):
    from mtgdeck.data import cards_fingerprint
    rows = [{'cards':['a','b']},{'cards':['a','c']}]
    source = write_rows(tmp_path/'raw.jsonl',rows)
    exclusion = {cards_fingerprint(['a','b']):'Confirmed cube, reviewed manually'}
    cfg = StaticConfig(min_count=1)
    original = prepare_corpus(source,tmp_path/'original',cfg)
    filtered = prepare_corpus(source,tmp_path/'filtered',cfg,exclusion)
    assert len(load_prepared(original)[2]) == 3
    assert len(load_prepared(filtered)[2]) == 2
    assert 'reviewed_non_deck' in (filtered/'context_audit.csv').read_text()
    records = write_rows(tmp_path/'decks.jsonl',[
        {'format':'modern','source':'moxfield','mainboard':[{'name':n} for n in cards]}
        for cards in [['a','b'],['a','c']]])
    provenance = recover_provenance([records],original,exclusion)
    assert provenance['formats']['modern'] == {1}
    assert len(load_prepared(original)[2]) == 3  # Re-evaluation never edits training data.


def test_source_exclusive_analysis_removes_shared_contexts(tmp_path):
    from mtgdeck.static_embeddings import source_associations
    rows = [{'cards':['lava spike','mountain',f'unique {i}']} for i in range(25)]
    prepared = prepare_corpus(write_rows(tmp_path/'raw.jsonl',rows),tmp_path/'p',StaticConfig(min_count=1))
    provenance = {'sources':{'one':set(range(15)),'two':set(range(10,25))}}
    overlapping = source_associations(prepared,provenance,min_contexts=1)
    exclusive = source_associations(prepared,provenance,min_contexts=1,exclusive=True)
    assert overlapping.iloc[0].overlapping_contexts == 5
    assert exclusive.iloc[0].overlapping_contexts == 0
    assert exclusive.iloc[0].contexts_a == 10 and exclusive.iloc[0].contexts_b == 10
    assert len(provenance['sources']['one']) == 15


def test_saved_run_reuse_validates_inputs_and_resolves_paths(tmp_path):
    from dataclasses import asdict
    from mtgdeck.static_embeddings import load_saved_runs, sha256_file
    cfg = StaticConfig(dimensions=(2,),subsampling=(0.0,),min_count=1)
    preparation = tmp_path/'prepared';preparation.mkdir()
    prep = {'sha256':'fixture','config':asdict(cfg)}
    (preparation/'preparation.json').write_text(json.dumps(prep))
    (preparation/'names.json').write_text(json.dumps(['a','b']))
    metadata = tmp_path/'metadata.json';metadata.write_text('[]')
    run = tmp_path/'2/sample_0';run.mkdir(parents=True)
    manifest = {'config':asdict(cfg),'preparation':prep,'metadata_sha256':sha256_file(metadata),
                'dimension':2,'subsampling':0.0}
    (run/'config.json').write_text(json.dumps(manifest))
    (run/'stats.json').write_text(json.dumps({'path':'Z:/other-machine/model','training_seconds':1}))
    np.save(run/'vectors.npy',np.eye(2,dtype=np.float32))
    loaded = load_saved_runs(tmp_path,cfg,metadata)
    assert loaded[0]['path'] == str(run)
    with pytest.raises(ValueError,match='mismatch'):
        load_saved_runs(tmp_path,replace(cfg,epochs=7),metadata)
    np.save(run/'vectors.npy',np.zeros((3,2)))
    with pytest.raises(ValueError,match='shape mismatch'):
        load_saved_runs(tmp_path,cfg,metadata)


def test_revised_notebook_parses_and_defaults_to_reuse():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root/'notebooks/card2vec_static_embeddings.ipynb').read_text(encoding='utf-8'))
    sources = []
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            source = ''.join(cell['source'])
            ast.parse(source)
            sources.append(source)
    assert 'RUN_SECOND_TRAINING = False' in '\n'.join(sources)
    assert 'RUN_SECOND_EVALUATION = False' in '\n'.join(sources)
