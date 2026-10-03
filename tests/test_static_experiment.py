"""Synthetic checks for the targeted experiment; no real training in tests."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mtgdeck import static_embeddings as se
from mtgdeck import static_experiment as experiment


def test_quality_rules_require_structural_evidence():
    assert experiment.quality_reason('Pauper Cube Archetypes',209)=='explicit_cube'
    assert experiment.quality_reason('My collection',150)=='collection_or_pool'
    assert experiment.quality_reason('Merged lists',160)=='aggregate_or_reference'
    assert experiment.quality_reason('Cube-loving commander',100) is None
    assert experiment.quality_reason(None,200) is None
    assert experiment.quality_reason('Normal Commander',200) is None


def test_training_seed_and_epoch_streams_do_not_overlap(tmp_path):
    from dataclasses import replace
    source=tmp_path/'raw.jsonl'
    source.write_text(json.dumps({'cards':list('abcdefghij')})+'\n')
    cfg=se.StaticConfig(min_count=1,pairs_per_context=30)
    prepared=se.prepare_corpus(source,tmp_path/'prepared',cfg)
    assert list(se.PairCorpus(prepared,cfg,0,1)) != list(se.PairCorpus(prepared,replace(cfg,seed=43),0,0))


def test_concepts_allow_overlap_but_exclude_ambiguous_metadata():
    names=['a','b','c']
    metadata={'a':{},'b':{}}
    (ids,y,labels),audit=experiment.concept_targets(names,metadata,{'one':['A','B'],'two':['B','C']})
    assert ids.tolist()==[0,1]
    assert y.tolist()==[[1,0],[1,1]]
    assert audit[audit.card=='C'].iloc[0].status=='ambiguous_or_missing_metadata'


def test_regression_outputs_finite_metrics_and_explains_constant_correlation():
    rng=np.random.default_rng(1)
    vectors=rng.normal(size=(48,8)).astype(np.float32)
    y=np.arange(48)%7
    names=[f'card {i}' for i in range(48)]
    counts=np.arange(48)+10
    rows,missing=experiment.probe_long(vectors,counts,names,{},(np.arange(48),y,['mana_value']),'mana',split_seeds=(101,))
    assert np.isfinite(rows.value).all()
    assert {'mae','median_ae','r2','spearman'} <= set(rows.metric)
    assert not ((rows.baseline=='constant') & (rows.metric=='spearman')).any()
    assert 'constant_target_or_prediction' in set(missing.reason)


def test_bucket_classifier_and_multilabel_probes_have_applicable_rows_only():
    rng=np.random.default_rng(2);vectors=rng.normal(size=(56,8)).astype(np.float32)
    names=[f'card {i}' for i in range(56)];counts=np.arange(56)+100
    metadata={name:{'type_line':'Artifact Creature'} for name in names}
    target=(np.arange(56),np.arange(56)%7,list('0123456'))
    rows,_=experiment.probe_long(vectors,counts,names,metadata,target,'mana_bucket',split_seeds=(101,))
    assert np.isfinite(rows.value).all()
    assert 'mae' not in set(rows.metric)
    y=np.ones((56,2),dtype=int)  # Enough multi-type test examples to clear the support floor.
    rows,_=experiment.probe_long(vectors,counts,names,metadata,(np.arange(56),y,['artifact','creature']),'type',split_seeds=(101,))
    assert np.isfinite(rows.value).all()
    assert 'types:multiple' in set(rows.group)


def test_summary_uses_training_seeds_and_requires_three():
    values=pd.DataFrame([{'dimension':d,'seed':s,'metric':'color_macro_f1','value':v}
                         for d in [128,256] for s,v in zip([42,43,44],[.7,.8,.9])])
    summary=experiment.summarize_seed_values(values)
    assert np.allclose(summary['mean'],.8)
    assert np.allclose(summary['std'],.1)
    with pytest.raises(ValueError,match='three'):
        experiment.summarize_seed_values(values[values.seed!=44])


def test_centroid_candidates_never_include_seed_cards():
    names=[f'card {i}' for i in range(12)]
    heldout,neighbors=experiment.expanded_centroids(np.eye(12),names,{'concept':names},seed=101)
    assert len(heldout)==6
    assert set(neighbors.card)==set(heldout.card)
    assert all(heldout['rank']==6)


def test_final_report_aggregates_finite_metrics_without_nan_tables(tmp_path):
    """Exercise report joins/CI/plots without model training or corpus evaluation."""
    evidence=[]
    for seed in experiment.SEEDS:
        for dimension in experiment.DIMENSIONS:
            out=tmp_path/f'seed_{seed}'/str(dimension)/'sample_0/evaluation_v2'
            out.mkdir(parents=True)
            (out/'complete.json').write_text('{}')
            rows=[]
            for split in [101,102,103]:
                for task in ['color','type','format','archetype','mana','mana_spells','mana_bucket']:
                    metric='mae' if task in ['mana','mana_spells'] else 'macro_f1'
                    for baseline in ['learned','frequency','random','constant']:
                        for group in ['all','frequency:medium']:
                            rows.append(dict(task=task,baseline=baseline,split_seed=split,group=group,
                                             n=20,mean_log_frequency=2.4,metric=metric,label='all',
                                             value=.6+.001*(seed-42)+dimension/100000,support=10))
            pd.DataFrame(rows).to_csv(out/'probes_long.csv',index=False)
            pairs=[]
            for kind in ['mechanical_combo','curated_synergy','corpus_supported']:
                for i in range(3):
                    pairs.append(dict(kind=kind,anchor='a',partner=f'b{i}',status='covered',
                                      anchor_frequency=100+i,partner_frequency=200+i,cooccurrence=20+i,
                                      pmi_smoothed_bits=.5+i,incidence_rank=i+1,rank=i+2,rr=1/(i+2),
                                      recall10=1.,recall25=1.,recall50=1.,frequency_bucket='medium'))
            pd.DataFrame(pairs).to_csv(out/'retrieval.csv',index=False)
            evidence=pairs
            pd.DataFrame([dict(concept='one',card='b',rank=2,rr=.5,recall10=1.,recall25=1.,recall50=1.)]).to_csv(out/'centroids.csv',index=False)
            pd.DataFrame([dict(card='a',frequency=100,bucket='medium',neighbors=json.dumps(['b','c',f'd{seed}']))]).to_csv(out/'neighborhoods.csv',index=False)
            (out/'costs.json').write_text(json.dumps(dict(seed=seed,dimension=dimension,embedding_parameters=dimension*10,
                inference_bytes=dimension*40,sgns_table_bytes=dimension*80,input_bytes_100_cards=dimension*400,
                serialized_bytes=dimension*40+128,training_seconds=10.,retrieval_seconds_per_query=.001)))
    pd.DataFrame(evidence).to_csv(tmp_path/'benchmark_evidence.csv',index=False)
    summary=experiment.build_report(tmp_path,tmp_path)
    assert np.isfinite(summary[['mean','std']]).all().all()
    paired=pd.read_csv(tmp_path/'paired_dimension_differences.csv')
    assert 'neighborhood_jaccard' not in set(paired.metric)
    assert np.isfinite(paired[['ci95_low','ci95_high']]).all().all()
    assert 'no meaningful difference yet' in (tmp_path/'report.md').read_text(encoding='utf-8')
    assert (tmp_path/'frequency_quality.png').exists()


def test_review_renderer_never_executes_compute_cells():
    from scripts.render_static_review import execute_review_cells
    notebook={'cells':[
        {'cell_type':'code','metadata':{'tags':['opt-in-compute']},'source':"raise AssertionError('must not execute')",'outputs':[],'execution_count':None},
        {'cell_type':'code','metadata':{'tags':['second-experiment-review']},'source':"from IPython.display import display\ndisplay({'measured_value': 0.87})",'outputs':[],'execution_count':None},
    ]}
    result=execute_review_cells(notebook)
    assert result['cells'][0]['execution_count'] is None
    assert result['cells'][1]['outputs']
    assert '0.87' in json.dumps(result['cells'][1]['outputs'])


def test_review_renderer_rejects_nan_table_cells():
    from scripts.render_static_review import execute_review_cells
    notebook={'cells':[{'cell_type':'code','metadata':{'tags':['second-experiment-review']},
                       'source':"from IPython.display import display, HTML\ndisplay(HTML('<table><td>NaN</td></table>'))",
                       'outputs':[],'execution_count':None}]}
    with pytest.raises(ValueError,match='NaN table cell'):
        execute_review_cells(notebook)
