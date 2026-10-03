"""Targeted second Card2Vec experiment, built on the original set-SGNS utilities."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
import json
import re
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from . import static_embeddings as se
from .data import cards_fingerprint, decklist_card_names, iter_jsonl, normalize_card_name
from .metadata import default_oracle_path

SEEDS = (42, 43, 44)
DIMENSIONS = (128, 256)
FORMATS = {'cedh','commander','legacy','modern','pioneer','standard','vintage','pauper',
           'duel-commander','historic','premodern','historic-brawl'}
SOURCES = {'moxfield','deckbox','mtgtop8'}


def experiment_config(seed=42):
    return se.StaticConfig(dimensions=DIMENSIONS, subsampling=(0.0,), seed=seed,
                           split_seeds=(101, 102, 103))


def quality_reason(title, size):
    """High-precision title+size evidence; keywords alone never reject a row."""
    if size < 120:
        return None
    title = str(title or '').casefold()
    if re.search(r'\bcube\b', title):
        return 'explicit_cube'
    if re.search(r'\b(collection|card pool|cardpool|inventory)\b', title):
        return 'collection_or_pool'
    if re.search(r'\b(archetype reference|reference list|all cards|merged lists?)\b', title):
        return 'aggregate_or_reference'
    return None


def prepare_second_experiment(root, output):
    root, output = Path(root), Path(output)
    prepared = output/'prepared'
    if (prepared/'preparation.json').exists():
        return prepared
    output.mkdir(parents=True, exist_ok=True)
    reviewed = json.loads((root/'configs/card2vec_reviewed_exclusions.json').read_text())
    exclusions = dict(reviewed)
    candidates = []
    source_manifest = []
    for path in sorted((root/'data/format_corpora').glob('*.jsonl')):
        if path.name.startswith('limited'):
            continue
        print(f'Quality inspection: {path.name}', flush=True)
        before = path.stat()
        for record in iter_jsonl(path):
            title = record.get('name', '')
            if not quality_reason(title, 120):
                continue
            names = sorted({normalize_card_name(n) for n in decklist_card_names(record) if normalize_card_name(n)})
            reason = quality_reason(title, len(names))
            if reason:
                fp = cards_fingerprint(names)
                exclusions[fp] = reason
                candidates.append(dict(fingerprint=fp, reason=reason, title=title, size=len(names),
                                       source=record.get('source','unknown'), url=record.get('url','')))
        after = path.stat()
        source_manifest.append({'path':str(path),'size_start':before.st_size,'size_end':after.st_size,
                                'changed_while_reading':before.st_mtime_ns != after.st_mtime_ns})
    pd.DataFrame(candidates, columns=['fingerprint','reason','title','size','source','url']).drop_duplicates().to_csv(output/'quality_candidates.csv',index=False)
    (output/'quality_source_manifest.json').write_text(json.dumps(source_manifest,indent=2))
    (output/'quality_exclusions.json').write_text(json.dumps(exclusions,indent=2))
    print(f'Preparing corpus; {len(exclusions)} candidate set exclusions, including reviewed cases.',flush=True)
    se.prepare_corpus(root/'data/cooccurence/embedding_corpus.jsonl', prepared, experiment_config(), exclusions)
    audit = pd.read_csv(prepared/'context_audit.csv')
    rejected = audit[audit.status!='accepted'].copy()
    rejected['reason'] = rejected.apply(lambda r: exclusions.get(r.fingerprint,r.status) if r.status=='reviewed_non_deck' else r.status,axis=1)
    rejected = rejected.merge(pd.DataFrame(candidates).drop_duplicates('fingerprint') if candidates else pd.DataFrame(columns=['fingerprint']),on='fingerprint',how='left',suffixes=('','_candidate'))
    rejected.to_csv(output/'quality_rejections.csv',index=False)
    audit.status.value_counts().rename_axis('status').reset_index(name='contexts').to_csv(output/'quality_counts.csv',index=False)
    pd.DataFrame({'before':audit['size'].describe(percentiles=[.25,.5,.75,.9,.99]),
                  'after':audit.loc[audit.status=='accepted','size'].describe(percentiles=[.25,.5,.75,.9,.99])}).to_csv(output/'quality_size_distribution.csv')
    return prepared


def train_one(root, output, seed, dimension):
    output, root = Path(output), Path(root)
    model_root = output/f'seed_{seed}'
    path = model_root/str(dimension)/'sample_0'
    config = experiment_config(seed)
    if (path/'stats.json').exists():
        manifest = json.loads((path/'config.json').read_text())
        preparation = json.loads((output/'prepared/preparation.json').read_text())
        if (manifest['config'] != json.loads(json.dumps(asdict(config)))
                or manifest['preparation'] != preparation
                or manifest['implementation_sha256'] != se.sha256_file(se.__file__)):
            raise ValueError('Refusing to reuse a mismatched training config, corpus or implementation')
        stats = json.loads((path/'stats.json').read_text())
    else:
        print(f'Train d={dimension}, seed={seed}',flush=True)
        stats = se.train_run(output/'prepared',model_root,config,dimension,0.0,
                             default_oracle_path(root/'data'),progress=True)
    return {**stats,'seed':seed,'path':str(path)}


def train_experiment(root, output, jobs=2):
    prepared = prepare_second_experiment(root,output)
    output = Path(output)
    config = {'config':asdict(experiment_config()),'training_seeds':SEEDS,
              'reason_for_three_seeds':'Six new runs budgeted from first-run timings; no high-dimension rerun.',
              'subsampling_policy':'Fixed at 0 before comparison; first sweep found no consistent benefit.',
              'module_sha256':se.sha256_file(__file__)}
    (output/'experiment.json').write_text(json.dumps(config,indent=2))
    with parallel_config(backend='loky',inner_max_num_threads=1):
        runs = Parallel(n_jobs=jobs,verbose=10)(delayed(train_one)(root,output,seed,dimension)
                    for seed in SEEDS for dimension in DIMENSIONS)
    pd.DataFrame(runs).to_csv(output/'runs.csv',index=False)
    print('Targeted training complete.',flush=True)
    return runs


def concept_targets(names, metadata, concepts):
    lookup = {name:i for i,name in enumerate(names)}
    membership = {}
    audit = []
    for label, members in concepts.items():
        for member in members:
            name = normalize_card_name(member)
            status = 'included' if name in lookup and name in metadata else 'missing_embedding' if name not in lookup else 'ambiguous_or_missing_metadata'
            audit.append(dict(concept=label,card=member,key=name,status=status))
            if status=='included':
                membership.setdefault(lookup[name],set()).add(label)
    ids = np.array(sorted(membership),dtype=int)
    labels = list(concepts)
    y = np.array([[int(label in membership[i]) for label in labels] for i in ids])
    return (ids,y,labels), pd.DataFrame(audit)


def build_targets(prepared, metadata, concepts, provenance):
    names, _, _, counts = se.load_prepared(prepared)
    targets = se.metadata_targets(names,metadata,mana_maximum=20)
    ids, values, _ = targets['mana']
    # Half-cost unusual values use explicit intervals [0,1), [1,2), ... [6,inf).
    targets['mana_bucket'] = (ids,np.minimum(np.floor(values),6).astype(int),['0','1','2','3','4','5','6+'])
    spell_mask = np.array(['land' not in metadata[names[i]].get('type_line','').lower() for i in ids])
    targets['mana_spells'] = (ids[spell_mask],values[spell_mask],['mana_value'])
    targets['archetype'], concept_audit = concept_targets(names,metadata,concepts)
    format_target = se.usage_targets(prepared,provenance)
    if format_target is not None:
        ids,y,labels = format_target
        keep = np.array([names[i] in metadata for i in ids],dtype=bool)
        targets['format'] = ids[keep],y[keep],labels
    return targets, concept_audit


def probe_long(vectors, counts, names, metadata, target, task, split_seeds=(101,102,103)):
    """Tidy finite results: one applicable metric/value per row, no NaN columns."""
    from scipy.stats import spearmanr
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import f1_score, mean_absolute_error, median_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ids,y,labels = target
    regression = task in {'mana','mana_spells'}
    multiclass = task=='mana_bucket'
    rows, unavailable = [], []
    if len(ids)<20:
        return pd.DataFrame(), pd.DataFrame([dict(task=task,reason='fewer_than_20_labelled_cards')])
    for split_seed in split_seeds:
        strata = None
        if multiclass:
            unique, group_counts = np.unique(y,return_counts=True)
            if group_counts.min()>=2:
                strata=y
        tr,te = train_test_split(np.arange(len(ids)),test_size=.25,random_state=split_seed,stratify=strata)
        if task=='archetype' and ((y[te].sum(axis=0)<2).any() or (y[tr].sum(axis=0)<5).any()):
            raise ValueError('Curated split lacks meaningful class support; expand/review labels before scoring')
        # Identical random features and card splits for all model seeds; dimensions are nested.
        random_x = np.random.default_rng(split_seed).normal(size=(len(ids),256)).astype(np.float32)[:,:vectors.shape[1]]
        features = {'learned':vectors[ids], 'frequency':np.log1p(counts[ids,None]),
                    'random':random_x,'constant':np.zeros((len(ids),1))}
        test_ids=ids[te]
        freq_groups=se.frequency_buckets(counts[test_ids])
        masks={'all':np.ones(len(te),dtype=bool)}
        masks.update({f'frequency:{b}':freq_groups==b for b in ['very_rare','rare','medium','common','very_common']})
        if task in {'type','color'}:
            type_lines=[metadata[names[i]].get('type_line','').lower() for i in test_ids]
            for kind in ['creature','artifact','land','instant','sorcery','enchantment','planeswalker']:
                masks[f'type:{kind}']=np.array([kind in line for line in type_lines])
            if task=='type':
                masks['types:single']=y[te].sum(axis=1)==1
                masks['types:multiple']=y[te].sum(axis=1)>1
        for group,mask in masks.items():
            if mask.sum()<5:
                unavailable.append(dict(task=task,split_seed=split_seed,group=group,
                                        n=int(mask.sum()),reason='fewer_than_five_test_cards'))
        for baseline,x in features.items():
            if regression:
                model=DummyRegressor(strategy='mean') if baseline=='constant' else make_pipeline(StandardScaler(),Ridge(alpha=10))
                model.fit(x[tr],y[tr]); predictions=model.predict(x[te])
            elif multiclass:
                model=DummyClassifier(strategy='most_frequent') if baseline=='constant' else make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,class_weight='balanced'))
                model.fit(x[tr],y[tr]);predictions=model.predict(x[te])
            else:
                predictions=[]
                for col in range(y.shape[1]):
                    model=DummyClassifier(strategy='most_frequent') if baseline=='constant' or len(np.unique(y[tr,col]))<2 else make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=2000,class_weight='balanced'))
                    model.fit(x[tr],y[tr,col]);predictions.append(model.predict(x[te]))
                predictions=np.array(predictions).T
            for group,mask in masks.items():
                if mask.sum()<5:
                    continue
                truth,pred=y[te][mask],predictions[mask]
                base=dict(task=task,baseline=baseline,split_seed=split_seed,group=group,n=int(mask.sum()),
                          mean_log_frequency=float(np.log10(counts[test_ids[mask]]+1).mean()))
                def emit(metric,value,label='all',support=None):
                    if np.isfinite(value):
                        rows.append({**base,'metric':metric,'label':label,'value':float(value),
                                     'support':int(support if support is not None else mask.sum())})
                    else:
                        unavailable.append({**base,'metric':metric,'label':label,'reason':'undefined_constant_target_or_prediction'})
                if regression:
                    emit('mae',mean_absolute_error(truth,pred))
                    emit('median_ae',median_absolute_error(truth,pred))
                    emit('r2',r2_score(truth,pred))
                    if np.ptp(truth)>0 and np.ptp(pred)>0:
                        emit('spearman',spearmanr(truth,pred).statistic)
                    else:
                        unavailable.append({**base,'metric':'spearman','reason':'constant_target_or_prediction'})
                else:
                    emit('macro_f1',f1_score(truth,pred,average='macro',zero_division=0))
                    emit('micro_f1',f1_score(truth,pred,average='micro',zero_division=0))
                    per_label = f1_score(truth,pred,labels=list(range(len(labels))) if multiclass else None,average=None,zero_division=0)
                    supports = np.bincount(truth,minlength=len(labels)) if multiclass else truth.sum(axis=0)
                    for label,score,support in zip(labels,per_label,supports):
                        if support:
                            emit('label_f1',score,label,int(support))
    return pd.DataFrame(rows),pd.DataFrame(unavailable)


def benchmark_evidence(prepared, concepts):
    names,tokens,offsets,counts=se.load_prepared(prepared)
    lookup={n:i for i,n in enumerate(names)}
    mechanical=[('mechanical_combo',a,b) for a,b in se.COMBOS for a,b in ((a,b),(b,a))]
    curated=[('curated_synergy',members[0],b) for members in concepts.values() for b in members[1:]]
    anchor_names=sorted({normalize_card_name(a) for _,a,_ in mechanical+curated if normalize_card_name(a) in lookup})
    anchor_ids=[lookup[a] for a in anchor_names]
    anchor_rows={index:row for row,index in enumerate(anchor_ids)}
    cooccur=np.zeros((len(anchor_ids),len(names)),dtype=np.int32)
    for row in range(len(offsets)-1):
        ids=tokens[offsets[row]:offsets[row+1]]
        for anchor in set(ids).intersection(anchor_rows):
            cooccur[anchor_rows[anchor],ids]+=1
    supported=[]
    incidence_cosines=cooccur/np.sqrt(counts[anchor_ids,None]*counts[None,:])
    for row,anchor in enumerate(anchor_names):
        lift=cooccur[row]*(len(offsets)-1)/(counts[anchor_ids[row]]*counts)
        eligible=(cooccur[row]>=100)&(lift>=2)
        eligible[anchor_ids[row]]=False
        candidates=np.flatnonzero(eligible)
        chosen=candidates[np.argsort(-incidence_cosines[row,candidates],kind='stable')[:5]]
        supported += [('corpus_supported',anchor,names[j]) for j in chosen]
    rows=[]
    for kind,anchor,partner in mechanical+curated+supported:
        a,b=normalize_card_name(anchor),normalize_card_name(partner)
        ia,ib=lookup.get(a),lookup.get(b)
        if ia is None or ib is None:
            rows.append(dict(kind=kind,anchor=a,partner=b,status='missing_vocabulary',anchor_frequency=int(counts[ia]) if ia is not None else 0,partner_frequency=int(counts[ib]) if ib is not None else 0))
            continue
        row=anchor_rows[ia]; joint=int(cooccur[row,ib]); baseline=incidence_cosines[row].copy();baseline[ia]=-np.inf
        rows.append(dict(kind=kind,anchor=a,partner=b,status='covered',anchor_frequency=int(counts[ia]),partner_frequency=int(counts[ib]),
                         cooccurrence=joint,conditional_partner_given_anchor=joint/counts[ia],
                         pmi_smoothed_bits=float(np.log2((joint+.5)*(len(offsets)-1)/((counts[ia]+.5)*(counts[ib]+.5)))),
                         incidence_cosine=float(baseline[ib]),incidence_rank=int((baseline>=baseline[ib]).sum())))
    return pd.DataFrame(rows)


def evaluate_pairs(vectors,names,evidence):
    lookup={n:i for i,n in enumerate(names)};unit=se.unit_vectors(vectors);rows=[]
    for anchor,group in evidence[evidence.status=='covered'].groupby('anchor'):
        ia=lookup[anchor];scores=unit@unit[ia];scores[ia]=-np.inf
        for record in group.to_dict('records'):
            ib=lookup[record['partner']];rank=int((scores>=scores[ib]).sum())
            rows.append({**record,'cosine':float(scores[ib]),'rank':rank,'rank_percentile':100*(rank-1)/max(len(names)-2,1),
                         'rr':1/rank,'recall10':float(rank<=10),'recall25':float(rank<=25),'recall50':float(rank<=50),
                         'frequency_bucket':str(se.frequency_buckets(np.array([record['anchor_frequency']]))[0])})
    return pd.DataFrame(rows)


def expanded_centroids(vectors,names,concepts,seed=101):
    lookup={n:i for i,n in enumerate(names)};unit=se.unit_vectors(vectors)
    rows,neighbors=[],[];rng=np.random.default_rng(seed)
    for label,members in concepts.items():
        ids=sorted({lookup[normalize_card_name(n)] for n in members if normalize_card_name(n) in lookup})
        if len(ids)<8:
            continue
        shuffled=rng.permutation(ids);train,test=shuffled[:len(ids)//2],shuffled[len(ids)//2:]
        query=unit[train].mean(axis=0);scores=unit@(query/max(np.linalg.norm(query),1e-12));scores[train]=-np.inf
        for i in test:
            rank=int((scores>=scores[i]).sum())
            rows.append(dict(concept=label,card=names[i],rank=rank,rr=1/rank,recall10=float(rank<=10),recall25=float(rank<=25),recall50=float(rank<=50),seed_cards=len(train),heldout_cards=len(test)))
        candidates=np.flatnonzero(np.isfinite(scores));top=candidates[np.argsort(-scores[candidates])[:10]]
        neighbors += [dict(concept=label,card=names[i],cosine=float(scores[i]),rank=r+1,is_heldout=bool(i in test)) for r,i in enumerate(top)]
    return pd.DataFrame(rows),pd.DataFrame(neighbors)


def fixed_stability_ids(names,counts,seed=101,per_bucket=50):
    rng=np.random.default_rng(seed);buckets=se.frequency_buckets(counts);selected=[]
    for bucket in np.unique(buckets):
        ids=np.flatnonzero(buckets==bucket);selected.extend(rng.choice(ids,min(per_bucket,len(ids)),replace=False).tolist())
    lookup={n:i for i,n in enumerate(names)}
    selected.extend(lookup[normalize_card_name(n)] for pair in se.COMBOS for n in pair if normalize_card_name(n) in lookup)
    return np.array(sorted(set(selected)),dtype=int)


def nearest_snapshot(vectors,names,counts,ids,k=20):
    unit=se.unit_vectors(vectors);rows=[];timings=[]
    for repeat in range(4):
        started=time.perf_counter()
        scores=unit[ids[:16]]@unit.T
        for j,index in enumerate(ids[:16]):scores[j,index]=-np.inf
        np.argpartition(scores,-k,axis=1)[:,-k:]
        if repeat:timings.append((time.perf_counter()-started)/min(16,len(ids)))
    for index in ids:
        scores=unit@unit[index];scores[index]=-np.inf
        top=np.argpartition(scores,-k)[-k:];top=top[np.argsort(-scores[top])]
        rows.append(dict(card=names[index],frequency=int(counts[index]),bucket=str(se.frequency_buckets(counts[index:index+1])[0]),neighbors=json.dumps([names[i] for i in top])))
    return pd.DataFrame(rows),float(np.median(timings))


def prepare_evaluation(root,output):
    import joblib
    root,output=Path(root),Path(output)
    cache=output/'evaluation_inputs.joblib'
    if cache.exists():
        policy=json.loads((output/'evaluation_policy.json').read_text())
        if (policy['concepts_sha256'] != se.sha256_file(root/'configs/card2vec_concepts.json')
                or policy['metadata_sha256'] != se.sha256_file(default_oracle_path(root/'data'))):
            raise ValueError('Cached evaluation inputs differ from current labels/metadata; use a fresh evaluation version')
        return joblib.load(cache)
    prepared=output/'prepared';names,_,_,counts=se.load_prepared(prepared)
    metadata,collisions=se.metadata_index(default_oracle_path(root/'data'))
    # Only probe fields cross joblib process boundaries; avoid copying complete
    # printing/image/legality payloads into every evaluation worker.
    metadata={name:{field:card[field] for field in ('name','cmc','type_line','color_identity') if field in card}
              for name,card in metadata.items()}
    concepts=json.loads((root/'configs/card2vec_concepts.json').read_text())['concepts']
    paths=sorted(p for p in (root/'data/format_corpora').glob('*.jsonl') if not p.name.startswith('limited'))
    print('Recovering provenance for filtered contexts.',flush=True)
    provenance=se.recover_provenance(paths,prepared)
    provenance['formats']={k:v for k,v in provenance['formats'].items() if k in FORMATS}
    provenance['sources']={k:v for k,v in provenance['sources'].items() if k in SOURCES}
    targets,concept_audit=build_targets(prepared,metadata,concepts,provenance)
    concept_audit.to_csv(output/'concept_coverage.csv',index=False)
    concept_counts=concept_audit.groupby(['concept','status']).size().rename('cards').reset_index()
    concept_counts.to_csv(output/'concept_counts.csv',index=False)
    joined=concept_counts[concept_counts.status=='included']
    if len(joined)!=len(concepts) or joined.cards.min()<10:
        raise ValueError('Curated benchmark needs at least 10 joined cards per concept; inspect concept_coverage.csv')
    for task,(ids,y,labels) in targets.items():
        values = y[:,None] if y.ndim==1 else y
        columns = ['bucket'] if task=='mana_bucket' else labels
        pd.DataFrame(values,columns=columns).assign(card=np.array(names)[ids]).to_csv(output/f'targets_{task}.csv',index=False)
    mana_audit=pd.DataFrame(se.mana_value_audit(names,metadata,20))
    mana_audit[mana_audit.status=='included'].to_csv(output/'mana_included.csv',index=False)
    # Invalid raw metadata has no numeric label; use readable reason, not NaN result tables.
    mana_audit[mana_audit.status!='included'].fillna('unavailable').to_csv(output/'mana_exclusions.csv',index=False)
    pd.DataFrame({'card':names,'metadata_status':['joined' if n in metadata else 'ambiguous' if n in collisions else 'missing' for n in names]}).to_csv(output/'metadata_coverage.csv',index=False)
    print('Counting benchmark pair evidence.',flush=True)
    evidence=benchmark_evidence(prepared,concepts)
    evidence[['kind','anchor','partner','status']].to_csv(output/'benchmark_coverage.csv',index=False)
    evidence=evidence[evidence.status=='covered'].copy()
    evidence.to_csv(output/'benchmark_evidence.csv',index=False)
    if not (evidence.kind=='corpus_supported').any():raise ValueError('No corpus-supported benchmark pairs qualified')
    overlaps=[]
    for a,b in combinations(provenance['sources'],2):
        left,right=provenance['sources'][a],provenance['sources'][b]
        overlaps.append(dict(source_a=a,source_b=b,contexts_a=len(left),contexts_b=len(right),shared=len(left&right),jaccard=len(left&right)/max(len(left|right),1)))
    pd.DataFrame(overlaps).to_csv(output/'source_overlap.csv',index=False)
    for exclusive in [False,True]:
        frame=se.source_associations(prepared,provenance,exclusive=exclusive)
        if not frame.empty:frame.to_csv(output/f'source_associations_{"disjoint" if exclusive else "overlapping"}.csv',index=False)
        else:(output/f'source_{exclusive}_unavailable.txt').write_text('Insufficient source/anchor support for this comparison.')
    provenance_summary={'training_contexts':provenance['total'],'matched_contexts':provenance['matched'],
                        'format_contexts':{k:len(v) for k,v in provenance['formats'].items()},
                        'source_contexts':{k:len(v) for k,v in provenance['sources'].items()}}
    (output/'provenance_summary.json').write_text(json.dumps(provenance_summary,indent=2))
    inputs=dict(names=names,counts=counts,metadata=metadata,concepts=concepts,targets=targets,evidence=evidence,
                stability_ids=fixed_stability_ids(names,counts))
    joblib.dump(inputs,cache)
    (output/'evaluation_policy.json').write_text(json.dumps({
        'mana_domain':[0,20],'mana_bins':'[0,1), [1,2), [2,3), [3,4), [4,5), [5,6), [6,inf)',
        'split_seeds':[101,102,103],'training_seeds':list(SEEDS),'dimensions':list(DIMENSIONS),
        'concepts_sha256':se.sha256_file(root/'configs/card2vec_concepts.json'),
        'metadata_sha256':se.sha256_file(default_oracle_path(root/'data')),
        'module_sha256':se.sha256_file(__file__),
        'statistical_pairs':'top five incidence-cosine partners per fixed anchor with joint count >=100 and lift >=2; transductive',
        'random_baseline':'seeded 256-dimensional draws truncated to each model dimension; same card splits',
        'frequency_buckets':{'very_rare':'<10','rare':'10-99','medium':'100-999','common':'1000-9999','very_common':'>=10000'},
    },indent=2))
    return inputs


def evaluate_one(output,run,inputs):
    output=Path(output);path=Path(run['path']);destination=path/'evaluation_v2'
    if (destination/'complete.json').exists():
        return str(destination)
    destination.mkdir(exist_ok=True)
    names,counts=inputs['names'],inputs['counts'];vectors=np.load(path/'vectors.npy',mmap_mode='r')
    frames,unavailable=[],[]
    print(f'Evaluate d={run["dimension"]}, seed={run["seed"]}',flush=True)
    for task,target in inputs['targets'].items():
        print(f'  d={run["dimension"]} seed={run["seed"]} task={task}',flush=True)
        frame,missing=probe_long(vectors,counts,names,inputs['metadata'],target,task)
        if not frame.empty:frames.append(frame)
        if not missing.empty:unavailable.append(missing)
    probes=pd.concat(frames,ignore_index=True)
    if not np.isfinite(probes.value).all():raise ValueError('Nonfinite reported probe metric')
    probes.to_csv(destination/'probes_long.csv',index=False)
    if unavailable:pd.concat(unavailable,ignore_index=True).fillna('not_applicable').to_csv(destination/'unavailable_metrics.csv',index=False)
    pairs=evaluate_pairs(vectors,names,inputs['evidence']);pairs.to_csv(destination/'retrieval.csv',index=False)
    joined_concepts={label:[card for card in members if normalize_card_name(card) in inputs['metadata']]
                     for label,members in inputs['concepts'].items()}
    heldout,nearby=expanded_centroids(vectors,names,joined_concepts)
    heldout.to_csv(destination/'centroids.csv',index=False);nearby.to_csv(destination/'centroid_neighbors.csv',index=False)
    neighborhood,seconds=nearest_snapshot(vectors,names,counts,inputs['stability_ids'])
    neighborhood.to_csv(destination/'neighborhoods.csv',index=False)
    costs={**run,'embedding_parameters':vectors.size,'sgns_embedding_parameters':2*vectors.size,
           'inference_bytes':vectors.nbytes,'sgns_table_bytes':2*vectors.nbytes,'input_bytes_100_cards':100*vectors.shape[1]*4,
           'retrieval_seconds_per_query':seconds}
    (destination/'costs.json').write_text(json.dumps(costs,indent=2))
    (destination/'complete.json').write_text(json.dumps({'dimension':run['dimension'],'seed':run['seed'],'metrics':len(probes),'module_sha256':se.sha256_file(__file__)}))
    return str(destination)


def evaluate_experiment(root,output,jobs=2):
    from threadpoolctl import threadpool_limits
    output=Path(output);inputs=prepare_evaluation(root,output)
    if (output/'runs.csv').exists():
        runs=pd.read_csv(output/'runs.csv').to_dict('records')
    else:
        runs=[]
        for seed in SEEDS:
            for dimension in DIMENSIONS:
                path=output/f'seed_{seed}'/str(dimension)/'sample_0'
                if (path/'stats.json').exists():
                    runs.append({**json.loads((path/'stats.json').read_text()),'seed':seed,'path':str(path)})
    if not runs:raise ValueError('No completed targeted model runs')
    with threadpool_limits(limits=1), parallel_config(backend='loky',inner_max_num_threads=1):
        Parallel(n_jobs=jobs,verbose=10)(delayed(evaluate_one)(output,run,inputs) for run in runs)
    if len(runs)==len(SEEDS)*len(DIMENSIONS):build_report(root,output)
    else:print(f'Evaluated {len(runs)} finished runs; full report waits for all six.',flush=True)


def summarize_seed_values(values):
    """Average split repeats before this function: uncertainty unit is training seed."""
    summary=values.groupby(['dimension','metric']).value.agg(['mean','std','count']).reset_index()
    if (summary['count']<3).any():raise ValueError('Headline uncertainty requires at least three training seeds')
    summary['mean_std']=summary.apply(lambda r:f"{r['mean']:.4f} ± {r['std']:.4f}",axis=1)
    return summary


def build_report(root,output):
    from scipy.stats import spearmanr, t
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,output=Path(root),Path(output)
    probe_frames,pair_frames,centroid_frames,neighborhood_frames,cost_rows=[],[],[],[],[]
    for seed in SEEDS:
        for dimension in DIMENSIONS:
            path=output/f'seed_{seed}'/str(dimension)/'sample_0/evaluation_v2'
            if not (path/'complete.json').exists():raise ValueError(f'Run not yet evaluated: {path}')
            tags={'seed':seed,'dimension':dimension}
            probe_frames.append(pd.read_csv(path/'probes_long.csv').assign(**tags))
            pair_frames.append(pd.read_csv(path/'retrieval.csv').assign(**tags))
            centroid_frames.append(pd.read_csv(path/'centroids.csv').assign(**tags))
            neighborhood_frames.append(pd.read_csv(path/'neighborhoods.csv').assign(**tags))
            cost_rows.append(json.loads((path/'costs.json').read_text()))
    probes=pd.concat(probe_frames,ignore_index=True);pairs=pd.concat(pair_frames,ignore_index=True)
    centroids=pd.concat(centroid_frames,ignore_index=True);neighbors=pd.concat(neighborhood_frames,ignore_index=True)
    costs=pd.DataFrame(cost_rows)
    probes.to_csv(output/'all_probes_long.csv',index=False);pairs.to_csv(output/'all_retrieval.csv',index=False)
    probes[(probes.task=='archetype')&(probes.baseline=='learned')&(probes.group=='all')&
           (probes.metric=='label_f1')][['split_seed','label','support','n']].drop_duplicates().to_csv(
               output/'archetype_split_support.csv',index=False)
    centroids.to_csv(output/'all_centroids.csv',index=False)
    main=probes[(probes.baseline=='learned')&(probes.group=='all')&probes.metric.isin(['macro_f1','mae','median_ae','r2','spearman'])]
    per_seed=main.groupby(['dimension','seed','task','metric']).value.mean().reset_index()
    per_seed['metric']=per_seed.task+'_'+per_seed.metric
    values=[per_seed[['dimension','seed','metric','value']]]
    for kind,g in pairs.groupby('kind'):
        summary=g.groupby(['dimension','seed'])[['rr','recall10','recall25','recall50']].mean().reset_index().melt(id_vars=['dimension','seed'],var_name='metric',value_name='value')
        summary['metric']=kind+'_'+summary.metric
        values.append(summary)
    concept_summary=centroids.groupby(['dimension','seed','concept'])[['rr','recall10','recall25','recall50']].mean().reset_index()
    concept_summary.to_csv(output/'centroid_by_seed.csv',index=False)
    centroid_overall=concept_summary.groupby(['dimension','seed'])[['rr','recall10','recall25','recall50']].mean().reset_index().melt(id_vars=['dimension','seed'],var_name='metric',value_name='value')
    centroid_overall['metric']='centroid_'+centroid_overall.metric;values.append(centroid_overall)
    stability=[]
    for dimension,g in neighbors.groupby('dimension'):
        for left,right in combinations(SEEDS,2):
            a=g[g.seed==left].set_index('card');b=g[g.seed==right].set_index('card')
            for card in a.index.intersection(b.index):
                x,y=set(json.loads(a.loc[card,'neighbors'])),set(json.loads(b.loc[card,'neighbors']))
                stability.append(dict(dimension=dimension,seed_a=left,seed_b=right,card=card,
                                      frequency=int(a.loc[card,'frequency']),bucket=a.loc[card,'bucket'],jaccard=len(x&y)/len(x|y)))
    stability=pd.DataFrame(stability);stability.to_csv(output/'neighborhood_stability.csv',index=False)
    stability.groupby(['dimension','bucket']).agg(mean=('jaccard','mean'),std=('jaccard','std'),comparisons=('jaccard','size'),cards=('card','nunique')).reset_index().to_csv(output/'stability_by_frequency.csv',index=False)
    # Each seed's score compares its neighborhoods with the other two seeds.
    stability_seed=[]
    for dimension,g in stability.groupby('dimension'):
        for seed in SEEDS:
            score=g[(g.seed_a==seed)|(g.seed_b==seed)].jaccard.mean()
            stability_seed.append(dict(dimension=dimension,seed=seed,metric='neighborhood_jaccard',value=score))
    values.append(pd.DataFrame(stability_seed))
    for col in ['embedding_parameters','inference_bytes','sgns_table_bytes','input_bytes_100_cards','serialized_bytes','training_seconds','retrieval_seconds_per_query']:
        values.append(costs[['dimension','seed',col]].rename(columns={col:'value'}).assign(metric=col))
    seed_values=pd.concat(values,ignore_index=True)
    if not np.isfinite(seed_values.value).all():raise ValueError('Nonfinite headline metric; report aborted')
    seed_values.to_csv(output/'individual_seed_metrics.csv',index=False)
    summary=summarize_seed_values(seed_values);summary.to_csv(output/'seed_summary.csv',index=False)
    comparison=summary.pivot(index='metric',columns='dimension',values='mean_std').reset_index()
    comparison.to_csv(output/'dimension_comparison.csv',index=False)
    paired=[]
    descriptive_only={'neighborhood_jaccard','embedding_parameters','inference_bytes','sgns_table_bytes',
                      'input_bytes_100_cards','serialized_bytes','training_seconds','retrieval_seconds_per_query'}
    for metric,g in seed_values.groupby('metric'):
        if metric in descriptive_only:
            continue  # Shared-seed Jaccard is dependent; costs use descriptive means/SDs only.
        pivot=g.pivot(index='seed',columns='dimension',values='value')
        delta=pivot[256]-pivot[128];mean=float(delta.mean());sd=float(delta.std());half=float(t.ppf(.975,len(delta)-1)*sd/np.sqrt(len(delta)))
        margin=.05 if metric.endswith(('mae','median_ae')) else .02 if metric=='neighborhood_jaccard' else .01
        low,high=mean-half,mean+half
        certainty='clear_practical_difference' if low>margin or high < -margin else 'weak_trend' if low>0 or high<0 else 'statistically_unresolved'
        paired.append(dict(metric=metric,delta_256_minus_128=mean,delta_sd=sd,ci95_low=low,ci95_high=high,practical_margin=margin,assessment=certainty))
    differences=pd.DataFrame(paired);differences.to_csv(output/'paired_dimension_differences.csv',index=False)
    # Task-specific presentation avoids a union of mostly inapplicable NaN columns.
    for task,g in probes.groupby('task'):
        aggregated=g.groupby(['dimension','seed','baseline','group','metric','label']).agg(value=('value','mean'),support=('support','mean'),n=('n','mean'),mean_log_frequency=('mean_log_frequency','mean')).reset_index()
        aggregated.to_csv(output/f'{task}_detail_by_seed.csv',index=False)
    baseline_seed=probes[(probes.group=='all')&probes.metric.isin(['macro_f1','mae','median_ae','r2','spearman'])].groupby(['dimension','seed','task','baseline','metric']).value.mean().reset_index()
    baseline=baseline_seed.groupby(['dimension','task','baseline','metric']).value.agg(['mean','std']).reset_index()
    baseline.to_csv(output/'baseline_comparison.csv',index=False)
    correlations=[]
    for (dimension,seed,kind),g in pairs.groupby(['dimension','seed','kind']):
        for feature in ['anchor_frequency','partner_frequency','cooccurrence','pmi_smoothed_bits']:
            if g[feature].nunique()>1 and g['rank'].nunique()>1:
                correlations.append(dict(dimension=dimension,seed=seed,kind=kind,feature=feature,
                    rank_spearman=float(spearmanr(g[feature],g['rank']).statistic),pairs=len(g),
                    distinct_undirected_pairs=len({tuple(sorted((a,b))) for a,b in zip(g.anchor,g.partner)})))
    pd.DataFrame(correlations).to_csv(output/'retrieval_evidence_correlations.csv',index=False)
    # Incidence similarity baseline reuses exact corpus pair counts, without a huge full matrix.
    evidence=pd.read_csv(output/'benchmark_evidence.csv')
    evidence['incidence_rr']=1/evidence.incidence_rank
    for k in [10,25,50]:evidence[f'incidence_recall{k}']=(evidence.incidence_rank<=k).astype(float)
    evidence.groupby('kind')[['incidence_rr','incidence_recall10','incidence_recall25','incidence_recall50']].mean().to_csv(output/'incidence_baseline.csv')
    freq=probes[(probes.baseline=='learned')&probes.group.str.startswith('frequency:')&(probes.metric=='macro_f1')&probes.task.isin(['color','type','format'])]
    freq=freq.groupby(['dimension','seed','task','group']).agg(value=('value','mean'),log_frequency=('mean_log_frequency','mean'),n=('n','mean')).reset_index()
    freq.to_csv(output/'semantic_frequency_by_seed.csv',index=False)
    pairs.groupby(['dimension','seed','kind','frequency_bucket']).agg(pairs=('rank','size'),mrr=('rr','mean'),recall50=('recall50','mean'),mean_rank=('rank','mean')).reset_index().to_csv(output/'retrieval_frequency_by_seed.csv',index=False)
    fig,axes=plt.subplots(1,4,figsize=(17,4))
    for axis,task in zip(axes[:3],['color','type','format']):
        for dimension,g in freq[freq.task==task].groupby('dimension'):
            points=g.groupby('group').agg(x=('log_frequency','mean'),mean=('value','mean'),sd=('value','std')).sort_values('x')
            axis.errorbar(points.x,points['mean'],yerr=points.sd,marker='o',label=str(dimension))
        axis.set(title=task,xlabel='Mean log10(frequency + 1)',ylabel='Macro F1');axis.legend()
    for dimension,g in stability.groupby('dimension'):
        points=g.assign(logf=np.log10(g.frequency+1)).groupby('bucket').agg(x=('logf','mean'),y=('jaccard','mean')).sort_values('x')
        axes[3].plot(points.x,points.y,marker='o',label=str(dimension))
    axes[3].set(title='Neighborhood stability',xlabel='Mean log10(frequency + 1)',ylabel='Top-20 Jaccard');axes[3].legend()
    fig.tight_layout();fig.savefig(output/'frequency_quality.png',dpi=150);plt.close(fig)
    thresholds=[]
    order=['very_rare','rare','medium','common','very_common'];lower=[0,10,100,1000,10000]
    for dimension in DIMENSIONS:
        for task,target in [('color',.8),('type',.5),('format',.6)]:
            means=freq[(freq.dimension==dimension)&(freq.task==task)].groupby('group').value.mean()
            qualifying=[i for i,b in enumerate(order) if means.get('frequency:'+b,-1)>=target]
            thresholds.append(dict(dimension=dimension,task=task,criterion=target,
                first_qualifying_bucket=order[qualifying[0]] if qualifying else 'none_observed',
                approximate_lower_frequency=lower[qualifying[0]] if qualifying else 'not_established'))
        means=stability[stability.dimension==dimension].groupby('bucket').jaccard.mean()
        qualifying=[i for i,b in enumerate(order) if means.get(b,-1)>=.5]
        thresholds.append(dict(dimension=dimension,task='neighborhood_stability',criterion=.5,
            first_qualifying_bucket=order[qualifying[0]] if qualifying else 'none_observed',
            approximate_lower_frequency=lower[qualifying[0]] if qualifying else 'not_established'))
    pd.DataFrame(thresholds).to_csv(output/'frequency_reliability_screen.csv',index=False)
    primary=['color_macro_f1','type_macro_f1','mana_mae','mana_bucket_macro_f1','format_macro_f1','archetype_macro_f1','corpus_supported_rr','mechanical_combo_rr','centroid_rr']
    clear=differences[differences.metric.isin(primary)&(differences.assessment=='clear_practical_difference')]
    advantages=[]
    for r in clear.itertuples():
        gain=-r.delta_256_minus_128 if r.metric.endswith('mae') else r.delta_256_minus_128
        advantages.append(256 if gain>0 else 128)
    decision='256 preferred' if advantages.count(256)>=2 and 128 not in advantages else '128 preferred' if advantages.count(128)>=2 and 256 not in advantages else 'no meaningful difference yet'
    chosen=comparison[comparison.metric.isin(primary+['neighborhood_jaccard','training_seconds','inference_bytes','retrieval_seconds_per_query'])]
    lines=['# Second Card2Vec experiment',f'**Decision: {decision}.**',
           'Three independent training seeds per dimension; three fixed card-split repeats are averaged within each seed. Values below are mean ± sample SD across training seeds.',
           chosen.to_markdown(index=False) if __import__('importlib').util.find_spec('tabulate') else chosen.to_string(index=False),
           '## Interpretation',
           'Semantic, retrieval and cost metrics above must be considered together. The paired differences file reports 95% t intervals across the three matched training seeds; these are noisy with only three seeds and uncorrected for multiple comparisons.',
           'A preference requires at least two primary metrics with paired intervals wholly beyond the prespecified practical margin (0.01 F1/MRR or 0.05 mana MAE), and no primary metric clearly favoring the other dimension. Otherwise the result remains unresolved. This is a research decision rule, not a formal equivalence claim.',
           '128 has half the input-vector parameter count and bytes of 256. Training times reflect two concurrent jobs; retrieval timings are CPU measurements and not a downstream-model benchmark.',
           'Mana targets are finite [0,20], with all excluded cards and reasons in mana_exclusions.csv. Land-free regression and bucketed classification are separate. Original contaminated mana scores are not a valid comparator.',
           'Corpus-supported pairs are selected using >=100 shared contexts and >=2 prevalence lift. The incidence baseline and benchmark selection share corpus evidence, so these are descriptive/transductive results, not held-out recommendation quality.',
           'Mechanical combo scores include per-pair co-occurrence, conditional prevalence, smoothed PMI and rank percentile. Five relationships are tested in both directions; those ten rows are not independent combo examples. Correlations on this tiny benchmark are descriptive. Sparse play evidence and rules-text dependence must not be conflated with a failure on frequent synergies.',
           'Archetype labels are manually selected and inspectable, with explicit overlap. This larger benchmark remains curated, not exhaustive ground truth. Centroid seed cards are excluded from candidate retrieval.',
           'Frequency thresholds are descriptive screens (color F1 >=.8, type >=.5, format >=.6), not guarantees. Bins span orders of magnitude; inspect support and variability before using a lower bound.',
           pd.DataFrame(thresholds).to_string(index=False),
           'Top-20 Jaccard compares the same frequency-stratified cards across seeds. Pairwise stability measurements share seeds and are not independent replicates.',
           'Source overlap is quantified separately; disjoint source associations still do not establish source-held-out embedding generalization. Unknown-source contexts remain in training.',
           'Extremely rare/new cards, sparse mechanical combos and rules-dependent relationships remain limitations of co-occurrence-only training. No text-model architecture is added here.',
           '## Files',
           'individual_seed_metrics.csv; seed_summary.csv; paired_dimension_differences.csv; all_probes_long.csv; all_retrieval.csv; neighborhood_stability.csv; quality_counts.csv; quality_rejections.csv; concept_counts.csv; source_overlap.csv.',
           'Undefined metrics (for example Spearman for a constant predictor) are omitted with explicit reasons in each run\'s unavailable_metrics.csv. Headline numeric results are required to be finite; missing data is never replaced with fabricated zeros.'
    ]
    supported_failures=pairs[(pairs.kind=='mechanical_combo')&(pairs.cooccurrence>=1000)].groupby(
        ['dimension','anchor','partner']).agg(mean_rank=('rank','mean'),cooccurrence=('cooccurrence','first')).reset_index()
    supported_failures=supported_failures[supported_failures.mean_rank>50].sort_values('mean_rank',ascending=False)
    if not supported_failures.empty:
        lines.extend(['## Supported mechanical relationships that remain difficult',
                      'These pairs occur in at least 1,000 retained contexts but average outside the top 50. '
                      'Their failures cannot be attributed simply to absent or extremely sparse training evidence. '
                      'Cosine geometry, multiple deck roles and the SGNS objective require further investigation.',
                      supported_failures.head(8).to_string(index=False)])
    spell_metrics=summary[summary.metric.isin(['mana_spells_mae','mana_spells_spearman','mana_bucket_macro_f1'])]
    lines.extend(['## Mana structure beyond the numeric target',spell_metrics[['dimension','metric','mean_std']].to_string(index=False)])
    (output/'methodology_and_diagnostics.md').write_text('\n\n'.join(lines),encoding='utf-8')
    compact_table=comparison[comparison.metric.isin(primary+['neighborhood_jaccard'])]
    compact_table_text=(compact_table.to_markdown(index=False) if __import__('importlib').util.find_spec('tabulate')
                        else '```text\n'+compact_table.to_string(index=False)+'\n```')
    compact=['# Second Card2Vec experiment',f'**Decision: {decision}.**',
             'Three independent training seeds per dimension; three fixed card splits are averaged within each seed. Values are mean ± sample SD across training seeds.',
             compact_table_text]
    for assessment,title in [('clear_practical_difference','Clear practical differences'),
                             ('weak_trend','Weak trends'),('statistically_unresolved','Unresolved differences')]:
        selected=differences[differences.metric.isin(primary)&(differences.assessment==assessment)]
        descriptions=[]
        for row in selected.itertuples():
            gain=-row.delta_256_minus_128 if row.metric.endswith('mae') else row.delta_256_minus_128
            description=('equal point estimates' if abs(gain)<1e-12 else
                         f'{"256" if gain>0 else "128"} has the better point estimate')
            descriptions.append(f'{row.metric} ({description})')
        compact.append(f'**{title}:** '+('; '.join(descriptions) if descriptions else 'none')+'.')
    recall50=summary[summary.metric=='centroid_recall50'].set_index('dimension')
    if set(DIMENSIONS).issubset(recall50.index):
        compact.append(f'**Supplementary retrieval trade-off:** held-out centroid Recall@50 is '
                       f'{recall50.loc[128,"mean"]:.1%} at 128 versus {recall50.loc[256,"mean"]:.1%} at 256. '
                       'Its paired interval is in the differences table. These correlated retrieval metrics are '
                       'not counted as independent votes in the overall decision. "No meaningful difference yet" '
                       'means no consistent overall capacity winner, not that every metric is equal.')
    rare=freq[freq.group.isin(['frequency:rare','frequency:very_rare'])].groupby(['dimension','task','group']).value.mean().reset_index()
    compact.append('**Long tail:** descriptive macro F1 by logarithmic frequency bucket:\n\n```text\n'+rare.to_string(index=False)+'\n```')
    memory=costs.groupby('dimension').inference_bytes.mean()/1024**2
    seconds=costs.groupby('dimension').training_seconds.mean()
    compact.append(f'**Cost:** input vectors use {memory.loc[128]:.2f} MiB at 128 versus {memory.loc[256]:.2f} MiB at 256. '
                   f'Mean training time was {seconds.loc[128]/60:.1f} versus {seconds.loc[256]/60:.1f} minutes. '
                   'The 256-dimensional downstream input tensor is twice as large. Timings reflect concurrent jobs and are approximate.')
    compact.append('**Stability and certainty:** top-20 Jaccard is shown above and by frequency in stability_by_frequency.csv. '
                   'Those comparisons share models; no independence-based interval is assigned to them. Paired 95% t intervals for primary scores are in paired_dimension_differences.csv; three seeds and multiple comparisons limit certainty.')
    if not supported_failures.empty:
        compact.append('**Retrieval limitation:** several mechanical pairs with at least 1,000 observed shared contexts still rank outside the top 50. '
                       'The problem is not simply absent training data. See all_retrieval.csv for support, PMI, ranks and percentiles.')
    mechanical_baseline=evidence[evidence.kind=='mechanical_combo']
    mechanical_models=summary[summary.metric=='mechanical_combo_rr']
    if not mechanical_baseline.empty and not mechanical_models.empty:
        compact.append(f'**Direct co-occurrence check:** mechanical-combo incidence MRR is {mechanical_baseline.incidence_rr.mean():.3f}, '
                       f'versus SGNS cosine MRR {mechanical_models["mean"].min():.3f}–{mechanical_models["mean"].max():.3f}. '
                       'If incidence is stronger, these failures concern the learned representation/scoring, not an intrinsic absence of co-occurrence evidence.')
    compact.append('**Scope:** mana uses finite [0,20] values with explicit exclusions; the original contaminated regression is not a comparator. '
                   'Expanded archetypes remain curated, and some split-level categories have only two positives. '
                   'Disjoint source agreement is not source-held-out generalization. Rare/new cards and rules-dependent relationships remain weaknesses of pure co-occurrence training.')
    compact.append('Details: methodology_and_diagnostics.md, individual_seed_metrics.csv, baseline_comparison.csv, and frequency_quality.png. '
                   'Only finite applicable metrics are reported; unavailable statistics have explicit reasons.')
    (output/'report.md').write_text('\n\n'.join(compact),encoding='utf-8')
    print('\n'.join(compact[:4]),flush=True)
    return summary


def evaluate_original_mana(root,output):
    """Correct the first-run mana evaluation without changing its embeddings."""
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    original=root/'artifacts/card2vec/static_v1'
    names,_,_,counts=se.load_prepared(original/'prepared')
    metadata,_=se.metadata_index(default_oracle_path(root/'data'))
    target=se.metadata_targets(names,metadata)['mana']
    frames=[]
    for dimension in DIMENSIONS:
        vectors=np.load(original/str(dimension)/'sample_0/vectors.npy',mmap_mode='r')
        frame,missing=probe_long(vectors,counts,names,metadata,target,'mana')
        frames.append(frame.assign(dimension=dimension))
    frame=pd.concat(frames,ignore_index=True)
    frame.to_csv(output/'first_run_mana_corrected.csv',index=False)
    result=frame[frame.group=='all'].groupby(['dimension','baseline','metric']).value.mean().reset_index()
    print(result.to_string(index=False),flush=True)
    return result
