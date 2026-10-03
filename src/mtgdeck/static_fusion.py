"""V3.2 evaluation and export using only frozen v2/v3.1 artifacts.

No training, corpus preparation, SVD fitting, or computation on import.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from . import static_relations as r
from .static_export import (DIMENSIONS, fuse_sources, load_static_card_embeddings,
                            require_disjoint, unit_rows, write_checkpoint)
from .artifacts import sha256_file

g = r.g
PRIMARY = ('sgns_input_128', 'ppmi_svd_128', 'concat_input_svd_256',
           'concat_input_context_svd_384')
FUSED = PRIMARY[2:]
RAW = ('concat_input_svd_raw', 'concat_input_context_svd_raw')
TASKS = ('color', 'type', 'format', 'archetype', 'mana', 'mana_spells')


def paths(root):
    base = Path(root).resolve() / 'artifacts/card2vec'
    return base / 'static_v2', base / 'static_v3_relations_v1'


def protected_roots(root):
    return [p for p in (Path(root) / 'artifacts/card2vec').glob('static_*')
            if p.name.startswith(('static_v1', 'static_v2', 'static_v3'))
            and not p.name.startswith('static_v3_2')]


def source_inventory(root):
    """Hash and validate real frozen inputs; vocabulary identity is mandatory."""
    source, relations = paths(root)
    provenance = g.read_json(relations / 'provenance.json')
    for relative, expected in provenance['source_hashes'].items():
        if sha256_file(source / relative) != expected:
            raise ValueError(f'Frozen source hash mismatch: {relative}')
    inputs = joblib.load(source / 'evaluation_inputs.joblib')
    names = g.read_json(source / 'prepared/names.json')
    if list(inputs['names']) != names or len(set(names)) != len(names):
        raise ValueError('Frozen evaluation and corpus vocabulary order disagree')
    index = g.read_json(relations / 'matrices/index.json')
    if index['shape'] != [len(names), len(names)]:
        raise ValueError('Frozen SVD parent matrix vocabulary shape mismatch')
    seeds, artifacts, configs, svd_configs, missing = [], {}, {}, {}, []
    for seed in g.read_json(source / 'evaluation_policy.json')['training_seeds']:
        model = source / f'seed_{seed}/128/sample_0/model.gensim'
        svd_dir = relations / f'svd/d128_s{seed}'
        required = [model, svd_dir / 'vectors.npy', svd_dir / 'complete.json']
        if not all(p.exists() for p in required):
            missing.append({'seed': seed, 'missing': [str(p.relative_to(root)) for p in required if not p.exists()]})
            continue
        files = [*sorted(model.parent.glob('model.gensim*')), model.parent / 'config.json',
                 svd_dir / 'vectors.npy', svd_dir / 'model.json', source / 'prepared/names.json',
                 relations / 'provenance.json', relations / 'matrices/index.json']
        artifacts[str(seed)] = [{'path': p.relative_to(root).as_posix(), 'sha256': sha256_file(p)} for p in files]
        configs[str(seed)] = g.read_json(model.parent / 'config.json')['config']
        svd_configs[str(seed)] = g.read_json(svd_dir / 'model.json')
        seeds.append(seed)
    if not seeds:
        raise ValueError('No corresponding 128d SGNS/SVD artifacts')
    preparation = g.read_json(source / 'prepared/preparation.json')
    metadata = {
        'created_from': ['static_v2', 'static_v3_relations_v1'],
        'corpus_version': 'static_v2/prepared',
        'training_corpus_hash': preparation['sha256'],
        'training_corpus_hash_policy': 'historically recorded source JSONL SHA256; exact filtered packed inputs hashed separately',
        'training_contexts': len(inputs['offsets']) - 1 if 'offsets' in inputs else preparation['contexts'],
        'packed_corpus_hashes': {k: v for k, v in provenance['source_hashes'].items() if k.startswith('prepared/')},
        'sgns_training_configuration': configs, 'ppmi_svd_configuration': svd_configs,
        'canonicalization': 'mtgdeck.data.normalize_card_name: Unicode normalization, remove diacritics, casefold, strip a- prefix, collapse whitespace, front face before slash',
        'alignment': 'exact canonical vocabulary names; SGNS input/context reindexed via aligned_model; SVD uses frozen prepared/names.json order recorded by v3.1 provenance',
        'source_provenance': provenance,
        'implementation_hashes': {Path(p).name: sha256_file(Path(p)) for p in
                                  (__file__, Path(__file__).with_name('static_export.py'), r.__file__)},
        'runtime': {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                    for p in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'gensim', 'joblib')}},
        'missing_artifacts': missing,
        'seed_policy': 'Pair SGNS and randomized SVD by seed; never average coordinates across seeds.',
        'known_discrepancies': ['Notebook accepted-context audit is 1002916; frozen packed preparation and matrix index are 1002915. Export uses packed training contexts.'],
    }
    if metadata['training_contexts'] != index['contexts']:
        raise ValueError('Packed corpus and matrix context counts disagree')
    return inputs, seeds, artifacts, metadata


def export(root, output):
    root = Path(root).resolve()
    output = require_disjoint(output, protected_roots(root))
    if output.exists():
        raise FileExistsError(f'Export already exists: {output}; use loader or a new version')
    inputs, seeds, artifacts, metadata = source_inventory(root)
    source, relations = paths(root)

    def matrices():
        for seed in seeds:
            print(f'Align and fuse seed {seed}', flush=True)
            u, v = g.aligned_model(source / f'seed_{seed}/128/sample_0/model.gensim', inputs['names'])
            if v is None:
                raise ValueError(f'Seed {seed} has no context matrix')
            svd = np.load(relations / f'svd/d128_s{seed}/vectors.npy', allow_pickle=False)
            yield seed, fuse_sources(u, v, svd)

    manifest = write_checkpoint(output, list(inputs['names']), matrices(), metadata, artifacts,
                                protected_roots(root))
    (output / 'README.md').write_text(EXPORT_README, encoding='utf-8')
    return manifest


EXPORT_README = '''# Static card export v1

One canonical card per UTF-8 vocab.txt line; every float32 matrix has that exact row order.
Manifest includes per-seed file hashes, vocabulary hash, source hashes, configurations,
packed corpus provenance, and normalization. No coordinates are averaged across seeds.
Seed 42 is conventional, not test-selected. Each seed has separate SGNS and SVD sources.
Raw matrices remain unchanged. Normalized concatenations L2-normalize each source block
per card before concatenation (zero blocks stay zero); cosine evaluation normalizes the
final vector. Raw concatenations are secondary diagnostics.

```python
from mtgdeck.static_export import load_static_card_embeddings
bundle = load_static_card_embeddings(artifact_root="artifacts/card2vec/static_export_v1")
card_id = bundle.index("lightning bolt")  # exact canonical key, clear error if absent
matrix = bundle.embedding_matrix  # writable contiguous float32; torch.from_numpy(matrix)
```

Downstream: SGNS input + SGNS context + PPMI-SVD -> concat -> trainable projection -> deck encoder.
Choose input width from matrix.shape[1], e.g. torch.nn.Linear(matrix.shape[1], d_model).
Train that projection end-to-end with the deck model; no static probe labels train it.
Compare random initialization, SGNS input, PPMI-SVD, 256d fusion, and 384d fusion.
Allow frozen and fine-tuned embedding ablations. Do not hardwire a 384d input.
Masked/partial deck completion, convergence, data efficiency, rare-card performance,
cross-format transfer and temporal/new-card generalization remain downstream experiments.
Absent cards require an explicit downstream cold-start policy; this export invents none.

Rebuild with scripts/run_static_fusion.py --stage export --export NEW_VERSION_PATH.
Requires existing frozen models/SVD; never trains or scrapes. Existing exports are immutable.
See ../static_v3_2_fusion_v1/report.md for evaluated results and limitations.
'''


def prepare(root, output, checkpoint):
    root, output, checkpoint = Path(root).resolve(), Path(output).resolve(), Path(checkpoint).resolve()
    require_disjoint(output, [*protected_roots(root), checkpoint])
    if (output / 'experiment.json').exists():
        expected = g.read_json(output / 'experiment.json')
        if expected['checkpoint_hash'] != sha256_file(checkpoint / 'manifest.json'):
            raise ValueError('Checkpoint changed; select a new evaluation version')
        if expected['implementation_hashes'] != implementation_hashes():
            raise ValueError('Evaluation code changed; select a new evaluation version')
        return
    source, relations = paths(root)
    manifest = g.read_json(checkpoint / 'manifest.json')
    with g.stage_directory(output):
        policy = g.read_json(source / 'evaluation_policy.json')
        records = g.read_json(relations / 'benchmark/queries.json')
        g.write_json(output / 'experiment.json', {
            'version': 'static_v3.2', 'checkpoint_hash': sha256_file(checkpoint / 'manifest.json'),
            'implementation_hashes': implementation_hashes(), 'training_seeds': manifest['training_seeds'],
            'evaluation_policy': policy, 'relation_policy': g.read_json(relations / 'relations_manifest.json'),
            'benchmark_hash': sha256_file(relations / 'benchmark/queries.json'),
            'frozen_inputs_hash': sha256_file(source / 'evaluation_inputs.joblib'),
            'primary': list(PRIMARY), 'raw_diagnostic': list(RAW),
            'probe_policy': 'unchanged v2 probe_long, learned rows only in comparisons; legacy random baseline caps at 256 features and is not a 384d control',
            'aggregation': 'mean over fixed split seeds within each training seed, then mean/sample SD across training seeds; no coordinate averaging',
        })
        g.write_json(output / 'queries.json', records)
        r.coverage(records).to_csv(output / 'coverage.csv', index=False)
        current_v31(root, output / 'completed_v3_1')


def implementation_hashes():
    return {Path(p).name: sha256_file(Path(p)) for p in
            (__file__, Path(__file__).with_name('static_export.py'), r.__file__, g.v2.__file__)}


def current_v31(root, dest):
    """Regenerate aggregate results outside immutable v3.1, including completed probes."""
    _, relations = paths(root)
    with g.stage_directory(dest):
        frames, probes, hashes = [], [], {}
        completed = []
        for marker in sorted((relations / 'relations').glob('*/complete.json')):
            directory = marker.parent
            info = g.read_json(directory / 'model.json')
            frame = pd.read_csv(directory / 'metrics.csv').assign(**info)
            frames.append(frame)
            completed.append(directory.name)
            hashes[str(directory.relative_to(relations) / 'metrics.csv')] = sha256_file(directory / 'metrics.csv')
        completed_probes = []
        for marker in sorted((relations / 'relation_probes').glob('*/complete.json')):
            directory = marker.parent
            name, dim, seed = directory.name.rsplit('_', 2)
            frame = pd.read_csv(directory / 'probes.csv')
            frame = frame[(frame.baseline == 'learned') & (frame.group == 'all') & (frame.label == 'all')]
            probes.append(frame.assign(representation=name, dimension=int(dim[1:]), seed=int(seed[1:])))
            completed_probes.append(directory.name)
            hashes[str(directory.relative_to(relations) / 'probes.csv')] = sha256_file(directory / 'probes.csv')
        expected = {f'{m}_d{d}_s{s}' for m in (*r.MODES, 'ppmi_svd_cosine', 'ppmi_svd_dot', 'compatibility')
                    for d in (128, 256) for s in (42, 43, 44)} | {'raw_incidence','incidence_cosine','pmi','ppmi','popularity'}
        expected_probes = {f'{m}_d{d}_s{s}' for m in ('input','context','combined','input_unit','context_unit','normalized_sum','ppmi_svd')
                           for d in (128,256) for s in (42,43,44)}
        pending, pending_probes = sorted(expected-set(completed)), sorted(expected_probes-set(completed_probes))
        g.write_json(dest / 'status.json', dict(status='complete' if not pending and not pending_probes else 'partial',
            completed=completed, pending=pending, completed_probes=completed_probes, pending_probes=pending_probes,
            note='Regenerated from completed v3.1 stages; historical aggregate snapshots are unchanged.'))
        group = ['scorer','dimension','seed','relation','source']
        frame = pd.concat(frames).groupby(group)[['mrr','recall@10','recall@20','recall@50']].mean().reset_index()
        frame.to_csv(dest / 'retrieval_per_seed.csv', index=False)
        frame.groupby(group[:-3]+['relation','source'])[['mrr','recall@10','recall@20','recall@50']].agg(['mean','std','count']).to_csv(dest / 'retrieval_summary.csv')
        pd.concat(probes).groupby(['representation','dimension','seed','task','metric']).value.mean().to_csv(dest / 'probe_per_seed.csv')
        g.write_json(dest / 'source_hashes.json', hashes)


def evaluate(root, output, checkpoint, stage):
    prepare(root, output, checkpoint)
    source, relations = paths(root)
    output, checkpoint = Path(output), Path(checkpoint)
    inputs = joblib.load(source / 'evaluation_inputs.joblib')
    records = g.read_json(output / 'queries.json')
    experiment = g.read_json(output / 'experiment.json')
    if sha256_file(source / 'evaluation_inputs.joblib') != experiment['frozen_inputs_hash']:
        raise ValueError('Frozen evaluation inputs changed')
    seeds = experiment['training_seeds']
    for seed in seeds:
        for name in (*PRIMARY, *RAW) if stage == 'retrieval' else PRIMARY:
            dest = output / stage / f'{name}_s{seed}'
            if (dest / 'complete.json').exists():
                continue
            print(f'{stage}: {name}, seed {seed}', flush=True)
            bundle = load_static_card_embeddings(name, seed, checkpoint)
            if bundle.vocab != list(inputs['names']):
                raise ValueError('Export and frozen evaluation vocabulary order disagree')
            with g.stage_directory(dest):
                if stage == 'retrieval':
                    baseline = {'sgns_input_128': 'input_cosine', 'ppmi_svd_128': 'ppmi_svd_cosine'}.get(name)
                    if baseline:
                        origin = relations / f'relations/{baseline}_d128_s{seed}/metrics.csv'
                        frame = pd.read_csv(origin)
                        g.write_json(dest / 'reuse.json', {'path': str(origin.relative_to(root)), 'sha256': sha256_file(origin)})
                    else:
                        frame, _ = r.evaluate_records(g.GeometryScorer(bundle.embedding_matrix), records, bundle.vocab, inputs['counts'])
                    frame.to_csv(dest / 'metrics.csv', index=False)
                else:
                    baseline = {'sgns_input_128': 'input', 'ppmi_svd_128': 'ppmi_svd'}.get(name)
                    if baseline:
                        origin = relations / f'relation_probes/{baseline}_d128_s{seed}/probes.csv'
                        pd.read_csv(origin).to_csv(dest / 'probes.csv', index=False)
                        g.write_json(dest / 'reuse.json', {'path': str(origin.relative_to(root)), 'sha256': sha256_file(origin)})
                        g.write_json(dest / 'unavailable.json', g.read_json(origin.parent / 'unavailable.json'))
                    else:
                        frames, unavailable = [], []
                        for task in TASKS:
                            frame, missing = g.v2.probe_long(bundle.embedding_matrix, inputs['counts'], bundle.vocab,
                                inputs['metadata'], inputs['targets'][task], task,
                                split_seeds=tuple(experiment['evaluation_policy']['split_seeds']))
                            frames.append(frame)
                            unavailable.extend(missing.to_dict('records'))
                        pd.concat(frames).to_csv(dest / 'probes.csv', index=False)
                        g.write_json(dest / 'unavailable.json', unavailable)


def redundancy(checkpoint, output):
    rows = []
    manifest = g.read_json(Path(checkpoint) / 'manifest.json')
    rng = np.random.default_rng(3202)
    ids = rng.choice(manifest['vocabulary_size'], min(2000, manifest['vocabulary_size']), replace=False)
    pairs = rng.integers(0, len(ids), (20000, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    for seed in manifest['training_seeds']:
        blocks = [load_static_card_embeddings(n, seed, checkpoint).embedding_matrix[ids].astype(float)
                  for n in ('sgns_input_128','sgns_context_128','ppmi_svd_128')]
        for a, b, label in ((0,2,'input_svd'),(1,2,'context_svd'),(0,1,'input_context')):
            x, y = (z-z.mean(axis=0) for z in (blocks[a],blocks[b]))
            cka = np.linalg.norm(x.T@y, 'fro')**2 / (np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro'))
            xu, yu = map(unit_rows, (blocks[a],blocks[b]))
            cos = [np.sum(z[pairs[:,0]]*z[pairs[:,1]],axis=1) for z in (xu,yu)]
            rows.append(dict(seed=seed, pair=label, linear_cka=cka, cosine_correlation=np.corrcoef(*cos)[0,1],
                             sample_cards=len(ids), sample_pairs=len(pairs), sample_seed=3202))
    pd.DataFrame(rows).to_csv(Path(output) / 'redundancy.csv', index=False)


def report(root, output, checkpoint):
    prepare(root, output, checkpoint)
    output = Path(output)
    dest = output / 'report'
    if dest.exists():
        raise FileExistsError('Report already exists; completed reports are immutable')
    seeds = g.read_json(output / 'experiment.json')['training_seeds']
    for stage, names in (('retrieval', (*PRIMARY, *RAW)), ('probes', PRIMARY)):
        for seed in seeds:
            for name in names:
                if not (output / stage / f'{name}_s{seed}/complete.json').exists():
                    raise ValueError(f'Incomplete {stage}: {name} seed {seed}')
    with g.stage_directory(dest):
        retrieval, probes = [], []
        for seed in seeds:
            for name in (*PRIMARY, *RAW):
                frame = pd.read_csv(output / 'retrieval' / f'{name}_s{seed}/metrics.csv')
                retrieval.append(frame.assign(representation=name, dimension=DIMENSIONS[name], seed=str(seed)))
            for name in PRIMARY:
                frame = pd.read_csv(output / 'probes' / f'{name}_s{seed}/probes.csv')
                frame = frame[(frame.baseline=='learned') & (frame.group=='all') & (frame.label=='all')]
                probes.append(frame.assign(representation=name, dimension=DIMENSIONS[name], seed=seed))
        _, relations = paths(root)
        for baseline in ('incidence_cosine', 'popularity'):
            frame = pd.read_csv(relations / 'relations' / baseline / 'metrics.csv')
            retrieval.append(frame.assign(representation=baseline,dimension=0,seed='deterministic'))
        metrics = ['mrr','recall@10','recall@20','recall@50']
        group = ['representation','dimension','relation','source']
        per_seed = pd.concat(retrieval).groupby(group+['seed'])[metrics].mean().reset_index()
        per_seed.to_csv(dest / 'retrieval_per_seed.csv',index=False)
        summary = per_seed.groupby(group)[metrics].agg(['mean','std','count'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary = summary.reset_index()
        for name, column in (('sgns_input_128','delta_vs_sgns_input'),('ppmi_svd_128','delta_vs_ppmi_svd')):
            ref = summary[summary.representation==name][['relation','source','mrr_mean']].rename(columns={'mrr_mean':column})
            summary = summary.merge(ref,on=['relation','source'],validate='many_to_one')
            summary[column] = summary.mrr_mean-summary[column]
        summary.to_csv(dest / 'retrieval_summary.csv',index=False)
        pgroup = ['representation','dimension','seed','task','metric']
        pseed = pd.concat(probes).groupby(pgroup).value.mean().reset_index()
        pseed.to_csv(dest / 'probe_per_seed.csv',index=False)
        ps = pseed.groupby(['representation','dimension','task','metric']).value.agg(['mean','std','count']).reset_index()
        ps.to_csv(dest / 'probe_summary.csv',index=False)
        primary = ps[((ps.task.isin(TASKS[:4])) & (ps.metric=='macro_f1')) | ((ps.task=='mana') & (ps.metric=='mae'))].copy()
        primary['score'] = primary.apply(lambda x: f'{x["mean"]:.4f} ± {x["std"]:.4f}',axis=1)
        compact = primary.pivot(index='representation',columns='task',values='score').reindex(PRIMARY)
        compact.to_csv(dest / 'probe_comparison.csv')
        redundancy(checkpoint,dest)
        norm = summary[summary.representation.isin(FUSED)].copy()
        raw = summary[summary.representation.isin(RAW)].copy()
        raw['representation'] = raw.representation.map(dict(zip(RAW,FUSED)))
        norm = norm.merge(raw[['representation','relation','source','mrr_mean']],on=['representation','relation','source'],suffixes=('','_raw'))
        norm['normalized_minus_raw_mrr'] = norm.mrr_mean-norm.mrr_mean_raw
        norm[['representation','relation','source','normalized_minus_raw_mrr']].to_csv(dest / 'block_normalization.csv',index=False)
        text = conclusion_text(compact, summary, ps, norm, checkpoint)
        (dest / 'report.md').write_text(text,encoding='utf-8')
    return dest


def conclusion_text(compact, retrieval, probes, norm, checkpoint):
    columns = ['representation','dimension','relation','source','mrr_mean','mrr_std','recall@10_mean','recall@20_mean','recall@50_mean','delta_vs_sgns_input','delta_vs_ppmi_svd']
    findings = []
    primary = probes[((probes.task.isin(TASKS[:4])) & (probes.metric=='macro_f1')) | ((probes.task=='mana') & (probes.metric=='mae'))]
    scores = primary.pivot(index='task',columns='representation',values='mean')
    for name in FUSED:
        better = []
        for task, row in scores.iterrows():
            if ((row[name] < min(row[PRIMARY[0]],row[PRIMARY[1]])) if task=='mana'
                    else (row[name] > max(row[PRIMARY[0]],row[PRIMARY[1]]))):
                better.append(task)
        findings.append(f'`{name}` beats both single-source means on: {", ".join(better) or "none of the primary probes"}. These are descriptive differences, not significance tests.')
        rows = retrieval[retrieval.representation==name]
        for _, row in rows.iterrows():
            findings.append(f'{name}, {row.relation}/{row.source}: MRR {row.mrr_mean:.4f}; '
                            f'Δ SGNS {row.delta_vs_sgns_input:+.4f}, Δ SVD {row.delta_vs_ppmi_svd:+.4f}.')
    gains = norm.normalized_minus_raw_mrr
    findings.append(f'Block normalization raises mean MRR in {int((gains>0).sum())}/{len(gains)} fusion/relation strata, '
                    f'lowers it in {int((gains<0).sum())}, and ties in {int((gains==0).sum())}; its effect is relation-specific.')
    findings.append('Probe improvements over both sources support complementary usable information for those frozen targets. '
                    'They do not isolate the benefit of context from added width; no context-only fusion control or width-matched random control was added.')
    manifest = g.read_json(Path(checkpoint) / 'manifest.json')
    blocks = ['# v3.2 conclusions and downstream handoff',
        '## Established',
        'All corresponding seeds were kept separate. The tables report means and sample SD across training seeds, after averaging the three frozen card splits for probes. Baselines reuse completed v3.1 results. Deterministic retrieval baselines have no seed SD (N/A).',
        '### Frozen probes (classification macro-F1 ↑; mana MAE ↓)', g.markdown_table(compact.reset_index()),
        '\n\n'.join(findings),
        '### Relation-specific retrieval', g.markdown_table(retrieval[columns].fillna('N/A')),
        'Curated relation sets are small and concept subsets overlap. Structural and cross-format rows are proxies; incidence-derived associations favor the incidence baseline by construction. No universal score is used.',
        '### Block normalization diagnostic (normalized minus raw MRR)',
        g.markdown_table(norm[['representation','relation','source','normalized_minus_raw_mrr']]),
        'Raw fusion is a retrieval diagnostic only; it does not change the exported primary definition. See redundancy.csv for fixed-sample CKA and pairwise cosine correlations; geometric difference alone does not prove useful information.',
        f'Checkpointed all seven representations per available seed under `{Path(checkpoint).name}/`: raw SGNS input/context and PPMI-SVD (128d each), normalized fusion (256d/384d), and raw fusion diagnostics (256d/384d). All share {manifest["vocabulary_size"]:,} canonical rows for this frozen corpus; the loader verifies actual manifest size, shapes, finite values and hashes.',
        '## Not established',
        'Static cosine is not ground-truth Magic synergy. Best static MRR does not establish the best deck recommender. Increased dimension is not isolated as the cause of any improvement. These results do not establish that embeddings should remain frozen downstream. The legacy random probe baseline stops at 256 features; it is excluded from fusion comparisons and cannot serve as a matched 384d control.',
        '## Downstream architecture',
        'SGNS input + SGNS context + PPMI-SVD → concatenate → trainable projection → deck encoder. Use `matrix.shape[1]` for the projection input width and train it end-to-end with deck objectives, never static probe labels.',
        'Carry both 256d and 384d normalized fusion into downstream ablations, alongside SGNS input, PPMI-SVD, and random initialization. Keep all seeds and all alternatives. Seed 42 is a conventional development default, not test-selected.',
        'The decisive next experiment is downstream: masked/partial deck completion, convergence speed, data efficiency, rare-card performance, cross-format transfer, and temporal/new-card generalization. Compare frozen and fine-tuned source features. These experiments are outside this notebook.',
        '## Artifact audit',
        'The old v3.1 aggregate snapshots were stale: 77 retrieval runs and 42 representation probe runs are complete. completed_v3_1 regenerates their aggregate tables without modifying any historical directory. The notebook accepted-context count (1,002,916) differs by one from the packed training corpus and incidence index (1,002,915); the manifest uses the latter. No new SGNS models, corpus, SVD, or learned fusion weights were trained.']
    return '\n\n'.join(blocks)+'\n'
