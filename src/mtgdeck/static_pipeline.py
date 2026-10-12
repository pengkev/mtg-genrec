"""Explicit cached-data run: frozen deck sources, adapted text and Spellbook."""
from pathlib import Path
import importlib.metadata
import subprocess
import sys

from . import static_v4 as v4
from .static_export import load_static_card_embeddings

RUN_NAME = 'pipeline_synthetic_rules_difference_v1'


def build_plan(root=v4.ROOT, run_name=RUN_NAME):
    root = Path(root).resolve()
    research = root / 'artifacts/card2vec/static_v4_multiview_v1'
    run = research / run_name
    v4.require_disjoint(run, [research / 'oracle', research / 'spellbook',
                             research / 'mlm_synthetic_cards_v1', *v4.historical_roots()])
    plan = []

    def add(name, script, args, expected):
        output = run / name
        plan.append({'name': name, 'output': str(output), 'expected_config': expected,
                     'command': [sys.executable, str(root / 'scripts' / script),
                                 *map(str, args), '--output', str(output), '--allow-expensive']})

    add('semantic_rules', 'run_static_semantic.py',
        ['--stage', 'export', '--prepared', research / 'oracle', '--model',
         research / 'mlm_synthetic_cards_v1/model', '--view', 'rules_only', '--label', 'mlm',
         '--batch-size', 16, '--max-length', 256],
        {'stage': 'semantic_export', 'label': 'mlm', 'view': 'rules_only', 'max_length': 256})
    add('semantic_evaluation', 'run_static_semantic.py',
        ['--stage', 'evaluate', '--prepared', research / 'oracle', '--vectors', run / 'semantic_rules'],
        {'stage': 'semantic_evaluation'})
    add('mechanical_s42', 'run_static_spellbook.py',
        ['--stage', 'train', '--prepared', research / 'spellbook', '--seed', 42,
         '--dimension', 128, '--epochs', 3, '--batch-size', 64],
        {'stage': 'spellbook_train', 'seed': 42, 'dimension': 128, 'epochs': 3,
         'batch_size': 64, 'auxiliary': False})
    for split in ('validation', 'test'):
        add('mechanical_' + split, 'run_static_spellbook.py',
            ['--stage', 'evaluate', '--prepared', research / 'spellbook',
             '--trained', run / 'mechanical_s42', '--split', split, '--seed', 42],
            {'stage': 'spellbook_evaluation', 'split': split, 'deck_seed': 42})
    add('export', 'run_static_multiview.py',
        ['--stage', 'export', '--export', run / 'export', '--semantic',
         'mlm=' + str(run / 'semantic_rules'), '--mechanical', run / 'mechanical_s42', '--seed', 42],
        {'stage': 'multiview_export', 'deck_seed': 42, 'primary_semantic': 'mlm'})
    add('evaluation', 'run_static_multiview.py',
        ['--stage', 'evaluate', '--export', run / 'export', '--oracle', research / 'oracle',
         '--spellbook', research / 'spellbook', '--probes'],
        {'stage': 'multiview_evaluation', 'probes': True})
    return plan


def preflight(root=v4.ROOT):
    root = Path(root)
    research = root / 'artifacts/card2vec/static_v4_multiview_v1'
    missing = []
    for name in ('oracle', 'mlm_synthetic_cards_v1', 'spellbook'):
        path = research / name
        if not (path / 'complete.json').exists():
            missing.append(str(path))
        else:
            v4.validate_stage(path)
    for relative in ('artifacts/card2vec/static_export_v1/manifest.json',
                     'artifacts/card2vec/static_v2/evaluation_inputs.joblib'):
        if not (root / relative).exists():
            missing.append(str(root / relative))
    packages = {}
    for name in ('torch', 'transformers', 'accelerate', 'numpy', 'scipy', 'scikit-learn', 'joblib'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append('Python package: ' + name)
    if (root / 'artifacts/card2vec/static_export_v1/manifest.json').exists():
        for representation in ('sgns_input_128', 'sgns_context_128', 'ppmi_svd_128',
                               'concat_input_context_svd_384'):
            load_static_card_embeddings(representation, 42, root / 'artifacts/card2vec/static_export_v1')
    split_counts = {}
    if (research / 'spellbook/complete.json').exists():
        split_counts = v4.read_json(research / 'spellbook/overlap.json')['eligible_split_counts']
        for split in ('train', 'validation', 'test'):
            if not split_counts.get(split):
                missing.append('Spellbook eligible partition: ' + split)
    return {'ready': not missing, 'missing': missing, 'python': sys.executable,
            'packages': packages, 'spellbook_eligible_splits': split_counts,
            'training': 'reuse MLM and frozen SGNS/PPMI; train Spellbook only',
            'evaluation': 'five linear-probe tasks, frozen card splits 101/102/103; single model seed 42'}


def run_pipeline(plan, root=v4.ROOT):
    """Reuse matching completed stages; stop on incomplete or changed inputs."""
    status = preflight(root)
    if not status['ready']:
        raise ValueError('Preflight failed: ' + ', '.join(status['missing']))
    for stage in plan:
        output = Path(stage['output'])
        if output.exists():
            if not (output / 'complete.json').exists():
                raise FileExistsError(f'Incomplete stage: {output}; select a fresh run_name')
            provenance = v4.validate_stage(output)
            if any(provenance['config'].get(k) != value for k, value in stage['expected_config'].items()):
                raise ValueError(f'Completed stage configuration differs: {output}')
            if any(not Path(p).is_file() or v4.sha256_file(Path(p)) != digest
                   for p, digest in provenance['inputs'].items()):
                raise ValueError(f'Completed stage input lineage differs: {output}')
            print('Reuse:', output, flush=True)
        else:
            print('Run:', stage['name'], flush=True)
            subprocess.run(stage['command'], cwd=root, check=True)
            v4.validate_stage(output)
