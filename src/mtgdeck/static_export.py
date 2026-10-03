"""Portable, immutable static card checkpoints. Loading requires only NumPy."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .artifacts import sha256_file

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / 'artifacts/card2vec/static_export_v1'
DEFAULT_SEED_POLICY = 'conventional fixed seed; not test-selected'
DIMENSIONS = {
    'sgns_input_128': 128, 'sgns_context_128': 128, 'ppmi_svd_128': 128,
    'concat_input_svd_256': 256, 'concat_input_context_svd_384': 384,
    'concat_input_svd_raw': 256, 'concat_input_context_svd_raw': 384,
}


def unit_rows(matrix):
    """L2-normalize each row; preserve zero rows as zero without mutating input."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


def fuse_sources(u, v, svd):
    sources = [np.asarray(x, dtype=np.float32) for x in (u, v, svd)]
    if any(x.ndim != 2 or x.shape != sources[0].shape or x.shape[1] != 128
           or not np.isfinite(x).all() for x in sources):
        raise ValueError('Expected three finite, aligned N x 128 source matrices')
    u, v, svd = sources
    un, vn, sn = map(unit_rows, sources)
    return dict(sgns_input_128=u, sgns_context_128=v, ppmi_svd_128=svd,
                concat_input_svd_256=np.concatenate((un, sn), axis=1),
                concat_input_context_svd_384=np.concatenate((un, vn, sn), axis=1),
                concat_input_svd_raw=np.concatenate((u, svd), axis=1),
                concat_input_context_svd_raw=np.concatenate((u, v, svd), axis=1))


def require_disjoint(output, protected):
    output = Path(output).resolve()
    for path in protected:
        path = Path(path).resolve()
        if output == path or output in path.parents or path in output.parents:
            raise ValueError(f'Output must be disjoint from protected artifacts: {path}')
    return output


def write_checkpoint(output, vocab, seeds, provenance, source_artifacts, protected=()):
    """Write once. ``seeds`` yields (seed, aligned source/fusion matrices)."""
    output = require_disjoint(output, protected)
    if output.exists():
        raise FileExistsError(f'Checkpoint already exists: {output}; choose a new version')
    if not vocab or len(set(vocab)) != len(vocab) or any(
            not isinstance(n, str) or not n or '\n' in n or '\r' in n for n in vocab):
        raise ValueError('Vocabulary must contain unique nonempty single-line canonical names')
    output.mkdir(parents=True)
    (output / 'vocab.txt').write_text('\n'.join(vocab) + '\n', encoding='utf-8', newline='\n')
    manifest = {**provenance, 'version': 'static_export_v1', 'dtype': 'float32',
                'vocabulary_size': len(vocab), 'vocabulary_hash': sha256_file(output / 'vocab.txt'),
                'training_seeds': [], 'default_seed': 42, 'default_seed_policy': DEFAULT_SEED_POLICY,
                'representations': {}}
    for seed, matrices in seeds:
        if seed in manifest['training_seeds'] or set(matrices) != set(DIMENSIONS):
            raise ValueError('Duplicate seed or incomplete representation set')
        manifest['training_seeds'].append(seed)
        directory = output / f'seed_{seed}'
        directory.mkdir()
        for name, dimension in DIMENSIONS.items():
            matrix = np.asarray(matrices[name], dtype=np.float32)
            if matrix.shape != (len(vocab), dimension) or not np.isfinite(matrix).all():
                raise ValueError(f'Invalid matrix: seed {seed}, {name}')
            path = directory / f'{name}.npy'
            np.save(path, matrix, allow_pickle=False)
            normalized = name.startswith('concat_') and not name.endswith('_raw')
            entry = manifest['representations'].setdefault(name, {
                'dimension': dimension,
                'normalization': 'per-card L2 on each source block; no final normalization; zero blocks stay zero'
                                 if normalized else 'raw source values; no normalization',
                'files': {}})
            entry['files'][str(seed)] = {'path': path.relative_to(output).as_posix(),
                'sha256': sha256_file(path), 'vocabulary_hash': manifest['vocabulary_hash'],
                'source_artifacts': source_artifacts[str(seed)]}
    if 42 not in manifest['training_seeds']:
        raise ValueError('Conventional default seed 42 is unavailable')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return manifest


@dataclass
class StaticCardEmbeddings:
    vocab: list[str]
    card_to_idx: dict[str, int]
    embedding_matrix: np.ndarray
    metadata: dict

    def index(self, canonical_card: str) -> int:
        """Exact canonical lookup; no silent unknown-card substitution."""
        try:
            return self.card_to_idx[canonical_card]
        except KeyError:
            raise KeyError(f'Card absent from frozen static vocabulary: {canonical_card!r}') from None


def load_static_card_embeddings(representation='concat_input_context_svd_384', seed=42,
                                artifact_root=None, verify_hashes=True):
    root = Path(artifact_root) if artifact_root is not None else DEFAULT_ROOT
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 'static_export_v1' or manifest.get('dtype') != 'float32':
        raise ValueError('Unsupported static checkpoint schema/dtype')
    if representation not in manifest['representations'] or seed not in manifest['training_seeds']:
        raise ValueError(f'Unavailable representation/seed: {representation}, {seed}')
    spec = manifest['representations'][representation]
    entry = spec['files'][str(seed)]
    path = (root / entry['path']).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Matrix path escapes checkpoint root')
    vocab_path = root / 'vocab.txt'
    if entry['vocabulary_hash'] != manifest['vocabulary_hash']:
        raise ValueError('Matrix vocabulary hash disagrees with manifest')
    if verify_hashes and (sha256_file(vocab_path) != manifest['vocabulary_hash']
                          or sha256_file(path) != entry['sha256']):
        raise ValueError('Static checkpoint hash mismatch')
    vocab = vocab_path.read_text(encoding='utf-8').splitlines()
    matrix = np.load(path, allow_pickle=False)
    if (len(vocab) != manifest['vocabulary_size'] or len(set(vocab)) != len(vocab)
            or any(not name for name in vocab)
            or matrix.shape != (len(vocab), spec['dimension'])
            or matrix.dtype != np.float32 or not np.isfinite(matrix).all()):
        raise ValueError('Invalid static vocabulary/matrix shape, dtype, or finite values')
    return StaticCardEmbeddings(vocab, {n: i for i, n in enumerate(vocab)}, matrix,
                                {**manifest, 'representation': representation, 'seed': seed})
