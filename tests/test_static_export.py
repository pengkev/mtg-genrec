"""Checkpoint contracts, including independent seed and normalized-block identity."""
import json
from pathlib import Path

import numpy as np
import pytest

from mtgdeck.static_export import (DEFAULT_SEED_POLICY, DIMENSIONS, fuse_sources,
    load_static_card_embeddings, require_disjoint, write_checkpoint)


@pytest.fixture
def checkpoint(tmp_path):
    names = ['alpha', 'beta', 'gamma']
    expected = {}
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        blocks = [rng.normal(size=(3, 128)).astype(np.float32) for _ in range(3)]
        blocks[1][0] = 0
        expected[seed] = fuse_sources(*blocks)
    path = tmp_path / 'static_export_v1'
    write_checkpoint(path, names, expected.items(), {'created_from': 'test'},
                     {str(s): [] for s in expected})
    return path, names, expected


def test_every_seed_round_trip_and_vocabulary(checkpoint):
    path, names, expected = checkpoint
    hashes = set()
    for seed, matrices in expected.items():
        for name, matrix in matrices.items():
            loaded = load_static_card_embeddings(name, seed, path)
            assert loaded.vocab == names
            assert loaded.card_to_idx == {'alpha': 0, 'beta': 1, 'gamma': 2}
            assert loaded.embedding_matrix.shape == (3, DIMENSIONS[name])
            assert loaded.embedding_matrix.dtype == np.float32
            assert np.isfinite(loaded.embedding_matrix).all()
            np.testing.assert_array_equal(loaded.embedding_matrix, matrix)
            assert loaded.index('beta') == 1
        hashes.add(loaded.metadata['representations']['sgns_input_128']['files'][str(seed)]['sha256'])
    assert len(hashes) == 3
    assert loaded.metadata['default_seed'] == 42
    assert loaded.metadata['default_seed_policy'] == DEFAULT_SEED_POLICY
    with pytest.raises(KeyError, match='Card absent'):
        loaded.index('absent card')
    with pytest.raises(ValueError, match='Unavailable'):
        load_static_card_embeddings(seed=99, artifact_root=path)


def test_normalized_blocks_exact_and_sources_unchanged():
    rng = np.random.default_rng(4)
    u, v, s = [rng.normal(size=(5,128)).astype(np.float32) * scale for scale in (1, 100, .01)]
    v[0] = 0
    copies = [x.copy() for x in (u,v,s)]
    result = fuse_sources(u,v,s)
    normalized = [x / np.where(np.linalg.norm(x,axis=1,keepdims=True)>0,
                              np.linalg.norm(x,axis=1,keepdims=True),1) for x in copies]
    np.testing.assert_array_equal(result['concat_input_svd_256'],np.concatenate([normalized[0],normalized[2]],axis=1))
    np.testing.assert_array_equal(result['concat_input_context_svd_384'],np.concatenate(normalized,axis=1))
    np.testing.assert_array_equal(result['concat_input_context_svd_raw'],np.concatenate(copies,axis=1))
    for original, copy in zip((u,v,s),copies):
        np.testing.assert_array_equal(original,copy)
    assert not np.allclose(result['concat_input_svd_256'],result['concat_input_svd_raw'])


@pytest.mark.parametrize('file', ['vocab.txt','seed_42/sgns_input_128.npy'])
def test_hash_corruption(checkpoint,file):
    path, _, _ = checkpoint
    with (path / file).open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_static_card_embeddings('sgns_input_128',42,path)


@pytest.mark.parametrize('kind', ['shape','nan','dtype','duplicate_vocab'])
def test_validation_without_hashes(checkpoint,kind):
    path, _, _ = checkpoint
    file = path / 'seed_42/sgns_input_128.npy'
    matrix = np.load(file)
    if kind=='shape': matrix = matrix[:-1]
    elif kind=='nan': matrix[0,0] = np.nan
    elif kind=='dtype': matrix = matrix.astype(np.float64)
    else: (path/'vocab.txt').write_text('alpha\nalpha\ngamma\n')
    np.save(file,matrix)
    with pytest.raises(ValueError,match='Invalid static'):
        load_static_card_embeddings('sgns_input_128',42,path,verify_hashes=False)


def test_immutable_and_protected(checkpoint,tmp_path):
    path, names, expected = checkpoint
    original = (path / 'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        write_checkpoint(path,names,expected.items(),{}, {})
    assert (path / 'manifest.json').read_bytes()==original
    historical = tmp_path / 'static_v3_relations_v1'
    historical.mkdir()
    marker = historical / 'historical.json'
    marker.write_text('frozen')
    for output in (historical,historical/'nested',tmp_path):
        with pytest.raises(ValueError,match='protected'):
            write_checkpoint(output,names,expected.items(),{}, {},protected=[historical])
    assert marker.read_text()=='frozen'
    link = tmp_path / 'alias'
    try:
        link.symlink_to(historical,target_is_directory=True)
    except OSError as exc:
        if getattr(exc, 'winerror', None) == 1314:
            pytest.skip('Windows symlink privilege unavailable; direct protection checks passed')
        raise
    with pytest.raises(ValueError):
        require_disjoint(link/'child',[historical])


def test_no_cross_seed_averaging(checkpoint):
    path, _, expected = checkpoint
    for seed in (42,43,44):
        actual = load_static_card_embeddings(seed=seed,artifact_root=path).embedding_matrix
        np.testing.assert_array_equal(actual,expected[seed]['concat_input_context_svd_384'])
        others = [expected[s]['concat_input_context_svd_384'] for s in (42,43,44)]
        assert not np.array_equal(actual,np.mean(others,axis=0))


def test_vocab_link_and_path_rejected(checkpoint):
    path, _, _ = checkpoint
    manifest = json.loads((path/'manifest.json').read_text())
    entry = manifest['representations']['sgns_input_128']['files']['42']
    entry['vocabulary_hash'] = 'bad'
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='vocabulary hash'):
        load_static_card_embeddings('sgns_input_128',42,path)
    entry['path'] = '../outside.npy'
    (path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='escapes'):
        load_static_card_embeddings('sgns_input_128',42,path)


def test_completed_repository_checkpoint():
    """Exercise real artifacts when available, without requiring binaries in CI."""
    root = Path(__file__).resolve().parents[1]
    path = root / 'artifacts/card2vec/static_export_v1'
    if not (path/'manifest.json').exists():
        pytest.skip('Versioned static checkpoint not present')
    canonical = json.loads((root/'artifacts/card2vec/static_v2/prepared/names.json').read_text())
    for seed in (42,43,44):
        loaded = {name: load_static_card_embeddings(name,seed,path) for name in DIMENSIONS}
        for bundle in loaded.values():
            assert bundle.vocab == canonical
            assert bundle.card_to_idx == {n:i for i,n in enumerate(canonical)}
        sources = [loaded[name].embedding_matrix for name in ('sgns_input_128','sgns_context_128','ppmi_svd_128')]
        expected = fuse_sources(*sources)
        for name, bundle in loaded.items():
            np.testing.assert_array_equal(bundle.embedding_matrix,expected[name])
        np.testing.assert_array_equal(sources[0],np.load(root/f'artifacts/card2vec/static_v2/seed_{seed}/128/sample_0/vectors.npy'))
        np.testing.assert_array_equal(sources[2],np.load(root/f'artifacts/card2vec/static_v3_relations_v1/svd/d128_s{seed}/vectors.npy'))
