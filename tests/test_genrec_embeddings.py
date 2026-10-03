from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mtgdeck.genrec_embeddings import align_static_embeddings, genrec_optimizer
from mtgdeck.legality import OracleCatalog
from mtgdeck.vae import Card2VecAttentionVAE, collate_token_rows, mask_present_logits, vae_loss
from mtgdeck.inference import _build_model, available_checkpoints


def fixture():
    source = np.arange(2 * 384, dtype=np.float32).reshape(2, 384) / 1000
    bundle = SimpleNamespace(embedding_matrix=source, card_to_idx={'fire': 0, 'sol ring': 1},
        metadata={'representation': 'fusion', 'seed': 42, 'vocabulary_hash': 'vocab-hash',
                  'representations': {'fusion': {'files': {'42': {'sha256': 'matrix-hash'}}}}})
    catalog = OracleCatalog([{'oracle_id': oid, 'name': name} for oid, name in
                            [('a', 'Sol Ring'), ('b', 'Fire // Ice'), ('c', 'Missing')]])
    vocab = {'<PAD>': 0, '<UNK>': 1, 'oid:a': 2, 'oid:b': 3, 'oid:c': 4}
    return bundle, vocab, catalog


def test_alignment_cold_start_and_immutable_source():
    bundle, vocab, catalog = fixture()
    original = bundle.embedding_matrix.copy()
    with pytest.raises(ValueError, match='1 cards absent'):
        align_static_embeddings(bundle, vocab, catalog)
    matrix, audit = align_static_embeddings(bundle, vocab, catalog, missing_policy='random')
    assert np.array_equal(matrix[2], original[1])
    assert np.array_equal(matrix[3], original[0])
    assert not matrix[:2].any()
    assert matrix[4].any()
    assert audit['covered'] == 2 and audit['missing'][0]['token'] == 'oid:c'
    again, _ = align_static_embeddings(bundle, vocab, catalog, missing_policy='random')
    assert np.array_equal(matrix, again)
    matrix[2] = 0
    assert np.array_equal(bundle.embedding_matrix, original)


@pytest.mark.parametrize('frozen', [True, False])
def test_fused_end_to_end_update_and_checkpoint_roundtrip(frozen):
    bundle, vocab, catalog = fixture()
    matrix, _ = align_static_embeddings(bundle, vocab, catalog, missing_policy='random')
    model = Card2VecAttentionVAE(torch.from_numpy(matrix.copy()), model_dim=16, num_heads=4,
                                num_layers=1, latent_dim=8, freeze_card2vec=frozen)
    before = model.card_embedding.weight.detach().clone()
    projection = model.input_projection.weight.detach().clone()
    optimizer = genrec_optimizer(model)
    assert [group['lr'] for group in optimizer.param_groups] == ([2e-4] if frozen else [2e-4, 2e-6])
    ids, roles, qty, padding = collate_token_rows([([2], [1], [1])])
    output = model(ids, roles, qty, padding)
    output['logits'] = mask_present_logits(output['logits'], ids, padding)
    output['logits'][:, :2] = -torch.inf
    target = torch.zeros(1, len(vocab)); target[0, 3] = 1
    loss, _, _ = vae_loss(output, target, beta=0.01)
    assert torch.isfinite(loss)
    loss.backward(); optimizer.step()
    assert torch.equal(before, model.card_embedding.weight) == frozen
    assert torch.equal(before[0], model.card_embedding.weight[0])
    assert not torch.equal(projection, model.input_projection.weight)
    checkpoint = {'state_dict': model.state_dict(), 'config': {'model_dim': 16, 'heads': 4,
                  'blocks': 1, 'latent_dim': 8}}
    restored = _build_model(checkpoint, torch.device('cpu'))
    model.eval()
    assert torch.allclose(model(ids, roles, qty, padding)['logits'],
                          restored(ids, roles, qty, padding)['logits'])


def test_discover_static_checkpoint(tmp_path):
    directory = tmp_path / 'pilot'
    directory.mkdir()
    path = directory / 'attention_oracleid_v2_static_fusion_variational_finetuned_384.pt'
    path.touch()
    assert path in available_checkpoints(tmp_path)
