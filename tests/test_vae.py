import pytest
import torch

from mtgdeck.card2vec import build_card2vec_corpus, embedding_matrix, train_card2vec
from mtgdeck.data import build_vocabulary, deck_to_tokens, mask_deck, normalize_card_name, normalize_deck_record
from mtgdeck.recommend import fit_baselines, ndcg_at_k, recall_at_k, recommend_baseline
from mtgdeck.vae import (
    Card2VecAttentionVAE,
    collate_token_rows,
    kl_beta,
    mask_present_logits,
    recommend_cards,
    recommend_cards_batch,
    score_cards_batch,
    vae_loss,
)


def make_model():
    torch.manual_seed(42)
    return Card2VecAttentionVAE(
        torch.randn(11, 12), model_dim=24, num_heads=4, num_layers=2,
        ff_dim=48, latent_dim=7, num_pool_queries=3, num_decoder_queries=4,
        role_dim=3, quantity_dim=4, dropout=0.0,
    )


def test_shapes_masks_losses_backward_and_optimizer_step():
    model = make_model()
    ids, roles, quantities, padding = collate_token_rows([
        ([2, 3, 4], [1, 1, 12], [1, 0, 0]),
        ([5, 6], [1, 1], [1, 0]),
    ])
    assert ids.shape == padding.shape == (2, 3)
    assert padding[1, 2]
    features = model.token_features(ids, roles, quantities)
    assert features.shape == (2, 3, 19)
    outputs = model(ids, roles, quantities, padding)
    assert outputs["pooled"].shape == (2, 3, 24)
    assert outputs["mu"].shape == outputs["logvar"].shape == outputs["latent"].shape == (2, 7)
    assert outputs["decoder_queries"].shape == (2, 4, 12)
    assert outputs["logits"].shape == (2, 11)
    assert model.logit_scale.exp().item() == pytest.approx(10.0)
    candidate_logits = mask_present_logits(outputs["logits"], ids, padding)
    assert torch.isneginf(candidate_logits[0, 2:5]).all()
    assert not torch.isneginf(candidate_logits[1, 0])

    targets = torch.zeros_like(outputs["logits"])
    targets[0, [7, 8]] = 1
    targets[1, 9] = 1
    total, recommendation, kl = vae_loss(outputs, targets, beta=0.01)
    assert all(torch.isfinite(value) for value in (total, recommendation, kl))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
    optimizer.step()
    assert model.logit_scale.grad is not None


def test_encoder_is_permutation_invariant_without_positional_embeddings():
    model = make_model().eval()
    ids = torch.tensor([[2, 3, 4, 5]])
    roles = torch.tensor([[1, 0, 0, 0]])
    quantities = torch.tensor([[1, 1, 2, 1]])
    padding = torch.zeros_like(ids, dtype=torch.bool)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        mu_a, logvar_a, pooled_a = model.encode(ids, roles, quantities, padding)
        mu_b, logvar_b, pooled_b = model.encode(ids[:, permutation], roles[:, permutation], quantities[:, permutation], padding[:, permutation])
    assert torch.allclose(pooled_a, pooled_b, atol=1e-5)
    assert torch.allclose(mu_a, mu_b, atol=1e-5)
    assert torch.allclose(logvar_a, logvar_b, atol=1e-5)


def test_deterministic_latent_ablation_uses_mu_and_zero_kl_loss():
    model = Card2VecAttentionVAE(
        torch.randn(11, 12), model_dim=24, num_heads=4, num_layers=1,
        ff_dim=48, latent_dim=7, num_pool_queries=2, num_decoder_queries=2,
        role_dim=3, quantity_dim=4, dropout=0.0, variational=False,
    ).train()
    ids, roles, quantities, padding = collate_token_rows([([2, 3, 4], [1, 1, 1], [1, 0, 0])])
    first = model(ids, roles, quantities, padding)
    second = model(ids, roles, quantities, padding)
    targets = torch.zeros_like(first["logits"]); targets[0, 7] = 1
    total, recommendation, kl = vae_loss(first, targets, beta=1.0, include_kl=False)

    assert torch.equal(first["latent"], first["mu"])
    assert torch.equal(first["latent"], second["latent"])
    assert kl.item() == 0.0
    assert torch.equal(total, recommendation)


def test_kl_beta_warmup_reaches_the_configured_maximum():
    assert kl_beta(0, 5_000, 0.01) == 0.0
    assert kl_beta(2_500, 5_000, 0.01) == pytest.approx(0.005)
    assert kl_beta(5_000, 5_000, 0.01) == pytest.approx(0.01)
    assert kl_beta(10_000, 5_000, 0.01) == pytest.approx(0.01)


def test_896_dimensional_embeddings_drive_projection_and_decoder_shapes():
    model = Card2VecAttentionVAE(
        torch.randn(11, 896), model_dim=24, num_heads=4, num_layers=1,
        ff_dim=48, latent_dim=7, num_pool_queries=2, num_decoder_queries=2,
        role_dim=3, quantity_dim=4, dropout=0.0, freeze_card2vec=False,
    )
    ids, roles, quantities, padding = collate_token_rows([([2, 3], [1, 1], [1, 0])])
    outputs = model(ids, roles, quantities, padding)

    assert model.card_dim == 896
    assert model.input_projection.in_features == 896 + 3 + 4
    assert model.card_embedding.weight.requires_grad
    assert outputs["decoder_queries"].shape == (1, 2, 896)
    assert outputs["logits"].shape == (1, 11)


def test_top_k_inference_excludes_visible_cards():
    model = make_model()
    vocab = {"<PAD>": 0, "<UNK>": 1, **{f"card {i}": i + 2 for i in range(9)}}
    partial = normalize_deck_record({
        "id": "partial", "commanders": [{"n": "Card 0", "q": 1}],
        "mainboard": [{"n": "Card 1", "q": 1}, {"n": "Card 2", "q": 3}],
    }, "moxfield")
    ranked = recommend_cards(model, partial, vocab, k=4)
    assert len(ranked) == 4
    assert not {"card 0", "card 1", "card 2"} & {name for name, _ in ranked}


def test_batched_inference_applies_per_deck_candidate_masks():
    model = make_model()
    vocab = {"<PAD>": 0, "<UNK>": 1, **{f"card {i}": i + 2 for i in range(9)}}
    partials = [normalize_deck_record({
        "id": str(index),
        "commanders": [{"n": f"Card {index}", "q": 1}],
        "mainboard": [{"n": "Card 2", "q": 1}],
    }, "moxfield") for index in range(2)]
    allowed = torch.zeros((2, len(vocab)), dtype=torch.bool)
    allowed[0, vocab["card 7"]] = True
    allowed[1, vocab["card 8"]] = True

    scores = score_cards_batch(model, partials, vocab, allowed_masks=allowed)
    ranked = recommend_cards_batch(model, partials, vocab, k=4, allowed_masks=allowed)

    assert scores.shape == (2, len(vocab))
    assert [name for name, _ in ranked[0]] == ["card 7"]
    assert [name for name, _ in ranked[1]] == ["card 8"]


def test_end_to_end_deck_completion_smoke():
    decks = []
    packages = [
        ["Sol Ring", "Island", "Ponder", "Counterspell", "Rhystic Study"],
        ["Sol Ring", "Island", "Ponder", "Swan Song", "Mystic Remora"],
        ["Sol Ring", "Forest", "Cultivate", "Llanowar Elves", "Beast Within"],
        ["Sol Ring", "Forest", "Cultivate", "Nature's Lore", "Beast Within"],
        ["Sol Ring", "Swamp", "Demonic Tutor", "Reanimate", "Feed the Swarm"],
        ["Sol Ring", "Swamp", "Demonic Tutor", "Animate Dead", "Feed the Swarm"],
    ]
    for index, cards in enumerate(packages):
        decks.append(normalize_deck_record({
            "id": str(index),
            "commanders": [{"n": f"Commander {index // 2}", "q": 1}],
            "mainboard": [{"n": card, "q": 8 if card in {"Island", "Forest", "Swamp"} else 1} for card in cards],
        }, "moxfield"))

    c2v = train_card2vec(build_card2vec_corpus(decks), vector_size=8, window=20, min_count=1, epochs=3, workers=1)
    vocab = build_vocabulary(decks)
    baseline = fit_baselines(decks[:-1])
    visible, hidden = mask_deck(decks[-1], mask_ratio=0.4, seed=7)
    hidden_names = [normalize_card_name(item["name"]) for item in hidden]
    baseline_ranked = recommend_baseline(
        baseline,
        "global",
        [item["name"] for item in visible["mainboard"]],
        [item["name"] for item in visible["commanders"]],
        k=3,
    )
    assert 0 <= recall_at_k(baseline_ranked, hidden_names, 3) <= 1
    assert 0 <= ndcg_at_k(baseline_ranked, hidden_names, 3) <= 1

    model = Card2VecAttentionVAE(
        torch.from_numpy(embedding_matrix(c2v, vocab)), model_dim=16, num_heads=4,
        num_layers=1, ff_dim=32, latent_dim=5, num_pool_queries=2,
        num_decoder_queries=2, role_dim=2, quantity_dim=3, dropout=0.0,
    )
    ids, roles, quantities, padding = collate_token_rows([deck_to_tokens(visible, vocab)], vocab["<PAD>"])
    outputs = model(ids, roles, quantities, padding)
    outputs["logits"] = mask_present_logits(outputs["logits"], ids, padding)
    targets = torch.zeros_like(outputs["logits"])
    for name in hidden_names:
        if name in vocab:
            targets[0, vocab[name]] = 1
    total, recommendation, kl = vae_loss(outputs, targets, beta=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad(); total.backward(); optimizer.step()
    assert all(torch.isfinite(value) for value in (total, recommendation, kl))
    ranked = recommend_cards(model, visible, vocab, k=3)
    present = {normalize_card_name(item["name"]) for zone in ("commanders", "mainboard") for item in visible[zone]}
    assert len(ranked) == 3 and not present.intersection(name for name, _ in ranked)
