"""Compact Card2Vec attention-VAE for partial Commander deck completion."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .data import PAD_TOKEN, UNK_TOKEN, deck_to_tokens


class SelfAttentionBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(model_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim), nn.Dropout(dropout),
        )

    def forward(self, tokens: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, key_padding_mask=padding_mask, need_weights=False)
        tokens = tokens + attended
        return tokens + self.feed_forward(self.norm2(tokens))


class Card2VecAttentionVAE(nn.Module):
    """Denoise an unordered partial deck through a configurable latent bottleneck."""

    def __init__(
        self,
        card_embeddings: Tensor,
        model_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_dim: int = 512,
        latent_dim: int = 64,
        num_pool_queries: int = 4,
        num_decoder_queries: int = 4,
        role_dim: int = 8,
        quantity_dim: int = 16,
        max_quantity: int = 50,
        dropout: float = 0.1,
        freeze_card2vec: bool = True,
        initial_logit_scale: float = 10.0,
        variational: bool = True,
    ) -> None:
        super().__init__()
        weights = torch.as_tensor(card_embeddings, dtype=torch.float32)
        if weights.ndim != 2:
            raise ValueError("card_embeddings must have shape [vocab, embedding_dim]")
        self.vocab_size, self.card_dim = weights.shape
        self.model_dim = model_dim
        self.latent_dim = latent_dim
        self.num_pool_queries = num_pool_queries
        self.num_decoder_queries = num_decoder_queries
        self.max_quantity = max_quantity
        self.variational = variational

        self.card_embedding = nn.Embedding.from_pretrained(weights, freeze=freeze_card2vec, padding_idx=0)
        self.role_embedding = nn.Embedding(2, role_dim)
        self.quantity_embedding = nn.Embedding(max_quantity + 1, quantity_dim, padding_idx=0)
        self.input_projection = nn.Linear(self.card_dim + role_dim + quantity_dim, model_dim)
        self.blocks = nn.ModuleList(SelfAttentionBlock(model_dim, num_heads, ff_dim, dropout) for _ in range(num_layers))
        self.pool_queries = nn.Parameter(torch.empty(num_pool_queries, model_dim))
        nn.init.normal_(self.pool_queries, std=0.02)
        self.pool_attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(model_dim)
        self.to_mu = nn.Linear(num_pool_queries * model_dim, latent_dim)
        self.to_logvar = nn.Linear(num_pool_queries * model_dim, latent_dim)
        self.to_decoder_queries = nn.Linear(latent_dim, num_decoder_queries * self.card_dim)
        if initial_logit_scale <= 0:
            raise ValueError("initial_logit_scale must be positive")
        self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_logit_scale), dtype=torch.float32))

    def token_features(self, card_ids: Tensor, roles: Tensor, quantities: Tensor) -> Tensor:
        quantities = quantities.clamp(min=0, max=self.max_quantity)
        return torch.cat(
            [self.card_embedding(card_ids), self.role_embedding(roles), self.quantity_embedding(quantities)],
            dim=-1,
        )

    def encode(self, card_ids: Tensor, roles: Tensor, quantities: Tensor, padding_mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        tokens = self.input_projection(self.token_features(card_ids, roles, quantities))
        for block in self.blocks:
            tokens = block(tokens, padding_mask)
        queries = self.pool_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        summaries, _ = self.pool_attention(queries, tokens, tokens, key_padding_mask=padding_mask, need_weights=False)
        summaries = self.pool_norm(summaries)
        flat = summaries.flatten(1)
        return self.to_mu(flat), self.to_logvar(flat), summaries

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor, sample: bool = True) -> Tensor:
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.to_decoder_queries(latent).view(-1, self.num_decoder_queries, self.card_dim)
        queries = F.normalize(queries, dim=-1)
        vocabulary = F.normalize(self.card_embedding.weight, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        per_query = scale * torch.einsum("bqe,ve->bqv", queries, vocabulary)
        logits = torch.logsumexp(per_query, dim=1) - math.log(self.num_decoder_queries)
        return logits, queries

    def forward(self, card_ids: Tensor, roles: Tensor, quantities: Tensor, padding_mask: Tensor | None = None, sample: bool | None = None) -> dict[str, Tensor]:
        mu, logvar, pooled = self.encode(card_ids, roles, quantities, padding_mask)
        should_sample = self.variational and (self.training if sample is None else sample)
        latent = self.reparameterize(mu, logvar, sample=should_sample) if self.variational else mu
        logits, decoder_queries = self.decode(latent)
        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "latent": latent,
            "pooled": pooled,
            "decoder_queries": decoder_queries,
        }


def recommendation_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Multi-positive categorical loss; each hidden identity is a positive."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same [batch, vocab] shape")
    positive_count = targets.sum(dim=1).clamp_min(1.0)
    log_probabilities = F.log_softmax(logits, dim=-1)
    positive_log_probabilities = log_probabilities.masked_fill(~targets.bool(), 0.0)
    return -positive_log_probabilities.sum(dim=1).div(positive_count).mean()


def mask_present_logits(logits: Tensor, card_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
    """Remove encoder-visible cards from the discrete candidate distribution."""

    masked = logits.clone()
    for batch_index in range(card_ids.shape[0]):
        present = card_ids[batch_index]
        if padding_mask is not None:
            present = present[~padding_mask[batch_index]]
        masked[batch_index, present.unique()] = -torch.inf
    return masked


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def vae_loss(
    outputs: Mapping[str, Tensor],
    targets: Tensor,
    beta: float,
    include_kl: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    recommendation = recommendation_loss(outputs["logits"], targets)
    kl = kl_divergence(outputs["mu"], outputs["logvar"]) if include_kl else recommendation.new_zeros(())
    return recommendation + beta * kl, recommendation, kl


def kl_beta(step: int, anneal_steps: int, maximum: float = 1.0) -> float:
    return maximum if anneal_steps <= 0 else maximum * min(1.0, max(0.0, step / anneal_steps))


def collate_token_rows(rows: Sequence[tuple[Sequence[int], Sequence[int], Sequence[int]]], pad_id: int = 0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    max_length = max(len(row[0]) for row in rows)
    batch = len(rows)
    ids = torch.full((batch, max_length), pad_id, dtype=torch.long)
    quantities = torch.zeros((batch, max_length), dtype=torch.long)
    roles = torch.zeros((batch, max_length), dtype=torch.long)
    padding = torch.ones((batch, max_length), dtype=torch.bool)
    for index, (row_ids, row_quantities, row_roles) in enumerate(rows):
        length = len(row_ids)
        ids[index, :length] = torch.tensor(row_ids)
        quantities[index, :length] = torch.tensor(row_quantities)
        roles[index, :length] = torch.tensor(row_roles)
        padding[index, :length] = False
    return ids, roles, quantities, padding


@torch.no_grad()
def score_cards_batch(
    model: Card2VecAttentionVAE,
    partial_decks: Sequence[Mapping],
    vocab: Mapping[str, int],
    device: torch.device | str | None = None,
    allowed_masks: Tensor | Sequence[Sequence[bool]] | None = None,
) -> Tensor:
    """Score a batch while excluding visible and optionally illegal cards."""

    model.eval()
    target_device = torch.device(device) if device is not None else next(model.parameters()).device
    if not partial_decks:
        return torch.empty((0, len(vocab)), device=target_device)
    rows = [deck_to_tokens(deck, vocab) for deck in partial_decks]
    if any(not row[0] for row in rows):
        raise ValueError("partial decks must contain at least one vocabulary token")
    ids, roles, quantities, padding = (
        tensor.to(target_device) for tensor in collate_token_rows(rows, vocab.get(PAD_TOKEN, 0))
    )
    outputs = model(ids, roles, quantities, padding, sample=False)
    scores = mask_present_logits(outputs["logits"], ids, padding)
    for special in (PAD_TOKEN, UNK_TOKEN):
        if special in vocab:
            scores[:, vocab[special]] = -torch.inf
    if allowed_masks is not None:
        allowed = torch.as_tensor(allowed_masks, dtype=torch.bool, device=target_device)
        if allowed.ndim == 1:
            allowed = allowed.unsqueeze(0).expand(len(rows), -1)
        if allowed.shape != scores.shape:
            raise ValueError("allowed_masks must have shape [vocab] or [batch, vocab]")
        scores = scores.masked_fill(~allowed, -torch.inf)
    return scores


@torch.no_grad()
def recommend_cards_batch(
    model: Card2VecAttentionVAE,
    partial_decks: Sequence[Mapping],
    vocab: Mapping[str, int],
    k: int = 20,
    device: torch.device | str | None = None,
    allowed_masks: Tensor | Sequence[Sequence[bool]] | None = None,
) -> list[list[tuple[str, float]]]:
    scores = score_cards_batch(model, partial_decks, vocab, device, allowed_masks)
    inverse_vocab = {index: name for name, index in vocab.items()}
    requested = min(k, scores.shape[1])
    values, indices = torch.topk(scores, requested, dim=1)
    results: list[list[tuple[str, float]]] = []
    for row_values, row_indices in zip(values.cpu(), indices.cpu().tolist()):
        results.append([
            (inverse_vocab[index], float(value))
            for value, index in zip(row_values, row_indices)
            if torch.isfinite(value)
        ])
    return results


@torch.no_grad()
def recommend_cards(
    model: Card2VecAttentionVAE,
    partial_deck: Mapping,
    vocab: Mapping[str, int],
    k: int = 20,
    device: torch.device | str | None = None,
    allowed_mask: Tensor | Sequence[bool] | None = None,
) -> list[tuple[str, float]]:
    return recommend_cards_batch(model, [partial_deck], vocab, k, device, allowed_mask)[0]
