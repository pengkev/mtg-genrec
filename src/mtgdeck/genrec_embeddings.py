"""Align immutable static card exports with GenRec's Oracle-ID vocabulary."""
from __future__ import annotations

import numpy as np

from .data import ORACLE_TOKEN_PREFIX, PAD_TOKEN, UNK_TOKEN, normalize_card_name


def align_static_embeddings(bundle, vocab, catalog, *, missing_policy="error", seed=42):
    """Return a fresh matrix and coverage audit; never mutate the source export.

    Missing identities either fail explicitly or get reproducible random vectors
    with the source's RMS scale. PAD/UNK stay zero. Random fallback rows also
    remain fixed in the frozen ablation, so coverage must accompany its results.
    """
    if missing_policy not in {"error", "random"}:
        raise ValueError("missing_policy must be 'error' or 'random'")
    if sorted(vocab.values()) != list(range(len(vocab))) or vocab.get(PAD_TOKEN) != 0:
        raise ValueError("Vocabulary must have contiguous indices and PAD at zero")
    source = bundle.embedding_matrix
    matrix = np.zeros((len(vocab), source.shape[1]), dtype=np.float32)
    missing = []
    for token, index in sorted(vocab.items(), key=lambda item: item[1]):
        if token in {PAD_TOKEN, UNK_TOKEN}:
            continue
        if not token.startswith(ORACLE_TOKEN_PREFIX):
            raise ValueError(f"Expected Oracle-ID token: {token}")
        card = catalog.resolve("", token[len(ORACLE_TOKEN_PREFIX):])
        if card is None:
            raise ValueError(f"Oracle identity absent from catalog: {token}")
        name = normalize_card_name(card["name"])
        row = bundle.card_to_idx.get(name)
        if row is None:
            missing.append({"token": token, "name": card["name"], "index": index})
        else:
            matrix[index] = source[row]
    if missing and missing_policy == "error":
        raise ValueError(f"{len(missing)} cards absent from static export: {missing[:10]}")
    if missing:
        rng = np.random.default_rng(seed)
        scale = float(np.sqrt(np.mean(np.square(source, dtype=np.float64))))
        matrix[[row["index"] for row in missing]] = rng.normal(
            0, scale, (len(missing), source.shape[1]))
    total = sum(token not in {PAD_TOKEN, UNK_TOKEN} for token in vocab)
    return matrix, {"cards": total, "covered": total - len(missing),
                    "missing_policy": missing_policy, "fallback_seed": seed,
                    "missing": missing, "dimension": source.shape[1],
                    "representation": bundle.metadata["representation"],
                    "seed": bundle.metadata["seed"],
                    "vocabulary_hash": bundle.metadata["vocabulary_hash"],
                    "matrix_sha256": bundle.metadata["representations"][bundle.metadata["representation"]]
                        ["files"][str(bundle.metadata["seed"])]["sha256"]}


def genrec_optimizer(model, learning_rate=2e-4, embedding_learning_rate=2e-6):
    """Fine-tune the shared encoder/decoder table at a separate, lower LR."""
    import torch

    if not 0 < embedding_learning_rate <= learning_rate:
        raise ValueError("Require 0 < embedding_learning_rate <= learning_rate")
    embedding = model.card_embedding.weight
    groups = [{"params": [p for p in model.parameters() if p.requires_grad and p is not embedding],
               "lr": learning_rate, "name": "model"}]
    if embedding.requires_grad:
        groups.append({"params": [embedding], "lr": embedding_learning_rate,
                       "weight_decay": 0.0, "name": "card_embeddings"})
    return torch.optim.AdamW(groups, lr=learning_rate)
