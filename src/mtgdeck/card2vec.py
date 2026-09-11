"""Gensim Card2Vec training and embedding-based deck completion."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .data import PAD_TOKEN, UNK_TOKEN, normalize_card_name


def build_card2vec_corpus(decks: Iterable[Mapping]) -> list[list[str]]:
    """Make one unordered, quantity-collapsed sentence per canonical deck."""

    corpus: list[list[str]] = []
    for deck in decks:
        cards = {
            normalize_card_name(item["name"])
            for zone in ("commanders", "mainboard")
            for item in deck.get(zone, [])
            if normalize_card_name(item.get("name", ""))
        }
        if cards:
            corpus.append(sorted(cards))
    return corpus


def train_card2vec(
    corpus: Iterable[Sequence[str]],
    vector_size: int = 896,
    window: int = 100,
    min_count: int = 2,
    epochs: int = 10,
    workers: int = 1,
    seed: int = 42,
):
    """Train skip-gram Word2Vec using a deck-wide context window."""

    from gensim.models import Word2Vec

    sentences = [list(sentence) for sentence in corpus]
    if not sentences:
        raise ValueError("Card2Vec corpus is empty")
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        workers=workers,
        epochs=epochs,
        seed=seed,
        sorted_vocab=1,
    )


def save_card2vec(model, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))


def load_card2vec(path: str | Path):
    from gensim.models import Word2Vec

    return Word2Vec.load(str(path))


def nearest_cards(model, card_name: str, topn: int = 10) -> list[tuple[str, float]]:
    key = normalize_card_name(card_name)
    if key not in model.wv:
        return []
    return [(name, float(score)) for name, score in model.wv.most_similar(key, topn=topn)]


def embedding_matrix(model, vocab: Mapping[str, int]) -> np.ndarray:
    """Align learned vectors with the project's explicit vocabulary indices."""

    matrix = np.zeros((len(vocab), model.wv.vector_size), dtype=np.float32)
    for name, index in vocab.items():
        if name not in (PAD_TOKEN, UNK_TOKEN) and name in model.wv:
            matrix[index] = model.wv[name]
    return matrix


def card2vec_scores(model, visible_cards: Iterable[str], candidates: Iterable[str] | None = None) -> dict[str, float]:
    """Score candidates by cosine similarity to the mean visible-card vector."""

    visible = {normalize_card_name(name) for name in visible_cards}
    vectors = [model.wv[name] for name in visible if name in model.wv]
    if not vectors:
        return {}
    query = np.mean(vectors, axis=0)
    norm = np.linalg.norm(query)
    if norm == 0:
        return {}
    query = query / norm
    names = list(candidates) if candidates is not None else list(model.wv.index_to_key)
    scores: dict[str, float] = {}
    for raw_name in names:
        name = normalize_card_name(raw_name)
        if name in visible or name not in model.wv:
            continue
        vector = model.wv[name]
        denominator = np.linalg.norm(vector)
        if denominator:
            scores[name] = float(np.dot(query, vector / denominator))
    return scores


def recommend_card2vec(model, visible_cards: Iterable[str], k: int = 20, candidates: Iterable[str] | None = None) -> list[tuple[str, float]]:
    scores = card2vec_scores(model, visible_cards, candidates)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
