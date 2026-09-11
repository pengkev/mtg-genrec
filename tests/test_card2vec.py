import inspect

import numpy as np

from mtgdeck.card2vec import (
    build_card2vec_corpus,
    embedding_matrix,
    nearest_cards,
    recommend_card2vec,
    train_card2vec,
)
from mtgdeck.data import build_vocabulary, normalize_deck_record


def tiny_decks():
    rows = [
        ("A", ["Sol Ring", "Island", "Counterspell"]),
        ("B", ["Sol Ring", "Island", "Ponder"]),
        ("C", ["Forest", "Llanowar Elves", "Cultivate"]),
        ("D", ["Forest", "Llanowar Elves", "Sol Ring"]),
    ]
    decks = []
    for deck_id, cards in rows:
        decks.append(normalize_deck_record({
            "id": deck_id,
            "mainboard": [{"n": card, "q": 12 if card == "Island" else 1} for card in cards],
            "commanders": [{"n": f"Commander {deck_id}", "q": 1}],
        }, "moxfield"))
    return decks


def test_default_embedding_dimension_is_896():
    assert inspect.signature(train_card2vec).parameters["vector_size"].default == 896


def test_corpus_collapses_quantities_but_canonical_records_do_not():
    decks = tiny_decks()
    corpus = build_card2vec_corpus(decks)
    assert corpus[0].count("island") == 1
    island = next(item for item in decks[0]["mainboard"] if item["name"] == "Island")
    assert island["quantity"] == 12


def test_tiny_training_lookup_neighbors_and_recommendations():
    decks = tiny_decks()
    model = train_card2vec(build_card2vec_corpus(decks), vector_size=12, window=20, min_count=1, epochs=8, workers=1)
    vocab = build_vocabulary(decks)
    matrix = embedding_matrix(model, vocab)
    assert matrix.shape == (len(vocab), 12)
    assert np.isfinite(matrix).all()
    assert nearest_cards(model, "Sol Ring", topn=2)
    recommendations = recommend_card2vec(model, ["Sol Ring"], k=3)
    assert len(recommendations) == 3
    assert all(name != "sol ring" for name, _ in recommendations)
