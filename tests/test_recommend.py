import pytest

from mtgdeck.data import normalize_deck_record
from mtgdeck.recommend import fit_baselines, ndcg_at_k, recall_at_k, recommend_baseline


def deck(deck_id, commander, cards):
    return normalize_deck_record({
        "id": deck_id,
        "commanders": [{"n": commander, "q": 1}],
        "mainboard": [{"n": name, "q": 1} for name in cards],
    }, "moxfield")


def test_rankers_exclude_present_cards_and_honor_k():
    index = fit_baselines([
        deck("1", "Alpha", ["Sol Ring", "A", "B"]),
        deck("2", "Alpha", ["Sol Ring", "A", "C"]),
        deck("3", "Beta", ["Sol Ring", "D", "E"]),
    ])
    ranked = recommend_baseline(index, "global", ["Sol Ring"], k=3)
    assert len(ranked) == 3
    assert "sol ring" not in {name for name, _ in ranked}


def test_commander_popularity_and_metrics_on_known_example():
    index = fit_baselines([
        deck("1", "Alpha", ["A", "Shared"]),
        deck("2", "Alpha", ["A", "Shared"]),
        deck("3", "Beta", ["B", "Shared"]),
    ])
    ranked = recommend_baseline(index, "commander", [], ["Alpha"], k=2)
    assert {name for name, _ in ranked} == {"a", "shared"}
    example = ["a", "x", "b"]
    assert recall_at_k(example, ["a", "b"], 2) == pytest.approx(0.5)
    assert ndcg_at_k(example, ["a", "b"], 3) == pytest.approx((1 + 1 / 2) / (1 + 1 / 1.5849625), rel=1e-5)


def test_parallel_baseline_matches_serial_baseline():
    decks = [
        deck("1", "Alpha", ["Sol Ring", "A", "B"]),
        deck("2", "Alpha", ["Sol Ring", "A", "C"]),
        deck("3", "Beta", ["Sol Ring", "D", "E"]),
    ]
    serial = fit_baselines(decks)
    parallel = fit_baselines(decks, n_jobs=2)

    assert parallel.global_counts == serial.global_counts
    assert parallel.commander_counts == serial.commander_counts
    assert parallel.cooccurrence == serial.cooccurrence
    assert parallel.vocabulary == serial.vocabulary
