"""Statistical, Card2Vec, and evaluation helpers for one held-out-card task."""

from __future__ import annotations

import math
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .card2vec import card2vec_scores
from .data import normalize_card_name


def deck_card_names(deck: Mapping, include_commanders: bool = True) -> set[str]:
    zones = ("commanders", "mainboard") if include_commanders else ("mainboard",)
    return {
        normalize_card_name(item["name"])
        for zone in zones
        for item in deck.get(zone, [])
        if normalize_card_name(item.get("name", ""))
    }


def commander_key(deck_or_names: Mapping | Iterable[str]) -> tuple[str, ...]:
    if isinstance(deck_or_names, Mapping):
        names = (item["name"] for item in deck_or_names.get("commanders", []))
    else:
        names = deck_or_names
    return tuple(sorted(normalize_card_name(name) for name in names if normalize_card_name(name)))


@dataclass
class BaselineIndex:
    global_counts: Counter
    commander_counts: dict[tuple[str, ...], Counter]
    cooccurrence: dict[str, Counter]
    vocabulary: set[str]


def _fit_cooccurrence_shard(
    flat_card_ids: np.ndarray,
    offsets: np.ndarray,
    card_names: Sequence[str],
    shard: int,
    shard_count: int,
) -> dict[str, Counter]:
    """Count pairs for a disjoint subset of source cards.

    The flat integer arrays are memmapped by Joblib for large corpora, avoiding a
    copy of the original nested deck dictionaries in every worker process.
    """

    cooccurrence: dict[str, Counter] = defaultdict(Counter)
    for row in range(len(offsets) - 1):
        ids = flat_card_ids[offsets[row] : offsets[row + 1]].tolist()
        names = [card_names[card_id] for card_id in ids]
        for card_id, card in zip(ids, names):
            if card_id % shard_count == shard:
                cooccurrence[card].update(names)
    for card, counts in cooccurrence.items():
        del counts[card]
    return dict(cooccurrence)


def fit_baselines(
    decks: Iterable[Mapping],
    n_jobs: int = 1,
    parallel_verbose: int = 0,
) -> BaselineIndex:
    global_counts: Counter = Counter()
    commander_counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)
    vocabulary: set[str] = set()
    card_rows: list[tuple[str, ...]] = []
    for deck in decks:
        main = deck_card_names(deck, include_commanders=False)
        key = commander_key(deck)
        global_counts.update(main)
        commander_counts[key].update(main)
        all_cards = main | set(key)
        vocabulary.update(all_cards)
        card_rows.append(tuple(sorted(all_cards)))

    if n_jobs == 1 or not card_rows:
        cooccurrence: dict[str, Counter] = defaultdict(Counter)
        for all_cards in card_rows:
            card_set = set(all_cards)
            for card in all_cards:
                cooccurrence[card].update(card_set - {card})
        return BaselineIndex(global_counts, dict(commander_counts), dict(cooccurrence), vocabulary)

    from joblib import Parallel, delayed, effective_n_jobs

    workers = effective_n_jobs(n_jobs)
    if workers == 1:
        cooccurrence: dict[str, Counter] = defaultdict(Counter)
        for all_cards in card_rows:
            card_set = set(all_cards)
            for card in all_cards:
                cooccurrence[card].update(card_set - {card})
        return BaselineIndex(global_counts, dict(commander_counts), dict(cooccurrence), vocabulary)

    card_names = tuple(sorted(vocabulary))
    card_ids = {name: index for index, name in enumerate(card_names)}
    offsets = np.zeros(len(card_rows) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(row) for row in card_rows])
    flat_card_ids = np.empty(int(offsets[-1]), dtype=np.int32)
    for row, (start, end) in zip(card_rows, zip(offsets[:-1], offsets[1:])):
        flat_card_ids[start:end] = [card_ids[name] for name in row]

    # More shards than workers balances popular-card skew while keeping each
    # returned dictionary small enough to merge without a large memory spike.
    shard_count = workers * 2
    partials = Parallel(
        n_jobs=workers,
        backend="loky",
        max_nbytes="1M",
        mmap_mode="r",
        pre_dispatch=workers,
        verbose=parallel_verbose,
    )(
        delayed(_fit_cooccurrence_shard)(flat_card_ids, offsets, card_names, shard, shard_count)
        for shard in range(shard_count)
    )
    cooccurrence = {}
    for partial in partials:
        cooccurrence.update(partial)
    return BaselineIndex(global_counts, dict(commander_counts), cooccurrence, vocabulary)


def global_popularity_scores(index: BaselineIndex) -> dict[str, float]:
    return {card: float(count) for card, count in index.global_counts.items()}


def commander_popularity_scores(index: BaselineIndex, commanders: Iterable[str]) -> dict[str, float]:
    counts = index.commander_counts.get(commander_key(commanders), Counter())
    return {card: float(count) for card, count in counts.items()}


def cooccurrence_scores(index: BaselineIndex, visible_cards: Iterable[str]) -> dict[str, float]:
    visible = {normalize_card_name(card) for card in visible_cards}
    scores: Counter = Counter()
    for card in visible:
        scores.update(index.cooccurrence.get(card, {}))
    return {card: float(score) for card, score in scores.items() if card not in visible}


def rank_scores(scores: Mapping[str, float], present_cards: Iterable[str] = (), k: int = 20, allowed_cards: set[str] | None = None) -> list[tuple[str, float]]:
    present = {normalize_card_name(card) for card in present_cards}
    ranked = []
    for raw_name, score in scores.items():
        name = normalize_card_name(raw_name)
        if name in present or (allowed_cards is not None and name not in allowed_cards) or not math.isfinite(float(score)):
            continue
        ranked.append((name, float(score)))
    return sorted(ranked, key=lambda item: (-item[1], item[0]))[:k]


def recommend_baseline(index: BaselineIndex, method: str, visible_cards: Iterable[str], commanders: Iterable[str] = (), k: int = 20) -> list[tuple[str, float]]:
    visible = list(visible_cards)
    if method == "global":
        scores = global_popularity_scores(index)
    elif method == "commander":
        scores = commander_popularity_scores(index, commanders)
    elif method == "cooccurrence":
        scores = cooccurrence_scores(index, [*visible, *commanders])
    else:
        raise ValueError("method must be global, commander, or cooccurrence")
    return rank_scores(scores, [*visible, *commanders], k)


def recommend_with_card2vec(model, visible_cards: Iterable[str], commanders: Iterable[str] = (), k: int = 20) -> list[tuple[str, float]]:
    all_visible = [*visible_cards, *commanders]
    return rank_scores(card2vec_scores(model, all_visible), all_visible, k)


def recall_at_k(ranked_cards: Sequence[str] | Sequence[tuple[str, float]], relevant_cards: Iterable[str], k: int) -> float:
    relevant = {normalize_card_name(card) for card in relevant_cards}
    if not relevant:
        return 0.0
    names = [normalize_card_name(item[0] if isinstance(item, tuple) else item) for item in ranked_cards[:k]]
    return len(set(names) & relevant) / len(relevant)


def ndcg_at_k(ranked_cards: Sequence[str] | Sequence[tuple[str, float]], relevant_cards: Iterable[str], k: int) -> float:
    relevant = {normalize_card_name(card) for card in relevant_cards}
    if not relevant:
        return 0.0
    names = [normalize_card_name(item[0] if isinstance(item, tuple) else item) for item in ranked_cards[:k]]
    dcg = sum(1.0 / math.log2(rank + 2) for rank, name in enumerate(names) if name in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def topk_overlap(first: Sequence[str], second: Sequence[str], k: int) -> dict[str, float]:
    a = {normalize_card_name(name) for name in first[:k]}
    b = {normalize_card_name(name) for name in second[:k]}
    union = a | b
    return {"overlap": len(a & b), "jaccard": len(a & b) / len(union) if union else 0.0}


def load_edhrec_benchmark(path: str | Path) -> list[dict]:
    """Load small, curated EDHREC snapshots without scraping during research runs."""

    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = ("commander", "partial_deck", "recommendations")
            if any(key not in row for key in required):
                raise ValueError(f"{path}:{line_number}: EDHREC snapshot is missing a required field")
            rows.append(row)
    return rows
