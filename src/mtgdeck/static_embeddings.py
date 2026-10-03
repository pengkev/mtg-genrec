"""Reproducible set-sampled Card2Vec experiments; no work occurs on import."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
import zlib

import numpy as np

from .card2vec import save_card2vec
from .data import cards_fingerprint, decklist_card_names, iter_jsonl, normalize_card_name
from .metadata import iter_oracle_cards


@dataclass(frozen=True)
class StaticConfig:
    dimensions: tuple[int, ...] = (128, 256, 512, 1024)
    subsampling: tuple[float, ...] = (0.0, 1e-3)
    epochs: int = 5
    negative: int = 10
    learning_rate: float = 0.025
    min_learning_rate: float = 0.0001
    min_count: int = 5
    seed: int = 42
    workers: int = 1
    pairs_per_context: int = 32
    pair_strategy: str = "fixed"  # or sqrt, capped at pairs_per_context
    max_context_size: int | None = 250
    min_context_size: int = 2
    split_seeds: tuple[int, ...] = (42, 43, 44)
    test_size: float = 0.25

    def __post_init__(self):
        if not self.dimensions or any(d < 1 for d in self.dimensions):
            raise ValueError("dimensions must be positive")
        if not self.subsampling or any(t < 0 for t in self.subsampling):
            raise ValueError("subsampling thresholds must be nonnegative")
        if min(self.epochs, self.negative, self.min_count, self.workers, self.pairs_per_context) < 1:
            raise ValueError("training counts and workers must be positive")
        if not 0 < self.min_learning_rate <= self.learning_rate:
            raise ValueError("learning-rate endpoints must be positive and decreasing")
        if self.pair_strategy not in {"fixed", "sqrt"}:
            raise ValueError("unknown pair strategy")
        if self.min_context_size < 2 or (self.max_context_size is not None and self.max_context_size < self.min_context_size):
            raise ValueError("invalid context-size range")
        if not self.split_seeds or not 0 < self.test_size < 1:
            raise ValueError("provide split seeds and a test fraction between zero and one")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_index(path):
    """Use project normalization; ambiguous identities remain unlabelled."""
    candidates = defaultdict(dict)
    for card in iter_oracle_cards(Path(path)):
        key = normalize_card_name(card.get("name", ""))
        if key:
            candidates[key][card.get("oracle_id", card.get("id", card["name"]))] = card
    unique = {key: next(iter(values.values())) for key, values in candidates.items() if len(values) == 1}
    collisions = {key: [c["name"] for c in values.values()] for key, values in candidates.items() if len(values) > 1}
    return unique, collisions


def normalized_context(row):
    if not isinstance(row, dict) or not isinstance(row.get("cards"), list):
        return None
    raw = row["cards"]
    if not all(isinstance(card, str) and card.strip() for card in raw):
        return None
    return sorted({name for card in raw if (name := normalize_card_name(card))})


def prepare_corpus(path, output, config, reviewed_exclusions=None):
    """Two streaming passes; packed int32 contexts avoid million Python lists.

    Unknown metadata/provenance never filters training. Invalid JSON is a hard
    error; malformed schemas, duplicates and size exclusions are audited.
    """
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Prepared corpus exists: {output}; choose a fresh artifact root")
    output.mkdir(parents=True, exist_ok=True)
    reviewed_exclusions = dict(reviewed_exclusions or {})
    before_hash = sha256_file(path)
    frequencies, raw_frequencies = Counter(), Counter()
    seen, accepted = set(), set()
    with (output / "context_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "size", "fingerprint", "status", "aliases_collapsed"])
        writer.writeheader()
        for row_id, row in enumerate(iter_jsonl(path)):
            cards = normalized_context(row)
            fp = cards_fingerprint(cards) if cards is not None else ""
            size = len(cards) if cards is not None else 0
            status = "accepted"
            if cards is None:
                status = "malformed"
            else:
                raw_frequencies.update(cards)
                if fp in seen:
                    status = "duplicate"
                elif size < config.min_context_size:
                    status = "too_small"
                elif config.max_context_size is not None and size > config.max_context_size:
                    status = "size_quarantine"
                elif fp in reviewed_exclusions:
                    status = "reviewed_non_deck"
                seen.add(fp)
            if status == "accepted":
                accepted.add(fp)
                frequencies.update(cards)
            writer.writerow(dict(row=row_id, size=size, fingerprint=fp, status=status,
                                 aliases_collapsed=len(row["cards"]) - size if cards is not None else 0))
    names = sorted(name for name, count in frequencies.items() if count >= config.min_count)
    if len(names) < 2:
        raise ValueError("Fewer than two training tokens remain")
    vocab = {name: i for i, name in enumerate(names)}
    offsets, fingerprints = [0], []
    with (output / "tokens.i32").open("wb") as handle:
        for row in iter_jsonl(path):
            cards = normalized_context(row)
            if cards is None:
                continue
            fp = cards_fingerprint(cards)
            if fp not in accepted:
                continue
            accepted.remove(fp)
            ids = np.array([vocab[name] for name in cards if name in vocab], dtype=np.int32)
            if len(ids) < 2:
                continue
            handle.write(ids.tobytes())
            offsets.append(offsets[-1] + len(ids))
            fingerprints.append(fp)
    if sha256_file(path) != before_hash:
        raise RuntimeError("Corpus changed during preparation; use a stable snapshot and rerun")
    if len(offsets) == 1:
        raise ValueError("No trainable contexts remain after min_count")
    # Vocabulary pruning can leave singleton contexts. Do not expose random,
    # never-trained vectors for tokens that only occur in those skipped rows.
    packed = np.memmap(output / "tokens.i32", dtype=np.int32, mode="r+")
    effective_counts = np.zeros(len(names), dtype=np.int64)
    for start in range(0, len(packed), 1_000_000):
        effective_counts += np.bincount(packed[start:start + 1_000_000], minlength=len(names))
    active = effective_counts > 0
    if not active.all():
        remap = np.cumsum(active) - 1
        for start in range(0, len(packed), 1_000_000):
            packed[start:start + 1_000_000] = remap[packed[start:start + 1_000_000]]
        packed.flush()
        names = [name for name, retain in zip(names, active) if retain]
    del packed
    np.save(output / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    np.save(output / "counts.npy", effective_counts[active])
    for filename, value in [("names.json", names), ("fingerprints.json", fingerprints),
                            ("raw_frequencies.json", raw_frequencies),
                            ("preparation.json", {"corpus": str(Path(path).resolve()), "sha256": before_hash,
                                                  "config": asdict(config), "contexts": len(fingerprints),
                                                  "reviewed_exclusions": reviewed_exclusions})]:
        (output / filename).write_text(json.dumps(value, indent=2), encoding="utf-8")
    return output


def load_prepared(path):
    path = Path(path)
    return (json.loads((path / "names.json").read_text(encoding="utf-8")),
            np.memmap(path / "tokens.i32", dtype=np.int32, mode="r"),
            np.load(path / "offsets.npy", mmap_mode="r"), np.load(path / "counts.npy"))


def load_saved_runs(root, config, metadata_path):
    """Resolve paths locally and reject mixing incompatible saved experiments."""
    root = Path(root)
    preparation = json.loads((root / "prepared/preparation.json").read_text())
    names = json.loads((root / "prepared/names.json").read_text())
    expected_config = json.loads(json.dumps(asdict(config)))
    metadata_hash = sha256_file(metadata_path)
    runs = []
    for dimension in config.dimensions:
        for threshold in config.subsampling:
            path = root / str(dimension) / f"sample_{threshold:g}"
            manifest = json.loads((path / "config.json").read_text())
            if (manifest["config"] != expected_config or manifest["preparation"] != preparation
                    or manifest["metadata_sha256"] != metadata_hash
                    or manifest["dimension"] != dimension or manifest["subsampling"] != threshold):
                raise ValueError(f"Saved run/config/metadata mismatch: {path}")
            vectors = np.load(path / "vectors.npy", mmap_mode="r")
            if vectors.shape != (len(names), dimension):
                raise ValueError(f"Saved vector/vocabulary shape mismatch: {path}")
            stats = json.loads((path / "stats.json").read_text())
            runs.append({**stats, "dimension": dimension, "subsampling": threshold, "path": str(path)})
    return runs


def recover_provenance(paths, prepared, reviewed_exclusions=None):
    """Exact normalized set matches only; multi-format/source matches retained.

    Labels come from record fields, never filenames, legality or guesses. Counts
    later use sets so copies in multiple files don't inflate prevalence.
    """
    fingerprints = json.loads((Path(prepared) / "fingerprints.json").read_text())
    lookup = {fp: i for i, fp in enumerate(fingerprints)}
    formats, sources, examples = defaultdict(set), defaultdict(set), {}
    matched = set()
    excluded = set(reviewed_exclusions or {})
    for path in paths:
        for record in iter_jsonl(path):
            if not isinstance(record, dict):
                continue
            cards = sorted({normalize_card_name(n) for n in decklist_card_names(record) if normalize_card_name(n)})
            fingerprint = cards_fingerprint(cards)
            if fingerprint in excluded:
                continue
            index = lookup.get(fingerprint)
            if index is None:
                continue
            fmt, source = str(record.get("format") or "").lower(), str(record.get("source") or "").lower()
            if fmt and fmt not in {"unknown", "other", "deckbox"} and not fmt.startswith("limited"):
                formats[fmt].add(index)
            if source and source not in {"unknown", "other"}:
                sources[source].add(index)
            matched.add(index)
            examples.setdefault(index, {k: record.get(k) for k in ("name", "format", "source", "url")})
    return {"formats": formats, "sources": sources, "examples": examples,
            "matched": len(matched), "total": len(fingerprints)}


class PairCorpus:
    """Re-iterable sampled two-token sentences, resampled every epoch.

    Gensim window=1 trains both directions. No serialization adjacency enters
    the objective. Sampling is with replacement and no self-pairs. Subsampling
    rejects proposed pairs without refilling the deck budget.
    """
    def __init__(self, prepared, config, threshold, epoch=0):
        self.names, self.tokens, self.offsets, self.counts = load_prepared(prepared)
        self.config, self.threshold, self.epoch = config, threshold, epoch

    def __iter__(self):
        # Separate seed/epoch coordinates: seed 42 epoch 1 must not reuse
        # the exact random stream of seed 43 epoch 0 in repeated experiments.
        rng = np.random.default_rng(np.random.SeedSequence([self.config.seed, self.epoch]))
        freq = self.counts / self.counts.sum()
        keep = np.ones(len(freq)) if not self.threshold else np.minimum(1, (np.sqrt(freq / self.threshold) + 1) * self.threshold / freq)
        for i in rng.permutation(len(self.offsets) - 1):
            ids = self.tokens[self.offsets[i]:self.offsets[i + 1]]
            n = len(ids)
            budget = self.config.pairs_per_context
            if self.config.pair_strategy == "sqrt":
                budget = min(budget, max(1, int(np.sqrt(n))))
            elif self.config.pair_strategy != "fixed":
                raise ValueError("pair_strategy must be fixed or sqrt")
            left = rng.integers(n, size=budget)
            right = rng.integers(n - 1, size=budget)
            right += right >= left
            a, b = ids[left], ids[right]
            accepted = (rng.random(budget) < keep[a]) & (rng.random(budget) < keep[b])
            for x, y in zip(a[accepted], b[accepted]):
                yield [self.names[x], self.names[y]]


def stable_hash(value):
    return zlib.crc32(value.encode("utf-8"))


def train_run(prepared, output, config, dimension, threshold, metadata_path, progress=False):
    from gensim.models import Word2Vec
    names, _, offsets, counts = load_prepared(prepared)
    run = Path(output) / str(dimension) / f"sample_{threshold:g}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "config.json").exists():
        raise FileExistsError(f"Run exists: {run}; choose a new artifact root to avoid overwriting")
    manifest = {"config": asdict(config), "dimension": dimension, "subsampling": threshold,
                "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in
                ("gensim", "numpy", "scipy", "pandas", "scikit-learn", "joblib")},
                "preparation": json.loads((Path(prepared) / "preparation.json").read_text()),
                "metadata": str(metadata_path), "metadata_sha256": sha256_file(metadata_path),
                "implementation_sha256": sha256_file(__file__)}
    (run / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    model = Word2Vec(vector_size=dimension, sg=1, hs=0, negative=config.negative,
                     window=1, shrink_windows=False, sample=0, min_count=1,
                     workers=config.workers, seed=config.seed, hashfxn=stable_hash,
                     alpha=config.learning_rate, min_alpha=config.min_learning_rate, sorted_vocab=1)
    model.build_vocab_from_freq(dict(zip(names, map(int, counts))))
    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        pairs = PairCorpus(prepared, config, threshold, epoch)
        alpha = config.learning_rate + (config.min_learning_rate - config.learning_rate) * epoch / config.epochs
        end_alpha = config.learning_rate + (config.min_learning_rate - config.learning_rate) * (epoch + 1) / config.epochs
        # total_examples is an upper bound for scheduling; explicit endpoints
        # preserve the same epoch-level learning-rate schedule across ablations.
        effective, raw = model.train(pairs, total_examples=(len(offsets) - 1) * config.pairs_per_context,
                                    epochs=1, start_alpha=alpha, end_alpha=end_alpha, compute_loss=True)
        loss = float(model.get_latest_training_loss())
        history.append({"epoch": epoch + 1, "loss": loss, "effective_tokens": effective,
                        "raw_tokens": raw, "loss_per_directed_pair": loss / max(effective, 1)})
        if progress:
            print(f"d={dimension} seed={config.seed} epoch={epoch + 1}/{config.epochs} "
                  f"elapsed={time.perf_counter() - started:.0f}s", flush=True)
    seconds = time.perf_counter() - started
    vectors = np.stack([model.wv[name] for name in names])
    np.save(run / "vectors.npy", vectors)
    save_card2vec(model, run / "model.gensim")
    (run / "history.json").write_text(json.dumps(history, indent=2))
    stats = {"dimension": dimension, "subsampling": threshold, "training_seconds": seconds,
             "embedding_bytes": vectors.nbytes, "serialized_bytes": (run / "vectors.npy").stat().st_size,
             "path": str(run)}
    (run / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


# Explicit small benchmarks, intentionally not mass-generated pseudo-labels.
COMBOS = [("Thassa's Oracle", "Demonic Consultation"), ("Thassa's Oracle", "Tainted Pact"),
          ("Niv-Mizzet, Parun", "Curiosity"), ("Niv-Mizzet, Parun", "Ophidian Eye"),
          ("Stuffy Doll", "Guilty Conscience")]
CONCEPTS = {
    "Burn": ["Lava Spike", "Lightning Bolt", "Rift Bolt", "Skewer the Critics", "Boros Charm", "Searing Blaze", "Eidolon of the Great Revel", "Monastery Swiftspear"],
    "Blink": ["Soulherder", "Ephemerate", "Restoration Angel", "Flickerwisp", "Momentary Blink", "Ghostly Flicker", "Teleportation Circle", "Conjurer's Closet"],
    "Storm": ["Grapeshot", "Tendrils of Agony", "Past in Flames", "Desperate Ritual", "Pyretic Ritual", "Manamorphose", "Baral, Chief of Compliance", "Goblin Electromancer"],
    "Artifacts": ["Tinker", "Urza, Lord High Artificer", "Goblin Welder", "Daretti, Scrap Savant", "Myr Battlesphere", "Blightsteel Colossus", "Thoughtcast", "Emry, Lurker of the Loch"],
    "Reanimator": ["Reanimate", "Animate Dead", "Exhume", "Entomb", "Buried Alive", "Unmarked Grave", "Persist", "Dance of the Dead"],
    "Control": ["Counterspell", "Supreme Verdict", "Wrath of God", "Teferi, Hero of Dominaria", "Jace, the Mind Sculptor", "Memory Deluge", "Cryptic Command", "Sphinx's Revelation"],
    "Spellslinger": ["Young Pyromancer", "Ponder", "Preordain", "Consider", "Third Path Iconoclast", "Murmuring Mystic", "Talrand, Sky Summoner", "Opt"],
}


def unit_vectors(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def retrieval(vectors, names):
    import pandas as pd
    unit = unit_vectors(vectors)
    lookup = {name: i for i, name in enumerate(names)}
    pairs = [("combo", a, b) for a, b in COMBOS for a, b in ((a, b), (b, a))]
    pairs += [("synergy", members[0], partner) for members in CONCEPTS.values() for partner in members[1:]]
    rows = []
    for kind, a, b in pairs:
        ia, ib = lookup.get(normalize_card_name(a)), lookup.get(normalize_card_name(b))
        row = {"kind": kind, "anchor": a, "partner": b, "covered": ia is not None and ib is not None}
        if row["covered"]:
            scores = unit @ unit[ia]
            scores[ia] = -np.inf
            # Conservative tie handling: every tied candidate shares worst rank.
            row.update(rank=int(np.sum(scores >= scores[ib])), cosine=float(scores[ib]))
        else:
            row.update(rank=np.nan, cosine=np.nan)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for k in (10, 20, 50):
        frame[f"recall@{k}"] = (frame["rank"] <= k).astype(float)
    frame["rr"] = 1 / frame["rank"]
    frame["rr"] = frame["rr"].fillna(0)
    return frame


def centroid_retrieval(vectors, names, seed=42):
    import pandas as pd
    unit, lookup = unit_vectors(vectors), {n: i for i, n in enumerate(names)}
    rng, rows, neighbors = np.random.default_rng(seed), [], []
    for concept, members in CONCEPTS.items():
        present = [lookup[normalize_card_name(n)] for n in members if normalize_card_name(n) in lookup]
        if len(present) < 4:
            rows.append({"concept": concept, "coverage": len(present) / len(members), "status": "insufficient members"})
            continue
        shuffled = rng.permutation(present)
        train, test = shuffled[:len(shuffled) // 2], shuffled[len(shuffled) // 2:]
        query = unit[train].mean(axis=0)
        scores = unit @ (query / max(np.linalg.norm(query), 1e-12))
        scores[train] = -np.inf
        for index in test:
            rank = int((scores >= scores[index]).sum())
            rows.append({"concept": concept, "card": names[index], "rank": rank,
                         "recall@10": float(rank <= 10), "recall@50": float(rank <= 50),
                         "coverage": len(present) / len(members), "status": "held-out"})
        candidates = np.flatnonzero(np.isfinite(scores))
        ranked = candidates[np.argsort(-scores[candidates])[:20]]
        neighbors += [{"concept": concept, "card": names[i], "cosine": float(scores[i])}
                      for i in ranked]
    return pd.DataFrame(rows), pd.DataFrame(neighbors)


def mana_value_audit(names, metadata, maximum=20.0):
    """Prespecified primary evaluation domain; never clip labels or training data.

    Excluded cards stay embedded. Keep the full audit so the domain restriction
    is visible, and reassess the bound when metadata changes.
    """
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("mana-value maximum must be finite and positive")
    rows = []
    for i, name in enumerate(names):
        value = metadata.get(name, {}).get("cmc")
        try:
            number = float(value) if value is not None and not isinstance(value, bool) else np.nan
        except (TypeError, ValueError, OverflowError):
            number = np.nan
        status = ("missing_or_invalid" if not np.isfinite(number) else
                  "negative" if number < 0 else "above_primary_range" if number > maximum else "included")
        rows.append({"index": i, "card": name, "mana_value": number, "status": status})
    return rows


def metadata_targets(names, metadata, mana_maximum=20.0):
    targets = {}
    for task, labels in [("color", list("WUBRG") + ["colorless"]),
                         ("type", ["creature", "instant", "sorcery", "artifact", "enchantment", "planeswalker", "land"])]:
        ids, values = [], []
        for i, name in enumerate(names):
            card = metadata.get(name)
            if not card or (task == "color" and "color_identity" not in card) or (task == "type" and not card.get("type_line")):
                continue
            colors = card.get("color_identity", [])
            value = [int((label in colors) if label != "colorless" else not colors) for label in labels] if task == "color" else [int(label in card["type_line"].lower()) for label in labels]
            ids.append(i)
            values.append(value)
        targets[task] = (np.array(ids, dtype=int), np.array(values, dtype=int), labels)
    mana = [row for row in mana_value_audit(names, metadata, mana_maximum) if row["status"] == "included"]
    targets["mana"] = (np.array([r["index"] for r in mana], dtype=int),
                       np.array([r["mana_value"] for r in mana], dtype=float), ["mana_value"])
    return targets


def usage_targets(prepared, provenance, min_contexts=100, min_observations=20, lift_threshold=2.0):
    """Observed format prevalence lift, not legality. Unobserved cards excluded."""
    names, tokens, offsets, _ = load_prepared(prepared)
    groups = {k: v for k, v in provenance["formats"].items() if len(v) >= min_contexts}
    if len(groups) < 2:
        return None
    known = set.union(*groups.values())
    total = np.zeros(len(names))
    counts = np.zeros((len(names), len(groups)))
    for row in known:
        total[tokens[offsets[row]:offsets[row + 1]]] += 1
    for col, rows in enumerate(groups.values()):
        for row in rows:
            counts[tokens[offsets[row]:offsets[row + 1]], col] += 1
    prevalence = counts / np.array([len(rows) for rows in groups.values()])
    background = total / len(known)
    lift = prevalence / np.maximum(background[:, None], 1e-12)
    ids = np.flatnonzero(total >= min_observations)
    return ids, ((lift[ids] >= lift_threshold) & (counts[ids] >= 5)).astype(int), list(groups)


def curated_targets(names):
    # Only curated members are examples. Other curated roles are the contrast
    # group; unlabelled vocabulary cards are NOT asserted to be negatives.
    lookup = {n: i for i, n in enumerate(names)}
    membership = defaultdict(set)
    for concept, members in CONCEPTS.items():
        for name in members:
            if normalize_card_name(name) in lookup:
                membership[lookup[normalize_card_name(name)]].add(concept)
    ids = np.array(sorted(membership), dtype=int)
    labels = list(CONCEPTS)
    return ids, np.array([[int(c in membership[i]) for c in labels] for i in ids]), labels


def frequency_buckets(counts):
    # Fixed absolute thresholds make bucket meanings comparable across runs.
    return np.array(["very_rare", "rare", "medium", "common", "very_common"])[np.digitize(counts, [10, 100, 1000, 10000])]


def probe_task(vectors, counts, target, task, config):
    """Frozen card-disjoint repeated splits, matched baselines and bucket scores."""
    import pandas as pd
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import f1_score, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    ids, y, labels = target
    if len(ids) < 8:
        return pd.DataFrame([{"task": task, "status": "insufficient labelled cards",
                              "baseline": "learned", "bucket": "all", "n": len(ids),
                              "macro_f1": np.nan, "micro_f1": np.nan, "mae": np.nan, "r2": np.nan}])
    rows = []
    for seed in config.split_seeds:
        # Stratify exact label combinations only if every stratum permits it.
        strata = None
        if task != "mana":
            _, classes, class_counts = np.unique(y, axis=0, return_inverse=True, return_counts=True)
            ntest = int(np.ceil(len(ids) * config.test_size))
            if class_counts.min() >= 2 and len(class_counts) <= min(ntest, len(ids) - ntest):
                strata = classes
        train, test = train_test_split(np.arange(len(ids)), test_size=config.test_size, random_state=seed, stratify=strata)
        random_features = np.random.default_rng(seed).normal(size=(len(ids), vectors.shape[1])).astype(np.float32)
        features = {"learned": vectors[ids], "frequency": np.log1p(counts[ids, None]),
                    "random": random_features, "majority": np.zeros((len(ids), 1))}
        for baseline, x in features.items():
            if task == "mana":
                estimator = DummyRegressor(strategy="mean") if baseline == "majority" else make_pipeline(StandardScaler(), Ridge(alpha=10))
                estimator.fit(x[train], y[train])
                pred = estimator.predict(x[test])
            else:
                predictions = []
                for column in range(y.shape[1]):
                    estimator = DummyClassifier(strategy="most_frequent") if baseline == "majority" or len(np.unique(y[train, column])) < 2 else make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
                    estimator.fit(x[train], y[train, column])
                    predictions.append(estimator.predict(x[test]))
                pred = np.array(predictions).T
            buckets = frequency_buckets(counts[ids[test]])
            for bucket in ["all", "very_rare", "rare", "medium", "common", "very_common"]:
                mask = np.ones(len(test), dtype=bool) if bucket == "all" else buckets == bucket
                if not mask.any():
                    continue
                truth, prediction = y[test][mask], pred[mask]
                row = dict(task=task, baseline=baseline, seed=seed, bucket=bucket, n=int(mask.sum()), status="ok")
                if task == "mana":
                    row.update(mae=mean_absolute_error(truth, prediction), r2=r2_score(truth, prediction) if mask.sum() > 1 else np.nan)
                else:
                    row.update(macro_f1=f1_score(truth, prediction, average="macro", zero_division=0), micro_f1=f1_score(truth, prediction, average="micro", zero_division=0))
                    for label, score, support in zip(labels, f1_score(truth, prediction, average=None, zero_division=0), truth.sum(axis=0)):
                        row[f"f1_{label}"] = score
                        row[f"support_{label}"] = int(support)
                rows.append(row)
    return pd.DataFrame(rows)


def diagnostics(vectors, names, counts, seed=42, sample_size=1000):
    import pandas as pd
    from scipy.stats import spearmanr
    rng, unit = np.random.default_rng(seed), unit_vectors(vectors)
    norms = np.linalg.norm(vectors, axis=1)
    ids = rng.choice(len(names), min(sample_size, len(names)), replace=False)
    neighbor_means, duplicates = [], []
    k = min(10, len(names) - 1)
    for start in range(0, len(ids), 64):
        chunk = ids[start:start + 64]
        scores = unit[chunk] @ unit.T
        scores[np.arange(len(chunk)), chunk] = -np.inf
        neighbor_means.extend(np.partition(scores, -k, axis=1)[:, -k:].mean(axis=1))
        for row, index in enumerate(chunk):
            matches = np.flatnonzero(scores[row] >= .9999)
            duplicates += [(names[index], names[j], float(scores[row, j])) for j in matches[:10]]
    a = rng.integers(len(names), size=10000)
    b = (a + rng.integers(1, len(names), size=len(a))) % len(names)
    random_cosines = (unit[a] * unit[b]).sum(axis=1)
    stats = {"frequency_norm_spearman": float(spearmanr(counts, norms).statistic),
             "frequency_neighborhood_spearman": float(spearmanr(counts[ids], neighbor_means).statistic),
             "zero_vectors": int((norms < 1e-10).sum()), "sampled_duplicate_pairs": len(duplicates)}
    return stats, pd.DataFrame({"card": names, "frequency": counts, "norm": norms}), pd.DataFrame({"card": np.array(names)[ids], "neighbor_cosine": neighbor_means}), random_cosines, pd.DataFrame(duplicates, columns=["card", "neighbor", "cosine"])


def source_associations(prepared, provenance, min_contexts=100, topk=20, exclusive=False):
    """Source-specific co-occurrence neighborhoods; not source-held-out models."""
    import pandas as pd
    from itertools import combinations
    names, tokens, offsets, _ = load_prepared(prepared)
    lookup = {name: i for i, name in enumerate(names)}
    groups = {k: set(v) for k, v in provenance["sources"].items()}
    if exclusive:
        multiplicity = Counter(row for rows in groups.values() for row in rows)
        groups = {k: {row for row in rows if multiplicity[row] == 1} for k, rows in groups.items()}
    sources = {k: v for k, v in groups.items() if len(v) >= min_contexts}
    anchors = [normalize_card_name(members[0]) for members in CONCEPTS.values()]
    neighborhoods, rows = {}, []
    for source, contexts in sources.items():
        associations = {lookup[a]: Counter() for a in anchors if a in lookup}
        occurrences = Counter()
        for row in contexts:
            ids = tokens[offsets[row]:offsets[row + 1]]
            for anchor in set(ids).intersection(associations):
                occurrences[anchor] += 1
                associations[anchor].update(int(i) for i in ids if i != anchor)
        for anchor, counts in associations.items():
            if occurrences[anchor] >= 10:
                neighborhoods[source, names[anchor]] = {i for i, _ in counts.most_common(topk)}
    for left, right in combinations(sources, 2):
        for anchor in anchors:
            a, b = neighborhoods.get((left, anchor)), neighborhoods.get((right, anchor))
            if a and b:
                rows.append({"source_a": left, "source_b": right, "anchor": anchor,
                             "jaccard": len(a & b) / len(a | b),
                             "exclusive": exclusive, "contexts_a": len(sources[left]),
                             "contexts_b": len(sources[right]),
                             "overlapping_contexts": len(sources[left] & sources[right])})
    return pd.DataFrame(rows)
