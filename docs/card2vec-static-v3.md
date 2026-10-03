# Static v3: geometry, compression and compatibility

The v3 section appended to `notebooks/card2vec_static_embeddings.ipynb` defaults to
reading saved outputs. Existing v2 cells and outputs are preserved. Computational
work lives in `src/mtgdeck/static_geometry.py`; the CLI never calls v1/v2 training,
preparation, or report writers.

## Inspection findings

- `static_embeddings.train_run` trains gensim SGNS and persists both `vectors.npy`
  and the full `model.gensim` through `card2vec.save_card2vec`.
- All six local v2 models retain `syn1neg`. Input vectors aligned by card name
  exactly match their exported arrays. Output vectors are aligned through
  `wv.key_to_index`, not assumed to follow packed vocabulary order. **No retraining
  is required.** Missing context vectors fail explicitly; no automatic replacement
  model is passed off as v2.
- `PairCorpus` samples 32 non-self pairs with replacement per context per epoch;
  two-token sentences train both directions. RNG coordinates are
  `SeedSequence([training_seed, epoch])`. Counts ignore card quantities.
- V2 quarantine combines structural checks, size limits, deduplication and reviewed
  title-plus-size exclusions. V3 reads the resulting packed corpus directly;
  it never reruns those rules over changing source files. The artifact contains
  **1,002,915 contexts and 33,623 cards**. Earlier source/audit counts are not
  substituted for the packed training population.
- V2 mechanics are input-vector cosine, both directions, pessimistic ties, at
  cutoffs 10/25/50. Its five pairs include Tainted Pact and Ophidian Eye and omit
  Painter/Grindstone and Depths/Stage. V3 keeps `mechanical_v2` unchanged and adds
  the requested five relationships as `complementary_mechanical`.
- V3 reuses frozen `evaluation_inputs.joblib`: metadata, labels, concepts,
  corpus-supported pairs, and stability anchors. It reuses `probe_long`,
  `expanded_centroids`, frequency buckets, normalization and hashing unchanged.
  No newer metadata snapshot silently alters the evaluation population.

## Execute

Use the repository's Python environment from the repository root. Audit first:

```bash
python scripts/run_static_geometry.py --stage audit
```

The following stages require explicit permission via the CLI flag. Stages run
sequentially with two BLAS threads by default (`--threads` overrides this).

```bash
# Existing models: input/context/combined geometry, retrieval and stability
python scripts/run_static_geometry.py --stage sgns --allow-expensive

# Full binary incidence and PPMI; default 12 GiB uncompressed shard budget
python scripts/run_static_geometry.py --stage matrices --allow-expensive --max-disk-gib 12
python scripts/run_static_geometry.py --stage baselines --allow-expensive

# Six randomized factorizations, then representation/scoring evaluation
python scripts/run_static_geometry.py --stage svd --allow-expensive
python scripts/run_static_geometry.py --stage svd-evaluate --allow-expensive

# Small frozen-vector association diagnostic; needs incidence shards
python scripts/run_static_geometry.py --stage compatibility --allow-expensive

# Expensive five-task probes for all completed card-vector representations
python scripts/run_static_geometry.py --stage probes --allow-expensive

# Regenerable report, then a new executed review-only archive
python scripts/run_static_geometry.py --stage report
python scripts/run_static_geometry.py --stage render
```

All outputs default to `artifacts/card2vec/static_v3/`. Every command accepts
`--output` for a separate experiment root; use the same root for every stage.
Historical directories and their parents/children are forbidden destinations.
Source hashes and implementation/policy snapshots are checked before each stage.
Completed model/evaluation directories are reused, never overwritten. An
interrupted directory lacks `complete.json`; move that incomplete directory
aside explicitly before retrying. Configuration/code changes require a fresh
output root. Report tables and plots are regenerable derived views. The executed
archive refuses overwrite; move it aside explicitly to render a newer review.

## Outputs

- `policy.json`, `provenance.json`, `model_inventory.json`: formulas,
  implementation/package versions, exact input hashes and vector availability.
- `matrices/`: sparse count/PPMI row shards and size/NNZ index.
- `svd/`: 128/256-dimensional vectors, singular values, seeds and runtime markers.
- `evaluations/`: identified scorer runs, pair ranks, per-query retrieval scores,
  centroid split results, fixed-anchor neighbors, unavailable-metric reasons.
- `probes/`: tidy per-split metrics, baseline comparisons, class support,
  frequency/type group details and explicit unavailable reasons.
- `compatibility_pairs.csv`, `compatibility/`: split pair supervision,
  learned diagonal weights, association validation scores or insufficiency reasons.
- `per_seed_metrics.csv`, `aggregate_metrics.csv`,
  `paired_dimension_differences.csv`: split-averaged model results and seed summaries.
- `mechanical_pairs.csv`, `semantic_pairs.csv`, `pair_results.csv`,
  `benchmark_coverage.csv`, `coverage_summary.csv`: inspectable benchmark results.
- `stability_pairs.csv`, `stability_by_frequency.csv`, plots: descriptive
  frequency-stratified neighborhood agreement, not independent seed-pair tests.
- `summary.json`, `report.md`, `notebook_executed.ipynb`: completion-aware review.

## Methodological and resource limits

Incidence uses one contribution per containing context per pair, unlike SGNS's
fixed 32 sampled pairs/context. This exposure-weight difference prevents a clean
causal attribution to objective versus compression. PMI uses binary context
marginals, not row sums of the pair matrix; no smoothing or negative-sampling
shift is applied. Diagonals are zero, absent PMI is negative infinity and absent
PPMI is zero. Pessimistic ranks include all tied candidate scores, excluding self.

Only 128 rows of count/PPMI products are materialized at once. The binary
context-card CSR and its transpose remain in RAM (roughly a gigabyte at this
corpus size, plus overhead). Sparse shards can still take gigabytes on disk.
SVD uses dense V×(d+10) workspaces, never V×V; several streaming passes per seed
trade disk I/O for bounded RAM. The disk budget refers to uncompressed sparse
storage, not a guarantee about operating-system peak RSS. Stage completion
markers record wall time; matrix metadata records major storage allocations.

SVD forms `U sqrt(S)` using three QR-normalized randomized power iterations and
10 oversamples. Cosine and Gram-dot retrieval answer different questions. For
an indefinite symmetric PPMI matrix, the Gram matrix is not its exact SVD
reconstruction. Factorization seeds measure algorithmic variability on one
fixed matrix, not independently sampled data or SGNS training.

The diagonal bilinear diagnostic deliberately has only d weights; it is less
expressive than a full W and depends on the embedding coordinate basis. All
benchmark endpoints are held out of scorer fitting/validation, including
corpus-supported endpoints. Embeddings and labels still use the same underlying
corpus. Positives are high-count/high-lift associations; zero-count negatives
are not proven incompatibilities. Balanced sampled pair validation does not
estimate deployment precision. Too little defensible supervision is explicitly
unavailable, not a claimed negative result. No MLP is trained on the combo pairs.

Three similar-role and five complementary pairs cannot establish a fundamental
semantic/compatibility distinction. Bidirectional scores are not independent
relationships. Corpus-supported partners favor incidence by construction.
Three model seeds are a small sample; card splits and correlated retrieval
cutoffs are not extra independent runs. Frequency-stability seed pairs overlap;
tie-breaking can inflate agreement in flat score regions. The later
format-specific recommendation architecture is outside this experiment.

## Completed initial review

The audit and all 42 SGNS scoring evaluations have run. Their report is explicitly
partial: matrix, SVD, diagnostic fitting, and probe stages remain opt-in.
See `artifacts/card2vec/static_v3/sgns_observations.md` for the initial measured
observations; the notebook interpretation section records them as partial findings.
