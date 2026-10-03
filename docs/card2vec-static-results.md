# Static Card2Vec research notes

## First run: `static_v1`

1,003,236 contexts; 33,635 tokens; dimensions 128/256/512/1024; five epochs;
32 proposed pairs/context; subsampling 0 versus 0.001; one training seed.
The original CSVs/report remain in `artifacts/card2vec/static_v1/`, with the
executed notebook archived there as `notebook_executed.ipynb`.

At subsampling 0.001:

| Dimension | Color macro F1 | Type macro F1 | Format macro F1 | Combo Recall@50 | Synergy MRR |
|---:|---:|---:|---:|---:|---:|
| 128 | .872 | .475 | .687 | .50 | .119 |
| 256 | .876 | .488 | .696 | .20 | .108 |
| 512 | .874 | .493 | .699 | .20 | .096 |
| 1024 | .874 | .493 | .699 | .20 | .095 |

- Color identity is strongly recoverable; exact combo retrieval is weaker.
- 256 is a provisional general-purpose candidate; retain 128 as a retrieval
  comparator. Little gain at 1024; no evidence yet for 2048/3072.
- Subsampling has no consistent benefit in this run.
- Very-rare-card type F1 at 256 is .206 versus .188 for random vectors.
- Archetype F1 .873 uses only 45 labelled cards and 12 test cards/split.
  Source Jaccard .395 includes overlapping contexts. Both are preliminary.
- **Original mana scores are unsuitable for ordinary-card conclusions.**
  Gleemax's mana value of 1,000,000 dominates fitting/scoring. A targeted corrected
  evaluation of the unchanged first-run vectors gives MAE 1.120 / 1.111 and
  R² .311 / .323 for 128 / 256 (no subsampling), versus constant MAE 1.380.
  These means use three card splits, not independent training seeds.

## Fixes and prevention

- Audit regression targets before fitting. Primary mana evaluation now uses
  finite values in [0, 20], with excluded identities/reasons saved separately.
  This is an explicit evaluation domain, not clipped labels or a training filter.
  Revisit the bound when metadata changes; don't select it by test performance.
- Size does not prove a context is a deck. Four explicitly named cubes have
  reviewed fingerprint exclusions, including the 209-card `Pauper Cube Archetypes`
  below the size cutoff. Preserve unknown-provenance contexts; never reject on
  missing labels or title keywords alone. Add reviewed evidence to
  `configs/card2vec_reviewed_exclusions.json` instead of editing source data.
- Re-evaluation excludes reviewed contexts from provenance labels. Existing
  embeddings still contain their original training influence; only new training
  applies the training quarantine. Format-score changes can reflect label changes.
- Source analysis now defaults to contexts exclusive to one known source and
  also saves the overlapping comparison. This is not source-held-out training.
- Keep ambiguous metadata joins excluded and report them separately from missing
  keys. Don't guess identities to improve apparent coverage.
- Report per-label support, small-test size, split variability, and benchmark
  coverage. Repeat training seeds before deciding tiny dimension differences.
- Preserve immutable training results. The notebook defaults to reading saved
  outputs. New training and evaluation are separate targeted stages, with their
  policies and targets saved in `static_v2`. Do not mix revised code with stale outputs.

## Second experiment: `static_v2`

- 128 and 256 dimensions; training seeds 42/43/44; no subsampling; five epochs;
  otherwise matched hyperparameters. Three card splits per training seed.
- Separate seed and epoch RNG coordinates prevent adjacent seeds from reusing
  one another's epoch sample streams.
- Conservative title-plus-size rules quarantined 321 additional contexts:
  258 cubes, 56 collections/pools and 7 aggregate/reference lists. Unknown
  provenance is retained. The original data and first-run models are unchanged.
- 244 manually selected concept memberships across 12 inspectable archetypes;
  actual vocabulary/metadata coverage is reported before classification.
- Tidy metric/value tables contain only applicable finite results. Undefined
  correlations have explicit reasons instead of NaNs or fabricated zeros.
- Report training-seed means/SDs and paired dimension intervals. Do not treat
  repeated card splits or pairwise stability comparisons as independent models.

### Measured second-run results

Three training seeds per dimension are complete. Values below are mean ± sample
SD across training seeds, after averaging the three card splits within each seed.

| Metric | 128 | 256 |
|---|---:|---:|
| Color macro F1 | .8708 ± .0007 | .8729 ± .0005 |
| Type macro F1 | .4752 ± .0022 | .4899 ± .0017 |
| Format macro F1 | .6874 ± .0007 | .6955 ± .0007 |
| Clean mana MAE | 1.1093 ± .0016 | 1.1003 ± .0009 |
| Archetype macro F1 | .8690 ± .0115 | .8646 ± .0080 |
| Corpus-supported partner MRR | .2082 ± .0020 | .2044 ± .0039 |
| Held-out centroid Recall@50 | .3568 ± .0117 | .2710 ± .0239 |

**Decision: no meaningful difference yet in the overall capacity choice.** This
is a trade-off, not a claim that every metric is equal. 256 has a clear practical
card-type improvement; 128 has stronger centroid retrieval and half the memory
(16.42 versus 32.83 MiB). Color's paired 95% interval includes zero. Three training
seeds are still a small sample; do not count correlated retrieval cutoffs as
independent evidence for a winner.

- Very-rare-card top-20 Jaccard is about .079 for both dimensions, versus .668 /
  .638 for very common cards. More dimensions do not fix rare-card instability.
- Descriptive frequency screens first qualify around 10 contexts for color,
  100 for format and neighborhood stability, and 1,000 for type. These depend on
  the stated thresholds and wide logarithmic bins; they are not guarantees.
  Type performance is nonmonotonic: the very-common bucket falls below .5 again.
  Class composition/support differs by bucket, so this does not show that more
  observations cause worse representations.
- Mechanical-combo incidence MRR is .384 versus SGNS cosine .034–.044. All five
  mechanical relationships reach the incidence top 50. Several SGNS failures
  have thousands of shared contexts: investigate representation/scoring rather
  than explaining them all as sparse data or absent rules-text understanding.
- Disjoint source agreement is .391 versus .395 with overlap. Exact duplicate
  removal changes little here, but this is still not source-held-out training.
- The expanded archetype population has 198 unique metadata-joined cards,
  15–22 per concept, and 2–8 positives per concept in each test split. It remains
  curated and may reflect color/type confounding as well as strategic roles.

Full evidence: `artifacts/card2vec/static_v2/report.md`,
`individual_seed_metrics.csv`, `paired_dimension_differences.csv`, and
`frequency_quality.png`. All headline metrics are finite. Only saved-result
review cells were executed in the notebook; its training/evaluation cells were
not executed. The six models were run through the targeted CLI instead.
