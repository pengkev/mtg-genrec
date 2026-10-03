# Relation-aware static Card2Vec evaluation

The authoritative report remains `notebooks/card2vec_static_embeddings.ipynb`.
Its 51 original cells and saved outputs are preserved exactly; an opening research
map and a new relation-aware section distinguish historical results from pending
experiments. The new helper is `src/mtgdeck/static_relations.py`, with the curated
configuration in `configs/card2vec_relations.json`.

## Audit and methodological changes

Historical static_v3 has 42 completed SGNS evaluations, 22 pending baseline/SVD/
compatibility evaluations, and 24 pending representation probe evaluations. Its
semantic benchmark covers only two directed queries; complementary coverage is six.
Matrix baselines, SVD and compatibility have no completed historical results.
The implementation hashes are frozen, so the existing geometry implementation is
reused without modifying it or its historical artifacts.

The revised pipeline adds:

- Nine SGNS scorers including reverse cross dot and the normalized input/context
  sum; 128/256d, seeds 42/43/44, with 128d the reference.
- Curated similarity, substitution and mechanical complementarity; up to 24
  seed subsets per frozen concept; up to 1,500 structural proxy queries; frozen
  corpus association queries; disjoint observed-format-usage structural proxies.
- A schema for curated cross-format transfer, currently empty. No format legality
  or strategically valid transfer is inferred from usage labels alone.
- Strict vocabulary coverage: any missing vocabulary endpoint excludes the entire
  query. Metadata-derived labels require metadata; curated and frozen corpus labels
  can be evaluated without it, with missing metadata reported separately. Targets
  are never silently dropped.
- First-hit MRR, mean target reciprocal rank, Recall/Precision/AP/NDCG@10/20/50,
  mean/median target rank, query medians, seed SD and descriptive t intervals.
- Identical benchmark/metrics for incidence, incidence cosine, PMI, PPMI,
  128/256d PPMI-SVD cosine/dot and popularity-only retrieval.
- A d+1 parameter frozen diagonal compatibility probe with card-disjoint
  train/validation/test splits, excluded curated endpoints, regularization chosen
  on validation, and test AUC/AP versus cosine, symmetric cross dot and PPMI.
- Reusable hard negatives matched on partner color, broad type, mana bucket and
  frequency bucket. No observed co-occurrence is not proof of incompatibility.
  Matching coverage and easy/hard examples are saved; no SGNS sweep is launched.
- Query-level cross-seed top-20 Jaccard by relation/provenance/frequency, neighbor
  popularity, rarity-weighted recall, and representative side-by-side neighborhoods.
- Frozen linear probes for input/context/raw sum/unit input/unit context/normalized
  sum/PPMI-SVD, preserving per-class results and the existing mana exclusion policy.

Historical pair-average reciprocal rank is not the revised first-hit MRR. Neither
is silently relabelled. All queries use full-vocabulary retrieval with seed cards
excluded. Multi-seed queries average seed scores, including for cross-space and
matrix scorers. PMI therefore requires joint support with every seed. Unseen PMI
pairs receive no retrieval credit and worst candidate rank in rank summaries.
PPMI zeros use canonical name order for ties; tie counts are logged. Precision/AP
are label retrieval under incomplete judgments, not exhaustive Magic relevance.

Metadata proxies match only color identity, broad type and mana bucket; the frozen
metadata cache lacks rules text. They are never pooled with curated similarity or
substitution scores. More queries do not establish representativeness. Concept
subsets overlap, as do seed-pair stability comparisons. Three model seeds give wide,
unadjusted uncertainty. SVD randomness is a different replicate source than SGNS.

PMI is log2(Cij*N/(fi*fj)), with binary-context marginals and no smoothing. PPMI
clips at zero. The existing 128-row sparse shards and streamed SVD avoid a dense
33,623-square matrix. Full incidence and bounded SGNS pair sampling have different
context weighting, so comparisons cannot isolate training objective effects alone.

## Run instructions

From the repository root, using the project's Python environment:

```bash
python scripts/run_static_relations.py --stage prepare
python scripts/run_static_relations.py --stage sgns --allow-expensive
python scripts/run_static_relations.py --stage matrices --allow-expensive
python scripts/run_static_relations.py --stage baselines --allow-expensive
python scripts/run_static_relations.py --stage svd --allow-expensive
python scripts/run_static_relations.py --stage svd-evaluate --allow-expensive
python scripts/run_static_relations.py --stage compatibility --allow-expensive
python scripts/run_static_relations.py --stage probes --allow-expensive
python scripts/run_static_relations.py --stage report
```

The notebook flags wrap these commands: `RUN_RELATION_PREPARATION`,
`RUN_RELATION_RETRIEVAL`, `RUN_EXPENSIVE_BASELINES`, `RUN_COMPATIBILITY_PROBE`,
`RUN_REPRESENTATION_PROBES`, and `RUN_RELATION_REPORT`. All default to false.
No new SGNS training is implemented. Run SVD before probes to include SVD probes.
Rerun report after any new stage. Two BLAS threads are the default; use `--threads`
and `--max-disk-gib` to control resources. Full retrieval and sparse factorization
are intentionally computationally pending after this code revision.

## Artifacts and reproducibility

New outputs live in `artifacts/card2vec/static_v3_relations_v1/`:

- `provenance.json`, `policy.json`, `relations_manifest.json`: exact frozen corpus,
  model, metadata/evaluation cache and implementation/config fingerprints.
- `benchmark/queries.json`, `coverage.csv`, `availability.json`: benchmark and exclusions.
- `matrices/`, `svd/`: expensive sparse counts/PPMI and low-rank representations.
- `relations/<scorer>_d<dimension>_s<seed>/`: metrics, neighbors, qualitative outputs.
- `supervision/`: canonical split pairs and negative matching diagnostics.
- `compatibility_v1/`: weights, validation/test scores and parameter counts.
- `relation_probes/`: per-class, per-split representation probes and unavailability reasons.
- `reports/<state-hash>/`: append-only status, per-seed/summary metrics, relation table,
  query medians, frequency strata, stability and qualitative comparisons.

Completed stages are reused. Partial stages have no completion marker and raise
instead of silently overwriting. Move an incomplete stage aside before retrying.
Changing implementation/config requires a fresh `--output` directory. Historical
static_v1/v2/v3 destinations are protected. No old conclusion or number was changed;
no revised scorer winner or export representation is asserted before computation.
Retain input and context vectors until the relation-specific comparisons justify an
export choice for the later, separately implemented format-specific models.

## Metadata coverage finding

The initial strict-metadata audit generated 3,194 queries but excluded all 288
archetype subsets, because the frozen cache lacks metadata for some prominent
curated members. It is preserved in
`artifacts/card2vec/static_v3_relations_v1_strict_coverage_audit/`. The final policy
uses vocabulary coverage for curated/frozen-corpus labels, records every missing
metadata endpoint, and requires metadata only for metadata-derived labels. This
avoids conflating metadata absence with retrieval failure or silently shrinking
concept targets. No corpus or metadata cache is rebuilt. Linear probes still use
the existing metadata joins and mana exclusions.

## Validation and prepared benchmark

The final benchmark contains 3,194 vocabulary-covered queries: 20 similarity,
13 substitutability, 20 complementarity, 288 archetype subsets, 20 corpus
associations, 1,500 structural proxies and 1,333 cross-format usage proxies.
2,863 queries have complete metadata; 331 explicitly record missing metadata.
The benchmark and initial pending-status report have been generated. The new
notebook review cells were executed with every compute flag disabled; all original
cells and saved outputs were compared structurally and remain identical.

Validation: 42 tests passed across `test_static_relations.py`,
`test_static_geometry.py`, `test_static_experiment.py`, and
`test_static_embeddings.py`, including exact metric examples, sparse identities,
endpoint/pair leakage tests, validation-only model selection and append-only
report reuse. Full-corpus expensive scorer/probe runs remain pending.
