# Prepared rules-text, deck-difference and Spellbook run

`notebooks/card2vec_static_embeddings.ipynb` contains only the current cached-data
workflow: configuration/readiness, execution, and saved-result review.
Use the repository `.venv` kernel and Run All.
`RUN_V4_PIPELINE = True` remains enabled in the configuration cell. Set it to
false to perform readiness checks and saved-result review only.
Historical experiments and their saved outputs are preserved in
[`notebooks/archive/card2vec_static_embeddings.ipynb`](../notebooks/archive/card2vec_static_embeddings.ipynb).
Their training flags remain false; the pipeline stages and settings are unchanged.

The equivalent CLI prints readiness and commands without running by default:

```powershell
.venv/Scripts/python.exe scripts/run_static_v4_pipeline.py
.venv/Scripts/python.exe scripts/run_static_v4_pipeline.py --run
```

This run reuses the frozen seed-42 SGNS input/context and PPMI-SVD matrices from
`static_export_v1`, and the completed `mlm_synthetic_cards_v1/model` checkpoint.
It extracts rules-only mean-pooled 768d card vectors, trains the independent 128d
Spellbook set-completion/replacement model for three epochs, exports independent
sources and fixed fusions, and evaluates them. It does not retrain SGNS, SVD or
MLM, and does not jointly tune text/deck sources with the hypergraph objective.
The mechanical representation adds an independently learned source to the
combined representations; whether it improves them is an evaluation question.

The new output parent is
`artifacts/card2vec/static_v4_multiview_v1/pipeline_synthetic_rules_difference_v1/`.
Stages are write-once, with input/code/runtime hashes and completion manifests.
Matching completed stages are reused after integrity/configuration/input checks.
An incomplete stage or a changed input stops the run. Use a fresh
`--run-name pipeline_synthetic_rules_difference_v2` (or notebook
`V4_PIPELINE_NAME`) for a changed experiment; never delete/overwrite a failed
stage to disguise its history. The pipeline operates offline from cached data.

## Prepared data

The explicit Commander Spellbook snapshot/preparation stages completed using
the official [bulk dataset](https://json.commanderspellbook.com/variants.json.gz)
and [schema](https://backend.commanderspellbook.com/schema/?format=json).
Credit: [Commander Spellbook](https://commanderspellbook.com).
The immutable snapshot records API/build version 7.1.4 and its actual retrieval
timestamp. A bulk build timestamp is not a variant creation date.

Of 113,013 raw variants, 54,429 are eligible against the frozen Oracle identities:
51,877 training, 1,511 validation and 1,041 test. There are 54,358 variants with
unresolved cards; template/multiplicity/legality exclusions also apply and their
counts can overlap. These experiments describe the resolved eligible subset.
One connected component contains 41,663 eligible variants, so the partitions
are imbalanced. Official-family/exact-set groups remain together. The saved
audit found no cross-split exact, subset/superset or high-Jaccard pairs under its
checks. Shared card identities still make this transductive combo retrieval.

## Representations

SGNS and PPMI coordinate systems are independently learned. Fit an orthogonal
Procrustes rotation on the row-normalized frozen deck-vocabulary matrices,
without labels or evaluation scores. Define
`difference = unit(SGNS input) - unit(PPMI) @ rotation`. Residual norms at most
`1e-6` become zero. Save raw difference vectors and the rotation separately.
Alignment uses all frozen deck cards; it is explicitly transductive.

| Representation | Dimensions |
|---|---:|
| SGNS input, SGNS context, PPMI-SVD, aligned difference (separately) | 128 each |
| SGNS input + PPMI | 256 |
| SGNS input + PPMI + aligned difference | 384 |
| Historical SGNS input + context + PPMI | 384 |
| Adapted ModernBERT rules text | 768 |
| Spellbook mechanical | 128 |
| SGNS + PPMI + difference + text | 1,152 |
| SGNS + PPMI + difference + mechanical | 512 |
| SGNS + PPMI + difference + text + mechanical | 1,280 |

The existing deck/text/mechanical combinations remain comparisons. All fusions
normalize each declared source block independently; their definitions are fixed
before scoring. No test-picked weighting is learned. A difference derived from
two sources is not assumed to add independent information; its contribution is
measured against the two-source concatenation. Every source keeps its own raw
matrix and coverage mask, and missing rows are zero.

## Stages and evaluation

| Stage | Computation / saved results |
|---|---|
| `semantic_rules` | Rules-only extraction from adapted MLM; mean/first vectors, descriptions, coverage, lineage |
| `semantic_evaluation` | Saved synthetic/description retrieval; manually reviewed test remains unavailable |
| `mechanical_s42` | Three-epoch hyperedge completion and conditional replacement heads; losses and coverage |
| `mechanical_validation`, `mechanical_test` | Learned heads versus train-only incidence/popularity and frozen deck controls; full candidate pool |
| `export` | Separate raw sources, alignment rotation, masks, and fixed fusions |
| `evaluation` | Linear probes, relations, held-out Spellbook cosine retrieval, neighbors, geometry and cold-start simulation |

Linear probes retain the original targets and card splits 101/102/103: color,
card type, observed format usage, mana value and archetype. They retain the same
training-only scaling, logistic regression/ridge settings, frequency controls
and constant controls. The historical random control caps at 256d and is excluded
from these comparisons. `probe_summary.csv` reports primary macro-F1/MAE means
and split SD; `probes.csv` retains per-split, per-label and frequency/type groups.
This single-model-seed run cannot report independent training-seed SD.

MLM does not consume synthetic role-pair supervision. Contrastive-family
exclusions apply only to actual contrastive checkpoints. Curated complementarity
targets exposed with their query in training hyperedges are removed and logged.
`relation_summary.csv` keeps similarity/substitution, complementarity, association,
archetype and structural proxies separate. Coverage can differ between sources;
the full candidate pool and missing-source policies are recorded.

Spellbook family holdouts evaluate partner retrieval, multi-card set completion
and conditional replacements. `retrieval_summary.csv` separates models/tasks and
whether targets were train-seen. Learned head scores are distinct from the
static cosine results in `spellbook_cosine_summary.csv`. MRR and Recall@10/50
are descriptive evidence, not a guarantee of deck recommendation quality.
All detailed rows, unavailable tasks and exclusion reasons remain available.

The new notebook reviews completed saved reports after the run. Preparation and
offline checks have completed; the full extraction/training/probe run has not
been launched as part of this preparation request.
