# Static v4: separate deck, Oracle, and Spellbook information

The established v3.2 baseline is the frozen 384d input/context/SVD concatenation.
V4 asks whether rules text adds functional similarity and Spellbook adds missing
mechanical components. No production v4 model has been trained, and no new
representation-quality result is claimed. Historical code, artifacts and the
first 80 notebook cells are preserved in `notebooks/archive/card2vec_static_embeddings.ipynb`.
Archive setup cells locate the repository from the new directory. Existing unrelated working-tree edits
are outside this extension.

## Implementation and artifact contracts

`static_semantic.py` prepares Oracle identities, serialization views, all observed
brace symbols, synthetic relations, local ModernBERT MLM and separate contrastive
training, and mean/first-token extraction. `static_spellbook.py` snapshots the
official bulk file, retains raw variant records and exclusions, constructs
grouped completion/conditional-slot examples, and trains two relation heads.
`static_multiview.py` exports independent raw blocks/masks and evaluates the
seven required views using existing frozen probes/relation scorers. `static_v4.py`
provides write-once stages, connected families, hashes and runtime provenance.
Four new scripts expose these stages and isolated notebook review.

Every stage refuses existing output directories, including incomplete ones, and
rejects paths inside or above historical roots. Input files, implementation,
configuration, installed packages, output files, and checkpoints are hashed.
Failed stages retain evidence and require a fresh output version. Loading saved
sources checks their saved integrity; their historical implementation hashes
describe that source, while a new consuming stage records its current code.

Default research root: `artifacts/card2vec/static_v4_multiview_v1/`.
Default portable checkpoint: `artifacts/card2vec/static_export_v2/`.
Generated artifacts are ignored by Git. These are new roots; v1/v2/v3/v3.2 and
`static_export_v1` are read-only. One explicit deck/model seed belongs to each
export: use fresh roots for seeds 43/44. Never average coordinates between seeds.

Local preparation completed: 38,906 Oracle identities, 33,415 unambiguous usable
cards, 31,088 overlapping the 33,623-card frozen deck vocabulary, and 2,327
deck-unseen cards. There are 2,560 canonical-name collision keys and no conflicting
identity records; 2,535 deck keys lack an unambiguous text source. This source
coverage limitation follows the existing normalization rather than guessing an
Oracle identity. The saved corpus has 4,567 Tier 1 and 20,595 Tier 2 train pairs,
plus 16,639 train descriptions; 478 cards are reserved through whole-family
blocking. These are preparation counts, not trained-model evaluation metrics.

The portable vocabulary starts with the exact frozen deck order and appends
Oracle cards absent from it. Each source has a float32 matrix and boolean mask.
Missing blocks deliberately become zero; train-unseen mechanical vectors are
zero rather than random initializations. Source raw matrices remain selectable;
normalized fusions are assembled on load without storing four redundant giant
arrays. Deck384 is one source block in new fusions, so it receives its own L2
normalization; its original three normalized 128d components remain unchanged
in `static_export_v1`. Fusions apply L2 per source block, no final normalization,
no trainable weights, and no averaging of incompatible coordinates.

```python
from mtgdeck.static_multiview import load_multiview_embeddings
bundle = load_multiview_embeddings(representation='text')
vectors = bundle.embedding_matrix
masks = bundle.metadata['coverage_masks']
# Also: deck, mechanical, semantic_vanilla, semantic_mlm,
# deck_text, deck_mechanical, text_mechanical, deck_text_mechanical.
```

## Oracle preparation and supervision

Use the local canonical `data/oracle_cards.jsonl.gz` through the existing reader
and name normalizer. Deduplicate by `oracle_id`; exclude/report contradictory
records and canonical-name collisions. Retain full card metadata for auditing,
but serialize no flavor or card name. Replace self-name references with `[SELF]`.
Both faces and optional PT/loyalty/defense fields are retained. Support
`rules_only`, `structured_no_name`, and heuristic `normalized_semantic` views.
Discover brace tokens from the actual corpus, including hybrids, Phyrexian,
energy and untap symbols, and persist tokens and corpus hashes.

MLM keeps all serialization views of a normalized rules family in one partition.
The recorded 80/20 train/validation split is independent of relation labels.
Defaults: length 256 with measured token truncation, dynamic padding, MLM .15,
LR 2e-5, decay .01, warmup .06, cosine scheduler, 1 epoch (CLI allows 1–3), and
BF16 when CUDA supports it, otherwise CUDA FP16 or CPU FP32. Save before/after
validation loss, train loss/history, effective arguments, exact cards, tokenizer,
checkpoint and runtime. MLM loss does not establish card-embedding quality.

Mean pooling over non-padding encoder states is fixed as primary. It includes
non-padding special tokens. First BOS/CLS is diagnostic; ModernBERT's alternating
local attention makes its interpretation different from a universally pooled
sentence vector. Both float32 matrices retain full Oracle and deck-aligned rows.
Length/truncation and deck-unseen coverage are explicit.

Synthetic examples store exact endpoints, rule, tier, weight, hard-negative reason,
source Oracle IDs/signature evidence, family, and split. Identical self-normalized
rules and curated slots produce Tier 1; auditable lexical effect signatures and
broad roles produce Tier 2 proposals. No claim of full Rules parsing is made.
Targets are lexical mentions, not resolved rules scope. Negative proposals are
bounded deterministic same-action samples, filtered against the positive family,
with target/cantrip/destroy-exile/temporary-answer/cost/timing/restriction tags.
Tier 2 examples are filterable and receive half the Tier 1 weight. Spellbook
slots retain source variant IDs, surrounding context and TRAIN-only provenance;
they are conditional slot evidence, not universal interchangeable-card judgments.

Tier 3 descriptions deterministically combine cost/type and exact Oracle wording.
They intentionally retain substantial lexical overlap; text-to-card retrieval
on these descriptions is a controlled diagnostic, not natural-user paraphrase
generalization. No external LLM generates labels. Contrastive training supports
card pairs and descriptions, multi-positive in-batch InfoNCE, explicit negatives
weighted .5, and family-level false-negative masking. Exact training examples
are saved separately; no evaluation probe targets train either representation.

All connected synthetic/curated families stay together. Reserved manual-test
candidates block the entire connected family from supervision. The new
`configs/card2vec_semantic_v4_test.json` expands difficult examples and distinctions,
but its status is **pending human review**. We cannot manufacture a human review:
manual-test metrics remain unavailable until a reviewer creates a new declared
version with `review_status: manually_reviewed` and records review provenance.
Existing role/slot evaluation removes training-family overlap. Historical
benchmarks remain untouched. Synthetic validation/test metrics are separately
labelled by confidence and task, never presented as reviewed ground truth.

Six explicit ablations are supported: `vanilla`, `mlm`, `semantic_mlm`, `tier1`,
`tier12`, `tier12_description`. Missing ablations are unavailable, never replaced
with another run's metrics. Evaluate untuned MLM vectors before contrastive
training. Keep raw/structured MLM and semantic-record MLM in distinct roots;
compare both extraction views via separate exports if desired. Fix architecture,
pooling, tiers and fusion before final test evaluation.

## Current Commander Spellbook schema and adjustments

On 2026-10-05, direct inspection of the [official live OpenAPI schema](https://backend.commanderspellbook.com/schema/?format=json)
and one REST variant reported API **7.1.4**. The schema requests the compressed
[official bulk snapshot](https://json.commanderspellbook.com/variants.json.gz)
rather than paginating the whole REST collection. Bulk objects contain
`timestamp`, `version`, `variants`, and `aliases`; REST pages use `results`,
`next` and nullable `count`. The snapshot stage stores raw bytes and schema,
UTC retrieval time, hashes, bulk/API version, actual total count and observed
field frequencies. Credit: [Commander Spellbook](https://commanderspellbook.com).

Verified variant fields: `id`, `of`, `includes`, `uses`, `requires`, `produces`,
`easyPrerequisites`, `notablePrerequisites`, `description`, `manaNeeded`,
`manaValueNeeded`, `notes`, `legalities`, `status`, `identity`, `spoiler`,
`prices`, `popularity`, `salt`, `saltVoteCount`, `bracketTag`, `variantCount`.
`uses[].card` includes `name`/`oracleId`; use records retain quantity, used face,
zone locations, commander requirement and zone-specific state. `requires` has
typed templates and Scryfall query/API metadata. `produces` has structured
feature IDs/names/status/quantity. `of` identifies generator combos; `includes`
identifies additional included combos. The raw record preserves everything.
There is no supported per-variant creation timestamp in the inspected schema;
bulk build time cannot justify a temporal split. Snapshot preparation reports
the actual downloaded fields rather than assuming this inspection never changes.

Resolve cards only with unambiguous canonical names and consistent Oracle IDs.
Unknown/ambiguous/mismatched members exclude the entire edge, avoiding partial
fake combos. Primary baseline eligibility requires verified `OK`, Commander
legal, at least two distinct real cards, unit quantities, and no template slots.
Template and multiplicity variants remain cached and counted. Encoding templates
or multiplicity is future work, not silently discarded evidence or fake cards.

Split connected components sharing official `of`/`includes` IDs or exact real
card sets. This conservative grouping can create large components; save sizes
and reject empty train/validation partitions. Audit cross-split exact, subset,
Jaccard ≥ .8 overlap and per-card frequencies, retaining overlap examples. Near
variants in unrelated official components can remain, so no unseen-combo claim
is automatic. Auditing costs depend on shared-card postings; a very large bulk
snapshot can take substantial CPU time. Training and review read cache only.

Generate every real-card leave-one-out target. The simple model has shared card
embeddings, order-invariant mean aggregation, separate complementarity and
replacement query/key projections, normalized dot scoring and full-vocabulary
softmax. Exclude visible cards. Replacement supervision additionally requires
same surrounding members, component, outcomes, and visible-card states; it is
split-local, auditable, and conditions on the current slot occupant. If no slots
exist, the replacement head is explicitly untrained/unavailable. Optional BCE
outcomes use train-only feature frequencies and the complete hyperedge.

Retrieval candidates are the full unambiguous Oracle vocabulary, including
train-unseen targets; report seen/unseen strata rather than omit hard targets.
Metrics: MRR, Recall@1/5/10/20/50, NDCG, mean/median target ranks. Deterministic
canonical-name ties match the current relation evaluator's policy. Comparators
are train-only popularity, raw co-incidence and incidence cosine, plus frozen
SGNS, SVD and deck384 with documented zero OOV rows. Incidence cosine supplies
the practical sparse incidence comparator; no new PPMI factorization is required.
Two-card forward/reverse targets are partner retrieval; multi-card queries are
set completion. Conditional replacement retrieval has its own result file.

## Evaluation interpretation

Multiview evaluation reuses frozen v2 targets and splits 101/102/103 for color,
type, observed format usage, clean mana value and archetype. All original label
IDs remain, so probe comparisons include the explicit zero/missing-source policy.
The historical random control caps at 256d; exclude it instead of presenting it
as a dimension-matched semantic control. Retrieval excludes uncovered query or
target sources but retains the full candidate vocabulary: compare coverage as
well as scores. Existing scorer preserves similarity, substitution,
complementarity, association, archetype/concept and structural-proxy strata.

Diagnostics save nearest neighbors, cosines and heuristic functional differences
for Remand, Reprieve, Bolt, Wrath, Llanowar Elves, Soulherder, Young Pyromancer and
Entomb; sampled CKA/cosine correlation use covered view intersections. Undefined
geometry is unavailable, not zero. Frequency strata come from frozen deck counts.
Cold-start deliberately hides deck rows (up to 50/bin; held-out functional anchors
first, seeded remainder), retains text, and compares intact fusion, hidden fusion
and text-only functional retrieval. This is a **simulation**. Full Oracle
description retrieval separately covers cards absent from the deck vocabulary.
The exported zero deck blocks demonstrate downstream compatibility; actual deck
encoder cold-start quality requires a separate downstream experiment.

Per-query/per-split and explicit model-seed provenance are retained. Do not call
three probe splits three independent models. For model-seed SD, complete genuinely
independent semantic/mechanical runs with seeds 42/43/44 and average probe split
values within each model first; cross-seed coordinate averaging is prohibited.
The current report makes no SD/significance claim from one seed or tiny curated
benchmarks. The downstream recommendation remains deck384 as the established
baseline, with separate trainable source projections as the v4 experiment.

## Reproducible commands (repository root, PowerShell)

Use Python 3.13 in a separate environment with `requirements-static-v4.txt`. Transformers 5.3.0
is the API exercised by the offline smoke test; all actual installed versions
are recorded per stage. Oracle preparation and notebook review were run locally;
production model/download/training/evaluation commands remain **unrun**.
Expensive stages and network stages require explicit opt-in flags. Notebook
Run All only reads saved results. No command retrains deck SGNS or SVD.

```powershell
python -m pip install -r requirements-static-v4.txt
# Already prepared locally: reuse the saved oracle/ stage.
# On a fresh checkout run prepare; existing output is intentionally refused.
python scripts/run_static_semantic.py --stage prepare
python scripts/run_static_semantic.py --stage download-model --allow-download
python scripts/run_static_semantic.py --stage mlm --epochs 1 --allow-expensive
python scripts/run_static_semantic.py --stage export --label vanilla --allow-expensive
python scripts/run_static_semantic.py --stage export --model artifacts/card2vec/static_v4_multiview_v1/mlm/model --label mlm --allow-expensive
python scripts/run_static_semantic.py --stage evaluate --label mlm --vectors artifacts/card2vec/static_v4_multiview_v1/semantic_mlm --allow-expensive

python scripts/run_static_spellbook.py --stage snapshot --allow-download
python scripts/run_static_spellbook.py --stage prepare
python scripts/run_static_spellbook.py --stage train --epochs 3 --allow-expensive
python scripts/run_static_spellbook.py --stage evaluate --split validation --allow-expensive
python scripts/run_static_spellbook.py --stage evaluate --split test --allow-expensive

python scripts/run_static_semantic.py --stage contrastive --model artifacts/card2vec/static_v4_multiview_v1/mlm/model --tiers 1 --allow-expensive
python scripts/run_static_semantic.py --stage export --model artifacts/card2vec/static_v4_multiview_v1/contrastive_tier1/model --label tier1 --allow-expensive

python scripts/run_static_multiview.py --stage export --semantic vanilla=artifacts/card2vec/static_v4_multiview_v1/semantic_vanilla --semantic mlm=artifacts/card2vec/static_v4_multiview_v1/semantic_mlm --semantic tier1=artifacts/card2vec/static_v4_multiview_v1/semantic_tier1 --allow-expensive
python scripts/run_static_multiview.py --stage evaluate --probes --spellbook artifacts/card2vec/static_v4_multiview_v1/spellbook --allow-expensive
python scripts/run_static_multiview.py --stage render
python -m pytest tests/test_static_v4.py -q
```

MLM training includes before/after held-out MLM evaluation in `mlm/losses.json`.
Semantic evaluation above evaluates untuned card vectors, both pooling choices,
and deterministic descriptions before any contrastive stage. Prefer validation
for configuration decisions; keep the final test readout frozen.

Remaining ablations and train-only Spellbook semantic augmentation:

```powershell
python scripts/run_static_semantic.py --stage mlm --semantic-records --epochs 1 --allow-expensive
python scripts/run_static_semantic.py --stage export --model artifacts/card2vec/static_v4_multiview_v1/mlm_semantic/model --label semantic_mlm --allow-expensive
python scripts/run_static_semantic.py --stage contrastive --model artifacts/card2vec/static_v4_multiview_v1/mlm/model --tiers 1 2 --allow-expensive
python scripts/run_static_semantic.py --stage export --model artifacts/card2vec/static_v4_multiview_v1/contrastive_tier12/model --label tier12 --allow-expensive
python scripts/run_static_semantic.py --stage contrastive --model artifacts/card2vec/static_v4_multiview_v1/mlm/model --tiers 1 2 --descriptions --allow-expensive
python scripts/run_static_semantic.py --stage export --model artifacts/card2vec/static_v4_multiview_v1/contrastive_tier12_description/model --label tier12_description --allow-expensive
python scripts/run_static_semantic.py --stage prepare --output artifacts/card2vec/static_v4_multiview_v1/oracle_with_slots --replacements artifacts/card2vec/static_v4_multiview_v1/spellbook/replacements.json
```

Use `--prepared .../oracle_with_slots` and fresh `--output` roots for subsequent
MLM/contrastive runs using slots. The metadata is identical but relation families
and splits change, so do not mix their training/evaluation lineage. Add all six
semantic exports to a new portable `--export` path with repeated `--semantic`
arguments; then evaluate that path into a fresh `--output`. `--primary mlm` is
the fixed default; changing it is a predeclared ablation, never a test selection.

Expected files:

| Stage | Files under the new research root |
|---|---|
| `oracle` | cards, corpus, tokens, synthetic pairs, split assignments, coverage, reserved benchmark |
| `model_download_v2` | local pretrained model/tokenizer and resolved file hashes; training files only |
| `mlm`, `mlm_semantic` | model/tokenizer, losses, exact cards, token lengths, effective arguments |
| `contrastive_*` | tuned model/tokenizer, exact train examples, train InfoNCE history |
| `semantic_*` | full Oracle mean/first vectors, deck-aligned matrices/mask, description vectors |
| `spellbook_snapshot` | raw immutable bulk bytes, raw schema, retrieval/version/schema summary |
| `spellbook` | raw-preserving normalized variants, completions, conditional replacements, overlap/frequency audit |
| `mechanical_s42` | raw vectors/mask, both heads checkpoint, outcomes, train examples, loss history |
| `mechanical_evaluation_*` | full-pool retrieval, replacement metrics, coverage, unavailable tasks |
| `multiview_evaluation_s42` | relation/probe tables, coverage, geometry, qualitative, cold-start, ablation status |

All stages also contain provenance and completion hash manifests. The portable
export has `vocab.txt`, `manifest.json`, independent source matrices/masks, and
fixed fusion definitions. The renderer updates only `static-v4-review` code
outputs and validates that every other cell stays exactly equivalent as a
JSON object. It never downloads, trains, refreshes metadata or recomputes history.

Intentionally unrun: Oracle-only production MLM, contrastive/mechanical training,
full-vocabulary production extraction and
evaluation, all six ablations, independent seed replication, and downstream
deck-encoder experiments. Small offline smoke tests establish executable
infrastructure only. Human review of the reserved benchmark remains pending.

The later user-requested model download and one-epoch mixed Oracle/synthetic-card
MLM run have completed. See [the synthetic-design MLM report](card2vec-synthetic-design-mlm.md)
for the separate immutable checkpoint, corpus policy and measured losses.

The next [prepared notebook/CLI run](card2vec-v4-training-run.md) reuses that MLM
checkpoint and frozen deck sources, adds the requested aligned SGNS/PPMI
difference features, trains cached Spellbook hyperedges, and runs the frozen
linear probes and relation tests. The explicit Spellbook snapshot/preparation
stages are now complete; full representation training/evaluation remains opt-in.
