# Static Card2Vec v3.2 fusion and downstream export

This extension reads the existing `static_v2` and `static_v3_relations_v1`
artifacts. It never trains SGNS, rebuilds the corpus, or refits SVD. Historical
modules remain unchanged because their hashes are part of saved provenance.

Use Python 3.13 and `requirements-static-fusion.txt` to reproduce the numerical
stack. Run from the repository root:

```bash
python scripts/run_static_fusion.py --stage export
python scripts/run_static_fusion.py --stage prepare
python scripts/run_static_fusion.py --stage retrieval --allow-expensive
python scripts/run_static_fusion.py --stage probes --allow-expensive
python scripts/run_static_fusion.py --stage report
python scripts/run_static_fusion.py --stage render
```

Export creates `artifacts/card2vec/static_export_v1/`. Evaluation stages write to
`artifacts/card2vec/static_v3_2_fusion_v1/`. Existing checkpoints and completed
stages are immutable. Evaluation resumes completed stages only with the same
checkpoint/code hashes. Choose fresh `--export`/`--output` version paths to run
changed experiments. Incomplete stages require explicit inspection and relocation.
`render` refreshes only the notebook's explicitly tagged v3.2 review cells and
the corrected current v3.1 aggregate cell; no computation flags are enabled.

`completed_v3_1/` regenerates the stale v3.1 aggregate outside its immutable
directory. It includes completed retrieval and probe tables plus original result
hashes. Earlier historical v2/v3 notebook snapshots remain unchanged.

For each seed, the checkpoint retains input, context, and SVD source matrices as
float32, plus normalized 256d/384d concatenations and raw fusion diagnostics.
All rows use the exact canonical corpus vocabulary. Every source block is
independently L2-normalized per card before primary concatenation; zero blocks
remain zero. Exported concatenations have no final normalization. Cosine scorers
normalize the full row. Source vectors and coordinates across seeds are never
averaged, rotated, or projected.

The primary experiment compares SGNS input, PPMI-SVD, input+SVD, and
input+context+SVD. The same frozen targets, splits 101/102/103, and `probe_long`
policies apply. The existing helper's random baseline caps at 256 features, so
it is excluded from primary comparisons and is not a matched 384d control.
Raw concatenation is a retrieval-only diagnostic. Retrieval reuses all frozen
queries and first-relevant-hit MRR, preserving each relation/provenance stratum,
Recall@10/20/50, incidence cosine, and popularity. Per-seed CSVs and sample SD
are retained; deterministic baselines have undefined seed SD, displayed as N/A.

The frozen packed corpus and incidence index record 1,002,915 training contexts;
the earlier notebook accepted-context audit says 1,002,916. The export uses the
packed count. A read-only fingerprint comparison confirms exactly one accepted
context (source row 362960) was omitted by post-vocabulary-pruning packing, which
skips contexts with fewer than two retained cards. `context_count_audit.json`
records the evidence; this is a pre-/post-pruning count distinction, not a missing
training artifact. Historical data is unchanged.

```python
from mtgdeck.static_export import load_static_card_embeddings

bundle = load_static_card_embeddings(
    representation="concat_input_context_svd_384", seed=42,
    artifact_root="artifacts/card2vec/static_export_v1",
)
card_id = bundle.index("lightning bolt")
vocab, card_to_idx = bundle.vocab, bundle.card_to_idx
matrix, metadata = bundle.embedding_matrix, bundle.metadata
# Optional PyTorch conversion:
# embeddings = torch.nn.Embedding.from_pretrained(torch.from_numpy(matrix), freeze=False)
# projection = torch.nn.Linear(matrix.shape[1], d_model)
# h = projection(embeddings(card_ids))
```

Loading requires only NumPy and standard-library modules. It verifies vocabulary
uniqueness, shape, dtype, finite values, and (by default) SHA256. Lookups accept
exact canonical keys and raise a clear error for absent cards. No OOV vector or
implicit seed averaging is introduced. Seed 42 is conventional, not test-selected.

Carry both normalized fusions into downstream ablations alongside random
initialization, SGNS input, and SVD. The trainable projection belongs to the
deck encoder and is trained end-to-end; the loader supports all input widths.
Compare frozen and fine-tuned embeddings. Static probe/retrieval improvements
do not establish mechanical synergy or deck-recommender quality, or isolate
dimension as the cause. The decisive experiment remains downstream completion,
convergence, data efficiency, rare cards, cross-format transfer, and temporal or
new-card generalization.

Completed numerical results and relation-specific normalization deltas live in
`artifacts/card2vec/static_v3_2_fusion_v1/report/` and the notebook's final section.
