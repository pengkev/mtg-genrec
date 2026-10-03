# Historical Commander static-384 pilot

This records the initial frozen/fine-tuned Commander pilot. The active notebook
now trains three format-specific models with premium refinement; see
[the current workflow](genrec-formats.md). The pilot metrics below remain historical.

The Commander completion pipeline now initializes its shared encoder/decoder card
lookup from `static_export_v1`, seed 42, representation
`concat_input_context_svd_384`: normalized 128d SGNS input, 128d SGNS context,
and 128d PPMI-SVD blocks, concatenated without further normalization. The existing
trainable input projection combines these card features with role and quantity
features before set attention. Decoder queries score the same card table.

The primary experiment fine-tunes the card table with AdamW at `2e-6` and no
embedding weight decay. Other trainable parameters use `2e-4`. The comparison
freezes the card table. Both start from the same weights, training random seed,
deck splits, masks, architecture, and validation checkpoint rule. Original static
exports are never modified; learned embeddings are stored in GenRec checkpoints.

Canonical export names are aligned to the downstream Oracle-ID vocabulary using
the local Oracle catalog. Missing embeddings are explicitly audited and either
raise (`missing_embedding_policy='error'`) or receive reproducible random vectors
at the source RMS scale (`'random'`, default). PAD and UNK begin at zero. A frozen
comparison also freezes any random fallback rows.

## Historical execution

These pilot artifacts were produced before the notebook gained format-specific
base/premium stages. The current CLI runs the new workflow and does not reproduce
the historical frozen-table ablation below. Original checkpoints and metrics are
retained in `checkpoints/static384_pilot_v1/` and `checkpoints/static384_pilot_v2/`.

## Pilot scope

The pilot uses the first 2,000 curated records, all from Moxfield: 1,600 training,
200 validation, and 200 test decks. Exact/near-duplicate grouping precedes the
split; the audit found no cross-split exact duplicates or sampled near duplicates.
Test results average three matched 20% masking tasks per deck (600 tasks).
All 10,808 non-special vocabulary entries have pretrained embeddings; no random
fallback is needed in this pilot. Validation Recall@20 selects checkpoints.

This is a bounded integration/transfer experiment, not a representative full-data
benchmark. The independent static pretraining corpus may contain downstream test
decks; pretraining overlap has not been excluded. The original 5,000-step KL
warmup is retained, so this 400-step pilot does not reach the maximum KL weight.
The historical 896d manual recommendation lists in the notebook are unchanged and
are explicitly labeled historical; they are not measurements of this experiment.

## Eight-epoch results

| Method | Recall@20 | NDCG@20 |
| --- | ---: | ---: |
| global | 0.1119 | 0.1664 |
| commander | 0.1582 | 0.1722 |
| cooccurrence | 0.1648 | 0.2141 |
| Static fusion cosine | 0.1392 | 0.1496 |
| Legal Variational Set Completion | 0.1894 | 0.2064 |
| Legal Variational + log_count | 0.2402 | 0.2687 |
| Frozen GenRec | 0.1877 | 0.2037 |

Both neural runs selected epoch 8. Fine-tuning improved Recall@20 by about 0.18
percentage points over frozen embeddings in this single-seed pilot; this is a
small observed difference, not evidence of a robust gain. GenRec improved recall
over direct static cosine and co-occurrence, although co-occurrence still had
higher NDCG@20 than the neural model alone. The validation-selected hybrid used
`log_count` with weight `2.0`; its gains should not be attributed to embeddings
alone. These results do not establish improvement over the previous 896d model.

The saved frozen table was verified bit-for-bit equal to the aligned source.
The fine-tuned table's RMS coordinate change was 0.00019143. The final checkpoint
loaded through the demo's inference adapter and returned legal recommendations
for the default Muldrotha partial deck (including Farseek, Nature's Lore, and
Three Visits); the complete example is saved as `recommendation_example.json`.

Artifacts: `checkpoints/static384_pilot_v2/`; execution log:
`logs/genrec_static384_pilot_v2.log`. The earlier two-epoch `static384_pilot_v1`
run is retained separately. Validation included 43 integration/model/demo tests
and 12 static-export tests (the latter run on Linux because Windows does not
permit the symlink test without additional privileges).
