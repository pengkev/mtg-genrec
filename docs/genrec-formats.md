# Format-specific GenRec and premium refinement

`notebooks/genrec.ipynb` trains independent Commander, Modern, and Legacy models.
`edh`, `cedh`, and `commander` normalize to Commander; Duel Commander does not.

Each format follows the same two-stage workflow:

1. Initialize from seed 42 of `concat_input_context_svd_384` in
   `artifacts/card2vec/static_export_v1`, aligning canonical card names to Oracle IDs.
2. Train on its broad corpus and select the best base checkpoint by base validation
   Recall@20. The deck model learns at `2e-4`, the shared card table at `2e-6`.
3. Reload that checkpoint and refine only on premium training decks. Rates decrease
   to `2e-5` and `2e-7`, respectively. Premium validation selects the refined model,
   with the unchanged base model as the epoch-zero candidate.
4. Evaluate both checkpoints on identical base and premium test tasks. If premium
   refinement hurts general-corpus performance, the comparison exposes that cost.

The default stages run eight and four epochs. Both fine-tune embeddings end to end;
there is no automatic frozen ablation in this workflow. Epoch-zero selection means
premium training did not beat the incoming model on premium validation. The source
export is never modified. Embedding width is inferred from the artifact.

## Inputs and identity completion

The notebook's `CORPORA` mapping is authoritative. Commander combines
`data/decks_clean.jsonl` with available Commander/cEDH/EDH files under
`data/format_corpora/`; Modern/Legacy use their matching format corpus files.
Premium inputs come only from matching files under `data/premium/formats/`.
Missing alias files are optional, but a missing entire tier is an error.

All inputs undergo schema/Oracle checks. Commander uses the existing full
Commander validator. Modern/Legacy require at least 60 mainboard cards, at most
15 sideboard cards, snapshot format legality, and copy limits across both zones
(with basic-land and Oracle-text exceptions). Companion-specific construction
conditions are not inferred. Rejected rows are counted with example reasons.
Old tournament lists containing cards now banned in the local snapshot are rejected;
this is not a historical-legality evaluation.

The model predicts missing mainboard **identities**, retaining quantity as an
encoder feature. Masking removes all copies of a selected identity. It does not
learn copy counts or sideboard recommendations. Commander retains its command-zone
roles and color restrictions; Modern/Legacy have no command zone or color mask.

## Splits and coverage

Both tiers are combined before splitting. Identical model-visible card sets are
deduplicated with premium origin taking precedence. Versions with the same source
identity are grouped, as are near duplicates found using deterministic MinHash
candidates and exact Jaccard verification at 0.90. Changing copies or sideboards
cannot move an identical mainboard identity set across splits. MinHash candidate
search may miss some near duplicates; stored group membership is not an exhaustive
all-pairs guarantee. Premium-bearing groups are balanced first across 80/10/10
partitions, followed by base-only groups.

A format vocabulary uses base-train plus premium-train cards only, allowing new
premium training cards without resizing the decoder mid-run. Missing static rows
get reproducible random initialization by default, audited in each format's
`data_audit.json`; `missing_embedding_policy='error'` instead requires full coverage.
Holdout-only cards remain unrecovered evaluation targets and target coverage is
reported. All six format/tier partitions must have nonempty splits.

This prevents downstream base training from seeing premium holdout groups. It does
not remove possible overlap from the independent static embedding pretraining corpus.
Premium provenance is retained, but training samples follow empirical deck frequency;
creator selection does not supply a power label and creators are not reweighted.

## Execution and artifacts

Run the notebook's cells through the matched-evaluation section, or:

```bash
python scripts/run_genrec.py --run-name genrec_formats_v1
python scripts/run_genrec.py --formats modern legacy --run-name constructed_v1
```

`--max-records 500 --epochs 1 --premium-epochs 1` is a bounded smoke configuration,
not a benchmark. The cap takes the first records across listed files per tier,
so it can omit later sources/aliases. Use fresh run names; existing runs are not
silently overwritten or automatically resumed. Data is loaded one format at a time.

Each format directory contains base and premium checkpoints, `data_audit.json`
with split membership and embedding hashes, and `results.json` with stage histories,
validation selection, and before/after test scores. `config.json` and a comparison
CSV live at the run root. Checkpoints include format, stage, vocabulary, encoder
weights, and premium-parent path.

After training, use `recommend_format('modern', {'Lightning Bolt':4, 'Mountain':8})`
in the notebook; Commander additionally takes a commander list. The demo serves one premium-refined checkpoint per format by default. Choose the
format at the top; each format retains its own partial-deck draft. Commander inputs
appear only for Commander. Modern/Legacy use snapshot format legality without a
color identity restriction. Recommendations are missing mainboard identities;
the card popup lets you choose how many copies to add, with limits across the
mainboard and sideboard. Basic lands and Oracle-text copy exceptions are retained.

Set `MTG_ALL_CHECKPOINTS=1` for local experimentation to load the full checkpoint
inventory and expose a format-filtered checkpoint selector in Advanced settings.
Historical checkpoints stay on disk. Asset export selects one model per format,
preferring premium refinement, then the latest matching checkpoint, and preserves
the format in inference metadata. The hosted manifest remains separately pinned;
new binary assets still require the normal asset publication workflow.

Companions must be chosen explicitly in the companion selector or a pasted
`Companion` section. Ordinary mainboard copies never activate companion rules.
The selector includes only companions legal in the local Oracle snapshot, with
Lutri excluded as a Commander companion even when it is legal as an ordinary card. Checks
cover all ten original companion requirements, constrain candidate recommendations
and additions, and include commanders in Commander starting-deck checks. Lutri's
singleton condition applies only to nonlands in the starting mainboard, while the
normal constructed copy limit counts mainboard plus sideboard (including the
chosen companion's sideboard slot). Umori requires a common card type across the
entire starting deck. Yorion reports the 80-card final-size requirement for partial
constructed decks and is unavailable as a Commander companion. Unknown card names
must be resolved before checking companion construction. Split cards use combined
characteristics; transforming/modal double-faced cards and adventures use their
front/normal characteristics outside the stack. Zirda recognizes explicit activated
abilities, intrinsic basic-land-type mana abilities, and activated keywords in the
snapshot. This is construction checking for partial decks, not a full game-rules
engine or a declaration that an unfinished deck is tournament-ready.

Rules reference: [Wizards' Ikoria release notes](https://magic.wizards.com/en/news/feature/ikoria-lair-behemoths-and-commander-2020-edition-release-notes-2020-04-10).
The Commander companion-only ban follows the [February 9, 2026 announcement](https://magic.wizards.com/en/news/announcements/commander-banned-and-restricted-february-9-2026).
Historical manual comparisons remain in the appendix.
