# Color-pie-consistent synthetic cards for ModernBERT MLM

This extends the v4 Oracle MLM corpus with hypothetical card designs. Existing
real cards remain the anchor. Synthetic **cards** are distinct from the previously
prepared synthetic card-relation pairs: no role/slot/probe labels enter MLM here.
The experiment asks whether broader combinations of costs, effect magnitudes,
spell speeds and supported mechanics improve rules understanding. MLM loss alone
cannot prove better functional embeddings or a complete understanding of design.

The user-selected policy is color-pie-consistent designs only. An explicit
[policy file](../configs/card2vec_synthetic_designs_v1.json) declares 20 restricted
effect templates and six compositions, using a conservative subset of Wizards'
[Mechanical Color Pie guidance](https://magic.wizards.com/en/news/making-magic/mechanical-color-pie-2021).
This whitelist is versioned guidance, not a complete classifier for every effect
or a claim about future policy. Historical bends and exceptions are not used to
justify additional synthetic color access.

For example, `{U}` with `Draw five cards.` is included: color access is appropriate
but power level is intentionally unconstrained. Unrestricted reanimation retains
black; a blue-black composition can reanimate and draw, but mono-blue reanimation
is not invented. Mono-green damage uses flying-target or creature-power
restrictions. Black draw retains life loss, green draw depends on creatures,
white reanimation is small-creature restricted, and ramp puts basic lands onto
the battlefield tapped. In compositions, every color supplying an effect gets
at least one corresponding colored mana pip. Ordinary instant and sorcery
templates do not assert balanced timing/costs or printability.

Each generated record stores a `synthetic:` identity, parameters, effect family,
color assignment, source citation, specific color-pie restriction and split. It
has no fabricated `oracle_id`, deck frequency, legality, or competitive label.
It is never appended to the real card vocabulary, used as a frozen probe target,
or turned into a contrastive role label by this stage.

All cost/amount/speed/color variations of one template or composition share a
split. Exclude exact existing structured cards and rules text matching the
opposite frozen Oracle split. Under seed 42 the current stage generates 2,140
designs, retains 1,956, and excludes 184 overlaps. There are 1,928 training
designs and 28 held-out designs after these conservative filters. The small
synthetic validation set is diagnostic; it cannot establish unseen-mechanic
generalization, since compositions share underlying effects and real Oracle
training already exposes many of those mechanics.

Training defaults to 80% real Oracle documents and 20% synthetic documents.
Real data keeps the existing rules-only and structured views. Synthetic cards
use only structured records, preserving the color-bearing mana cost. Sample
synthetic families uniformly, then sample parameter instances uniformly within
each family, with replacement. Persist every sampled ID/view and family exposure.
Validation uses each held-out population once, without oversampling, and saves
real-Oracle and synthetic losses separately before and after training. Reset the
masking RNG to the same recorded seed for before/after comparisons.

The baseline is `answerdotai/ModernBERT-base`; the initial requested run uses one
epoch, seed 42, dynamic padding, length 256, MLM probability .15, LR 2e-5,
weight decay .01, warmup .06, cosine schedule and BF16 on supported CUDA.
Checkpoint/tokenizer, exact training documents, package/code/input hashes,
effective arguments, truncation counts, losses and family exposure are saved.
Before claiming synthetic benefit, also train an Oracle-only checkpoint under
the same seed/hyperparameters and evaluate both card representations. Changes
in overall mixed MLM loss are not an apples-to-apples comparison with Oracle-only
loss; use the saved real-Oracle validation losses and downstream frozen tasks.

Run from the repository root with the v4 Python environment:

```powershell
$V = 'artifacts/card2vec/static_v4_multiview_v1'
python scripts/run_static_semantic.py --stage prepare-designs
python scripts/run_static_semantic.py --stage download-model --allow-download
python scripts/run_static_semantic.py --stage mlm --synthetic-cards "$V/synthetic_cards_v2" --synthetic-fraction 0.2 --epochs 1 --batch-size 16 --allow-expensive
```

The first preparation attempt is preserved at `synthetic_cards_v1/`; v2 corrects
the one-land search wording to `up to one basic land card`. Completed and
incomplete stages are never overwritten. Reuse completed local preparation/model
downloads; choose fresh `--output` paths to repeat or change a run.

Expected training artifacts: `$V/mlm_synthetic_cards_v1/model/`, `losses.json`,
`mixing.json`, `mlm_documents.json`, `synthetic_family_exposure.json`,
`cards_used.json`, `training_arguments.json`, `token_lengths.json`, provenance
and the final completion manifest. A folder without `complete.json` is still
running or incomplete; a model quality result must not be inferred from it.

To extract **real-card** vectors after the mixed MLM completes:

```powershell
python scripts/run_static_semantic.py --stage export --model "$V/mlm_synthetic_cards_v1/model" --label mlm --output "$V/semantic_mlm_synthetic_cards_v1" --allow-expensive
python scripts/run_static_semantic.py --stage evaluate --label mlm --vectors "$V/semantic_mlm_synthetic_cards_v1" --output "$V/evaluation_mlm_synthetic_cards_v1" --allow-expensive
python scripts/run_static_multiview.py --stage render
```

These retain the original Oracle identity/vocabulary lineage. Synthetic cards
remain MLM data only. Any subsequent card-level contrastive training is a
separate stage. Results should distinguish grammar/effect comprehension, color
access, balance/cost realism, and embedding geometry rather than treating them
as interchangeable achievements.

## Completed initial run

`mlm_synthetic_cards_v1/` completed one epoch with seed 42 and batch size 16.
The model and tokenizer are saved under its `model/` directory; all three input,
design and training stages passed their completion-manifest hash checks.
Training used 53,430 real Oracle documents and 13,358 sampled synthetic
documents (20% after rounding). Mean training loss was 1.011507. The 95th
percentile input length was 91 tokens; 60 train/validation records were truncated
at 256 tokens.

| Validation population | Documents | MLM loss before | MLM loss after |
|---|---:|---:|---:|
| Real Oracle | 13,400 | 9.892061 | 0.513984 |
| Synthetic designs | 28 | 16.236116 | 0.277780 |
| Combined | 13,428 | 9.904242 | 0.513789 |

Before/after evaluations reset masking to seed 1042. The before checkpoint is
the pretrained model **after adding the field and MTG symbol tokens**; learning
those new tokens contributes to the loss reduction. Combined and per-source
evaluations have their own masking draws and batch averages, so the combined
number is not an exact weighted average of the two source numbers.

This is one training seed, with no Oracle-only adapted control or downstream
evaluation yet. The loss reduction establishes corpus adaptation, not the
isolated benefit of synthetic cards, better functional embeddings, or proven
knowledge of the color pie. The synthetic holdout remains a small diagnostic.
