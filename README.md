# MTG GenRec

MTG GenRec learns MTG card representations with Card2Vec and an attention VAE to recommend missing cards for a Commander deck. Recommendations use Oracle card identities and Commander legality filters.

**scrape → data → train → demo**

```text
scrape/
  scraper.py             Single-writer CLI and discovery scheduler
  state.py               JSONL output, deduplication and checkpoint recovery
  http.py                Retries, rate limits and pagination validation
  metadata.py            Scryfall refresh CLI
  sources/               Moxfield, MTGTop8 and Deckbox adapters
configs/commander.json    Training subset input manifest
scripts/curate_decks.py   Normalize, validate and deduplicate training data
src/mtgdeck/
  data.py                Canonical records and dataset preparation
  metadata.py            Shared Oracle readers and snapshot selection
  legality.py            Commander validation and candidate filters
  card2vec.py             Card representation learning
  vae.py                 GenRec model and training primitives
  recommend.py           Recommendations, baselines and evaluation
  inference.py           Shared demo parsing, checkpoint loading and scoring
notebooks/genrec.ipynb    Training, evaluation and model comparisons
demo/app.py              Gradio deck-completion demo
demo/space/              Space metadata, inference requirements and asset pins
tests/                   Offline tests
docs/                    AWS migration handoff
data/                    Local datasets, metadata and scraper state (ignored)
checkpoints/             Local trained GenRec models (ignored)
```

## Local setup

Use Python 3.11 or newer. The full suite is verified in the existing Python 3.13 environment. Run commands from the repository root.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

GPU training needs a PyTorch installation compatible with your CUDA environment. The demo also runs on CPU. Large datasets and trained weights are local assets; installing dependencies does not download them.

## 1. Scrape

Refresh card metadata and explicit commander eligibility:

```bash
python -m scrape.metadata
```

Collect Commander and cEDH decks with the single scraper:

```bash
python -m scrape.scraper --sources moxfield mtgtop8 --formats commander cedh
```

Each source runs only its supported formats. Omit the source/format flags to collect all configured constructed formats, including Deckbox. Use `python -m scrape.scraper --help` for page limits, discovery controls, per-source delays and Oracle refresh options. `--mtgtop8-event-metadata` adds tournament placement/player counts by fetching event pages. Moxfield requires `MOXFIELD_USER_AGENT` in the environment or local `.env` (the existing `user-agent` key also works). Keep `.env` private. Use `--sources mtgtop8 deckbox` if that setting is unavailable.

**Run only one scraper process against a given output directory and checkpoint.** There is no cross-process writer lock. Stop with Ctrl+C to flush partial progress. Rerun the same command to resume; `--fresh` resets search progress and is not the normal resume command. Collection is unbounded unless limits are supplied.

Existing storage paths remain unchanged so prior progress resumes:

- `data/format_corpora/<format>.jsonl`: full schema-v1 decklists with quantities and zones.
- `data/format_corpora/deckbox.jsonl`: Deckbox records, retaining the source's format labels.
- `data/cooccurence/embedding_corpus.jsonl`: deduplicated, count-free card contexts; a derived representation, not the current notebook's training input.
- `data/diverse_scraper.checkpoint.json` and `data/format_corpora/.diverse_scraper.seen.sqlite3`: active search progress and seen IDs.

The spelling and filenames above are intentional compatibility with local state. Source retries, result-window expansion, repeated-page checks, atomic checkpoint replacement, and restartable format corrections remain in the consolidated scraper. Startup still scans existing corpora and loads the search checkpoint; large-state resource limits remain work for the cloud migration.

## 2. Prepare data

```bash
python scripts/curate_decks.py --manifest configs/commander.json
```

The manifest selects Commander/cEDH records from existing `data/decks.jsonl` and current scraper outputs. Paths are relative to the manifest. Explicitly optional missing inputs are reported; preparation fails if none exist. Edit or add a manifest to choose a different subset without copying raw files. A single older harvest can instead be supplied with `--input path/to/harvest.jsonl`.

Preparation streams input records, adapts older Moxfield/MTGTop8 schemas, checks Commander legality, resolves Oracle IDs and removes duplicate accepted decks. It produces `data/decks_clean.jsonl`, a rejection manifest and an audit report. These are derived outputs that preparation replaces; raw inputs are never rewritten. Keep the scraper stopped while preparing a consistent dataset snapshot. Run training after preparation finishes.

Schema v1 retains source identity, URL, date, format, mainboard, sideboard, commanders, optional companions and source metadata. Card entries retain display names and quantities; preparation can attach Oracle IDs. Companion legality is not inferred. Training converts cards to `oid:` tokens without replacing the stored display names.

Preparation and the demo prefer the updater's `data/oracle_cards.jsonl.gz`, falling back to the newest dated snapshot or `oracle_cards.json`. The notebook uses the same selector; its configuration can pin a specific snapshot for reproducible experiments.

## 3. Train

```bash
jupyter notebook notebooks/genrec.ipynb
```

Run the notebook from the first cell. It requires the prepared `data/decks_clean.jsonl` and Oracle metadata. The configuration cell controls data paths, seed, device, model size and ablations. It trains/loads Card2Vec, creates grouped train/validation/test splits, trains GenRec and compares recommendations using matched held-out cards. Manual EDHREC comparisons remain in the final section.

The default 896-dimensional Oracle-ID models use `data/card2vec_clean_oracleid_v2_896.model` and its NumPy sidecars, plus `checkpoints/attention_oracleid_v2_*.pt`. Keep sidecars with their Card2Vec model. The notebook retains its current algorithms; it loads datasets into memory and is not yet a resumable batch-training CLI.

## 4. Demo

```bash
python demo/app.py
```

Select an available GenRec checkpoint, enter a Commander and partial deck, and request recommendations. The demo loads the checkpoint, Oracle metadata and commander eligibility. It does not require the training corpus or separate Card2Vec files. A fresh checkout needs trained checkpoints copied into `checkpoints/` or produced by the notebook.

The Gradio demo preserves checkpoint selection, Commander/partner inputs, deck
text parsing, legality and color-identity filters, seeded latent sampling,
score tables, resolved-card warnings and CSV download. CPU inference is the
default after benchmarking; set `MTG_DEVICE=cuda` for a local GPU or
`MTG_DEVICE=zerogpu` on ZeroGPU hardware. Models load once at startup.

Hosted demo: [pengkev/mtg-genrec](https://huggingface.co/spaces/pengkev/mtg-genrec).
GitHub `main` is the source of truth. GitHub Actions tests the repository, builds
an allowlisted inference artifact and uploads it to the Space. See
[Space deployment](docs/space-deployment.md) for asset bootstrapping, keyless
Trusted Publisher setup, manual redeployment and verification.

## Tests and migration

```bash
python -m pytest -q
```

Tests use fixtures and temporary directories; they do not scrape live sites or train on local corpora. See [the AWS migration handoff](docs/aws-migration-audit.md) for current resource findings and remaining work. No AWS infrastructure is included.
