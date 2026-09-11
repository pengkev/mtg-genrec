# GenRec cleanup report

Reviewed and completed 2026-09-11 against the requirements in the local `instruction.txt`. The repository now follows **scrape → data → train → demo**. No Git history was rewritten and no AWS infrastructure was added.

## Resulting structure

```text
configs/commander.json
scrape/
  scraper.py
  state.py
  http.py
  metadata.py
  sources/{moxfield,mtgtop8,deckbox}.py
src/mtgdeck/
  data.py
  metadata.py
  legality.py
  card2vec.py
  vae.py
  recommend.py
scripts/curate_decks.py
notebooks/genrec.ipynb
demo/app.py
tests/
docs/{aws-migration-audit,cleanup-report}.md
data/          # preserved, ignored
checkpoints/   # current Oracle-ID models, ignored
README.md
requirements.txt
pytest.ini
.gitignore
```

Package `__init__.py` files and private local environment/tool directories are omitted from this view.

## Major removals

- Root `score_decklist.py`, `streamlit_app.py`, the old Oracle embedding tensor, SetTransformer checkpoint and isotonic calibrator. Removed their dedicated LFS rules; the old HTML export remains deleted.
- `models/`: obsolete scoring baselines, heuristics, MiniLM/embedding experiments, DeepSets and SetTransformer notebooks, checkpoints, plots and generated analysis.
- `helpers/`: removed redundant collector generations, one-off corpus builders, obsolete diagnostics, unused Game Knights/Archidekt and tournament benchmark collectors, and inactive Limited bulk ingestion. The modern scraper was moved and consolidated first.
- `scripts/collect_decks.py`: retired the competing live collector; preserved older-record import in offline preparation and tournament metadata parsing in the MTGTop8 adapter.
- `scripts/separate_limited_embedding_corpus.py`: retired the one-off separation utility. Existing Limited datasets were preserved.
- `client/`: moved the current app to `demo/app.py`; removed the stale HTML demo and redundant launchers/documentation.
- Three pre-Oracle-ID checkpoints: `attention_vae.pt`, `attention_vae_clean.pt`, `attention_variational_frozen_896.pt`.
- Tests solely for retired Limited ingestion, the old error log, stale notebook execution outputs, generated migration inventory/hash files, and stale caches.

About **14.04 GiB** was removed from the old paths, chiefly unused model artifacts. Deletion accounting covers 204 files, including former paths of relocated code; this is logical file size, not a filesystem free-space measurement.

## Consolidation

| Former functionality | Current home |
| --- | --- |
| Modern scraper CLI, scheduling and source circuits | `scrape/scraper.py` |
| JSONL writers, temporary SQLite dedup indexes, durable seen IDs, atomic checkpoint writes and restartable format migrations | `scrape/state.py` |
| Retry/backoff, Retry-After and repeated-page validation | `scrape/http.py` |
| Moxfield/MTGTop8/Deckbox parsing and discovery | `scrape/sources/` |
| Canonical record creation, normalization, quantities, date parsing and corpus fingerprints | `src/mtgdeck/data.py` |
| Shared Oracle reading, discovery profiles and default snapshot selection | `src/mtgdeck/metadata.py` |
| Scryfall refresh command | `scrape/metadata.py` |
| Older-harvest import, manifest subset selection, legal-deck curation and cross-input deduplication | `scripts/curate_decks.py` |
| Current training/evaluation and manual EDHREC comparisons | `notebooks/genrec.ipynb` |
| Current checkpoint-powered Streamlit interface | `demo/app.py` |

The default Commander manifest connects modern scraper outputs and existing canonical data to the notebook's prepared input. Preparation now protects its input/metadata paths against output collisions. Companion zones survive normalized imports. The shared Oracle selector includes the refreshed undated snapshot that the previous notebook/demo selectors missed.

No model architecture, loss, masking, optimizer, evaluation algorithm or recommendation-ranking change was made. The notebook now explicitly requires prepared data instead of retaining its old raw-data/model-path fallback.

## Intentionally retained

- **All 281 local `data/` files**, including older harvests, Limited corpora and generated data/model assets, because cleanup was not authorization to delete user datasets. Unselected data is not part of the default training manifest.
- **Three current `attention_oracleid_v2_*_896.pt` checkpoints**, including the deterministic and frozen ablations, because the current notebook and demo use them.
- **Existing scraper filenames and migration routines**, including `diverse_scraper.checkpoint.json`, `.diverse_scraper.seen.sqlite3`, and the `cooccurence` directory spelling, because changing them could strand progress. The single-writer assumption is explicit; no new state migration was executed.
- **Count-free embedding contexts**, which are a genuinely different derived representation. The README makes clear that the current notebook builds Card2Vec input from its own training split instead.
- **Older-record normalization and metadata fields**, so preserved local harvests remain usable without retaining their collectors.
- **Manual EDHREC comparison cells**, because they evaluate the current GenRec models; their saved execution outputs were cleared.
- **`joblib`, plotting and notebook dependencies**, because current baselines/research use them. Removed the unused scikit-learn requirement and obsolete dependency labeling. SciPy remains an explicit GenSim numerical dependency.
- **`.env`, the local virtual environment and the user-provided instruction file**, kept private/ignored. The instruction file is not application configuration.

## Verification

- Full surviving suite: `.venv/Scripts/python.exe -m pytest -q` — **101 passed in 11.49 seconds**.
- Refactored scraper/data/legality subset also passed in Linux Python: **86 passed**.
- Added/adjusted coverage for manifest selection, cross-source deduplication, optional/missing inputs, input-overwrite prevention, companion preservation, shared Oracle selection and retained tournament metadata parsing. Existing retry, pagination, interruption, deduplication and migration coverage survives under the new modules.
- All three command entry points passed `--help` in the installed Windows environment.
- Streamlit AppTest on CPU loaded the preserved model and current Oracle metadata, then successfully executed the recommendation button and rendered two tables without application exceptions or API deprecation warnings.
- All retained Python files and all notebook code cells parsed successfully; notebook execution counts/outputs were cleared.
- All 281 data files matched their pre-cleanup filenames, sizes and nanosecond modification times. `.env` and all three retained model checkpoints matched pre-cleanup SHA-256 hashes. Dataset contents were not exhaustively rehashed.
- Git whitespace checks passed and active state, datasets, models and secrets are ignored. No live scrape or full training run was started.

## Before AWS migration

See [the updated migration handoff](aws-migration-audit.md). Remaining priorities are an actual OOM diagnosis and resource profile, bounded/incremental scraper state, enforced writer ownership and service shutdown recovery, versioned dataset/model bundles, a locked environment and resumable training command, and demo concurrency/capacity testing. The Streamlit width deprecations are also worth removing when the deployment version is pinned.

No AWS capacity or cost estimate has been validated.
