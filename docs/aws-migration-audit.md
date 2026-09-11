# GenRec AWS migration handoff

Updated after the repository cleanup on 2026-09-10. This describes the retained application and the decisions to take to an AWS expert; it does not provision infrastructure or specify a priced architecture. Resource measurements below were taken during the preceding local audit and are baselines, not post-cleanup production benchmarks. The old generated file inventory and code-hash evidence were removed because the cleanup invalidated their paths and contents.

## Current application boundary

The project now has one pipeline:

1. `python -m scrape.metadata` refreshes Scryfall Oracle cards and commander eligibility.
2. `python -m scrape.scraper` collects Moxfield, MTGTop8 and Deckbox records. `scrape/sources/` owns source parsing/discovery; `scrape/http.py` handles retries and pagination validation; `scrape/state.py` owns durable output, deduplication and search checkpoints.
3. `python scripts/curate_decks.py --manifest configs/commander.json` selects existing input files and formats, normalizes older records, checks legality and deduplicates accepted decks. It writes the canonical training dataset and rejection/report artifacts.
4. `notebooks/genrec.ipynb` trains Card2Vec and the attention VAE, evaluates held-out-card completion and saves Oracle-ID checkpoints.
5. `python -m streamlit run demo/app.py` serves recommendations from a trained checkpoint and card metadata.

There is no second live collector, old power-scoring application, SetTransformer artifact loader, or old experimental model directory. The algorithms in `src/mtgdeck/{card2vec,vae,recommend,legality}.py` remain the current GenRec stack.

## Storage and data contracts

| Asset | Purpose and durability |
| --- | --- |
| `data/format_corpora/*.jsonl` | Append-only full deck records, with format corrections performed by restartable migrations when necessary. Durable source data. |
| `data/format_corpora/deckbox.jsonl` | Shared Deckbox corpus; individual records retain format labels for manifest selection. |
| `data/cooccurence/embedding_corpus.jsonl` | Derived unique normalized card-name contexts. Current training builds Card2Vec contexts from its own training split instead. |
| `data/diverse_scraper.checkpoint.json` | Active discovery queues and page progress. Keep alongside corpora and seen state. |
| `data/format_corpora/.diverse_scraper.seen.sqlite3` | Persistent processed source IDs, including skipped records. Do not confuse with temporary dedup indexes. |
| Temporary SQLite indexes | Reconstructed from corpus JSONL at startup; bounded page cache; disposable on restart. |
| `data/decks.jsonl` | Existing canonical harvest, accepted as one input by the Commander manifest. |
| `data/decks_clean.jsonl` | Derived legal Commander training dataset; preparation replaces this output. |
| `data/*_rejected.jsonl`, `data/*_report.json` | Preparation diagnostics and dataset provenance. |
| Oracle snapshot and commander-eligibility JSON | Shared metadata needed by preparation/training/inference. Pin immutable versions for deployable model bundles. |
| `data/card2vec_clean_oracleid_v2_896.model` and NumPy sidecars | Active representation-learning assets; sidecars must travel with the model. |
| `checkpoints/attention_oracleid_v2_*.pt` | Current demo/training checkpoints. |

Schema v1 preserves source/deck identity, URL, format, date, board zones, card quantities and metadata. Preparation attaches Oracle IDs; model tokens use those IDs. Companions are retained but their legality is not inferred. Storage deduplication distinguishes all zones; training/legality fingerprints have task-specific semantics. Existing hashes, filenames and checkpoint layout were retained rather than migrating active state during cleanup.

All 281 files under local `data/` were preserved during cleanup. They include older harvests, Limited corpora and older generated models that the default pipeline does not consume. These are not an instruction to copy everything into every cloud workload. Select the required assets explicitly. Three current Oracle-ID checkpoints and `.env` were also preserved. Secrets and local assets are ignored by Git.

The default manifest reads available Commander/cEDH inputs, including the modern scraper's output. It reports absent optional inputs and fails if none exist. This closes the previous disconnect between the scraper output and notebook input. Stop collection while preparing a consistent dataset snapshot; do not train while preparation is replacing its outputs.

## Resource evidence and unresolved OOM diagnosis

The user's OOM appeared to happen at scraper startup. No kernel OOM event or definitive failing allocation was captured, so the cause is still unconfirmed.

The preceding offline startup profile indexed 676,664 embedding records and 357,875 format records. Moving corpus identity/content deduplication to a shared temporary SQLite index reduced measured peak RSS from about 677 MiB to 330.4 MiB; corpus indexing alone fell from about 316.8 MiB to 71.9 MiB. Startup time rose from about 149 to 177.7 seconds. These were Linux Python 3.14 diagnostic measurements with local data, not cloud capacity targets.

The active checkpoint was about 84 MiB with 54 buckets, 232,176 searches and 229,341 queued searches. It is still loaded as one Python object graph and serialized in full after completed pages. At this size, 1,000 full checkpoint writes imply roughly 82 GiB of logical writes before filesystem/device effects. Temporary SQLite also writes during indexing. Moving to AWS does not itself remove this write amplification or unbounded queue growth.

The previous Windows CPU inference smoke used one fine-tuned checkpoint (~111.8 MiB), with Oracle snapshot and eligibility bringing the minimal asset bundle to about 135.3 MiB. The model had 30,110 vocabulary entries and 28,898,241 parameters. Loading took 3.51 seconds; the first query took about 706.5 ms; five warm queries had a median around 23.5 ms and maximum around 24.7 ms. Process RSS was about 1,436.5 MiB with peak 1,539.4 MiB. This was one offline process, not a Streamlit HTTP, concurrent-user or load test. Metadata selection was subsequently unified during cleanup, so remeasure the final pinned deployment bundle.

The local training machine previously reported 12 logical CPUs, 63.79 GiB RAM and an RTX 5060 Ti with 15.93 GiB VRAM. The notebook loads the full dataset, constructs grouped splits and retains training/evaluation structures in memory. Hardware presence is not evidence of measured peak training demand.

## Decisions and engineering work before deployment

1. **Measure actual workload limits.** Capture OOM/exit diagnostics, peak RSS, scratch usage, queue growth, daily corpus growth and checkpoint write volume. Profile a complete training run and concurrent demo traffic before choosing compute capacity or estimating costs.
2. **Define durable scraper ownership.** The implementation assumes one writer per corpus/checkpoint and has no cross-process lock or lease. It flushes on Ctrl+C; add explicit service shutdown/SIGTERM handling and test bounded shutdown before using managed task termination. Keep corpus flush-before-checkpoint ordering.
3. **Bound state growth and checkpoint writes.** Consider incremental durable search state and persistent dedup indexes, with a separately tested migration from the current JSON checkpoint. Do not reset or discard local progress to deploy the cleaned code. Test interrupted writes, partial JSONL tails and recovery with realistic state sizes.
4. **Separate workload lifecycles.** Scraping needs long-running outbound requests and durable state; preparation/training need versioned datasets and batch resources; the demo needs only model/metadata assets and request-serving capacity. Decide their scheduling, restart policies and access boundaries independently.
5. **Make runs reproducible.** Add a locked environment and an explicit training entry point/configuration. Preserve optimizer, scheduler, RNG, epoch and data-split state for resumable training. Current model checkpoints are not a complete training-resume contract.
6. **Publish immutable model bundles.** Record checkpoint, vocabulary/identity schema, Oracle snapshot, commander eligibility, dependency versions and hashes together. Preparation, training and demo now share snapshot selection, but defaulting to the refreshed snapshot is not version pinning.
7. **Test dataset publication.** Preparation uses temporary output files but publishes the accepted file, rejected file and report separately; these are not one transaction. Use versioned run directories/manifests and a final publication marker before concurrent consumers read cloud outputs. Its deduplication set also grows with accepted data.
8. **Package and operate the demo.** Add the agreed deployment packaging, health checks, authentication if required, bounded model caching and concurrency/load tests. The present Streamlit app lets users choose models/devices; memory can grow with multiple cached bundles. No container, infrastructure-as-code or CI deployment was added by cleanup.
9. **Plan source access and observability.** Validate each source from the deployment network, keep per-source rate limits and Retry-After behavior, and define response/error/blocked-source alerts. Cloud placement does not guarantee source access. Keep credentials outside images and source control.
10. **Select assets and recovery objectives.** Decide which preserved datasets to migrate, retention, backups, restore tests and acceptable recovery point/time. Keep raw source records, derived training snapshots and deployable model bundles distinct. Estimate storage, compute, outbound networking and operational costs with the expert after these inputs are known.

## Verification scope

Cleanup verification covers the full surviving offline test suite, CLI help/import checks, notebook syntax, a demo smoke check, and preservation of local data/state and active model files. It does not include live scraping, a new full training run, cloud deployment, an actual OOM reproduction or cloud sizing validation. See `cleanup-report.md` for final test results and the deletion/consolidation record.
