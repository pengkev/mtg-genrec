# Premium deck collection

`python -m scrape.premium` collects only the eight Moxfield accounts, Andrea
Mengucci's TCGplayer author feed, the Archidekt folder 614906 and its public
subfolders, and MTGTop8. The allowlist lives in `configs/premium_sources.json`.
Deckbox and general Moxfield discovery are excluded from this collection.
Only Legacy, Modern, and multiplayer EDH are collected across all premium
sources. EDH uses `commander` for creator sites and `cedh` for MTGTop8. Other
formats, including Duel Commander, cannot be selected with `--formats`.

Run from the repository root, with scraper dependencies installed and the local
Oracle snapshot available:

```powershell
.\.venv\Scripts\python.exe -m scrape.premium --max-pages 1 --limit 2
# Full collection, resumable by repeating the command:
.\.venv\Scripts\python.exe -m scrape.premium
# Revisit completed listings to discover new decks:
.\.venv\Scripts\python.exe -m scrape.premium --refresh
# Tournament-only collection:
.\.venv\Scripts\python.exe -m scrape.premium --sources mtgtop8
```

Moxfield uses the existing `MOXFIELD_USER_AGENT` configuration. Requests use
bounded retries and pacing; denied requests remain failures rather than empty
successful collections. `--formats legacy modern` restricts collection by
format. `--max-pages` caps each existing source/format bucket or each additional
creator source; `--limit` caps new decks per existing source/format, or per
TCGplayer/Archidekt source. Small page caps may visit only the first Moxfield
creator; subsequent runs resume the queue. Archidekt traverses folders breadth
first, including an initially empty root folder.

All outputs are isolated under `data/premium/`:

- `formats/<format>.jsonl`: canonical lists with quantities, zones and provenance.
- `embedding_corpus.jsonl`: unique card sets for co-occurrence training.
- `sources.json`: the source selection used for this corpus.
- Checkpoints and seen-ID databases: resume partial pages without appending duplicates.

A changed allowlist requires a new `--output-dir`, preventing old selections from
silently remaining in a new collection. Existing broad corpus files and scraper
schedule are unchanged. Do not run two premium collectors concurrently. The
existing Windows operational wrapper targets the general scraper, not this new
entry point.

Creator rows carry `metadata.quality_tier`, `creator` and `curation_url`.
Archidekt also preserves `folder_path` and `theorycrafted`; its collection includes
Brewing, Deck Doctor and Patreon Deck Reviews, which may contain experimental or
submitted lists. TCGplayer preserves test-deck and available event-result fields.
MTGTop8 event metadata is enabled automatically. Scraping does not infer wins,
legality, or a numerical strength label from creator reputation.

Archidekt currently maps Standard, Modern, Commander, Legacy, Vintage and Pauper;
an unknown format stops that source with a resumable error. TCGplayer and
Archidekt reject lists containing unknown Oracle names rather than dropping cards.
Refresh the local Oracle snapshot if this occurs. Completed source IDs are not
re-downloaded by `--refresh`; use a new output directory for revised deck snapshots.
Exact duplicate lists retain the first record's provenance in format corpora.

## Using the data for refinement

Use the quantity-preserving format corpora for deck cohesion or deck-level
fine-tuning. Keep formats separate, balance creators against tournament volume,
and use tournament results as stronger evidence of competitive strength than
creator selection alone. Filter test decks, theorycrafts and review submissions
before assigning positive strength labels. Group duplicates and close deck
variants together when splitting train/evaluation data to reduce leakage.

For the existing Commander legality pipeline, the premium-only manifest excludes
all older broad-source inputs:

```powershell
.\.venv\Scripts\python.exe scripts/curate_decks.py --manifest configs/premium_commander.json --output data/premium/decks_clean.jsonl
```

This manifest includes Commander and cEDH, not Duel Commander or 60-card formats.
Use the resulting clean corpus as an explicit training input. Collection alone
does not retrain models or demonstrate improved strength/cohesion; compare a
held-out evaluation before replacing a model checkpoint.
