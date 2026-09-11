# Gradio Space deployment

The source of truth is `pengkev/mtg-genrec` on GitHub, branch `main`.
`pengkev/mtg-genrec` on Hugging Face is a generated deployment, not another
codebase. Never clone it into this repository or edit its application manually.
Scraping, training and evaluation retain their existing local architecture.

## Runtime and behavior

`demo/app.py` calls `demo/adapter.py` for Gradio formatting and
`src/mtgdeck/inference.py` for the parsing, loading and ranking functions extracted
from the previous Streamlit demo. The model and legality implementations are
unchanged. The UI retains all three checkpoint choices, Commander/partner input,
partial-deck sections and quantities, unknown/out-of-vocabulary notices, 5–100
recommendations, posterior mean or seeded sampling, 1–16 draws, resolved-deck and
score tables, validation metadata and CSV export. Scores remain raw relative
logits rounded to four decimals; there is no new calibration or hybrid ranking.
Requests are serialized to preserve the original global PyTorch sampling seed.
Invalid requests clear stale output and downloads. CSV files expire from the
Gradio cache after an hour.

CPU is the local default. On the local machine with two CPU threads, normal
requests took 18–31 ms and the default
deck with 16 draws took about 275 ms. A 100-card input with 16 draws took
0.63–0.68 seconds in Linux. CUDA normal requests took 5–7 ms and 16 draws about
60 ms. CPU is already interactive and avoids GPU allocation/quota overhead.
Hardware differs on the Space; these are local measurements, not live latency
guarantees. Sampling parity is evaluated on the same device; PyTorch CPU and
CUDA random streams are not interchangeable.

The target Space uses ZeroGPU. CPU performance is sufficient, but ZeroGPU
refused startup without a GPU function, and the hardware API refused a switch
to free `cpu-basic` without PRO. No paid hardware or subscription was enabled.
The app detects `SPACES_ZERO_GPU=true` and selects `zerogpu` automatically;
explicit CPU mode on ZeroGPU hardware is rejected with a clear setup error.

`MTG_DEVICE=zerogpu` can also explicitly enable ZeroGPU inference. `spaces` is imported
before models are constructed; all three models are loaded and placed on CUDA
at module scope. Only the scoring function has `@spaces.GPU(duration=5)`. Five
seconds provides headroom over the measured inference times; benchmark on the
host before tuning further. There is no per-request loading or lazy CUDA move.
`MTG_DEVICE=cuda` supports a normal local GPU. See the official
[ZeroGPU loading pattern](https://huggingface.co/docs/hub/spaces-zerogpu).

## Deployment boundary

`scripts/build_space.py` copies an explicit allowlist into an empty build
directory: `app.py`, `adapter.py`, Space README/requirements/asset manifest, and
the seven required `mtgdeck` modules. These modules are copied byte-for-byte from
`src/mtgdeck`; no second implementation is maintained. `data.py`, `vae.py` and
`legality.py` also contain local development helpers, but are required to define
the existing inference types and model. Their presence does not require a
training corpus or training packages. `card2vec.py` and `recommend.py` are not
shipped because the demo does not import their training/baseline logic.

The only binary assets are three existing VAE checkpoints (~112 MiB each), a
compact Oracle reference snapshot (~5 MiB) and commander eligibility (~162 KiB).
Export strips local paths and training settings, preserves tensors/vocab/metrics,
and includes every Oracle record and every field used by the catalog and legality
rules. No deck dataset, optimizer, notebook, test, scraper, separate Card2Vec
model, credential or cache is included. The total is approximately 340.5 MiB.
Keep all generated output under ignored `build/`; never Git-add it.

`demo/space/artifacts.json` records SHA-256, byte length and content-addressed
Hub paths. After initial upload it also pins the immutable Hub revision. Both
builder and app verify checksums. The Space stores these inference assets;
GitHub stores only their manifest. Source is never pulled from the Space.
`source.json` in each deployment records the GitHub source commit and whether
the build source was dirty. Production deployments must use clean GitHub source.

## Dependencies

Space direct requirements: `gradio==6.27.0`, `spaces==0.51.3`,
`torch==2.11.0`, `numpy==2.5.3`. Gradio supplies its normal transitive web,
dataframe and Hub-client packages. Python is 3.12, matching the existing Space
metadata. The test job installs separate `requirements-test.txt`; the deploy
job installs only the Space requirements plus `huggingface_hub==1.31.0` for CI.
Gensim, SciPy, joblib, Jupyter, matplotlib, Streamlit and scraping packages are
not required by the application. CPU-only PyTorch can be installed locally/CI
first from its official CPU wheel index; the Space keeps the regular wheel for
ZeroGPU compatibility.

## First asset publication or a new locally trained model

Authenticate with `hf auth login` using a fine-grained token with write access
only to `spaces/pengkev/mtg-genrec`. Never put a token in a command argument,
source file, notebook, `.env` committed to Git, or documentation.

```bash
python scripts/export_space_assets.py
python scripts/build_space.py --output build/space --assets-dir build/space-assets
python -m pytest -q
python scripts/smoke_space.py --app-dir build/space
hf upload pengkev/mtg-genrec build/space-assets . --repo-type space \
  --commit-message "Publish inference assets"
```

Record the resulting full Hub commit SHA as `revision` in
`demo/space/artifacts.json`, then commit the manifest and any source changes to
GitHub. The first publication contains only inference assets, without changing
the Space app. Subsequent CI builds fetch these exact pinned files and verify
their SHA-256. Asset refresh is an intentional local export/publication; CI never
trains, scrapes or regenerates a live Oracle snapshot. Export into a new empty
directory for each refresh to avoid uploading obsolete local assets.

## Automatic GitHub deployment

`.github/workflows/deploy-space.yml` runs the full test suite for pull requests
and pushes. Only successful `main` runs can deploy. The deploy job creates a
minimal artifact from the same checkout and pinned Hub assets, starts Gradio,
checks a real recommendation and CSV against direct inference, then runs:

```bash
hf upload pengkev/mtg-genrec build/space . --repo-type space --delete '*'
```

This mirrors the complete generated artifact, removing stale deployment files.
All current pinned assets are included in that upload. Nothing is pushed to
Hugging Face with Git. This uses the CLI build-step option recommended in
[Hugging Face's GitHub synchronization documentation](https://huggingface.co/docs/hub/spaces-github-actions).
The workflow disables Python bytecode writes and excludes `**/__pycache__/*`
and `*.pyc` from uploads. The smoke test also checks that it leaves the staging
file set unchanged.

Prefer a [Trusted Publisher](https://huggingface.co/docs/hub/trusted-publishers):

1. Open the Space's **Settings → Trusted Publishers**.
2. Add provider **GitHub Actions** with repository `pengkev/mtg-genrec`, branch
   `main`, and workflow `deploy-space.yml`.
3. The workflow has `id-token: write` and uses
   `HF_OIDC_RESOURCE=spaces/pengkev/mtg-genrec` when no token secret is set.
   No long-lived secret is required.

If Trusted Publishing is unavailable, add GitHub Actions secret `HF_TOKEN`
containing a fine-grained token scoped to this Space. The workflow supports
that fallback. The default GitHub `GITHUB_TOKEN` needs only `contents: read`.

For an intentional redeploy, use GitHub **Actions → Test and deploy Gradio
Space → Run workflow**, selecting `main`. Tests and the real smoke request run
again before upload. The workflow waits for the live Space and exercises its API.
To reproduce locally from pinned assets without local training files:

```bash
python scripts/build_space.py --output build/reproduction
python scripts/smoke_space.py --app-dir build/reproduction
```

## Migration verification

All 101 original tests plus 19 adapter/build/export tests passed (120 total). A separate
migration parity check compared the original Streamlit functions and exported
Gradio inference on CPU: all 12 checkpoint/sampling cases matched exactly,
including ordered card names and rounded scores. Every model tensor and
vocabulary entry and every catalog name/face alias matched. Local Gradio API
checks returned 25 recommendations, 11 resolved cards and a matching CSV;
invalid commander input cleared results. The default top result was
`Opulent Palace` with score `10.9694`.
