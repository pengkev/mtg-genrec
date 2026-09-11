"""Gradio entry point, used unchanged locally and in the generated HF Space."""

from __future__ import annotations

import os
import json
from pathlib import Path
import sys

import gradio as gr
import spaces
import torch

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR if (APP_DIR / "mtgdeck").exists() else ROOT / "src"))
sys.path.insert(0, str(APP_DIR))

from adapter import generate, resolve_device_mode
from mtgdeck.artifacts import load_manifest, verify_assets
from mtgdeck.inference import (
    DEFAULT_COMMANDER, DEFAULT_DECK, RECOMMENDATION_COLUMNS, VISIBLE_COLUMNS,
    _checkpoint_label, available_checkpoints, load_bundle, recommend,
)
from mtgdeck.legality import OracleCatalog
from mtgdeck.metadata import default_oracle_path


# Device choice is server configuration: models are never reloaded per request.
DEVICE_MODE = resolve_device_mode(os.environ)
DEVICE = "cpu" if DEVICE_MODE == "cpu" else "cuda"
torch.set_num_threads(int(os.environ.get("MTG_CPU_THREADS", "2")))

if (APP_DIR / "artifacts.json").exists():
    manifest = load_manifest(APP_DIR / "artifacts.json")
    verify_assets(manifest, APP_DIR)
    checkpoint_paths = [(item["name"], APP_DIR / item["path"]) for item in manifest["checkpoints"]]
    oracle_path = APP_DIR / manifest["oracle"]["path"]
    eligibility_path = APP_DIR / manifest["eligibility"]["path"]
else:
    checkpoint_paths = [(path.name, path) for path in available_checkpoints(ROOT / "checkpoints")]
    oracle_path = default_oracle_path(ROOT / "data")
    eligibility_path = ROOT / "data" / "commander_eligible_oracle_ids.json"
if not checkpoint_paths:
    raise RuntimeError("No Oracle-ID checkpoints found. See docs/space-deployment.md for setup.")

# Construct/load once at module scope. ZeroGPU CUDA emulation handles the
# model.to('cuda') in load_bundle here, before a real GPU is requested.
CATALOG = OracleCatalog.from_path(oracle_path, eligibility_path)
BUNDLES = {
    name: load_bundle(str(path), str(oracle_path), DEVICE, catalog=CATALOG)
    for name, path in checkpoint_paths
}


def score(checkpoint, partial, count, sample, draws, seed):
    return recommend(BUNDLES[checkpoint], partial, count, sample, draws, seed)


if DEVICE_MODE == "zerogpu":
    @spaces.GPU(duration=5)
    def gpu_score(checkpoint, partial, count, sample, draws, seed):
        return score(checkpoint, partial, count, sample, draws, seed)

    SCORE = gpu_score
else:
    SCORE = score


def generate_recommendations(checkpoint, commander, deck, count, sample, draws, seed):
    return generate(BUNDLES, SCORE, checkpoint, commander, deck, count, sample, draws, seed)


def model_details(checkpoint):
    bundle = BUNDLES[checkpoint]
    saved, model = bundle["checkpoint"], bundle["model"]
    metrics = saved.get("val_metrics", {})
    detail = (
        f"Epoch {saved.get('epoch', '?')} · "
        f"{'Variational' if model.variational else 'Deterministic'} latent · "
        f"{model.card_dim}-d embeddings · {len(bundle['vocab']):,} vocabulary entries · {DEVICE_MODE.upper()}"
    )
    if metrics:
        detail += f" · Validation Recall@20: {metrics.get('Recall@20', 0):.3f}"
    return detail


def change_model(checkpoint):
    return model_details(checkpoint), gr.Checkbox(value=False, interactive=BUNDLES[checkpoint]["model"].variational), gr.Slider(value=1, interactive=False)


def deployment_info() -> dict:
    path = APP_DIR / "source.json"
    return json.loads(path.read_text()) if path.exists() else {"commit": None, "dirty": True}


def create_demo():
    first = checkpoint_paths[0][0]
    with gr.Blocks(title="MTG GenRec", delete_cache=(3600, 3600)) as app:
        gr.api(deployment_info, api_name="deployment", api_visibility="undocumented", queue=False)
        gr.Markdown("# MTG variational deck completion\nPaste a Commander and any partial deck. The model ranks legal missing cards; scores are relative model logits, not probabilities or power ratings.")
        checkpoint = gr.Dropdown([(_checkpoint_label(Path(name)), name) for name, _ in checkpoint_paths], value=first, label="Checkpoint")
        details = gr.Markdown(model_details(first))
        with gr.Row():
            with gr.Column(scale=1):
                commander = gr.Textbox(value=DEFAULT_COMMANDER, label="Commander(s)", lines=3, info="One commander per line. Partner pairs are supported.")
                count = gr.Slider(5, 100, value=25, step=5, label="Recommendations")
                sample = gr.Checkbox(value=False, label="Sample the variational latent z", interactive=BUNDLES[first]["model"].variational, info="Off uses the stable posterior mean; on samples the learned posterior.")
                draws = gr.Slider(1, 16, value=1, step=1, label="Latent draws to average", interactive=False)
                seed = gr.Number(value=42, minimum=0, maximum=2_147_483_647, precision=0, label="Sampling seed")
            with gr.Column(scale=2):
                deck = gr.Textbox(value=DEFAULT_DECK, label="Partial mainboard", lines=14, info="Accepts 1 Card Name, 1x Card Name, or one bare name per line. Commander/Deck/Sideboard headings are recognized.")
        submit = gr.Button("Recommend missing cards", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        results = gr.Dataframe(headers=RECOMMENDATION_COLUMNS, datatype=["number", "str", "number", "str", "str"], interactive=False, label="Recommendations")
        download = gr.File(label="Download CSV", interactive=False)
        with gr.Accordion("Resolved partial deck", open=False):
            visible = gr.Dataframe(headers=VISIBLE_COLUMNS, datatype=["str", "number", "str"], interactive=False)
        gr.Markdown("Commander legality and color identity are enforced. The notebook's optional commander-count hybrid is not included because its training index is not stored in the checkpoint.")
        checkpoint.change(change_model, checkpoint, [details, sample, draws], api_visibility="private")
        sample.change(lambda enabled: gr.Slider(interactive=enabled), sample, draws, api_visibility="private")

        def submit_request(checkpoint, commander, deck, count, sample, draws, seed):
            results, visible, status, produced = generate_recommendations(checkpoint, commander, deck, count, sample, draws, seed)
            if produced is None:
                return results, visible, status, None
            try:
                cached = download.move_resource_to_block_cache(produced)
            finally:
                Path(produced).unlink(missing_ok=True)
            return results, visible, status, cached

        # Serial scoring protects the original global torch sampling seed.
        submit.click(submit_request, [checkpoint, commander, deck, count, sample, draws, seed], [results, visible, status, download], api_name="recommend", concurrency_limit=1, concurrency_id="inference")
    return app.queue(max_size=32, default_concurrency_limit=1)


demo = create_demo()

if __name__ == "__main__":
    default_host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", default_host), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
