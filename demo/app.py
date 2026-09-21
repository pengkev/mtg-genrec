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

from adapter import (
    add_to_deck, generate, recommendation_gallery, resolve_device_mode,
    select_recommendation,
)
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
        gr.Markdown("# MTG GenRec\nNeural Commander deck completion")
        recommendations = gr.State([])
        selected = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                commander = gr.Dropdown(
                    CATALOG.commander_choices(), value=[DEFAULT_COMMANDER],
                    multiselect=True, max_choices=2, filterable=True,
                    label="Commander(s)", info="Search for one commander or a legal pair.",
                )
                deck = gr.Textbox(value=DEFAULT_DECK, label="Partial mainboard", lines=14,
                                  info="Paste Arena/Moxfield text, quantities, or one card per line.")
                submit = gr.Button("Recommend", variant="primary")
                status = gr.Textbox(label="Deck status", interactive=False)
            with gr.Column(scale=3, min_width=300):
                gallery = gr.Gallery(label="Recommended cards", columns=5, height="auto",
                                     object_fit="contain", allow_preview=False,
                                     interactive=False, elem_id="recommendation-gallery")
                selected_details = gr.HTML("Select a card to inspect it.")
                add = gr.Button("Add to deck", interactive=False, variant="primary")
        with gr.Accordion("Advanced / model settings", open=False):
            checkpoint = gr.Dropdown([(_checkpoint_label(Path(name)), name) for name, _ in checkpoint_paths], value=first, label="Checkpoint")
            details = gr.Markdown(model_details(first))
            count = gr.Slider(5, 100, value=25, step=5, label="Recommendations")
            sample = gr.Checkbox(value=False, label="Sample the variational latent z", interactive=BUNDLES[first]["model"].variational, info="Off uses the stable posterior mean; on samples the learned posterior.")
            draws = gr.Slider(1, 16, value=1, step=1, label="Latent draws to average", interactive=False)
            seed = gr.Number(value=42, minimum=0, maximum=2_147_483_647, precision=0, label="Sampling seed")
            results = gr.Dataframe(headers=RECOMMENDATION_COLUMNS, datatype=["number", "str", "number", "str", "str"], interactive=False, label="Ranked recommendations")
            download = gr.File(label="Download CSV", interactive=False)
            visible = gr.Dataframe(headers=VISIBLE_COLUMNS, datatype=["str", "number", "str"], interactive=False, label="Resolved partial deck")
            gr.Markdown("Scores are relative model logits, not probabilities or power ratings. Commander legality and color identity are enforced. The optional commander-count hybrid is not included because its training index is not stored in the checkpoint.")
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

        # Retain the text-based public API and its original four outputs.
        api_commander = gr.Textbox(value=DEFAULT_COMMANDER, visible=False)
        api_submit = gr.Button(visible=False)
        api_submit.click(submit_request, [checkpoint, api_commander, deck, count, sample, draws, seed],
                         [results, visible, status, download], api_name="recommend",
                         concurrency_limit=1, concurrency_id="inference")

        def visual_request(checkpoint, commanders, deck, count, sample, draws, seed):
            table, resolved, message, csv = submit_request(
                checkpoint, "\n".join(commanders or []), deck, count, sample, draws, seed,
            )
            records, images = recommendation_gallery(table, CATALOG)
            return (table, resolved, message, csv, records,
                    gr.Gallery(value=images, selected_index=None), None,
                    "Select a card to inspect it.", gr.Button(interactive=False))

        def select_card(records, event: gr.SelectData):
            card, detail = select_recommendation(records, event.index if event.selected else None)
            return card, detail, gr.Button(interactive=card is not None)

        def add_request(card, deck):
            updated, message = add_to_deck(deck, card, CATALOG)
            return updated, message + "\nClick Recommend when you're ready to refresh recommendations."

        inputs = [checkpoint, commander, deck, count, sample, draws, seed]
        outputs = [results, visible, status, download, recommendations, gallery, selected, selected_details, add]
        # Both entry points and selection share a queue, protecting sampling and state.
        submit.click(visual_request, inputs, outputs, api_visibility="private", concurrency_limit=1, concurrency_id="inference")
        gallery.select(select_card, recommendations, [selected, selected_details, add], api_visibility="private", concurrency_limit=1, concurrency_id="inference")
        add.click(add_request, [selected, deck], [deck, status], api_visibility="private", concurrency_limit=1, concurrency_id="inference")
    return app.queue(max_size=32, default_concurrency_limit=1)


demo = create_demo()

if __name__ == "__main__":
    default_host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    demo.launch(css="""
        #recommendation-gallery .grid-container {grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)) !important;}
        #recommendation-gallery {max-width: 1000px;}
    """, server_name=os.environ.get("GRADIO_SERVER_NAME", default_host), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
