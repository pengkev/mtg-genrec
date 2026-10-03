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
    add_to_deck, card_modal, generate, recommendation_gallery, resolve_device_mode,
    select_recommendation,
)
from card_dialog import ADD_CONFIRMATION_JS, APP_CSS, DIALOG_CSS, DIALOG_JS, DIALOG_TEMPLATE
from mtgdeck.deck_rules import companion_choices
from mtgdeck.artifacts import load_manifest, verify_assets
from mtgdeck.inference import (
    DEFAULT_COMMANDER, DEFAULT_DECK, RECOMMENDATION_COLUMNS, VISIBLE_COLUMNS,
    _checkpoint_label, available_checkpoints, serving_checkpoints, load_bundle, recommend,
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
    checkpoint_root = ROOT / "checkpoints"
    checkpoint_paths = [
        ((path.relative_to(checkpoint_root) if path.is_relative_to(checkpoint_root) else path).as_posix(), path)
        for path in available_checkpoints(checkpoint_root)
    ]
    oracle_path = default_oracle_path(ROOT / "data")
    eligibility_path = ROOT / "data" / "commander_eligible_oracle_ids.json"
ALL_CHECKPOINTS = os.environ.get("MTG_ALL_CHECKPOINTS", "").lower() in {"1", "true"} and not (APP_DIR / "artifacts.json").exists()
if not ALL_CHECKPOINTS:
    chosen_paths = serving_checkpoints([Path(name) if (APP_DIR / "artifacts.json").exists() else path for name, path in checkpoint_paths])
    chosen = {path.as_posix() for path in chosen_paths}
    # Selection must use source names, because manifest paths are content hashes.
    checkpoint_paths = [(name, path) for name, path in checkpoint_paths if (Path(name) if (APP_DIR / "artifacts.json").exists() else path).as_posix() in chosen]
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


def generate_recommendations(checkpoint, commander, deck, count, sample, draws, seed, companion=""):
    return generate(BUNDLES, SCORE, checkpoint, commander, deck, count, sample, draws, seed, companion)


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
    return (model_details(checkpoint), gr.Checkbox(value=False, interactive=BUNDLES[checkpoint]["model"].variational),
            gr.Slider(value=1, interactive=False))


def format_models(format_name):
    return [name for name, _ in checkpoint_paths if BUNDLES[name].get("format", "commander") == format_name]


def preferred_model(format_name):
    names = format_models(format_name)
    return next((name for name in names if "_premium_" in name), names[0])


def default_deck(format_name):
    return DEFAULT_DECK if format_name == "commander" else "4 Lightning Bolt\n8 Mountain"


def deployment_info() -> dict:
    path = APP_DIR / "source.json"
    return json.loads(path.read_text()) if path.exists() else {"commit": None, "dirty": True}


def create_demo():
    formats = [fmt for fmt in ("commander", "modern", "legacy") if format_models(fmt)]
    initial_format = formats[0]
    first = preferred_model(initial_format)
    with gr.Blocks(title="MTG GenRec", delete_cache=(3600, 3600)) as app:
        gr.api(deployment_info, api_name="deployment", api_visibility="undocumented", queue=False)
        gr.Markdown("# MTG GenRec\nComplete your deck. Select a format, paste a partial mainboard, and explore recommendations.")
        format_select = gr.Radio([(fmt.title(), fmt) for fmt in formats], value=initial_format, label="Format", elem_id="format-select")
        drafts = gr.State({})
        recommendations = gr.State([])
        selected = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                commander = gr.Dropdown(
                    CATALOG.commander_choices(), value=[DEFAULT_COMMANDER],
                    multiselect=True, max_choices=2, filterable=True,
                    visible=initial_format == "commander",
                    label="Commander(s)", info="Search for one commander or a legal pair.",
                )
                companion = gr.Dropdown(companion_choices(CATALOG, initial_format), value=[],
                                        multiselect=True, max_choices=1, filterable=True,
                                        label="Companion (optional)", info="Choose only when using its companion ability. Restrictions apply to the starting deck.")
                deck = gr.Textbox(value=default_deck(initial_format), label="Partial mainboard", lines=10,
                                  info="Paste Arena/Moxfield text. Include quantities; Sideboard and Companion sections are recognized.")
                submit = gr.Button("Recommend", variant="primary")
                status = gr.Textbox(label="Deck status", interactive=False)
            with gr.Column(scale=3, min_width=280):
                gallery = gr.Gallery(label="Recommended cards · click for details and to add", columns=3, height="auto", show_label=False, container=False,
                                     object_fit="contain", allow_preview=False,
                                     interactive=False, elem_id="recommendation-gallery")
        modal = gr.HTML(value="", html_template=DIALOG_TEMPLATE, css_template=DIALOG_CSS,
                        js_on_load=DIALOG_JS, elem_id="card-modal")
        with gr.Accordion("Advanced settings", open=False):
            checkpoint = gr.Dropdown([(_checkpoint_label(Path(name)), name) for name in format_models(initial_format)],
                                     value=first, label="Testing checkpoint", visible=ALL_CHECKPOINTS)
            details = gr.Markdown(model_details(first), visible=ALL_CHECKPOINTS)
            count = gr.Slider(5, 100, value=25, step=5, label="Recommendations")
            sample = gr.Checkbox(value=False, label="Sample the variational latent z", interactive=BUNDLES[first]["model"].variational,
                                 info="Off uses the stable posterior mean; on samples the learned posterior.")
            draws = gr.Slider(1, 16, value=1, step=1, label="Latent draws to average", interactive=False)
            seed = gr.Number(value=42, minimum=0, maximum=2_147_483_647, precision=0, label="Sampling seed")
            results = gr.Dataframe(headers=RECOMMENDATION_COLUMNS, datatype=["number", "str", "number", "str", "str"], interactive=False, label="Ranked recommendations")
            download = gr.File(label="Download CSV", interactive=False)
            visible = gr.Dataframe(headers=VISIBLE_COLUMNS, datatype=["str", "number", "str"], interactive=False, label="Resolved partial deck")
            gr.Markdown("Scores are relative model logits. Recommendations use snapshot format legality and selected companion restrictions. Modern and Legacy predict missing mainboard identities; Add to deck adds one copy per click. Sideboards are checked for copy limits but are not recommendation targets.")
        sample.change(lambda enabled: gr.Slider(interactive=enabled), sample, draws, api_visibility="private")

        def switch_format(format_name, previous_checkpoint, deck_text, commanders, companion_name, saved_drafts):
            saved_drafts = dict(saved_drafts)
            previous_format = BUNDLES[previous_checkpoint].get("format", "commander")
            saved_drafts[previous_format] = (deck_text, commanders, companion_name)
            text, leaders, chosen_companion = saved_drafts.get(format_name, (default_deck(format_name), [DEFAULT_COMMANDER], []))
            name = preferred_model(format_name)
            return (gr.Dropdown(choices=[(_checkpoint_label(Path(n)), n) for n in format_models(format_name)], value=name),
                    gr.Textbox(value=text), gr.Dropdown(value=leaders, visible=format_name == "commander"),
                    gr.Dropdown(choices=companion_choices(CATALOG, format_name), value=chosen_companion), saved_drafts,
                    *change_model(name), [], gr.Gallery(value=[], selected_index=None), None, "", [], [], "", None)

        # Changing format clears previous cards and CSVs, and restores that format's draft.
        format_select.change(switch_format, [format_select, checkpoint, deck, commander, companion, drafts],
                             [checkpoint, deck, commander, companion, drafts, details, sample, draws,
                              recommendations, gallery, selected, modal, results, visible, status, download],
                             api_visibility="private", concurrency_limit=1, concurrency_id="inference")

        def change_checkpoint(name):
            return (*change_model(name), [], gr.Gallery(value=[], selected_index=None), None, "", [], [], "", None)

        checkpoint.input(change_checkpoint, checkpoint,
                         [details, sample, draws, recommendations, gallery, selected, modal, results, visible, status, download],
                         api_visibility="private", concurrency_limit=1, concurrency_id="inference")

        def submit_request(checkpoint, commander, deck, count, sample, draws, seed, companion=""):
            results, visible, status, produced = generate_recommendations(checkpoint, commander, deck, count, sample, draws, seed, companion or "")
            if produced is None:
                return results, visible, status, None
            try:
                cached = download.move_resource_to_block_cache(produced)
            finally:
                Path(produced).unlink(missing_ok=True)
            return results, visible, status, cached

        # Keep the seven-input text API; Companion sections are accepted in deck text.
        api_checkpoint = gr.Dropdown(choices=[name for name, _ in checkpoint_paths], value=first, label="API checkpoint", visible=False)
        api_commander = gr.Textbox(value=DEFAULT_COMMANDER, visible=False)
        api_submit = gr.Button(visible=False)
        api_submit.click(submit_request, [api_checkpoint, api_commander, deck, count, sample, draws, seed],
                         [results, visible, status, download], api_name="recommend",
                         concurrency_limit=1, concurrency_id="inference")

        def visual_request(checkpoint, commanders, deck, count, sample, draws, seed, companion=""):
            table, resolved, message, csv = submit_request(
                checkpoint, "\n".join(commanders or []), deck, count, sample, draws, seed, next(iter(companion or []), ""),
            )
            records, images = recommendation_gallery(table, CATALOG)
            return table, resolved, message, csv, records, gr.Gallery(value=images, selected_index=None), None, ""

        def select_card(records, checkpoint, deck, companion, event: gr.SelectData):
            card, _ = select_recommendation(records, event.index if event.selected else None)
            return card, card_modal(card, CATALOG, deck, BUNDLES[checkpoint].get("format", "commander"), CATALOG.resolve(next(iter(companion or []), "")))

        def add_request(card, deck, checkpoint, companion, commanders, event: gr.EventData):
            format_name = BUNDLES[checkpoint].get("format", "commander")
            quantity = 1
            updated, message = add_to_deck(deck, card, CATALOG, format_name, quantity, next(iter(companion or []), ""), "\n".join(commanders or []))
            # Leave the dialog DOM, focus, and scroll position untouched on Add.
            return updated, message

        inputs = [checkpoint, commander, deck, count, sample, draws, seed, companion]
        outputs = [results, visible, status, download, recommendations, gallery, selected, modal]
        submit.click(visual_request, inputs, outputs, api_visibility="private", concurrency_limit=1, concurrency_id="inference")
        gallery.select(select_card, [recommendations, checkpoint, deck, companion], [selected, modal],
                       api_visibility="private", concurrency_limit=1, concurrency_id="inference")
        modal.add_copies(add_request, [selected, deck, checkpoint, companion, commander], [deck, status],
                         show_progress="hidden", api_visibility="private", concurrency_limit=1, concurrency_id="inference").then(
                             fn=None, inputs=[status], outputs=[], js=ADD_CONFIRMATION_JS,
                             api_visibility="private", show_progress="hidden",
                         )
    return app.queue(max_size=32, default_concurrency_limit=1)


demo = create_demo()

if __name__ == "__main__":
    default_host = "0.0.0.0" if os.environ.get("SPACE_ID") else "127.0.0.1"
    demo.launch(css=APP_CSS, server_name=os.environ.get("GRADIO_SERVER_NAME", default_host), server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
