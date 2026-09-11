"""Gradio output formatting; all card resolution and scoring live in mtgdeck."""

from __future__ import annotations

import csv
import tempfile
from typing import Callable, Mapping, Any

from mtgdeck.inference import RECOMMENDATION_COLUMNS, prepare_request


def resolve_device_mode(environment: Mapping[str, str]) -> str:
    """Match serving mode to the host without requesting a GPU locally."""
    zero_gpu = environment.get("SPACES_ZERO_GPU", "").lower() in {"1", "t", "true"}
    mode = environment.get("MTG_DEVICE", "zerogpu" if zero_gpu else "cpu")
    if mode not in {"cpu", "cuda", "zerogpu"}:
        raise ValueError("MTG_DEVICE must be cpu, cuda, or zerogpu")
    if zero_gpu and mode != "zerogpu":
        raise ValueError("ZeroGPU hardware requires MTG_DEVICE=zerogpu. CPU mode requires CPU hardware.")
    return mode


def generate(
    bundles: Mapping[str, Any], score: Callable, checkpoint: str,
    commander: str, deck: str, count: int, sample: bool, draws: int, seed: int,
) -> tuple[list[list[Any]], list[list[Any]], str, str | None]:
    """One queued request, including a unique per-request downloadable CSV."""
    try:
        if checkpoint not in bundles:
            raise ValueError("Select an available checkpoint.")
        request = prepare_request(bundles[checkpoint], commander, deck, count, sample, draws, seed)
        rows = score(checkpoint, request.partial, request.count, request.sample_latent, request.draws, request.seed)
    except (ValueError, AssertionError) as exc:
        # Returning empty outputs also clears any previous request's CSV/results.
        return [], [], str(exc), None
    # The UI moves this producer file into its expiring cache, then removes it.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", prefix="mtg_recommendations_", newline="", encoding="utf-8", delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=RECOMMENDATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        output_path = handle.name
    status = "\n".join([*request.notices, f"Generated {len(rows)} recommendations."])
    return [[row[column] for column in RECOMMENDATION_COLUMNS] for row in rows], request.visible, status, output_path
