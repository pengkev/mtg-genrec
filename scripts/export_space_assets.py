"""Export inference assets from local training outputs; never upload raw datasets."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from mtgdeck.artifacts import sha256_file
from mtgdeck.inference import available_checkpoints
from mtgdeck.metadata import default_oracle_path, iter_oracle_cards

MODEL_CONFIG_KEYS = {
    "variational", "experiment", "model_dim", "heads", "blocks", "latent_dim",
    "pool_queries", "decoder_queries", "initial_logit_scale",
}
# Every field used by OracleCatalog / CommanderCandidateIndex, including alias
# preference and commander-pair rules. Preserve every card, its order and faces.
ORACLE_FIELDS = {
    "name", "oracle_id", "color_identity", "type_line", "oracle_text", "legalities",
    "games", "layout", "lang", "released_at", "card_faces",
}


def export_assets(output: Path, manifest_path: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)

    def store_asset(path: Path, name: str, suffix: str) -> dict:
        digest = sha256_file(path)
        relative = f"artifacts/{digest}{suffix}"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        return {"name": name, "path": relative, "sha256": digest, "size": destination.stat().st_size}

    manifest = {"schema_version": 1, "repo_id": "pengkev/mtg-genrec", "repo_type": "space", "checkpoints": []}
    checkpoints = available_checkpoints(ROOT / "checkpoints")
    if not checkpoints:
        raise FileNotFoundError("No Oracle-ID checkpoints in checkpoints/")
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        for path in checkpoints:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            # Keep tensors and vocabulary exactly; exclude local paths and all
            # training-only settings. No optimizer, corpus or Card2Vec sidecars.
            minimal = {key: checkpoint[key] for key in ("state_dict", "vocab", "epoch", "val_metrics") if key in checkpoint}
            minimal["config"] = {key: value for key, value in checkpoint.get("config", {}).items() if key in MODEL_CONFIG_KEYS}
            saved = staging / "model.pt"
            torch.save(minimal, saved)
            manifest["checkpoints"].append(store_asset(saved, path.name, ".pt"))
        oracle = staging / "oracle.jsonl.gz"
        with oracle.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for card in iter_oracle_cards(default_oracle_path(ROOT / "data")):
                compact = {key: value for key, value in card.items() if key in ORACLE_FIELDS}
                compressed.write((json.dumps(compact, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        manifest["oracle"] = store_asset(oracle, "oracle_cards.jsonl.gz", ".jsonl.gz")
        manifest["eligibility"] = store_asset(ROOT / "data" / "commander_eligible_oracle_ids.json", "commander_eligible_oracle_ids.json", ".json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "space-assets")
    parser.add_argument("--manifest", type=Path, default=ROOT / "demo" / "space" / "artifacts.json")
    args = parser.parse_args()
    manifest = export_assets(args.output, args.manifest)
    print(f"Exported {len(manifest['checkpoints'])} checkpoints and Oracle reference metadata to {args.output}")
