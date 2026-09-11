"""Build an allowlisted Space artifact from GitHub source and pinned assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtgdeck.artifacts import asset_entries, load_manifest, verify_assets

# These existing modules are required by the inference import graph. Copy them
# byte-for-byte at build time; never maintain a second mtgdeck implementation.
INFERENCE_MODULES = ("__init__.py", "data.py", "metadata.py", "legality.py", "vae.py", "inference.py", "artifacts.py")
SOURCE_FILES = {
    "demo/app.py": "app.py", "demo/adapter.py": "adapter.py",
    "demo/space/README.md": "README.md", "demo/space/requirements.txt": "requirements.txt",
    "demo/space/artifacts.json": "artifacts.json",
    **{f"src/mtgdeck/{name}": f"mtgdeck/{name}" for name in INFERENCE_MODULES},
}


def build(output: Path, assets_dir: Path | None = None, revision: str | None = None) -> Path:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    manifest = load_manifest(ROOT / "demo" / "space" / "artifacts.json")
    if assets_dir is None:
        asset_revision = revision or manifest.get("revision")
        if not asset_revision or not re.fullmatch(r"[0-9a-f]{40}", asset_revision):
            raise ValueError("Pin the full Hub asset commit as revision in demo/space/artifacts.json before a remote build")
        from huggingface_hub import snapshot_download
        assets_dir = Path(snapshot_download(
            repo_id=manifest["repo_id"], repo_type=manifest["repo_type"],
            revision=asset_revision,
            allow_patterns=[entry["path"] for entry in asset_entries(manifest)],
        ))
    verify_assets(manifest, assets_dir)
    output.mkdir(parents=True, exist_ok=True)
    for source, destination in SOURCE_FILES.items():
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    for entry in asset_entries(manifest):
        destination = output / entry["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(assets_dir / entry["path"], destination)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    (output / "source.json").write_text(json.dumps({"repository": "https://github.com/pengkev/mtg-genrec", "commit": source_sha, "dirty": dirty}, indent=2) + "\n")
    # Hub's upload CLI manages binary blobs; GitHub never tracks model weights.
    (output / ".gitattributes").write_text("artifacts/** filter=lfs diff=lfs merge=lfs -text\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "space")
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--asset-revision", help="Optional immutable Hub commit override")
    args = parser.parse_args()
    output = build(args.output, args.assets_dir, args.asset_revision)
    print(f"Built {len(list(output.rglob('*.*')))} files in {output}")
