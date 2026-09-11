"""Content-addressed inference assets shared by the Space builder and loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def asset_entries(manifest: dict) -> list[dict]:
    return [*manifest["checkpoints"], manifest["oracle"], manifest["eligibility"]]


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("checkpoints"):
        raise ValueError("Expected a version 1 inference artifact manifest with checkpoints")
    for entry in asset_entries(manifest):
        relative = PurePosixPath(entry["path"])
        digest = entry["sha256"]
        if (relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2
                or relative.parts[0] != "artifacts" or "\\" in entry["path"]
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or not relative.name.startswith(digest + ".")):
            raise ValueError(f"Invalid content-addressed artifact path: {entry['path']}")
    return manifest


def verify_assets(manifest: dict, directory: Path) -> None:
    for entry in asset_entries(manifest):
        path = directory / entry["path"]
        if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Inference artifact checksum mismatch: {entry['name']}")
