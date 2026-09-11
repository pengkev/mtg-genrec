import json
from pathlib import Path

import pytest

from mtgdeck.artifacts import load_manifest, sha256_file, verify_assets
from scripts import build_space


def test_build_is_allowlisted_and_verifies_assets(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    assets = tmp_path / "assets"
    assets.mkdir()
    payload = assets / "fixture"
    payload.write_bytes(b"test-only asset")
    digest = sha256_file(payload)
    relative = f"artifacts/{digest}.pt"
    (assets / "artifacts").mkdir()
    payload.rename(assets / relative)
    entry = {"name": "fixture", "path": relative, "sha256": digest, "size": 15}
    manifest = {"schema_version": 1, "checkpoints": [entry], "oracle": entry, "eligibility": entry}
    for name in build_space.SOURCE_FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest) if name.endswith("artifacts.json") else "source\n")
    (root / ".env").write_text("not for deployment")
    (root / "training.pt").write_bytes(b"not for deployment")
    monkeypatch.setattr(build_space, "ROOT", root)
    monkeypatch.setattr(build_space.subprocess, "check_output", lambda cmd, **kw: "a" * 40 if "rev-parse" in cmd else "")
    output = build_space.build(tmp_path / "space", assets)
    files = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    assert files == {*build_space.SOURCE_FILES.values(), relative, "source.json", ".gitattributes"}
    for source, destination in build_space.SOURCE_FILES.items():
        assert (root / source).read_bytes() == (output / destination).read_bytes()
    assert not json.loads((output / "source.json").read_text())["dirty"]
    with pytest.raises(ValueError, match="empty"):
        build_space.build(output, assets)
    with pytest.raises(ValueError, match="Pin the full Hub asset commit"):
        build_space.build(tmp_path / "unpinned-space")
    (assets / relative).write_bytes(b"tampered asset!")
    with pytest.raises(ValueError, match="checksum"):
        verify_assets(manifest, assets)


def test_manifest_rejects_path_traversal(tmp_path):
    entry = {"path": "artifacts/../../secret.pt", "sha256": "a" * 64}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "checkpoints": [entry], "oracle": entry, "eligibility": entry}))
    with pytest.raises(ValueError, match="path"):
        load_manifest(path)


def test_export_strips_training_details_without_changing_tensors_or_aliases(tmp_path, monkeypatch):
    import torch
    from scripts import export_space_assets
    from mtgdeck.metadata import iter_oracle_cards
    root = tmp_path / "repo"
    (root / "checkpoints").mkdir(parents=True)
    (root / "data").mkdir()
    checkpoint = {"state_dict": {"weights": torch.arange(6)}, "vocab": {"oid:card": 2},
                  "config": {"variational": True, "data": "private/local/path", "epochs": 99},
                  "epoch": 4, "val_metrics": {"Recall@20": 0.4}, "optimizer": {"ignored": True}}
    torch.save(checkpoint, root / "checkpoints/attention_oracleid_v2_variational_finetuned_896.pt")
    card = {"name": "Front // Back", "oracle_id": "card", "color_identity": ["U"],
            "type_line": "Legendary Creature", "oracle_text": "Partner", "games": ["paper"],
            "legalities": {"commander": "legal"}, "lang": "en", "layout": "transform",
            "released_at": "2026-01-01", "card_faces": [{"name": "Front"}, {"name": "Back"}],
            "prices": {"usd": "irrelevant"}}
    (root / "data/oracle_cards.json").write_text(json.dumps([card]))
    (root / "data/commander_eligible_oracle_ids.json").write_text('["card"]')
    monkeypatch.setattr(export_space_assets, "ROOT", root)
    output, manifest_path = tmp_path / "assets", tmp_path / "manifest.json"
    manifest = export_space_assets.export_assets(output, manifest_path)
    verify_assets(load_manifest(manifest_path), output)
    saved = torch.load(output / manifest["checkpoints"][0]["path"], weights_only=True)
    assert torch.equal(saved["state_dict"]["weights"], checkpoint["state_dict"]["weights"])
    assert saved["vocab"] == checkpoint["vocab"]
    assert saved["val_metrics"] == checkpoint["val_metrics"]
    assert saved["config"] == {"variational": True}
    assert "optimizer" not in saved
    compact = list(iter_oracle_cards(output / manifest["oracle"]["path"]))
    assert compact == [{key: value for key, value in card.items() if key != "prices"}]
