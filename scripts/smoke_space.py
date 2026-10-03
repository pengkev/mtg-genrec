"""Exercise the real Gradio API and CSV; optionally compare with direct inference."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

from gradio_client import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mtgdeck.inference import DEFAULT_COMMANDER, DEFAULT_DECK, RECOMMENDATION_COLUMNS


def exercise(url: str, app_dir: Path | None = None) -> dict:
    client = Client(url, verbose=False)
    from mtgdeck.inference import checkpoint_format, serving_checkpoints
    choices = next(component["props"]["choices"] for component in client.config["components"]
                   if component["type"] == "dropdown" and component["props"].get("label") == "API checkpoint")
    paths = [Path(choice[1] if isinstance(choice, (list, tuple)) else choice) for choice in choices]
    checkpoints = serving_checkpoints(paths)
    checked = []
    if app_dir:
        served = client.predict(api_name="/deployment")
        assert served == json.loads((app_dir / "source.json").read_text())
        import torch
        from mtgdeck.artifacts import load_manifest
        from mtgdeck.inference import load_bundle, prepare_request, recommend
        torch.set_num_threads(2)
        manifest = load_manifest(app_dir / "artifacts.json")
    for path in checkpoints:
        checkpoint = path.as_posix()
        format_name = checkpoint_format(path)
        commander = DEFAULT_COMMANDER if format_name == "commander" else ""
        deck = DEFAULT_DECK if format_name == "commander" else "4 Lightning Bolt\n8 Mountain"
        result, visible, status, download = client.predict(
            checkpoint, commander, deck, 25, False, 1, 42, api_name="/recommend",
        )
        assert result["headers"] == RECOMMENDATION_COLUMNS, result
        rows = result["data"]
        assert len(rows) == 25, status
        assert len(visible["data"]) == (11 if format_name == "commander" else 2)
        assert "Generated 25 recommendations." in status
        present = {line.split(" ", 1)[1] for line in deck.splitlines()}
        assert not {row[1] for row in rows} & present
        if format_name == "commander":
            assert all(row[3] == "Colorless" or set(row[3]) <= set("UBG") for row in rows)
        with open(download, encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        assert [row["Card"] for row in csv_rows] == [row[1] for row in rows]
        if app_dir:
            entry = next(item for item in manifest["checkpoints"] if item["name"] == checkpoint)
            bundle = load_bundle(str(app_dir / entry["path"]), str(app_dir / manifest["oracle"]["path"]), "cpu", str(app_dir / manifest["eligibility"]["path"]))
            assert bundle["format"] == format_name
            request = prepare_request(bundle, commander, deck)
            expected = recommend(bundle, request.partial, 25, False, 1, 42)
            assert [row[1] for row in rows] == [row["Card"] for row in expected]
            assert all(abs(row[2] - direct["Score"]) < 0.001 for row, direct in zip(rows, expected))
            assert all(bundle["catalog"].resolve(row[1])["legalities"].get(format_name) == "legal" for row in rows)
        failed = client.predict(checkpoint, "Not a real Commander name", "Not a real card name", 25, False, 1, 42, api_name="/recommend")
        assert not failed[0]["data"] and failed[3] is None
        checked.append({"format": format_name, "recommendations": len(rows), "csv": "passed", "invalid_input": "passed", "top_card": rows[0][1]})
    return {"url": url, "formats": checked}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-dir", type=Path)
    parser.add_argument("--url", help="Existing local or live Gradio URL")
    args = parser.parse_args()
    if args.url:
        print(json.dumps(exercise(args.url, args.app_dir), indent=2))
    else:
        if not args.app_dir:
            parser.error("--app-dir or --url is required")
        original_files = {p.relative_to(args.app_dir) for p in args.app_dir.rglob("*") if p.is_file()}
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        env = {**os.environ, "GRADIO_SERVER_PORT": str(port), "GRADIO_SERVER_NAME": "127.0.0.1", "GRADIO_ANALYTICS_ENABLED": "False", "MTG_DEVICE": "cpu"}
        with tempfile.TemporaryFile(mode="w+") as log:
            process = subprocess.Popen([sys.executable, "-B", "app.py"], cwd=args.app_dir, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                for _ in range(120):
                    if process.poll() is not None:
                        raise RuntimeError("Gradio process exited during startup")
                    try:
                        with urllib.request.urlopen(url + "/config", timeout=2):
                            break
                    except (OSError, TimeoutError):
                        time.sleep(0.5)
                else:
                    raise TimeoutError("Gradio did not start within 60 seconds")
                print(json.dumps(exercise(url, args.app_dir), indent=2))
            except Exception:
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        resulting_files = {p.relative_to(args.app_dir) for p in args.app_dir.rglob("*") if p.is_file()}
        if resulting_files != original_files:
            raise RuntimeError(f"Smoke test changed the deployment file set: {resulting_files ^ original_files}")
