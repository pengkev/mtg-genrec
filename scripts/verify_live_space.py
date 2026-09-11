"""Wait for a deployment to start, verify provenance, then call its real API."""

import argparse
import json
import time

from huggingface_hub import HfApi, hf_hub_download
from gradio_client import Client
from smoke_space import exercise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    repo = "pengkev/mtg-genrec"
    api = HfApi()
    for _ in range(90):
        runtime = api.get_space_runtime(repo)
        print(f"Space stage: {runtime.stage}", flush=True)
        if runtime.stage == "RUNNING":
            try:
                served = Client("https://pengkev-mtg-genrec.hf.space", verbose=False).predict(api_name="/deployment")
                if served.get("commit") == args.source_commit and not served.get("dirty"):
                    break
                print("Waiting for the requested source commit to be served", flush=True)
            except Exception as exc:
                print(f"Waiting for Gradio API: {type(exc).__name__}", flush=True)
        if runtime.stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR"}:
            raise RuntimeError(f"Space failed: {runtime.raw}")
        time.sleep(10)
    else:
        raise TimeoutError("Space did not reach RUNNING within 15 minutes")
    with open(hf_hub_download(repo, "source.json", repo_type="space")) as handle:
        source = json.load(handle)
    if source["commit"] != args.source_commit or source["dirty"]:
        raise RuntimeError(f"Unexpected deployment provenance: {source}")
    print(json.dumps(exercise("https://pengkev-mtg-genrec.hf.space"), indent=2))
