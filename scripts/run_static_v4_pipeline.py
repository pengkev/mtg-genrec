"""Print readiness/plan by default; --run explicitly executes cached stages."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtgdeck.static_pipeline import RUN_NAME, build_plan, preflight, run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--run-name', default=RUN_NAME)
    args = parser.parse_args()
    plan = build_plan(ROOT, args.run_name)
    print(json.dumps({'preflight': preflight(ROOT), 'plan': plan}, indent=2))
    if args.run:
        run_pipeline(plan, ROOT)


if __name__ == '__main__':
    main()
