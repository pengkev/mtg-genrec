"""Run isolated relation-aware stages against frozen static_v2 models."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare', 'matrices', 'svd', 'sgns', 'baselines', 'svd-evaluate', 'compatibility', 'probes', 'report'])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/card2vec/static_v3_relations_v1')
    parser.add_argument('--allow-expensive', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--max-disk-gib', type=float, default=12)
    args = parser.parse_args()
    if args.threads < 1 or args.max_disk_gib <= 0:
        parser.error('Resource limits must be positive')
    if args.stage not in ('prepare', 'report') and not args.allow_expensive:
        parser.error('Computation requires --allow-expensive')
    from threadpoolctl import threadpool_limits
    from mtgdeck import static_relations as r
    with threadpool_limits(limits=args.threads):
        if args.stage == 'prepare':
            r.prepare(ROOT, args.output)
        else:
            r.validate(ROOT, args.output)
            if args.stage == 'matrices':
                if not (args.output / 'matrices/complete.json').exists():
                    r.g.build_matrices(ROOT, args.output, args.max_disk_gib)
            elif args.stage == 'svd':
                r.g.build_svd(ROOT, args.output)
            elif args.stage == 'probes':
                r.probes(ROOT, args.output)
            elif args.stage == 'report':
                print(r.report(ROOT, args.output))
            else:
                r.evaluate(ROOT, args.output, {'svd-evaluate': 'svd'}.get(args.stage, args.stage))


if __name__ == '__main__':
    main()
