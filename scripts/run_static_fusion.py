"""Explicit frozen-model v3.2 stages; never trains SGNS or rebuilds the corpus/SVD."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['export','prepare','retrieval','probes','report','render'])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/card2vec/static_v3_2_fusion_v1')
    parser.add_argument('--export', dest='checkpoint', type=Path, default=ROOT / 'artifacts/card2vec/static_export_v1')
    parser.add_argument('--allow-expensive', action='store_true')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.stage in ('retrieval','probes') and not args.allow_expensive:
        parser.error('Evaluation requires --allow-expensive')
    from threadpoolctl import threadpool_limits
    from mtgdeck import static_fusion as f
    with threadpool_limits(limits=args.threads):
        if args.stage == 'export':
            f.export(ROOT,args.checkpoint)
        elif args.stage == 'prepare':
            f.prepare(ROOT,args.output,args.checkpoint)
        elif args.stage in ('retrieval','probes'):
            f.evaluate(ROOT,args.output,args.checkpoint,args.stage)
        elif args.stage == 'report':
            print(f.report(ROOT,args.output,args.checkpoint))
        else:
            from render_static_fusion_review import render
            render(ROOT)


if __name__ == '__main__':
    main()
