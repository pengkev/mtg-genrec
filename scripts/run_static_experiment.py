"""Execute a selected research stage without executing the notebook."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['train','evaluate','report'],required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/card2vec/static_v2')
    parser.add_argument('--jobs',type=int,default=2)
    args = parser.parse_args()
    from mtgdeck import static_experiment as experiment
    if args.stage=='train':
        experiment.train_experiment(ROOT,args.output,args.jobs)
    elif args.stage=='evaluate':
        experiment.evaluate_experiment(ROOT,args.output,args.jobs)
    else:
        experiment.build_report(ROOT,args.output)
