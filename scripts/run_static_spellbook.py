"""Explicit snapshot, preparation, hyperedge training and retrieval stages."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtgdeck import static_spellbook as spellbook
from mtgdeck import static_v4 as v4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['snapshot', 'prepare', 'train', 'evaluate'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--snapshot', type=Path, default=v4.DEFAULT_ROOT / 'spellbook_snapshot')
    parser.add_argument('--oracle', type=Path, default=v4.DEFAULT_ROOT / 'oracle')
    parser.add_argument('--prepared', type=Path, default=v4.DEFAULT_ROOT / 'spellbook')
    parser.add_argument('--trained', type=Path, default=v4.DEFAULT_ROOT / 'mechanical_s42')
    parser.add_argument('--deck-export', type=Path, default=ROOT / 'artifacts/card2vec/static_export_v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dimension', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--auxiliary', action='store_true')
    parser.add_argument('--split', choices=['validation', 'test'], default='test')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--allow-expensive', action='store_true')
    args = parser.parse_args()
    if args.stage == 'snapshot' and not args.allow_download:
        parser.error('snapshot requires --allow-download')
    if args.stage in ('train', 'evaluate') and not args.allow_expensive:
        parser.error(f'{args.stage} requires --allow-expensive')
    defaults = {'snapshot': 'spellbook_snapshot', 'prepare': 'spellbook', 'train': f'mechanical_s{args.seed}',
                'evaluate': f'mechanical_evaluation_s{args.seed}_{args.split}'}
    output = args.output or v4.DEFAULT_ROOT / defaults[args.stage]
    if args.stage == 'snapshot':
        spellbook.snapshot(output)
    elif args.stage == 'prepare':
        spellbook.prepare(args.snapshot, args.oracle, output, args.seed)
    elif args.stage == 'train':
        spellbook.train(args.prepared, output, args.seed, args.dimension, args.epochs, args.batch_size, args.auxiliary)
    else:
        spellbook.evaluate(args.prepared, args.trained, args.deck_export, output, args.split, args.seed)
    print(output)


if __name__ == '__main__':
    main()
