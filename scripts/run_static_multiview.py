"""Export separate views, evaluate fixed fusion, or review saved v4 results."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mtgdeck import static_multiview as multiview
from mtgdeck import static_v4 as v4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['export', 'evaluate', 'render'])
    parser.add_argument('--output', type=Path, default=v4.DEFAULT_ROOT / 'multiview_evaluation_s42')
    parser.add_argument('--export', type=Path, default=v4.DEFAULT_EXPORT)
    parser.add_argument('--deck-export', type=Path, default=ROOT / 'artifacts/card2vec/static_export_v1')
    parser.add_argument('--semantic', action='append', default=[], metavar='LABEL=PATH')
    parser.add_argument('--primary', default='mlm', choices=multiview.ABLATIONS)
    parser.add_argument('--mechanical', type=Path, default=v4.DEFAULT_ROOT / 'mechanical_s42')
    parser.add_argument('--oracle', type=Path, default=v4.DEFAULT_ROOT / 'oracle')
    parser.add_argument('--spellbook', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--probes', action='store_true')
    parser.add_argument('--allow-expensive', action='store_true')
    args = parser.parse_args()
    if args.stage in ('export', 'evaluate') and not args.allow_expensive:
        parser.error(f'{args.stage} requires --allow-expensive')
    if args.stage == 'export':
        semantics = {}
        for value in args.semantic:
            label, separator, path = value.partition('=')
            if not separator or label in semantics:
                parser.error('Use unique --semantic LABEL=PATH arguments')
            semantics[label] = Path(path)
        multiview.export(args.deck_export, semantics, args.mechanical, args.export, args.seed, args.primary)
    elif args.stage == 'evaluate':
        multiview.evaluate(ROOT, args.export, args.oracle, args.output, args.probes, args.spellbook)
    else:
        from render_static_v4_review import render
        render(ROOT)


if __name__ == '__main__':
    main()
