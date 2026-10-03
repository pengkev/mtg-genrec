"""Execute the notebook's three format-specific base/premium training workflows."""
from pathlib import Path
import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formats', nargs='+', choices=('commander', 'edh', 'cedh', 'modern', 'legacy'),
                        default=['commander', 'modern', 'legacy'])
    parser.add_argument('--max-records', '--max-decks', dest='max_records', type=int, default=None,
                        help='Cap raw records per format/tier for smoke runs only')
    parser.add_argument('--epochs', type=int, default=8, help='Base training epochs')
    parser.add_argument('--premium-epochs', type=int, default=4)
    parser.add_argument('--run-name', required=True, help='Fresh directory under checkpoints/')
    args = parser.parse_args()
    if min(args.epochs, args.premium_epochs) < 1 or (args.max_records is not None and args.max_records < 30):
        parser.error('Use positive epoch counts and at least 30 records')
    if Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}:
        parser.error('run-name must be a single directory name')
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    cells = json.loads((root / 'notebooks/genrec.ipynb').read_text(encoding='utf-8'))['cells']
    namespace = {'display': print}
    for index, cell in enumerate(cells):
        tags = cell.get('metadata', {}).get('tags', [])
        if cell['cell_type'] != 'code' or not {'genrec_training', 'genrec_config'}.intersection(tags):
            continue
        print(f'Running cell {index}', flush=True)
        exec(compile(''.join(cell['source']), f'genrec.ipynb:cell{index}', 'exec'), namespace)
        if 'genrec_config' in tags:
            namespace['RUN_DIR'] = root / 'checkpoints' / args.run_name
            namespace['FORMATS'] = tuple(dict.fromkeys(namespace['canonical_format'](fmt) for fmt in args.formats))
            namespace['CFG']['max_records'] = args.max_records
            namespace['CFG']['base']['epochs'] = args.epochs
            namespace['CFG']['premium']['epochs'] = args.premium_epochs
            namespace['torch'].set_num_threads(4)
    print(f"Completed: {namespace['RUN_DIR']}", flush=True)


if __name__ == '__main__':
    main()
