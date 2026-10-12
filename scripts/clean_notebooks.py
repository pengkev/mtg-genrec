"""Clear generated output from active notebooks, preserving historical archives."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Report output without changing files')
    args = parser.parse_args()
    dirty = []
    for path in sorted((Path(__file__).resolve().parents[1] / 'notebooks').glob('*.ipynb')):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        changed = False
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                changed |= bool(cell.get('outputs')) or cell.get('execution_count') is not None
                cell['outputs'] = []
                cell['execution_count'] = None
        if changed:
            dirty.append(path.name)
            if not args.check:
                path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    if dirty:
        print(('Generated output: ' if args.check else 'Cleaned: ') + ', '.join(dirty))
    return int(args.check and bool(dirty))


if __name__ == '__main__':
    raise SystemExit(main())
