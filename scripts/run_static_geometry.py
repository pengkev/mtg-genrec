"""Explicit, isolated stages for static_v3. No historical writes."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def render_archive(root, output):
    """Render archived review cells without changing the frozen geometry module."""
    import nbformat
    from nbclient import NotebookClient
    from mtgdeck import static_geometry as v3

    root, _, output = v3.validate(root, output)
    archive = output / 'notebook_executed.ipynb'
    if archive.exists():
        raise FileExistsError('Executed archive exists; move it aside explicitly before rendering a newer review')
    original = nbformat.read(root / 'notebooks/archive/card2vec_static_embeddings.ipynb', as_version=4)
    review = nbformat.v4.new_notebook(cells=[c for c in original.cells
                                          if 'static-v3-review' in c.metadata.get('tags', [])])
    if not review.cells:
        raise ValueError('No v3 review cells')
    review.cells.insert(0, nbformat.v4.new_code_cell(
        f'from pathlib import Path\nV3_OUTPUT_OVERRIDE = Path({str(output)!r})'))
    NotebookClient(review, timeout=600, kernel_name='python3',
                   resources={'metadata': {'path': str(root)}}).execute()
    nbformat.write(review, archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=[
        'audit', 'matrices', 'svd', 'sgns', 'baselines', 'svd-evaluate',
        'compatibility', 'probes', 'report', 'render'])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/card2vec/static_v3')
    parser.add_argument('--allow-expensive', action='store_true', help='Required for computation beyond audit/report/review')
    parser.add_argument('--max-disk-gib', type=float, default=12, help='Uncompressed sparse shard budget')
    parser.add_argument('--threads', type=int, default=2, help='BLAS threads; models run sequentially')
    args = parser.parse_args()
    if args.stage not in ('audit', 'report', 'render') and not args.allow_expensive:
        parser.error('This stage requires --allow-expensive; default notebook behavior is saved-output review')
    if args.max_disk_gib <= 0 or args.threads < 1:
        parser.error('Resource budgets must be positive')
    from threadpoolctl import threadpool_limits
    from mtgdeck import static_geometry as v3
    with threadpool_limits(limits=args.threads):
        if args.stage == 'audit':
            v3.audit(ROOT, args.output)
        elif args.stage == 'matrices':
            v3.build_matrices(ROOT, args.output, args.max_disk_gib)
        elif args.stage == 'svd':
            v3.build_svd(ROOT, args.output)
        elif args.stage in ('sgns', 'baselines', 'svd-evaluate', 'compatibility'):
            v3.evaluate(ROOT, args.output, family={'svd-evaluate': 'svd'}.get(args.stage, args.stage))
        elif args.stage == 'probes':
            v3.run_probes(ROOT, args.output)
        elif args.stage == 'report':
            v3.build_report(ROOT, args.output)
        else:
            print(render_archive(ROOT, args.output))


if __name__ == '__main__':
    main()
