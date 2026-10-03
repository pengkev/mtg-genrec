"""Execute only v3.2 review-tagged cells, preserving historical notebook outputs."""
import json
from pathlib import Path


def render(root):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    import nbformat

    path = Path(root) / 'notebooks/card2vec_static_embeddings.ipynb'
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    selected = [c for c in notebook.cells if c.cell_type == 'code'
                and 'static-v3.2-review' in c.metadata.get('tags', [])]
    if not selected:
        raise ValueError('No v3.2 review cells')
    for count, cell in enumerate(selected, 1):
        print(f'Render v3.2 review {count}/{len(selected)}', flush=True)
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(cell.source, store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Review failed: {captured.stdout}\n{captured.stderr}') from (result.error_before_exec or result.error_in_exec)
        outputs = []
        for name, value in [('stdout',captured.stdout),('stderr',captured.stderr)]:
            if value:
                outputs.append(nbformat.v4.new_output('stream',name=name,text=value))
        outputs.extend(nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata or {})
                       for o in captured.outputs)
        cell.outputs = outputs
        cell.execution_count = count
    nbformat.validate(notebook)
    nbformat.write(notebook,path)


if __name__ == '__main__':
    render(Path(__file__).resolve().parents[1])
