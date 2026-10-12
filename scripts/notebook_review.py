"""Shared execution of explicitly tagged notebook review cells."""
import copy
import json
from pathlib import Path
import re


def execute_review_cells(notebook, tag, *, reject_nan=False):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    original = copy.deepcopy(notebook['cells'])
    indices = [i for i, cell in enumerate(notebook['cells'])
               if cell['cell_type'] == 'code'
               and tag in cell.get('metadata', {}).get('tags', [])]
    if not indices:
        raise ValueError(f'No explicitly tagged review cells: {tag}')
    shell = InteractiveShell.instance()
    for count, index in enumerate(indices, 1):
        cell = notebook['cells'][index]
        print(f'Review cell {count}/{len(indices)}', flush=True)
        with capture_output(stdout=True, stderr=True, display=True) as captured:
            result = shell.run_cell(''.join(cell['source']), store_history=False)
        error = result.error_before_exec or result.error_in_exec
        if error:
            raise RuntimeError(f'Review cell {index} failed: {captured.stdout}\n{captured.stderr}') from error
        outputs = [{'output_type': 'stream', 'name': name, 'text': value}
                   for name, value in [('stdout', captured.stdout), ('stderr', captured.stderr)] if value]
        for rich in captured.outputs:
            if reject_nan and re.search(r'>\s*(?:NaN|nan)\s*<', rich.data.get('text/html', '')):
                raise ValueError(f'Review cell {index} produced a NaN table cell')
            outputs.append({'output_type': 'display_data', 'data': rich.data,
                            'metadata': rich.metadata or {}})
        cell['outputs'] = outputs
        cell['execution_count'] = count
    if any(cell != original[i] for i, cell in enumerate(notebook['cells']) if i not in indices):
        raise AssertionError('Non-review notebook cells changed')
    return notebook


def render_notebook(path, tag, *, reject_nan=False):
    import nbformat

    path = Path(path)
    notebook = json.loads(path.read_text(encoding='utf-8'))
    execute_review_cells(notebook, tag, reject_nan=reject_nan)
    nbformat.validate(nbformat.from_dict(notebook))
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
