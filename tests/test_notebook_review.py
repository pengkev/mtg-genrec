"""Review rendering must preserve history and never execute compute cells."""
import nbformat
import pytest

from scripts.notebook_review import render_notebook


def test_tagged_review_preserves_other_cells(tmp_path):
    path = tmp_path / 'review.ipynb'
    frozen = nbformat.v4.new_code_cell('raise AssertionError("compute executed")',
        outputs=[nbformat.v4.new_output('stream', name='stdout', text='saved result\n')],
        execution_count=7)
    review = nbformat.v4.new_code_cell('print("review")', metadata={'tags': ['review']})
    nbformat.write(nbformat.v4.new_notebook(cells=[frozen, review]), path)
    render_notebook(path, 'review')
    updated = nbformat.read(path, as_version=4)
    assert updated.cells[0] == frozen
    assert updated.cells[1].outputs[0].text == 'review\n'


@pytest.mark.parametrize('source, tag, error', [
    ('print("unused")', 'missing', ValueError),
    ('raise RuntimeError("failed review")', 'review', RuntimeError),
    ('from IPython.display import display, HTML\ndisplay(HTML("<td>NaN</td>"))', 'review', ValueError),
])
def test_rejected_review_leaves_file_unchanged(tmp_path, source, tag, error):
    path = tmp_path / 'review.ipynb'
    cell = nbformat.v4.new_code_cell(source, metadata={'tags': ['review']})
    nbformat.write(nbformat.v4.new_notebook(cells=[cell]), path)
    before = path.read_bytes()
    with pytest.raises(error):
        render_notebook(path, tag, reject_nan=True)
    assert path.read_bytes() == before
