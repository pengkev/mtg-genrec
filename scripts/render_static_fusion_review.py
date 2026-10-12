"""Render saved-result review cells without executing training cells."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook_review import render_notebook

def render(root):
    render_notebook(Path(root) / 'notebooks/archive/card2vec_static_embeddings.ipynb', 'static-v3.2-review')


if __name__ == '__main__':
    render(Path(__file__).resolve().parents[1])
