import json

import numpy as np
import pytest

from mtgdeck import static_fusion as f


def test_fusion_cosine_is_equal_block_mean():
    rng = np.random.default_rng(42)
    u,v,s = [rng.normal(size=(12,128)).astype(np.float32) for _ in range(3)]
    fused = f.fuse_sources(u,v,s)
    for name, blocks in [('concat_input_svd_256',[u,s]),('concat_input_context_svd_384',[u,v,s])]:
        score = f.g.GeometryScorer(fused[name])(3)
        expected = np.mean([f.g.GeometryScorer(block)(3) for block in blocks],axis=0)
        np.testing.assert_allclose(score,expected,atol=2e-7)


def test_report_requires_all_stages(tmp_path,monkeypatch):
    monkeypatch.setattr(f,'prepare',lambda *args:None)
    (tmp_path/'experiment.json').write_text(json.dumps({'training_seeds':[42,43,44]}))
    with pytest.raises(ValueError,match='Incomplete retrieval'):
        f.report(tmp_path,tmp_path,tmp_path/'export')
    assert not (tmp_path/'report').exists()


def test_historical_and_checkpoint_output_guards(tmp_path):
    historical = tmp_path/'artifacts/card2vec/static_v3_relations_v1'
    historical.mkdir(parents=True)
    with pytest.raises(ValueError,match='protected'):
        f.prepare(tmp_path,historical/'new',tmp_path/'checkpoint')
    with pytest.raises(ValueError,match='protected'):
        f.prepare(tmp_path,tmp_path/'checkpoint',tmp_path/'checkpoint')


def test_review_renderer_preserves_compute_and_history(tmp_path):
    import nbformat
    from scripts.render_static_fusion_review import render
    directory = tmp_path/'notebooks'
    directory.mkdir()
    path = directory/'card2vec_static_embeddings.ipynb'
    historical = nbformat.v4.new_code_cell('raise RuntimeError("historical cell executed")',
        outputs=[nbformat.v4.new_output('stream',name='stdout',text='frozen output\n')],execution_count=19)
    compute = nbformat.v4.new_code_cell('raise RuntimeError("compute executed")')
    review = nbformat.v4.new_code_cell('print("review only")\nfrom IPython.display import display, Markdown\ndisplay(Markdown("**Rendered report**"))',metadata={'tags':['static-v3.2-review']})
    notebook = nbformat.v4.new_notebook(cells=[historical,compute,review])
    nbformat.write(notebook,path)
    render(tmp_path)
    updated = nbformat.read(path,as_version=4)
    assert updated.cells[:2] == notebook.cells[:2]
    assert updated.cells[2].outputs[0].text == 'review only\n'
    assert updated.cells[2].outputs[1].data['text/markdown'] == '**Rendered report**'
