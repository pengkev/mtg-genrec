"""Execute only saved-result review cells; never execute training/evaluation cells."""
from pathlib import Path
import json
import re

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]


def execute_review_cells(notebook):
    shell=InteractiveShell.instance()
    indices=[i for i,c in enumerate(notebook['cells']) if c['cell_type']=='code'
             and 'second-experiment-review' in c.get('metadata',{}).get('tags',[])]
    if not indices:
        raise RuntimeError('No explicitly tagged review cells')
    for count,index in enumerate(indices,1):
        cell=notebook['cells'][index]
        print(f'Review cell {count}/{len(indices)}',flush=True)
        with capture_output(stdout=True,stderr=True,display=True) as captured:
            result=shell.run_cell(''.join(cell['source']),store_history=False)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(f'Review cell {index} failed: {captured.stdout}\n{captured.stderr}') from (result.error_before_exec or result.error_in_exec)
        outputs=[]
        if captured.stdout:outputs.append({'output_type':'stream','name':'stdout','text':captured.stdout})
        if captured.stderr:outputs.append({'output_type':'stream','name':'stderr','text':captured.stderr})
        for rich in captured.outputs:
            html=rich.data.get('text/html','')
            if re.search(r'>\s*(?:NaN|nan)\s*<', html):
                raise ValueError(f'Review cell {index} produced a NaN table cell')
            outputs.append({'output_type':'display_data','data':rich.data,'metadata':rich.metadata or {}})
        cell['outputs']=outputs
        cell['execution_count']=count
    return notebook


if __name__=='__main__':
    path=ROOT/'notebooks/card2vec_static_embeddings.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    notebook=execute_review_cells(notebook)
    path.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    print('Saved review outputs only; all compute cells were excluded.',flush=True)
