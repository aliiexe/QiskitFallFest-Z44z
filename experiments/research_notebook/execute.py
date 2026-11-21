"""Execute the .ipynb with a fresh existing project kernel; keep mode outputs separate."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import uuid
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['saved','quick','full'],default='saved')
    p.add_argument('--seeds',default='0,1,2')
    args=p.parse_args()
    path=ROOT/'notebooks/QFF_Quantum_Crop_Research.ipynb'
    notebook=nbformat.read(path,as_version=4)
    config=next(c for c in notebook.cells if c.cell_type=='code' and c.source.startswith('RUN_MODE ='))
    config.source=config.source.replace("RUN_MODE = 'saved'",f"RUN_MODE = {args.mode!r}",1)
    seeds=[int(s) for s in args.seeds.split(',')]
    config.source=config.source.replace('TRAINING_SEEDS = [0, 1, 2]',f'TRAINING_SEEDS = {seeds!r}',1)
    # Existing kernel specification points at ROOT/.venv/bin/python.
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/qff-research-notebook-mpl')
    os.environ.setdefault('IPYTHONDIR','/private/tmp/qff-research-notebook-ipython')
    os.environ.setdefault('JUPYTER_RUNTIME_DIR','/private/tmp/qff-research-notebook-runtime')
    for variable in ('MPLCONFIGDIR','IPYTHONDIR','JUPYTER_RUNTIME_DIR'):Path(os.environ[variable]).mkdir(parents=True,exist_ok=True)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output=ROOT/'outputs/research_notebook_verification'/tag;output.mkdir(parents=True,exist_ok=False)
    client=NotebookClient(notebook,timeout=600,kernel_name='crop-challenge',
                          resources={'metadata':{'path':str(ROOT/'notebooks')}},record_timing=True)
    start=perf_counter()
    def before_cell(cell,cell_index,**kwargs):
        first=cell.source.splitlines()[0][:90] if cell.source else ''
        print(f'[{args.mode}] cell {cell_index+1}/{len(notebook.cells)} {first}',flush=True)
    client.on_cell_start=before_cell
    try:
        client.execute()
        errors=[(i,o) for i,c in enumerate(notebook.cells) for o in c.get('outputs',[]) if o.output_type=='error']
        assert not errors
        code_cells=[c for c in notebook.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code_cells)
        pngs=sum('image/png' in o.get('data',{}) for c in code_cells for o in c.get('outputs',[]))
        assert pngs==4,'The compact ring notebook should contain exactly four essential inline figures'
        report={'status':'passed','mode':args.mode,'seeds':seeds,'kernel':'crop-challenge',
                'executed_code_cells':len(code_cells),'notebook_cells':len(notebook.cells),
                'inline_png_outputs':pngs,'wall_seconds':perf_counter()-start,
                'fresh_kernel':True,'errors':0}
        notebook.metadata['execution_verification']=report
        nbformat.validate(notebook)
        destination=path if args.mode=='saved' else output/f'QFF_Quantum_Crop_Research_{args.mode}.ipynb'
        nbformat.write(notebook,destination)
        report['executed_notebook']=str(destination.relative_to(ROOT))
        (output/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
    except BaseException:
        nbformat.write(notebook,output/f'failed_{args.mode}.ipynb')
        print('Failed execution preserved in',output,flush=True)
        raise


if __name__=='__main__':main()
