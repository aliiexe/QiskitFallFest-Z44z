"""Check saved-model demonstration in fresh project kernels; no training/search."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
import uuid
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'notebooks/QFF_Ring_Demo.ipynb'
TEMPLATE=ROOT/'notebooks/Qiskit_QuantumAfrica_Crop_Challenge.ipynb'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def execute(seed):
    notebook=nbformat.read(TARGET,as_version=4);original=nbformat.read(TEMPLATE,as_version=4)
    assert len(notebook.cells)==len(original.cells)==18
    assert [c.cell_type for c in notebook.cells]==[c.cell_type for c in original.cells]
    config=next(c for c in notebook.cells if c.cell_type=='code' and 'DEMO_SEED = 0' in c.source)
    config.source=config.source.replace('DEMO_SEED = 0',f'DEMO_SEED = {seed}',1)
    tracked=[ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p]
    locked={str(p.relative_to(ROOT)):sha(p) for p in tracked if p.is_file() and p!=TARGET}
    for k,v in [('MPLCONFIGDIR','qff-simple-demo-mpl'),('IPYTHONDIR','qff-simple-demo-ipython'),('JUPYTER_RUNTIME_DIR','qff-simple-demo-runtime')]:
        os.environ.setdefault(k,str(Path('/private/tmp')/v));Path(os.environ[k]).mkdir(parents=True,exist_ok=True)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    out=ROOT/'outputs/ring_demo_verification'/tag;out.mkdir(parents=True,exist_ok=False)
    client=NotebookClient(notebook,kernel_name='crop-challenge',timeout=300,
                          resources={'metadata':{'path':str(ROOT/'notebooks')}},record_timing=True)
    def progress(cell,cell_index,**kwargs):
        if cell.cell_type=='code':print(f'[saved seed {seed}] cell {cell_index+1}/18: {cell.source.splitlines()[0][:70]}',flush=True)
    client.on_cell_start=progress;start=perf_counter()
    try:
        client.execute()
        code=[c for c in notebook.cells if c.cell_type=='code']
        assert len(code)==10 and all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
        assert images==5
        texts='\n'.join(o.get('text','') for c in code for o in c.outputs if o.output_type=='stream')
        assert 'Retraining skipped.' in texts
        assert 'Validation predictions/probabilities reproduce for both QNNs and SVM.' in texts
        assert all(sha(ROOT/p)==h for p,h in locked.items())
        report={'status':'passed','default_models_loaded_without_training':True,'demo_seed':seed,
                'fresh_kernel':True,'kernel':'crop-challenge','cells':18,'code_cells_executed':10,
                'inline_figures':5,'same_cell_type_order_as_actual_organizer_notebook':True,
                'template_sha256_unchanged':sha(TEMPLATE),'all_prior_tracked_files_unchanged':True,
                'both_QNNs_and_SVM_reproduce_verified_validation_predictions':True,
                'new_experiments':False,'wall_seconds':perf_counter()-start}
        notebook.metadata['execution_verification']=report
        nbformat.validate(notebook)
        target=TARGET if seed==0 else out/f'QFF_Ring_Demo_seed_{seed}.ipynb'
        nbformat.write(notebook,target)
        report['executed_notebook']=str(target.relative_to(ROOT))
        (out/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        return report
    except BaseException:
        nbformat.write(notebook,out/'failed.ipynb');print('Failed run saved in',out,flush=True);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seeds',default='0')
    for seed in [int(s) for s in p.parse_args().seeds.split(',')]:execute(seed)
