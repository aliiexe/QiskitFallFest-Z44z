"""Fresh-kernel checks of notebook two, without touching notebook one."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
from time import perf_counter
import uuid
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]
TARGET=ROOT/'notebooks/QFF_PCA_SVM_Feature_Research.ipynb'
REFERENCE=ROOT/'notebooks/QFF_Quantum_Crop_Research.ipynb'


def execute(mode,repeats):
    notebook=nbformat.read(TARGET,as_version=4)
    reference=nbformat.read(REFERENCE,as_version=4)
    before=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    assert len(notebook.cells)==len(reference.cells)==25
    assert [c.cell_type for c in notebook.cells]==[c.cell_type for c in reference.cells]
    configuration=next(c for c in notebook.cells if c.cell_type=='code' and c.source.startswith('RUN_MODE ='))
    configuration.source=configuration.source.replace("RUN_MODE = 'saved'",f'RUN_MODE = {mode!r}',1)
    configuration.source=configuration.source.replace('CV_REPEATS = [0, 1, 2]',f'CV_REPEATS = {repeats!r}',1)
    if 0 not in repeats:configuration.source=configuration.source.replace('MATRIX_REPEAT = 0',f'MATRIX_REPEAT = {repeats[0]}',1)
    for variable,folder in [('MPLCONFIGDIR','qff-pca-svm-mpl'),('IPYTHONDIR','qff-pca-svm-ipython'),('JUPYTER_RUNTIME_DIR','qff-pca-svm-runtime')]:
        os.environ.setdefault(variable,str(Path('/private/tmp')/folder))
        Path(os.environ[variable]).mkdir(parents=True,exist_ok=True)
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    output=ROOT/'outputs/pca_svm_notebook_verification'/tag
    output.mkdir(parents=True,exist_ok=False)
    client=NotebookClient(notebook,timeout=600,kernel_name='crop-challenge',
                          resources={'metadata':{'path':str(ROOT/'notebooks')}},record_timing=True)
    def progress(cell,cell_index,**kwargs):
        if cell.cell_type=='code':print(f'[{mode}] cell {cell_index+1}/25: {cell.source.splitlines()[0][:85]}',flush=True)
    client.on_cell_start=progress
    start=perf_counter()
    try:
        client.execute()
        code=[c for c in notebook.cells if c.cell_type=='code']
        assert len(code)==12 and all(c.execution_count is not None for c in code)
        assert not any(o.output_type=='error' for c in code for o in c.outputs)
        images=sum('image/png' in o.get('data',{}) for c in code for o in c.outputs)
        assert images==4
        final_text='\n'.join(o.get('text','') for o in notebook.cells[-1].outputs if o.output_type=='stream')
        match=re.search(r'New plots, metrics and provenance: (.+)',final_text)
        assert match,final_text
        session_path=Path(match.group(1).strip())/'session.json'
        session=json.loads(session_path.read_text())
        assert session['status']=='complete' and session['previous_files_unchanged']
        assert session['archived_OOF_files_rescored']==3*len(repeats)
        assert len(session['full_reproduction_checks'])==(9*len(repeats) if mode=='full' else 0)
        assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest()==before
        report={'status':'passed','mode':mode,'repeats':repeats,'split_seeds':[2026+r for r in repeats],
                'fresh_kernel':True,'kernel':'crop-challenge','notebook_cells':25,'executed_code_cells':12,
                'inline_figures':images,'errors':0,'same_cell_layout_as_first_notebook':True,
                'first_notebook_sha256_unchanged':before,
                'archived_OOF_files_rescored':session['archived_OOF_files_rescored'],
                'full_outer_folds_parameters_inner_scores_predictions_reproduced':len(session['full_reproduction_checks']),
                'session_record':str(session_path.relative_to(ROOT)),
                'wall_seconds':perf_counter()-start}
        notebook.metadata['execution_verification']=report
        nbformat.validate(notebook)
        destination=TARGET if mode=='saved' else output/f'QFF_PCA_SVM_Feature_Research_{mode}.ipynb'
        nbformat.write(notebook,destination)
        report['executed_notebook']=str(destination.relative_to(ROOT))
        (output/'verification.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        return report
    except BaseException:
        nbformat.write(notebook,output/f'failed_{mode}.ipynb')
        print('Failed execution saved in',output,flush=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes',default='saved',help='Comma-separated saved,quick,full modes; each gets a fresh kernel')
    parser.add_argument('--repeats',default='0,1,2')
    args=parser.parse_args()
    modes=args.modes.split(',');repeats=[int(r) for r in args.repeats.split(',')]
    if any(m not in ['saved','quick','full'] for m in modes):parser.error('Unknown mode')
    for mode in modes:execute(mode,repeats)


if __name__=='__main__':main()
