"""Profile the unchanged twelve-angle ring trainer on training data only."""
import argparse
import cProfile
from importlib.metadata import version
import json
from pathlib import Path
import pstats
import subprocess
import sys
from time import perf_counter
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from challenge.data import load_data
from experiments.qnn_optimization.run import preprocessing,sha,now
from experiments.entanglement_topology.model import TopologyClassifier


class TimedRing(TopologyClassifier):
    def prepare_states(self,X):
        start=perf_counter();result=super().prepare_states(X)
        self.prepare_wall=perf_counter()-start
        return result

    def statevector_proba(self,states,weights,count=True):
        start=perf_counter();result=super().statevector_proba(states,weights,count)
        self.probability_wall+=perf_counter()-start
        return result


def profile(out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    tracked=[ROOT/p for p in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if p]
    protected=set(tracked)
    for folder in (ROOT/'outputs').iterdir():
        if folder.resolve()!=out and folder.is_dir():
            protected.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    locked={str(p.relative_to(ROOT)):sha(p) for p in protected if p.is_file()}
    (out/'preservation.json').write_text(json.dumps(locked,indent=2))
    tables,_,manifest,_=load_data();t=tables['train'];raw=t[manifest['features']].to_numpy();y=t.target.to_numpy()
    start=perf_counter();A=preprocessing().fit_transform(raw);prep_seconds=perf_counter()-start
    start=perf_counter();baseline=TopologyClassifier(topology='ring');construction=perf_counter()-start
    baseline.fit(A,y,seed=0)
    start=perf_counter();measured=TimedRing(topology='ring');profile_constructor=perf_counter()-start
    measured.probability_wall=0.
    p=cProfile.Profile();p.enable();measured.fit(A,y,seed=0);p.disable()
    np.testing.assert_allclose(baseline.model.weights,measured.model.weights,atol=1e-10,rtol=0)
    with np.load(ROOT/'outputs/entanglement_topology_2026-11-21/ring_seed_0_weights.npz',allow_pickle=False) as w:
        np.testing.assert_allclose(baseline.model.weights,w['final'],atol=1e-10,rtol=0)
    stats=pstats.Stats(p)
    rows=[]
    for (filename,line,function),(primitive,total,self_time,cumulative,_) in stats.stats.items():
        rows.append({'file':filename,'line':line,'function':function,'primitive_calls':primitive,
                     'total_calls':total,'self_seconds':self_time,'cumulative_seconds':cumulative})
    top=sorted(rows,key=lambda r:r['cumulative_seconds'],reverse=True)[:35]
    report={'status':'passed','scope':'914 training fields only, fixed original seed 0; no validation inference',
            'profiled_timing_caveat':'cProfile adds overhead; use separate unprofiled runs for speed claims',
            'recorded_at':now(),'preprocessing_seconds':prep_seconds,
            'unprofiled_model_construction_seconds':construction,'unprofiled_fit_seconds':baseline.fit_seconds,
            'profiled_constructor_seconds':profile_constructor,'profiled_fit_seconds':measured.fit_seconds,
            'profiled_fixed_input_state_preparation_seconds':measured.prepare_wall,
            'profiled_probability_passes_seconds':measured.probability_wall,
            'profiled_remaining_optimizer_gradient_python_seconds':measured.fit_seconds-measured.prepare_wall-measured.probability_wall,
            'probability_share_of_profiled_fit':measured.probability_wall/measured.fit_seconds,
            'calls':len(measured.history),'probability_passes':measured.forward_passes,
            'logical_field_evaluations':measured.logical_field_evaluations,
            'top_cumulative_functions':top,'baseline_weights_reproduce':True,
            'versions':{p:version(p) for p in ['qiskit','qiskit-machine-learning','numpy','scipy']}}
    (out/'profile.json').write_text(json.dumps(report,indent=2))
    assert all(sha(ROOT/p)==h for p,h in locked.items())
    print(json.dumps({k:v for k,v in report.items() if k!='top_cumulative_functions'},indent=2))
    print('Top cumulative functions:',[(r['function'],round(r['cumulative_seconds'],3)) for r in top[:12]])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);profile(p.parse_args().results)
