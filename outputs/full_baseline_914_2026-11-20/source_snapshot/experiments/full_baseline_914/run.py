"""Record and run the original coverage trainer without changing any model.

From the repository root:
  .venv/bin/python experiments/full_baseline_914/run.py --results outputs/NEW_DIRECTORY
Result directories must be new. Test features are never used for model fitting/prediction.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from challenge.coverage import train
from analyze import analyze


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def now():
    return datetime.now(ZoneInfo('Africa/Casablanca')).isoformat()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, message):
        for stream in self.streams:
            stream.write(message)
            stream.flush()
        return len(message)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    destination = args.results if args.results.is_absolute() else ROOT / args.results
    destination.mkdir(parents=True, exist_ok=False)
    protected = [p for folder in ('challenge', 'configs', 'data', 'results', 'notebooks')
                 for p in (ROOT / folder).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    snapshot_files = [*sorted((ROOT / 'challenge').glob('*.py')),
                      ROOT / 'configs/reference.json', ROOT / 'configs/pilot.json',
                      ROOT / 'requirements.txt', ROOT / 'data/manifest.json',
                      ROOT / 'experiments/full_baseline_914/run.py',
                      ROOT / 'experiments/full_baseline_914/analyze.py']
    for source in snapshot_files:
        target = destination / 'source_snapshot' / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    metadata = {
        'status': 'running', 'started_at': now(),
        'branch': git('branch', '--show-current'), 'commit': git('rev-parse', 'HEAD'),
        'git_status_before': git('status', '--short'),
        'command': '.venv/bin/python experiments/full_baseline_914/run.py --results ' + str(args.results),
        'trainer': 'challenge.coverage.train(914, results); original implementation, unmodified',
        'configuration': {'training_fields': 914, 'validation_fields': 150, 'seeds': [0, 1, 2],
                          'labels': ['alfalfa', 'maize', 'wheat'],
                          'preprocessing': 'StandardScaler -> PCA(4, full SVD) -> StandardScaler; fitted on training only',
                          'qnn': {'qubits': 4, 'trainable_angles': 8, 'encoding': 'ry',
                                  'optimizer': 'COBYLA', 'max_evaluations': 100, 'shots': None},
                          'control': {'architecture': '4 -> tanh(1) -> softmax(3)',
                                      'trainable_parameters': 11, 'optimizer': 'COBYLA', 'max_evaluations': 100},
                          'svm': json.loads((ROOT / 'configs/reference.json').read_text()),
                          'test_inference': False},
        'environment': {'python': sys.version, 'executable': sys.executable,
                        'platform': platform.platform(), 'machine': platform.machine(),
                        'processor': platform.processor()},
        'package_versions': dict(sorted((d.metadata['Name'], d.version) for d in distributions() if d.metadata['Name'])),
        'protected_file_sha256_before': before,
    }
    meta_path = destination / 'run_metadata.json'
    meta_path.write_text(json.dumps(metadata, indent=2))
    start = perf_counter()
    try:
        with (destination / 'training.log').open('w', buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)):
                print('Running original 914-field baseline, seeds 0/1/2. Destination:', destination, flush=True)
                train(914, destination)
                print('Training complete. Verifying predictions and building diagnostic figures.', flush=True)
                analyze(destination)
        after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
        changed = [path for path in before if before[path] != after[path]]
        if changed:
            raise RuntimeError('Unexpected changes to protected project files: ' + ', '.join(changed))
        metadata.update(status='complete', completed_at=now(), wall_seconds=perf_counter()-start,
                        protected_files_unchanged=True)
    except BaseException:
        metadata.update(status='failed', completed_at=now(), wall_seconds=perf_counter()-start,
                        error=traceback.format_exc())
        raise
    finally:
        meta_path.write_text(json.dumps(metadata, indent=2))
    print('Complete. Source, notebook, data, quick-run artifacts and supplied results preserved.', flush=True)


if __name__ == '__main__':
    main()
