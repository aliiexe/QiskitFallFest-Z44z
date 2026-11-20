"""Verify saved baseline predictions and draw graphs without retraining models."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from challenge.data import load_data
from challenge.metrics import score

LABELS = ['alfalfa', 'maize', 'wheat']
MODELS = ['reference', 'qnn', 'matched_head']
NAMES = {'reference': 'SVM', 'qnn': 'QNN', 'matched_head': 'Tiny classical control'}
COLORS = {'reference': '#64748b', 'qnn': '#176f9e', 'matched_head': '#ba852d'}


def analyze(folder):
    folder = Path(folder)
    figures = folder / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 140, 'axes.spines.top': False,
                         'axes.spines.right': False})
    tables, _, manifest, _ = load_data()
    validation = tables['validation']
    records = json.loads((folder / 'runs.json').read_text())
    assert len(records) == 15
    train_digest = hashlib.sha256(tables['train'].to_csv(index=False).encode()).hexdigest()
    prediction_tables = {}
    for name in ['dummy_majority', 'dummy_stratified'] + MODELS:
        selected = [r for r in records if r['model'] == name]
        assert sorted(r['seed'] for r in selected) == [0, 1, 2]
        for r in selected:
            assert r['training_rows'] == 914 and r['training_sha256'] == train_digest
            assert r['validation_sha256'] == manifest['files']['validation.csv']
            p = pd.read_csv(folder / f'{name}_{r["seed"]}_predictions.csv')
            assert p.sample_id.is_unique and len(p) == 150
            assert set(p.sample_id) == set(validation.sample_id)
            p = p.set_index('sample_id').loc[validation.sample_id].reset_index()
            np.testing.assert_array_equal(p.target, validation.target)
            assert score(p.target, p.prediction, LABELS) == r['metrics']
            prediction_tables[(name, r['seed'])] = p
            if name in ('qnn', 'matched_head'):
                history = json.loads((folder / f'{name}_{r["seed"]}_history.json').read_text())
                assert len(history) == r['resources']['objective_evaluations'] == 100
    for seed in (1, 2):
        np.testing.assert_array_equal(prediction_tables[('reference', 0)].prediction,
                                      prediction_tables[('reference', seed)].prediction)

    rows = []
    for r in records:
        item = {'model': r['model'], 'seed': r['seed'], 'macro_f1': r['metrics']['macro_f1'],
                'accuracy': r['metrics']['accuracy'],
                'correct_fields': int(np.trace(np.asarray(r['metrics']['confusion']))),
                'fit_seconds': r['fit_seconds'], 'predict_seconds': r['predict_seconds'],
                'preprocessing_seconds': r['preprocessing_seconds'],
                'total_seconds': sum(r[k] for k in ('fit_seconds', 'predict_seconds', 'preprocessing_seconds'))}
        for j, label in enumerate(LABELS):
            for key in ('precision', 'recall'):
                item[f'{key}_{label}'] = r['metrics'][key][j]
        rows.append(item)
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'per_seed_metrics.csv', index=False)
    numeric = [c for c in frame if c not in ('model', 'seed')]
    summary = {'class_order': LABELS, 'standard_deviation': 'population SD across seeds, ddof=0; not a confidence interval',
               'metrics': [], 'verification': {'all_15_prediction_files_rescored': True,
                                               'all_training_and_validation_hashes_match': True,
                                               'all_qnn_control_histories_have_100_evaluations': True,
                                               'svm_predictions_identical_across_seeds': True,
                                               'test_inference': False}}
    for name in ['dummy_majority', 'dummy_stratified'] + MODELS:
        selected = frame.loc[frame.model.eq(name)]
        item = {'model': name}
        for metric in numeric:
            item[f'{metric}_mean'] = float(selected[metric].mean())
            item[f'{metric}_sd'] = float(selected[metric].std(ddof=0))
        rs = [r for r in records if r['model'] == name]
        item['confusion_sum_three_seeds'] = np.asarray([r['metrics']['confusion'] for r in rs]).sum(axis=0).tolist()
        item['resources'] = rs[0]['resources']
        summary['metrics'].append(item)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for x, name in enumerate(MODELS):
        chosen = frame.loc[frame.model.eq(name)].sort_values('seed')
        f1, seconds = chosen.macro_f1.to_numpy(), chosen.total_seconds.to_numpy()
        axes[0].bar(x, f1.mean(), width=.65, yerr=f1.std(ddof=0), capsize=5, color=COLORS[name], alpha=.35)
        for j, marker in enumerate(['o', 's', '^']):
            axes[0].scatter(x + (j-1)*.13, f1[j], marker=marker, s=50, color=COLORS[name],
                            label=f'seed {j}' if x == 0 else None, zorder=4)
        axes[0].text(x, max(f1)+.045, f'{f1.mean():.4f}', ha='center')
        axes[1].bar(x, seconds.mean(), width=.65, color=COLORS[name], alpha=.8)
        axes[1].text(x, seconds.mean()*1.3, f'{seconds.mean():.3g} s', ha='center')
    axes[0].set(ylabel='Validation macro-F1', ylim=(0, 1.05), title='Dots: seeds; bars: mean +/- seed SD')
    axes[0].legend(loc='lower left', ncols=3)
    axes[1].set(ylabel='Preprocessing + fit + prediction (seconds; log scale)', yscale='log',
                title='Runtime on this computer (mean across seeds)')
    bottom, top = axes[1].get_ylim()
    axes[1].set_ylim(bottom/2, top*5)
    for ax in axes:
        ax.set_xticks(range(3), ['SVM', 'QNN', 'Tiny control'])
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Fresh full baseline: 914 training / 150 validation fields', fontsize=13)
    fig.savefig(figures / '01_scores_and_runtime.png', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    qnn_records = sorted([r for r in records if r['model']=='qnn'], key=lambda r:r['seed'])
    for seed, ax, r in zip([0, 1, 2], axes, qnn_records):
        cm = np.asarray(r['metrics']['confusion'])
        ax.imshow(cm/50, vmin=0, vmax=1, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm[i,j]} / 50\n{cm[i,j]/50:.0%}', ha='center', va='center',
                        color='white' if cm[i,j]/50 > .55 else '#18202a')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
               xlabel='Predicted crop', ylabel='Actual crop',
               title=f'QNN seed {seed}: F1 {r["metrics"]["macro_f1"]:.4f}')
    fig.suptitle('QNN: does a different starting point change which crops are missed?', fontsize=13)
    fig.savefig(figures / '02_qnn_confusion_by_seed.png', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout='constrained')
    for ax, name in zip(axes, MODELS):
        cm = np.asarray([r['metrics']['confusion'] for r in records if r['model']==name]).sum(axis=0)
        ax.imshow(cm/150, vmin=0, vmax=1, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{cm[i,j]} / 150\n{cm[i,j]/150:.0%}', ha='center', va='center',
                        color='white' if cm[i,j]/150 > .55 else '#18202a')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
               xlabel='Predicted crop', ylabel='Actual crop', title=NAMES[name])
    fig.suptitle('Confusions summed across 3 runs of the SAME 150 fields (450 decisions)', fontsize=13)
    fig.savefig(figures / '03_all_models_confusion.png', bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
    seed_colors = ['#176f9e', '#cf8b25', '#35825b']
    loss_stats = {}
    for name, ax in zip(['qnn', 'matched_head'], axes):
        loss_stats[name] = []
        for seed, color in zip([0, 1, 2], seed_colors):
            h = pd.DataFrame(json.loads((folder / f'{name}_{seed}_history.json').read_text()))
            loss = h.objective.to_numpy() * (np.log(2) if name == 'qnn' else 1)
            ax.plot(h.evaluation, np.minimum.accumulate(loss), label=f'Seed {seed}', color=color)
            loss_stats[name].append({'seed': seed, 'initial_loss_nats': float(loss[0]),
                                     'best_at_50': float(loss[:50].min()), 'best_at_80': float(loss[:80].min()),
                                     'best_at_100': float(loss.min()),
                                     'best_evaluation': int(h.evaluation.iloc[loss.argmin()])})
        ax.set(xlabel='Optimizer evaluations (100 per run)', ylabel='Best training loss so far (nats; lower is better)',
               title=NAMES[name])
        ax.legend()
        ax.grid(alpha=.15)
    fig.suptitle('Original optimizer budget: are the models still learning at the limit?', fontsize=13)
    fig.savefig(figures / '04_training_loss.png', bbox_inches='tight')
    plt.close(fig)
    summary['loss_diagnostics'] = loss_stats

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    for ax, metric in zip(axes, ['recall', 'precision']):
        for offset, name in zip([-.24, 0, .24], MODELS):
            selected = frame.loc[frame.model.eq(name)]
            means = np.array([selected[f'{metric}_{label}'].mean() for label in LABELS])*100
            sd = np.array([selected[f'{metric}_{label}'].std(ddof=0) for label in LABELS])*100
            ax.bar(np.arange(3)+offset, means, width=.22, yerr=sd, capsize=3,
                   label=NAMES[name], color=COLORS[name])
        ax.set(xticks=range(3), xticklabels=LABELS, ylim=(0, 110), yticks=[0,25,50,75,100],
               ylabel='Percent; mean +/- seed SD', title=f'Per-crop {metric}')
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3)
    fig.savefig(figures / '05_per_crop_metrics.png', bbox_inches='tight')
    plt.close(fig)

    # A field remains one field even though it receives predictions from three seeds.
    field_review = validation[['sample_id', 'target']].copy()
    for name in MODELS:
        for seed in [0,1,2]:
            field_review[f'{name}_seed_{seed}'] = prediction_tables[(name,seed)].prediction.to_numpy()
        field_review[f'{name}_wrong_seed_count'] = sum(
            field_review[f'{name}_seed_{seed}'].ne(field_review.target).astype(int) for seed in [0,1,2])
    field_review.to_csv(folder / 'validation_field_review.csv', index=False)
    summary['persistent_qnn_errors'] = field_review.loc[field_review.qnn_wrong_seed_count.eq(3)].to_dict(orient='records')
    summary['qnn_error_stability'] = {str(k): int((field_review.qnn_wrong_seed_count==k).sum()) for k in range(4)}

    supplied = json.loads((ROOT / 'results/runs.json').read_text())
    supplied_deltas = []
    for r in records:
        if r['model'] in MODELS:
            old = next(s for s in supplied if s['model']==r['model'] and s['seed']==r['seed'])
            supplied_deltas.append({'model': r['model'], 'seed': r['seed'],
                                    'macro_f1_difference': r['metrics']['macro_f1']-old['metrics']['macro_f1'],
                                    'accuracy_difference': r['metrics']['accuracy']-old['metrics']['accuracy']})
    summary['comparison_with_supplied_reference_runs'] = supplied_deltas

    quick_path = ROOT / 'outputs/mentor_review/quick_run_review.json'
    if quick_path.exists():
        quick = json.loads(quick_path.read_text())
        summary['quick_vs_full_seed0'] = {}
        fig, ax = plt.subplots(figsize=(8,3.8), layout='constrained')
        for x, (name, key) in enumerate([('reference','svm'),('qnn','qnn'),('matched_head','control')]):
            old = quick['validation'][key]
            new = next(r['metrics'] for r in records if r['model']==name and r['seed']==0)
            summary['quick_vs_full_seed0'][name] = {'quick': old, 'full': new}
            for offset, metric, label, color in [(-.18,old,'100 training fields','#aab6c2'),
                                                (.18,new,'914 training fields','#176f9e')]:
                bar=ax.bar(x+offset,metric['macro_f1'],width=.34,label=label if x==0 else None,color=color)
                ax.bar_label(bar,fmt='%.3f',padding=3)
        ax.set(xticks=range(3),xticklabels=['SVM','QNN','Tiny control'],ylim=(0,1.07),
               ylabel='Validation macro-F1',title='Same seed 0: quick run versus full training')
        ax.legend(loc='lower left')
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
        fig.savefig(figures/'06_quick_vs_full_seed0.png',bbox_inches='tight')
        plt.close(fig)

    (folder / 'summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(summary['metrics']).drop(columns=['resources','confusion_sum_three_seeds']).to_csv(folder/'summary.csv',index=False)
    print(frame.loc[frame.model.isin(MODELS), ['model','seed','macro_f1','accuracy','correct_fields','total_seconds']].to_string(index=False),flush=True)
    print('Verified 15 prediction files, all data hashes, original optimizer budgets, and deterministic SVM predictions.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    analyze(parser.parse_args().results)
