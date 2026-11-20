"""Show individual-seed confusions so averaging cannot hide a failed fit."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    parser.add_argument('--seed',type=int,default=1)
    args=parser.parse_args()
    records=json.loads((args.results/'runs.json').read_text())
    labels=['alfalfa','maize','wheat']
    plt.rcParams.update({'font.size':11,'figure.dpi':140})
    fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
    for ax,scale in zip(axes,(1.,.5,.25)):
        r=next(r for r in records if r['scale']==scale and r['seed']==args.seed)
        cm=np.asarray(r['metrics']['confusion'])
        ax.imshow(cm/50,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f'{cm[i,j]} / 50\n{cm[i,j]/50:.0%}',ha='center',va='center',
                        color='white' if cm[i,j]/50>.55 else '#18202a')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=labels,yticklabels=labels,
               xlabel='Predicted crop',ylabel='Actual crop',title=f'Scale {scale:g}: F1 {r["metrics"]["macro_f1"]:.4f}')
    fig.suptitle(f'Same seed {args.seed}, same 150 validation fields: individual-run trade-offs',fontsize=13)
    target=args.results/'figures'/f'07_seed_{args.seed}_confusion_comparison.png'
    fig.savefig(target,bbox_inches='tight')
    plt.close(fig)
    print(target)


if __name__=='__main__':
    main()
