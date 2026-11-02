import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',default='results/raw');ap.add_argument('--figures',default='results/figures');a=ap.parse_args()
    raw=Path(a.raw);fig=Path(a.figures);fig.mkdir(parents=True,exist_ok=True)
    metas=[]
    for path in sorted(raw.glob('*.json')):
        m=json.loads(path.read_text());m['stem']=path.stem;metas.append(m)
    if not metas:raise SystemExit('No results found')
    frame=pd.DataFrame(metas)
    frame.to_csv(fig/'runs.csv',index=False)
    keys=['experiment','objective','D','d','irrelevant','budget','algorithm','kernel']
    rows=[]
    for key,g in frame.groupby(keys,dropna=False):
        values=g.best_so_far.to_numpy(dtype=float)
        # percentile bootstrap CI on mean (not a significance test)
        rng=np.random.default_rng(42012)
        samples=rng.choice(values,size=(2000,len(values)),replace=True).mean(axis=1)
        rows.append(dict(zip(keys,key),n=len(values),mean=float(values.mean()),median=float(np.median(values)),
          std=float(values.std(ddof=1)) if len(values)>1 else float('nan'),
          mean_ci_low=float(np.quantile(samples,.025)),mean_ci_high=float(np.quantile(samples,.975)),
          mean_runtime_seconds=float(g.runtime_seconds.mean())))
    pd.DataFrame(rows).to_csv(fig/'summary.csv',index=False)
    for (exp,D,d,budget),g in frame.groupby(['experiment','D','d','budget']):
        plt.figure(figsize=(7,4))
        for (alg,kernel),sub in g.groupby(['algorithm','kernel']):
            traces=[]
            for stem in sub.stem:
                data=pd.read_csv(raw/(stem+'.csv'))
                traces.append(data.best_so_far.to_numpy())
            shortest=min(map(len,traces)); y=np.asarray([t[:shortest] for t in traces])
            plt.plot(np.arange(1,shortest+1),np.median(y,axis=0),label=f'{alg} {kernel}')
        plt.yscale('symlog',linthresh=1e-8);plt.xlabel('Objective evaluations');plt.ylabel('Median best-so-far')
        plt.title(f'{exp} D={D}, d={d}, budget={budget}');plt.legend();plt.tight_layout()
        plt.savefig(fig/f'{exp}_D{D}_d{d}_b{budget}.png',dpi=140);plt.close()
    print('Saved',len(rows),'summary groups and plots to',fig)
if __name__=='__main__':main()
