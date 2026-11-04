"""Consolidate raw GP-BO/CMA-ES experiments without rerunning optimisation.

Usage: python -m src.analysis.consolidate_results [--root .] [--allow-incomplete]
Requires: numpy, pandas, matplotlib.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GROUPS = {'sphere_full': ('sphere', [(2,2),(5,5),(10,10),(20,20)]),
          'sphere_effective': ('sphere', [(20,2),(20,5),(20,10),(20,20)]),
          'ellipsoid_full': ('ellipsoid', [(2,2),(5,5),(10,10),(20,20)]),
          'ellipsoid_effective': ('ellipsoid', [(20,2),(20,5),(20,10),(20,20)])}
KEY = ['objective','D','d','algorithm','seed','budget','kernel']


def phase_for(meta, path):
    p = {x.lower() for x in path.parts}
    if 'investigations' in p or 'consolidated' in p:
        return None
    if 'extension' in p: return 'extension'
    if 'robustness' in p: return 'robustness'
    if 'budget' in p: return 'budget'
    if 'pilot' in p or 'raw' in p: return 'pilot'
    return None


def scan(root):
    records, traces, problems = [], {}, []
    for meta_file in sorted((root/'results').rglob('*.json')):
        try: meta = json.loads(meta_file.read_text())
        except (ValueError, OSError): continue
        phase = phase_for(meta, meta_file)
        if phase is None or not all(k in meta for k in ('experiment','objective','D','d','seed','budget','algorithm')):
            continue
        exp = str(meta['experiment'])
        if exp not in GROUPS: continue
        obj, expected_pairs = GROUPS[exp]
        D, d, seed, budget = (int(meta[k]) for k in ('D','d','seed','budget'))
        alg = str(meta['algorithm']); kernel = str(meta.get('kernel','ard' if alg=='bo' else 'na'))
        if alg == 'cma': kernel = 'na'
        if obj != meta['objective'] or (D,d) not in expected_pairs or alg not in ('bo','cma'):
            problems.append(f'Unexpected configuration: {meta_file}'); continue
        csv_file = meta_file.with_suffix('.csv')
        if not csv_file.exists():
            problems.append(f'Missing trace: {csv_file}'); continue
        try:
            trace = pd.read_csv(csv_file)
            if not {'evaluation','x','value','best_so_far'}.issubset(trace.columns):
                raise ValueError('missing trace columns')
            if len(trace) != budget or list(trace.evaluation.astype(int)) != list(range(1,budget+1)):
                raise ValueError(f'evaluation count/index incorrect: {len(trace)} vs {budget}')
            x = np.array([json.loads(s) for s in trace.x], dtype=float)
            if x.shape != (budget,D) or not np.isfinite(x).all() or (x < -5-1e-9).any() or (x > 5+1e-9).any():
                raise ValueError('invalid or out-of-bounds evaluated vectors')
            w = np.ones(d) if obj=='sphere' else np.geomspace(1,100,d)
            expected_y = ((x[:,:d]-1)**2 @ w)
            y = trace.value.to_numpy(dtype=float)
            b = trace.best_so_far.to_numpy(dtype=float)
            if not np.isfinite(y).all() or not np.allclose(y,expected_y,atol=1e-7,rtol=1e-10):
                raise ValueError('objective mismatch')
            if not np.allclose(b,np.minimum.accumulate(y),atol=1e-7,rtol=1e-10):
                raise ValueError('best-so-far mismatch')
            if not np.isclose(b[-1],float(meta.get('best_so_far',b[-1])),atol=1e-7,rtol=1e-10):
                raise ValueError('metadata final mismatch')
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            problems.append(f'{csv_file}: {exc}'); continue
        record = dict(experiment=exp,objective=obj,D=D,d=d,irrelevant=D-d,algorithm=alg,
                      seed=seed,budget=budget,kernel=kernel,phase=phase,final=float(b[-1]),
                      runtime_seconds=float(meta.get('runtime_seconds',np.nan)),source=str(csv_file.relative_to(root)))
        records.append(record)
        traces[record['source']] = b
    return pd.DataFrame(records), traces, problems


def expected_keys():
    out = []
    for exp,(obj,pairs) in GROUPS.items():
        for D,d in pairs:
            for seed in range(10):
                for alg in ('bo','cma'):
                    if seed < 5:
                        out.append((exp,D,d,seed,100,alg,'ard' if alg=='bo' else 'na','pilot'))
                        for budget in (250,500): out.append((exp,D,d,seed,budget,alg,'ard' if alg=='bo' else 'na','budget'))
                        if alg=='bo': out.append((exp,D,d,seed,100,alg,'isotropic','robustness'))
                    else:
                        out.append((exp,D,d,seed,100,alg,'ard' if alg=='bo' else 'na','extension'))
    return out


def summary(df, keys):
    if df.empty: return pd.DataFrame()
    return df.groupby(keys,dropna=False).agg(n=('final','size'),mean=('final','mean'),median=('final','median'),
        std=('final','std'),q25=('final',lambda x:x.quantile(.25)),q75=('final',lambda x:x.quantile(.75)),
        median_runtime_seconds=('runtime_seconds','median')).reset_index()


def paired(a, b, label_a, label_b, by):
    aa = a[by+['final']].rename(columns={'final':label_a})
    bb = b[by+['final']].rename(columns={'final':label_b})
    merged = aa.merge(bb,on=by,how='inner',validate='one_to_one')
    if not merged.empty:
        merged['difference'] = merged[label_a]-merged[label_b]
        merged['winner'] = np.select([merged['difference']<0,merged['difference']>0],[label_a,label_b],default='tie')
    return merged


def plot_kernel(df, out):
    if df.empty:return
    ss=summary(df,['objective','D','d','kernel'])
    pv=ss.pivot(index=['objective','D','d'],columns='kernel',values='median').dropna(subset=['ard','isotropic'])
    if pv.empty:return
    vals=np.log10(np.maximum(pv['isotropic'].to_numpy(),1e-15)/np.maximum(pv['ard'].to_numpy(),1e-15))
    labels=[f'{o} D={D}, d={d}' for o,D,d in pv.index]
    fig,ax=plt.subplots(figsize=(10,max(5,len(labels)*.42)))
    ax.barh(labels,vals);ax.axvline(0,color='black',linewidth=.8)
    ax.set_xlabel('log10(isotropic median / ARD median); negative favours isotropic')
    ax.set_title('Kernel sensitivity — unique configurations')
    fig.tight_layout();fig.savefig(out/'kernel_sensitivity.png',dpi=180);plt.close(fig)


def plot_budget(df,out):
    if df.empty:return
    cases=[('sphere',10,10),('sphere',20,2),('ellipsoid',20,10)]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,(obj,D,d) in zip(axes,cases):
        sub=df[(df.objective==obj)&(df.D==D)&(df.d==d)]
        for alg in ('bo','cma'):
            g=sub[sub.algorithm==alg].groupby('budget')['final'].median().sort_index()
            if len(g):ax.plot(g.index,g.values,marker='o',label=alg.upper())
        ax.set_title(f'{obj} D={D},d={d}');ax.set_xlabel('Budget');ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].set_ylabel('Median final best (log scale)');axes[-1].legend()
    fig.tight_layout();fig.savefig(out/'budget_sensitivity.png',dpi=180);plt.close(fig)


def plot_convergence(df,traces,out):
    cases=[('sphere',2,2),('sphere',10,10),('sphere',20,20),('ellipsoid',20,2)]
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,(obj,D,d) in zip(axes.flat,cases):
        for alg in ('bo','cma'):
            subset=df[(df.objective==obj)&(df.D==D)&(df.d==d)&(df.budget==100)&(df.seed<5)&(df.algorithm==alg)&(df.kernel.isin(['ard','na']))]
            curves=[traces[p] for p in subset.source if p in traces]
            if curves:
                yy=np.array(curves)
                ax.plot(np.arange(1,101),np.median(yy,axis=0),label=alg.upper())
        ax.set_title(f'{obj} D={D},d={d}');ax.set_yscale('log');ax.set_xlabel('Evaluations');ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Median best-so-far');axes[1,0].set_ylabel('Median best-so-far');axes[0,0].legend()
    fig.tight_layout();fig.savefig(out/'pilot_convergence.png',dpi=180);plt.close(fig)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--allow-incomplete',action='store_true',help='Create partial outputs; mark missing coverage explicitly')
    args=ap.parse_args(argv)
    root=args.root.resolve(); out=root/'results'/'consolidated'; figures=out/'figures'
    df,traces,problems=scan(root)
    if df.empty:raise SystemExit('No valid raw traces found under results/. See instructions.')
    ids=['experiment','D','d','seed','budget','algorithm','kernel','phase']
    dup=df.duplicated(ids,keep=False)
    if dup.any():problems.append(f'Duplicate run identities: {df.loc[dup,ids].to_dict("records")[:8]}')
    found={tuple(x) for x in df[ids].itertuples(index=False,name=None)}
    missing=sorted(set(expected_keys())-found)
    unexpected=sorted(found-set(expected_keys()))
    if unexpected:problems.append(f'Unexpected runs: {unexpected[:8]}')
    out.mkdir(parents=True,exist_ok=True);figures.mkdir(exist_ok=True)
    pd.DataFrame(missing,columns=ids).to_csv(out/'missing_runs.csv',index=False)
    (out/'validation_report.json').write_text(json.dumps(dict(valid_runs=len(df),missing_expected=len(missing),
        missing_examples=missing[:12],problems=problems),indent=2,default=str))
    print(f'Validated raw runs: {len(df)}; missing expected: {len(missing)}; problems: {len(problems)}')
    if (missing or problems) and not args.allow_incomplete:
        raise SystemExit('Incomplete/invalid dataset. Review results/consolidated/validation_report.json; '
                         'use --allow-incomplete for clearly labelled exploratory outputs.')
    df.to_csv(out/'all_runs.csv',index=False)
    cross=df[(df.D==20)&(df.d==20)].copy()
    cross.to_csv(out/'reproducibility_crosschecks.csv',index=False)
    unique=df.loc[~((df.D==20)&(df.d==20)&df.experiment.str.endswith('_effective'))].copy()
    unique.to_csv(out/'unique_runs.csv',index=False)
    pilot=unique[(unique.budget==100)&(unique.seed<5)&(unique.phase=='pilot')]
    extension=unique[(unique.budget==100)&(unique.seed>=5)&(unique.phase=='extension')]
    budget=unique[((unique.phase=='budget')&(unique.budget.isin([250,500])))|((unique.phase=='pilot')&(unique.budget==100))]
    kernel=unique[(unique.budget==100)&(unique.seed<5)&(((unique.phase=='pilot')&(unique.algorithm=='bo'))|((unique.phase=='robustness')&(unique.kernel=='isotropic')))]
    ten=pd.concat([pilot,extension],ignore_index=True)
    for name,data,keys in [('pilot_summary',pilot,['objective','D','d','algorithm']),
                           ('budget_summary',budget,['objective','D','d','budget','algorithm']),
                           ('extension_ten_seed_summary',ten,['objective','D','d','algorithm']),
                           ('kernel_summary',kernel,['objective','D','d','kernel'])]:
        summary(data,keys).to_csv(out/f'{name}.csv',index=False)
    by=['objective','D','d','seed','budget']
    for name,data in [('pilot',pilot),('budget',budget),('ten_seed',ten)]:
        a=data[data.algorithm=='bo'];b=data[data.algorithm=='cma']
        paired(a,b,'bo','cma',by).to_csv(out/f'{name}_paired.csv',index=False)
    paired(kernel[kernel.kernel=='ard'],kernel[kernel.kernel=='isotropic'],
           'ard','isotropic',['objective','D','d','seed','budget']).to_csv(out/'kernel_paired.csv',index=False)
    plot_kernel(kernel,figures);plot_budget(budget,figures);plot_convergence(pilot,traces,figures)
    print('Saved consolidated tables and figures:',out)
    if missing or problems:print('WARNING: outputs are PARTIAL. See validation_report.json')

if __name__=='__main__':main()
