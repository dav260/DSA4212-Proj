import argparse, csv, json, platform, sys
from pathlib import Path
import numpy as np
import scipy, sklearn
from dataclasses import asdict, replace
from .config import Settings, EXPERIMENTS
from .record import Recorder
from . import bo, cma_es

def run_one(experiment, objective, D, d, algorithm, seed, cfg, outdir):
    rec = Recorder(objective,D,d,cfg.budget)
    extra = (bo.optimize(rec,cfg,seed) if algorithm=='bo' else cma_es.optimize(rec,cfg,seed))
    meta = dict(experiment=experiment, objective=objective,D=D,d=d,irrelevant=D-d,
                algorithm=algorithm,seed=seed,budget=cfg.budget,kernel=cfg.kernel if algorithm=='bo' else 'na',
                config=asdict(cfg), **extra)
    if len(rec.rows)!=cfg.budget and extra['termination_reason'].startswith('budget_exhausted'):
        raise AssertionError('Budget mismatch')
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    stem=f'{experiment}_D{D}_d{d}_{algorithm}_{cfg.kernel if algorithm=="bo" else "standard"}_b{cfg.budget}_s{seed}'
    with (outdir/(stem+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['evaluation','x','value','best_so_far','elapsed_seconds']);w.writeheader()
        for r in rec.rows: w.writerow({**r,'x':json.dumps(r['x'])})
    meta.update(evaluations=len(rec.rows),best_so_far=rec.rows[-1]['best_so_far'],
                runtime_seconds=rec.rows[-1]['elapsed_seconds'])
    (outdir/(stem+'.json')).write_text(json.dumps(meta,indent=2))
    return meta

def main(argv=None):
    a=argparse.ArgumentParser()
    a.add_argument('--experiment',choices=[*EXPERIMENTS,'all'],default='all')
    a.add_argument('--phase',choices=['pilot','budget','robustness'],default='pilot')
    a.add_argument('--budget',type=int,default=None)
    a.add_argument('--seeds',type=int,nargs='+',default=None)
    a.add_argument('--kernel',choices=['ard','isotropic'],default='ard')
    a.add_argument('--outdir',default='results/raw')
    a.add_argument('--algorithms',nargs='+',choices=['bo','cma'],default=['bo','cma'])
    a.add_argument('--max-cases',type=int,default=None,help='Smoke-test limiter')
    args=a.parse_args(argv)
    budgets=([100,250,500] if args.phase=='budget' and args.budget is None else [args.budget or 100])
    kernels=(['ard','isotropic'] if args.phase=='robustness' and args.kernel=='ard' else [args.kernel])
    names=EXPERIMENTS if args.experiment=='all' else {args.experiment:EXPERIMENTS[args.experiment]}
    cases=0
    for budget in budgets:
      for kernel in kernels:
       cfg=Settings(budget=budget,kernel=kernel)
       for experiment,(obj,pairs) in names.items():
        for D,d in pairs:
         for seed in (args.seeds if args.seeds is not None else cfg.seeds):
          for alg in args.algorithms:
           if alg=='cma' and kernel!='ard':continue
           m=run_one(experiment,obj,D,d,alg,seed,cfg,args.outdir)
           print(experiment,D,d,alg,seed,budget,m['evaluations'],m['best_so_far'],flush=True)
           cases+=1
           if args.max_cases is not None and cases>=args.max_cases:return
if __name__=='__main__': main()
