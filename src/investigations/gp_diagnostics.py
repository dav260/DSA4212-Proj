"""Non-invasive GP diagnostic rerun: wraps sklearn GP.fit to record kernel length scales.
This does not change the acquisition policy or the original BO implementation.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from src.config import Settings
from src.record import Recorder
from src.bo import optimize
from .common import write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--objective', choices=['sphere','ellipsoid'], required=True)
    p.add_argument('--D', type=int, required=True)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--budget', type=int, default=100)
    p.add_argument('--kernel', choices=['ard','isotropic'], default='ard')
    p.add_argument('--output-dir', default='results/investigations/gp_diagnostics')
    a = p.parse_args()
    if not 1 <= a.d <= a.D:
        p.error('Require 1 <= d <= D')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Settings(budget=a.budget, kernel=a.kernel)
    rec = Recorder(a.objective, a.D, a.d, a.budget)
    diagnostics = []
    original_fit = GaussianProcessRegressor.fit

    def traced_fit(self, X, y):
        result = original_fit(self, X, y)
        n = len(y)
        # Standard shared implementation fits at n=8,18,28,... and conditions at every n.
        is_hyper = n == cfg.initial_points or (n-cfg.initial_points) % cfg.hyper_interval == 0
        kern = self.kernel_
        # Existing shared implementation uses ConstantKernel * Matern.
        matern = kern.k2 if hasattr(kern, 'k2') else kern
        scales = np.atleast_1d(matern.length_scale).astype(float).tolist()
        diagnostics.append({'observations': n, 'hyperparameters_optimized': is_hyper, 'length_scales': json.dumps(scales), 'min_length_scale': min(scales), 'max_length_scale': max(scales), 'log_marginal_likelihood': float(self.log_marginal_likelihood_value_)})
        return result

    try:
        GaussianProcessRegressor.fit = traced_fit
        extra = optimize(rec, cfg, a.seed)
    finally:
        GaussianProcessRegressor.fit = original_fit
    stem = f'{a.objective}_D{a.D}_d{a.d}_{a.kernel}_b{a.budget}_s{a.seed}'
    write_csv(out / f'{stem}_gp.csv', diagnostics, ['observations','hyperparameters_optimized','length_scales','min_length_scale','max_length_scale','log_marginal_likelihood'])
    trace = []
    for row in rec.rows:
        trace.append({'evaluation':row['evaluation'], 'x':json.dumps(row['x']), 'value':row['value'], 'best_so_far':row['best_so_far'], 'elapsed_seconds':row['elapsed_seconds'], 'improvement': 0.0 if row['evaluation']==1 else max(0.0, rec.rows[row['evaluation']-2]['best_so_far'] - row['best_so_far'])})
    write_csv(out / f'{stem}_trace.csv', trace, ['evaluation','x','value','best_so_far','elapsed_seconds','improvement'])
    jumps = sorted((r for r in trace if r['improvement']>0), key=lambda r:r['improvement'], reverse=True)[:10]
    (out / f'{stem}_findings.json').write_text(json.dumps({'objective':a.objective,'D':a.D,'d':a.d,'seed':a.seed,'budget':a.budget,'kernel':a.kernel,'final_best':rec.rows[-1]['best_so_far'],'hyperparameter_refits':extra['hyperparameter_refits_at_n'],'top_improvements':[{k:r[k] for k in ('evaluation','improvement','best_so_far')} for r in jumps], 'note':'Length scales are descriptive, not proof of recovered variable relevance. GP fits are instrumented via temporary wrapper.'}, indent=2))
    print(f'Saved {stem}; {len(rec.rows)} evaluations; final={rec.rows[-1]["best_so_far"]:.8g}')

if __name__ == '__main__':
    main()
