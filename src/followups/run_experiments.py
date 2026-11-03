"""Additive follow-up dispatcher: reuses existing code.src.runner.run_one unchanged.
Run from repository root: python code/followups/run_experiments.py --help
"""
import argparse
import json
import sys
from pathlib import Path
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT if (ROOT / 'src' / 'runner.py').exists() else ROOT / 'code'))
from src.config import Settings, EXPERIMENTS  # noqa: E402
from src.runner import run_one  # noqa: E402

PLAN = json.loads((ROOT / 'configs' / 'followup_plan.json').read_text())


def tasks(phase, experiment):
    spec = PLAN[phase]
    obj, pairs = EXPERIMENTS[experiment]
    budgets = spec.get('budgets', [spec.get('budget', 100)])
    kernels = spec.get('kernels', [spec.get('kernel', 'ard')])
    for budget in budgets:
        for D, d in pairs:
            for seed in spec['seeds']:
                for kernel in kernels:
                    for algorithm in spec['algorithms']:
                        yield obj, D, d, seed, budget, kernel, algorithm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['budget', 'robustness', 'extension'], required=True)
    p.add_argument('--experiment', choices=PLAN['experiments'], required=True)
    p.add_argument('--dry-run', action='store_true', help='Print planned jobs; no imports of CMA until execution')
    p.add_argument('--limit', type=int, default=None, help='Limit number of runs for smoke testing')
    p.add_argument('--overwrite', action='store_true', help='Explicitly replace existing run files')
    args = p.parse_args()
    out = ROOT / 'results' / args.phase / args.experiment / 'raw'
    selected = list(tasks(args.phase, args.experiment))
    if args.limit is not None:
        if args.limit < 1: p.error('--limit must be positive')
        selected = selected[:args.limit]
    print(f'Phase={args.phase}; experiment={args.experiment}; planned={len(selected)}; output={out}', flush=True)
    for idx, (obj, D, d, seed, budget, kernel, algorithm) in enumerate(selected, 1):
        cfg = replace(Settings(), budget=budget, kernel=kernel)
        stem = f'{args.experiment}_D{D}_d{d}_{algorithm}_{kernel if algorithm == "bo" else "standard"}_b{budget}_s{seed}'
        csv_path, json_path = out / (stem + '.csv'), out / (stem + '.json')
        if csv_path.exists() or json_path.exists():
            if not args.overwrite:
                if csv_path.exists() and json_path.exists():
                    print(f'[{idx}/{len(selected)}] SKIP existing {stem}', flush=True)
                    continue
                raise RuntimeError(f'Incomplete existing output for {stem}; inspect files before rerunning')
        if args.dry_run:
            print(f'[{idx}/{len(selected)}] PLAN {stem}', flush=True)
            continue
        print(f'[{idx}/{len(selected)}] RUN {stem}', flush=True)
        meta = run_one(args.experiment, obj, D, d, algorithm, seed, cfg, out)
        if meta['evaluations'] != budget:
            raise RuntimeError(f'Unexpected early termination: {stem}: {meta["evaluations"]}/{budget}')
    print('Finished selected jobs.', flush=True)


if __name__ == '__main__':
    main()
