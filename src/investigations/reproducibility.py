"""Compare equivalent D=d=20 conditions, rerun using identical configurations."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.config import Settings
from src.runner import run_one
from .common import validate_trace, write_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--objective', choices=['sphere', 'ellipsoid'], required=True)
    p.add_argument('--algorithm', choices=['bo', 'cma'], default='bo')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    p.add_argument('--budget', type=int, default=100)
    p.add_argument('--raw-dir', default='results/raw')
    p.add_argument('--output-dir', default='results/investigations/reproducibility')
    p.add_argument('--rerun', action='store_true', help='Run controlled identical-config cross-checks; can take time')
    a = p.parse_args()
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in a.seeds:
        names = [f'{a.objective}_full', f'{a.objective}_effective']
        kernel = 'ard' if a.algorithm == 'bo' else 'standard'
        paths = [Path(a.raw_dir) / f'{name}_D20_d20_{a.algorithm}_{kernel}_b{a.budget}_s{seed}.csv' for name in names]
        if a.rerun:
            cfg = Settings(budget=a.budget, kernel='ard')
            for name in names:
                run_one(name, a.objective, 20, 20, a.algorithm, seed, cfg, output / 'controlled_raw')
            paths = [output / 'controlled_raw' / path.name for path in paths]
        if not all(path.exists() for path in paths):
            records.append({'seed': seed, 'status': 'missing_trace', 'details': '; '.join(str(x) for x in paths if not x.exists())})
            continue
        left, right = [validate_trace(path, a.objective, 20, 20, a.budget) for path in paths]
        differences = np.where((np.max(np.abs(left['X'] - right['X']), axis=1) > 1e-10) | (np.abs(left['Y'] - right['Y']) > 1e-9))[0]
        first = int(differences[0] + 1) if len(differences) else ''
        records.append({'seed': seed, 'status': 'invalid_trace' if left['errors'] or right['errors'] else ('diverged' if len(differences) else 'identical'), 'first_difference_evaluation': first, 'initial_eight_identical': bool(np.allclose(left['X'][:8], right['X'][:8], atol=1e-10, rtol=0)), 'final_full': float(np.min(left['Y'])), 'final_effective': float(np.min(right['Y'])), 'details': '; '.join(left['errors'] + right['errors'])})
    path = write_csv(output / f'{a.objective}_{a.algorithm}_crosscheck.csv', records, ['seed','status','first_difference_evaluation','initial_eight_identical','final_full','final_effective','details'])
    print(path)
    for r in records:
        print(json.dumps(r))
    if any(r['status'] in ('invalid_trace','missing_trace') for r in records):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
