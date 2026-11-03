"""Validate follow-up raw CSV/JSON pairs, objective values and complete budgets."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT if (ROOT / 'src' / 'objectives.py').exists() else ROOT / 'code'))
from src.objectives import objective  # noqa: E402


def check_folder(folder):
    problems = []
    files = sorted(folder.rglob('*.json'))
    if not files:
        return ['No JSON results found'], 0
    for path in files:
        meta = json.loads(path.read_text())
        csv_path = path.with_suffix('.csv')
        if not csv_path.exists():
            problems.append(f'{path}: missing CSV')
            continue
        with csv_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        budget = int(meta['budget'])
        if len(rows) != budget or int(meta['evaluations']) != budget:
            problems.append(f'{path}: expected {budget}, got {len(rows)} rows, metadata={meta["evaluations"]}')
        running = float('inf')
        for i, r in enumerate(rows, 1):
            try:
                x = np.asarray(json.loads(r['x']), dtype=float)
                value = float(r['value'])
                running = min(running, value)
                assert int(r['evaluation']) == i
                assert np.isfinite(value)
                assert x.shape == (int(meta['D']),)
                assert np.all(x >= -5.0 - 1e-9) and np.all(x <= 5.0 + 1e-9)
                assert np.isclose(value, objective(x, meta['objective'], int(meta['D']), int(meta['d'])), rtol=1e-9, atol=1e-9)
                assert np.isclose(running, float(r['best_so_far']), rtol=1e-9, atol=1e-9)
            except (AssertionError, ValueError, KeyError, TypeError) as e:
                problems.append(f'{path}: invalid evaluation {i}: {e}')
                break
        if rows and not np.isclose(float(meta['best_so_far']), running):
            problems.append(f'{path}: final best mismatch')
    return problems, len(files)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['budget', 'robustness', 'extension'], required=True)
    p.add_argument('--experiment', choices=['sphere_full','ellipsoid_full','sphere_effective','ellipsoid_effective'])
    args = p.parse_args()
    folder = ROOT / 'results' / args.phase
    if args.experiment:
        folder = folder / args.experiment
    problems, count = check_folder(folder)
    print(f'Checked {count} runs in {folder}')
    if problems:
        print('\n'.join(problems[:30]))
        raise SystemExit(1)
    print('PASS: budget, rows, objective recomputation and best-so-far validated')


if __name__ == '__main__':
    main()
