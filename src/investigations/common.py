"""Shared input/output helpers. Run modules from repository root with python -m."""
import csv
import json
from pathlib import Path
import numpy as np
from src.objectives import objective


def read_trace(path):
    with Path(path).open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f'Empty trace: {path}')
    return rows


def validate_trace(path, name, D, d, expected=None):
    rows = read_trace(path)
    values = []
    X = []
    errors = []
    for i, row in enumerate(rows, 1):
        x = np.asarray(json.loads(row['x']), dtype=float)
        y = float(row['value'])
        best = float(row['best_so_far'])
        if int(row['evaluation']) != i:
            errors.append(f'evaluation index at {i}')
        if x.shape != (D,) or not np.all(np.isfinite(x)) or np.any(x < -5 - 1e-10) or np.any(x > 5 + 1e-10):
            errors.append(f'candidate bounds/shape at {i}')
        else:
            if not np.isclose(y, objective(x, name, D, d), rtol=1e-9, atol=1e-9):
                errors.append(f'objective mismatch at {i}')
        values.append(y)
        X.append(x)
        if not np.isclose(best, min(values), rtol=1e-9, atol=1e-9):
            errors.append(f'best-so-far mismatch at {i}')
    if expected is not None and len(rows) != expected:
        errors.append(f'count {len(rows)} != {expected}')
    return {'file': str(path), 'count': len(rows), 'errors': errors, 'X': np.asarray(X), 'Y': np.asarray(values)}


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return path
