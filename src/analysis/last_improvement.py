"""Regenerate report Table 4 from saved 500-evaluation GP-BO ARD traces.
Run from repository root: python src/analysis/last_improvement.py
"""
from pathlib import Path
import pandas as pd

CASES = [
    ("sphere_full", "sphere", 5, 5),
    ("sphere_full", "sphere", 10, 10),
    ("sphere_effective", "sphere", 20, 2),
    ("ellipsoid_effective", "ellipsoid", 20, 2),
]

rows = []
for experiment, objective, D, d in CASES:
    last_evals = []
    for seed in range(5):
        name = f"{experiment}_D{D}_d{d}_bo_ard_b500_s{seed}.csv"
        path = Path("results/budget") / experiment / "raw" / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing trace: {path}")

        trace = pd.read_csv(path)
        if len(trace) != 500:
            raise ValueError(f"Expected 500 evaluations in {path}; got {len(trace)}")
        if "best_so_far" not in trace.columns:
            raise ValueError(f"Missing best_so_far in {path}")
        best = pd.to_numeric(trace["best_so_far"], errors="raise")
        if best.isna().any() or not (best.diff().iloc[1:] <= 0).all():
            raise ValueError(f"Invalid best_so_far sequence in {path}")

        # The first observation sets the initial best; later strict decreases improve it.
        improved = best.diff().lt(0)
        improvement_positions = [1] + (improved[improved].index + 1).tolist()
        last_evals.append(max(improvement_positions))

    rows.append({
        "Objective (D, d)": f"{objective.title()} ({D}, {d})",
        "Median last improvement": int(pd.Series(last_evals).median()),
        "Latest last improvement": max(last_evals),
        "Seed-level last improvements": str(last_evals),
    })

result = pd.DataFrame(rows)
print(result.to_string(index=False))
out = Path("results/consolidated/last_improvement_table.csv")
out.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out, index=False)
print(f"\nSaved: {out}")
