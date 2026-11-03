"""Summarise follow-up runs without touching existing analyze.py.
Creates per-condition final-value summaries and per-seed BO-minus-CMA comparisons.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['budget', 'robustness', 'extension'], required=True)
    args = p.parse_args()
    base = ROOT / 'results' / args.phase
    metas = [json.loads(p.read_text()) for p in sorted(base.rglob('*.json'))]
    if not metas:
        raise SystemExit(f'No run metadata found in {base}')
    groups = defaultdict(list)
    paired = defaultdict(dict)
    for m in metas:
        key = (m['experiment'], m['objective'], m['D'], m['d'], m['budget'], m['algorithm'], m['kernel'])
        groups[key].append(float(m['best_so_far']))
        if m['kernel'] in ('ard', 'na'):
            pairkey = (m['experiment'], m['D'], m['d'], m['budget'], m['seed'])
            paired[pairkey][m['algorithm']] = float(m['best_so_far'])
    summary = []
    for key, vals in sorted(groups.items()):
        v = np.array(vals)
        summary.append(dict(zip(('experiment','objective','D','d','budget','algorithm','kernel'),key),
                            n=len(v),mean=float(v.mean()),median=float(np.median(v)),
                            std=float(v.std(ddof=1)) if len(v)>1 else '',
                            q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75))))
    write_csv(base / 'followup_summary.csv', summary,
              ['experiment','objective','D','d','budget','algorithm','kernel','n','mean','median','std','q25','q75'])
    differences=[]
    for key, vals in sorted(paired.items()):
        if 'bo' in vals and 'cma' in vals:
            differences.append(dict(zip(('experiment','D','d','budget','seed'),key),
                                    bo=vals['bo'],cma=vals['cma'],bo_minus_cma=vals['bo']-vals['cma']))
    write_csv(base / 'followup_paired.csv', differences,
              ['experiment','D','d','budget','seed','bo','cma','bo_minus_cma'])
    print(f'Wrote {len(summary)} summary groups and {len(differences)} paired rows in {base}')
    print('Positive BO-minus-CMA means CMA has lower objective (better).')


if __name__ == '__main__':
    main()
