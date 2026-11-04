"""Seed-matched ARD versus isotropic comparisons from original raw JSON metadata.
No additional optimiser evaluations are performed.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .common import write_csv, validate_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--raw-dirs', nargs='+', default=['results/raw','results/pilot','results/robustness'])
    p.add_argument('--output-dir', default='results/investigations/kernel_comparison')
    p.add_argument('--seeds', nargs='+', type=int, default=[0,1,2,3,4])
    a = p.parse_args()
    candidates = {}
    for directory in a.raw_dirs:
        for path in Path(directory).rglob('*.json') if Path(directory).exists() else []:
            try:
                data = json.loads(path.read_text())
            except (ValueError, OSError):
                continue
            if data.get('algorithm') != 'bo' or data.get('kernel') not in ('ard','isotropic') or data.get('budget') != 100:
                continue
            key = (data.get('experiment'),data.get('objective'),int(data['D']),int(data['d']),int(data['seed']),data['kernel'])
            if key in candidates and candidates[key][0] != data.get('best_so_far'):
                raise ValueError(f'Conflicting duplicate metadata for {key}; inspect {path}')
            candidates[key] = (float(data['best_so_far']),path,data)
    rows = []
    for (exp,obj,D,d,seed,kernel), (best,path,meta) in sorted(candidates.items()):
        if kernel != 'ard' or seed not in a.seeds:
            continue
        other = candidates.get((exp,obj,D,d,seed,'isotropic'))
        if other is None:
            continue
        iso, ipath, imeta = other
        ard_trace, iso_trace = path.with_suffix('.csv'), ipath.with_suffix('.csv')
        issues = []
        for trace in (ard_trace, iso_trace):
            if trace.exists():
                issues += validate_trace(trace,obj,D,d,100)['errors']
            else:
                issues.append(f'missing raw trace {trace}')
        rows.append({'experiment':exp,'objective':obj,'D':D,'d':d,'seed':seed,'ard_final':best,'isotropic_final':iso,'isotropic_minus_ard':iso-best,'winner':'ard' if best<iso else ('isotropic' if iso<best else 'tie'),'trace_validation':'ok' if not issues else '; '.join(issues)})
    out = write_csv(Path(a.output_dir)/'paired_kernel_comparison.csv',rows,['experiment','objective','D','d','seed','ard_final','isotropic_final','isotropic_minus_ard','winner','trace_validation'])
    print(f'{len(rows)} matched seed pairs saved: {out}')
    if not rows:
        print('No matches. This script requires raw per-run JSON metadata for both kernels; aggregate followup_summary.csv alone is insufficient for paired comparison.')
    elif any(r['trace_validation'] != 'ok' for r in rows):
        print('WARNING: some raw trace files are absent or invalid; see trace_validation column')

if __name__ == '__main__':
    main()
