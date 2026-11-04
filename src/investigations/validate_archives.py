"""Audit all available per-run CSV+JSON pairs; do not modify saved results."""
import argparse
import json
from pathlib import Path
from .common import validate_trace, write_csv


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw-dirs',nargs='+',default=['results/raw','results/pilot','results/budget','results/robustness'])
    p.add_argument('--output-dir',default='results/investigations/validation')
    a=p.parse_args()
    results=[]
    for directory in a.raw_dirs:
        root=Path(directory)
        if not root.exists():
            continue
        for meta_path in sorted(root.rglob('*.json')):
            try:
                meta=json.loads(meta_path.read_text())
            except (OSError,ValueError):
                continue
            if not all(k in meta for k in ('objective','D','d','budget','algorithm')):
                continue
            trace_path=meta_path.with_suffix('.csv')
            if not trace_path.exists():
                results.append({'file':str(meta_path),'status':'missing_trace','details':str(trace_path)})
                continue
            try:
                checked=validate_trace(trace_path,meta['objective'],int(meta['D']),int(meta['d']),int(meta['budget']))
                issues=checked['errors']
                if int(meta.get('evaluations',-1))!=checked['count']:
                    issues.append('metadata evaluation count mismatch')
                if abs(float(meta.get('best_so_far',float('nan')))-float(min(checked['Y'])))>1e-7:
                    issues.append('metadata final best mismatch')
                results.append({'file':str(meta_path),'status':'fail' if issues else 'pass','details':'; '.join(issues)})
            except Exception as e:
                results.append({'file':str(meta_path),'status':'fail','details':str(e)})
    out=write_csv(Path(a.output_dir)/'raw_validation.csv',results,['file','status','details'])
    counts={status:sum(r['status']==status for r in results) for status in ('pass','fail','missing_trace')}
    print(f'{out}: {counts}')
    if not results:
        print('No per-run JSON metadata found. Aggregate summary CSVs cannot verify raw candidate bounds.')
    if counts['fail'] or counts['missing_trace']:
        raise SystemExit(2)

if __name__=='__main__':
    main()
