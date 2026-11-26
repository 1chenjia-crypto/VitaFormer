#!/usr/bin/env python3
"""Re-run the toy example into a new folder and compare every output table with examples/toy/expected_output.
Usage (from the package root): python examples/check_toy.py <new_output_dir>"""
import json, subprocess, sys
from pathlib import Path
import pandas as pd
new = Path(sys.argv[1]); cfg = json.loads(Path('configs/toy_example.json').read_text()); cfg['output'] = str(new)
tmp = new.parent / (new.name + '_config.json'); tmp.write_text(json.dumps(cfg))
subprocess.run([sys.executable, '-m', 'oreval.run', '--config', str(tmp)], check=True)
ok = True
for f in sorted(Path('examples/toy/expected_output').glob('*.csv')):
    a, b = pd.read_csv(new / f.name), pd.read_csv(f)
    same = a.shape == b.shape and all((a[c].astype(str) == b[c].astype(str)).all() if a[c].dtype == object else
                                      ((a[c] - b[c]).abs().max() < 1e-9 or a[c].isna().equals(b[c].isna())) for c in b.columns)
    ok &= same; print(f.name, 'identical' if same else 'DIFFERENT')
print('PASS' if ok else 'FAIL'); sys.exit(0 if ok else 1)
