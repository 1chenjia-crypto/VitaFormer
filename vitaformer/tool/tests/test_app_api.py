#!/usr/bin/env python3
"""End-to-end check of the local user interface through its HTTP API (the same calls the page makes).
Usage: python tests/test_app_api.py <base_url> <results_root (Chinese/space allowed)> <cli_run_dir for comparison>
Uploads the NViR example (6 images) + 1 invalid file, runs square_box 256, checks masks and readouts against the CLI run
(same core, same device -> identical expected), previews and applies control normalisation with the example plate map,
downloads the ZIP, and checks that a second launch detects the running instance."""
import io
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

base, out_root, cli_run = sys.argv[1], sys.argv[2], Path(sys.argv[3])
here = Path(__file__).resolve().parents[1]; ex = here / 'examples_nvir'
res = []


def ok(n, c, d=''):
    res.append((n, bool(c), d)); print(('PASS ' if c else 'FAIL ') + n, d)


def call(method, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=600) as r:
        b = r.read(); return json.loads(b) if 'json' in r.headers.get('Content-Type', '') else b


call('POST', '/api/settings', {'out_root': out_root})
s = call('POST', '/api/session', {'name': 'nvir example'})
ok('session folder created under Chinese/space path', Path(s['run_dir']).exists() and out_root in s['run_dir'], s['run_dir'])
names = sorted(p.name for p in (ex / 'input').iterdir())
for n in names:
    r = call('PUT', f"/api/upload?session={urllib.parse.quote(s['session'])}&name={urllib.parse.quote(n)}", raw=(ex / 'input' / n).read_bytes())
    ok(f'upload {n[-11:]}', r['ok'], r['reason'])
r = call('PUT', f"/api/upload?session={urllib.parse.quote(s['session'])}&name={urllib.parse.quote('坏 文件.png')}", raw=b'garbage')
ok('invalid upload reported, not converted', not r['ok'], r['reason'])
run = call('POST', '/api/run', {'session': s['session'], 'resize_mode': 'square_box', 'resize_value': 256})['run']
t0 = time.time()
while True:
    p = call('GET', f'/api/progress?run={urllib.parse.quote(run)}')
    if p['state'] in ('finished', 'cancelled', 'error'):
        break
    time.sleep(2)
ok('run finished with 6 ok and 1 failed', p['state'] == 'finished' and p['ok'] == 6 and p['failed'] == 1, json.dumps(p, ensure_ascii=False))
rr = call('GET', f'/api/results?run={urllib.parse.quote(run)}'); rd = Path(rr['run_dir'])
ui = pd.DataFrame(rr['rows']).set_index('image'); cli = pd.read_csv(cli_run / 'readouts_image_measurements.csv').set_index('image')
same_mask = all((np.asarray(Image.open(rd / ui.loc[n, 'mask_file'])) == np.asarray(Image.open(cli_run / cli.loc[n, 'mask_file']))).all() for n in names)
ok('UI masks identical to CLI masks (same core, CPU)', same_mask)
num = [c for c in cli.columns if cli[c].dtype.kind in 'fi' and c not in ('index',)]
uic = pd.read_csv(rd / 'readouts_image_measurements.csv').set_index('image')
ok('exported UI readouts identical to CLI readouts', float((uic.loc[names, num] - cli.loc[names, num]).abs().max().max()) == 0.0,
   '(compared on the exported CSV; the on-screen JSON is rounded to 10 significant digits)')
img = call('GET', f'/api/image?run={urllib.parse.quote(run)}&index=1&kind=mask')
ok('mask preview served as RGBA PNG', Image.open(io.BytesIO(img)).mode == 'RGBA')
pm = pd.read_csv(ex / 'plate_map.csv')[['image', 'group', 'role']].to_dict('records')
pv = call('POST', '/api/controls/preview', {'run': run, 'plate_map': pm})
ok('control preview shows group, 3 controls, status ok', pv['summary'][0]['n_controls'] == 3 and pv['summary'][0]['status'] == 'ok', json.dumps(pv['summary'][0])[:160])
bad = [dict(x, role='sample') for x in pm]
pv2 = call('POST', '/api/controls/preview', {'run': run, 'plate_map': bad})
ok('group without controls is flagged before normalising', pv2['summary'][0]['status'] == 'NO CONTROLS in this group')
ap = call('POST', '/api/controls/apply', {'run': run, 'plate_map': pm})
nd = pd.read_csv(Path(ap['folder']) / 'readouts_relative_to_controls.csv')
cli_n = pd.read_csv(cli_run / 'normalised/readouts_relative_to_controls.csv').set_index('image')
ok('normalised values identical to CLI', np.allclose(nd.set_index('image').loc[names, 'foreground_area_px_rel_control'],
                                                       cli_n.loc[names, 'foreground_area_px_rel_control']))
ap2 = call('POST', '/api/controls/apply', {'run': run, 'plate_map': pm})
ok('second normalisation goes to a new folder (no overwrite)', ap2['folder'] != ap['folder'], Path(ap2['folder']).name)
z = zipfile.ZipFile(io.BytesIO(call('GET', f'/api/zip?run={urllib.parse.quote(run)}')))
zn = z.namelist()
ok('ZIP contains masks, overlays, readouts, errors, manifest, normalised', all(any(k in n for n in zn) for k in
   ['masks/', 'overlays/', 'readouts_image_measurements.csv', 'errors.csv', 'run_manifest.json', 'normalised_1/']), f'{len(zn)} files')
man = json.loads((rd / 'run_manifest.json').read_text(encoding='utf-8'))
ok('manifest records square_box 256 and weights SHA256', man['resize_mode'] == 'square_box' and man['resize_value'] == 256 and len(man['weights_sha256']) == 64)
port = urllib.parse.urlparse(base).port
o = subprocess.run([sys.executable, '-m', 'vitaformer', 'app', '--port', str(port), '--no-browser'], cwd=here, capture_output=True, text=True, timeout=60)
ok('second launch detects running instance and exits', 'already running' in o.stdout, o.stdout.strip())
print(f'\nrun id: {run}\n{sum(p for _, p, _ in res)} / {len(res)} passed')
(Path(out_root) / 'app_api_test_results.json').write_text(json.dumps({'run': run, 'checks': res}, ensure_ascii=False, indent=1), encoding='utf-8')
sys.exit(0 if all(p for _, p, _ in res) else 1)
