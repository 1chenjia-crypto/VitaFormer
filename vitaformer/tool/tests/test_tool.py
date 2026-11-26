#!/usr/bin/env python3
"""Automated checks of the shared core (CPU). Usage: python tests/test_tool.py <scratch_dir> [weights]
Covers: input validation (formats, multi-page TIFF, alpha, palette, 16-bit, corrupt, tiny), batch continues after failures,
cancel keeps finished results, duplicate file names, never overwriting, Chinese/space paths, control normalisation flags
(no control, zero control median, not in plate map), square-mode rejection of non-square images, manifest content."""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vitaformer.io import InputError, check_image  # noqa: E402
from vitaformer.pipeline import Run  # noqa: E402
from vitaformer.readouts import control_normalise  # noqa: E402

root = Path(sys.argv[1]) / '测试 目录 with space'; root.mkdir(parents=True, exist_ok=True)
W = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / 'weights/vitaformer_m2_seed42_final.pth'
inp = root / '输入 images'; inp.mkdir(exist_ok=True)
res = []


def ok(name, cond, detail=''):
    res.append((name, bool(cond), detail)); print(('PASS ' if cond else 'FAIL ') + name, detail)


rng = np.random.default_rng(0)
real = Path(__file__).resolve().parents[1] / 'examples_nvir/input/B_ms12_echo_revolve_v0_P0002_D04_2022-09-18_H03.png'
small = np.asarray(Image.open(real).resize((320, 320), Image.BOX))
Image.fromarray(small).save(inp / '类器官 样本1.png')                                   # 8-bit L, Chinese + space name
Image.fromarray(np.stack([small] * 3, -1)).save(inp / 'rgb sample.jpg', quality=95)     # 8-bit RGB
Image.fromarray((small.astype(np.uint16) * 200)).save(inp / 'sixteen_bit.tif')          # 16-bit greyscale
Image.fromarray(np.full((320, 320), 180, np.uint8)).save(inp / 'blank.png')             # no foreground expected
Image.fromarray(small[:, :300]).save(inp / 'non_square.png')
Image.fromarray(np.stack([small] * 3 + [np.full_like(small, 255)], -1), 'RGBA').save(inp / 'alpha.png')
Image.fromarray(small).convert('P').save(inp / 'palette.png')
Image.fromarray(small[:40, :40]).save(inp / 'tiny.png')
(inp / 'corrupt.png').write_bytes(b'not an image')
(inp / 'notes.txt').write_text('x')
fr = [Image.fromarray(small), Image.fromarray(small)]; fr[0].save(inp / 'multipage.tif', save_all=True, append_images=fr[1:])
sub = inp / 'sub'; sub.mkdir(exist_ok=True); Image.fromarray(small).save(sub / '类器官 样本1.png')  # duplicate name, different folder

expect = {'类器官 样本1.png': None, 'rgb sample.jpg': None, 'sixteen_bit.tif': None, 'blank.png': None, 'non_square.png': None,
          'alpha.png': 'alpha', 'palette.png': 'palette', 'tiny.png': 'too small', 'corrupt.png': 'cannot be read', 'notes.txt': 'unsupported',
          'multipage.tif': 'pages'}
for n, e in expect.items():
    try:
        check_image(inp / n); ok(f'validate {n}', e is None, 'accepted')
    except InputError as x:
        ok(f'validate {n}', e is not None and e in str(x), str(x))

files = [inp / n for n in expect] + [sub / '类器官 样本1.png']
r = Run(files, root / '结果 results', W, 'cpu', 'native', None, 50, 'edge cases').run()
t = pd.read_csv(r.dir / 'readouts_image_measurements.csv'); e = pd.read_csv(r.dir / 'errors.csv')
ok('batch continues after failures', r.state == 'finished' and len(t) == 6 and len(e) == 6, f'{len(t)} ok (5 accepted + 1 duplicate name), {len(e)} errors')
ok('duplicate names get distinct mask files', t[t.image == '类器官 样本1.png'].mask_file.nunique() == 2)
ok('16-bit image stretched and recorded', 'stretch' in t[t.image == 'sixteen_bit.tif'].preprocessing.iloc[0])
ok('blank image gives zero foreground', int(t[t.image == 'blank.png'].foreground_area_px.iloc[0]) == 0)
man = json.loads((r.dir / 'run_manifest.json').read_text(encoding='utf-8'))
ok('manifest has weights SHA256 + versions + input checksums',
   man['weights_sha256'] and 'torch' in man['software'] and all(i['sha256'] for i in man['inputs'] if Path(i['source_path']).exists()))
ok('original files unchanged', (inp / '类器官 样本1.png').stat().st_size > 0 and all(f.exists() for f in files))
r2 = Run(files[:1], root / '结果 results', W, 'cpu', 'native', None, 50, 'edge cases').run()
ok('second run with same name gets a new folder', r2.dir != r.dir and r.dir.exists() and r2.dir.exists(), r2.dir.name)
r3 = Run([inp / 'non_square.png', inp / '类器官 样本1.png'], root / '结果 results', W, 'cpu', 'square_box', 256, 50, 'square').run()
e3 = pd.read_csv(r3.dir / 'errors.csv')
ok('square mode rejects non-square image explicitly', len(e3) == 1 and 'square' in e3.reason.iloc[0], e3.reason.iloc[0] if len(e3) else '')
# cancel
many = [inp / '类器官 样本1.png'] * 30
rc = Run(many, root / '结果 results', W, 'cpu', 'native', None, 50, 'cancel')
th = threading.Thread(target=rc.run); th.start()
while rc.done < 2 and th.is_alive():
    time.sleep(.2)
rc.cancel(); th.join()
tc = pd.read_csv(rc.dir / 'readouts_image_measurements.csv')
ok('cancel stops early and keeps finished images', rc.state == 'cancelled' and 2 <= len(tc) < 30, f'{len(tc)} kept')
# control normalisation flags
pm = pd.DataFrame([{'image': 'rgb sample.jpg', 'group': 'g1', 'role': 'control'}, {'image': 'sixteen_bit.tif', 'group': 'g1', 'role': 'sample'},
                   {'image': 'blank.png', 'group': 'g2', 'role': 'control'}, {'image': 'non_square.png', 'group': 'g2', 'role': 'sample'},
                   {'image': '类器官 样本1.png', 'group': 'g3', 'role': 'sample'}])
t1 = t.drop_duplicates('image')
n = control_normalise(t1, pm, ['foreground_area_px']).set_index('image')
st = n['foreground_area_px_rel_status']
ok('normal group ok', st['sixteen_bit.tif'] == 'ok' and abs(n.loc['rgb sample.jpg', 'foreground_area_px_rel_control'] - 1) < 1e-12)
ok('zero control median flagged, no value', st['non_square.png'] == 'control median is zero' and np.isnan(n.loc['non_square.png', 'foreground_area_px_rel_control']))
ok('group without control flagged', st['类器官 样本1.png'] == 'no control in group')
n2 = control_normalise(t1, pm[pm.group != 'g3'], ['foreground_area_px']).set_index('image')
ok('image missing from plate map flagged', n2.loc['类器官 样本1.png', 'foreground_area_px_rel_status'] == 'not in plate map' and np.isnan(n2.loc['类器官 样本1.png', 'foreground_area_px_rel_control']))
try:
    control_normalise(t1, pm.assign(role=['control', 'Sample?', 'control', 'sample', 'sample']), ['foreground_area_px']); ok('invalid role rejected', False)
except ValueError:
    ok('invalid role rejected', True)
pd.DataFrame(res, columns=['check', 'pass', 'detail']).to_csv(root / 'test_results.csv', index=False, encoding='utf-8-sig')
print(f"\n{sum(p for _, p, _ in res)} / {len(res)} passed"); sys.exit(0 if all(p for _, p, _ in res) else 1)
