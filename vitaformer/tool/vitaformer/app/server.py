"""Local user interface: a small HTTP server on 127.0.0.1 (standard library only) serving a single page.
Images never leave the computer. All computation goes through vitaformer.pipeline / vitaformer.readouts (same code as the CLI).

Start:  python -m vitaformer app   (or the desktop launcher). If an instance is already running, the browser is pointed to it."""
from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .. import __version__
from ..io import EXTENSIONS, InputError, check_image, load_image
from ..pipeline import COLORS, Run, new_run_dir
from ..readouts import NORMALISABLE, control_normalise, control_summary

STATIC = Path(__file__).parent / 'static'
APP_ID = 'VitaFormer-local-ui'
STATE = {'out_root': Path.home() / 'VitaFormer_results', 'weights': None, 'sessions': {}, 'runs': {}, 'lock': threading.Lock(),
         'weights_check': None}
MAX_UPLOAD = 512 * 1024 * 1024


def default_weights():
    from ..cli import DEFAULT_WEIGHTS
    return Path(os.environ.get('VITAFORMER_WEIGHTS', DEFAULT_WEIGHTS))


def weights_status():
    if STATE['weights_check'] is None:
        w = STATE['weights']
        if not w.exists():
            STATE['weights_check'] = {'ok': False, 'message': f'model file not found: {w}'}
        else:
            from ..infer import KNOWN_WEIGHTS
            from ..pipeline import sha256_file
            d = sha256_file(w)
            STATE['weights_check'] = {'ok': d in KNOWN_WEIGHTS, 'sha256': d, 'label': KNOWN_WEIGHTS.get(d, 'unknown file'),
                                      'size_mb': round(w.stat().st_size / 1e6, 1), 'path': str(w),
                                      'message': 'verified' if d in KNOWN_WEIGHTS else 'SHA256 does not match the frozen VitaFormer M2 weights'}
    return STATE['weights_check']


def device_info():
    try:
        import torch
        cuda = torch.cuda.is_available()
        return {'cuda': cuda, 'name': torch.cuda.get_device_name(0) if cuda else 'CPU', 'torch': torch.__version__}
    except Exception as e:
        return {'cuda': False, 'name': f'unavailable ({e})', 'torch': None}


def color_mask_png(mask):
    rgba = np.zeros(mask.shape + (4,), np.uint8)
    for c in (1, 2, 3):
        rgba[mask == c, :3] = COLORS[c]; rgba[mask == c, 3] = 255
    b = io.BytesIO(); Image.fromarray(rgba, 'RGBA').save(b, 'PNG'); return b.getvalue()


def png_bytes(arr):
    b = io.BytesIO(); Image.fromarray(arr).save(b, 'PNG'); return b.getvalue()


class Handler(BaseHTTPRequestHandler):
    server_version = 'VitaFormerLocal/' + __version__

    def log_message(self, fmt, *args):  # quiet console
        pass

    # ---------------------------------------------------------------- helpers
    def _send(self, code, body: bytes, ctype='application/json; charset=utf-8', extra=None):
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode('utf-8'))

    def _err(self, msg, code=400):
        self._json({'error': msg}, code)

    def _q(self):
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).items()}

    def _body_json(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n).decode('utf-8')) if n else {}

    def _run(self, q):
        r = STATE['runs'].get(q.get('run', ''))
        if r is None:
            raise KeyError('unknown run')
        return r

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            if path in ('/', '/index.html'):
                return self._send(200, (STATIC / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if path.startswith('/static/'):
                f = (STATIC / path[len('/static/'):]).resolve()
                if STATIC.resolve() not in f.parents or not f.is_file():
                    return self._err('not found', 404)
                return self._send(200, f.read_bytes(), (mimetypes.guess_type(f.name)[0] or 'application/octet-stream') + '; charset=utf-8')
            if path == '/api/ping':
                return self._json({'app': APP_ID, 'version': __version__})
            if path == '/api/config':
                return self._json({'version': __version__, 'out_root': str(STATE['out_root']), 'weights': weights_status(),
                                   'device': device_info(), 'extensions': EXTENSIONS,
                                   'runs': [{'id': k, 'dir': str(r.dir), 'state': r.state} for k, r in STATE['runs'].items()]})
            q = self._q()
            if path == '/api/progress':
                return self._json(self._run(q).progress())
            if path == '/api/results':
                r = self._run(q)
                return self._json({'rows': json.loads(pd.DataFrame(r.rows).to_json(orient='records')) if r.rows else [],
                                   'errors': r.errors, 'run_dir': str(r.dir), 'state': r.state})
            if path == '/api/image':
                r = self._run(q); idx = int(q['index']); kind = q.get('kind', 'original')
                row = next(x for x in r.rows if x['index'] == idx)
                if kind == 'original':
                    rgb, _ = load_image(Path(row['source_path']))
                    return self._send(200, png_bytes(rgb), 'image/png')
                mask = np.asarray(Image.open(r.dir / row['mask_file']))
                if kind == 'mask':
                    return self._send(200, color_mask_png(mask), 'image/png')
                return self._err('unknown kind')
            if path == '/api/last_plate_map':
                r = self._run(q); ds = sorted(r.dir.glob('normalised_*/plate_map_used.csv'), key=lambda f: int(f.parent.name.split('_')[1]))
                rows = json.loads(pd.read_csv(ds[-1]).fillna('').to_json(orient='records')) if ds else []
                return self._json({'rows': rows})
            if path == '/api/plate_template':
                r = self._run(q)
                df = pd.DataFrame({'image': [x['image'] for x in r.rows], 'group': '', 'role': 'sample'})
                return self._send(200, df.to_csv(index=False).encode('utf-8-sig'), 'text/csv; charset=utf-8',
                                  {'Content-Disposition': 'attachment; filename="plate_map_template.csv"'})
            if path == '/api/zip':
                r = self._run(q); b = io.BytesIO()
                with zipfile.ZipFile(b, 'w', zipfile.ZIP_DEFLATED) as z:
                    for f in sorted(r.dir.rglob('*')):
                        if f.is_file() and 'input' not in f.relative_to(r.dir).parts[:1]:
                            z.write(f, f.relative_to(r.dir.parent))
                name = urllib.parse.quote(r.dir.name + '.zip')
                return self._send(200, b.getvalue(), 'application/zip', {'Content-Disposition': f"attachment; filename*=UTF-8''{name}"})
            return self._err('not found', 404)
        except KeyError as e:
            return self._err(str(e), 404)
        except Exception as e:
            return self._err(f'{e.__class__.__name__}: {e}', 500)

    # ---------------------------------------------------------------- POST / PUT
    def do_PUT(self):
        q = self._q()
        if urllib.parse.urlparse(self.path).path != '/api/upload':
            return self._err('not found', 404)
        s = STATE['sessions'].get(q.get('session', ''))
        if s is None:
            return self._err('unknown session', 404)
        n = int(self.headers.get('Content-Length', 0))
        if n > MAX_UPLOAD:
            return self._err('file too large')
        name = Path(urllib.parse.unquote(q.get('name', 'image'))).name
        data = self.rfile.read(n)
        with STATE['lock']:
            k = len(s['files']) + 1
        dest = s['dir'] / 'input' / f'{k:04d}_{name}'
        dest.write_bytes(data)
        try:
            info = check_image(dest); ok = True; reason = info['preprocessing'] + f" ({info['width']} x {info['height']})"
        except InputError as e:
            ok, reason, info = False, str(e), None
        with STATE['lock']:
            s['files'].append({'name': name, 'path': dest, 'ok': ok, 'reason': reason, 'info': info})
        return self._json({'name': name, 'ok': ok, 'reason': reason, 'square': bool(info and info['width'] == info['height'])})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        try:
            b = self._body_json()
            if path == '/api/settings':
                if b.get('out_root'):
                    p = Path(b['out_root']).expanduser(); p.mkdir(parents=True, exist_ok=True)
                    t = p / '.vitaformer_write_test'; t.write_text('ok', encoding='utf-8'); t.unlink()
                    STATE['out_root'] = p
                return self._json({'out_root': str(STATE['out_root'])})
            if path == '/api/session':
                d = new_run_dir(STATE['out_root'], b.get('name') or 'run'); (d / 'input').mkdir()
                sid = d.name; STATE['sessions'][sid] = {'dir': d, 'files': []}
                return self._json({'session': sid, 'run_dir': str(d)})
            if path == '/api/run':
                s = STATE['sessions'].get(b.get('session', ''))
                if s is None:
                    return self._err('unknown session', 404)
                if not weights_status()['ok']:
                    return self._err('model weights not verified: ' + weights_status()['message'])
                files = [f for f in s['files']]
                mode = b.get('resize_mode', 'native'); val = None if mode == 'native' else float(b.get('resize_value'))
                r = Run([f['path'] for f in files], STATE['out_root'], STATE['weights'], b.get('device', 'auto'), mode, val,
                        int(b.get('min_region_px', 50)), s['dir'].name, True, [f['name'] for f in files], run_dir=s['dir'])
                STATE['runs'][s['dir'].name] = r
                threading.Thread(target=r.run, daemon=True).start()
                return self._json({'run': s['dir'].name})
            if path == '/api/cancel':
                self._run(b).cancel(); return self._json({'ok': True})
            if path in ('/api/controls/preview', '/api/controls/apply'):
                r = self._run(b)
                pm = pd.DataFrame(b['plate_map'])
                pm['group'] = pm['group'].astype(str).str.strip(); pm['role'] = pm['role'].astype(str).str.strip().str.lower()
                pm = pm[pm.group != '']
                if pm.empty:
                    return self._err('no image has a group; assign groups first')
                rd = pd.DataFrame(r.rows)
                summ = control_summary(rd, pm)
                if path.endswith('preview'):
                    return self._json({'summary': json.loads(summ.to_json(orient='records')),
                                       'unassigned': int(len(rd) - rd.image.isin(pm.image).sum())})
                k = 1
                while (r.dir / f'normalised_{k}').exists():
                    k += 1
                out = r.dir / f'normalised_{k}'; out.mkdir()
                pm.to_csv(out / 'plate_map_used.csv', index=False, encoding='utf-8-sig')
                summ.to_csv(out / 'control_groups_summary.csv', index=False, encoding='utf-8-sig')
                nd = control_normalise(rd, pm, NORMALISABLE)
                nd.to_csv(out / 'readouts_relative_to_controls.csv', index=False, encoding='utf-8-sig')
                st = nd['foreground_area_px_rel_status'].value_counts().to_dict()
                (out / 'README.txt').write_text('Control-relative readouts (tier 2): each readout divided by the median of the control images in '
                                                'the same group. Rows marked with a status other than "ok" have no value on purpose.\n'
                                                f'Status counts (foreground area): {st}\n', encoding='utf-8')
                return self._json({'folder': str(out), 'status_counts': st})
            if path == '/api/open_folder':
                r = self._run(b); d = str(r.dir)
                if sys.platform.startswith('win'):
                    os.startfile(d)  # noqa
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', d])
                else:
                    subprocess.Popen(['xdg-open', d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return self._json({'ok': True, 'dir': d})
            return self._err('not found', 404)
        except KeyError as e:
            return self._err(f'missing or unknown: {e}', 404)
        except Exception as e:
            return self._err(f'{e.__class__.__name__}: {e}', 500)


def already_running(port):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/api/ping', timeout=1.5) as r:
            return json.loads(r.read()).get('app') == APP_ID
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument('--port', type=int, default=8765); ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--out-root', default=None); ap.add_argument('--weights', default=None)
    a = ap.parse_args(argv)
    STATE['weights'] = Path(a.weights) if a.weights else default_weights()
    if a.out_root:
        STATE['out_root'] = Path(a.out_root)
    for port in range(a.port, a.port + 10):
        if already_running(port):
            print(f'VitaFormer is already running at http://127.0.0.1:{port} - opening it.')
            if not a.no_browser:
                webbrowser.open(f'http://127.0.0.1:{port}')
            return
        try:
            srv = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            break
        except OSError:
            continue
    else:
        sys.exit('no free port between %d and %d' % (a.port, a.port + 9))
    url = f'http://127.0.0.1:{port}'
    print(f'VitaFormer {__version__} running at {url} (local only; close this window to stop).', flush=True)
    print(f'Results folder: {STATE["out_root"]}', flush=True)
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
