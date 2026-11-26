"""Command line (advanced users). Uses the same core as the local user interface.
  python -m vitaformer predict   --input <file|folder> --output <folder> [--weights ...] [--resize native|factor_bilinear|square_box --resize-value v]
  python -m vitaformer normalize --run <run folder> --plate-map <csv: image,group,role> [--features ...]
  python -m vitaformer app       (start the local user interface; same as the desktop launcher)"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__
from .io import EXTENSIONS

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / 'weights' / 'vitaformer_m2_seed42_final.pth'


def collect(inp: Path):
    if inp.is_dir():
        return sorted(p for p in inp.iterdir() if p.is_file() and not p.name.startswith('.'))
    return [inp]


def cmd_predict(a):
    from .pipeline import Run
    files = collect(Path(a.input))
    skipped = [p for p in files if p.suffix.lower() not in EXTENSIONS]
    if not files:
        sys.exit(f'no files found in {a.input}')
    val = None if a.resize == 'native' else float(a.resize_value)
    r = Run(files, a.output, a.weights, a.device, a.resize, val, a.min_region_px, a.name, not a.no_overlays)
    r.run()
    p = r.progress()
    print(f"{p['state']}: {p['ok']} ok, {p['failed']} failed/rejected (incl. {len(skipped)} with unsupported extensions) -> {p['run_dir']}")
    if r.state == 'error':
        sys.exit(r.message)


def cmd_normalize(a):
    from .readouts import NORMALISABLE, control_normalise, control_summary
    run = Path(a.run)
    r = pd.read_csv(run / 'readouts_image_measurements.csv'); m = pd.read_csv(a.plate_map)
    feats = a.features.split(',') if a.features else NORMALISABLE
    s = control_summary(r, m); out = run / 'normalised'
    out.mkdir(exist_ok=False)
    s.to_csv(out / 'control_groups_summary.csv', index=False, encoding='utf-8-sig')
    m.to_csv(out / 'plate_map_used.csv', index=False, encoding='utf-8-sig')
    control_normalise(r, m, feats).to_csv(out / 'readouts_relative_to_controls.csv', index=False, encoding='utf-8-sig')
    print(s.to_string(index=False)); print(f'-> {out}')


def cmd_app(a):
    from .app.server import main as app_main
    app_main(['--port', str(a.port)] + (['--no-browser'] if a.no_browser else []))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='vitaformer', description=f'VitaFormer M2 frozen bright-field organoid segmentation {__version__}')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('predict'); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--weights', default=str(DEFAULT_WEIGHTS)); p.add_argument('--device', default='auto')
    p.add_argument('--resize', default='native', choices=['native', 'factor_bilinear', 'square_box'])
    p.add_argument('--resize-value', default=None, help='factor (factor_bilinear) or side in px (square_box)')
    p.add_argument('--min-region-px', type=int, default=50); p.add_argument('--name', default='run'); p.add_argument('--no-overlays', action='store_true')
    p.set_defaults(fn=cmd_predict)
    q = sub.add_parser('normalize'); q.add_argument('--run', required=True); q.add_argument('--plate-map', required=True)
    q.add_argument('--features', default=None); q.set_defaults(fn=cmd_normalize)
    s = sub.add_parser('app'); s.add_argument('--port', type=int, default=8765); s.add_argument('--no-browser', action='store_true'); s.set_defaults(fn=cmd_app)
    a = ap.parse_args(argv); a.fn(a)
