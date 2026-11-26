"""Batch run shared by the command line and the local user interface (one computation core).

A run writes into a NEW folder (never overwrites):
  <out_root>/<YYYYMMDD-HHMMSS>_<name>/
     masks/<n>_<stem>.png          uint8 labels 0 background, 1 Alive, 2 Dead (not externally validated), 3 Bubble
     overlays/<n>_<stem>.jpg       preview (green Alive, red Dead, blue Bubble)
     readouts_image_measurements.csv   tier 1, one row per successfully processed image (written after every image)
     errors.csv                    one row per rejected/failed image, with the reason
     run_manifest.json             model, weights SHA256, settings, software versions, input SHA256s, counts, timing
The `<n>_` prefix keeps outputs unique when two inputs share a file name; `image` and `source_path` columns map outputs to inputs.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import __version__
from .io import InputError, load_image
from .readouts import image_readouts

COLORS = np.array([[0, 0, 0], [45, 180, 75], [220, 50, 47], [60, 130, 220]], np.uint8)
CLASS_NAMES = {0: 'background', 1: 'Alive (model class)', 2: 'Dead (model class; not externally validated)', 3: 'Bubble'}
_MODEL_CACHE: dict = {}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def overlay(rgb, mask, alpha=0.6):
    out = rgb.copy()
    for c in (1, 2, 3):
        m = mask == c
        out[m] = ((1 - alpha) * out[m] + alpha * COLORS[c]).astype(np.uint8)
    return out


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r'[\\/:*?"<>|]', '_', stem)[:120] or 'image'


def new_run_dir(out_root: Path, name: str) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    base = f"{dt.datetime.now():%Y%m%d-%H%M%S}_{re.sub(r'[^0-9A-Za-z_-]+', '_', name)[:40] or 'run'}"
    for k in range(1000):
        p = out_root / (base if k == 0 else f'{base}_{k}')
        try:
            p.mkdir()
            return p
        except FileExistsError:
            continue
    raise RuntimeError('could not create a new run folder')


def get_model(weights: Path, device_name: str, allow_unknown=False):
    """Returns (model, sha256, label, device, note). 'auto' tries the GPU and falls back to the CPU if the GPU cannot be used
    (for example when it is occupied); the reason is returned in `note` and written to the run manifest."""
    import torch
    from .infer import load_model
    note = ''
    if device_name == 'auto':
        if torch.cuda.is_available():
            try:
                key = (str(weights), 'cuda')
                if key not in _MODEL_CACHE:
                    m, d, lab = load_model(Path(weights), torch.device('cuda'), allow_unknown)
                    with torch.inference_mode():
                        m(torch.zeros(1, 3, 224, 224, device='cuda'))
                    _MODEL_CACHE[key] = (m, d, lab, torch.device('cuda'), '')
                return _MODEL_CACHE[key]
            except RuntimeError as e:
                _MODEL_CACHE.pop((str(weights), 'cuda'), None)
                note = f'GPU not usable ({str(e).splitlines()[0]}); running on CPU'
        device_name = 'cpu'
    dev = torch.device(device_name)
    key = (str(weights), str(dev))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = (*load_model(Path(weights), dev, allow_unknown), dev, note)
    m, d, lab, dv, _ = _MODEL_CACHE[key]
    return m, d, lab, dv, note


class Run:
    """Process a list of input files. Thread-safe progress fields; call cancel() to stop after the current image."""

    def __init__(self, files, out_root, weights, device='auto', resize_mode='native', resize_value=None, min_region_px=50,
                 name='run', save_overlays=True, source_names=None, run_dir=None):
        self.files = [Path(f) for f in files]
        self.source_names = source_names or [f.name for f in self.files]
        self.out_root = Path(out_root); self.weights = Path(weights); self.device = device
        self.mode, self.value, self.min_region_px, self.name, self.save_overlays = resize_mode, resize_value, min_region_px, name, save_overlays
        self.dir = Path(run_dir) if run_dir else None; self.state = 'pending'; self.done = 0; self.ok = 0; self.failed = 0; self.current = ''
        self.message = ''; self._cancel = threading.Event(); self.rows = []; self.errors = []; self.t0 = None

    def cancel(self):
        self._cancel.set()

    def progress(self):
        return {'state': self.state, 'total': len(self.files), 'done': self.done, 'ok': self.ok, 'failed': self.failed,
                'current': self.current, 'message': self.message, 'run_dir': str(self.dir) if self.dir else None,
                'device': str(self.dev) if hasattr(self, 'dev') else None,
                'elapsed_s': round((getattr(self, 't_end', None) or time.time()) - self.t0, 1) if self.t0 else 0}

    def _write_tables(self):
        cols = ['index', 'image', 'source_path', 'mask_file', 'overlay_file', 'preprocessing', 'resize_mode', 'resize_value']
        if self.rows:
            pd.DataFrame(self.rows).to_csv(self.dir / 'readouts_image_measurements.csv', index=False, encoding='utf-8-sig')
        else:
            pd.DataFrame(columns=cols).to_csv(self.dir / 'readouts_image_measurements.csv', index=False, encoding='utf-8-sig')
        pd.DataFrame(self.errors, columns=['index', 'image', 'source_path', 'reason']).to_csv(self.dir / 'errors.csv', index=False,
                                                                                              encoding='utf-8-sig')

    def _manifest(self, extra=None):
        import torch, scipy, PIL, timm
        m = {'tool': 'VitaFormer', 'tool_version': __version__, 'created': dt.datetime.now().isoformat(timespec='seconds'),
             'state': self.state, 'weights': str(self.weights), 'weights_sha256': getattr(self, 'digest', None),
             'weights_label': getattr(self, 'label', None), 'device': str(getattr(self, 'dev', self.device)), 'device_note': getattr(self, 'device_note', ''),
             'fp16_autocast_on_gpu': True, 'resize_mode': self.mode, 'resize_value': self.value, 'min_region_px': self.min_region_px,
             'inference': '224 px window, stride 112, reflect padding, ImageNet normalisation, softmax average, argmax; no threshold, '
                          'no test-time augmentation, no post-processing',
             'units': 'pixels (no physical calibration applied)', 'n_inputs': len(self.files), 'n_ok': self.ok, 'n_failed': self.failed,
             'elapsed_seconds': round((getattr(self, 't_end', None) or time.time()) - self.t0, 1) if self.t0 else None,
             'software': {'python': platform.python_version(), 'torch': torch.__version__, 'timm': timm.__version__, 'numpy': np.__version__,
                          'scipy': scipy.__version__, 'pandas': pd.__version__, 'pillow': PIL.__version__, 'platform': platform.platform()},
             'classes': CLASS_NAMES,
             'inputs': [{'index': i + 1, 'image': n, 'source_path': str(f), 'sha256': sha256_file(f) if f.exists() else None}
                        for i, (f, n) in enumerate(zip(self.files, self.source_names))]}
        m.update(extra or {})
        (self.dir / 'run_manifest.json').write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding='utf-8')

    def run(self):
        self.t0 = time.time(); self.state = 'loading model'
        try:
            if self.dir is None:
                self.dir = new_run_dir(self.out_root, self.name)
            (self.dir / 'masks').mkdir(exist_ok=False); (self.dir / 'overlays').mkdir(exist_ok=False)
            model, self.digest, self.label, self.dev, self.device_note = get_model(self.weights, self.device)
            self.message = self.device_note
        except Exception as e:
            self.state = 'error'; self.message = f'could not start: {e}'
            if self.dir:
                self._write_tables(); self._manifest({'start_error': str(e)})
            return self
        self.state = 'running'; self._manifest()
        for i, (f, name) in enumerate(zip(self.files, self.source_names), start=1):
            if self._cancel.is_set():
                break
            self.current = name
            try:
                rgb, info = load_image(f)
                from .infer import predict_mask
                mask = predict_mask(model, rgb, self.dev, self.mode, self.value)
                stem = f'{i:04d}_{safe_stem(name)}'
                Image.fromarray(mask).save(self.dir / 'masks' / f'{stem}.png')
                if self.save_overlays:
                    Image.fromarray(overlay(rgb, mask)).save(self.dir / 'overlays' / f'{stem}.jpg', quality=90)
                self.rows.append({'index': i, 'image': name, 'source_path': str(f), 'mask_file': f'masks/{stem}.png',
                                  'overlay_file': f'overlays/{stem}.jpg' if self.save_overlays else '',
                                  'preprocessing': info['preprocessing'], 'resize_mode': self.mode, 'resize_value': self.value,
                                  **image_readouts(mask, rgb, self.min_region_px)})
                self.ok += 1
            except InputError as e:
                self.errors.append({'index': i, 'image': name, 'source_path': str(f), 'reason': f'rejected: {e}'}); self.failed += 1
            except Exception as e:  # keep going; record the failure
                self.errors.append({'index': i, 'image': name, 'source_path': str(f),
                                    'reason': f'failed: {e.__class__.__name__}: {e}'}); self.failed += 1
                (self.dir / 'error_traceback.log').open('a', encoding='utf-8').write(f'--- {name}\n{traceback.format_exc()}\n')
            self.done += 1
            self._write_tables()
        self.state = 'cancelled' if self._cancel.is_set() else 'finished'
        self.t_end = time.time(); self.current = ''; self._write_tables(); self._manifest()
        return self
