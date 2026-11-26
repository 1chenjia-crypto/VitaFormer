"""Input validation and loading. Only input types that the validated analyses actually used are accepted; everything else is
rejected with a message (no silent conversion).

Accepted (and where they were validated):
  * 8-bit greyscale (PIL mode 'L')            - used as is, replicated to 3 channels      (NViR crops, PNG)
  * 8-bit RGB (mode 'RGB')                    - used as is                                (CRC training/test images, JPEG)
  * 16-bit greyscale (modes 'I;16', 'I;16B', 'I') - per-image 1st-99th percentile stretch to 8 bit, replicated
                                                                                          (DeepOrganoid IXM TIFF; GemcitabineScreen)
Rejected: multi-page/multi-frame files, alpha channels (RGBA/LA), palette/1-bit/CMYK/float images, 16-bit colour, unreadable files,
and extensions other than .png .jpg .jpeg .tif .tiff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
ACCEPTED_MODES = {'L': '8-bit greyscale, used as is', 'RGB': '8-bit RGB, used as is',
                  'I;16': '16-bit greyscale, 1-99th percentile stretch', 'I;16B': '16-bit greyscale, 1-99th percentile stretch',
                  'I;16L': '16-bit greyscale, 1-99th percentile stretch', 'I': '16/32-bit integer greyscale, 1-99th percentile stretch'}
MIN_SIDE = 64


class InputError(ValueError):
    pass


def check_image(path: Path) -> dict:
    """Return image info or raise InputError with a user-readable reason."""
    path = Path(path)
    if path.suffix.lower() not in EXTENSIONS:
        raise InputError(f'unsupported file type "{path.suffix}" (accepted: {", ".join(EXTENSIONS)})')
    try:
        im = Image.open(path)
    except (UnidentifiedImageError, OSError) as e:
        raise InputError(f'cannot be read as an image ({e.__class__.__name__})')
    n = getattr(im, 'n_frames', 1)
    if n > 1:
        raise InputError(f'contains {n} pages/frames; export single images (one field of view per file)')
    if im.mode not in ACCEPTED_MODES:
        hint = {'RGBA': 'has an alpha channel', 'LA': 'has an alpha channel', 'P': 'is a palette image', '1': 'is 1-bit',
                'CMYK': 'is CMYK', 'F': 'is floating point', 'I;16S': 'is signed 16-bit'}.get(im.mode, f'has mode {im.mode}')
        raise InputError(f'{hint}; accepted: 8-bit greyscale, 8-bit RGB or 16-bit greyscale')
    w, h = im.size
    if min(w, h) < MIN_SIDE:
        raise InputError(f'too small ({w} x {h}); minimum side {MIN_SIDE} px')
    return {'mode': im.mode, 'width': w, 'height': h, 'preprocessing': ACCEPTED_MODES[im.mode]}


def load_image(path: Path):
    """Validated load -> (uint8 RGB array, info). Raises InputError."""
    info = check_image(path)
    im = Image.open(path)
    if im.mode == 'L':
        a = np.asarray(im)
        rgb = np.repeat(a[:, :, None], 3, axis=2)
    elif im.mode == 'RGB':
        rgb = np.asarray(im)
    else:
        raw = np.asarray(im).astype(np.float32)
        if raw.ndim != 2:
            raise InputError('16-bit image is not single-channel')
        lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
        if hi <= lo:
            raise InputError('16-bit image has no intensity range between the 1st and 99th percentile (blank image?)')
        g = (np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        rgb = np.stack([g] * 3, -1)
    return np.ascontiguousarray(rgb), info
