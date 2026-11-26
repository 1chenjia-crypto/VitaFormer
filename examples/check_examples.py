#!/usr/bin/env python3
"""Check a VitaFormer run of one of the bundled example sets against its expected output.

Usage
  python check_examples.py "<run folder>"

The run folder is the folder VitaFormer creates for each analysis, for example
  C:\\Users\\<you>\\VitaFormer_results\\20261124-141553_run
It contains run_manifest.json, readouts_image_measurements.csv and a masks folder.
The example set (crc or nvir) is recognised automatically from the image names.

What is checked
  1. The run used the frozen VitaFormer seed-42 weights (SHA256 in run_manifest.json).
  2. The run used the input size recommended for the example set.
  3. For every example image: the input file is unchanged (SHA256) and the predicted mask agrees
     with the expected mask pixel by pixel.

Interpreting the result
  Masks computed on a CPU (fp32) and on a GPU (fp16) can differ in a few pixels at class boundaries.
  The CRC reference masks were computed on a GPU; the NViR reference masks on a CPU.
  The check passes when every image agrees on at least 99.9 % of its pixels.

Only the Python standard library, NumPy and Pillow are needed; the Python runtime installed with
VitaFormer contains both (see check_examples.bat).
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SETS = [p for p in sorted(HERE.iterdir()) if (p / 'example.json').exists()]


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as fh:
        return list(csv.DictReader(fh))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def strip_index(name):
    """VitaFormer prefixes saved files with the input number, e.g. 0001_<name>."""
    return re.sub(r'^\d{4}_', '', name)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run = Path(sys.argv[1].strip('"'))
    if not (run / 'run_manifest.json').exists() or not (run / 'readouts_image_measurements.csv').exists():
        print(f'FAIL: "{run}" is not a VitaFormer run folder (run_manifest.json or readouts_image_measurements.csv missing).')
        return 1
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8-sig'))
    rows = {strip_index(r['image']): r for r in read_csv(run / 'readouts_image_measurements.csv')}

    # which example set was run?
    best, overlap = None, 0
    for s in SETS:
        names = {r['image'] for r in read_csv(s / 'expected_output' / 'expected_counts.csv')}
        if len(names & set(rows)) > overlap:
            best, overlap = s, len(names & set(rows))
    if best is None:
        print('FAIL: none of the images in this run belongs to an example set '
              f'({", ".join(s.name for s in SETS)}).')
        return 1
    meta = json.loads((best / 'example.json').read_text(encoding='utf-8'))
    expected = read_csv(best / 'expected_output' / 'expected_counts.csv')
    print(f'Example set: {best.name} - {meta["name"]}')
    print(f'VitaFormer version: {manifest.get("tool_version", "?")}   device: {manifest.get("device", "?")}')

    ok = True
    if manifest.get('weights_sha256') != meta['weights_sha256']:
        print(f'FAIL  weights: SHA256 {manifest.get("weights_sha256")} is not the frozen VitaFormer seed-42 file.')
        ok = False
    else:
        print('OK    weights: frozen VitaFormer seed 42')
    mode, value = manifest.get('resize_mode'), manifest.get('resize_value')
    same_value = (value is None and meta['resize_value'] is None) or \
                 (value is not None and meta['resize_value'] is not None and float(value) == float(meta['resize_value']))
    if mode != meta['resize_mode'] or not same_value:
        print(f'FAIL  input size: run used {mode} {value}; this example needs {meta["resize_mode"]} '
              f'{meta["resize_value"] if meta["resize_value"] is not None else ""}'.rstrip())
        ok = False
    else:
        print(f'OK    input size: {mode} {value if value is not None else ""}'.rstrip())

    inputs = {strip_index(Path(i.get('image', '')).name): i for i in manifest.get('inputs', [])}
    worst, missing = 1.0, []
    print()
    print(f'{"image":58s} {"agreement":>10s} {"Alive px run/exp":>18s} {"Dead px run/exp":>17s}')
    for e in expected:
        name = e['image']
        if name not in rows:
            missing.append(name)
            continue
        r = rows[name]
        if name in inputs and inputs[name].get('sha256') and inputs[name]['sha256'] != e['input_sha256']:
            print(f'FAIL  {name}: the input file differs from the example file (SHA256).')
            ok = False
        got = np.asarray(Image.open(run / r['mask_file']))
        exp = np.asarray(Image.open(best / e['expected_mask']))
        if got.shape != exp.shape:
            print(f'FAIL  {name}: mask size {got.shape} differs from expected {exp.shape}.')
            ok = False
            continue
        agree = float((got == exp).mean())
        worst = min(worst, agree)
        print(f'{name[:58]:58s} {agree:10.6f} {int((got == 1).sum()):>8d}/{int(e["alive_px"]):<9d}'
              f' {int((got == 2).sum()):>7d}/{int(e["dead_px"]):<9d}')
    print()
    if missing:
        print(f'INCOMPLETE: {len(missing)} of {len(expected)} example images were not in this run:')
        for m in missing:
            print('   ', m)
        ok = False
    if worst < meta['min_agreement']:
        print(f'FAIL  lowest per-image agreement {worst:.6f} (required >= {meta["min_agreement"]}).')
        ok = False
    if ok:
        kind = 'identical' if worst == 1.0 else f'lowest per-image agreement {worst:.6f}; boundary pixels can differ between CPU and GPU'
        print(f'PASS ({kind})')
        return 0
    print('NOT PASSED - see the lines above.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
