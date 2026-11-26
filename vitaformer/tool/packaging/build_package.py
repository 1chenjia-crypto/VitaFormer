#!/usr/bin/env python3
"""Assemble the distributable folder dist/VitaFormer_v<version>/ (+ zip). Copies, never modifies, the source files.
Includes the frozen seed-42 weights (checksum verified). Excludes development-only material with an unconfirmed licence (examples/ CRC)."""
import hashlib, shutil, sys, zipfile
from pathlib import Path
T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T)); from vitaformer import __version__  # noqa
from vitaformer.infer import KNOWN_WEIGHTS  # noqa
D = T / 'dist' / f'VitaFormer_v{__version__}'
if D.exists():
    sys.exit(f'{D} exists - remove it first (no overwriting)')
ign = shutil.ignore_patterns('__pycache__', '*.pyc', '.DS_Store')
D.mkdir(parents=True)
shutil.copytree(T / 'vitaformer', D / 'vitaformer', ignore=ign)
shutil.copytree(T / 'examples_nvir', D / 'examples_nvir', ignore=ign)
shutil.copytree(T / 'tests', D / 'tests', ignore=ign)
(D / 'docs').mkdir(); [shutil.copy2(f, D / 'docs' / f.name) for f in (T / 'docs').glob('*.md')]
shutil.copytree(T / 'docs/screenshots', D / 'docs/screenshots')
for f in ['install_windows.bat', 'start_VitaFormer.bat', 'install.sh', 'start.sh', 'requirements_cpu.txt', 'requirements_gpu_cu124.txt']:
    shutil.copy2(T / 'packaging' / f, D / f)
for f in ['README.md', 'CHANGELOG.md']:
    shutil.copy2(T / f, D / f)
(D / 'weights').mkdir()
w = (T / 'weights/vitaformer_m2_seed42_final.pth').resolve(); dst = D / 'weights/vitaformer_m2_seed42_final.pth'
shutil.copy2(w, dst)
h = hashlib.sha256(dst.read_bytes()).hexdigest()
assert h in KNOWN_WEIGHTS, 'weights checksum mismatch'
lines = []
for f in sorted(D.rglob('*')):
    if f.is_file():
        lines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(D).as_posix()}')
(D / 'SHA256SUMS.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
z = D.parent / (D.name + '.zip')
with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zz:
    for f in sorted(D.rglob('*')):
        if f.is_file():
            zz.write(f, f.relative_to(D.parent).as_posix())
print(D, '\n', z, round(z.stat().st_size / 1e6, 1), 'MB;', len(lines), 'files; weights', h[:12])
