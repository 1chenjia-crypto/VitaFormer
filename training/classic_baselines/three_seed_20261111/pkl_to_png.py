"""Convert a baseline script's test_best_predictions.pkl (list of uint8 HxW, test_cond order) to PNG masks."""
import pickle, sys
from pathlib import Path
import numpy as np
from PIL import Image
pkl, out = Path(sys.argv[1]), Path(sys.argv[2])
VOC = Path('OrganoidSegBaseline/VOC2012')
ids = [s.strip() for s in (VOC / 'ImageSets/Segmentation/test_cond.txt').read_text().splitlines() if s.strip()]
preds = pickle.load(open(pkl, 'rb'))
assert len(preds) == len(ids) == 167, (len(preds), len(ids))
out.mkdir(parents=True, exist_ok=True)
for i, p in zip(ids, preds):
    p = np.asarray(p, dtype=np.uint8); assert p.max() < 4
    Image.fromarray(p).save(out / f'{i}.png')
print('wrote', len(ids), 'masks to', out)
