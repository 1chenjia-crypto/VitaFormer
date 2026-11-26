"""Stage G shared helpers (AMENDMENT_1_20261106.md). numpy/scipy/PIL only, so both the VF env and the nnU-Net env can use it.

Score s = log p_dead - log max(p_bg, p_alive, p_bubble). Dead iff s > delta, else argmax of the other three.
Fine edges e_k = k/100, k = -2000..2000 (exact 0.0 at k = 0). bin(s) = searchsorted(e, s, 'left') in 0..4001, so
s > e_k  <=>  bin(s) > k + 2000. Main grid delta_j = (j - 40)/10, j = 0..80; level(s) = #{j : s > delta_j} in 0..81, so
Dead at delta_j <=> level > j.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

VOC_GT = 'OrganoidSegBaseline/VOC2012/SegmentationClass'
NC = 4
EDGES = np.arange(-2000, 2001) / 100.0
NB = len(EDGES) + 1  # 4002 bins
MAIN = np.arange(-40, 41) / 10.0
OTHER = np.array([0, 1, 3])  # class ids of the three non-Dead classes
PBINS = 1000


def score_from_probs(p):
    """p: (4, H, W) float probabilities -> s (float32), other-argmax class ids (uint8)."""
    p = p.astype(np.float32)
    po = p[OTHER]
    oi = po.argmax(0)
    pmax = np.take_along_axis(po, oi[None], 0)[0]
    s = np.log(np.maximum(p[2], 1e-30)) - np.log(np.maximum(pmax, 1e-30))
    return s.astype(np.float32), OTHER[oi].astype(np.uint8)


def decide(s, other, delta=0.0):
    return np.where(s > delta, 2, other).astype(np.uint8)


def farbg(gt):
    valid = gt < NC
    nonbg = np.where(valid, gt, 0) > 0
    d = ndi.distance_transform_edt(~nonbg) if nonbg.any() else np.full(gt.shape, np.inf)
    return valid & (d > 8)


def summarize(s, other, pdead, gt):
    """Per-image sufficient statistics. Returns dict of arrays + uint8 level map."""
    valid = gt < NC
    b = np.searchsorted(EDGES, s, side='left')  # 0..4001
    oidx = np.searchsorted(OTHER, other)       # 0..2
    H = np.zeros((NC, 3, NB), np.int64)
    for g in range(NC):
        m = valid & (gt == g)
        if m.any():
            H[g] = np.bincount(oidx[m] * NB + b[m], minlength=3 * NB).reshape(3, NB)
    fb = farbg(gt) & (gt != 2)
    HF = np.bincount(b[fb], minlength=NB).astype(np.int64)
    pq = np.minimum((pdead * PBINS).astype(np.int64), PBINS - 1)
    HP = np.stack([np.bincount(pq[valid & (gt != 2)], minlength=PBINS), np.bincount(pq[valid & (gt == 2)], minlength=PBINS)])
    level = np.searchsorted(MAIN, s, side='left').astype(np.uint8)  # #{j: delta_j < s}
    return {'H': H, 'HF': HF, 'HP': HP.astype(np.int64)}, level


def load_gt(stem):
    return np.asarray(Image.open(f'{VOC_GT}/{stem}.png'))
