#!/usr/bin/env python3
"""Stage A smoke test (no validation/test metric is computed; no checkpoint is saved).

For M0 (model_v2, no MPM/BAM), M1 (zero CNN input) and M2 (CNN multiscale):
  1. parameter counts (total / trainable with freeze_blocks=4, per module);
  2. GFLOPs per 224 window (torch FlopCounterMode);
  3. 3 training steps on real Rare-Dead-augmented TRAIN crops (same dataset_v2 settings as M0's launch script),
     same CombinedLoss / optimiser groups as train_v2_stratified.py, fp32: shape, finite loss, gradient reaching
     the CNN branch, step time, peak training memory (batch 16);
  4. whole-image inference latency on one TRAIN image (infer_hard protocol, autocast), median of 5, peak memory.
Weights: encoder from BiomedCLIP, rest random (timing/shape only).
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('3D_OrganoidSeg')
sys.path.insert(0, str(ROOT / 'vco/organoid_seg'))
sys.path.insert(0, str(ROOT / 'vco/runs/arch_exploration_20261104/scripts'))
from model_v2 import OrganoidSegV2  # noqa: E402
from model_v3_multiscale import OrganoidSegMultiScale  # noqa: E402
from dataset import build_dataset_index, stratified_train_val_test_split  # noqa: E402
from dataset_v2 import PatchOrganoidDataset  # noqa: E402
import train_v2_stratified as T  # noqa: E402  (guarded by __main__)
from a0_m0_val_inference import infer_hard  # noqa: E402

BCLIP = 'pretrained/biomedclip/open_clip_pytorch_model.bin'
OUT = ROOT / 'vco/runs/arch_exploration_20261104/A_design_smoke'
dev = torch.device('cuda')
torch.manual_seed(0)
np.random.seed(0)


def build(name):
    if name == 'M0':
        return OrganoidSegV2(weight_path=BCLIP, num_classes=4, freeze_blocks=4, use_mpm=False, use_bam=False)
    return OrganoidSegMultiScale(BCLIP, num_classes=4, freeze_blocks=4, zero_cnn_input=(name == 'M1'))


def counts(m):
    tot = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    per = {n: sum(p.numel() for p in c.parameters()) for n, c in m.named_children()}
    return tot, tr, per


def gflops(m):
    from torch.utils.flop_counter import FlopCounterMode
    m.eval()
    x = torch.randn(1, 3, 224, 224, device=dev)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x)
    return fc.get_total_flops() / 1e9


def main():
    all_samples, wells = build_dataset_index('OrganoidSegBaseline/VOC2012')
    tr, va, te = stratified_train_val_test_split(wells, 0.6, 0.2, seed=42, use_condition=True)
    assert (len(tr), len(va), len(te)) == (481, 164, 167), (len(tr), len(va), len(te))
    ds = PatchOrganoidDataset(tr[:64], 224, augment=True, minority_oversample=0.4, use_dead_center_crop=True,
                              dead_center_crop_prob=0.6, sparse_dead_zoom_prob=0.8, sparse_dead_threshold=5000,
                              sparse_dead_zoom_factor=2)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True, num_workers=4, drop_last=True)
    batches = []
    for i, b in enumerate(loader):
        batches.append(b)
        if len(batches) == 3:
            break
    cw = [0.5, 0.5, 10.0, 0.9007]
    img_full = np.asarray(Image.open(tr[0]['image']).convert('RGB'))
    res = {'train_batch': {'n_batches': 3, 'batch_size': 16, 'img_shape': list(batches[0][0].shape)},
           'full_image_shape': list(img_full.shape), 'models': {}}
    for name in ('M0', 'M1', 'M2'):
        torch.cuda.empty_cache()
        m = build(name).to(dev)
        tot, trn, per = counts(m)
        r = {'params_total': tot, 'params_trainable': trn, 'params_by_module': per, 'gflops_224': gflops(m)}
        m.train()
        crit = T.CombinedLoss(class_weights=cw, use_focal=True, gamma=2.0, ce_w=0.4, tversky_w=0.6, lovasz_w=0.0).to(dev)
        enc = [p for p in m.encoder.parameters() if p.requires_grad]
        rest = [p for p in m.parameters() if p.requires_grad and id(p) not in {id(q) for q in enc}]
        opt = torch.optim.AdamW([{'params': enc, 'lr': 1e-5}, {'params': rest, 'lr': 1e-4}], weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats()
        losses, times = [], []
        for imgs, masks, _ in batches:
            imgs, masks = imgs.to(dev), masks.to(dev)
            torch.cuda.synchronize(); t = time.time()
            seg, _ = m(imgs)
            assert seg.shape == (imgs.shape[0], 4, 224, 224), seg.shape
            ld = crit(seg, masks)
            loss = ld['total'] if isinstance(ld, dict) else ld
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            if name != 'M0':
                g = sum(float(p.grad.abs().sum()) for p in m.cnn.parameters() if p.grad is not None)
                r.setdefault('cnn_grad_abs_sum', []).append(g)
            opt.step(); torch.cuda.synchronize(); times.append(time.time() - t)
            losses.append(float(loss))
        r['train_losses'] = losses
        r['all_finite'] = bool(np.all(np.isfinite(losses)))
        r['train_step_s_median_excl_first'] = float(np.median(times[1:]))
        r['peak_train_mem_MB'] = torch.cuda.max_memory_allocated() / 2 ** 20
        m.eval()
        torch.cuda.reset_peak_memory_stats()
        lat = []
        for _ in range(5):
            torch.cuda.synchronize(); t = time.time()
            pred = infer_hard(m, img_full, dev)
            torch.cuda.synchronize(); lat.append(time.time() - t)
        r['full_image_latency_s_median'] = float(np.median(lat))
        r['full_image_latency_s_all'] = lat
        r['peak_infer_mem_MB'] = torch.cuda.max_memory_allocated() / 2 ** 20
        r['pred_shape'] = list(pred.shape)
        res['models'][name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != 'params_by_module'}), flush=True)
        del m, opt
    res['note'] = ('GPU shared with the concurrent A0 validation inference job and possibly other users; '
                   'timings are relative (M0 vs M2 measured in the same session), not absolute benchmarks.')
    (OUT / 'smoke_results.json').write_text(json.dumps(res, indent=2))
    print('done', flush=True)


if __name__ == '__main__':
    main()
