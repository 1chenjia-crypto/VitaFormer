#!/usr/bin/env python3
"""ImageNet-pretrained SegFormer-B2 baseline on the frozen CRC condition-aware split (P0-B_PROTOCOL.md §4).

- Encoder: nvidia/mit-b2 (revision 3bb39e87; loaded with torch.load(weights_only=True); no repo code run).
- Augmentation policy identical to the existing OrganoidSegBaseline baselines (random scale 0.5-2.0, hflip,
  zero-pad to 512, random 512 crop), re-implemented with PIL/numpy. ImageNet normalisation.
- Loss: weighted CE with the existing baselines' weights (1/sqrt(freq), mean-normalised, clipped [0.2, 10]).
- AdamW 6e-5 (head x10), wd 0.01, linear warmup 1500 then poly(1.0), batch 8, 40k iters, AMP.
- Validation every 1000 iters (sliding 512/256 on 164 val images); best val mIoU checkpoint.
- Test once at the end with the best checkpoint; hard masks saved as PNG for the shared evaluator.
"""

from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerConfig, SegformerForSemanticSegmentation

VOC = Path('OrganoidSegBaseline/VOC2012')
SPLITS = {k: VOC / f'ImageSets/Segmentation/{k}_cond.txt' for k in ('train', 'val', 'test')}
PRE = Path('3D_OrganoidSeg/baselines_fair_20261030/pretrained/nvidia_mit-b2')
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
CROP, WIN, STRIDE, NC = 512, 512, 256, 4


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ids(split):
    return [s.strip() for s in SPLITS[split].read_text().splitlines() if s.strip()]


def load(stem):
    img = np.asarray(Image.open(VOC / 'JPEGImages' / f'{stem}.jpg').convert('RGB'))
    return img, np.asarray(Image.open(VOC / 'SegmentationClass' / f'{stem}.png'))


def augment(img, mask, rng):
    s = rng.uniform(0.5, 2.0)
    h, w = mask.shape
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR))
    mask = np.asarray(Image.fromarray(mask).resize((nw, nh), Image.NEAREST))
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    ph, pw = max(CROP - nh, 0), max(CROP - nw, 0)
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)))
        mask = np.pad(mask, ((0, ph), (0, pw)))
    h, w = mask.shape
    y, x = rng.integers(0, h - CROP + 1), rng.integers(0, w - CROP + 1)
    return np.ascontiguousarray(img[y:y + CROP, x:x + CROP]), np.ascontiguousarray(mask[y:y + CROP, x:x + CROP])


def norm(img):
    return torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())


class TrainSet(Dataset):
    def __init__(self, seed):
        self.ids, self.seed = ids('train'), seed

    def __len__(self):
        return 10 ** 7  # iteration-based sampling

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        stem = self.ids[rng.integers(len(self.ids))]
        img, mask = augment(*load(stem), rng)
        return norm(img), torch.from_numpy(mask.astype(np.int64))


def class_weights():
    c = np.zeros(NC, np.int64)
    for s in ids('train'):
        c += np.bincount(np.asarray(Image.open(VOC / 'SegmentationClass' / f'{s}.png')).ravel(), minlength=NC)[:NC]
    f = c / c.sum()
    w = 1 / np.sqrt(np.maximum(f, 1e-8))
    return c, np.clip(w / w.mean(), 0.2, 10.0).astype(np.float32)


def build_model():
    """transformers' own loader handles the legacy->current key renaming of the mit-b2 checkpoint. Hard checks:
    no encoder parameter may be missing, and every checkpoint tensor must be found (by exact value) in the model."""
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        PRE, num_labels=NC, ignore_mismatched_sizes=True, output_loading_info=True)
    missing_enc = [k for k in info['missing_keys'] if not k.startswith('decode_head')]
    if missing_enc:
        raise RuntimeError(f'pretrained encoder not loaded: {missing_enc[:5]} ... ({len(missing_enc)})')
    sd = torch.load(PRE / 'pytorch_model.bin', map_location='cpu', weights_only=True)
    model_tensors = {}
    for p in model.segformer.state_dict().values():
        model_tensors.setdefault(tuple(p.shape), []).append(p)
    ck = [v for k, v in sd.items() if k.startswith('segformer.')]
    matched = sum(any(torch.equal(v, m) for m in model_tensors.get(tuple(v.shape), [])) for v in ck)
    if matched != len(ck):
        raise RuntimeError(f'only {matched}/{len(ck)} checkpoint encoder tensors found in the model')
    return model, {'missing_keys': sorted(info['missing_keys']), 'unexpected_keys': sorted(info['unexpected_keys']),
                   'mismatched_keys': sorted(str(m) for m in info.get('mismatched_keys', [])),
                   'checkpoint_encoder_tensors_verified_equal': f'{matched}/{len(ck)}'}


@torch.inference_mode()
def predict(model, img, device):
    h, w = img.shape[:2]
    ph, pw = max(WIN - h, 0), max(WIN - w, 0)
    x = norm(np.pad(img, ((0, ph), (0, pw), (0, 0)), mode='reflect'))
    H, W = x.shape[1:]
    ys = list(range(0, H - WIN + 1, STRIDE)); xs = list(range(0, W - WIN + 1, STRIDE))
    if ys[-1] + WIN < H: ys.append(H - WIN)
    if xs[-1] + WIN < W: xs.append(W - WIN)
    score = torch.zeros((NC, H, W), device=device); cnt = torch.zeros((H, W), device=device)
    coords = [(y, xx) for y in ys for xx in xs]
    for b in range(0, len(coords), 16):
        cc = coords[b:b + 16]
        batch = torch.stack([x[:, y:y + WIN, xx:xx + WIN] for y, xx in cc]).to(device)
        with torch.amp.autocast('cuda'):
            lg = model(pixel_values=batch).logits
        pr = F.softmax(F.interpolate(lg.float(), size=(WIN, WIN), mode='bilinear', align_corners=False), 1)
        for j, (y, xx) in enumerate(cc):
            score[:, y:y + WIN, xx:xx + WIN] += pr[j]; cnt[y:y + WIN, xx:xx + WIN] += 1
    return (score / cnt.clamp_min(1)).argmax(0)[:h, :w].byte().cpu().numpy()


def miou(model, split, device, save_dir=None):
    tp = np.zeros(NC); fp = np.zeros(NC); fn = np.zeros(NC)
    for s in ids(split):
        img, gt = load(s)
        p = predict(model, img, device)
        if save_dir is not None:
            Image.fromarray(p).save(save_dir / f'{s}.png')
        v = gt < NC
        for c in range(NC):
            pc, tc = (p == c) & v, (gt == c) & v
            tp[c] += (pc & tc).sum(); fp[c] += (pc & ~tc).sum(); fn[c] += (~pc & tc).sum()
    iou = tp / np.maximum(tp + fp + fn, 1)
    return float(iou.mean() * 100), (iou * 100).round(3).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=40000)
    ap.add_argument('--val-every', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')
    counts, w = class_weights()
    model, load_info = build_model()
    model.to(device)
    head = [p for n, p in model.named_parameters() if n.startswith('decode_head')]
    enc = [p for n, p in model.named_parameters() if not n.startswith('decode_head')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': 6e-5}, {'params': head, 'lr': 6e-4}], weight_decay=0.01)
    base = [6e-5, 6e-4]
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, device=device), ignore_index=255)
    scaler = torch.amp.GradScaler('cuda')
    loader = iter(DataLoader(TrainSet(args.seed), batch_size=args.batch, num_workers=args.workers,
                             pin_memory=True, persistent_workers=True))
    meta = {'created': '2026-10-30', 'script': __file__, 'script_sha256': sha256(__file__), 'args': vars(args),
            'pretrained': str(PRE), 'pretrained_sha256': sha256(PRE / 'pytorch_model.bin'), 'load_info': load_info,
            'class_counts_train': counts.tolist(), 'class_weights': w.tolist(),
            'splits_sha256': {k: sha256(v) for k, v in SPLITS.items()},
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'torch': torch.__version__}
    (out / 'run_meta.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ('load_info', 'class_weights', 'trainable_params')}), flush=True)
    best, hist, t0 = -1.0, [], time.time()
    model.train()
    for it in range(1, args.iters + 1):
        f = it / 1500 if it <= 1500 else (1 - (it - 1500) / (args.iters - 1500))
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * f
        x, y = next(loader)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda'):
            lg = model(pixel_values=x).logits
            lg = F.interpolate(lg, size=y.shape[-2:], mode='bilinear', align_corners=False)
            loss = crit(lg.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if it % 100 == 0:
            print(f'iter {it} loss {loss.item():.4f} lr {opt.param_groups[0]["lr"]:.2e} {time.time() - t0:.0f}s', flush=True)
        if it % args.val_every == 0 or it == args.iters:
            model.eval()
            m, per = miou(model, 'val', device)
            model.train()
            hist.append({'iter': it, 'val_mIoU': m, 'val_iou_per_class': per})
            (out / 'val_history.json').write_text(json.dumps(hist, indent=1))
            if m > best:
                best = m
                torch.save({'model': model.state_dict(), 'iter': it, 'val_mIoU': m}, out / 'best_val_miou.pth')
            print(f'VAL iter {it} mIoU {m:.3f} best {best:.3f} per-class {per}', flush=True)
    ck = torch.load(out / 'best_val_miou.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(ck['model']); model.eval()
    (out / 'test_masks').mkdir()
    tm, tper = miou(model, 'test', device, save_dir=out / 'test_masks')
    (out / 'test_summary.json').write_text(json.dumps({'best_iter': ck['iter'], 'best_val_mIoU': ck['val_mIoU'],
                                                       'test_mIoU_internal': tm, 'test_iou_per_class': tper,
                                                       'elapsed_seconds': time.time() - t0}, indent=2))
    print(f'TEST best_iter {ck["iter"]} mIoU {tm:.3f} per-class {tper}', flush=True)


if __name__ == '__main__':
    main()
