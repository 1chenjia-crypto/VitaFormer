#!/usr/bin/env python3
"""External-dataset (OrgaSegment / OrganoIDNetData) protocol-B training of frozen VitaFormer M2 (2026-11-08).

Derived from train_brain_m2_20261108.py; ONLY the data layer changed (source-paper split readers, grayscale->3ch rule,
PIL mask reader, --dataset instead of --fold). Model, transfer, augmentation, loss, optimiser, schedule, inference unchanged.
Plans: baselines_fair_20261030/external_m2_20261108/{orgasegment,organoidnet}/ANALYSIS_PLAN.md.

Original docstring follows:

Copy of train_brain_strong_baseline_20261104.py; ONLY change: the "vitaformer" arm builds M2
(model_v3_multiscale.OrganoidSegMultiScale) initialised from the frozen M2 seed-42 final.pth
(MODEL_FREEZE_M2_20261108.md). Data, folds, augmentation, loss, optimiser, schedule, inference, evaluator unchanged.

The split membership comes only from the source authors' released splits.csv.
Checkpoint selection uses the complete official validation partition and the
mean image-level Dice. Test images are evaluated once after checkpoint
selection. Existing runs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF


ROOT = Path("3D_OrganoidSeg")
BASELINE_ROOT = Path("OrganoidSegBaseline")
DATA_ROOT = ROOT / "external_datasets/BrainOrganoid2024/data"
OFFICIAL_SPLITS = (
    BASELINE_ROOT
    / "results/brain_organoid2024_metric_alignment_audit/source_cache/"
    "robust_monitoring_organoid_growth-main/model_training/splits.csv"
)
VITA_CKPT = ROOT / "vco/runs/arch_exploration_20261104/stageB_m2_seed42/final.pth"
SEG_CKPT = (
    ROOT
    / "baselines_fair_20261030/segformer_b2_stable_runs/"
    "baseline_seed42/best_val_miou.pth"
)
BCLIP = Path("pretrained/biomedclip/open_clip_pytorch_model.bin")
SEG_PRE = ROOT / "baselines_fair_20261030/pretrained/nvidia_mit-b2"

EXPECTED_SPLIT_SHA = "493c7635286b1150c627c604d76238d897c9bc1e2ca5c4e6df14308d06382e22"
EXPECTED_CKPT_SHA = {
    "vitaformer": "0f12be44859d06723ebfa35e8416df8cd4b02ff83f3ac153654d0f68df226101",
    "segformer": "ba47f5e0de1da43fe8a8601fd010472c6ef6473930b36aa514729dd328aa60e1",
}
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], np.float32)
VITA_MEAN = [0.485, 0.456, 0.406]
VITA_STD = [0.229, 0.224, 0.225]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, obj) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


EXT = ROOT / "external_datasets"
ORGASEG = EXT / "OrgaSegment/OrganoidBasic_v20211206"
OIDN = EXT / "OrganoIDNetData/OrganoIDNetData/Dataset"
EXPECTED_COUNTS = {"orgasegment": (184, 35, 12), "organoidnet": (713 + 1426, 179, 10)}


def load_image(path: Path) -> np.ndarray:
    # Plan rule (2026-11-08): every image -> 8-bit grayscale (PIL "L"), replicated to 3 channels.
    im = Image.open(path)
    if im.mode not in ("L", "RGB", "RGBA", "P"):
        x = np.asarray(im).astype(np.float32)
        x -= x.min()
        im = Image.fromarray((255 * x / max(float(x.max()), 1.0)).clip(0, 255).astype(np.uint8))
    x = np.asarray(im.convert("L"))
    return np.repeat(x[..., None], 3, axis=2)


def load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path)) > 0).astype(np.uint8)


def _row(stem, img, lab, group):
    if not Path(img).exists() or not Path(lab).exists():
        raise FileNotFoundError(f"{img} / {lab}")
    return dict(stem=stem, image_path=str(img), label_path=str(lab), org_id=stem, Clone=group, Day="", Imaging=group)


def read_rows(dataset: str) -> Dict[str, List[dict]]:
    out = {"train": [], "val": [], "test": []}
    if dataset == "orgasegment":
        for split, folder in (("train", "train"), ("val", "val"), ("test", "eval")):
            for img in sorted((ORGASEG / folder).glob("*_img.jpg")):
                stem = img.name[: -len("_img.jpg")]
                lab = img.with_name(f"{stem}_masks_organoid.png")
                out[split].append(_row(stem, img, lab, f"{Image.open(img).size[0]}px"))
    elif dataset == "organoidnet":
        for split, folders in (("train", ("Training", "Augmented")), ("val", ("Validation",)), ("test", ("Test",))):
            for folder in folders:
                for img in sorted((OIDN / folder / "Images").glob("*.tif")):
                    lab = OIDN / folder / "Masks" / f"{img.stem}_mask.tif"
                    group = "human" if "Human" in img.stem else "mouse"
                    out[split].append(_row(img.stem, img, lab, group))
    else:
        raise ValueError(dataset)
    got = tuple(len(out[k]) for k in ("train", "val", "test"))
    if got != EXPECTED_COUNTS[dataset]:
        raise RuntimeError(f"source split count mismatch: got {got}, expected {EXPECTED_COUNTS[dataset]}")
    stems = [set(r["image_path"] for r in out[k]) for k in ("train", "val", "test")]
    if stems[0] & stems[1] or stems[0] & stems[2] or stems[1] & stems[2]:
        raise RuntimeError("file overlap between partitions")
    return out


def normalize_segformer(img: np.ndarray) -> torch.Tensor:
    x = ((img.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(x))


class TrainDataset(Dataset):
    def __init__(self, rows: Sequence[dict], model_name: str, seed: int):
        self.rows = list(rows)
        self.model_name = model_name
        self.seed = seed
        self.crop = 224 if model_name == "vitaformer" else 512

    def __len__(self) -> int:
        return 10**7

    def __getitem__(self, index: int):
        rng = np.random.default_rng((self.seed, index))
        row = self.rows[int(rng.integers(len(self.rows)))]
        img = load_image(Path(row["image_path"]))
        mask = load_mask(Path(row["label_path"]))
        if img.shape[:2] != mask.shape:
            mask = np.asarray(
                Image.fromarray(mask).resize((img.shape[1], img.shape[0]), Image.Resampling.NEAREST)
            )
        if self.model_name == "segformer":
            scale = float(rng.uniform(0.5, 2.0))
            h, w = mask.shape
            nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
            img = np.asarray(Image.fromarray(img).resize((nw, nh), Image.Resampling.BILINEAR))
            mask = np.asarray(Image.fromarray(mask).resize((nw, nh), Image.Resampling.NEAREST))
        if float(rng.random()) < 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        if self.model_name == "vitaformer" and float(rng.random()) < 0.5:
            img, mask = img[::-1].copy(), mask[::-1].copy()
        h, w = mask.shape
        ph, pw = max(self.crop - h, 0), max(self.crop - w, 0)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
        h, w = mask.shape
        y = int(rng.integers(0, h - self.crop + 1))
        x = int(rng.integers(0, w - self.crop + 1))
        img = np.ascontiguousarray(img[y : y + self.crop, x : x + self.crop])
        mask = np.ascontiguousarray(mask[y : y + self.crop, x : x + self.crop])
        if self.model_name == "vitaformer":
            image_t = TF.normalize(TF.to_tensor(Image.fromarray(img)), VITA_MEAN, VITA_STD)
        else:
            image_t = normalize_segformer(img)
        return image_t, torch.from_numpy(mask.astype(np.int64))


def load_matching_weights(
    model,
    checkpoint: Path,
    expected_sha256: str,
    container_key: str,
    excluded_prefixes: Tuple[str, ...],
):
    actual_sha256 = sha256(checkpoint)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"source checkpoint SHA256 mismatch: got {actual_sha256}, expected {expected_sha256}"
        )
    obj = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = obj[container_key] if container_key else obj
    target = model.state_dict()
    copied = {}
    skipped = []
    for key, value in source.items():
        if key.startswith(excluded_prefixes) or key not in target or target[key].shape != value.shape:
            skipped.append(key)
            continue
        copied[key] = value
    result = model.load_state_dict(copied, strict=False)
    nonhead_missing = [k for k in result.missing_keys if not k.startswith(excluded_prefixes)]
    if nonhead_missing or result.unexpected_keys:
        raise RuntimeError(
            f"non-head transfer mismatch missing={nonhead_missing[:8]} unexpected={result.unexpected_keys[:8]}"
        )
    return {"copied_tensors": len(copied), "skipped_keys": sorted(skipped)}


def build_vitaformer():
    model_dir = ROOT / "vco/organoid_seg"
    sys.path.insert(0, str(model_dir))
    from model_v3_multiscale import OrganoidSegMultiScale

    model = OrganoidSegMultiScale(str(BCLIP), num_classes=2, freeze_blocks=0, zero_cnn_input=False)
    info = load_matching_weights(
        model,
        VITA_CKPT,
        EXPECTED_CKPT_SHA["vitaformer"],
        "model_state",
        ("seg_head",),
    )
    return model, info


def build_segformer():
    from transformers import SegformerForSemanticSegmentation

    model = SegformerForSemanticSegmentation.from_pretrained(
        SEG_PRE, num_labels=2, ignore_mismatched_sizes=True
    )
    info = load_matching_weights(
        model,
        SEG_CKPT,
        EXPECTED_CKPT_SHA["segformer"],
        "model",
        ("decode_head.classifier",),
    )
    return model, info


def model_logits(model_name: str, model, x: torch.Tensor) -> torch.Tensor:
    if model_name == "vitaformer":
        out = model(x)
        return out[0] if isinstance(out, tuple) else out
    return model(pixel_values=x).logits


@torch.inference_mode()
def predict_probability(model_name: str, model, image: np.ndarray, device: torch.device) -> np.ndarray:
    window = 224 if model_name == "vitaformer" else 512
    stride = window // 2
    h, w = image.shape[:2]
    ph, pw = max(window - h, 0), max(window - w, 0)
    padded = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    H, W = padded.shape[:2]
    ys = list(range(0, H - window + 1, stride))
    xs = list(range(0, W - window + 1, stride))
    if ys[-1] + window < H:
        ys.append(H - window)
    if xs[-1] + window < W:
        xs.append(W - window)
    score = torch.zeros((H, W), dtype=torch.float32, device=device)
    count = torch.zeros((H, W), dtype=torch.float32, device=device)
    coords = [(y, x) for y in ys for x in xs]
    infer_batch = 16 if model_name == "segformer" else 8
    amp_dtype = torch.bfloat16 if model_name == "segformer" else torch.float16
    for start in range(0, len(coords), infer_batch):
        group = coords[start : start + infer_batch]
        tensors = []
        for y, x in group:
            patch = padded[y : y + window, x : x + window]
            if model_name == "vitaformer":
                t = TF.normalize(TF.to_tensor(Image.fromarray(patch)), VITA_MEAN, VITA_STD)
            else:
                t = normalize_segformer(patch)
            tensors.append(t)
        batch = torch.stack(tensors).to(device)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model_logits(model_name, model, batch)
        logits = F.interpolate(logits.float(), (window, window), mode="bilinear", align_corners=False)
        probs = logits.softmax(1)[:, 1]
        for j, (y, x) in enumerate(group):
            score[y : y + window, x : x + window] += probs[j]
            count[y : y + window, x : x + window] += 1
    return (score / count.clamp_min(1))[:h, :w].cpu().numpy()


def evaluate(model_name: str, model, rows: Sequence[dict], device: torch.device, save_dir: Path | None = None):
    per_image = []
    total = dict(tp=0, fp=0, fn=0, tn=0)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=False)
    for row in rows:
        image = load_image(Path(row["image_path"]))
        gt = load_mask(Path(row["label_path"])).astype(bool)
        prob = predict_probability(model_name, model, image, device)
        pred = prob >= 0.5
        tp = int((pred & gt).sum())
        fp = int((pred & ~gt).sum())
        fn = int((~pred & gt).sum())
        tn = int((~pred & ~gt).sum())
        for key, value in zip(("tp", "fp", "fn", "tn"), (tp, fp, fn, tn)):
            total[key] += value
        dice = 2 * tp / max(2 * tp + fp + fn, 1)
        iou = tp / max(tp + fp + fn, 1)
        per_image.append(
            {
                "image": row["stem"],
                "org_id": row["org_id"],
                "clone": row["Clone"],
                "day": row["Day"],
                "lab": row["Imaging"],
                "dice": dice,
                "iou": iou,
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
        if save_dir is not None:
            Image.fromarray(pred.astype(np.uint8) * 255).save(save_dir / f"{row['stem']}.png")
    t = total
    summary = {
        "n_images": len(per_image),
        "image_dice_mean": float(np.mean([r["dice"] for r in per_image])),
        "image_dice_sd": float(np.std([r["dice"] for r in per_image], ddof=1)),
        "image_iou_mean": float(np.mean([r["iou"] for r in per_image])),
        "micro_dice": 2 * t["tp"] / max(2 * t["tp"] + t["fp"] + t["fn"], 1),
        "micro_iou": t["tp"] / max(t["tp"] + t["fp"] + t["fn"], 1),
        **total,
    }
    return summary, per_image


def class_weights(rows: Sequence[dict]) -> np.ndarray:
    counts = np.zeros(2, dtype=np.int64)
    for row in rows:
        counts += np.bincount(load_mask(Path(row["label_path"])).ravel(), minlength=2)
    freq = counts / counts.sum()
    weights = 1 / np.sqrt(np.maximum(freq, 1e-8))
    return np.clip(weights / weights.mean(), 0.2, 10.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("vitaformer", "segformer"), required=True)
    ap.add_argument("--dataset", choices=("orgasegment", "organoidnet"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--val-every", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.fold = 0  # single source-split run; keeps seed = 42 + 0 as in Brain fold0

    if not args.dry_run and args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    random.seed(args.seed + args.fold)
    np.random.seed(args.seed + args.fold)
    torch.manual_seed(args.seed + args.fold)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.fold)
    rows = read_rows(args.dataset)
    model, load_info = build_vitaformer() if args.model == "vitaformer" else build_segformer()
    batch_size = args.batch or 8
    dataset = TrainDataset(rows["train"], args.model, args.seed + args.fold)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=args.device != "cpu" and torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
    )
    use_cuda = torch.cuda.is_available() if args.device == "auto" else args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)
    weights = class_weights(rows["train"])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    if args.dry_run:
        x, y = next(iter(loader))
        x, y = x.to(device), y.to(device)
        logits = model_logits(args.model, model, x)
        logits = F.interpolate(logits.float(), y.shape[-2:], mode="bilinear", align_corners=False)
        loss = criterion(logits, y)
        print(
            json.dumps(
                {
                    "model": args.model,
                    "fold": args.fold,
                    "device": str(device),
                    "batch_shape": list(x.shape),
                    "logits_shape": list(logits.shape),
                    "loss": float(loss),
                    "split_counts": {k: len(v) for k, v in rows.items()},
                    "load_info": load_info,
                },
                indent=2,
            )
        )
        return

    args.out.mkdir(parents=True, exist_ok=False)
    checkpoint = VITA_CKPT if args.model == "vitaformer" else SEG_CKPT
    meta = {
        "created": "2026-11-08", "arm": "VitaFormer M2 (frozen, seed42 final.pth)",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "args": {**vars(args), "out": str(args.out)},
        "dataset": args.dataset,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "load_info": load_info,
        "split_counts": {k: len(v) for k, v in rows.items()},
        "class_weights": weights.tolist(),
        "threshold": 0.5,
        "tta": False,
        "primary_checkpoint_metric": "validation image-level mean Dice",
        "torch": torch.__version__,
        "device": str(device),
    }
    dump_json(args.out / "run_meta.json", meta)

    if args.model == "vitaformer":
        enc = [p for n, p in model.named_parameters() if n.startswith("encoder")]
        other = [p for n, p in model.named_parameters() if not n.startswith("encoder")]
        optimizer = torch.optim.AdamW(
            [{"params": enc, "lr": 2e-6}, {"params": other, "lr": 2e-5}],
            weight_decay=1e-4,
        )
        base_lrs = [2e-6, 2e-5]
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:
        enc = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
        head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
        optimizer = torch.optim.AdamW(
            [{"params": enc, "lr": 6e-5}, {"params": head, "lr": 6e-4}],
            weight_decay=0.01,
        )
        base_lrs = [6e-5, 6e-4]
        amp_dtype = torch.bfloat16
        scaler = None
    scheduler = None
    if args.model == "vitaformer":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iters, eta_min=1e-7
        )
    iterator = iter(loader)
    best = -1.0
    history = []
    started = time.time()
    for step in range(1, args.iters + 1):
        model.train()
        if args.model == "segformer":
            warmup = 1500
            lr_factor = (
                step / warmup
                if step <= warmup
                else 1 - (step - warmup) / (args.iters - warmup)
            )
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * lr_factor
        x, y = next(iterator)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model_logits(args.model, model, x)
            logits = F.interpolate(logits, y.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits.float(), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % 100 == 0:
            print(
                f"step={step}/{args.iters} loss={loss.item():.6f} "
                f"grad_norm_preclip={grad_norm:.4f} lr={optimizer.param_groups[0]['lr']:.3e} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
        if step % args.val_every == 0 or step == args.iters:
            model.eval()
            val_summary, _ = evaluate(args.model, model, rows["val"], device)
            record = {"step": step, **val_summary}
            history.append(record)
            dump_json(args.out / "val_history.json", history)
            if val_summary["image_dice_mean"] > best:
                best = val_summary["image_dice_mean"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "step": step,
                        "val_image_dice_mean": best,
                    },
                    args.out / "best.pth",
                )
            print(
                f"VAL step={step} image_dice_mean={val_summary['image_dice_mean']:.6f} "
                f"micro_dice={val_summary['micro_dice']:.6f} best={best:.6f}",
                flush=True,
            )
    torch.save(
        {"model_state": model.state_dict(), "step": args.iters},
        args.out / "last.pth",
    )
    best_obj = torch.load(args.out / "best.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(best_obj["model_state"])
    model.to(device).eval()
    test_summary, per_image = evaluate(
        args.model, model, rows["test"], device, save_dir=args.out / "test_masks"
    )
    dump_json(
        args.out / "result.json",
        {
            "model": args.model,
            "fold": args.fold,
            "best_step": best_obj["step"],
            "best_val_image_dice_mean": best_obj["val_image_dice_mean"],
            "test": test_summary,
            "per_image": per_image,
            "elapsed_seconds": time.time() - started,
        },
    )
    print(json.dumps({"TEST": test_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
