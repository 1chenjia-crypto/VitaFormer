"""
dataset_v2.py — Native-resolution patch dataset with minority-class oversampling.

Key difference from dataset.py:
  - Images are NOT downscaled to 128px. Instead we crop 256×256 patches from
    the original 2048×2044 images. An organoid (~100px diameter) is thus still
    100px in the model input, not compressed to 6px.
  - Minority oversampling: ~50% of crops are centred on alive/dead pixels so
    the model trains on rare classes far more often.
  - Elastic deformation (scipy) for microscopy-appropriate augmentation.
"""

import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from dataset import build_dataset_index, train_val_split, NUM_CLASSES


CROP = 256        # native-res patch size
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]


# ── Elastic augmentation ──────────────────────────────────────────────────────

def elastic_transform(img_np, mask_np, alpha=40, sigma=5):
    """Elastic deformation — standard microscopy augmentation."""
    from scipy.ndimage import map_coordinates, gaussian_filter
    rng = np.random.RandomState()
    h, w = img_np.shape[:2]
    dx = gaussian_filter(rng.randn(h, w), sigma) * alpha
    dy = gaussian_filter(rng.randn(h, w), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xi = np.clip((xx + dx).ravel(), 0, w - 1)
    yi = np.clip((yy + dy).ravel(), 0, h - 1)
    coords = [yi, xi]
    out_img = np.stack([
        map_coordinates(img_np[:, :, c], coords, order=1).reshape(h, w)
        for c in range(img_np.shape[2])
    ], axis=2).astype(np.uint8)
    out_mask = map_coordinates(mask_np.astype(float), coords, order=0).reshape(h, w).astype(mask_np.dtype)
    return out_img, out_mask


# ── Patch dataset ─────────────────────────────────────────────────────────────

class PatchOrganoidDataset(Dataset):
    """
    Random 256×256 crops from native-resolution images.

    minority_oversample: fraction of crops biased toward alive/dead/bubble pixels.

    Dead-center crop (use_dead_center_crop=True, training only):
      With probability dead_center_crop_prob, crops are centred on a Dead pixel /
      component centroid with random jitter, retrying up to max_dead_crop_retry
      times to ensure >= min_dead_pixels_in_crop Dead pixels land inside the crop.
      Falls back to minority / random crop on failure or when Dead is absent.
    """

    DEAD_CLASS = 2  # class index for Dead

    def __init__(self, samples, crop_size=CROP, augment=False, minority_oversample=0.5,
                 use_dead_center_crop=False, dead_center_crop_prob=0.6,
                 min_dead_pixels_in_crop=20, max_dead_crop_retry=20,
                 dead_pixel_dropout_prob=0.0,
                 sparse_dead_zoom_prob=0.0, sparse_dead_threshold=5000, sparse_dead_zoom_factor=2,
                 copy_paste_dead_prob=0.0):
        self.samples = samples
        self.crop_size = crop_size
        self.augment = augment
        self.minority_oversample = minority_oversample

        # Dead-center crop config (only meaningful during training)
        self.use_dead_center_crop = use_dead_center_crop and augment
        self.dead_center_crop_prob = dead_center_crop_prob
        self.min_dead_pixels_in_crop = min_dead_pixels_in_crop
        self.max_dead_crop_retry = max_dead_crop_retry

        # Dead Pixel Dropout: randomly zero out 60-90% of Dead pixels to simulate sparse Dead
        self.dead_pixel_dropout_prob = dead_pixel_dropout_prob if augment else 0.0

        # Sparse Dead zoom crop: zoom into <threshold-px Dead images to make sparse Dead visible
        self.sparse_dead_zoom_prob = sparse_dead_zoom_prob if augment else 0.0
        self.sparse_dead_threshold = sparse_dead_threshold
        self.sparse_dead_zoom_factor = sparse_dead_zoom_factor

        # Copy-Paste Dead augmentation
        self.copy_paste_dead_prob = copy_paste_dead_prob if augment else 0.0

        self._minority_idx = self._build_minority_index()
        # Build Dead-only index when either dead-center crop or zoom crop is enabled
        need_dead_idx = (self.use_dead_center_crop or self.sparse_dead_zoom_prob > 0
                         or self.copy_paste_dead_prob > 0)
        self._dead_idx = self._build_dead_index() if need_dead_idx else {}
        self._dead_pixel_count = {i: len(v) for i, v in self._dead_idx.items()}

        # Build Dead crop bank for Copy-Paste (training only)
        self._dead_bank = self._build_dead_bank() if self.copy_paste_dead_prob > 0 else []
        if self._dead_bank:
            print(f"[CopyPaste] Dead bank: {len(self._dead_bank)} crops "
                  f"(prob={self.copy_paste_dead_prob})")

    def _build_minority_index(self):
        """Find (y, x) coordinates of alive/dead/bubble pixels per sample."""
        idx = {}
        for i, s in enumerate(self.samples):
            mask = np.array(Image.open(s['mask']))
            minority = np.argwhere((mask == 1) | (mask == 2) | (mask == 3))
            idx[i] = minority
        return idx

    def _build_dead_index(self):
        """Find (y, x) coordinates of Dead pixels per sample (class 2 only)."""
        idx = {}
        for i, s in enumerate(self.samples):
            mask = np.array(Image.open(s['mask']))
            dead = np.argwhere(mask == self.DEAD_CLASS)
            idx[i] = dead  # empty array if no Dead pixels
        return idx

    def _try_dead_center_crop(self, img, mask, sample_idx):
        """
        Attempt to crop a patch that contains >= min_dead_pixels_in_crop Dead pixels,
        centred near a randomly chosen Dead pixel with random jitter.
        Returns (img_crop, mask_crop) on success, None on failure.
        """
        dead_pts = self._dead_idx.get(sample_idx, np.empty((0, 2), dtype=int))
        if len(dead_pts) == 0:
            return None

        H, W = img.shape[:2]
        cs = self.crop_size
        jitter = cs // 4  # ±25% of patch size

        for _ in range(self.max_dead_crop_retry):
            py, px = dead_pts[random.randrange(len(dead_pts))]
            py = py + random.randint(-jitter, jitter)
            px = px + random.randint(-jitter, jitter)
            y0 = int(np.clip(py - cs // 2, 0, max(0, H - cs)))
            x0 = int(np.clip(px - cs // 2, 0, max(0, W - cs)))
            img_c  = img[y0:y0+cs, x0:x0+cs]
            mask_c = mask[y0:y0+cs, x0:x0+cs]
            if (mask_c == self.DEAD_CLASS).sum() >= self.min_dead_pixels_in_crop:
                return img_c, mask_c

        return None  # all retries failed → caller falls back to random crop

    def _try_sparse_dead_zoom_crop(self, img, mask, sample_idx):
        """Crop a smaller window around sparse Dead and resize to crop_size.

        For images with very few Dead pixels (<sparse_dead_threshold), a standard
        256×256 crop may contain almost no Dead pixels even when centred correctly.
        This zooms in (crops crop_size/zoom_factor window, then upsamples) so that
        the same few Dead pixels fill more of the model input.
        """
        dead_pts = self._dead_idx.get(sample_idx, np.empty((0, 2), dtype=int))
        if len(dead_pts) == 0:
            return None

        H, W = img.shape[:2]
        cs = self.crop_size
        small_cs = max(cs // self.sparse_dead_zoom_factor, 64)
        jitter = small_cs // 4
        min_dead_in_crop = max(1, self.min_dead_pixels_in_crop // 4)

        for _ in range(self.max_dead_crop_retry):
            py, px = dead_pts[random.randrange(len(dead_pts))]
            py = py + random.randint(-jitter, jitter)
            px = px + random.randint(-jitter, jitter)
            y0 = int(np.clip(py - small_cs // 2, 0, max(0, H - small_cs)))
            x0 = int(np.clip(px - small_cs // 2, 0, max(0, W - small_cs)))
            img_c  = img[y0:y0+small_cs, x0:x0+small_cs]
            mask_c = mask[y0:y0+small_cs, x0:x0+small_cs]
            if (mask_c == self.DEAD_CLASS).sum() >= min_dead_in_crop:
                img_up  = np.array(Image.fromarray(img_c).resize((cs, cs), Image.BILINEAR))
                mask_up = np.array(Image.fromarray(mask_c).resize((cs, cs), Image.NEAREST))
                return img_up, mask_up
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img  = np.array(Image.open(s['image']).convert('RGB'))
        mask = np.array(Image.open(s['mask']))

        H, W = img.shape[:2]
        cs = self.crop_size

        img_c = mask_c = None

        # ── Sparse Dead zoom crop (takes priority for sparse Dead images) ─────
        n_dead_total = self._dead_pixel_count.get(i, 0)
        if (self.sparse_dead_zoom_prob > 0
                and 0 < n_dead_total < self.sparse_dead_threshold
                and random.random() < self.sparse_dead_zoom_prob):
            result = self._try_sparse_dead_zoom_crop(img, mask, i)
            if result is not None:
                img_c, mask_c = result

        # ── Dead-center crop (training only, when enabled) ────────────────────
        if (img_c is None
                and self.use_dead_center_crop
                and random.random() < self.dead_center_crop_prob):
            result = self._try_dead_center_crop(img, mask, i)
            if result is not None:
                img_c, mask_c = result

        # ── Fallback: minority oversample or pure random crop ─────────────────
        if img_c is None:
            if (self.augment
                    and self.minority_oversample > 0
                    and random.random() < self.minority_oversample
                    and len(self._minority_idx[i]) > 0):
                pts = self._minority_idx[i]
                py, px = pts[random.randrange(len(pts))]
                y0 = int(np.clip(py - cs // 2, 0, H - cs))
                x0 = int(np.clip(px - cs // 2, 0, W - cs))
            else:
                y0 = random.randint(0, max(0, H - cs))
                x0 = random.randint(0, max(0, W - cs))
            img_c  = img[y0:y0+cs, x0:x0+cs]
            mask_c = mask[y0:y0+cs, x0:x0+cs]

        if self.augment:
            img_c, mask_c = self._augment(img_c, mask_c)
            # Copy-Paste Dead augmentation (applied after standard augment)
            if self.copy_paste_dead_prob > 0 and random.random() < self.copy_paste_dead_prob:
                img_c, mask_c = self._copy_paste_dead(img_c, mask_c)

        img_t = TF.to_tensor(Image.fromarray(img_c))
        img_t = TF.normalize(img_t, IMG_MEAN, IMG_STD)

        mask_np = mask_c.astype(np.int64)
        mask_np[(mask_np >= 250) | (mask_np > NUM_CLASSES - 1)] = -1
        mask_t = torch.from_numpy(mask_np)

        return img_t, mask_t, s['stem']

    def _build_dead_bank(self):
        """Build a bank of Dead-region crops for Copy-Paste augmentation.

        For each training image that contains Dead pixels, extract up to
        max_crops small patches centred on Dead regions. These are stored
        in memory and used during training to synthesise additional Dead
        occurrences in batches that would otherwise lack Dead pixels.
        """
        bank = []
        bank_cs = self.crop_size // 2   # paste-crop size (half of training crop)
        max_crops_per_img = 4
        for i, s in enumerate(self.samples):
            dead_pts = self._dead_idx.get(i, np.empty((0, 2), dtype=int))
            if len(dead_pts) == 0:
                continue
            img  = np.array(Image.open(s['image']).convert('RGB'))
            mask = np.array(Image.open(s['mask']))
            H, W = img.shape[:2]
            added = 0
            # Shuffle dead points to get varied crops
            perm = np.random.permutation(len(dead_pts))
            for idx in perm:
                if added >= max_crops_per_img:
                    break
                py, px = dead_pts[idx]
                y0 = int(np.clip(py - bank_cs // 2, 0, max(0, H - bank_cs)))
                x0 = int(np.clip(px - bank_cs // 2, 0, max(0, W - bank_cs)))
                img_crop  = img[y0:y0+bank_cs, x0:x0+bank_cs].copy()
                mask_crop = mask[y0:y0+bank_cs, x0:x0+bank_cs].copy()
                if (mask_crop == self.DEAD_CLASS).sum() >= 5:
                    bank.append((img_crop, mask_crop))
                    added += 1
        return bank

    def _copy_paste_dead(self, img, mask):
        """Paste a Dead-region crop from the bank onto the current patch.

        Only Dead-labelled pixels are pasted (non-Dead pixels of the source
        crop are transparent). The pasted region is randomly scaled
        (0.5×–1.5×) and flipped to improve diversity.
        """
        if not self._dead_bank:
            return img, mask

        src_img, src_mask = random.choice(self._dead_bank)

        # Random scale
        scale = random.uniform(0.5, 1.5)
        new_h = int(np.clip(src_img.shape[0] * scale, 32, self.crop_size))
        new_w = int(np.clip(src_img.shape[1] * scale, 32, self.crop_size))
        src_img_r  = np.array(Image.fromarray(src_img).resize(
            (new_w, new_h), Image.BILINEAR))
        src_mask_r = np.array(Image.fromarray(src_mask).resize(
            (new_w, new_h), Image.NEAREST))

        # Random flip
        if random.random() > 0.5:
            src_img_r  = src_img_r[:, ::-1].copy()
            src_mask_r = src_mask_r[:, ::-1].copy()

        # Random paste position
        cs = self.crop_size
        max_y = cs - new_h
        max_x = cs - new_w
        if max_y < 0 or max_x < 0:
            return img, mask   # source larger than target — skip

        y0 = random.randint(0, max_y)
        x0 = random.randint(0, max_x)

        dead_in_src = src_mask_r == self.DEAD_CLASS
        if dead_in_src.sum() == 0:
            return img, mask

        img_out  = img.copy()
        mask_out = mask.copy()
        # Only overwrite pixels that were Dead in the source
        img_out[y0:y0+new_h, x0:x0+new_w][dead_in_src]  = src_img_r[dead_in_src]
        mask_out[y0:y0+new_h, x0:x0+new_w][dead_in_src] = self.DEAD_CLASS
        return img_out, mask_out

    def _apply_dead_pixel_dropout(self, mask):
        """Randomly zero out 60-90% of Dead pixels to synthesize sparse Dead patterns."""
        dead_ys, dead_xs = np.where(mask == self.DEAD_CLASS)
        n_dead = len(dead_ys)
        if n_dead == 0:
            return mask
        keep_frac = random.uniform(0.1, 0.4)
        n_keep = max(1, int(n_dead * keep_frac))
        keep_idx = np.random.choice(n_dead, n_keep, replace=False)
        new_mask = mask.copy()
        new_mask[dead_ys, dead_xs] = 0  # zero out all Dead pixels
        new_mask[dead_ys[keep_idx], dead_xs[keep_idx]] = self.DEAD_CLASS  # restore kept
        return new_mask

    def _augment(self, img, mask):
        # Horizontal / vertical flip
        if random.random() > 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        if random.random() > 0.5:
            img, mask = img[::-1].copy(), mask[::-1].copy()

        # 90° rotations
        k = random.randint(0, 3)
        if k:
            img  = np.rot90(img,  k).copy()
            mask = np.rot90(mask, k).copy()

        # Random brightness / contrast / gamma
        img = img.astype(np.float32)
        img *= 1.0 + random.uniform(-0.2, 0.2)          # brightness
        mean = img.mean()
        img  = (img - mean) * (1.0 + random.uniform(-0.3, 0.3)) + mean  # contrast
        gamma = random.uniform(0.8, 1.2)
        img = np.clip(img, 0, 255) / 255.0
        img = img ** gamma * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)

        # Elastic deformation (50% prob)
        if random.random() > 0.5:
            img, mask = elastic_transform(img, mask, alpha=30, sigma=4)

        # Dead pixel dropout — simulate sparse Dead images (only when Dead pixels present)
        if self.dead_pixel_dropout_prob > 0 and random.random() < self.dead_pixel_dropout_prob:
            mask = self._apply_dead_pixel_dropout(mask)

        return img, mask


# ── Temporal patch dataset ────────────────────────────────────────────────────

class TemporalPatchDataset(Dataset):
    """
    Patch dataset for the 2D+T model.

    Returns T=4 crops (one per available time-point) ALIGNED to the same
    spatial crop window so temporal correspondences are preserved.
    Missing time-points are filled with zeros.
    """

    DAY_ORDER = {'D0': 0, 'D6': 1, 'D9': 2, 'D12': 3}
    MAX_T = 4

    def __init__(self, well_groups, well_ids, crop_size=CROP, augment=False,
                 minority_oversample=0.5):
        self.well_ids   = well_ids
        self.well_groups = well_groups
        self.crop_size  = crop_size
        self.augment    = augment
        self.minority_oversample = minority_oversample

        # Build minority index: for each well, union of minority pixels across all days
        self._minority_idx = self._build_minority_index()

    def _build_minority_index(self):
        idx = {}
        for wid in self.well_ids:
            pts = []
            for e in self.well_groups[wid]:
                mask = np.array(Image.open(e['mask']))
                pts.append(np.argwhere((mask == 1) | (mask == 2) | (mask == 3)))
            idx[wid] = np.concatenate(pts) if pts else np.empty((0, 2), dtype=int)
        return idx

    def __len__(self):
        return len(self.well_ids)

    def __getitem__(self, i):
        wid = self.well_ids[i]
        entries = sorted(
            self.well_groups[wid],
            key=lambda e: self.DAY_ORDER.get(e['day'], 99),
        )

        # Load first available image to get spatial dims
        ref = np.array(Image.open(entries[0]['image']).convert('RGB'))
        H, W = ref.shape[:2]
        cs = self.crop_size

        # Choose crop origin
        if (self.augment
                and self.minority_oversample > 0
                and random.random() < self.minority_oversample
                and len(self._minority_idx[wid]) > 0):
            pts = self._minority_idx[wid]
            py, px = pts[random.randrange(len(pts))]
            y0 = int(np.clip(py - cs // 2, 0, H - cs))
            x0 = int(np.clip(px - cs // 2, 0, W - cs))
        else:
            y0 = random.randint(0, max(0, H - cs))
            x0 = random.randint(0, max(0, W - cs))

        # Augmentation params (same for all frames in sequence)
        flip_h = self.augment and random.random() > 0.5
        flip_v = self.augment and random.random() > 0.5
        rot_k  = random.randint(0, 3) if self.augment else 0
        bright = 1.0 + random.uniform(-0.2, 0.2) if self.augment else 1.0
        gamma  = random.uniform(0.8, 1.2)         if self.augment else 1.0

        imgs  = torch.zeros(self.MAX_T, 3, cs, cs)
        masks = torch.full((self.MAX_T, cs, cs), -1, dtype=torch.long)
        valid = torch.zeros(self.MAX_T, dtype=torch.bool)

        for e in entries:
            t = self.DAY_ORDER.get(e['day'])
            if t is None:
                continue
            img_np  = np.array(Image.open(e['image']).convert('RGB'))
            mask_np = np.array(Image.open(e['mask']))

            img_c  = img_np[y0:y0+cs, x0:x0+cs]
            mask_c = mask_np[y0:y0+cs, x0:x0+cs]

            if self.augment:
                if flip_h: img_c, mask_c = img_c[:, ::-1].copy(), mask_c[:, ::-1].copy()
                if flip_v: img_c, mask_c = img_c[::-1].copy(),    mask_c[::-1].copy()
                if rot_k:
                    img_c  = np.rot90(img_c,  rot_k).copy()
                    mask_c = np.rot90(mask_c, rot_k).copy()
                # Colour jitter (per-frame variation allowed)
                img_f = img_c.astype(np.float32) * (bright + random.uniform(-0.05, 0.05))
                mean  = img_f.mean()
                img_f = (img_f - mean) * (1 + random.uniform(-0.15, 0.15)) + mean
                img_f = np.clip(img_f / 255.0, 0, 1) ** gamma * 255.0
                img_c = np.clip(img_f, 0, 255).astype(np.uint8)

            img_t = TF.normalize(TF.to_tensor(Image.fromarray(img_c)), IMG_MEAN, IMG_STD)
            m_np  = mask_c.astype(np.int64)
            m_np[(m_np >= 250) | (m_np > NUM_CLASSES - 1)] = -1

            imgs[t]  = img_t
            masks[t] = torch.from_numpy(m_np)
            valid[t] = True

        return imgs, masks, valid, wid
