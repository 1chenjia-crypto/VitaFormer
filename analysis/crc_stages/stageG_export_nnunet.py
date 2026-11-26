#!/usr/bin/env python3
"""Stage G (AMENDMENT_1_20261106.md §4): nnU-Net run2 checkpoint_final, NO TTA, validation probability export, one seed.
Run with env_totalseg2 python. Usage: stageG_export_nnunet.py SEED
Settings = nnUNetv2_predict defaults used for the primary test predictions (nnunet_post_train.sh: -c 2d -f 0
-chk checkpoint_final.pth --disable_tta): tile_step_size 0.5, gaussian on, mirroring off. Image reader from plans.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
EXP = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104')
NN = Path('3D_OrganoidSeg/baselines_fair_20261030/nnunet')
sys.path.insert(0, str(EXP / 'scripts'))
import stageG_common as C  # noqa
os.environ.setdefault('nnUNet_raw', str(NN / 'nnUNet_raw')); os.environ.setdefault('nnUNet_preprocessed', str(NN / 'nnUNet_preprocessed'))
import torch  # noqa
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa
from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO  # noqa
VAL = Path('OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/val_cond.txt')
TEST = Path('OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/test_cond.txt')
RAW = NN / 'nnUNet_raw/Dataset801_CRCOrganoid'
OUT = EXP / 'stageG_dead_pr_val/export'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    s = int(sys.argv[1])
    mdir = NN / f'nnUNet_results_run2_seed{s}/Dataset801_CRCOrganoid/nnUNetTrainer__nnUNetPlans__2d'
    plans = json.loads((mdir / 'plans.json').read_text()); dj = json.loads((mdir / 'dataset.json').read_text())
    assert plans['image_reader_writer'] == 'NaturalImage2DIO'
    assert dj['labels'] == {'background': 0, 'alive': 1, 'dead': 2, 'bubble': 3}, dj['labels']
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, perform_everything_on_device=True,
                           device=torch.device('cuda'), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(mdir), use_folds=(0,), checkpoint_name='checkpoint_final.pth')
    io = NaturalImage2DIO()

    def run(path):
        img, props = io.read_images([str(path)])
        seg, prob = pred.predict_single_npy_array(img, {'spacing': props['spacing']}, None, None, True)
        return np.asarray(seg)[0].astype(np.uint8), np.asarray(prob)[:, 0].astype(np.float32)

    val = [x.strip() for x in VAL.read_text().splitlines() if x.strip()]
    test = [x.strip() for x in TEST.read_text().splitlines() if x.strip()]
    d = OUT / f'nnUNet_seed{s}'; (d / 'level').mkdir(parents=True, exist_ok=False); (d / 'other').mkdir()
    checks = {}
    # same-configuration reproduction: 2 test images vs existing primary noTTA masks
    checks['test2_vs_pred_noTTA_px_diff'] = [int((run(RAW / f'imagesTs/{t}_0000.png')[0] !=
                                                  np.asarray(Image.open(NN / f'pred_run2_seed{s}_noTTA/{t}.png'))).sum()) for t in test[:2]]
    # repeatability: 2 val images twice
    rep = []
    for v in val[:2]:
        a, pa = run(RAW / f'imagesTr/{v}_0000.png'); b, pb = run(RAW / f'imagesTr/{v}_0000.png')
        rep.append({'seg_px_diff': int((a != b).sum()), 'prob_max_absdiff': float(np.abs(pa - pb).max())})
    checks['repeat_val2'] = rep
    oldtta = NN / f'nnUNet_results_run2_seed{s}/Dataset801_CRCOrganoid/nnUNetTrainer__nnUNetPlans__2d/fold_0/validation'
    stats = {'H': [], 'HF': [], 'HP': []}
    mism = {'seg_vs_probargmax_px': 0, 'seg_vs_delta0_px': 0, 'oldTTA_vs_delta0_px': 0, 'n_px': 0}
    t0 = time.time()
    for i, v in enumerate(val, 1):
        seg, prob = run(RAW / f'imagesTr/{v}_0000.png')
        gt = C.load_gt(v); assert seg.shape == gt.shape == prob.shape[1:], (v, seg.shape, gt.shape, prob.shape)
        sc, oth = C.score_from_probs(prob); d0 = C.decide(sc, oth, 0.0)
        mism['seg_vs_probargmax_px'] += int((prob.argmax(0) != seg).sum()); mism['seg_vs_delta0_px'] += int((d0 != seg).sum())
        mism['oldTTA_vs_delta0_px'] += int((d0 != np.asarray(Image.open(oldtta / f'{v}.png'))).sum()); mism['n_px'] += d0.size
        res, level = C.summarize(sc, oth, prob[2], gt)
        for k in stats: stats[k].append(res[k])
        Image.fromarray(level).save(d / 'level' / f'{v}.png'); Image.fromarray(oth).save(d / 'other' / f'{v}.png')
        if i % 40 == 0: print(f'nnU-Net s{s}: {i}/{len(val)} {time.time()-t0:.0f}s {mism}', flush=True)
    np.savez_compressed(d / 'stats.npz', stems=np.array(val), **{k: np.stack(v) for k, v in stats.items()})
    checks['val'] = mism
    man = {'arm': 'nnUNet', 'seed': s, 'checkpoint': str(mdir / 'fold_0/checkpoint_final.pth'),
           'checkpoint_sha256': sha(mdir / 'fold_0/checkpoint_final.pth'), 'plans_sha256': sha(mdir / 'plans.json'),
           'settings': 'tile_step 0.5, gaussian, no mirroring', 'checks': checks, 'seconds': round(time.time() - t0)}
    (d / 'manifest.json').write_text(json.dumps(man, indent=2)); print('done', json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
