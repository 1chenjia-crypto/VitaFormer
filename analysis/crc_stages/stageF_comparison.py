#!/usr/bin/env python3
"""Stage F statistics (stageF_m2_test/ANALYSIS_PLAN.md). Reuses scripts/unified_comparison_20261102.py unchanged;
only GROUPS / CONTRASTS / OUT are replaced. (Its manifest 'created' field is hard-coded 2026-11-02 in the reused code.)"""
import sys
from pathlib import Path

sys.path.insert(0, '3D_OrganoidSeg/scripts')
import unified_comparison_20261102 as U  # noqa: E402

E = Path('3D_OrganoidSeg/vco/runs/arch_exploration_20261104/stageF_m2_test')
U.OUT = E / 'comparison'
G = U.GROUPS
U.GROUPS = {
    'M2-final': {s: E / f'shared_eval/M2_final_seed{s}.json' for s in U.SEEDS},
    'M0-final': {s: E / f'shared_eval/M0_final_seed{s}.json' for s in U.SEEDS},
    'VF-Simplified': G['VF-Simplified'],
    'SegFormer-B2': G['SegFormer-B2'],
    'SegFormer-B2-stable': {s: U.SE / f'segformer_b2_stable_baseline_seed{s}.json' for s in U.SEEDS},
    'nnU-Net': G['nnU-Net'],
}
U.CONTRASTS = [
    ('M2-final - nnU-Net', [(1, 'M2-final'), (-1, 'nnU-Net')]),
    ('M2-final - M0-final', [(1, 'M2-final'), (-1, 'M0-final')]),
    ('M2-final - VF-Simplified', [(1, 'M2-final'), (-1, 'VF-Simplified')]),
    ('M2-final - SegFormer-B2', [(1, 'M2-final'), (-1, 'SegFormer-B2')]),
    ('M2-final - SegFormer-B2-stable', [(1, 'M2-final'), (-1, 'SegFormer-B2-stable')]),
    ('M0-final - VF-Simplified', [(1, 'M0-final'), (-1, 'VF-Simplified')]),
]

if __name__ == '__main__':
    U.main()
