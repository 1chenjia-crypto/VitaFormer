# P0-B 强基线训练协议(2026-10-30 定稿,训练前写定)

在 `audit_final_experiments_20261030/BASELINE_FAIRNESS_PLAN.md` 的基础上,按作者 10-30 的要求逐条补全。凡与该计划不同之处,在第 6 节列出。

## 1. 数据划分与评价程序(与 VitaFormer 完全一致)
- 划分:`OrganoidSegBaseline/VOC2012/ImageSets/Segmentation/{train,val,test}_cond.txt`,481/164/167 张,是 VitaFormer 使用的同一份 condition-aware 划分(split seed 42,物理孔级别,0 重叠)。
  - sha256 记录在各 run 的 manifest 中。
- 测试集只在训练结束后用于一次评估,不参与任何选择。
- **统一评价程序**:新脚本 `baselines_fair_20261030/eval_masks_shared.py`。
  - 从各方法在测试集上的 PNG 硬 mask 逐图计算每类 tp/fp/fn,保存成与 `test_best_miou_uniform.json` 相同的 `confusion_counts` 结构。
  - 汇总公式与 `analyze_multiseed_ablation.metric_values` 相同:像素在全部图像上合并;mIoU 为 4 类(含背景)的平均。
  - 然后接入同一个 3 种子配对层级 bootstrap(种子 + 物理孔)。
- **一致性检查**:用 P0-M 生成的 VitaFormer mask 跑该程序,必须重现各 run 已有的 JSON 指标(差异 < 0.01 pp)。不通过则停止。

## 2. 预训练差异(如实说明,不声称等同)
| 方法 | 编码器 | 预训练数据 | 可训练参数 |
|---|---|---|---|
| VitaFormer | ViT-B/16(BiomedCLIP) | PMC-15M 生物医学图文对(对比学习) | 约 60.6M |
| VitaFormer(ImageNet 消融) | ViT-B/16 | ImageNet | 约 60.6M;已证明与 BiomedCLIP 无显著差异(+0.45 [−0.92, +1.98]) |
| SegFormer-B2 | MiT-B2 | ImageNet-1k 分类(`nvidia/mit-b2`,revision 3bb39e87,sha256 见 `pretrained/WEIGHTS_PROVENANCE.json`) | 约 27.4M |
| nnU-Net v2 | 自配置 2D U-Net | **无预训练**(nnU-Net 标准做法) | 由 plans 决定,训练前记录 |

因为 BiomedCLIP 与 ImageNet 初始化已被证明等效,"ImageNet 预训练的 SegFormer vs VitaFormer"在预训练这一点上大体可比。nnU-Net 没有预训练,这是它的官方设定,论文中说明。

## 3. Dead 类权重与过采样(不平衡处理)
- VitaFormer 使用 Rare-Dead 包(6 项干预)。
- **主比较中,两个基线都用各自方法的标准不平衡处理,不加入 Rare-Dead 包:**
  - SegFormer-B2:加权 CE。权重与现有所有基线相同:1/√频率,归一化后截断到 [0.2, 10],即 bg 0.20、alive 0.59、dead 1.75、bubble 1.63。
  - nnU-Net:官方 Dice + CE,不加类别权重,使用 nnU-Net 自带的前景过采样(oversample_foreground_percent = 0.33)。
- 理由:主问题是"VitaFormer(含其训练策略)与通用方法开箱即用相比如何"。
- 论文中必须说明:**Dead 召回的差异可能部分来自 Rare-Dead 训练策略,而不只是架构。**
- 可选 P1-S(只有当 SegFormer-B2 与 VitaFormer 差距 < 3 pp 时才做,该条件预先设定,与结果是否报告无关):SegFormer-B2 + 相同的 Rare-Dead 包,用来分离架构与训练策略。

## 4. 训练配置与预算

### SegFormer-B2(脚本 `baselines_fair_20261030/train_segformer_b2.py`)
- 模型:transformers `SegformerForSemanticSegmentation`,编码器加载 mit-b2(`torch.load(weights_only=True)`,不执行仓库代码),解码头随机初始化,4 类。
- 数据增强:与现有基线策略相同(随机缩放 0.5–2.0、水平翻转、补零到 512、随机 512 crop),用 PIL/numpy 重新实现(当前环境没有 cv2)。ImageNet 归一化。
- 优化:AdamW,lr 6e-5,解码头 ×10,weight decay 0.01;线性 warmup 1,500 iter,之后 poly(power 1.0);batch 8;40,000 iter;AMP。
- 验证:每 1,000 iter 在 164 张验证图上做 sliding window 512/256 推理,取最佳验证 mIoU 的 checkpoint。
- 测试:最佳 checkpoint,sliding window 512/256,原分辨率,无 TTA,只做一次。
- 种子:**固定 3 个种子:42、3407、2026。全部运行、全部报告,不依赖任何一个种子的结果。**
- GPU:7(另一用户只占 2.6 GB、利用率 0%)。3 个种子先顺序运行;若显存和吞吐允许,改为并发 2 个。估计每个种子 4–6 小时。

### nnU-Net v2(`env_totalseg2`,nnunetv2 2.8.1)
- 数据:Dataset801_CRCOrganoid,`imagesTr` = train + val(645 张),`imagesTs` = test(167 张);RGB 自然图像格式,标签 0–3。
- `nnUNetv2_plan_and_preprocess -d 801 -c 2d --verify_dataset_integrity`,全部使用官方自动配置。
- 自定义 `splits_final.json`:fold 0 为 train 481 / val 164,与 VitaFormer 一致。
- 训练 `nnUNetv2_train 801 2d 0`,默认 1000 epoch,用 `checkpoint_final`(nnU-Net 默认,不按验证集挑选)。
- 预测:`nnUNetv2_predict` 不带 TTA(主)和带 TTA(次)各一次。
- 预算:每个 fold 训练约 15–30 GPU 小时,以预处理后的 plans 和前几个 epoch 的实测为准。

## 5. 多种子计划(避免按首个种子的表现选择性报告)
- SegFormer-B2:固定 3 个种子(见上)。
- **nnU-Net:种子数必须在训练前固定,不能再用"首个种子落后 ≥ 3 pp 就不补"的规则**(原计划中的这条规则已作废,见第 6 节)。两个选项,待作者选择:
  - (a) **3 个种子**(推荐,与 VitaFormer 口径一致):约 3 × 1 天的 GPU,可在 GPU 3 和 5 上并行(P0-M 结束后),墙钟约 1.5–2 天。
  - (b) **1 个种子,并在论文中预先声明是单次运行**:不做 3 种子的配对推断,只做描述性比较。
- 所有已运行的种子结果全部报告。

## 6. 与原计划(BASELINE_FAIRNESS_PLAN)的差异
1. 原计划的 nnU-Net "首种子落后 ≥ 3 pp 则不补种子"规则不符合"避免按首个种子的表现决定"的要求,**作废**,改为第 5 节的预先固定方案(待选择)。
2. SegFormer 数据增强改为 PIL 重新实现,策略不变。
3. 新增统一的 mask 级评价程序及其一致性检查(第 1 节)。

## 7. Author decision and execution details (2026-10-30, before any nnU-Net training)
- **Option (a): nnU-Net fixed at 3 seeds (42/3407/2026).** All three use the same fold 0 (custom `splits_final.json`, 481/164). Different folds are not treated as different seeds. All three are fully reported, whatever the first seed shows.
- Launcher `nnunet_train_seed.py`:
  - sets python/numpy/torch seeds in the main process, then calls the official `run_training('801', '2d', 0)`;
  - each seed writes to its own `nnUNet_results_seed{N}/`.
- **Seeding limitation (upstream behaviour):** nnU-Net v2 starts its augmentation workers with `seeds=None`, so augmentation randomness cannot be fixed. The seed controls network initialisation and the main-process sampling only. The three runs are therefore independent repeats with fixed initialisation seeds; state this in the paper.
- Before launch, check:
  1. the training log shows "Using splits from existing split file" with 481/164;
  2. the plans (patch size, batch size, network) are recorded.
- GPUs: start only after P0-M frees GPUs 3 and 5; re-check other users' processes before launch.
- Evaluation: run `nnUNetv2_predict` on imagesTs, without TTA (primary) and with TTA (secondary). Feed the PNGs to the shared evaluator `eval_masks_shared.py`, then the same 3-seed comparison as VitaFormer and SegFormer-B2.

## 8. nnU-Net automatic configuration (generated 2026-10-30; nothing adjusted by hand)
- `nnUNetPlans.json`, 2d:
  - PlainConvUNet, 9 stages, features 32/64/128/256/512×5, 3×3 kernels;
  - patch size 1024×1280, batch size 2;
  - Z-score normalisation per channel, spacing 1×1 (median image 2044×2048).
- `--verify_dataset_integrity` passed. Preprocessing used only the 645 training and validation images; none of the 167 test images was preprocessed.
- `splits_final.json` installed (1 fold: 481 / 164); all IDs exist in the preprocessed data.
- Budget: 1000 epochs × 250 iterations × batch 2 at 1024×1280. The actual time per epoch will be read from the first epochs of the log. With the current CPU contention (load average about 110–130), expect several minutes per epoch, possibly well above the earlier estimate of about 1 GPU-day per seed. If the measured time puts 3 seeds above 6 GPU-days, report to the author before continuing. No settings will be changed to save time.
- 2026-10-30 17:10: nnU-Net launch automated with `nnunet_autolaunch.sh` (nohup, independent of the terminal session):
  - seed 42 on GPU 3 and seed 3407 on GPU 5, after P0-M exits;
  - seed 2026 on GPU 7, after SegFormer-B2 finishes;
  - each launch requires ≤ 40 GB GPU memory in use. Log: `nnunet/logs/autolaunch.log`.

## 9. Execution log (2026-10-31)
- **SegFormer-B2, all 3 seeds finished.**
  - Seeds 42 and 3407 diverged to NaN loss under fp16 AMP (validation mIoU fell to 24.87 = all background, from iteration 21,000 and 29,000). Seed 2026 did not diverge.
  - Per protocol, test used the best-validation checkpoint (iterations 18k, 26k, 32k), all before divergence.
  - Test mIoU: 83.60 / 82.32 / 82.83. The shared evaluator (`shared_eval/`) reproduces the internal numbers exactly.
  - The divergence is a training defect of these two runs and is reported as such. No rerun without the author's decision.
- **nnU-Net first launch (00:05) crashed at epoch 0, all three seeds:**
  - seeds 42 and 3407: "can't start new thread". The launcher bypassed the official entry point's OMP/MKL = 1 thread settings, so the 4,096 per-user thread limit was reached.
  - seed 2026: `torch.compile` failed, because gcc 4.8.5 lacks `stdatomic.h`.
  - The crashed outputs (`nnUNet_results_seed*/`, `logs/train_seed*.full.log`) are kept.
- **nnU-Net run 2 (09:22) launched on GPUs 3, 5 and 7:**
  - the launcher now mirrors the official entry thread settings;
  - `nnUNet_compile=f` (speed only);
  - outputs in `nnUNet_results_run2_seed*/`, logs in `logs/train_run2_seed*.full.log`;
  - all three logs confirm "481 training and 164 validation cases".
  - No plans, trainer or epoch settings were changed.
- **nnU-Net run 2, measured speed** (from `fold_0/training_log_*.txt`, since nnU-Net does not print epochs to stdout): about 27 s per epoch on each of GPUs 3/5/7 in parallel. That is about 7.5 h per seed, about 0.95 GPU-days for 3 seeds, well under the 6 GPU-day report threshold. Expected finish about 17:00 on 2026-10-31.
- 2026-10-31 15:33: `nnunet_post_train.sh` launched (nohup), one per seed. After the seed's training exits with `checkpoint_final.pth`, it predicts imagesTs on the same GPU into `nnunet/pred_run2_seed{N}_{noTTA,TTA}/` and runs `eval_masks_shared.py` (output in `shared_eval/`). Log: `nnunet/logs/post_train.log`.
