# BASELINE_FAIRNESS_PLAN — SegFormer 审计与 nnU-Net v2 公平基线方案(2026-10-30,待批准,未启动任何训练)

## 1. 现有基线审计结论
数据来源:`OrganoidSegBaseline/scripts/train_*_baseline.py`、`run_*_nohup.sh`、`runs/*/run_meta.json`、`val_iter_*.json`。

| 项目 | U-Net / U-Net++ / DeepLabv3 | SegFormer(现有) | OrganoidViT(复现) | VitaFormer |
|---|---|---|---|---|
| 骨干 | 手写(DeepLabv3 为手写 ResNet) | 手写 MiT-B0,约 3.7M 参数 | 官方仓库 MiT-B0 | BiomedCLIP ViT-B/16(可训练 60.6M) |
| **预训练** | **无,从头训练** | **无,从头训练** | ImageNet `mit_b0.pth` | BiomedCLIP |
| 输入 | 512 crop,随机缩放 0.5–2,水平翻转;推理整图 | 同左 | mmseg 配置 | 224 crop + Rare-Dead 采样;推理 sliding 224/112 |
| 归一化 | `normalize_to_tensor`(ImageNet 均值方差) | 同左 | mmseg | ImageNet |
| 类别映射 | bg / live / death / bubble | 同左 | 同左 | 同左 ✅ |
| 损失和不平衡处理 | 加权 CE(Dead 权重约 1.75) | 同左 | mmseg 默认 | Focal + Tversky,Dead 权重 10,6 项 Rare-Dead 干预 |
| 优化 | AdamW 2e-4,poly,batch 4 | AdamW 2e-4,poly,**batch 2** | mmseg | AdamW 1e-4 / 1e-5,batch 16 |
| 步数 | 40k | 40k | 40k(取 iter 32000) | 40k |
| ckpt 选择 | 最佳验证 mIoU | 同左 | 最佳验证 | 最佳验证 mIoU ✅ |
| 测试指标 | 同一 167 张图,`evaluate_test_full_metrics` | 同左 | 同左 | 自有评估;混淆计数口径相同 |

### SegFormer 低性能的原因(判断)
1. **没有 ImageNet 预训练。** SegFormer 原论文和所有常规用法都依赖预训练的 MiT 编码器;从头训练的 Transformer 在 481 张图上不可能发挥。
2. **batch 2 加上 poly lr,验证 mIoU 在 0.33–0.53 之间来回震荡,从未收敛。**
3. **Dead 类处理极弱**(权重 1.75,对比 VitaFormer 的 10 加过采样),所以 Dead IoU 只有 9.5。

这是实验设置问题,不是实现 bug。类别映射、划分和评估都正确。但在论文里把它称为 "SegFormer" 会误导读者:它代表的是"从头训练的 MiT-B0"。

### 其他基线
- U-Net、U-Net++、DeepLabv3 同样从头训练,并出现后期崩溃。
- 它们可以保留,但**表中必须注明 "trained from scratch"**。
- 正文的主要对照应换成下文的 nnU-Net 和预训练 SegFormer。

## 2. 方案 S:预训练 SegFormer(MiT-B2,ImageNet-1k)
- 目的:回答"一个标准的、预训练的通用 Transformer 分割器能做到多少"。
- 实现:HuggingFace `transformers` 的 `SegformerForSemanticSegmentation`,权重来自 `nvidia/mit-b2`。本地未缓存,需要联网下载约 100 MB。
- 训练协议:SegFormer 官方 ADE20K 配方,适配本数据。
  - 512 crop,随机缩放 0.5–2.0,翻转;
  - AdamW 6e-5,head 学习率 ×10,poly 衰减,warmup 1,500 iter;
  - batch 8,40k iter;
  - 加权 CE 使用与现有基线相同的权重。
  - **不加入** Rare-Dead 包。理由:我们比较的是"通用方法开箱即用"。另报一项可选变体 S+,加入相同的 Rare-Dead 采样和损失,用来分离"架构"和"训练策略"的贡献;S+ 列为 P2。
- 推理:sliding window 512/256,原分辨率(与训练尺度一致;现有整图推理与训练尺度不匹配)。
- ckpt 选择:每 1k iter 在验证集上评估,取最佳 mIoU。测试集只在最后评估一次。
- 指标:同一个 `evaluate_test_full_metrics` 程序,同时导出逐图混淆计数,以接入 3 种子统计框架。
- 种子:3 个(42/3407/2026),与 VitaFormer 的统计口径一致。
- 资源:A100 上每个 run 约 6–8 GPU 小时,显存约 20 GB;3 个种子共约 20–24 GPU 小时。

## 3. 方案 N:nnU-Net v2 2D
- 目的:生物医学分割的事实标准强基线,自动配置,无预训练。
- 环境:`env_totalseg2`(nnunetv2 2.8.1,torch 2.5.1)。新建独立目录 `nnunet_baseline_<date>/`,包含 `nnUNet_raw`、`nnUNet_preprocessed`、`nnUNet_results`。
- 数据转换:
  - Dataset ID 例如 `Dataset801_CRCOrganoid`;
  - RGB PNG 按 nnU-Net 自然图像格式存放(3 个通道,spacing 1×1);
  - 标签 0–3;
  - imagesTr 为 train + val 共 645 张,imagesTs 为 test 167 张。
  - 测试集不参与预处理统计(nnU-Net 指纹只在 imagesTr 上计算)。
- 划分:写入自定义 `splits_final.json`,与 VitaFormer 的样本归属完全一致。
  - **fold 0:train = 481,val = 164**,作为主比较,对应 VitaFormer 的单模型。
  - fold 1–4(可选,P2):在 645 张中按物理孔分组生成 4 个附加折,只用于 5 折集成。
- 训练:
  - `nnUNetv2_plan_and_preprocess -c 2d --verify_dataset_integrity`,使用官方自配置,不手动调 patch 或 batch;
  - `nnUNetv2_train 801 2d 0`,默认 1000 epoch × 250 iter;
  - 记录实际 plans(patch size、batch、网络深度);
  - 不为了对齐 VitaFormer 的 40k step 而缩短训练。
- 模型选择:nnU-Net 默认用 `checkpoint_final.pth`,验证集只做监控。这与 VitaFormer 的"最佳验证 mIoU"选择不同,在论文中如实注明。不用测试集选择任何东西。
- 推理:`nnUNetv2_predict`,默认镜像 TTA。另报一次不带 TTA 的结果,因为 VitaFormer 也没有 TTA。主比较用不带 TTA 的结果,两者都报告。
- 指标:把预测 PNG 交给同一个评估程序(混淆计数 → mIoU、Dead P/R/IoU、面积误差、边界指标)。
- 比较口径:

| 口径 | nnU-Net | VitaFormer |
|---|---|---|
| 主 | fold 0 单模型,3 个种子(若资源允许;否则 1 个种子并注明) | Simplified 3 种子 |
| 次 | 5 折集成 | 不作同口径比较,注明集成优势 |

预算不对等(nnU-Net 1000 epoch,VitaFormer 40k step)在论文中明说,**不称为等预算比较**。
- 资源估计(A100,2D,2048² 图):
  - 预处理约 1–2 小时 CPU,磁盘约 30–60 GB;
  - 每个 fold 训练约 15–30 GPU 小时(视 plans 而定,以前 5 个 epoch 的实测再更新);
  - 推理 167 张约 0.5–1 GPU 小时;
  - **主方案 fold 0 × 1 种子约 1 天;3 种子约 3 GPU 天,可在 3 张卡上并行,约 1 天**;
  - 5 折集成再加约 4 GPU 天。

## 4. 建议的最小方案(请批准)
1. N-main:nnU-Net fold 0,**1 个种子**,不带 TTA 和带 TTA 各报一次。
2. S-main:预训练 SegFormer-B2,**3 个种子**。
3. 停止规则(预先设定):
   - 若 N-main 的 mIoU 比 VitaFormer 3 种子均值低 ≥ 3 pp,不再补 nnU-Net 种子,单种子并注明即可。
   - 若差距 < 3 pp,或 nnU-Net 更好,则补到 3 个种子,用配对层级 bootstrap 正式比较。
4. 无论结果如何:
   - 全部写入主表;
   - 若 nnU-Net 优于 VitaFormer,论文的分割优越性主张随之调整为"可比"或"不如",但功能验证(DeepOrganoid、患者系排序)的价值不变。

合计约 1.5–2.5 GPU 天,可并行。需要你确认:
- 是否批准方案 N-main 和 S-main;
- 可用的 GPU 编号;
- 是否允许联网下载 `nvidia/mit-b2` 权重。
