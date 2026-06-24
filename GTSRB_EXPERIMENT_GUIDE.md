# GTSRB 交通标志鲁棒性实验代码说明

> 当前论文级实验建议优先使用 `run_py/run_gtsrb_experiments.py`。该入口会自动运行 `classical_strong`、`hybrid_quantum`、`hybrid_noquantum`、`hybrid_mlp` 的 3-seed 训练、FGSM/PGD/C&W 评估和 mean/std 汇总。旧的 `classical` 别名对应 8 维瓶颈 baseline，只用于兼容旧结果，不建议作为论文主 baseline。

## 1. 改动概览

本次实验不直接重写原有 MNIST/CIFAR 脚本，而是在保留原代码的基础上新增 GTSRB 实验入口。

新增文件：

- `run_py/gtsrb_common.py`
  - GTSRB 数据加载
  - 数据增强与 Normalize
  - 设备选择，支持 `cuda/mps/cpu`
  - `NormalizeWrapper`，用于在像素空间攻击时自动归一化输入

- `run_py/train_gtsrb.py`
  - 训练 `WideResNet` 或 `HybridQWideResNet`
  - 输出 43 类 GTSRB checkpoint
  - 保存训练 history 和曲线

- `run_py/attack_eval_gtsrb.py`
  - 加载 GTSRB checkpoint
  - 评估 clean accuracy
  - 评估 FGSM、PGD、C&W
  - 输出 JSON、TXT、PNG 结果

- `run_py/run_gtsrb_experiments.py`
  - 编排多 seed 训练、攻击评估和结果汇总

- `run_py/summarize_gtsrb_results.py`
  - 汇总 clean accuracy、robust accuracy、ASR 的 mean/std

修改文件：

- `run_py/attack.py`
  - 原攻击函数默认仍使用 `[0, 1]` 边界，旧脚本可继续运行
  - 新增 `clip_min/clip_max` 参数，支持归一化输入空间
  - GTSRB 评估脚本默认在像素空间生成攻击，因此 `epsilon=8/255` 含义明确

## 2. 环境准备

当前项目依赖见 `requirements.txt`。如果是新环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

MacBook Air M5 可以先做小样本调试。完整训练和 C&W/EOT 类攻击建议使用 CUDA GPU。

## 3. 快速小样本调试

先用少量样本确认代码链路可跑：

```bash
python run_py/train_gtsrb.py \
  --model classical_strong \
  --epochs 1 \
  --batch-size 32 \
  --widen-factor 1 \
  --train-samples 512 \
  --val-samples 128
```

量子混合模型小样本调试：

```bash
python run_py/train_gtsrb.py \
  --model hybrid_quantum \
  --epochs 1 \
  --batch-size 16 \
  --widen-factor 1 \
  --train-samples 256 \
  --val-samples 64 \
  --device cpu
```

注意：`HybridQWideResNet` 使用 PennyLane `default.qubit`，在 Mac 上默认建议先用 CPU。

## 4. 正式多 seed 实验

论文级实验使用一键 runner：

```bash
python run_py/run_gtsrb_experiments.py \
  --seeds 42,123,2024 \
  --models classical_strong,hybrid_quantum,hybrid_noquantum,hybrid_mlp \
  --epochs 80 \
  --batch-size 64 \
  --eval-batch-size 64 \
  --optimizer sgd \
  --lr 0.05 \
  --widen-factor 4 \
  --attacks fgsm,pgd,cw \
  --epsilons 0,0.0039215686,0.0078431373,0.0156862745,0.031372549,0.062745098 \
  --test-samples 2000 \
  --pgd-steps 20 \
  --cw-steps 50 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --output-dir /data/q3d/outputs/gtsrb_paper
```

输出结构：

```text
/data/q3d/outputs/gtsrb_paper/
├── seed_42/
├── seed_123/
├── seed_2024/
├── summary_clean.csv
├── summary_robustness.csv
└── summary_ablation.csv
```

## 5. 单次数字攻击评估

如果只想对某个 seed 目录重新评估：

```bash
python run_py/attack_eval_gtsrb.py \
  --model classical_strong,hybrid_quantum,hybrid_noquantum,hybrid_mlp \
  --attacks fgsm,pgd,cw \
  --epsilons 0,0.0039215686,0.0078431373,0.0156862745,0.031372549,0.062745098 \
  --test-samples 2000 \
  --batch-size 64 \
  --pgd-steps 20 \
  --cw-steps 50 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/gtsrb_paper/seed_42 \
  --output-dir /data/q3d/outputs/gtsrb_paper/seed_42
```

## 6. 为什么攻击评估这样写

训练时模型输入是 normalized tensor：

```python
transforms.Normalize(GTSRB_MEAN, GTSRB_STD)
```

如果直接在 normalized tensor 上做攻击，`epsilon=8/255` 不再是像素空间扰动，而且 clamp 到 `[0,1]` 是错误的。

因此 GTSRB 评估脚本采用：

1. DataLoader 输出像素空间图像，范围 `[0,1]`
2. 对像素空间图像生成 FGSM/PGD/C&W 对抗样本
3. 用 `NormalizeWrapper` 在模型前自动做 normalize
4. 梯度仍然可以从模型反传到像素空间输入

这使得 `epsilon=8/255` 的实验含义和自动驾驶视觉攻击论文中的常见设置一致。

## 7. 下一步扩展

当前代码覆盖最小可行实验：

- Clean accuracy
- FGSM
- PGD
- C&W

下一步建议新增：

- `run_py/patch_eval_gtsrb.py`
- universal adversarial patch
- patch size sweep
- EOT 变换，包括旋转、缩放、亮度、模糊和平移
- 类别级 confusion matrix

这些扩展应继续使用像素空间输入和 `NormalizeWrapper`，避免归一化边界错误。
