# q3d - Quantum-Augmented Traffic Sign Robustness

## 项目简介

本项目当前主线是 **GTSRB 交通标志分类** 场景下的量子经典混合模型鲁棒性评估。

需要注意：当前任务是 **traffic sign classification**，不是 detection。GTSRB 数据集提供的是裁剪后的交通标志图像和类别标签，当前代码不包含 bounding box、YOLO/Faster R-CNN、mAP 或 IoU 评估。

核心实验目标：

- 数据集：GTSRB, 43 类交通标志。
- 强经典基线：`classical_strong`，标准 WideResNet 分类头。
- 兼容旧结果基线：`classical_legacy`，8 维瓶颈 WideResNet，仅用于解释旧实验。
- 量子经典混合模型：`hybrid_quantum`，当前 HybridQWideResNet。
- 量子层消融：`hybrid_noquantum`、`hybrid_mlp`。
- 攻击方法：FGSM、PGD、C&W。
- 研究问题：量子层是否带来超出强经典基线和经典替代层的对抗鲁棒性差异。

原始 MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100 脚本仍保留在 `run_py/` 中，作为 legacy/reference 代码；当前远程服务器运行主线以 GTSRB 为准。

## 项目结构

GitHub 仓库只保存代码和文档，不保存数据集、模型权重和训练输出。

```text
q3d/
├── run_py/
│   ├── network.py                    # CNN、HybridQCNN、WideResNet、HybridQWideResNet
│   ├── attack.py                     # FGSM、BIM、MIM、PGD、C&W 攻击实现
│   ├── gtsrb_common.py               # GTSRB 数据加载、Normalize、设备选择
│   ├── train_gtsrb.py                # GTSRB 经典/量子混合模型训练入口
│   ├── attack_eval_gtsrb.py          # GTSRB clean/FGSM/PGD/C&W 评估入口
│   ├── run_gtsrb_experiments.py      # 多 seed 训练、评估、汇总编排入口
│   ├── summarize_gtsrb_results.py    # 多 seed 结果 mean/std 汇总
│   │
│   ├── quantum_train_*.py            # legacy: 原四数据集量子模型训练脚本
│   ├── classical_train_*.py          # legacy: 原四数据集经典模型训练脚本
│   ├── attack_eval_*_pb.py           # legacy: 原四数据集扰动预算评估脚本
│   ├── attack_eval_*_it.py           # legacy: 原四数据集迭代次数评估脚本
│   ├── circuit_visualization.py      # 量子电路可视化工具
│   └── convert_results_to_excel.py   # legacy: 旧结果文本转 Excel
│
├── requirements.txt
├── GTSRB_EXPERIMENT_GUIDE.md         # GTSRB 实验代码说明
├── REMOTE_SERVER_GUIDE.md            # 远程服务器运行说明
├── OPENCODE_REMOTE_RUNBOOK.md        # 给 opencode 使用的远程运行手册
└── traffic_sign_qaml_research_plan.md
```

远程服务器建议使用系统盘放代码、数据盘放数据和输出：

```text
~/q3d/                                  # GitHub 项目代码，系统盘
/data/q3d/datasets/gtsrb/               # GTSRB 数据集，数据盘
/data/q3d/outputs/gtsrb_paper/          # paper-grade 多 seed 输出，数据盘
/data/q3d/outputs/model_history_gtsrb/  # 单次调试输出，数据盘
```

`.gitignore` 已忽略：

```text
data_*/
model_history_*/
*.pth
*.pt
```

因此数据集和模型权重不会被提交到 GitHub。

## 环境依赖

核心依赖见 `requirements.txt`：

- Python 3.8+
- PyTorch
- torchvision
- PennyLane
- pennylane-lightning
- numpy
- scipy
- matplotlib
- pandas
- openpyxl
- rich
- tqdm

远程服务器建议使用 CUDA GPU。量子混合模型包含 PennyLane `default.qubit` 模拟，训练和攻击会明显慢于纯经典模型。

## 远程服务器快速开始

```bash
cd ~
git clone https://github.com/xuyinmiao/q3d.git
cd q3d

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p /data/q3d/datasets/gtsrb
mkdir -p /data/q3d/outputs/gtsrb_paper
```

检查 CUDA 和依赖：

```bash
python - <<'PY'
import torch
import torchvision
import pennylane as qml
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("pennylane", qml.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

如果 `cuda_available` 是 `False`，不要直接开始完整训练，先修复 CUDA/PyTorch 环境。

## 数据集位置

GTSRB 数据集放在数据盘：

```text
/data/q3d/datasets/gtsrb/
```

脚本运行时使用：

```bash
--data-dir /data/q3d/datasets/gtsrb
```

如果服务器能联网，`torchvision.datasets.GTSRB` 会在首次运行时自动下载到该目录。如果服务器不能联网，需要手动把 GTSRB 数据上传到该目录。

不要把数据集放进 `~/q3d/` 仓库目录。

## Smoke Test

先跑一个 1 epoch 小样本测试，确认数据、依赖、CUDA、输出路径都正常：

```bash
python run_py/train_gtsrb.py \
  --model classical_strong \
  --epochs 1 \
  --batch-size 32 \
  --widen-factor 1 \
  --train-samples 512 \
  --val-samples 128 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/gtsrb_paper/smoke
```

预期产生：

```text
/data/q3d/outputs/gtsrb_paper/smoke/best_classical_strong_gtsrb.pth
/data/q3d/outputs/gtsrb_paper/smoke/classical_strong_gtsrb_history.json
```

## 模型命名

新实验使用以下模型名：

- `classical_strong`：标准 WideResNet 强经典基线，论文主 baseline。
- `hybrid_quantum`：当前量子经典混合模型，论文主模型。
- `hybrid_noquantum`：移除量子层的结构消融。
- `hybrid_mlp`：用经典 MLP 替代量子层的结构消融。
- `classical_legacy`：旧 8 维瓶颈 WideResNet，仅用于复现旧结果。

兼容别名：

- `classical` 等同于 `classical_legacy`。
- `hybrid` 等同于 `hybrid_quantum`。
- `paper_core` 等同于 `classical_strong,hybrid_quantum`。
- `ablation` 等同于 `hybrid_quantum,hybrid_noquantum,hybrid_mlp`。

## Paper-grade 一键实验

推荐远程服务器使用一键编排脚本完成 3 个 seed、强 baseline、量子层消融和对抗评估：

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

输出目录：

```text
/data/q3d/outputs/gtsrb_paper/seed_42/
/data/q3d/outputs/gtsrb_paper/seed_123/
/data/q3d/outputs/gtsrb_paper/seed_2024/
/data/q3d/outputs/gtsrb_paper/summary_clean.csv
/data/q3d/outputs/gtsrb_paper/summary_robustness.csv
/data/q3d/outputs/gtsrb_paper/summary_ablation.csv
```

如果训练已经完成，只想重新汇总：

```bash
python run_py/summarize_gtsrb_results.py \
  --results-dir /data/q3d/outputs/gtsrb_paper \
  --output-dir /data/q3d/outputs/gtsrb_paper
```

## 单模型调试训练

训练强经典 WideResNet：

```bash
python run_py/train_gtsrb.py \
  --model classical_strong \
  --epochs 80 \
  --batch-size 64 \
  --widen-factor 4 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb
```

训练量子经典混合 HybridQWideResNet：

```bash
python run_py/train_gtsrb.py \
  --model hybrid_quantum \
  --epochs 80 \
  --batch-size 64 \
  --widen-factor 4 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb
```

如果 hybrid 在 CUDA 上因为 PennyLane 量子层兼容性或速度问题失败，可先用 CPU 验证链路：

```bash
python run_py/train_gtsrb.py \
  --model hybrid_quantum \
  --epochs 1 \
  --batch-size 16 \
  --widen-factor 1 \
  --train-samples 256 \
  --val-samples 64 \
  --device cpu \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb
```

## 单次对抗鲁棒性评估

完成对应模型训练后运行：

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
  --save-dir /data/q3d/outputs/model_history_gtsrb \
  --output-dir /data/q3d/outputs/model_history_gtsrb
```

输出文件：

```text
/data/q3d/outputs/model_history_gtsrb/gtsrb_robustness_results.json
/data/q3d/outputs/model_history_gtsrb/gtsrb_robustness_results.txt
/data/q3d/outputs/model_history_gtsrb/gtsrb_robustness_results.png
```

## 为什么 GTSRB 攻击脚本单独写

GTSRB 训练时模型输入是 normalized tensor。若直接在 normalized tensor 上攻击，`epsilon=8/255` 不再表示像素空间扰动。

当前 `attack_eval_gtsrb.py` 使用：

1. DataLoader 输出像素空间图像，范围 `[0, 1]`。
2. 在像素空间生成 FGSM/PGD/C&W 对抗样本。
3. 用 `NormalizeWrapper` 在模型前自动 normalize。
4. 梯度仍可从模型反传到像素空间输入。

这样 `epsilon=1/255, 2/255, 4/255, 8/255` 的含义更清楚。

## Legacy 脚本说明

仓库中仍保留原始四数据集脚本：

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100

这些脚本用于参考原项目实现，不是当前 GTSRB 自动驾驶交通标志实验的主入口。

如果继续使用旧 CIFAR 评估脚本，需要注意：旧脚本对 normalized 彩色图像的攻击边界处理不如 GTSRB 新脚本严谨。当前研究结论应以 GTSRB 新入口为准。

## 更多说明

- `GTSRB_EXPERIMENT_GUIDE.md`：GTSRB 实验代码说明。
- `REMOTE_SERVER_GUIDE.md`：远程服务器目录和命令。
- `OPENCODE_REMOTE_RUNBOOK.md`：给远程服务器 opencode 使用的执行手册。
