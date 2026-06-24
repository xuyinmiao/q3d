# Remote Server Guide

This project keeps source code on the system disk and training data/results on the data disk.

Current paper-grade workflow uses `run_py/run_gtsrb_experiments.py` to run `classical_strong`, `hybrid_quantum`, `hybrid_noquantum`, and `hybrid_mlp` across seeds `42,123,2024`, then writes mean/std summaries. The old `classical` alias is a legacy 8-d bottleneck baseline and should not be used as the main paper baseline.

## Directory Layout

Use this layout on the remote server:

```text
~/q3d/                                  # source code, system disk
/data/q3d/datasets/gtsrb/               # GTSRB dataset, data disk
/data/q3d/outputs/gtsrb_paper/          # paper-grade multi-seed outputs, data disk
/data/q3d/outputs/model_history_gtsrb/  # optional one-off debugging outputs
```

If the server uses a different data mount, replace `/data/q3d/...` in the commands below.

## Setup

```bash
cd ~
git clone https://github.com/xuyinmiao/q3d.git
cd q3d

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p /data/q3d/datasets/gtsrb
mkdir -p /data/q3d/outputs/gtsrb_paper
```

## Data

The GTSRB scripts use `--data-dir /data/q3d/datasets/gtsrb`.

If the server has internet access, torchvision can download GTSRB automatically on the first run.
If the server cannot access the dataset source, upload the GTSRB files to:

```text
/data/q3d/datasets/gtsrb/
```

Do not place datasets inside the Git repository.

## Smoke Test

Run a small classical training job first:

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

## Paper-grade Run

Run the full multi-seed experiment:

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

## Optional One-off Evaluation

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

Use this only when checkpoints already exist in the selected seed directory. The paper-grade runner already performs this step automatically.
