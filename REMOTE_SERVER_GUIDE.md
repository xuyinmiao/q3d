# Remote Server Guide

This project keeps source code on the system disk and training data/results on the data disk.

## Directory Layout

Use this layout on the remote server:

```text
~/q3d/                                  # source code, system disk
/data/q3d/datasets/gtsrb/               # GTSRB dataset, data disk
/data/q3d/outputs/model_history_gtsrb/  # checkpoints and experiment results, data disk
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
mkdir -p /data/q3d/outputs/model_history_gtsrb
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
  --model classical \
  --epochs 1 \
  --batch-size 32 \
  --widen-factor 1 \
  --train-samples 512 \
  --val-samples 128 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb
```

## Full Training

Train the classical baseline:

```bash
python run_py/train_gtsrb.py \
  --model classical \
  --epochs 50 \
  --batch-size 128 \
  --widen-factor 4 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb
```

Train the hybrid quantum-classical model:

```bash
python run_py/train_gtsrb.py \
  --model hybrid \
  --epochs 50 \
  --batch-size 64 \
  --widen-factor 4 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb
```

## Adversarial Evaluation

```bash
python run_py/attack_eval_gtsrb.py \
  --model both \
  --attacks fgsm,pgd,cw \
  --test-samples 2000 \
  --batch-size 64 \
  --pgd-steps 20 \
  --cw-steps 50 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb \
  --output-dir /data/q3d/outputs/model_history_gtsrb
```

Outputs are written to `/data/q3d/outputs/model_history_gtsrb/`.
