# Opencode Remote Runbook

Use this file as the instruction source for an opencode session on the remote server. The goal is to keep source code on the system disk, put datasets and experiment outputs on the data disk, then start GTSRB training in a reproducible way.

## Fixed Paths

Use these paths unless the server administrator confirms a different data mount:

```text
~/q3d/                                  # source code on system disk
/data/q3d/datasets/gtsrb/               # GTSRB dataset on data disk
/data/q3d/outputs/model_history_gtsrb/  # checkpoints, logs, plots, results on data disk
```

Do not put datasets, checkpoints, or generated experiment results inside `~/q3d`.

## Opencode Instruction

Give opencode this task on the remote server:

```text
Clone or update https://github.com/xuyinmiao/q3d.git into ~/q3d.
Create a Python virtual environment in ~/q3d/.venv.
Install requirements.txt.
Create /data/q3d/datasets/gtsrb and /data/q3d/outputs/model_history_gtsrb.
Use /data/q3d/datasets/gtsrb as --data-dir and /data/q3d/outputs/model_history_gtsrb as --save-dir.
Run a 1-epoch classical GTSRB smoke test first.
If the smoke test passes, start full classical training.
Then start hybrid training only after the classical checkpoint is saved.
Do not move datasets or outputs into the git repository.
Record commands and failures in /data/q3d/outputs/model_history_gtsrb/run_notes.md.
```

## Environment Setup

Run from the remote server shell:

```bash
cd ~
if [ ! -d q3d ]; then
  git clone https://github.com/xuyinmiao/q3d.git q3d
fi

cd ~/q3d
git pull --ff-only

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p /data/q3d/datasets/gtsrb
mkdir -p /data/q3d/outputs/model_history_gtsrb
```

Check the runtime:

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

If `cuda_available` is `False`, do not start full training. Fix the CUDA/PyTorch environment first.

## Dataset Import

Preferred mode: let torchvision download GTSRB into the data disk.

The first training run will use:

```bash
--data-dir /data/q3d/datasets/gtsrb
```

If the server has no internet access, upload the GTSRB dataset manually into:

```text
/data/q3d/datasets/gtsrb/
```

After import, check the directory exists and is non-empty:

```bash
du -sh /data/q3d/datasets/gtsrb
find /data/q3d/datasets/gtsrb -maxdepth 3 -type f | head
```

If torchvision download fails, keep the failed command output in:

```text
/data/q3d/outputs/model_history_gtsrb/run_notes.md
```

## Smoke Test First

Run this before any long job:

```bash
cd ~/q3d
source .venv/bin/activate

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

Expected result:

```text
/data/q3d/outputs/model_history_gtsrb/best_wideresnet_gtsrb.pth
/data/q3d/outputs/model_history_gtsrb/wideresnet_gtsrb_history.json
```

If the checkpoint is not created, stop and inspect the error before starting full training.

## Full Classical Training

Start after the smoke test passes:

```bash
cd ~/q3d
source .venv/bin/activate

python run_py/train_gtsrb.py \
  --model classical \
  --epochs 50 \
  --batch-size 128 \
  --widen-factor 4 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb
```

Expected checkpoint:

```text
/data/q3d/outputs/model_history_gtsrb/best_wideresnet_gtsrb.pth
```

## Full Hybrid Training

Start only after the classical baseline has completed:

```bash
cd ~/q3d
source .venv/bin/activate

python run_py/train_gtsrb.py \
  --model hybrid \
  --epochs 50 \
  --batch-size 64 \
  --widen-factor 4 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb
```

Expected checkpoint:

```text
/data/q3d/outputs/model_history_gtsrb/best_hybrid_qwideresnet_gtsrb.pth
```

Hybrid training uses PennyLane `default.qubit`; it can be much slower than classical training. If CUDA execution fails due to device incompatibility in the quantum layer, rerun hybrid training with:

```bash
--device cpu
```

and reduce batch size to `16` or `32`.

## Robustness Evaluation

Run this only after both checkpoints exist:

```bash
cd ~/q3d
source .venv/bin/activate

python run_py/attack_eval_gtsrb.py \
  --model both \
  --attacks fgsm,pgd,cw \
  --epsilons 0,0.0039215686,0.0078431373,0.0156862745,0.031372549 \
  --test-samples 2000 \
  --batch-size 64 \
  --pgd-steps 20 \
  --cw-steps 50 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb \
  --output-dir /data/q3d/outputs/model_history_gtsrb
```

Expected outputs:

```text
/data/q3d/outputs/model_history_gtsrb/gtsrb_robustness_results.json
/data/q3d/outputs/model_history_gtsrb/gtsrb_robustness_results.txt
/data/q3d/outputs/model_history_gtsrb/gtsrb_robustness_results.png
```

## Logging

For long jobs, prefer `tmux` or `nohup`.

Example:

```bash
nohup python run_py/train_gtsrb.py \
  --model classical \
  --epochs 50 \
  --batch-size 128 \
  --widen-factor 4 \
  --device cuda \
  --data-dir /data/q3d/datasets/gtsrb \
  --save-dir /data/q3d/outputs/model_history_gtsrb \
  > /data/q3d/outputs/model_history_gtsrb/classical_train.log 2>&1 &
```

Monitor logs:

```bash
tail -f /data/q3d/outputs/model_history_gtsrb/classical_train.log
```

## Safety Checks

Before committing or pushing from the server, verify that large files are not tracked:

```bash
cd ~/q3d
git status --short --ignored
git ls-files | grep -E '^(data_|model_history_)|\\.pth$|\\.pt$' || true
```

The second command should print nothing.

## Common Failures

- `ModuleNotFoundError`: activate `.venv` and rerun `python -m pip install -r requirements.txt`.
- `cuda_available False`: install the CUDA-compatible PyTorch build for the server.
- GTSRB download fails: manually upload the dataset to `/data/q3d/datasets/gtsrb`.
- Hybrid training is too slow: lower `--batch-size`, lower `--widen-factor`, or use `--device cpu` for correctness testing.
- Attack evaluation cannot find checkpoints: run both full training commands first or pass explicit `--classical-ckpt` and `--hybrid-ckpt`.
