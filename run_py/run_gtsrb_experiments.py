import argparse
import os
import subprocess
import sys


DEFAULT_MODELS = "classical_strong,hybrid_quantum,hybrid_noquantum,hybrid_mlp"
DEFAULT_SEEDS = "42,123,2024"
DEFAULT_EPSILONS = "0,0.0039215686,0.0078431373,0.0156862745,0.031372549,0.062745098"


def parse_csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def add_flag(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def checkpoint_path(seed_dir, model_name):
    return os.path.join(seed_dir, f"best_{model_name}_gtsrb.pth")


def run_command(cmd, dry_run=False):
    print(" ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def train_model(args, seed, model_name, seed_dir):
    ckpt = checkpoint_path(seed_dir, model_name)
    if os.path.exists(ckpt) and not args.overwrite:
        print(f"[skip] checkpoint exists: {ckpt}", flush=True)
        return

    cmd = [
        sys.executable,
        "run_py/train_gtsrb.py",
        "--model",
        model_name,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--optimizer",
        args.optimizer,
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--depth",
        str(args.depth),
        "--widen-factor",
        str(args.widen_factor),
        "--drop-rate",
        str(args.drop_rate),
        "--latent-dim",
        str(args.latent_dim),
        "--n-qubits",
        str(args.n_qubits),
        "--n-layers",
        str(args.n_layers),
        "--device",
        args.device,
        "--data-dir",
        args.data_dir,
        "--save-dir",
        seed_dir,
    ]
    add_flag(cmd, "--train-samples", args.train_samples)
    add_flag(cmd, "--val-samples", args.val_samples)
    if args.no_download:
        cmd.append("--no-download")
    if args.allow_mps_quantum:
        cmd.append("--allow-mps-quantum")
    run_command(cmd, dry_run=args.dry_run)


def evaluate_seed(args, seed, models, seed_dir):
    result_path = os.path.join(seed_dir, "gtsrb_robustness_results.json")
    if os.path.exists(result_path) and not args.overwrite:
        print(f"[skip] robustness results exist: {result_path}", flush=True)
        return

    cmd = [
        sys.executable,
        "run_py/attack_eval_gtsrb.py",
        "--model",
        ",".join(models),
        "--attacks",
        args.attacks,
        "--epsilons",
        args.epsilons,
        "--test-samples",
        str(args.test_samples),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--data-dir",
        args.data_dir,
        "--save-dir",
        seed_dir,
        "--output-dir",
        seed_dir,
        "--depth",
        str(args.depth),
        "--widen-factor",
        str(args.widen_factor),
        "--drop-rate",
        str(args.drop_rate),
        "--latent-dim",
        str(args.latent_dim),
        "--n-qubits",
        str(args.n_qubits),
        "--n-layers",
        str(args.n_layers),
        "--pgd-steps",
        str(args.pgd_steps),
        "--cw-steps",
        str(args.cw_steps),
        "--cw-c",
        str(args.cw_c),
        "--cw-lr",
        str(args.cw_lr),
    ]
    if args.no_download:
        cmd.append("--no-download")
    if args.allow_mps_quantum:
        cmd.append("--allow-mps-quantum")
    run_command(cmd, dry_run=args.dry_run)


def summarize(args):
    summary_path = os.path.join(args.output_dir, "summary_robustness.csv")
    if os.path.exists(summary_path) and not args.overwrite:
        print(f"[skip] summary exists: {summary_path}", flush=True)
        return

    cmd = [
        sys.executable,
        "run_py/summarize_gtsrb_results.py",
        "--results-dir",
        args.output_dir,
        "--output-dir",
        args.output_dir,
    ]
    run_command(cmd, dry_run=args.dry_run)


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-seed GTSRB training, attacks, and summaries.")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--data-dir", default="/data/q3d/datasets/gtsrb")
    parser.add_argument("--output-dir", default="/data/q3d/outputs/gtsrb_paper")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen-factor", type=int, default=4)
    parser.add_argument("--drop-rate", type=float, default=0.3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-mps-quantum", action="store_true")

    parser.add_argument("--attacks", default="fgsm,pgd,cw")
    parser.add_argument("--epsilons", default=DEFAULT_EPSILONS)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--cw-steps", type=int, default=50)
    parser.add_argument("--cw-c", type=float, default=1.0)
    parser.add_argument("--cw-lr", type=float, default=0.01)

    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = parse_csv(args.seeds, int)
    models = parse_csv(args.models)

    os.makedirs(args.output_dir, exist_ok=True)
    for seed in seeds:
        seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        if not args.skip_train:
            for model_name in models:
                train_model(args, seed, model_name, seed_dir)

        if not args.skip_eval:
            evaluate_seed(args, seed, models, seed_dir)

    if not args.skip_summary:
        summarize(args)


if __name__ == "__main__":
    main()
