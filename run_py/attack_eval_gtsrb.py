import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from attack import generate_adversarial_loader
from gtsrb_common import (
    GTSRB_NUM_CLASSES,
    NormalizeWrapper,
    make_gtsrb_pixel_test_loader,
    project_root,
    save_json,
    seed_everything,
    select_device,
)
from network import HybridQWideResNet, WideResNet


console = Console()


def build_model(args, model_name):
    if model_name == "classical":
        return WideResNet(
            n_classes=GTSRB_NUM_CLASSES,
            depth=args.depth,
            widen_factor=args.widen_factor,
            drop_rate=args.drop_rate,
            latent_dim=args.latent_dim,
        )
    if model_name == "hybrid":
        return HybridQWideResNet(
            n_classes=GTSRB_NUM_CLASSES,
            depth=args.depth,
            widen_factor=args.widen_factor,
            drop_rate=args.drop_rate,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
        )
    raise ValueError(f"Unknown model: {model_name}")


def default_checkpoint(save_dir, model_name):
    if model_name == "classical":
        return os.path.join(save_dir, "best_wideresnet_gtsrb.pth")
    return os.path.join(save_dir, "best_hybrid_qwideresnet_gtsrb.pth")


def resolve_device(args, model_name):
    device = select_device(args.device)
    if model_name == "hybrid" and device.type == "mps" and not args.allow_mps_quantum:
        console.print("[yellow]PennyLane default.qubit is CPU-oriented; using CPU for hybrid model.[/yellow]")
        return torch.device("cpu")
    return device


def load_wrapped_model(args, model_name, save_dir):
    device = resolve_device(args, model_name)
    model = build_model(args, model_name).to(device)
    ckpt = args.classical_ckpt if model_name == "classical" else args.hybrid_ckpt
    ckpt = ckpt or default_checkpoint(save_dir, model_name)

    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            f"Train it first, for example: python run_py/train_gtsrb.py --model {model_name}"
        )

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    wrapped = NormalizeWrapper(model).to(device)
    wrapped.eval()
    return wrapped, device, ckpt


@torch.no_grad()
def evaluate_accuracy(model, loader, device, description="Evaluating"):
    model.eval()
    total_correct, total = 0, 0
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=len(loader))
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.update(task, advance=1)
    return 100.0 * total_correct / max(total, 1)


def parse_float_list(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    items = [item.strip() for item in value.split(",") if item.strip()]
    return [float(item) for item in items]


def parse_attack_list(value):
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    if "all" in items:
        return ["fgsm", "pgd", "cw"]
    allowed = {"fgsm", "pgd", "cw"}
    unknown = sorted(set(items) - allowed)
    if unknown:
        raise ValueError(f"Unknown attacks: {unknown}. Allowed: {sorted(allowed)}")
    return items


def attack_params(args, attack_name, epsilon):
    if attack_name == "fgsm":
        return {}
    if attack_name == "pgd":
        alpha = args.pgd_alpha if args.pgd_alpha is not None else epsilon / args.pgd_alpha_divisor
        return {
            "alpha": alpha,
            "iterations": args.pgd_steps,
            "random_start": True,
        }
    if attack_name == "cw":
        return {
            "c": args.cw_c,
            "iterations": args.cw_steps,
            "learning_rate": args.cw_lr,
        }
    raise ValueError(f"Unknown attack: {attack_name}")


def plot_results(results, epsilons, path):
    plt.figure(figsize=(12, 7))
    attack_colors = {
        "fgsm": "#1f77b4",
        "pgd": "#d62728",
        "cw": "#9467bd",
    }
    model_styles = {
        "classical": "--",
        "hybrid": "-",
    }

    for model_name, model_results in results.items():
        for attack_name, values in model_results.items():
            label = f"{model_name} {attack_name.upper()}"
            plt.plot(
                epsilons,
                np.array(values["accuracy"]) / 100.0,
                linestyle=model_styles[model_name],
                linewidth=2.3,
                marker="o",
                color=attack_colors[attack_name],
                label=label,
            )

    plt.xlabel("Perturbation budget")
    plt.ylabel("Accuracy")
    plt.title("GTSRB adversarial robustness")
    plt.xticks(epsilons, [f"{eps:.4f}" for eps in epsilons], rotation=30)
    plt.yticks(np.arange(0.0, 1.01, 0.1))
    plt.ylim(-0.01, 1.01)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_text_results(results, epsilons, path):
    with open(path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("GTSRB ADVERSARIAL ROBUSTNESS RESULTS\n")
        f.write("=" * 100 + "\n\n")

        for model_name, model_results in results.items():
            f.write(f"{model_name.upper()} MODEL\n")
            f.write("-" * 80 + "\n")
            for attack_name, values in model_results.items():
                f.write(f"\n{attack_name.upper()} Attack Results:\n")
                f.write(f"{'Epsilon':<12} {'Accuracy':<12} {'ASR':<12}\n")
                clean_acc = values["accuracy"][0] if epsilons and epsilons[0] == 0 else None
                for eps, acc in zip(epsilons, values["accuracy"]):
                    asr = 0.0 if clean_acc in (None, 0.0) else max(0.0, (clean_acc - acc) / clean_acc * 100.0)
                    f.write(f"{eps:<12.6f} {acc:<12.2f} {asr:<12.2f}\n")
                f.write("\n")


def run_eval_for_model(args, model_name, test_loader, save_dir, attacks, epsilons):
    model, device, ckpt = load_wrapped_model(args, model_name, save_dir)
    console.print(Panel.fit(
        f"Model: {model_name}\nDevice: {device}\nCheckpoint: {ckpt}",
        title="Evaluation model",
        border_style="cyan",
    ))

    model_results = {}
    clean_acc = evaluate_accuracy(model, test_loader, device, f"{model_name} clean")
    console.print(f"[green]{model_name} clean accuracy: {clean_acc:.2f}%[/green]")

    for attack_name in attacks:
        accuracies = []
        console.rule(f"[bold blue]{model_name} {attack_name.upper()}[/bold blue]")
        for epsilon in epsilons:
            if epsilon == 0.0:
                acc = clean_acc
            else:
                params = attack_params(args, attack_name, epsilon)
                adv_loader = generate_adversarial_loader(
                    model,
                    test_loader,
                    epsilon,
                    attack_type=attack_name,
                    model_name=f"{model_name} GTSRB",
                    clip_min=0.0,
                    clip_max=1.0,
                    **params,
                )
                acc = evaluate_accuracy(
                    model,
                    adv_loader,
                    device,
                    f"{model_name} {attack_name} eps={epsilon:.6f}",
                )
            accuracies.append(acc)

            table = Table(title=f"{model_name} {attack_name.upper()} eps={epsilon:.6f}")
            table.add_column("Metric")
            table.add_column("Value", justify="right")
            table.add_row("Accuracy", f"{acc:.2f}%")
            table.add_row("Attack success rate", f"{max(0.0, (clean_acc - acc) / max(clean_acc, 1e-12) * 100.0):.2f}%")
            console.print(table)

        model_results[attack_name] = {
            "epsilon": epsilons,
            "accuracy": accuracies,
        }
    return model_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GTSRB adversarial robustness.")
    parser.add_argument("--model", choices=["classical", "hybrid", "both"], default="both")
    parser.add_argument("--attacks", default="fgsm,pgd,cw")
    parser.add_argument(
        "--epsilons",
        default="0,0.0039215686,0.0078431373,0.0156862745,0.031372549,0.062745098",
        help="Comma-separated pixel-space epsilons. Defaults to 0,1/255,2/255,4/255,8/255,16/255.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--classical-ckpt", default=None)
    parser.add_argument("--hybrid-ckpt", default=None)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", dest="download", action="store_true", default=True)
    parser.add_argument("--no-download", dest="download", action="store_false")

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--allow-mps-quantum", action="store_true")
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen-factor", type=int, default=4)
    parser.add_argument("--drop-rate", type=float, default=0.3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)

    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--pgd-alpha", type=float, default=None)
    parser.add_argument("--pgd-alpha-divisor", type=float, default=4.0)
    parser.add_argument("--cw-steps", type=int, default=50)
    parser.add_argument("--cw-c", type=float, default=1.0)
    parser.add_argument("--cw-lr", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    root = project_root()
    data_dir = args.data_dir or os.path.join(root, "data_gtsrb")
    save_dir = args.save_dir or os.path.join(root, "model_history_gtsrb")
    output_dir = args.output_dir or save_dir
    os.makedirs(output_dir, exist_ok=True)

    attacks = parse_attack_list(args.attacks)
    epsilons = parse_float_list(args.epsilons)
    base_device = select_device(args.device)
    test_loader = make_gtsrb_pixel_test_loader(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=base_device,
        download=args.download,
        test_samples=args.test_samples,
    )

    model_names = ["classical", "hybrid"] if args.model == "both" else [args.model]
    results = {}
    for model_name in model_names:
        results[model_name] = run_eval_for_model(
            args,
            model_name,
            test_loader,
            save_dir,
            attacks,
            epsilons,
        )

    json_path = os.path.join(output_dir, "gtsrb_robustness_results.json")
    txt_path = os.path.join(output_dir, "gtsrb_robustness_results.txt")
    plot_path = os.path.join(output_dir, "gtsrb_robustness_results.png")
    save_json(json_path, results)
    write_text_results(results, epsilons, txt_path)
    plot_results(results, epsilons, plot_path)

    console.print(Panel.fit(
        f"JSON: {json_path}\nTXT: {txt_path}\nPlot: {plot_path}",
        title="Saved results",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
