import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from gtsrb_common import (
    GTSRB_NUM_CLASSES,
    make_gtsrb_loaders,
    project_root,
    save_json,
    seed_everything,
    select_device,
)
from network import (
    HybridQWideResNet,
    HybridQWideResNetMLP,
    HybridQWideResNetNoQuantum,
    WideResNet,
    WideResNetStrong,
)


console = Console()

MODEL_ALIASES = {
    "classical": "classical_legacy",
    "hybrid": "hybrid_quantum",
}

MODEL_GROUPS = {
    "paper_core": ["classical_strong", "hybrid_quantum"],
    "ablation": ["hybrid_quantum", "hybrid_noquantum", "hybrid_mlp"],
    "all": ["classical_legacy", "classical_strong", "hybrid_quantum", "hybrid_noquantum", "hybrid_mlp"],
    "both": ["classical_legacy", "hybrid_quantum"],
}

CANONICAL_MODELS = (
    "classical_legacy",
    "classical_strong",
    "hybrid_quantum",
    "hybrid_noquantum",
    "hybrid_mlp",
)


def normalize_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def expand_model_selection(value):
    selected = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        item = normalize_model_name(item)
        if item in MODEL_GROUPS:
            names = MODEL_GROUPS[item]
        else:
            names = [item]
        for name in names:
            if name not in CANONICAL_MODELS:
                allowed = sorted(set(CANONICAL_MODELS) | set(MODEL_ALIASES) | set(MODEL_GROUPS))
                raise ValueError(f"Unknown model '{name}'. Allowed values: {allowed}")
            if name not in selected:
                selected.append(name)
    return selected


def build_model(args, model_name):
    model_name = normalize_model_name(model_name)
    if model_name == "classical_legacy":
        return WideResNet(
            n_classes=GTSRB_NUM_CLASSES,
            depth=args.depth,
            widen_factor=args.widen_factor,
            drop_rate=args.drop_rate,
            latent_dim=args.latent_dim,
        )
    if model_name == "classical_strong":
        return WideResNetStrong(
            n_classes=GTSRB_NUM_CLASSES,
            depth=args.depth,
            widen_factor=args.widen_factor,
            drop_rate=args.drop_rate,
        )
    if model_name == "hybrid_quantum":
        return HybridQWideResNet(
            n_classes=GTSRB_NUM_CLASSES,
            depth=args.depth,
            widen_factor=args.widen_factor,
            drop_rate=args.drop_rate,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            input_scaling=args.quantum_input_scaling,
        )
    if model_name == "hybrid_noquantum":
        return HybridQWideResNetNoQuantum(
            n_classes=GTSRB_NUM_CLASSES,
            depth=args.depth,
            widen_factor=args.widen_factor,
            drop_rate=args.drop_rate,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
        )
    if model_name == "hybrid_mlp":
        return HybridQWideResNetMLP(
            n_classes=GTSRB_NUM_CLASSES,
            depth=args.depth,
            widen_factor=args.widen_factor,
            drop_rate=args.drop_rate,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
        )
    raise ValueError(f"Unknown model: {model_name}")


def checkpoint_name(model_name):
    return f"best_{normalize_model_name(model_name)}_gtsrb.pth"


def history_stem(model_name):
    return f"{normalize_model_name(model_name)}_gtsrb"


def make_optimizer(args, model, model_name=""):
    """Build optimizer. For hybrid_quantum, use a separate (smaller) lr on the
    quantum layer parameters to stabilize PennyLane default.qubit training."""
    canonical = normalize_model_name(model_name) if model_name else ""
    use_split = (
        canonical == "hybrid_quantum"
        and getattr(args, "quantum_lr", None) is not None
        and hasattr(model, "quantum_parameters")
    )

    if use_split:
        classical_params = model.classical_parameters()
        quantum_params = model.quantum_parameters()
        if args.optimizer == "sgd":
            return optim.SGD(
                [
                    {"params": classical_params, "lr": args.lr},
                    {"params": quantum_params, "lr": args.quantum_lr},
                ],
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
                nesterov=True,
            )
        if args.optimizer == "adamw":
            return optim.AdamW(
                [
                    {"params": classical_params, "lr": args.lr},
                    {"params": quantum_params, "lr": args.quantum_lr},
                ],
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def train_epoch(model, loader, criterion, optimizer, device, grad_clip=None):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Training", total=len(loader))
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.update(task, advance=1, description=f"Training loss={loss.item():.4f}")

    return total_loss / max(len(loader), 1), 100.0 * total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, description="Validation"):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=len(loader))
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
            progress.update(task, advance=1)

    return total_loss / max(len(loader), 1), 100.0 * total_correct / max(total, 1)


def plot_history(history, path, title):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title(f"{title} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.title(f"{title} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_metadata(args, model_name, train_loader, val_loader, test_loader, device):
    return {
        "model": normalize_model_name(model_name),
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "depth": args.depth,
        "widen_factor": args.widen_factor,
        "drop_rate": args.drop_rate,
        "latent_dim": args.latent_dim,
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "quantum_input_scaling": getattr(args, "quantum_input_scaling", "2pi"),
        "quantum_lr": getattr(args, "quantum_lr", None),
        "grad_clip": getattr(args, "grad_clip", None),
        "image_size": args.image_size,
        "val_fraction": args.val_fraction,
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "data_dir": args.data_dir,
    }


def run_training(args, model_name, train_loader, val_loader, test_loader, save_dir):
    model_name = normalize_model_name(model_name)
    device = select_device(args.device)
    if model_name == "hybrid_quantum" and device.type == "mps" and not args.allow_mps_quantum:
        console.print("[yellow]PennyLane default.qubit is CPU-oriented; using CPU for hybrid_quantum.[/yellow]")
        device = torch.device("cpu")

    model = build_model(args, model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(args, model, model_name=model_name)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = os.path.join(save_dir, checkpoint_name(model_name))
    stem = history_stem(model_name)
    history_path = os.path.join(save_dir, f"{stem}_history.json")
    plot_path = os.path.join(save_dir, f"{stem}_history.png")

    console.rule(f"[bold cyan]GTSRB {model_name} training[/bold cyan]")
    console.print(Panel.fit(
        f"Model: {model_name}\n"
        f"Device: {device}\n"
        f"Seed: {args.seed}\n"
        f"Epochs: {args.epochs}\n"
        f"Batch size: {args.batch_size}\n"
        f"Checkpoint: {ckpt_path}",
        title="Configuration",
        border_style="cyan",
    ))

    history = {
        "metadata": make_metadata(args, model_name, train_loader, val_loader, test_loader, device),
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_loss": None,
        "test_acc": None,
        "best_val_acc": 0.0,
        "best_epoch": 0,
        "checkpoint": ckpt_path,
    }
    best_val_acc = float("-inf")

    for epoch in range(args.epochs):
        console.rule(f"[bold blue]Epoch {epoch + 1}/{args.epochs}[/bold blue]")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, grad_clip=getattr(args, "grad_clip", None))
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        table = Table(title=f"Epoch {epoch + 1} Summary")
        table.add_column("Metric")
        table.add_column("Train", justify="right")
        table.add_column("Val", justify="right")
        table.add_row("Loss", f"{train_loss:.4f}", f"{val_loss:.4f}")
        table.add_row("Accuracy", f"{train_acc:.2f}%", f"{val_acc:.2f}%")
        console.print(table)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history["best_val_acc"] = best_val_acc
            history["best_epoch"] = epoch + 1
            torch.save(model.state_dict(), ckpt_path)
            console.print(f"[green]Saved new best checkpoint: {ckpt_path}[/green]")

        save_json(history_path, history)
        plot_history(history, plot_path, model_name)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, description="Test")
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    save_json(history_path, history)
    plot_history(history, plot_path, model_name)

    console.print(Panel.fit(
        f"Best val accuracy: {best_val_acc:.2f}%\n"
        f"Best epoch: {history['best_epoch']}\n"
        f"Test accuracy: {test_acc:.2f}%",
        title=f"{model_name} result",
        border_style="green",
    ))
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train GTSRB model variants.")
    parser.add_argument(
        "--model",
        default="paper_core",
        help="Comma-separated models or group. Examples: classical_strong,hybrid_quantum,paper_core,ablation,all.",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--seed-subdir", action="store_true", help="Write outputs under save-dir/seed_<seed>.")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--val-samples", type=int, default=None)
    parser.add_argument("--download", dest="download", action="store_true", default=True)
    parser.add_argument("--no-download", dest="download", action="store_false")

    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--allow-mps-quantum", action="store_true")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--depth", type=int, default=28)
    parser.add_argument("--widen-factor", type=int, default=4)
    parser.add_argument("--drop-rate", type=float, default=0.3)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--n-qubits", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)

    # --- Quantum-layer training stabilization (opt-in; defaults preserve legacy behavior) ---
    parser.add_argument(
        "--quantum-input-scaling",
        choices=["2pi", "none"],
        default="2pi",
        help="Scaling applied to sigmoid output before the quantum layer. "
             "'2pi' is legacy (multiply by 2*pi); 'none' feeds raw [0,1] to AngleEmbedding "
             "and avoids PauliZ gradient saturation. Use 'none' if hybrid_quantum fails to converge.",
    )
    parser.add_argument(
        "--quantum-lr",
        type=float,
        default=None,
        help="Separate learning rate for the quantum layer parameters of HybridQWideResNet. "
             "If unset, a single lr is used for all parameters (legacy). "
             "Try 0.01 when the classical lr (0.05) destabilizes quantum training.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=None,
        help="Max L2 norm for gradient clipping on all parameters. "
             "Recommended 1.0 for unstable quantum training. Disabled if unset.",
    )
    parser.add_argument(
        "--no-2pi-scaling",
        dest="quantum_input_scaling",
        action="store_const",
        const="none",
        help="Shorthand for --quantum-input-scaling none.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    root = project_root()
    args.data_dir = args.data_dir or os.path.join(root, "data_gtsrb")
    save_dir = args.save_dir or os.path.join(root, "model_history_gtsrb")
    if args.seed_subdir:
        save_dir = os.path.join(save_dir, f"seed_{args.seed}")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    base_device = select_device(args.device)
    train_loader, val_loader, test_loader = make_gtsrb_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
        device=base_device,
        seed=args.seed,
        download=args.download,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )

    for model_name in expand_model_selection(args.model):
        run_training(args, model_name, train_loader, val_loader, test_loader, save_dir)


if __name__ == "__main__":
    main()
