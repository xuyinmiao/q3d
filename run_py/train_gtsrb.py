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


def checkpoint_name(model_name):
    if model_name == "classical":
        return "best_wideresnet_gtsrb.pth"
    return "best_hybrid_qwideresnet_gtsrb.pth"


def history_stem(model_name):
    if model_name == "classical":
        return "wideresnet_gtsrb"
    return "hybrid_qwideresnet_gtsrb"


def make_optimizer(args, model):
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


def train_epoch(model, loader, criterion, optimizer, device):
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


def run_training(args, model_name, train_loader, val_loader, test_loader, save_dir):
    device = select_device(args.device)
    if model_name == "hybrid" and device.type == "mps" and not args.allow_mps_quantum:
        console.print("[yellow]PennyLane default.qubit is CPU-oriented; using CPU for hybrid model.[/yellow]")
        device = torch.device("cpu")

    model = build_model(args, model_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(args, model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_path = os.path.join(save_dir, checkpoint_name(model_name))
    stem = history_stem(model_name)
    history_path = os.path.join(save_dir, f"{stem}_history.json")
    plot_path = os.path.join(save_dir, f"{stem}_history.png")

    console.rule(f"[bold cyan]GTSRB {model_name} training[/bold cyan]")
    console.print(Panel.fit(
        f"Model: {model_name}\n"
        f"Device: {device}\n"
        f"Epochs: {args.epochs}\n"
        f"Batch size: {args.batch_size}\n"
        f"Checkpoint: {ckpt_path}",
        title="Configuration",
        border_style="cyan",
    ))

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "test_acc": None,
        "best_val_acc": 0.0,
        "best_epoch": 0,
    }
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        console.rule(f"[bold blue]Epoch {epoch + 1}/{args.epochs}[/bold blue]")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
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
    _, test_acc = evaluate(model, test_loader, criterion, device, description="Test")
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train WideResNet/HybridQWideResNet on GTSRB.")
    parser.add_argument("--model", choices=["classical", "hybrid", "both"], default="both")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--save-dir", default=None)
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
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    root = project_root()
    data_dir = args.data_dir or os.path.join(root, "data_gtsrb")
    save_dir = args.save_dir or os.path.join(root, "model_history_gtsrb")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    base_device = select_device(args.device)
    train_loader, val_loader, test_loader = make_gtsrb_loaders(
        data_dir=data_dir,
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

    model_names = ["classical", "hybrid"] if args.model == "both" else [args.model]
    for model_name in model_names:
        run_training(args, model_name, train_loader, val_loader, test_loader, save_dir)


if __name__ == "__main__":
    main()
