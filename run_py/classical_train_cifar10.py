import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from network import WideResNet

# ===================== 设置数据和结果保存目录 =====================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

save_dir = os.path.join(project_root, "model_history_cifar10")
os.makedirs(save_dir, exist_ok=True)

data_dir = os.path.join(project_root, "data_cifar10")
os.makedirs(data_dir, exist_ok=True)

# ===================== 数据准备 =====================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

cifar10_dataset = CIFAR10(root=data_dir, train=True, download=True)

val_size = int(0.2 * len(cifar10_dataset))
train_size = len(cifar10_dataset) - val_size

generator = torch.Generator().manual_seed(42)
train_subset, val_subset = random_split(range(len(cifar10_dataset)), [train_size, val_size], generator=generator)
train_indices, val_indices = list(train_subset.indices), list(val_subset.indices)

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = TransformedSubset(cifar10_dataset, train_indices, transform_train)
val_dataset = TransformedSubset(cifar10_dataset, val_indices, transform_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

# ===================== 训练准备 =====================
model = WideResNet(n_classes=10, latent_dim=8)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

console = Console()
console.rule("[bold cyan]🚀 WideResNet Training Started[/bold cyan]")
console.print(Panel.fit(
    "[bold bright_cyan]Model:[/bold bright_cyan] WideResNet\n"
    f"[bold bright_cyan]Device:[/bold bright_cyan] [yellow]{device}[/yellow]\n"
    f"[bold bright_cyan]Data Directory:[/bold bright_cyan] [yellow]{data_dir}[/yellow]\n"
    f"[bold bright_cyan]Model Save Path:[/bold bright_cyan] [yellow]{save_dir}[/yellow]",
    border_style="cyan",
    title="[bold bright_white]Training Configuration[/bold bright_white]",
    title_align="left"
))

# ===================== 训练过程 =====================
history = {'batch_loss': [], 'epoch_loss': [], 'epoch_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc, best_epoch = 0.0, 0
num_epochs = 50
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    console.rule(f"[bold blue]🧩 Epoch {epoch+1}/{num_epochs}[/bold blue]")

    # ---------- Training ----------
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="cyan", finished_style="bright_cyan"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        model.train()
        train_task = progress.add_task("[cyan]Training", total=len(train_loader))
        epoch_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            history['batch_loss'].append(batch_loss)
            progress.update(train_task, advance=1,
                            description=f"[cyan]Training - Loss: {batch_loss:.4f}")

        train_accuracy = 100 * correct / total
        history['epoch_loss'].append(epoch_loss / len(train_loader))
        history['epoch_acc'].append(train_accuracy)

    scheduler.step()

    # ---------- Validation ----------
    with Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="bright_green"),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        model.eval()
        val_task = progress.add_task("[green]Validating", total=len(val_loader))
        val_loss_sum, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                predicted = outputs.argmax(dim=1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                progress.update(val_task, advance=1)

        val_accuracy = 100 * val_correct / val_total
        val_loss_avg = val_loss_sum / len(val_loader)
        history['val_loss'].append(val_loss_avg)
        history['val_acc'].append(val_accuracy)

    # ---------- Epoch Summary ----------
    table = Table(title=f"[bold bright_white]📊 Epoch {epoch+1} Summary[/bold bright_white]",
                  header_style="bold magenta", box=None)
    table.add_column("Metric", style="bold white")
    table.add_column("Training", justify="right", style="bright_cyan")
    table.add_column("Validation", justify="right", style="bright_green")
    table.add_row("Loss", f"{history['epoch_loss'][-1]:.4f}", f"{val_loss_avg:.4f}")
    table.add_row("Accuracy", f"{train_accuracy:.2f}%", f"{val_accuracy:.2f}%")
    console.print(table)

    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        best_epoch = epoch
        model_path = os.path.join(save_dir, "best_fully_classical_model.pth")
        torch.save(model.state_dict(), model_path)
        console.print(
            f"[bold bright_green]✅ New best model saved![/bold bright_green] "
            f"[white](epoch {epoch+1}, val_acc = {val_accuracy:.2f}%)[/white]"
        )

# ===================== 保存训练过程 =====================
history_path = os.path.join(save_dir, "fully_classical_history.json")
with open(history_path, 'w') as f:
    json.dump(history, f)
console.print(f"[green]📁 Training history saved to:[/green] [yellow]{history_path}[/yellow]")

# ===================== 绘制结果 =====================
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(history['batch_loss'])
plt.title('Batch Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot(history['epoch_loss'], 'd-', label='Training')
plt.plot(history['val_loss'], 'd-', label='Validation')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(history['epoch_acc'], 'd-', label='Training')
plt.plot(history['val_acc'], 'd-', label='Validation')
plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(history['val_acc'], 'd-', color='green')
plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f'Best: {best_val_acc:.2f}%')
plt.title('Best Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plot_path = os.path.join(save_dir, "fully_classical_training_history.png")
plt.savefig(plot_path)
plt.close()
console.rule("[bold bright_white]📈 Training Results[/bold bright_white]")
console.print(f"[green]✅ Training plot saved to:[/green] [yellow]{plot_path}[/yellow]")

# ===================== 总结 =====================
console.rule("[bold bright_green]🎉 Training Complete![/bold bright_green]")
console.print(Panel.fit(
    f"[bold bright_white]Best Validation Accuracy:[/bold bright_white] [bold green]{best_val_acc:.2f}%[/bold green]\n"
    f"[bold bright_white]Best Epoch:[/bold bright_white] [cyan]{best_epoch+1}[/cyan]",
    border_style="bright_green",
    title="[bold bright_yellow]Summary[/bold bright_yellow]",
    title_align="left"
))
