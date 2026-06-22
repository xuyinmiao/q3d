import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
from attack import generate_adversarial_loader
from network import CNN, HybridQCNN

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_dir = os.path.join(project_root, "data_fmnist")
os.makedirs(data_dir, exist_ok=True)

save_dir = os.path.join(project_root, "model_history_fmnist")
os.makedirs(save_dir, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

X_test_list, y_test_list = [], []
for img, lbl in test_dataset:
    X_test_list.append(img)
    y_test_list.append(torch.tensor(lbl))
X_test_tensor = torch.stack(X_test_list)
y_test_tensor = torch.stack(y_test_list)

USE_SUBSET = True  
TEST_SAMPLES = 256  

if USE_SUBSET:
    subset_n = min(TEST_SAMPLES, X_test_tensor.size(0))
    X_test_tensor = X_test_tensor[:subset_n]
    y_test_tensor = y_test_tensor[:subset_n]

val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

console = Console()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_and_collect(model, data_loader, attack_name=""):
    model.eval()
    total_correct, total = 0, 0
    desc = f"Evaluating {attack_name}" if attack_name else "Evaluating model"
    with Progress(
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"{desc} ({len(data_loader)} batches)", total=len(data_loader))
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                total_correct += (predicted == labels).sum().item()
                progress.update(task, advance=1)
    acc = 100.0 * total_correct / total
    console.print(f"[green]✅ Accuracy: {acc:.2f}%[/green]")
    return acc

hybrid_model = HybridQCNN(n_qubits=8, n_layers=5, n_classes=10).to(device)
classical_model = CNN(latent_dim=8, n_classes=10).to(device)

hybrid_ckpt = os.path.join(save_dir, "best_hybrid_qcnn_model.pth")
classical_ckpt = os.path.join(save_dir, "best_fully_classical_model.pth")
hybrid_model.load_state_dict(torch.load(hybrid_ckpt, map_location=device))
classical_model.load_state_dict(torch.load(classical_ckpt, map_location=device))

console.print("[green]✅ Checkpoints loaded successfully![/green]")

# ===================== 定义攻击配置 =====================
attack_configs = {
    'fgsm': {
        'name': 'FGSM',
        'params': {}
    },
    'bim': {
        'name': 'BIM',
        'params': {'alpha': 0.01, 'iterations': 10}
    },
    'mim': {
        'name': 'MIM',
        'params': {'alpha': 0.01, 'iterations': 10, 'decay': 1.0}
    },
    'pgd': {
        'name': 'PGD',
        'params': {'alpha': 0.01, 'iterations': 10, 'random_start': True}
    },
    'cw': {
        'name': 'C&W',
        'params': {'c': 1.0, 'iterations': 10, 'learning_rate': 0.1}
    }
}

# ===================== 扰动点设置 =====================
epsilons = np.linspace(0.0, 0.2, 11)

all_results = {}
console.rule("[bold blue]⚔ Multi-Attack Adversarial Robustness Evaluation[/bold blue]")

# ===================== 为每个模型单独生成对抗样本并评估 =====================
for attack_key, attack_config in attack_configs.items():
    attack_name = attack_config['name']
    attack_params = attack_config['params']
    
    console.print(f"\n{'=' * 100}")
    console.print(f"[bold yellow]🎯 Testing Attack Method: {attack_name}[/bold yellow]", justify="center")
    console.print(f"{'=' * 100}\n")
    
    hybrid_accs, classical_accs = [], []
    
    for eps in epsilons:
        console.print(f"\n[bold cyan]>>> Evaluating {attack_name} with ε = {eps:.3f} <<<[/bold cyan]")
        console.print("─" * 80)
        
        # ====== 为 Hybrid QCNN 生成针对性的对抗样本 ======
        console.print(f"\n[bold cyan]→ Generating adversarial set for Hybrid QCNN[/bold cyan]")
        hybrid_adv_loader = generate_adversarial_loader(
            hybrid_model, val_loader, eps, 
            attack_type=attack_key, 
            model_name="Hybrid QCNN",
            **attack_params
        )
        
        console.print(f"[bold cyan]→ Evaluating Hybrid QCNN on adversarial examples[/bold cyan]")
        hybrid_acc = evaluate_and_collect(hybrid_model, hybrid_adv_loader, f"{attack_name} ε={eps:.3f}")
        
        # ====== 为 Classical CNN 生成针对性的对抗样本 ======
        console.print(f"\n[bold green]→ Generating adversarial set for Classical CNN[/bold green]")
        classical_adv_loader = generate_adversarial_loader(
            classical_model, val_loader, eps, 
            attack_type=attack_key,
            model_name="Classical CNN",
            **attack_params
        )
        
        console.print(f"[bold green]→ Evaluating Classical CNN on adversarial examples[/bold green]")
        classical_acc = evaluate_and_collect(classical_model, classical_adv_loader, f"{attack_name} ε={eps:.3f}")
        
        hybrid_accs.append(hybrid_acc)
        classical_accs.append(classical_acc)
        
        console.print(Panel.fit(
            f"[bold white]{attack_name} Attack Results (ε = {eps:.3f})[/bold white]\n"
            f"[cyan]Hybrid QCNN Accuracy:[/cyan] {hybrid_acc:.2f}%\n"
            f"[green]Classical CNN Accuracy:[/green] {classical_acc:.2f}%",
            title=f"[bold magenta]Summary[/bold magenta]",
            border_style="bright_blue",
        ))
    
    all_results[attack_key] = {
        'name': attack_name,
        'hybrid_accs': hybrid_accs,
        'classical_accs': classical_accs
    }

console.rule("[bold green]✅ All evaluations completed![/bold green]")

# ===================== 生成综合对比图 =====================
console.print("\n[bold yellow]📊 Generating comprehensive robustness comparison...[/bold yellow]")

plt.figure(figsize=(14, 8))
plt.style.use("seaborn-v0_8-whitegrid")

# 定义深色颜色方案（每种攻击一个颜色）
attack_colors = {
    'fgsm': '#1f77b4',      # 深蓝
    'bim': '#ff7f0e',       # 深橙
    'mim': '#2ca02c',       # 深绿
    'pgd': '#d62728',       # 深红
    'cw': '#9467bd',        # 深紫
}

# 定义不同的 marker 方案（每种政击一个 marker）
attack_markers = {
    'fgsm': 'o',      # Circle
    'bim': 's',       # Square
    'mim': '^',       # Triangle Up
    'pgd': 'D',       # Diamond
    'cw': 'v',        # Triangle Down
}

# 绘制所有曲线
for attack_key, results in all_results.items():
    attack_name = results['name']
    hybrid_accs_frac = np.array(results['hybrid_accs']) / 100.0
    classical_accs_frac = np.array(results['classical_accs']) / 100.0
    
    color = attack_colors[attack_key]
    marker = attack_markers[attack_key]
    
    plt.plot(epsilons, hybrid_accs_frac,
             linestyle='-',           # 混合网络使用实线
             linewidth=2.5,
             color=color,
             marker=marker,
             markersize=8,
             solid_capstyle='round')
    
    plt.plot(epsilons, classical_accs_frac,
             linestyle='--',          # 经典网络使用虚线
             linewidth=2.0,
             color=color,
             marker=marker,
             markersize=8,
             dash_capstyle='round')

# 设置坐标轴
plt.xlabel("Perturbation Budget", fontsize=16, fontweight='bold')
plt.ylabel("Accuracy", fontsize=16, fontweight='bold')
plt.title("Adversarial Robustness Comparison (Fashion-MNIST)", fontsize=18, fontweight='bold', pad=20)

plt.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
plt.xticks(epsilons, [f"{e:.2f}" for e in epsilons], fontsize=12)
plt.yticks(np.arange(0.0, 1.01, 0.1), [f"{y:.1f}" for y in np.arange(0.0, 1.01, 0.1)], fontsize=12)
plt.ylim(-0.01, 1.01)
plt.xlim(-0.01, 0.21)

# 自定义图例句柄
legend_elements = [
    # 模型类型图例（黑白线型区分）
    Line2D([0], [0], color='black', lw=2.5, linestyle='-', label='Hybrid QCNN'),
    Line2D([0], [0], color='black', lw=2.0, linestyle='--', label='CNN'),
    Line2D([0], [0], color='white', label=''),  # 空行分隔
]

# 攻击方法图例（颜色和Marker区分）
for attack_key, results in all_results.items():
    attack_name = results['name']
    color = attack_colors[attack_key]
    marker = attack_markers[attack_key]
    legend_elements.append(
        Line2D([0], [0], color=color, marker=marker, linestyle='None', 
               markersize=8, label=attack_name)
    )

# 优化图例布局
plt.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
          frameon=True, fancybox=True, shadow=True,
          ncol=1)
plt.tight_layout()

# 保存高分辨率图片
final_save_path = os.path.join(save_dir, "adv_robustness_vs_pbs.png")
plt.savefig(final_save_path, dpi=400, bbox_inches='tight', facecolor='white')
plt.close()

console.print(f"[bold cyan]✅ Saved robustness comparison to: {final_save_path}[/bold cyan]")

# ===================== 生成结果表格 =====================
console.print("\n[bold yellow]📋 Generating results summary table...[/bold yellow]\n")

for eps in epsilons:  
    table = Table(title=f"Accuracy Results at ε = {eps:.3f}", show_header=True, header_style="bold magenta")
    table.add_column("Attack Method", style="cyan", width=15)
    table.add_column("Hybrid QCNN", style="green", justify="right")
    table.add_column("CNN", style="yellow", justify="right")
    
    for attack_key, results in all_results.items():
        attack_name = results['name']
        idx = np.argmin(np.abs(epsilons - eps))
        hybrid_acc = results['hybrid_accs'][idx]
        classical_acc = results['classical_accs'][idx]
        table.add_row(attack_name, f"{hybrid_acc:.2f}%", f"{classical_acc:.2f}%")
    
    console.print(table)

# ===================== 保存数值结果到文件 =====================
results_file = os.path.join(save_dir, "robustness_vs_pbs_results.txt")
with open(results_file, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("ADVERSARIAL ROBUSTNESS EVALUATION RESULTS (Perturbation Budgets)\n")
    f.write("=" * 100 + "\n\n")
    
    for attack_key, results in all_results.items():
        attack_name = results['name']
        f.write(f"\n{attack_name} Attack Results:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Epsilon':<12} {'Hybrid QCNN':<15} {'CNN':<15}\n")
        f.write("-" * 80 + "\n")
        
        for i, eps in enumerate(epsilons):
            hybrid_acc = results['hybrid_accs'][i]
            classical_acc = results['classical_accs'][i]
            f.write(f"{eps:<12.3f} {hybrid_acc:<15.2f} {classical_acc:<15.2f}\n")
        f.write("\n")

console.print(f"[bold cyan]✅ Saved detailed numerical results to: {results_file}[/bold cyan]")
console.print("\n[bold green]🎉 All evaluations and visualizations completed successfully![/bold green]\n")