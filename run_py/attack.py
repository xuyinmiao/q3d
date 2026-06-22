import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

console = Console()

def _infer_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _to_input_tensor(value, inputs):
    if torch.is_tensor(value):
        return value.to(device=inputs.device, dtype=inputs.dtype)
    return torch.as_tensor(value, device=inputs.device, dtype=inputs.dtype)

def _clamp_inputs(inputs, clip_min=0.0, clip_max=1.0):
    lower = _to_input_tensor(clip_min, inputs)
    upper = _to_input_tensor(clip_max, inputs)
    return torch.max(torch.min(inputs, upper), lower)

def _project_linf(x_adv, inputs, epsilon, clip_min=0.0, clip_max=1.0):
    eps = _to_input_tensor(epsilon, inputs)
    x_adv = torch.max(torch.min(x_adv, inputs + eps), inputs - eps)
    return _clamp_inputs(x_adv, clip_min, clip_max)

def _to_tanh_space(inputs, clip_min=0.0, clip_max=1.0):
    lower = _to_input_tensor(clip_min, inputs)
    upper = _to_input_tensor(clip_max, inputs)
    scaled = (inputs - lower) / (upper - lower).clamp_min(1e-12)
    scaled = scaled.clamp(1e-6, 1 - 1e-6)
    return torch.atanh(scaled * 2 - 1)

def _from_tanh_space(w, reference_inputs, clip_min=0.0, clip_max=1.0):
    lower = _to_input_tensor(clip_min, reference_inputs)
    upper = _to_input_tensor(clip_max, reference_inputs)
    scaled = 0.5 * (torch.tanh(w) + 1)
    return lower + scaled * (upper - lower)

# ===================== 攻击方法实现 =====================
def fgsm_attack(model, inputs, labels, epsilon, clip_min=0.0, clip_max=1.0):
    """FGSM Attack"""
    x_adv = inputs.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = F.cross_entropy(logits, labels)
    model.zero_grad(set_to_none=True)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.sign()
    x_adv = _project_linf(x_adv, inputs, epsilon, clip_min, clip_max)
    return x_adv.detach()

def bim_attack(model, inputs, labels, epsilon, alpha=0.01, iterations=10,
               clip_min=0.0, clip_max=1.0):
    """Basic Iterative Method (BIM) Attack"""
    x_adv = inputs.clone().detach()
    
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        x_step = x_adv + alpha * x_adv.grad.sign()
        x_adv = _project_linf(x_step, inputs, epsilon, clip_min, clip_max).detach()
    
    return x_adv

def mim_attack(model, inputs, labels, epsilon, alpha=0.01, iterations=10, decay=1.0,
               clip_min=0.0, clip_max=1.0):
    """Momentum Iterative Method (MIM) Attack"""
    x_adv = inputs.clone().detach()
    momentum = torch.zeros_like(inputs)
    
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()

        grad = x_adv.grad
        grad_norm = grad.abs().flatten(1).sum(dim=1).view(-1, 1, 1, 1) + 1e-12
        momentum = decay * momentum + grad / grad_norm
        
        x_step = x_adv + alpha * momentum.sign()
        x_adv = _project_linf(x_step, inputs, epsilon, clip_min, clip_max).detach()
    
    return x_adv

def pgd_attack(model, inputs, labels, epsilon, alpha=0.01, iterations=10,
               random_start=True, clip_min=0.0, clip_max=1.0):
    """Projected Gradient Descent (PGD) Attack"""
    x_adv = inputs.clone().detach()
    
    if random_start:
        eps = _to_input_tensor(epsilon, inputs)
        x_adv = x_adv + torch.empty_like(x_adv).uniform_(-1.0, 1.0) * eps
        x_adv = _clamp_inputs(x_adv, clip_min, clip_max).detach()
    
    for _ in range(iterations):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, labels)
        model.zero_grad(set_to_none=True)
        loss.backward()
        
        x_step = x_adv + alpha * x_adv.grad.sign()
        x_adv = _project_linf(x_step, inputs, epsilon, clip_min, clip_max).detach()
    
    return x_adv

def cw_attack(model, inputs, labels, epsilon, c=1.0, iterations=100, learning_rate=0.01,
              clip_min=0.0, clip_max=1.0):
    """Carlini & Wagner (C&W) L-infinity Attack"""
    batch_size = inputs.size(0)
    
    w = torch.zeros_like(inputs, requires_grad=True)
    w.data = _to_tanh_space(inputs, clip_min, clip_max)
    
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    
    for _ in range(iterations):
        optimizer.zero_grad()
        
        x_adv = _from_tanh_space(w, inputs, clip_min, clip_max)
        logits = model(x_adv)
        
        # 计算C&W损失函数
        # f(x) = max(Z(x)_y - max_{i≠y} Z(x)_i, -κ)
        # 其中y是真实类别，我们希望降低真实类别的logit
        real_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # 获取除真实类别外的最大logit
        other_logits = logits.clone()
        other_logits.scatter_(1, labels.unsqueeze(1), float('-inf'))
        other_max_logits = other_logits.max(1)[0]
        
        # C&W loss: 鼓励真实类别logit小于其他类别
        # 对于untargeted攻击，我们希望 other_max > real
        f_loss = torch.clamp(real_logits - other_max_logits, min=0.0)
        
        # L-infinity扰动损失
        perturbation = (x_adv - inputs).view(batch_size, -1)
        linf_loss = torch.norm(perturbation, p=float('inf'), dim=1)
        
        # 总损失
        loss = (c * f_loss + linf_loss).mean()
        
        loss.backward()
        optimizer.step()
        
        # 在每次迭代后投影到L-infinity球内
        with torch.no_grad():
            x_adv_temp = _from_tanh_space(w, inputs, clip_min, clip_max)
            x_adv_proj = _project_linf(x_adv_temp, inputs, epsilon, clip_min, clip_max)
            # 更新w以对应投影后的x_adv
            w.data = _to_tanh_space(x_adv_proj, clip_min, clip_max)
    
    # 最终对抗样本
    x_adv = _from_tanh_space(w, inputs, clip_min, clip_max)
    x_adv = _project_linf(x_adv, inputs, epsilon, clip_min, clip_max).detach()
    
    return x_adv

def generate_adversarial_loader(model, loader, epsilon, attack_type='fgsm', model_name="Model",
                                clip_min=0.0, clip_max=1.0, **attack_params):
    """Generate adversarial examples using specified attack method for a specific model"""
    if epsilon == 0.0:
        console.print(f"[yellow]ε = 0.0 — reuse original validation loader (no perturbation).[/yellow]")
        return loader

    device = _infer_device(model)
    model = model.to(device)
    model.eval()
    adv_images, adv_labels = [], []

    with Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(
            f"Generating {attack_type.upper()} adv for {model_name} (ε={epsilon:.3f})", 
            total=len(loader)
        )
        
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            if attack_type == 'fgsm':
                perturbed = fgsm_attack(model, inputs, labels, epsilon, clip_min=clip_min, clip_max=clip_max)
            elif attack_type == 'bim':
                perturbed = bim_attack(model, inputs, labels, epsilon, clip_min=clip_min, clip_max=clip_max, **attack_params)
            elif attack_type == 'mim':
                perturbed = mim_attack(model, inputs, labels, epsilon, clip_min=clip_min, clip_max=clip_max, **attack_params)
            elif attack_type == 'pgd':
                perturbed = pgd_attack(model, inputs, labels, epsilon, clip_min=clip_min, clip_max=clip_max, **attack_params)
            elif attack_type == 'cw':
                perturbed = cw_attack(model, inputs, labels, epsilon, clip_min=clip_min, clip_max=clip_max, **attack_params)
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")

            adv_images.append(perturbed.detach().cpu())
            adv_labels.append(labels.detach().cpu())
            progress.update(task, advance=1)

    adv_dataset = TensorDataset(torch.cat(adv_images), torch.cat(adv_labels))
    return DataLoader(adv_dataset, batch_size=loader.batch_size, shuffle=False)
