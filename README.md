# QAML_Rob - Quantum-Augmented Machine Learning for Adversarial Robustness

## 项目简介

本项目实现了量子增强的卷积神经网络，并对其在对抗样本攻击下的鲁棒性进行了系统评估。项目包含了对四个数据集（MNIST、Fashion-MNIST、CIFAR-10、CIFAR-100）的完整实验流程。

## 项目结构

```
QAML_Rob/
├── run_py/                           # 主要代码目录
│   ├── network.py                    # 网络架构定义（CNN、QCNN、WideResNet、QWideResNet）
│   ├── attack.py                     # 对抗攻击实现（FGSM、BIM、MIM、PGD、C&W）
│   ├── circuit_visualization.py      # 量子电路可视化工具
│   ├── convert_results_to_excel.py   # 结果转换为Excel表格
│   │
│   ├── quantum_train_*.py            # 量子增强模型训练脚本
│   ├── classical_train_*.py          # 经典模型训练脚本
│   ├── attack_eval_*_pb.py          # 扰动预算评估脚本
│   └── attack_eval_*_it.py          # 迭代次数评估脚本
│
├── data_mnist/                       # MNIST 数据集
├── data_fmnist/                      # Fashion-MNIST 数据集
├── data_cifar10/                     # CIFAR-10 数据集
├── data_cifar100/                    # CIFAR-100 数据集
│
├── model_history_mnist/              # MNIST 实验结果
├── model_history_fmnist/             # Fashion-MNIST 实验结果
├── model_history_cifar10/            # CIFAR-10 实验结果
└── model_history_cifar100/           # CIFAR-100 实验结果
```

## 环境依赖

### 核心依赖
- Python 3.8+
- PyTorch 1.12+
- PennyLane 0.30+ (量子机器学习框架)
- torchvision
- numpy
- matplotlib
- pandas
- openpyxl
- rich (终端美化输出)

详见 `requirements.txt`

## 模型架构

### 数据集与模型配置

| 数据集 | 类别数 | 输入尺寸 | 经典模型 | 量子模型 | 量子层数 | 量子比特数 | 潜在维度 |
|--------|--------|----------|----------|----------|----------|------------|----------|
| MNIST | 10 | 28×28×1 | CNN | HybridQCNN | 3 | 8 | 8 |
| Fashion-MNIST | 10 | 28×28×1 | CNN | HybridQCNN | 5 | 8 | 8 |
| CIFAR-10 | 10 | 32×32×3 | WideResNet | QWideResNet | 6 | 8 | 8 |
| CIFAR-100 | 100 | 32×32×3 | WideResNet | QWideResNet | 6 | 8 | 8 |

### 架构说明

- **CNN/WideResNet**: 经典卷积神经网络基线
- **HybridQCNN/QWideResNet**: 量子增强版本，在分类层前集成变分量子电路

## 对抗攻击方法

项目实现了5种主流对抗攻击方法：

1. **FGSM** (Fast Gradient Sign Method) - 单步梯度攻击
2. **BIM** (Basic Iterative Method) - 迭代FGSM
3. **MIM** (Momentum Iterative Method) - 带动量的迭代攻击
4. **PGD** (Projected Gradient Descent) - 投影梯度下降攻击
5. **C&W** (Carlini & Wagner) - 优化基础的L2+L∞混合攻击

### 统一的攻击参数

```python
# 迭代攻击参数
alpha = 0.01           # 步长
iterations = 10        # 迭代次数

# C&W 攻击参数
c = 1.0               # 平衡因子
learning_rate = 0.1   # 学习率
```

## 使用指南

### 1. 训练模型

```bash
# 训练量子增强模型
python run_py/quantum_train_mnist.py
python run_py/quantum_train_fmnist.py
python run_py/quantum_train_cifar10.py
python run_py/quantum_train_cifar100.py

# 训练经典模型
python run_py/classical_train_mnist.py
python run_py/classical_train_fmnist.py
python run_py/classical_train_cifar10.py
python run_py/classical_train_cifar100.py
```

### 2. 评估对抗鲁棒性

#### 扰动预算评估 (Perturbation Budget)
评估不同ε值下的模型准确率：

```bash
python run_py/attack_eval_mnist_pb.py
python run_py/attack_eval_fmnist_pb.py
python run_py/attack_eval_cifar10_pb.py
python run_py/attack_eval_cifar100_pb.py
```

#### 迭代次数评估 (Iterations)
评估不同迭代次数下的攻击效果：

```bash
python run_py/attack_eval_mnist_it.py
python run_py/attack_eval_fmnist_it.py
python run_py/attack_eval_cifar10_it.py
python run_py/attack_eval_cifar100_it.py
```

### 3. 结果转换

将文本结果转换为Excel表格：

```bash
# 扰动预算结果
python run_py/convert_results_to_excel.py -m pbs

# 迭代次数结果
python run_py/convert_results_to_excel.py -m it

# 指定输入输出文件
python run_py/convert_results_to_excel.py -m pbs \
  -i model_history_cifar10/robustness_vs_pbs_results.txt \
  -o model_history_cifar10/robustness_vs_pbs_results.xlsx
```

## 评估指标

### 扰动预算评估 (pb)
- **epsilon 范围**: 0.0 ~ 0.2 (MNIST/Fashion-MNIST), 0.0 ~ 0.1 (CIFAR-10/100)
- **评估点**: 11个等间隔点
- **固定参数**: iterations=10

### 迭代次数评估 (it)
- **迭代范围**: 0 ~ 10 (步长为2)
- **评估点**: 0, 2, 4, 6, 8, 10
- **固定参数**: ε=0.1 (MNIST/Fashion-MNIST), ε=0.05 (CIFAR-10/100)

## 结果文件

每次评估生成三类文件：

1. **高分辨率图表**: `adv_robustness_vs_pbs.png` / `adv_robustness_vs_iterations.png`
2. **文本结果**: `robustness_vs_pbs_results.txt` / `robustness_vs_iterations_results.txt`
3. **Excel表格**: `robustness_vs_pbs_results.xlsx` / `robustness_vs_iterations_results.xlsx`

## 可视化风格

### 绘图规范
- **线型区分模型**: 实线 (量子增强), 虚线 (经典模型)
- **颜色和标记区分攻击**: 每种攻击方法有独特的颜色和marker
- **图例**: 分层显示，上层为模型类型，下层为攻击方法
- **分辨率**: 400 DPI，适合论文发表

### 攻击方法可视化

| 攻击 | 颜色 | 标记 |
|------|------|------|
| FGSM | 深蓝 | ○ (Circle) |
| BIM | 深橙 | □ (Square) |
| MIM | 深绿 | △ (Triangle Up) |
| PGD | 深红 | ◇ (Diamond) |
| C&W | 深紫 | ▽ (Triangle Down) |

## 主要发现

1. **量子增强效果**: 量子层提升了模型对迭代攻击的鲁棒性
2. **攻击强度比较**: C&W > PGD > MIM > BIM > FGSM
3. **数据集难度**: CIFAR-100 > CIFAR-10 > Fashion-MNIST > MNIST

## 注意事项

1. **CUDA支持**: 建议使用GPU加速训练和评估
2. **内存需求**: CIFAR-100评估需要较大内存，已优化测试样本数为256
3. **时间成本**: C&W攻击最耗时，建议在GPU上运行
4. **可复现性**: 所有随机种子已固定，确保结果可复现

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{qaml_rob_2024,
  title={Quantum-Augmented Machine Learning for Adversarial Robustness},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/QAML_Rob}}
}
```

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：
- Email: your.email@example.com
- GitHub Issues: [项目Issues页面]

---

**最后更新**: 2024年11月
