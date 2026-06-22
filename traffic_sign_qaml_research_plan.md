# 面向自动驾驶交通标志识别的量子经典混合模型鲁棒性研究计划

## 1. 研究题目

面向自动驾驶交通标志识别的量子经典混合模型在视觉对抗攻击下的鲁棒性评估

## 2. 研究背景

自动驾驶视觉系统依赖深度学习模型完成交通标志识别、目标检测、车道线检测等任务。已有研究表明，视觉模型可能受到数字扰动、物理贴纸、局部补丁、光照和视角变化等攻击影响，从而产生误分类或检测失败。

现有项目已经实现了经典模型与量子经典混合模型在 MNIST、Fashion-MNIST、CIFAR-10 和 CIFAR-100 上的对抗鲁棒性评估。该研究计划将项目迁移到自动驾驶视觉子任务，重点研究交通标志识别场景下，量子经典混合模型是否表现出不同于经典模型的鲁棒性特征。

## 3. 研究问题

本研究重点回答以下问题：

1. 在交通标志分类任务中，量子经典混合模型能否保持与经典模型接近的 clean accuracy？
2. 面对白盒数字攻击时，HybridQWideResNet 是否比 WideResNet 更鲁棒？
3. 面向自动驾驶场景的局部补丁、RP2 风格扰动和 EOT 物理变换攻击下，量子经典混合模型是否仍能保持鲁棒性优势？
4. 如果量子混合模型表现出鲁棒性差异，这种差异更可能来自量子特征映射、低维特征压缩，还是模型容量变化？

## 4. 研究假设

在交通标志分类任务中，量子经典混合模型可能通过低维量子特征映射改变决策边界，从而在部分白盒攻击和物理模拟视觉攻击下表现出不同于经典模型的鲁棒性特征。

该假设不预设量子模型一定优于经典模型，而是通过系统实验评估两类模型在不同攻击条件下的表现差异。

## 5. 研究范围

本研究聚焦自动驾驶视觉中的交通标志分类子任务，不直接评估完整自动驾驶系统安全性。

### 5.1 包含内容

- GTSRB 交通标志分类任务
- 经典 WideResNet 与 HybridQWideResNet 对比
- FGSM、PGD、C&W 等白盒数字攻击
- RP2 风格局部扰动
- DARTS 风格定向误分类攻击
- Adversarial Patch 风格局部补丁攻击
- EOT 物理变换仿真，包括旋转、缩放、亮度、模糊和视角扰动

### 5.2 不包含内容

- 不直接评估真实车辆或真实道路系统
- 不做自动驾驶目标检测全链路攻击
- 不做车道线检测或语义分割攻击
- 不做真实世界贴纸部署实验，除非后续具备安全、合规的测试条件

## 6. 数据集设计

### 6.1 主数据集

GTSRB, German Traffic Sign Recognition Benchmark

- 类别数：43 类
- 任务类型：交通标志分类
- 输入尺寸：优先使用 32x32，便于复用当前 WideResNet 结构
- 后续扩展：可尝试 64x64 输入以提高视觉细节保留

### 6.2 可选扩展数据集

LISA Traffic Sign Dataset

用途：

- 跨数据集泛化评估
- 更贴近真实道路图像分布
- 检验模型在不同采集条件下的鲁棒性

## 7. 模型设计

### 7.1 经典基线模型

WideResNet

- 输入：交通标志图像
- 输出：43 类交通标志类别
- 用作经典深度学习基线

### 7.2 量子经典混合模型

HybridQWideResNet

- 前端卷积特征提取与 WideResNet 保持一致
- 分类头前加入量子层
- 量子比特数初始设为 8
- 量子层数初始设为 6
- 输出类别数改为 43

### 7.3 公平对照原则

两类模型应尽量保持：

- 相同训练集和验证集划分
- 相同数据增强策略
- 相同 optimizer 和学习率策略
- 相同训练 epoch
- 相同 batch size
- 相同输入尺寸

核心区别仅保留为是否加入量子层。

## 8. 攻击方法设计

### 8.1 数字白盒攻击

用于建立基础鲁棒性曲线。

攻击方法：

- FGSM
- PGD
- C&W

评估维度：

- 不同 epsilon 下的 robust accuracy
- attack success rate
- confidence drop
- targeted attack 与 untargeted attack 差异

### 8.2 RP2 风格物理扰动攻击

RP2 适合交通标志场景。该攻击通过优化局部、可打印的扰动，使交通标志在不同视角和环境变化下仍被模型误分类。

实验实现方式：

- 限制扰动区域为交通标志内部局部区域
- 加入 EOT 变换
- 比较 WideResNet 与 HybridQWideResNet 的攻击成功率

### 8.3 DARTS 风格定向误分类攻击

该方向关注交通标志识别中的定向误分类，例如将 stop sign 攻击成 speed limit sign。

实验实现方式：

- 选择若干高风险类别对
- 进行 targeted attack
- 统计目标类别成功率

### 8.4 Adversarial Patch 攻击

用于模拟局部贴纸、污渍、广告贴片等局部视觉干扰。

实验变量：

- patch 面积比例
- patch 位置
- patch 透明度
- patch 是否固定
- patch 是否经过 EOT 训练

## 9. EOT 物理变换设计

EOT, Expectation over Transformation, 用于模拟真实道路视觉变化。

推荐变换：

- 旋转：-15 度到 15 度
- 缩放：0.8 到 1.2
- 平移：水平和垂直方向小幅偏移
- 亮度变化：0.7 到 1.3
- 对比度变化：0.8 到 1.2
- 高斯模糊：轻度模糊
- JPEG 压缩：模拟摄像头和传输压缩

## 10. 评估指标

### 10.1 分类性能

- Clean accuracy
- Top-1 accuracy
- Per-class accuracy
- Confusion matrix

### 10.2 鲁棒性指标

- Robust accuracy
- Attack success rate
- Targeted attack success rate
- Accuracy vs epsilon 曲线
- Accuracy vs patch size 曲线
- Accuracy vs EOT strength 曲线

### 10.3 计算成本

- 训练时间
- 单次推理时间
- 单次攻击生成时间
- GPU/CPU 内存占用

## 11. 实验流程

### 第一阶段：代码整理与数据迁移

目标：

- 加入 GTSRB 数据加载
- 将输出类别数改为 43
- 修复彩色图像攻击中的归一化边界问题
- 将重复脚本改造成可配置入口

产出：

- GTSRB 训练脚本
- GTSRB 评估脚本
- 统一配置文件

### 第二阶段：模型训练

目标：

- 训练 WideResNet
- 训练 HybridQWideResNet
- 确认两类模型 clean accuracy 接近

建议判断标准：

- 如果 clean accuracy 差距小于 2%，可直接比较鲁棒性
- 如果差距明显，需要调整训练策略或补充容量对照实验

### 第三阶段：数字攻击评估

目标：

- 复现 FGSM、PGD、C&W
- 绘制 robust accuracy 曲线
- 分析攻击强度变化下的模型差异

产出：

- accuracy vs epsilon 图
- attack success rate 表格
- 不同类别的鲁棒性分析

### 第四阶段：自动驾驶视觉攻击评估

目标：

- 实现 RP2 风格局部扰动
- 实现 patch attack
- 加入 EOT 物理变换
- 分析自动驾驶相关攻击下的模型鲁棒性

产出：

- patch size vs accuracy 图
- EOT strength vs attack success rate 图
- 典型攻击样本可视化

### 第五阶段：结果整理与论文分析

目标：

- 整理图表
- 分析鲁棒性差异
- 讨论限制条件
- 完成论文实验部分初稿

## 12. 时间计划

### 第 1 周

- 阅读并整理当前代码
- 抽取统一训练和评估配置
- 加入 GTSRB 数据集
- 修复归一化攻击边界

### 第 2 周

- 训练 WideResNet
- 训练 HybridQWideResNet
- 记录 clean accuracy 和训练曲线

### 第 3 周

- 完成 FGSM、PGD、C&W 数字攻击评估
- 生成基础鲁棒性曲线和表格

### 第 4 周

- 实现 RP2 风格局部扰动
- 实现 adversarial patch
- 加入 EOT 物理变换

### 第 5 周

- 跑完整对比实验
- 生成所有图表
- 做类别级分析和失败案例分析

### 第 6 周

- 整理实验结论
- 撰写论文实验部分
- 总结限制和未来工作

## 13. 预期结果

可能出现三类结果：

### 13.1 量子混合模型在部分攻击下更鲁棒

说明量子特征映射可能改变了模型局部梯度或决策边界，对某些攻击有抑制作用。

### 13.2 两类模型鲁棒性接近

说明量子层在该任务和该模型规模下没有提供显著鲁棒性收益，但仍可作为负结果进行分析。

### 13.3 量子混合模型 clean accuracy 或鲁棒性更差

需要分析是否由量子层表达能力不足、训练不稳定、特征压缩过强或参数配置不合理导致。

## 14. 风险与应对

### 14.1 训练速度慢

量子层会显著增加训练和攻击耗时，尤其是 PGD、C&W 和 EOT 攻击。

应对：

- 先使用小样本子集调试
- 降低 batch size
- 减少量子层数
- 优先使用 CUDA GPU 完整运行

### 14.2 MacBook Air 不适合完整实验

MacBook Air M5 可以做小规模验证，但完整实验建议使用 NVIDIA CUDA GPU。

应对：

- 本地只做代码调试和小样本验证
- 完整训练和攻击迁移到 GPU 服务器

### 14.3 归一化边界错误影响结论

当前项目中彩色图像攻击函数将样本统一 clamp 到 [0, 1]，但 CIFAR/GTSRB 通常会使用 Normalize，必须修复。

应对：

- 在 normalized space 中计算正确上下界
- 或在 pixel space 中生成攻击后再 normalize 输入模型

### 14.4 对比不公平

如果 HybridQWideResNet clean accuracy 明显低于 WideResNet，鲁棒性比较会失去公平性。

应对：

- 调整训练超参数
- 增加相同 latent dimension 的经典压缩头对照
- 报告 clean accuracy 和 robust accuracy 的共同变化

## 15. 论文贡献点

潜在贡献可以表述为：

1. 将量子经典混合模型鲁棒性评估扩展到自动驾驶交通标志识别任务。
2. 系统比较经典 WideResNet 与 HybridQWideResNet 在数字攻击和物理模拟攻击下的表现。
3. 分析量子层在低维特征映射、决策边界和攻击迁移性方面可能带来的影响。
4. 给出量子经典混合模型用于安全关键视觉任务时的局限性和实验建议。

## 16. 推荐论文结构

1. Introduction
2. Related Work
   - Adversarial attacks in autonomous driving
   - Physical adversarial examples
   - Quantum machine learning robustness
3. Method
   - HybridQWideResNet architecture
   - Attack formulation
   - EOT physical simulation
4. Experiments
   - Dataset and setup
   - Clean accuracy
   - Digital attacks
   - Physical simulation attacks
5. Results and Analysis
6. Limitations
7. Conclusion

## 17. 阶段性最小可行版本

最小可行版本建议如下：

1. 只使用 GTSRB。
2. 只比较 WideResNet 和 HybridQWideResNet。
3. 只实现 FGSM、PGD 和 patch attack。
4. 输入尺寸固定为 32x32。
5. 评估 clean accuracy、robust accuracy 和 attack success rate。
6. 本地先用 1000 到 5000 张样本调试，最终在 GPU 上完整运行。

这个版本最容易形成闭环，也最适合从当前项目快速扩展。
