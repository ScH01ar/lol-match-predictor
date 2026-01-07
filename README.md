# League of Legends Match Prediction

本项目旨在通过英雄联盟（League of Legends, LoL）对局数据预测比赛胜负。我们在官方提供的 Baseline 基础上，通过引入特征工程和深度卷积神经网络（ResNet-18 1D），显著提升了模型的预测准确率。

## 项目背景

比赛任务是根据提供的 18 万条对局数据（包含击杀、助攻、伤害、视野得分等统计信息），预测玩家所在队伍的胜负情况（Win/Loss，二分类任务）。

## 方法介绍

### 1. Baseline
官方 Baseline 使用了一个简单的多层感知机（MLP）网络。虽然能跑通流程，但模型结构较浅，难以捕捉特征之间复杂的非线性交互关系。

### 2. 本项目改进方案 (Our Approach)

为了提升预测性能，我们从数据和模型两个维度进行了深度优化：

#### A. 特征工程 (Feature Engineering)
除了使用原始的统计数据外，我们结合游戏理解构建了多个关键的高阶特征：
- **KDA**: `(kills + assists) / deaths`，衡量生存与击杀效率的核心指标。
- **Kills Participation (参团率)**: 衡量玩家对团队击杀的贡献度。
- **Damage Efficiency (伤害转化)**: 各类伤害（物理/魔法/真实）占总伤害的比例。
- **Tankiness (承伤能力)**: 承受伤害与死亡次数的比值。
- **Vision Score (视野得分)**: 插眼与排眼的总和，反映视野控制能力。

#### B. 模型架构：ResNet-18 (1D)
我们将模型从简单的 MLP 升级为 **一维 ResNet-18 (ResNet-18 1D)**。
- **卷积特征提取**：利用 1D 卷积层自动提取特征序列中的局部模式。
- **残差连接 (Residual Connections)**：引入残差块（BasicBlock），解决了深层网络训练中的梯度消失问题，允许模型学习更深层次的特征表达。
- **结构细节**：
    - 初始卷积层 + 最大池化。
    - 4 个残差层（Layer1-Layer4），通道数依次增加（64 -> 512）。
    - 全局平均池化 + 全连接层输出预测概率。

#### C. 训练策略
- **数据预处理**：使用最大值归一化（Max Normalization）将所有特征缩放到 [0, 1] 区间，加速模型收敛。
- **损失函数**：`BCEWithLogitsLoss`（二元交叉熵损失）。
- **优化器**：`Adam`，初始学习率 0.001。
- **学习率调度**：`ReduceLROnPlateau`，当验证集准确率不再提升时自动降低学习率。
- **验证集划分**：从训练集中预留最后 1000 条数据作为验证集，用于监控模型性能并保存最佳权重。

## 快速开始

### 环境依赖
- Python 3.x
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### 运行训练
直接运行 `train_improved.py` 即可开始训练、验证并生成提交文件：

```bash
python train_improved.py
```

### 输出文件
- `submission_resnet_torch.csv`: 预测结果文件。
- `submission_resnet_torch.zip`: 打包好的提交文件。
- `training_log.txt`: 训练过程日志。
- `plots/`: 包含训练 Loss 和验证 Accuracy 的变化曲线图。
- `best_model.pth`: 验证集准确率最高的模型权重。
