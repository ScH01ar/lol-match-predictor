import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 设置随机种子
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# 创建保存图片的目录
if not os.path.exists('plots'):
    os.makedirs('plots')

# 重定向print输出到文件
class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

sys.stdout = Logger()

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 特征工程函数
def feature_engineering(df):
    # 避免除以0
    epsilon = 1e-6
    
    # KDA 相关
    df['kda'] = (df['kills'] + df['assists']) / (df['deaths'].clip(lower=1))
    df['kills_participation'] = df['kills'] / (df['kills'] + df['assists'] + epsilon) # 击杀占比
    
    # 伤害转化率
    df['damage_to_champ_ratio'] = df['totdmgtochamp'] / (df['totdmgdealt'] + epsilon)
    df['magic_damage_ratio'] = df['magicdmgtochamp'] / (df['totdmgtochamp'] + epsilon)
    df['phys_damage_ratio'] = df['physdmgtochamp'] / (df['totdmgtochamp'] + epsilon)
    df['true_damage_ratio'] = df['truedmgtochamp'] / (df['totdmgtochamp'] + epsilon)
    
    # 承伤能力
    df['tankiness'] = df['totdmgtaken'] / (df['deaths'].clip(lower=1))
    
    # 视野控制
    df['vision_score'] = df['wardsplaced'] + df['wardskilled']
    
    return df

# 读取数据
print("Loading data...")
train_df = pd.read_csv('train.csv.zip')
test_df = pd.read_csv('test.csv.zip')

# 删除无关列
train_df = train_df.drop(['id', 'timecc'], axis=1)
test_df = test_df.drop(['id', 'timecc'], axis=1)

# 应用特征工程
print("Applying feature engineering...")
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# 数据归一化 (Max Normalization)
print("Applying Max Normalization...")
for col in train_df.columns:
    if col == 'win':
        continue
    max_val = train_df[col].max()
    if max_val != 0:
        train_df[col] /= max_val
        if col in test_df.columns:
            test_df[col] /= max_val

# 分离特征和标签
y = train_df['win'].values
X = train_df.drop(['win'], axis=1).values
X_test = test_df.values

# 自定义 Dataset
class LolDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        else:
            return self.x[idx]

# --- 改进的模型结构：ResNet-18 1D ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, 1)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: [batch, features] -> [batch, 1, features]
        x = x.unsqueeze(1)
        
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 划分验证集
val_size = 1000
X_train = X[:-val_size]
y_train = y[:-val_size]
X_val = X[-val_size:]
y_val = y[-val_size:]

train_dataset = LolDataset(X_train, y_train)
val_dataset = LolDataset(X_val, y_val)
test_dataset = LolDataset(X_test)

# DataLoader 配置
BATCH_SIZE = 256
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

model = ResNet18().to(device)
print("Model structure: ResNet-18 (1D)")

# 训练配置
EPOCH_NUM = 50
learning_rate = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5) # 学习率衰减

# 记录训练过程
train_losses = []
val_accuracies = []
best_acc = 0.0

# 训练循环
print("Start training...")
for epoch in range(EPOCH_NUM):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_epoch_loss = running_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    
    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    val_accuracies.append(val_acc)
    
    print(f"Epoch: {epoch}, Loss: {avg_epoch_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 更新学习率
    scheduler.step(val_acc)
    
    # 保存最佳模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved with accuracy: {best_acc:.4f}")

print(f"Training finished. Best Validation Accuracy: {best_acc:.4f}")

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.title('Validation Accuracy')
plt.legend()

plt.savefig('plots/training_curve_resnet_torch.png')
plt.close()

# 加载最佳模型进行预测
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
        all_preds.extend(preds)

# 保存结果
pd.DataFrame({'win': all_preds}).to_csv('submission_resnet_torch.csv', index=None)
os.system('zip submission_resnet_torch.zip submission_resnet_torch.csv')
print("Submission file saved as submission_resnet_torch.zip")
