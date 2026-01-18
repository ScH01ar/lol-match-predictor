import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


def feature_engineering(df):
    epsilon = 1e-6
    df["kda"] = (df["kills"] + df["assists"]) / (df["deaths"].clip(lower=1))
    df["kills_participation"] = df["kills"] / (df["kills"] + df["assists"] + epsilon)
    df["damage_to_champ_ratio"] = df["totdmgtochamp"] / (df["totdmgdealt"] + epsilon)
    df["magic_damage_ratio"] = df["magicdmgtochamp"] / (df["totdmgtochamp"] + epsilon)
    df["phys_damage_ratio"] = df["physdmgtochamp"] / (df["totdmgtochamp"] + epsilon)
    df["true_damage_ratio"] = df["truedmgtochamp"] / (df["totdmgtochamp"] + epsilon)
    df["tankiness"] = df["totdmgtaken"] / (df["deaths"].clip(lower=1))
    df["vision_score"] = df["wardsplaced"] + df["wardskilled"]
    return df


def max_normalize(train_df, test_df, label_col="win"):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in train_df.columns:
        if col == label_col:
            continue
        max_val = float(train_df[col].max())
        if max_val != 0:
            train_df[col] = train_df[col] / max_val
            if col in test_df.columns:
                test_df[col] = test_df[col] / max_val
    return train_df, test_df


def standardize(train_df, test_df, label_col="win"):
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col in train_df.columns:
        if col == label_col:
            continue
        mean = float(train_df[col].mean())
        std = float(train_df[col].std())
        if std == 0:
            continue
        train_df[col] = (train_df[col] - mean) / std
        if col in test_df.columns:
            test_df[col] = (test_df[col] - mean) / std
    return train_df, test_df


class LolDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.float32)
        if y is None:
            self.y = None
        else:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class MLPClassifier(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18_1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, 1)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
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


@dataclass
class ExperimentConfig:
    name: str
    model_type: str
    use_feature_engineering: bool
    normalization: str
    use_scheduler: bool
    select_best: bool
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42
    val_size: int = 1000


def load_data(use_feature_engineering, normalization):
    train_df = pd.read_csv("train.csv.zip")
    test_df = pd.read_csv("test.csv.zip")

    train_df = train_df.drop(["id", "timecc"], axis=1)
    test_df = test_df.drop(["id", "timecc"], axis=1)

    if use_feature_engineering:
        train_df = feature_engineering(train_df)
        test_df = feature_engineering(test_df)

    if normalization == "max":
        train_df, test_df = max_normalize(train_df, test_df, label_col="win")
    elif normalization == "standard":
        train_df, test_df = standardize(train_df, test_df, label_col="win")
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    y = train_df["win"].values.astype(np.float32)
    X = train_df.drop(["win"], axis=1).values.astype(np.float32)
    X_test = test_df.values.astype(np.float32)
    return X, y, X_test


def split_last(X, y, val_size):
    X_train = X[:-val_size]
    y_train = y[:-val_size]
    X_val = X[-val_size:]
    y_val = y[-val_size:]
    return X_train, y_train, X_val, y_val


def build_model(model_type, num_features, device):
    if model_type == "mlp":
        return MLPClassifier(num_features).to(device)
    if model_type == "resnet18":
        return ResNet18_1D().to(device)
    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    if total == 0:
        return 0.0
    return correct / total


def train_one_experiment(cfg, device, make_submission=False):
    set_seed(cfg.seed)

    X, y, X_test = load_data(cfg.use_feature_engineering, cfg.normalization)
    X_train, y_train, X_val, y_val = split_last(X, y, cfg.val_size)

    train_dataset = LolDataset(X_train, y_train)
    val_dataset = LolDataset(X_val, y_val)
    test_dataset = LolDataset(X_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = build_model(cfg.model_type, num_features=X.shape[1], device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = None
    if cfg.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    train_losses = []
    val_accuracies = []
    best_acc = -1.0
    best_state = None

    print(f"\n=== Experiment: {cfg.name} ===")
    print(f"model={cfg.model_type}, fe={cfg.use_feature_engineering}, norm={cfg.normalization}, "
          f"scheduler={cfg.use_scheduler}, select_best={cfg.select_best}")
    print(f"train_size={len(train_dataset)}, val_size={len(val_dataset)}")

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        avg_loss = running_loss / max(1, len(train_loader))
        train_losses.append(avg_loss)
        val_acc = evaluate_accuracy(model, val_loader, device)
        val_accuracies.append(val_acc)

        print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        if scheduler is not None:
            scheduler.step(val_acc)

        if cfg.select_best and val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    last_acc = val_accuracies[-1] if val_accuracies else 0.0
    selected_acc = best_acc if cfg.select_best else last_acc

    if cfg.select_best and best_state is not None:
        model.load_state_dict(best_state)

    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.title(f"{cfg.name} - Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy", color="orange")
    plt.title(f"{cfg.name} - Val Acc")
    plt.legend()
    plt.savefig(f"plots/{cfg.name}_curve.png")
    plt.close()

    if make_submission:
        all_preds = []
        model.eval()
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
                all_preds.extend(preds.tolist())
        out_csv = f"submission_{cfg.name}.csv"
        out_zip = f"submission_{cfg.name}.zip"
        pd.DataFrame({"win": all_preds}).to_csv(out_csv, index=None)
        os.system(f"zip {out_zip} {out_csv}")
        print(f"Submission file saved as {out_zip}")

    return {
        "name": cfg.name,
        "model": cfg.model_type,
        "feature_engineering": int(cfg.use_feature_engineering),
        "normalization": cfg.normalization,
        "scheduler": int(cfg.use_scheduler),
        "select_best": int(cfg.select_best),
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "val_size": cfg.val_size,
        "val_acc_selected": float(selected_acc),
        "val_acc_last": float(last_acc),
        "val_acc_best": float(best_acc if best_acc >= 0 else last_acc),
    }


def run_ablations(device, seed=42):
    configs = [
        ExperimentConfig(
            name="ablation_0_baseline_mlp",
            model_type="mlp",
            use_feature_engineering=False,
            normalization="max",
            use_scheduler=False,
            select_best=False,
            epochs=50,
            lr=1e-3,
            seed=seed,
        ),
        ExperimentConfig(
            name="ablation_1_plusA_resnet",
            model_type="resnet18",
            use_feature_engineering=False,
            normalization="max",
            use_scheduler=False,
            select_best=False,
            epochs=50,
            lr=1e-3,
            seed=seed,
        ),
        ExperimentConfig(
            name="ablation_2_plusAB_resnet_fe",
            model_type="resnet18",
            use_feature_engineering=True,
            normalization="max",
            use_scheduler=False,
            select_best=False,
            epochs=50,
            lr=1e-3,
            seed=seed,
        ),
        ExperimentConfig(
            name="ablation_3_plusABC_resnet_fe_sched_best",
            model_type="resnet18",
            use_feature_engineering=True,
            normalization="max",
            use_scheduler=True,
            select_best=True,
            epochs=50,
            lr=1e-3,
            seed=seed,
        ),
    ]

    results = []
    for cfg in configs:
        results.append(train_one_experiment(cfg, device=device, make_submission=False))

    results_df = pd.DataFrame(results)
    results_df.to_csv("ablation_results.csv", index=False)
    print("\nAblation results saved to ablation_results.csv\n")
    print(results_df[["name", "val_acc_selected", "val_acc_last", "val_acc_best"]].to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "ablation"])
    parser.add_argument("--model", type=str, default="resnet18", choices=["mlp", "resnet18"])
    parser.add_argument("--fe", type=int, default=1, choices=[0, 1])
    parser.add_argument("--norm", type=str, default="max", choices=["max", "standard"])
    parser.add_argument("--scheduler", type=int, default=1, choices=[0, 1])
    parser.add_argument("--select_best", type=int, default=1, choices=[0, 1])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--make_submission", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def main():
    sys.stdout = Logger()
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "ablation":
        run_ablations(device=device, seed=args.seed)
        return

    cfg = ExperimentConfig(
        name=f"train_{args.model}_fe{args.fe}_norm{args.norm}_sched{args.scheduler}_best{args.select_best}",
        model_type=args.model,
        use_feature_engineering=bool(args.fe),
        normalization=args.norm,
        use_scheduler=bool(args.scheduler),
        select_best=bool(args.select_best),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        val_size=args.val_size,
    )
    train_one_experiment(cfg, device=device, make_submission=bool(args.make_submission))


if __name__ == "__main__":
    main()
