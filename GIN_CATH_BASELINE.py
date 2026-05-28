from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm
import torch.nn.functional as F
import networkx as nx
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

from cath_dataset import Dataset
dataset = Dataset(root='data/cath', min_fold_count=10, radius=8.0, max_len=600)

print(dataset)
print(dataset.num_classes)
print(dataset.num_node_features)
print(dataset[0])

torch.manual_seed(42)
y = dataset.y.numpy()
idx = np.arange(len(dataset))


class GNNClassifier(torch.nn.Module):
    def __init__(self):
        super(GNNClassifier,self).__init__()
        self.nn1 = nn.Sequential(
            nn.Linear(dataset.num_node_features, 64,),
            nn.ReLU(),
            nn.Linear(64,64)
        )
        self.conv1 = GINConv(self.nn1)
        self.nn2 = nn.Sequential(
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )
        self.conv2 = GINConv(self.nn2)
        self.nn3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.conv3 = GINConv(self.nn3)
        self.Linear = nn.Linear(192, dataset.num_classes)
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(64)
        self.bn3 = BatchNorm(64)
        self.dropout = nn.Dropout(0.3)

    def forward(self,x,edge_index,batch):
        x1 = self.conv1(x,edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
        p1 = global_add_pool(x1,batch)
        x2 = self.conv2(x1,edge_index)
        x2 = self.bn2(x1)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        p2 = global_add_pool(x2,batch)
        x3 = self.conv3(x2,edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        p3 = global_add_pool(x3,batch)
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        p3 = F.normalize(p3, dim=1)
        f = torch.cat([p1,p2,p3],dim=1)
        f = self.dropout(f)
        x4 = self.Linear(f)
        return x4


def run_one_fold(train_idx, val_idx, test_idx, fold_num):
    train, val, test = dataset[train_idx], dataset[val_idx], dataset[test_idx]
    print(f"\n=== Fold {fold_num} === train:{len(train)} val:{len(val)} test:{len(test)}")

    train_loader = DataLoader(dataset=train, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset=val, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=128, shuffle=True)

    model = GNNClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005)
    criterion = torch.nn.CrossEntropyLoss()
    acc_train = []
    acc_test = []
    for epoch in range(200):
        lossT = 0
        acc = 0
        total = 0
        for data in train_loader:
            data = data.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(data.x, data.edge_index, data.batch)
            loss = criterion(output, data.y)
            loss.backward()
            optimizer.step()

            lossT += loss.item()
            total += data.y.size(0)
            acc += (output.argmax(dim=1) == data.y).sum().item()
        acc_train.append(acc / total)

        with torch.no_grad():
            l = 0
            a = 0
            t = 0
            for data in test_loader:
                model.eval()
                data = data.to(device)
                output = model(data.x, data.edge_index, data.batch)
                loss = criterion(output, data.y)
                l += loss.item()
                t += data.y.size(0)
                a += (output.argmax(dim=1) == data.y).sum().item()
        acc_test.append(a / t)


        print(f"  fold {fold_num} epoch {epoch+1}: train {acc_train[-1]:.4f}  test {acc_test[-1]:.4f}")

    return max(acc_train), max(acc_test), acc_train, acc_test


# === 10-fold stratified CV ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_train_max = []
fold_test_max  = []
all_train_curves = []
all_test_curves  = []

for fold_num, (trainval_idx, test_idx) in enumerate(skf.split(idx, y), start=1):
    # carve a val split out of trainval (same ~11% ratio you used before)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=0.111,
        stratify=y[trainval_idx], random_state=42)

    best_tr, best_te, curve_tr, curve_te = run_one_fold(
        train_idx, val_idx, test_idx, fold_num)
    fold_train_max.append(best_tr)
    fold_test_max.append(best_te)
    all_train_curves.append(curve_tr)
    all_test_curves.append(curve_te)
    print(f"Fold {fold_num}: best train {best_tr:.4f}  best test {best_te:.4f}")

# === Aggregate results ===
train_mean = np.mean(fold_train_max) * 100
train_std  = np.std(fold_train_max)  * 100
test_mean  = np.mean(fold_test_max)  * 100
test_std   = np.std(fold_test_max)   * 100

print("\n" + "=" * 50)
print("10-fold CV results on ENZYMES")
print("=" * 50)
print(f"Per-fold test:  {[f'{a*100:.1f}' for a in fold_test_max]}")
print(f"Train accuracy: {train_mean:.1f} ± {train_std:.1f}")
print(f"Test  accuracy: {test_mean:.1f} ± {test_std:.1f}")


# === Plot mean ± std curves across folds ===
train_curves = np.array(all_train_curves)   # [10, 350]
test_curves  = np.array(all_test_curves)
epochs = np.arange(1, train_curves.shape[1] + 1)

tr_mu, tr_sd = train_curves.mean(axis=0), train_curves.std(axis=0)
te_mu, te_sd = test_curves.mean(axis=0),  test_curves.std(axis=0)

plt.figure(figsize=(8, 5))
plt.plot(epochs, tr_mu, label='Train (mean)', color='C0')
plt.fill_between(epochs, tr_mu - tr_sd, tr_mu + tr_sd, alpha=0.2, color='C0')
plt.plot(epochs, te_mu, label='Test (mean)', color='C1')
plt.fill_between(epochs, te_mu - te_sd, te_mu + te_sd, alpha=0.2, color='C1')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title(f'GIN on ENZYMES — 10-fold CV (test {test_mean:.1f} ± {test_std:.1f})')
plt.legend()
plt.tight_layout()
plt.show()






