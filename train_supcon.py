import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from supcon import SupConLoss
from pk_sampler import PKSampler
from projection_head import ProjectionHead
from cath_dataset import Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

import numpy as np
from torch_geometric.nn import GINConv,global_mean_pool,BatchNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

TRAIN = False
data = Dataset(root='data/cath', min_fold_count=10, radius=8.0, max_len=600)
label = []
for d in data:
    label.append(int(d.y))
idx = np.arange(len(data))
y = np.array(label)

print(f"dataset has {len(data)} graphs with {data.num_classes} classes and {data.num_node_features} node features")

train_idx, test_idx = train_test_split(idx,test_size=0.2,random_state=42,stratify=y)
train_idx, val_idx = train_test_split(train_idx,test_size=0.111,random_state=42,stratify=y[train_idx])

train = data[train_idx]
val = data[val_idx]
test = data[test_idx]

train_label = []
for d in train:
    train_label.append(int(d.y))
train_label = np.array(train_label)

sampler = PKSampler(train_label,P=32,K=4,num_batches = 700)
train_loader = DataLoader(dataset=train,batch_sampler=sampler)
test_loader  = DataLoader(dataset=test,  batch_size=64, shuffle=False)
full_train_loader = DataLoader(train, batch_size=64, shuffle=False, num_workers=0)
full_test_loader  = DataLoader(test,  batch_size=64, shuffle=False, num_workers=0)

print(f"folds in sampler: {len(sampler.folds)}")
print(f"folds in dataset: {data.num_classes}")
print(f"folds missing from training: {data.num_classes - len(sampler.folds)}")

class GNNEncoder(nn.Module):
    def __init__(self,in_dim,hid_dim=64):
        super(GNNEncoder,self).__init__()
        self.nn1 = nn.Sequential(
            nn.Linear(in_dim,hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,hid_dim)
        )
        self.conv1 = GINConv(self.nn1)
        self.bn1 = BatchNorm(hid_dim)

        self.nn2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.conv2 = GINConv(self.nn2)
        self.bn2 = BatchNorm(hid_dim)

        self.nn3 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim)
        )
        self.conv3 = GINConv(self.nn3)
        self.bn3 = BatchNorm(hid_dim)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self,x,edge_index,batch):
        x = F.relu(self.bn1(self.conv1(x,edge_index)))
        p1 = global_mean_pool(x,batch)
        x = self.dropout(x)
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x,edge_index)))
        p2 = global_mean_pool(x,batch)
        x = self.dropout(x)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x,edge_index)))
        p3 = global_mean_pool(x,batch)
        f = torch.cat((p1,p2,p3),1)
        return f

encoder = GNNEncoder(in_dim=data.num_node_features,hid_dim=128).to(device)
proj_head = ProjectionHead(input_dim=384,hidden_dim=256,output_dim=128 ).to(device)
criterion = SupConLoss(temperature = 0.1)
optimizer = optim.AdamW(list(encoder.parameters())+list(proj_head.parameters()),lr=0.001,weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=400,eta_min=1e-5)

def augment(batch):
    x = batch.x.clone()
    mask = torch.rand_like(x) < 0.15
    x[mask] = 0
    return x


if TRAIN:
    print("Contrastive Learning")
    for epoch in range(400):
        encoder.train()
        proj_head.train()
        lossT = 0
        batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x1 = augment(batch)
            x2 = augment(batch)
            h1 = encoder(x1, batch.edge_index, batch.batch)
            h2 = encoder(x2, batch.edge_index, batch.batch)
            z1 = proj_head(h1)
            z2 = proj_head(h2)
            z = torch.cat([z1,z2],0)
            y = torch.cat([batch.y.squeeze(),batch.y.squeeze()],dim=0)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            lossT += loss.item()
            batches += 1
        scheduler.step()
        print(f"Epoch {epoch}, Loss {lossT / batches}")

    torch.save(encoder.state_dict(), './encoder.pth')

else:
    encoder.load_state_dict(torch.load('./encoder.pth'))

print("Linear Classification")

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.lin1 = nn.Linear(in_features=384,out_features=128)
        self.lin2 = nn.Linear(in_features=128,out_features=128)
        self.lin3 = nn.Linear(in_features=128,out_features=data.num_classes)
    def forward(self,x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
linear = Classifier().to(device)
optimizer_lin = optim.AdamW((list(encoder.parameters()) + list(linear.parameters())),lr =1e-4,weight_decay=1e-4)
criterion_lin = nn.CrossEntropyLoss()

for epoch in range(200):
    encoder.train()
    linear.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in full_train_loader:
        batch = batch.to(device)

        optimizer_lin.zero_grad()

        # forward THROUGH encoder
        emb = encoder(batch.x, batch.edge_index, batch.batch)

        # forward THROUGH classifier
        logits = linear(emb)

        loss = criterion_lin(logits, batch.y)

        loss.backward()
        optimizer_lin.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)

    acc = correct / total
    print(f"Epoch {epoch} | Loss {total_loss/len(full_train_loader):.4f} | Train Acc {acc:.4f}")

encoder.eval()
linear.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch in full_test_loader:
        batch = batch.to(device)

        emb = encoder(batch.x, batch.edge_index, batch.batch)
        logits = linear(emb)

        preds = logits.argmax(dim=1)

        correct += (preds == batch.y).sum().item()
        total += batch.y.size(0)

print("Test Acc:", correct / total)



































