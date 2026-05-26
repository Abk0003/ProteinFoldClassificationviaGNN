from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
import torch.nn.functional as F
import networkx as nx
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = TUDataset(root='./data/', name='ENZYMES', use_node_attr=True,)

#Information about dataset
print(dataset)                  # ENZYMES(600)
print(dataset.num_classes)      # 6
print(dataset.num_node_features) # 21
print(dataset[0])                # Data(edge_index=[2, 168], x=[37, 21], y=[1])

#Spliting the dataset for testing and training
torch.manual_seed(42)
y = dataset.y.numpy()
idx = list(range(len(dataset)))

train_idx, test_idx = train_test_split(
    idx, test_size=0.1, stratify=y, random_state=42)
train_idx, val_idx  = train_test_split(
    train_idx, test_size=0.111, stratify=y[train_idx], random_state=42)

train, val, test = dataset[train_idx], dataset[val_idx], dataset[test_idx]
print(len(train), len(val), len(test))   # 480 60 60

train_loader = DataLoader(dataset=train,batch_size=32,shuffle=True)
val_loader = DataLoader(dataset=val,batch_size=32,shuffle=True)
test_loader = DataLoader(dataset=test,batch_size=32,shuffle=True)

class GNNClassifier(torch.nn.Module):
    def __init__(self):
        super(GNNClassifier,self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features,64)
        self.conv2 = GCNConv(64,64)
        self.conv3  = GCNConv(64,64)
        self.conv4 = GCNConv(64,64)
        self.linear = torch.nn.Linear(64,6)
        self.bn1 = BatchNorm(64)
        self.bn2 = BatchNorm(64)
        self.bn3 = BatchNorm(64)
        self.bn4 = BatchNorm(64)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self,x,edge_index,batch):
        x1 = F.relu(self.bn1(self.conv1(x,edge_index)))
        x2 = self.conv2(x1,edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
        x3 = self.conv3(x2,edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x4 = self.conv4(x3,edge_index)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x4 = self.dropout(x4)
        x5 = global_mean_pool(x4, batch)
        x5 = self.linear(x5)
        return x5


model = GNNClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001)
criterion = torch.nn.CrossEntropyLoss()
acc_train = []
acc_test = []
for epoch in range(300):
    lossT = 0
    acc = 0
    total = 0
    for  data in train_loader:
        data = data.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(data.x,data.edge_index,data.batch)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        lossT += loss.item()
        total += data.y.size(0)
        acc += (output.argmax(dim=1) == data.y).sum().item()
    acc_train.append(acc/total)
    print(f'Epoch: {epoch+1}, Loss: {lossT/len(train_loader)}, Accuracy: {acc/total:.4f}')

    with torch.no_grad():
        l = 0
        a = 0
        t = 0
        for data in val_loader:
            model.eval()
            data = data.to(device)
            output = model(data.x,data.edge_index,data.batch)
            loss = criterion(output, data.y)
            l += loss.item()
            t += data.y.size(0)
            a += (output.argmax(dim=1) == data.y).sum().item()
    print(f"Epoch: {epoch+1}, Test Accuracy: {a/t:.4f}")
    acc_test.append(a/t)

plt.plot(acc_train, label='Train Accuracy')
plt.plot(acc_test, label= 'Test Accuracy')
plt.show()

print(f"Maximum Accuracy for training : {max(acc_train)})")
print(f"Maximum Accuracy for testing : {max(acc_test)})")














