from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from train_supcon import GNNEncoder
from cath_dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
data = Dataset(root='data/cath', min_fold_count=10, radius=8.0, max_len=600)
encoder = GNNEncoder(in_dim=data.num_node_features,hid_dim=128).to(device)
encoder.load_state_dict(torch.load('./encoder.pth'))

encoder.eval()
@torch.no_grad()
def extract_viz(loader):
    embs, labels = [] , []
    for batch in loader:
        batch = batch.to(device)
        h = encoder(batch.x,batch.edge_index,batch.batch)
        embs.append(h.cpu().numpy())
        labels.append(batch.y.cpu().numpy())
    return np.concatenate(embs), np.concatenate(labels)

x,y = extract_viz(data.full_test_loader)

np.random.seed(42)
selected = []
for fold in np.unique(y):
    fold_idx = np.where(y == fold)[0]
    chosen = np.random.choice(fold_idx,
                              min(50, len(fold_idx)),
                              replace=False)
    selected.extend(chosen)
selected = np.array(selected)
X_sub = x[selected]
y_sub = y[selected]
print(f"subsampled to {len(X_sub)} proteins across {len(np.unique(y_sub))} folds")

# run t-SNE
print("running t-SNE (takes 1-3 minutes)...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    n_iter=1000,
    metric='cosine',
    init='pca',
    random_state=42,
    verbose=1,
)
Z = tsne.fit_transform(X_sub)

# plot
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    Z[:, 0], Z[:, 1],
    c=y_sub,
    cmap='tab20',
    s=8,
    alpha=0.7,
)
ax.set_title(
    f'SupCon GIN Embeddings — CATH test set\n'
    f'({len(np.unique(y_sub))} folds, t-SNE)',
    fontsize=13
)
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig('tsne_supcon.png', dpi=200, bbox_inches='tight')
plt.show()
print("saved tsne_supcon.png")



