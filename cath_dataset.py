import pandas as pd
from pathlib import Path
from torch_geometric.data import InMemoryDataset
from building_graph import build_graph, parse_pdb

def cath_label(label_path):
    rows = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or len(line) == 0 :
                continue
            part = line.split()
            rows.append((part[0], part[1]+"."+part[2]+"."+part[3]))

    return rows

result = cath_label("data/cath/cath-domain-list.txt")
df = pd.DataFrame(result, columns=["domain_id", "fold_label"])
df.to_csv("data/cath/labels.tsv", sep="\t", index=False)

"""print(f"total rows: {len(result)}")
print(f"first 5:    {result[:5]}")

# count unique folds
folds = set(fold for _, fold in result)
print(f"unique folds: {len(folds)}")"""


df = pd.read_csv("data/cath/labels.tsv", sep="\t")
print(f"number of rows: {len(df)}")

pdb_dir = Path("data/cath/dompdb")
select = {p.name for p in pdb_dir.iterdir()}
print(f"number of rows: {len(select)}")

filtered = df[df["domain_id"].isin(select)]
print(f"after filtering: {len(filtered)} rows")

counts = filtered["fold_label"].value_counts()
print(f"unique folds:              {len(counts)}")
print(f"folds with >= 5 examples:  {(counts >= 5).sum()}")
print(f"folds with >= 10 examples: {(counts >= 10).sum()}")
print(f"median fold size:          {int(counts.median())}")

Path("data/cath/raw").mkdir(parents=True, exist_ok=True)
filtered.to_csv("data/cath/raw/labels_s40.tsv", sep="\t", index=False)
print("saved labels_s40.tsv")

class Dataset(InMemoryDataset):
    def __init__(self,root,min_fold_count=10,radius=8.0,max_len=600,transform=None):
        self.min_fold_count = min_fold_count
        self.radius = radius
        self.max_len = max_len
        super(Dataset, self).__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["labels_s40.tsv"]

    @property
    def processed_file_names(self):
        return [f"cath_min{self.min_fold_count}_r{int(self.radius)}_m{self.max_len}.pt"]

    def process(self):
        # 1. load
        df = pd.read_csv(Path(self.raw_dir) / "labels_s40.tsv", sep="\t")

        # 2. filter
        counts = df["fold_label"].value_counts()
        keep_folds = counts[counts >= self.min_fold_count].index
        df = df[df["fold_label"].isin(keep_folds)].reset_index(drop=True)
        print(f"after filter: {len(df)} domains, {df['fold_label'].nunique()} folds")

        # 3. fold -> int
        unique_folds = sorted(df["fold_label"].unique())
        fold_to_int = {f: i for i, f in enumerate(unique_folds)}
        df["fold_int"] = df["fold_label"].map(fold_to_int)

        # 4. pdb dir
        pdb_dir = Path(self.root) / "dompdb"

        # 5. build graphs
        data_list = []
        for i, row in df.iterrows():
            path = pdb_dir / row["domain_id"]
            try:
                coords, aa = parse_pdb(str(path))
            except Exception:
                continue
            if len(aa) < 20 or len(aa) > self.max_len:
                continue
            try:
                g = build_graph(coords, aa,
                                label=int(row["fold_int"]),
                                radius=self.radius)
                data_list.append(g)
            except Exception:
                continue
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(df)}  kept {len(data_list)}")

        print(f"done: kept {len(data_list)}")


        self.save(data_list, self.processed_paths[0])

if __name__ == "__main__":
        ds = Dataset(root="data/cath", min_fold_count=10,
                         radius=8.0, max_len=600)
        print(ds)
        print(f"num classes:  {ds.num_classes}")
        print(f"feature dim:  {ds[0].x.shape[1]}")
        print(f"sample graph: {ds[0]}")























