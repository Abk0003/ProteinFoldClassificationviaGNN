import urllib.request, os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from Bio.PDB import PDBParser
from torch_geometric.data import Data

os.makedirs("pdbs", exist_ok=True)
urllib.request.urlretrieve(
    "https://files.rcsb.org/download/1UBQ.pdb",
    "pdbs/1ubq.pdb"
)

amino = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
letter = "ACDEFGHIKLMNPQRSTVWY"
index = {aa: i for i, aa in enumerate(letter)}
numin = 20

def parse_pdb(pdb_path,chain_id=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(chain_id, pdb_path)
    model = next(structure.get_models())
    coord, seq = [], []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for residue in chain:
            if residue.id[0] != " ": #exclude water/ligands
                continue
            resname = residue.get_resname()
            if resname not in amino: #exclude non-standard
                continue
            if "CA" not in residue:
                continue
            coord.append(residue["CA"].get_coord())
            seq.append(amino[resname])
        if chain_id is not None:
            break
    return np.array(coord), seq


coords, seq = parse_pdb("pdbs/1ubq.pdb", chain_id="A")
""" check for correct parsing 
print(f"length: {len(aa)}")              # should be 76
print(f"sequence: {''.join(aa)}")
print(f"coords shape: {coords.shape}")   # (76, 3)
print(f"first CA at: {coords[0]}")       # something like [27.4, 24.4, 5.2] """

def build_graph(coords, seq, label, radius=8.0):
    N = len(seq)
    assert coords.shape == (N, 3)

    # Node features: one-hot amino acid identity, 21-dim
    tok = torch.tensor([index.get(a, numin) for a in seq], dtype=torch.long)
    x = F.one_hot(tok, num_classes=21).float()

    # Sequential edges: i <-> i+1, bidirectional
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    seq_edges = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ], dim=0)
    seq_type = torch.zeros(seq_edges.size(1), dtype=torch.long)

    # Spatial edges: CA pairs within radius, excluding i, i±1
    c = torch.tensor(coords, dtype=torch.float)
    dist = torch.cdist(c, c)
    idx = torch.arange(N)
    mask = (dist < radius) & ((idx.view(-1, 1) - idx.view(1, -1)).abs() > 1)
    sp_src, sp_dst = mask.nonzero(as_tuple=True)
    spatial_edges = torch.stack([sp_src, sp_dst], dim=0)
    sp_type = torch.ones(spatial_edges.size(1), dtype=torch.long)

    edge_index = torch.cat([seq_edges, spatial_edges], dim=1)
    edge_type  = torch.cat([seq_type, sp_type])

    return Data(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        y=torch.tensor([label], dtype=torch.long),
        num_nodes=N,
    )


g = build_graph(coords, seq, label=0, radius=8.0)
"""print(g)
print(f"nodes:              {g.num_nodes}")
print(f"node feature dim:   {g.x.shape[1]}")
print(f"total edges:        {g.num_edges}")
print(f"sequential edges:   {(g.edge_type == 0).sum().item()}")
print(f"spatial edges:      {(g.edge_type == 1).sum().item()}")

# distance diagnostic on spatial edges
sp_mask = (g.edge_type == 1)
src, dst = g.edge_index[:, sp_mask]
distances = np.linalg.norm(coords[src.numpy()] - coords[dst.numpy()], axis=1)
print(f"spatial dist mean:  {distances.mean():.2f} Å")
print(f"spatial dist max:   {distances.max():.2f} Å")
print(f"avg neighbors/node: {g.num_edges / g.num_nodes:.1f}")"""




