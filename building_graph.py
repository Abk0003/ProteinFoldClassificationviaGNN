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

""" Check for correct parsing
coords, aa = parse_pdb("pdbs/1ubq.pdb", chain_id="A")
print(f"length: {len(aa)}")              # should be 76
print(f"sequence: {''.join(aa)}")
print(f"coords shape: {coords.shape}")   # (76, 3)
print(f"first CA at: {coords[0]}")       # something like [27.4, 24.4, 5.2] """

def building_graph(coords,seq,label,radius= 8.0):
    N = len(seq)
    for a in seq:
        emb = torch.tensor(index.get(a,numin), dtype=torch.long)
    x = F.one_hot(emb,N)


