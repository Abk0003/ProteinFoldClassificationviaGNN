import torch
import numpy as np

def tree_aff(i,j):
    parti = i.split(".")
    partj = j.split(".")
    depth = len(parti)

    common = 0
    for p1, p2 in zip(parti, partj):
        if p1  == p2:
            common += 1
        else:
            break

    return depth - common

def affinity(folds,a=0.7):
    B = len(folds)
    parts = np.array([s.split(".") for s in folds])
    depth = parts.shape[1]

    level_match = np.stack([parts[:,d:d+1]==parts[:,d:d+1].T for d in range(depth)])
    prefix_match = np.cumprod(level_match,axis=0)
    shared = prefix_match.sum(axis=0).astype(float)
    d_tree = depth - shared
    A = np.exp(-a*d_tree)
    np.fill_diagonal(A,0)
    A = torch.tensor(A,dtype=torch.float)
    return A







