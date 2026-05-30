from hierarchy import affinity
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalSupCon(nn.Module):
    def __init__(self,a,temperature):
        super(HierarchicalSupCon, self).__init__()
        self.a = a
        self.temperature = temperature

    def forward(self,z,fold):
        B = z.size(0)
        S = (z@z.T)/self.temperature
        S = S - S.max(dim=1,keepdim=True).values.detach()
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        exp_S = torch.exp(S)*~eye
        log_denom = torch.log(torch.sum(exp_S,dim=1,keepdim=True))
        log_prob = S - log_denom
        A = affinity(fold,a = self.a).to(z.device)
        weight_sum = A.sum(dim=1)
        mean_log_prob = (A*log_prob).sum(dim=1)/(weight_sum + 1e-12)
        valid = weight_sum > 0
        return -mean_log_prob[valid].mean()



