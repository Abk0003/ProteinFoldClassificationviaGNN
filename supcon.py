from cmath import exp

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self,temperature=1.0):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self,z,y):
        B = z.size(0)
        S = (z@z.T)/self.temperature
        S = S - S.max(dim=1, keepdim=True).values.detach()
        same = (y.view(-1, 1) == y.view(1, -1))
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask = same & ~eye
        exp_S = torch.exp(S) * ~eye
        log_denom = torch.log(torch.sum(exp_S,dim=1,keepdim=True))
        log_prob = S - log_denom
        count = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask*log_prob).sum(dim=1)/(count + 1e-12)
        valid = count>0
        loss = -mean_log_prob[valid].mean()
        return loss

if __name__ == "__main__":
    z = F.normalize(torch.randn(8,64),dim=1)
    y = torch.tensor([0,0,1,1,2,2,3,3])
    loss = SupConLoss()
    res = loss(z,y)
    print(res)










