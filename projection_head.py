import torch
import torch.nn.functional as F
import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(self,input_dim=192,hidden_dim=128,output_dim=64):
        super(ProjectionHead,self).__init__()
        self.linear1 = nn.Linear(input_dim,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = F.normalize(x,p=2,dim=1)
        return x


