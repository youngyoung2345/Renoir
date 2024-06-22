import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s, m):
        super(ArcFace, self).__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.empty(in_dim, out_dim))

        nn.init.xavier_uniform_(self.W)
        
    def forward(self, x):
        normalized_x = F.normalize(x, p=2, dim=1)
        normalized_W = F.normalize(self.W, p=2, dim=0)
    
        
        cosine = torch.matmul(normalized_x.view(normalized_x.size(0), -1), normalized_W)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        probability = self.s * torch.cos(theta+self.m)
        
        return probability