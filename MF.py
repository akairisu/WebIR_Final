import torch.nn as nn
import torch

class MF(nn.Module):
    def __init__(self, n_users, n_items, n_factors=512):
        super().__init__()
        self.user_mat = nn.Parameter(torch.empty(n_users, n_factors))
        self.item_mat = nn.Parameter(torch.empty(n_items, n_factors))
        nn.init.xavier_normal_(self.user_mat.data)
        nn.init.xavier_normal_(self.item_mat.data)
    
    def forward(self, user, pos, neg):
        u = self.user_mat[user]
        i = self.item_mat[pos]
        j = self.item_mat[neg]
        return torch.mul(u, i - j).sum(dim=1)