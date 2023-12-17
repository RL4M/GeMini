import numpy as np
import math
import torch
import torch.nn as nn
from einops import rearrange


from transformers import set_seed
set_seed(2023)
"""
Code from the DICE paper 
https://github.com/wjdghks950/Methods-for-Numeracy-Preserving-Word-Embeddings/blob/master/dice_embedding.ipynb
"""
class DICE:
    '''
    DICE class turns numbers into their respective DICE embeddings

    Since the cosine function decreases monotonically between 0 and pi, simply employ a linear mapping
    to map distances s_n \in [0, |a-b|] to angles \theta \in [0, pi]
    '''
    def __init__(self, d=768, min_bound=0, max_bound=100, norm="l2"):
        self.d = d
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.norm = norm  # Restrict x and y to be of unit length
        self.M = torch.normal(0, 1, size=(self.d, self.d))
        self.Q, self.R = torch.linalg.qr(self.M, mode="complete")  # QR decomposition for orthonormal basis, Q
        self.Q = self.Q.unsqueeze(0)

    def __linear_mapping(self, num):
        '''Eq. (4) from DICE'''
        norm_diff = num / torch.abs(torch.tensor(self.min_bound,device=num.device) - torch.tensor(self.max_bound,device=num.device) )
        theta = norm_diff * torch.tensor(math.pi, device=norm_diff.device)
        return theta

    def make_dice(self, num):
        b = num.size(0)
        # self.Q = self.Q.to(num.device)
        self.Q = rearrange(self.Q, 'b d1 d2 -> b d2 d1')
        # print(self.Q)

        theta = self.__linear_mapping(num)

        polar_coord = torch.concat([(torch.sin(theta) ** (dim - 1) * torch.cos(theta)).unsqueeze(2) if dim < self.d else (torch.sin(theta) ** (self.d)).unsqueeze(2) for dim
         in range(1, self.d + 1)], dim=-1)

        dice = torch.matmul(polar_coord, self.Q.to(polar_coord.device))  # DICE-D embedding for `num`

        return dice


"""
Implementation of Picewise linear embedding https://arxiv.org/pdf/2203.05556.pdf
"""

def get_bin_feature(x, min_bound=-1000, max_bound=1000):
    n_bins = max_bound - min_bound
    if type(x) == list:
        x= np.array(x)
    else:
        x = np.array([x])

    bin_positives = x - min_bound

    bin_feature = np.concatenate([np.concatenate([np.ones([1, int(num)]), \
                                              np.array([[(num - int(num))]]),
                                              np.zeros([1, n_bins - int(num) - 1],)],
                                             axis=1) if int(num) < n_bins else np.ones([1, n_bins])
                                for num in bin_positives], axis=0)

    return bin_feature


class bin_feature(nn.Module):
    def __init__(self, d=768, min_bound=-1000, max_bound=1000) -> None:
        super().__init__()
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.n_bins = max_bound - min_bound

        self.layers = nn.Sequential(nn.Linear(self.n_bins,d), nn.ReLU())

    def forward(self, x):
        b, n_feature = x.size()

        bin_positives = x - torch.tensor(self.min_bound,device=x.device)
        bin_feature = torch.concat([ torch.concat([torch.ones([1,int(num.detach().item())], device=bin_positives.device), \
                                         (num-num.long()).view(1,1),
                                    torch.zeros([1,self.n_bins - int(num.detach().item())-1], device=bin_positives.device) ], dim=1) if num.long() < self.n_bins else torch.ones([1,self.n_bins], device=bin_positives.device) for num in bin_positives.flatten()],dim=0).view(b,n_feature, self.n_bins)

        return self.layers(bin_feature)