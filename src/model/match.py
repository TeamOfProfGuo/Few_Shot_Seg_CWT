# encoding:utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .conv4d import CenterPivotConv4d, Conv4d

conv4_dt = {'cv4': Conv4d, 'red': CenterPivotConv4d}


def MutualMatching(corr4d):
    # mutual matching
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  # parenthesis are important for symmetric output
    return corr4d


class NeighConsensus(torch.nn.Module):
    def __init__(self, kernel_sizes=[3,3,3], channels=[10,10,1], symmetric_mode=True, conv='cv4'):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = 1
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(conv4_dt[conv](in_channels=ch_in,out_channels=ch_out,kernel_size=k_size,bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)

    def forward(self, x):
        if self.symmetric_mode:
            # apply network on the input and its "transpose" (swapping A-B to B-A ordering of the correlation tensor),
            # this second result is "transposed back" to the A-B ordering to match the first result and be able to add together
            x = self.conv(x)+self.conv(x.permute(0,1,4,5,2,3)).permute(0,1,4,5,2,3)
            # because of the ReLU layers in between linear layers,
            # this operation is different than convolving a single time with the filters+filters^T
            # and therefore it makes sense to do this.
        else:
            x = self.conv(x)
        return x


class MatchNet(nn.Module):
    def __init__(self, temp=3.0, cv_type='red', sym_mode=True, cv_kernels=[3,3,3], cv_channels=[10,10,1]):
        super(self).__init__()
        self.temp = temp
        self.NeighConsensus = NeighConsensus(kernel_sizes=cv_kernels,channels=cv_channels, symmetric_mode=sym_mode, conv=cv_type)

    def forward(self, corr, v, ig_mask=None):  # ig_mask [1, 3600]
        if corr.dim() == 5:
            corr = corr.unsqueeze(1)        # [B, 1, h, w, h_s, w_s]
        B, _, h, w, h_s, w_s = corr.shape

        corr4d = self.run_match_model(corr).squeeze(1)
        corr2d = corr4d.view(B, h*w, h_s*w_s)

        if ig_mask is not None:
            ig_mask = ig_mask.view(B, -1, h_s*w_s).expand(corr2d.shape)
            corr2d[ig_mask==True] = 0.0001         # [B, N_q, N_s]

        attn = F.softmax( corr2d*self.temp, dim=-1 )
        weighted_v = torch.bmm(v, attn.permute(0, 2, 1))  # [B, 512, N_s] * [B, N_s, N_q] -> [1, 512, N_q]
        weighted_v = weighted_v.view(B, -1, h, w)
        return weighted_v

    def run_match_model(self,corr4d):
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)
        corr4d = MutualMatching(corr4d)
        return corr4d

