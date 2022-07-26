# encoding:utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .conv4d import CenterPivotConv4d, Conv4d
from .base import SpatialContextEncoder
from .model_util import get_corr

from .base.correlation import Correlation
from .base.geometry import Geometry
from .base.chm import CHM4d, CHM6d

conv4_dt = {'cv4': Conv4d, 'red': CenterPivotConv4d}

# input arguments
# Conv4d: ( in_channels, out_channels, kernel_size, bias=True, pre_permuted_filters=True, padding=True )
# CenterPivotConv4d: (in_channels, out_channels, kernel_size, stride=(1,)*4, padding=(1,)*4, bias=True))

def MutualMatching(corr4d):
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_lst = []
    for chn in range(ch):
        corr4d_chn = corr4d[:, chn, :, :, :, :]   # [B, h, w, h_s, w_s]
        corr4d_chn = MutualMatching_chn(corr4d_chn).unsqueeze(1)
        corr4d_lst.append(corr4d_chn)

    out = torch.cat(corr4d_lst, dim=1)
    return out


def MutualMatching_chn(corr4d):
    # mutual matching
    batch_size, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1*fs2, fs3, fs4)  # [batch_idx,k_A,i_B,j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3*fs4)

    # get max
    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  # parenthesis are important for symmetric output
    return corr4d


class NeighConsensus(torch.nn.Module):
    def __init__(self, kernel_sizes=[3,3,3], channels=[10,10,1], symmetric_mode=True, conv='cv4', in_channel=1):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i==0:
                ch_in = in_channel
            else:
                ch_in = channels[i-1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(conv4_dt[conv](in_channels=ch_in,out_channels=ch_out,kernel_size=(k_size,)*4,padding=(1,)*4,bias=True))
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
    def __init__(self, temp=3.0, cv_type='red', in_channel=1, sce=False, cyc=False, sym_mode=True, cv_kernels=[3,3,3], cv_channels=[10,10,1]):
        super().__init__()
        self.temp = temp
        self.sce = sce
        self.cyc = cyc
        self.in_channel = in_channel
        if self.sce:
            sce_ksz = 25
            self.SpatialContextEncoder = SpatialContextEncoder(kernel_size=sce_ksz, input_dim=sce_ksz * sce_ksz + 2048, hidden_dim=2048)
        if self.cyc:
            self.ass_drop  = nn.Dropout(0.1)

        self.NeighConsensus = NeighConsensus(kernel_sizes=cv_kernels,channels=cv_channels, symmetric_mode=sym_mode, conv=cv_type, in_channel=in_channel)

    def forward(self, fq_fea, fs_fea, v, s_mask=None, ig_mask=None, ret_corr=False, use_cyc=False, ret_cyc=False):  # ig_mask [1, 3600]
        B, ch, h, w = fq_fea.shape
        if v.dim() == 4:
            v = v.flatten(2)

        fq_fea = F.normalize(fq_fea, dim=1)
        fs_fea = F.normalize(fs_fea, dim=1)

        if self.sce:
            fq_fea = self.SpatialContextEncoder(fq_fea)  # [B, ch, h, w]
            fs_fea = self.SpatialContextEncoder(fs_fea)  # [B, ch, h, w]

        corr = get_corr(fq_fea, fs_fea)
        corr = corr.view(B, -1, h, w, h, w)  # [B, 1, h, w, h_s, w_s]

        corr4d = self.run_match_model(corr).squeeze(1)  # [B, h, w, h_s, w_s]
        corr2d = corr4d.view(B, h*w, h*w)

        if ig_mask is not None:
            ig_mask = ig_mask.view(B, -1, h*w).expand(corr2d.shape)
            corr2d[ig_mask==True] = 0.0001         # [B, N_q, N_s]
        if self.cyc and use_cyc:
            inconsistent_mask = self.run_cyc(corr2d, s_mask)   # positive means inconsistent [B, N_s]
            inconsistent_mask = inconsistent_mask.unsqueeze(1)   # [B, 1, N_s]
            corr2d = corr2d + inconsistent_mask * (-1000.0)

        attn = F.softmax( corr2d*self.temp, dim=-1 )
        weighted_v = torch.bmm(v, attn.permute(0, 2, 1))  # [B, 512, N_s] * [B, N_s, N_q] -> [1, 512, N_q]
        weighted_v = weighted_v.view(B, -1, h, w)

        if ret_corr and ret_cyc:
            return weighted_v, corr2d.reshape(B, h, w, h, w), inconsistent_mask
        elif ret_cyc:
            return weighted_v, inconsistent_mask,
        elif ret_corr:
            return weighted_v, corr2d.reshape(B, h, w, h, w)
        else:
            return weighted_v

    def corr_forward(self, corr4d, v, ret_attn=False):  # ig_mask [1, 3600]
        if v.dim() == 4:
            v = v.flatten(2)
        B, ch, h, w, h, w = corr4d.shape
        assert ch == self.in_channel, 'input corr channel inconsistent with in_channel of NCNet'

        corr4d = self.run_match_model(corr4d).squeeze(1)  # [B, h, w, h_s, w_s]
        corr2d = corr4d.view(B, h*w, h*w)

        attn = F.softmax( corr2d*self.temp, dim=-1 )
        weighted_v = torch.bmm(v, attn.permute(0, 2, 1))  # [B, 512, N_s] * [B, N_s, N_q] -> [1, 512, N_q]
        weighted_v = weighted_v.view(B, -1, h, w)

        if ret_attn:
            return corr2d, weighted_v
        return weighted_v  # [B, 512, h, w]

    def run_match_model(self,corr4d):
        corr4d = MutualMatching(corr4d)
        corr4d = self.NeighConsensus(corr4d)
        corr4d = MutualMatching(corr4d)
        return corr4d

    def run_cyc(self, corr2d, s_mask):

        if s_mask is not None:  # [B, 1, 60, 60]

            B, n_q, n_s = corr2d.shape
            s_mask = s_mask.view(B, n_s)

            k2q_sim_idx = corr2d.max(1)[1]  # [B, n_s]
            q2k_sim_idx = corr2d.max(2)[1]  # [B, n_q]

            re_map_idx = torch.gather(q2k_sim_idx, 1, k2q_sim_idx)  # [B, n_s]  k->q->k
            re_map_mask = torch.gather(s_mask, 1, re_map_idx)       # [B, n_s]

            association = (s_mask == re_map_mask).to(corr2d.device)   # [B, n_s], True means matched position in supp

            inconsistent = ~association            # True means unmatched
            inconsistent = inconsistent.float()
            inconsistent = self.ass_drop(inconsistent)
            return inconsistent


#############

r""" Conovlutional Hough matching layers """


class CHMLearner(nn.Module):

    def __init__(self, ktype, feat_dim, temp=20.0):
        super(CHMLearner, self).__init__()
        self.temp = temp

        # Scale-wise feature transformation
        self.scales = [0.5, 1, 2]
        self.conv2ds = nn.ModuleList([nn.Conv2d(feat_dim, feat_dim // 4, kernel_size=3, padding=1, bias=False) for _ in self.scales])

        # CHM layers
        ksz_translation = 5
        ksz_scale = 3
        self.chm6d = CHM6d(1, 1, ksz_scale, ksz_translation, ktype)
        self.chm4d = CHM4d(1, 1, ksz_translation, ktype, bias=True)

        # Activations
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, src_feat, trg_feat, v, ig_mask=None, ret_corr=False):

        corr = Correlation.build_correlation6d(src_feat, trg_feat, self.scales, self.conv2ds).unsqueeze(1)   # [1, 1, 3, 3, 30, 30, 30 30]
        bsz, ch, s, s, h, w, h, w = corr.size()

        # CHM layer (6D)
        corr = self.chm6d(corr)
        corr = self.sigmoid(corr)   # [1, 1, 3, 3, 30, 30, 30 30]

        # Scale-space maxpool
        corr = corr.view(bsz, -1, h, w, h, w).max(dim=1)[0]    # max pool along scale  [1, 30, 30, 30, 30]
        corr = Geometry.interpolate4d(corr, [h * 2, w * 2]).unsqueeze(1)             # [1, 1, 60, 60, 60, 60]

        # CHM layer (4D)
        corr = self.chm4d(corr).squeeze(1)     # [1, 60, 60, 60, 60]

        # To ensure non-negative vote scores & soft cyclic constraints
        corr = self.softplus(corr)
        corr = Correlation.mutual_nn_filter(corr.view(bsz, corr.size(-1) ** 2, corr.size(-1) ** 2).contiguous())  # [1, 3600, 3600]

        corr2d = corr.view(bsz, 4*h*w, 4*h*w)

        if ig_mask is not None:
            ig_mask = ig_mask.view(bsz, -1, 4*h*w).expand(corr2d.shape)
            corr2d[ig_mask == True] = 0.0001  # [B, N_q, N_s]
        attn = F.softmax(corr2d * self.temp, dim=-1)
        weighted_v = torch.bmm(v, attn.permute(0, 2, 1))  # [B, 512, N_s] * [B, N_s, N_q] -> [1, 512, N_q]
        weighted_v = weighted_v.view(bsz, -1, 2*h, 2*w)

        if ret_corr:
            return weighted_v, corr2d
        else:
            return weighted_v
