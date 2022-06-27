# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .match import MatchNet
from .ops.modules import MSDeformAttn
from .positional_encoding import SinePositionalEncoding
from .msm import MSBlock

in_fea_dim_lookup = {'l3': 1024, 'l4': 2048, 'l34': 1024+2048, 'l23':512+1024}


class MMN(nn.Module):
    def __init__(self, args, sf_att=False, cs_att=True, reduce_dim=512):
        super().__init__()
        self.args = args     # rmid
        self.reduce_dim = reduce_dim
        rate = 4

        if self.args.layers == 50:
            self.nbottlenecks = [3, 4, 6, 3]
            self.feature_channels = [256, 512, 1024, 2048]

        in_fea_dim = in_fea_dim_lookup[args.rmid]
        self.adjust_feature = nn.Sequential(nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                                            nn.ReLU(inplace=True))

        for i, k in enumerate(self.nbottlenecks):
            for j in range(k):
                setattr(self, "msblock"+str(i+1)+"_"+str(j+1), MSBlock(self.feature_channels[i], rate))
                setattr(self, "conv"+str(i+1)+"_"+str(j+1)+"_down", nn.Conv2d(32, inner_channel, (1, 1), stride=1))
            setattr(self, "conv"+str(i+1)+"_kernel", nn.Conv2d(inner_channel,
                                                               inner_channel, (3, 3), stride=1, padding=1))
            if i != len(self.nbottlenecks)-1:
                setattr(self, "conv"+str(i+1)+"_scale", nn.Conv2d(2 *
                                                                  inner_channel, inner_channel, (3, 3), stride=1, padding=1))
            if i != 0:
                setattr(self, "feat_upsample_"+str(i+1), nn.ConvTranspose2d(
                    inner_channel, inner_channel, 4, stride=2, padding=1, bias=False))
            setattr(self, "wa_"+str(i+1), WeightAverage(inner_channel))

        self.corr_net = MatchNet(temp=args.temp, cv_type='red', sce=False, sym_mode=True)

        self.print_model()

    def forward(self, fq_lst, fs_lst, f_q, f_s, padding_mask=None, s_padding_mask=None):
        fq_fea, fs_fea = self.get_feat(fq_lst, fs_lst)

        if self.cs_att:
            ca_fq = self.cross_trans(fq_fea, fs_fea, f_s, ig_mask=None, ret_corr=False)
            f_q = F.normalize(f_q, p=2, dim=1) + F.normalize(ca_fq, p=2, dim=1) * self.args.att_wt

        if self.sf_att:
            sa_fq = self.self_trans(fq_fea, f_q, padding_mask=padding_mask)   # [B, 512, 60, 60]
            f_q = F.normalize(f_q, p=2, dim=1) + F.normalize(sa_fq, p=2, dim=1) * self.args.att_wt

        return f_q, sa_fq if self.sf_att else None, ca_fq if self.cs_att else None

    def get_feat(self, fq_lst, fs_lst):
        if self.args.rmid == 'nr':
            idx = [-1]
        elif self.args.rmid in ['l2', 'l3', 'l4', 'l34', 'l23', 'l234']:
            rmid = self.args.rmid[1:]
            idx = [int(num) - 2 for num in list(rmid)]

        fq_fea = torch.cat( [fq_lst[id] for id in idx], dim=1 )
        fs_fea = torch.cat( [fs_lst[id] for id in idx], dim=1 )

        fq_fea = self.adjust_feature(fq_fea)
        fs_fea = self.adjust_feature(fs_fea)
        return fq_fea, fs_fea

    def print_model(self):
        repr_str = self.__class__.__name__
        repr_str += f'reduce_dim={self.reduce_dim}, '
        repr_str += f'with_self_transformer={self.sf_att})'
        repr_str += f'with_cross_transformer={self.cs_att})'
        print(repr_str)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
