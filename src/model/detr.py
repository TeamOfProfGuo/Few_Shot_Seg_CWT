# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .match import MatchNet
from .ops.modules import MSDeformAttn
from .positional_encoding import SinePositionalEncoding

in_fea_dim_lookup = {'l3': 1024, 'l4': 2048, 'l34': 1024+2048, 'l23':512+1024}


class DeTr(nn.Module):
    def __init__(self, args, sf_att=False, cs_att=True, reduce_dim=512):
        super().__init__()
        self.args = args     # rmid
        self.reduce_dim = reduce_dim
        self.sf_att = sf_att
        self.cs_att = cs_att

        in_fea_dim = in_fea_dim_lookup[args.rmid]
        self.adjust_feature = nn.Sequential(nn.Conv2d(in_fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
                                            nn.ReLU(inplace=True))

        if args.get('drop', False) != False:
            drop_out = 0.5
            self.adjust_feature.add_module('drop', nn.Dropout2d(p=drop_out))

        if cs_att:
            self.cross_trans = MatchNet(temp=args.temp, cv_type='red', sce=False, sym_mode=True)
        if sf_att:
            self.self_trans =  DeformAtt(embed_dims=reduce_dim, n_levels=1, n_heads=8, n_points=9)

        self.print_model()

    def forward(self, fq_lst, fs_lst, f_q, f_s, padding_mask=None, s_padding_mask=None):
        fq_fea, fs_fea = self.compute_feat(fq_lst, fs_lst)

        if self.cs_att:
            ca_fq = self.cross_trans(fq_fea, fs_fea, f_s, ig_mask=None, ret_corr=False)
            f_q = F.normalize(f_q, p=2, dim=1) + F.normalize(ca_fq, p=2, dim=1) * self.args.att_wt

        if self.sf_att:
            sa_fq = self.self_trans(fq_fea, f_q, padding_mask=padding_mask)   # [B, 512, 60, 60]
            f_q = F.normalize(f_q, p=2, dim=1) + F.normalize(sa_fq, p=2, dim=1) * self.args.att_wt

        return f_q, sa_fq if self.sf_att else None, ca_fq if self.cs_att else None

    def compute_feat(self, fq_lst, fs_lst):
        if self.args.rmid == 'nr':
            idx = [-1]
        elif self.args.rmid in ['l2', 'l3', 'l4', 'l34', 'l23']:
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


class DeformAtt(nn.Module):
    def __init__(self, embed_dims = 512, n_heads=8, n_points=9, n_levels=1):
        super().__init__()

        self.num_levels = n_levels
        self.level_embed = nn.Parameter(torch.rand(n_levels, embed_dims))
        self.positional_encoding = SinePositionalEncoding(embed_dims // 2, normalize=True)
        self.self_trans =  MSDeformAttn(d_model=embed_dims, n_levels=n_levels, n_heads=n_heads, n_points=n_points)

    def forward(self, fq_fea, f_q, padding_mask=None):
        if not isinstance(fq_fea, list):
            fq_fea = [fq_fea]

        q_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index = self.get_qry_flatten_input(fq_fea, qry_masks=padding_mask)
        reference_points = self.get_reference_points(spatial_shapes, device=q_flatten.device)
        input_flatten = f_q.flatten(2).permute(0, 2, 1)
        sa_fq = self.self_trans(q_flatten + pos_embed_flatten, reference_points, input_flatten, spatial_shapes, level_start_index, input_padding_mask=None)
        sa_fq = sa_fq.permute(0, 2, 1).view(f_q.shape)
        return sa_fq

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points.unsqueeze(2).repeat(1, 1, len(spatial_shapes), 1)
        return reference_points

    def get_qry_flatten_input(self, x, qry_masks):
        src_flatten = []
        qry_valid_masks_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []
        for lvl in range(self.num_levels):
            src = x[lvl]
            bs, c, h, w = src.shape

            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).permute(0, 2, 1)  # [bs, c, h*w] -> [bs, h*w, c]
            src_flatten.append(src)

            if qry_masks is not None:
                qry_mask = qry_masks[lvl]
                qry_valid_mask = []
                qry_mask = F.interpolate(qry_mask.unsqueeze(1), size=(h, w), mode='nearest').squeeze(1)
                for img_id in range(bs):
                    qry_valid_mask.append(qry_mask[img_id] == 255)
                qry_valid_mask = torch.stack(qry_valid_mask, dim=0)
            else:
                qry_valid_mask = torch.zeros((bs, h, w), device=src.device).long()

            pos_embed = self.positional_encoding(qry_valid_mask)  # [bs, num_feats, h, w]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)      # [bs, hw, num_feats]
            if self.num_levels>1:
                pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            pos_embed_flatten.append(pos_embed)                   # [bs, hw, num_feats]

            qry_valid_masks_flatten.append(qry_valid_mask.flatten(1))  # [bs, hw]

        src_flatten = torch.cat(src_flatten, 1)  # [bs, num_elem, c]   num_elem = \sum_{0}^{L}h*w
        qry_valid_masks_flatten = torch.cat(qry_valid_masks_flatten, dim=1)  # [bs, num_elem]
        pos_embed_flatten = torch.cat(pos_embed_flatten, dim=1)  # [bs, num_elem, c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  # [num_lvl, 2]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))  # [num_lvl]

        return src_flatten, qry_valid_masks_flatten, pos_embed_flatten, spatial_shapes, level_start_index
