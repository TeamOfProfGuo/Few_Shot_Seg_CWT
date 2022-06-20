# encoding:utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .conv4d import CenterPivotConv4d




class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):   # q: [4, 2, 512], k: [4, 3600, 512],  4: num_head*batch_size batch_size=1
        attn = torch.bmm(q, k.transpose(1, 2))  # [4, 2, 3600]
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)       # [4, 2, 3600]
        attn = self.softmax(attn)               # [4, 2, 3600]
        attn = self.dropout(attn)               # [4, 2, 3600]                        # dropOut on the attention weight!
        output = torch.bmm(attn, v)             # [4, 2, 3600].[4, 3600, 512]->[4, 2, 512]
        return output, attn, log_attn


class MultiHeadAttentionOne(nn.Module):
    """
    Multi-Head Attention module with shared projection
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttentionOne, self).__init__()
        self.n_head = n_head   # 4
        self.d_k = d_k         # 512
        self.d_v = d_v         # 512

        self.w_qkvs = nn.Linear(d_model, n_head * d_k, bias=False)                                         # Q, K, V 共享一个weight matrix ?
        nn.init.normal_(self.w_qkvs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, query_input=False):   # q: [1, 2, 512], k,v: [1, 512, 60, 60]
        k = k.view(k.size()[0], k.size()[1], -1)  # [bz, c, hw]
        v = v.view(v.size()[0], v.size()[1], -1)  # [bz, c, hw]

        k = k.permute(0, 2, 1).contiguous()  # [bz, hw, c]
        v = v.permute(0, 2, 1).contiguous()  # [bz, hw, c]

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()   # batch_size, num_weight
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qkvs(q).view(sz_b, len_q, n_head, d_k)  # [1, 2, 512] -> [1, 2, 2048] -> [1, 2, 4, 512] : 4 head
        k = self.w_qkvs(k).view(sz_b, len_k, n_head, d_k)  # [1, 3600, 2048] -> [1, 3600, 4, 512]
        v = self.w_qkvs(v).view(sz_b, len_v, n_head, d_v)  # [1, 3600, 2048] -> [1, 3600, 4, 512]

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # [(n*b), lq, dk]  head, batch, num_w, d_k -> [n*b, lq, dk]
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # [(n*b), lk, dk]
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # [(n*b), lv, dv]

        output, attn, log_attn = self.attention(q, k, v)             # output: [4, 2, 512]

        output = output.view(n_head, sz_b, len_q, d_v)               # [4, 1, 2, 512]
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # [b, lq, (n*dv)]

        output = self.dropout(self.fc(output))                       # [1, 2, 512]
        output = self.layer_norm(output + residual)

        return output


class CrossAttention(nn.Module):
    def __init__(self, n_head, dim, dim_v, ln=None, fv=None, fc=None, dropout=0.1, temp=None, trans_vn=False):
        super(CrossAttention, self).__init__()
        self.n_head = n_head   # 4
        head_dim = dim // n_head
        self.temperature = temp or head_dim ** -0.5
        self.trans_vn = trans_vn

        self.qk_fc = nn.Linear(dim, dim, bias=False)
        self.layer_norm_q = nn.LayerNorm(dim) if ln == 'ln' else nn.Identity()
        self.layer_norm_k = nn.LayerNorm(dim) if ln == 'ln' else nn.Identity()
        self.v_fc = nn.Linear(dim_v, dim_v, bias=False) if fv == 'fv' else nn.Identity()

        self.fc = nn.Linear(dim_v, dim_v) if fc=='fc' else nn.Identity()
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_v)

    def forward(self, k, v, q, idt, s_valid_mask):   # k, v: support pixels, q：query pixels
        B, N_q, C = q.shape
        B, N_s, D = v.shape

        if self.trans_vn:
            v = F.normalize(v, dim=-1)
            idt = F.normalize(idt, dim=-1)

        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)

        q = self.qk_fc(q).reshape(B, N_q, self.n_head, C//self.n_head).permute(0, 2, 1, 3)  #[B, N, nH, d_k] -> [B, nH, N, d_k]
        k = self.qk_fc(k).reshape(B, N_s, self.n_head, C//self.n_head).permute(0, 2, 1, 3)
        v = self.v_fc(v).reshape(B, N_s, self.n_head, D//self.n_head).permute(0, 2, 1, 3)
        q = q.contiguous().view(B*self.n_head, N_q, -1)  # [B*nH, N, d_k]
        k = k.contiguous().view(B*self.n_head, N_s, -1)
        v = v.contiguous().view(B*self.n_head, N_s, -1)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.temperature   # [B*nH, N_q, N_s]
        if s_valid_mask is not None:
            s_valid_mask = s_valid_mask.unsqueeze(1).repeat(1, self.n_head, 1)  # [B, N_s] ->  [B, nH, N_s]
            s_valid_mask = s_valid_mask.unsqueeze(-2).float()                   # [B, nH, 1, N_s]
            s_valid_mask = s_valid_mask.view(B*self.n_head, 1, N_s) * -1000.0             # [B*nH, 1, N_s]
            attn = attn + s_valid_mask                                          # [B*nH, N_q, N_s]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)                          # dropOut on the attention weight!
        output = torch.bmm(attn, v)                        # [B*nH, N_q, d_v]

        output = output.view(B, self.n_head, N_q, -1)                # [B, nH, N_q, d_v]
        output = output.permute(0, 2, 1, 3).contiguous().view(B, N_q, -1)  # [B, N_q, nH, d_v] -> [B, N_q, nH*d_v]

        output = self.dropout(self.fc(output))                             # [B, N_q, d_v]
        output = self.layer_norm(output + idt)
        return output, attn


class MHA(nn.Module):
    def __init__(self, n_head, dim, dim_v, ln=True, fv=True, fc=True, qkv_bias=False, qk_scale=None,
                 proj_drop=0.1, attn_drop=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim_v)

        self.n_head = n_head
        head_dim = dim // n_head
        self.scale = qk_scale or head_dim ** -0.5

        self.qk_fc = nn.Linear(dim, dim, bias=qkv_bias)
        # self.qk_fc.weight.data.copy_(torch.eye(dim, dim) + torch.randn(dim, dim) * 0.001)
        self.v_fc = nn.Linear(dim_v, dim_v, bias=qkv_bias) if (fv=='fv' or fv==True) else nn.Identity()
        self.proj = nn.Linear(dim_v, dim_v) if (fc=='fc' or fc==True) else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.norm2 = norm_layer(dim_v)

    def forward(self, k, v, q, idt=None, s_valid_mask=None, return_attention=True):
        q, k, v = self.norm1_q(q), self.norm1_k(k), self.norm1_v(v)

        B, N_q, C = q.shape
        B, N_s, D = v.shape

        q = self.qk_fc(q).reshape(B, N_q, self.n_head, C//self.n_head).permute(0, 2, 1, 3)  # [B, N, nH, d] -> [B, nH, N, d_k]
        k = self.qk_fc(k).reshape(B, N_s, self.n_head, C//self.n_head).permute(0, 2, 1, 3)
        v = self.v_fc(v).reshape(B, N_s, self.n_head, D//self.n_head).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale    # [B, nH, N_q, N_s]
        if s_valid_mask is not None:
            s_valid_mask = s_valid_mask.unsqueeze(1).repeat(1, self.n_head, 1)  # [B, N_s] ->  [B, nH, N_s]
            s_valid_mask = s_valid_mask.unsqueeze(-2).float() * (-1000.0)       # [B, nH, 1, N_s]
            attn = attn + s_valid_mask                                          # [B, nH, N_q, N_s]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, -1)  # [B, nH, N_q, d_v] -> [B, N_q, nH, d_v] -> [B, N_q, D]
        x = self.proj(x)
        x = self.proj_drop(x)

        output = x + idt
        return output, attn


class AttentionBlock(nn.Module):
    def __init__(self, n_head=1, dim=2048, dim_v=512, v_norm=False, mode='l', scale_att='sc',):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.v_norm = v_norm
        self.qk_fc = nn.Linear(dim, dim)
        self.qk_fc.weight.data.copy_(torch.eye(dim, dim) + torch.randn(dim, dim)*0.001)
        self.qk_fc.bias.data.zero_()

        if scale_att == 'sc':
            self.scale_att = nn.Parameter(torch.FloatTensor(1).fill_(20.0), requires_grad=True)
        else:
            self.scale_att = 20.0

        self.att_wt = LinearDiag(dim_v, mode=mode, wt=0.2)
        self.org_wt = LinearDiag(dim_v, mode=mode)

    def forward(self, k, v, q, idt, s_valid_mask):
        B, N_q, C = q.shape
        B, N_s, D = v.shape

        if self.v_norm is True or self.v_norm == 'vn':
            v = F.normalize(v, p=2, dim=-1)
            idt = F.normalize(idt, p=2, dim=-1)

        q = self.qk_fc(q)  # [B, N_q, C]
        k = self.qk_fc(k)  # [B, N_s, C]
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        attn = self.scale_att * torch.bmm(q, k.permute(0, 2, 1))
        if s_valid_mask is not None:
            s_valid_mask = s_valid_mask.unsqueeze(1).repeat(1, N_q, 1)  # [B, N_s] ->  [B, N_q, N_s]
            attn = attn + s_valid_mask * (-1000.0)
        attn = F.softmax(attn, dim=-1)     # [B, N_q, N_s]
        fq_att = torch.bmm(attn, v)        # [B, N_q, D]

        out = self.att_wt(fq_att) + self.org_wt(idt)
        return out, attn


class LinearDiag(nn.Module):
    def __init__(self, num_features, mode='l', wt=1.0, bias=False):
        super(LinearDiag, self).__init__()
        if mode == 'l':
            self.weight = nn.Parameter(torch.tensor(wt))
        elif mode == 'ld':
            weight = torch.FloatTensor(num_features).fill_(wt)  # initialize to the identity transform
            self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_features).fill_(0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, X):
        out = X * self.weight
        if self.bias is not None:
            out = out + self.bias.expand_as(out)
        return out


class DynamicFusion(nn.Module):

    def __init__(self, im_size=30, mid_dim=256):
        super().__init__()
        self.im_size = im_size

        self.conv4d = CenterPivotConv4d(in_channels=1, out_channels=1, kernel_size=(3,) *4, stride=(1, 1, 2, 2), padding=(1,)*4)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        in_dim = im_size*im_size * 2
        self.att = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(mid_dim, 1, kernel_size=1, padding=0)
        )

    def forward(self, corr, s_mask):    # corr: [1, 60, 60, 60, 60], s_mask: [1, 1, 60, 60]
        B, h, w, h_s, w_s = corr.shape
        B, c, h, w = s_mask.shape      # ignore should be considered as BG

        corr = corr.unsqueeze(1)    # [B, 1, h, w, h_s, w_s]
        corr = self.conv4d(corr)    # [B, 1, h, w, 30, 30]
        corr = corr.reshape(B, h, w, self.im_size**2).permute(0, 3, 1, 2)   # [B, 900, h, w]

        s_mask = self.pool(s_mask)  # [B, 1, 30, 30]
        s_mask = s_mask.view(B, self.im_size**2, 1, 1).expand(corr.shape)   # [B, 900, h, w]

        corr = torch.cat((corr, s_mask), dim=1)      # [B, 1800, h, w]
        wt = self.att(corr)                          # [B, 1, h, w]
        wt = F.sigmoid(wt)

        return wt


class FuseNet1(nn.Module):

    def __init__(self, im_size=30, mid_dim=256):
        super().__init__()
        self.im_size = im_size

        self.conv4d = nn.Sequential(
            CenterPivotConv4d(in_channels=1, out_channels=16, kernel_size=(3,)*4, stride=(1, 1, 2, 2), padding=(1,)*4),
            nn.ReLU(inplace=True),
            CenterPivotConv4d(in_channels=16, out_channels=1, kernel_size=(3,)*4, stride=(1, 1, 1, 1), padding=(1,)*4),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        in_dim = im_size*im_size * 3 + 4
        self.att = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(mid_dim, 2, kernel_size=1, padding=0)
        )

    def forward(self, corr_lst, s_mask, pd_lst):    # corr: [1, 60, 60, 60, 60], s_mask: [1, 1, 60, 60]
        B, h, w, h_s, w_s = corr_lst[0].shape
        B, c, h_sm, w_sm = s_mask.shape             # ignore should be considered as BG

        att_in = []
        for corr in corr_lst:
            corr = corr.unsqueeze(1)    # [B, 1, h, w, h_s, w_s]
            corr = self.conv4d(corr)    # [B, 1, h, w, 30, 30]
            corr = corr.reshape(B, h, w, self.im_size**2).permute(0, 3, 1, 2)   # [B, 900, h, w]
            att_in.append(corr)

        if h_sm == 2*self.im_size:
            s_mask = self.pool(s_mask)  # [B, 1, 30, 30]
        s_mask = s_mask.view(B, self.im_size**2, 1, 1).expand(-1, -1, h, w)   # [B, 900, h, w]
        att_in.append(s_mask)

        for pd in pd_lst:
            att_in.append(pd)           # [B, 2, 60]

        att_in = torch.cat(att_in, dim=1)      # [B, 1800, h, w]
        wt = self.att(att_in)                  # [B, 2, h, w]
        wt = F.softmax(wt, dim=1)

        return wt


class FuseNet(nn.Module):

    def __init__(self, im_size=30, mid_dim=256):
        super().__init__()
        self.im_size = im_size

        self.conv4d = nn.Sequential(
            CenterPivotConv4d(in_channels=1, out_channels=16, kernel_size=(3,)*4, stride=(1, 1, 2, 2), padding=(1,)*4),
            nn.ReLU(inplace=True),
            CenterPivotConv4d(in_channels=16, out_channels=1, kernel_size=(3,)*4, stride=(1, 1, 1, 1), padding=(1,)*4),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        in_dim = im_size*im_size * 4 + 1
        self.att = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(mid_dim, 1, kernel_size=1, padding=0)
        )

    def forward(self, corr, pd_mask0, corr_fg, corr_bg, s_mask):    # corr: [1, 60, 60, 60, 60], s_mask: [1, 1, 60, 60]
        B, h, w, h_s, w_s = corr.shape
        B, c, h_sm, w_sm = s_mask.shape      # ignore should be considered as BG

        att_in = []
        for corr in [corr]:
            corr = corr.unsqueeze(1)    # [B, 1, h, w, h_s, w_s]
            corr = self.conv4d(corr)    # [B, 1, h, w, 30, 30]
            corr = corr.reshape(B, h, w, self.im_size**2).permute(0, 3, 1, 2)   # [B, 900, h, w]
            att_in.append(corr)
        att_in.append(pd_mask0)

        for mask in [corr_fg, corr_bg, s_mask]:
            mask = mask.view(B, self.im_size**2, 1, 1).expand(-1, -1, h, w)   # [B, 900, h, w]
            att_in.append(mask)

        att_in = torch.cat(att_in, dim=1)      # [B, 1800, h, w]
        wt = self.att(att_in)                  # [B, 2, h, w]
        wt = F.sigmoid(wt)

        return wt
