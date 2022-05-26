# encoding:utf-8

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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
    def __init__(self, n_head, dim_k, dim_v, d_k, d_v, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.n_head = n_head   # 4
        self.d_k = d_k         # 512
        self.d_v = d_v         # 512

        self.qk_fc = nn.Linear(dim_k, n_head * d_k, bias=False)
        self.v_fc = nn.Linear(dim_v, n_head * d_v, bias=False)
        nn.init.normal_(self.qk_fc.weight, mean=0, std=np.sqrt(2.0 / (dim_k + d_k)))
        nn.init.normal_(self.v_fc.weight, mean=0, std=np.sqrt(2.0 / (dim_v + d_v)))

        self.temperature=np.power(d_k, 0.5)
        self.layer_norm = nn.LayerNorm(d_v)

        self.fc = nn.Linear(n_head * d_v, d_v)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, s_valid_mask, idt):   # k, v: support pixels, q：query pixels
        B, N_q, C = q.shape
        N_s = k.size(1)

        q = self.qk_fc(q).reshape(B, N_q, self.n_head, self.d_k).permute(0, 2, 1, 3)  #[B, N, nH, d_k] -> [B, nH, N, d_k]
        k = self.qk_fc(k).reshape(B, N_s, self.n_head, self.d_k).permute(0, 2, 1, 3)
        v = self.v_fc(v).reshape(B, N_s, self.n_head, self.d_v).permute(0, 2, 1, 3)
        q = q.contiguous().view(-1, N_q, self.d_k)  # [B*nH, N, d_k]
        k = k.contiguous().view(-1, N_s, self.d_k)
        v = v.contiguous().view(-1, N_s, self.d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) / self.temperature   # [B*nH, N_q, N_s]
        if s_valid_mask is not None:
            s_valid_mask = s_valid_mask.unsqueeze(1).repeat(1, self.n_head, 1)  # [B, N_s] ->  [B, nH, N_s]
            s_valid_mask = s_valid_mask.unsqueeze(-2).float()                   # [B, nH, 1, N_s]
            s_valid_mask = s_valid_mask.view(-1, 1, N_s) * -10000.0             # [B*nH, 1, N_s]
            attn = attn + s_valid_mask                                          # [B*nH, N_q, N_s]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)                          # dropOut on the attention weight!
        output = torch.bmm(attn, v)                        # [B*nH, N_q, d_v]

        output = output.view(B, self.n_head, N_q, self.d_v)                # [B, nH, N_q, d_v]
        output = output.permute(0, 2, 1, 3).contiguous().view(B, N_q, -1)  # [B, N_q, nH, d_v] -> [B, N_q, nH*d_v]

        output = self.dropout(self.fc(output))                             # [B, N_q, d_v]
        output = self.layer_norm(output + idt)
        return output

