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

