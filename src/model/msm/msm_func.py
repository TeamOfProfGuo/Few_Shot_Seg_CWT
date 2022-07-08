import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules import padding

from functools import reduce
from operator import add
from torchvision import models

import torch.nn.functional as F

class MSBlock(nn.Module):
    def __init__(self, c_in, c_out=32, rate=4):
        super(MSBlock, self).__init__()
        self.rate = rate

        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(c_out, c_out, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)

        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)

        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(c_out, c_out, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class WeightAverage(nn.Module):
    def __init__(self, c_in, args, R=3):
        super(WeightAverage, self).__init__()
        c_out = c_in // 2

        self.conv_theta = nn.Conv2d(c_in, c_out, 1)
        self.conv_phi = nn.Conv2d(c_in, c_out, 1)
        self.conv_g = nn.Conv2d(c_in, c_out, 1)
        self.conv_back = nn.Conv2d(c_out, c_in, 1)
        self.CosSimLayer = nn.CosineSimilarity(dim=3)  # norm

        self.R = R
        self.c_out = c_out
        self.att_drop = nn.Dropout(args.get('att_drop', 0.0))
        self.proj_drop = nn.Dropout(args.get('proj_drop', 0.0))

    def forward(self, x):
        """
        x: torch.Tensor(batch_size, channel, h, w)
        """

        batch_size, c, h, w = x.size()
        padded_x = F.pad(x, (1, 1, 1, 1), 'replicate')
        neighbor = F.unfold(padded_x, kernel_size=self.R, dilation=1, stride=1)  # BS, C*R*R, H*W
        neighbor = neighbor.contiguous().view(batch_size, c, self.R, self.R, h, w)
        neighbor = neighbor.permute(0, 2, 3, 1, 4, 5)  # BS, R, R, c, h ,w
        neighbor = neighbor.reshape(batch_size * self.R * self.R, c, h, w)

        theta = self.conv_theta(x)  # BS, C', h, w               # Q
        phi = self.conv_phi(neighbor)   # BS*R*R, C', h, w       # K
        g = self.conv_g(neighbor)     # BS*R*R, C', h, w         # V

        phi = phi.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)                           # K
        phi = phi.permute(0, 4, 5, 3, 1, 2)  # BS, h, w, c, R, R                                            # K
        theta = theta.permute(0, 2, 3, 1).contiguous().view(batch_size, h, w, self.c_out)   # BS, h, w, c   # Q
        theta_dim = theta                                                                                   # Q

        cos_sim = self.CosSimLayer(phi, theta_dim[:, :, :, :, None, None])  # BS, h, w, c, R, R

        softmax_sim = F.softmax(cos_sim.contiguous().view(batch_size, h, w, -1), dim=3).contiguous().view_as(cos_sim)  # BS, h, w, R, R
        softmax_sim = self.att_drop(softmax_sim)

        g = g.contiguous().view(batch_size, self.R, self.R, self.c_out, h, w)
        g = g.permute(0, 4, 5, 1, 2, 3)  # BS, h, w, R, R, c_out

        weighted_g = g * softmax_sim[:, :, :, :, :, None]
        weighted_average = torch.sum(weighted_g.contiguous().view(batch_size, h, w, -1, self.c_out), dim=3)
        weight_average = weighted_average.permute(0, 3, 1, 2).contiguous()  # BS, c_out, h, w

        x_res = self.conv_back(weight_average)
        x_res = self.proj_drop(x_res)

        ret = x + x_res

        return ret
