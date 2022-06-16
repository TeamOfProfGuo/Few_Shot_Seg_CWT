# encoding:utf-8

r""" 4D and 6D convolutional Hough matching layers """

from torch.nn.modules.conv import _ConvNd
import torch.nn.functional as F
import torch.nn as nn
import torch
from . import chm_kernel


def fast4d(corr, kernel, bias=None):
    r""" Optimized implementation of 4D convolution """
    bsz, ch, srch, srcw, trgh, trgw = corr.size()
    out_channels, _, kernel_size, kernel_size, kernel_size, kernel_size = kernel.size()
    psz = kernel_size // 2

    out_corr = torch.zeros((bsz, out_channels, srch, srcw, trgh, trgw))
    if torch.cuda.is_available():
        out_corr = out_corr.cuda()
    corr = corr.transpose(1, 2).contiguous().view(bsz * srch, ch, srcw, trgh, trgw)

    for pidx, k3d in enumerate(kernel.permute(2, 0, 1, 3, 4, 5)):  # kernel: [ch, 1, 5, 5, 5, 5]
        inter_corr = F.conv3d(corr, k3d, bias=None, stride=1, padding=psz)
        inter_corr = inter_corr.view(bsz, srch, out_channels, srcw, trgh, trgw).transpose(1, 2).contiguous()

        add_sid = max(psz - pidx, 0)
        add_fid = min(srch, srch + psz - pidx)
        slc_sid = max(pidx - psz, 0)
        slc_fid = min(srch, srch - psz + pidx)

        out_corr[:, :, add_sid:add_fid, :, :, :] += inter_corr[:, :, slc_sid:slc_fid, :, :, :]

    if bias is not None:
        out_corr += bias.view(1, out_channels, 1, 1, 1, 1)

    return out_corr


def fast6d(corr, kernel, bias, diagonal_idx):
    r""" Optimized implementation of 6D convolutional Hough matching
         NOTE: this function only supports kernel size of (3, 3, 5, 5, 5, 5).
    r"""
    bsz, _, s6d, s6d, s4d, s4d, s4d, s4d = corr.size()
    _, _, ks6d, ks6d, ks4d, ks4d, ks4d, ks4d = kernel.size()
    corr = corr.permute(0, 2, 3, 1, 4, 5, 6, 7).contiguous().view(-1, 1, s4d, s4d, s4d, s4d)
    kernel = kernel.view(-1, ks6d ** 2, ks4d, ks4d, ks4d, ks4d).transpose(0, 1)
    corr = fast4d(corr, kernel).view(bsz, s6d * s6d, ks6d * ks6d, s4d, s4d, s4d, s4d)
    corr = corr.view(bsz, s6d, s6d, ks6d, ks6d, s4d, s4d, s4d, s4d).transpose(2, 3).\
        contiguous().view(-1, s6d * ks6d, s4d, s4d, s4d, s4d)

    ndiag = s6d + (ks6d // 2) * 2
    first_sum = []
    for didx in diagonal_idx:
        first_sum.append(corr[:, didx, :, :, :, :].sum(dim=1))
    first_sum = torch.stack(first_sum).transpose(0, 1).view(bsz, s6d * ks6d, ndiag, s4d, s4d, s4d, s4d)

    corr = []
    for didx in diagonal_idx:
        corr.append(first_sum[:, didx, :, :, :, :, :].sum(dim=1))
    sidx = ks6d // 2
    eidx = ndiag - sidx
    corr = torch.stack(corr).transpose(0, 1)[:, sidx:eidx, sidx:eidx, :, :, :, :].unsqueeze(1).contiguous()
    corr += bias.view(1, -1, 1, 1, 1, 1, 1, 1)

    reverse_idx = torch.linspace(s6d * s6d - 1, 0, s6d * s6d).long()
    if torch.cuda.is_available():
        reverse_idx = reverse_idx.cuda()
    corr = corr.view(bsz, 1, s6d * s6d, s4d, s4d, s4d, s4d)[:, :, reverse_idx, :, :, :, :].\
        view(bsz, 1, s6d, s6d, s4d, s4d, s4d, s4d)
    return corr

def init_param_idx4d(param_dict):
    param_idx = []
    for key in param_dict:
        curr_offset = int(key.split('_')[-1])
        idx = torch.tensor(param_dict[key]).cuda() if torch.cuda.is_available() else torch.tensor(param_dict[key])
        param_idx.append(idx)
    return param_idx

class CHM4d(_ConvNd):
    r""" 4D convolutional Hough matching layer
         NOTE: this function only supports in_channels=1 and out_channels=1.
    r"""
    def __init__(self, in_channels, out_channels, ksz4d, ktype, bias=True):
        super(CHM4d, self).__init__(in_channels, out_channels, (ksz4d,) * 4,
                                    (1,) * 4, (0,) * 4, (1,) * 4, False, (0,) * 4,
                                    1, bias, padding_mode='zeros')

        # Zero kernel initialization
        self.zero_kernel4d = torch.zeros((in_channels, out_channels, ksz4d, ksz4d, ksz4d, ksz4d))
        if torch.cuda.is_available():
            self.zero_kernel4d = self.zero_kernel4d.cuda()
        self.nkernels = in_channels * out_channels

        # Initialize kernel indices
        param_dict4d = chm_kernel.KernelGenerator(ksz4d, ktype).generate()
        param_shared =  param_dict4d is not None

        if param_shared:
            # Initialize the shared parameters (multiplied by the number of times being shared)
            self.param_idx = init_param_idx4d(param_dict4d)   # list of list of idx_1d of the 'param4d params sharing wt', 'psi': 55 learnable params
            weights = torch.abs(torch.randn(len(self.param_idx) * self.nkernels)) * 1e-3  # [55]
            for i in range(len(weights)):
                weights[i] = weights[i] * len( self.param_idx[i] )
            # for weight, param_idx in zip(weights.sort()[0], self.param_idx):    weight *= len(param_idx)    # modify the weight in place?

            self.weight = nn.Parameter(weights)
        else:  # full kernel initialziation
            self.param_idx = None
            self.weight = nn.Parameter(torch.abs(self.weight))
            if bias: self.bias = nn.Parameter(torch.tensor(0.0))
        print('(%s) # params in CHM 4D: %d' % (ktype, len(self.weight.view(-1))))

    def forward(self, x):
        kernel = self.init_kernel()
        x = fast4d(x, kernel, self.bias)
        return x

    def init_kernel(self):
        # Initialize CHM kernel (divided by the number of times being shared)
        ksz = self.kernel_size[-1]
        if self.param_idx is None:
            kernel = self.weight
        else:
            kernel = torch.zeros_like(self.zero_kernel4d).view(-1, ksz**4)  # [1, 1, 5, 5, 5, 5] -> [1, 5, 5, 5, 5] 为了支持有多个kernel的情况
            for idx, pdx in enumerate(self.param_idx):     # list of list (sublist 为 row_idx of i, j, k, l share weight)
                for jdx in range(len(kernel)):
                    weight = self.weight[idx + jdx * len(self.param_idx)].repeat(len(pdx)) / len(pdx)
                    kernel[jdx, pdx] = kernel[jdx, pdx] + weight
            kernel = kernel.view(self.in_channels, self.out_channels, ksz, ksz, ksz, ksz)
        return kernel


class CHM6d(_ConvNd):
    r""" 6D convolutional Hough matching layer with kernel (3, 3, 5, 5, 5, 5)
         NOTE: this function only supports in_channels=1 and out_channels=1.
    r"""
    def __init__(self, in_channels, out_channels, ksz6d, ksz4d, ktype):
        kernel_size = (ksz6d, ksz6d, ksz4d, ksz4d, ksz4d, ksz4d)
        super(CHM6d, self).__init__(in_channels, out_channels, kernel_size, (1,) * 6,
                                    (0,) * 6, (1,) * 6, False, (0,) * 6,
                                    1, bias=True, padding_mode='zeros')

        # Zero kernel initialization
        self.zero_kernel4d = torch.zeros((ksz4d, ksz4d, ksz4d, ksz4d))
        self.zero_kernel6d = torch.zeros((ksz6d, ksz6d, ksz4d, ksz4d, ksz4d, ksz4d))
        if torch.cuda.is_available():
            self.zero_kernel4d = self.zero_kernel4d.cuda()
            self.zero_kernel6d = self.zero_kernel6d.cuda()

        self.nkernels = in_channels * out_channels

        # Initialize kernel indices
        # Indices in scale-space where 4D convolutions are performed (3 by 3 scale-space)
        if torch.cuda.is_available():
            self.diagonal_idx = [torch.tensor(x).cuda() for x in [[6], [3, 7], [0, 4, 8], [1, 5], [2]]]
        else:
            self.diagonal_idx = [torch.tensor(x) for x in [[6], [3, 7], [0, 4, 8], [1, 5], [2]]]
        param_dict4d = chm_kernel.KernelGenerator(ksz4d, ktype).generate()
        param_shared =  param_dict4d is not None

        if param_shared:  # psi & iso kernel initialization
            if ktype == 'psi':
                self.param_dict6d = [[4], [0, 8], [2, 6], [1, 3, 5, 7]]
            elif ktype == 'iso':
                self.param_dict6d = [[0, 4, 8], [2, 6], [1, 3, 5, 7]]

            if torch.cuda.is_available():
                self.param_dict6d = [torch.tensor(i).cuda() for i in self.param_dict6d]
            else:
                self.param_dict6d = [torch.tensor(i) for i in self.param_dict6d]

            # Initialize the shared parameters (multiplied by the number of times being shared)
            self.param_idx = init_param_idx4d(param_dict4d)     # list of list for idx1d of the (4d kernel weight sharing)
            self.param = []
            for param_dict6d in self.param_dict6d:
                weights = torch.abs(torch.randn(len(self.param_idx))) * 1e-3    # each cross scale combination's conv4d params [55]
                for i in range(len(weights)):
                    weights[i] = weights[i] * (len(self.param_idx[i]) * len(param_dict6d))  # 为什么要先乘再除
                self.param.append(nn.Parameter(weights))        # ordered by param_dict6d, param_dict4d    # total 55*
            self.param = nn.ParameterList(self.param)    # size:[sn, ln], row idx: scale_offset(param_dict6d) column idx
        else:  # full kernel initialziation
            self.param_idx = None
            self.param = nn.Parameter(torch.abs(self.weight) * 1e-3)
        print('(%s) # params in CHM 6D: %d' % (ktype, sum([len(x.view(-1)) for x in self.param])))
        self.weight = None

    def forward(self, corr):
        kernel = self.init_kernel()
        corr = fast6d(corr, kernel, self.bias, self.diagonal_idx)
        return corr

    def init_kernel(self):
        # Initialize CHM kernel (divided by the number of times being shared)
        if self.param_idx is None:
            return self.param

        kernel6d = torch.zeros_like(self.zero_kernel6d)
        for idx, (param, param_dict6d) in enumerate(zip(self.param, self.param_dict6d)):
            ksz4d = self.kernel_size[-1]   # 此处param只 针对当前scale pair (所有share 参数的scale pair)
            kernel4d = torch.zeros_like(self.zero_kernel4d).view(-1)   # [5, 5, 5, 5] -> [525]
            for jdx, pdx in enumerate(self.param_idx):  # list of list (sublist 为 row_idx of i, j, k, l share weight)
                kernel4d[pdx] = ((param[jdx] / len(pdx)) / len(param_dict6d))
            kernel6d.view(-1, ksz4d, ksz4d, ksz4d, ksz4d)[param_dict6d] = kernel4d.view(ksz4d, ksz4d, ksz4d, ksz4d)   # [9, 5, 5, 5]
        kernel6d = kernel6d.unsqueeze(0).unsqueeze(0)

        return kernel6d

