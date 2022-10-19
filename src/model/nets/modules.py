
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True))
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim=256, proj='convmlp', bn_type='torchsyncbn'):
        super(ProjectionHead, self).__init__()
        print('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(in_dim, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=1),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(),
                nn.Conv2d(in_dim, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class CosCls(nn.Module):
    def __init__(self, in_dim=512, n_classes=2, cls_type = '00000'):
        super(CosCls, self).__init__()
        self.xNorm, self.bias, self.WeightNormR, self.weight_norm, self.temp = parse_param_coscls(cls_type)
        self.cls = nn.Conv2d(in_dim, n_classes, kernel_size=1, bias=self.bias)
        if self.WeightNormR:
            WeightNorm.apply(self.cls, 'weight', dim=0)  # split the weight update component to direction and norm

    def forward(self, x):
        if self.xNorm:
            x = F.normalize(x, p=2, dim=1, eps=0.00001)  # [B, ch, h, w]
        if self.weight_norm:
            self.cls.weight.data = F.normalize(self.cls.weight.data, p=2, dim=1, eps=0.00001)

        cos_dist = self.cls(x)   #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.temp * cos_dist
        return scores

    def reset_parameters(self):   # 与torch自己的method同名
        self.cls.reset_parameters()


def parse_param_coscls(cls_type):
    xNorm_dt = {'x': True, '0': False, 'o':False}
    WeightNormR_dt = {'r': True, '0': False, 'o': False}
    weight_norm_dt = {'n': True, '0': False, 'o': False}
    bias_dt = {'b': True, '0': False, 'o': False}
    temp = 1 if cls_type[4] in ['0', 'o'] else int(cls_type[4])
    print('classifier X_norm: {} weight_norm_Regular: {} weight_norm: {}, bias: {}, temp: {}'.format( xNorm_dt[cls_type[0]],
          bias_dt[cls_type[1]], WeightNormR_dt[cls_type[2]], weight_norm_dt[cls_type[3]], temp))
    return xNorm_dt[cls_type[0]], bias_dt[cls_type[1]], WeightNormR_dt[cls_type[2]], weight_norm_dt[cls_type[3]], temp
