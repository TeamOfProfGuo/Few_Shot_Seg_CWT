# encoding:utf-8

import torch
from torch import nn
import torch.nn.functional as F
from src.model.nets.resnet import resnet50, resnet101
from src.model.nets.vgg import vgg16_bn
from src.model.model_util import get_corr, get_ig_mask, Adapt_SegLoss, SegLoss
from torch.nn.utils.weight_norm import WeightNorm


def get_model(args) -> nn.Module:
    return PSPNet(args, zoom_factor=8, use_ppm=True)


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


class PSPNet(nn.Module):
    def __init__(self, args, zoom_factor, use_ppm):
        super(PSPNet, self).__init__()
        # assert args.layers in [50, 101, 152]
        assert 2048 % len(args.bins) == 0
        assert args.num_classes_tr > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.m_scale = args.m_scale
        self.bottleneck_dim = args.bottleneck_dim
        self.rmid = args.get('rmid', None)     # 是否返回中间层
        self.args = args                 # all_lr

        resnet_kwargs = {}
        if self.rmid == 'nr':
            resnet_kwargs['no_relu'] = True
        if args.arch == 'resnet':
            if args.layers == 50:
                resnet = resnet50(pretrained=args.pretrained, **resnet_kwargs)  # nbottlenecks = [3, 4, 6, 3]   # channels [256, 512, 1024, 2048]
            else:
                resnet = resnet101(pretrained=args.pretrained, **resnet_kwargs) # nbottlenecks = [3, 4, 23, 3]  # channels [256, 512, 1024, 2048]

            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                        resnet.conv2, resnet.bn2, resnet.relu,
                                        resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)

            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        elif args.arch == 'vgg':
            vgg = vgg16_bn(pretrained=args.pretrained)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        if self.m_scale:
            fea_dim = 1024 + 512
        else:
            if args.arch == 'resnet':
                fea_dim = 2048
            elif args.arch == 'vgg':
                fea_dim = 512
        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(args.bins)), args.bins)
            fea_dim *= 2
            self.bottleneck = nn.Sequential(
                nn.Conv2d(fea_dim, self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.bottleneck_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=args.dropout)
            )

        self.classifier = CosCls(in_dim=self.bottleneck_dim, n_classes=args.num_classes_tr, cls_type=args.cls_type)

        if args.get('inherit_base', False):
            self.val_classifier = nn.Conv2d(self.bottleneck_dim, args.num_classes_tr + 1, kernel_size=1, bias=False)

    def freeze_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        H = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        W = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x, fea_lst = self.extract_features(x)
        x = self.classify(x, (H, W))
        if self.rmid:
            return x, fea_lst
        else:
            return x

    def get_feat_backbone(self, x):
        x = self.layer0(x)
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        if self.rmid == 'nr':
            x_4, x4_nr = self.layer4(x_3)
        else:
            x_4 = self.layer4(x_3)
        return [x_1, x_2, x_3, x_4]

    def get_feat_list(self, img,):
        feats = dict()

        # Layer 0 and layer1
        feat = self.layer0(img)
        feat = self.layer1(feat)

        # Layer 2,3,4
        for lid in [2, 3, 4]:
            n_bottleneck = len(self.__getattr__('layer'+str(lid)))
            for bid in range(n_bottleneck):
                feat = self.__getattr__('layer'+str(lid))[bid](feat)
                if str(lid) in self.args.get('all_lr', 'l') or bid == n_bottleneck-1:  # to decide whether to to return intermediate layers
                    feats[lid] = feats.get(lid, []) + [feat]

        return feat, feats

    def extract_features(self, x):
        x_4, feat_lst = self.get_feat_list(x)    # feat_lst 其实是 dict

        x = self.ppm(x_4)
        x = self.bottleneck(x)

        if self.rmid is not None and ('l' in self.rmid or 'mid' in self.rmid):
            return x, feat_lst
        else:
            return x, []

    def classify(self, features, shape):
        x = self.classifier(features)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x

    def inner_loop(self, f_s, s_label):
        # input: f_s 为feature extractor输出的 feature map
        self.classifier.reset_parameters()

        # optimizer and loss function
        optimizer = torch.optim.SGD(self.classifier.parameters(), lr=self.args.cls_lr)

        criterion = SegLoss(loss_type=self.args.inner_loss_type)

        # inner loop 学习 classifier的params
        for index in range(self.args.adapt_iter):
            pred_s_label = self.classifier(f_s)  # [n_shot, 2(cls), 60, 60]
            pred_s_label = F.interpolate(pred_s_label, size=s_label.size()[1:],mode='bilinear', align_corners=True)
            s_loss = criterion(pred_s_label, s_label)  # pred_label: [n_shot, 2, 473, 473], label [n_shot, 473, 473]
            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()

    def increment_inner_loop(self, f_s, s_label, cls_idx, meta_train=True):
        """cls_idx: set weight in loss"""
        classifier = self.classifier if meta_train else self.val_classifier
        num_cls = self.args.num_classes_tr if meta_train else self.args.num_classes_tr+1

        optimizer = torch.optim.SGD(classifier.parameters(), lr=self.args.cls_lr)
        criterion = Adapt_SegLoss(num_cls=num_cls, fg_idx=cls_idx, tp=self.args.tp)

        for index in range(self.args.adapt_iter):
            pred_s_label = classifier(f_s)  # [n_shot, 2(cls), 60, 60]
            pred_s_label = F.interpolate(pred_s_label, size=s_label.size()[1:], mode='bilinear', align_corners=True)
            s_loss = criterion(pred_s_label, s_label)  # pred_label: [n_shot, 2, 473, 473], label [n_shot, 473, 473]
            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()


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
