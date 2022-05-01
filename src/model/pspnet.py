# encoding:utf-8

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .resnet import resnet50, resnet101
from .vgg import vgg16_bn


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
        self.rmid = args.get('rmid', False)     # 是否返回中间层
        self.args = args

        if args.arch == 'resnet':
            if args.layers == 50:
                resnet = resnet50(pretrained=args.pretrained)
            else:
                resnet = resnet101(pretrained=args.pretrained)
            self.layer0 = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu,
                resnet.conv2, resnet.bn2, resnet.relu,
                resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)

            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            self.feature_res = (53, 53)

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
        self.classifier = nn.Conv2d(self.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

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

    def extract_features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x_2 = self.layer2(x)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)
        if self.m_scale:
            x = torch.cat([x_2, x_3], dim=1)
        else:
            x = x_4
        x = self.ppm(x)
        x = self.bottleneck(x)
        if self.rmid:
            return x, [x_2, x_3, x_4]
        else:
            return x, []

    def extract_features_backbone(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x_2 = self.layer2(x)
        x_3 = self.layer3(x_2)
        if self.m_scale:
            x = torch.cat([x_2, x_3], dim=1)
        else:
            x = self.layer4(x_3)
        return x

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

        s_label_arr = s_label.cpu().numpy().copy()  # [ n_shots, img_size, img_size]
        back_pix = np.where(s_label_arr == 0)
        target_pix = np.where(s_label_arr == 1)
        weight = torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])])  # bg的weight: num of gf pixels
        if torch.cuda.is_available():
            weight = weight.cuda()
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=255)

        # inner loop 学习 classifier的params
        for index in range(self.args.adapt_iter):
            pred_s_label = self.classifier(f_s)  # [n_shot, 2(cls), 60, 60]
            pred_s_label = F.interpolate(pred_s_label, size=s_label.size()[1:],mode='bilinear', align_corners=True)
            s_loss = criterion(pred_s_label, s_label)  # pred_label: [n_shot, 2, 473, 473], label [n_shot, 473, 473]
            optimizer.zero_grad()
            s_loss.backward()
            optimizer.step()

    def outer_forward(self, f_q, f_s, fq_fea, fs_fea):
        # 最后为了finetune f_q
        bs, C, height, width = f_q.size()

        # 基于attention, refine f_q, 并对query img做prediction
        proj_q = fq_fea.view(bs, -1, height * width).permute(0, 2, 1)  # [1, 2048, hw] -> [1, hw, 2048]
        proj_k = fs_fea.view(bs, -1, height * width)  # [1, 2048, hw]
        proj_v = f_s.view(bs, -1, height * width)
        sim = torch.bmm(proj_q, proj_k)  # [1, 3600 (q_hw), 3600(k_hw)]
        attention = F.softmax(sim, dim=-1)
        weighted_v = torch.bmm(proj_v, attention.permute(0, 2, 1))  # [1, 512, hw_k] * [1, hw_k, hw_q] -> [1, 512, hw_q]
        weighted_v = weighted_v.view(bs, C, height, width)

        out = (weighted_v * self.gamma + f_q)/(1+self.gamma)
        pred_q_label = self.classifier(out)
        return pred_q_label


        # pred_q_label = F.interpolate(pred_q_label, size=q_label.shape[1:],mode='bilinear', align_corners=True)
        #
        # # loss function for outer loop
        # # Dynamic class weights used for query image only during training
        # q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
        # q_back_pix = np.where(q_label_arr == 0)
        # q_target_pix = np.where(q_label_arr == 1)
        # loss_weight = torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)])
        # loss_weight = loss_weight.cuda() if torch.cuda.is_available() else loss_weight
        # criterion = nn.CrossEntropyLoss(weight=loss_weight,ignore_index=255)
        # q_loss = criterion(pred_q_label, q_label.long())







