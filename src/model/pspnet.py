# encoding:utf-8

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .resnet import resnet50, resnet101
from .vgg import vgg16_bn
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
        self.args = args

        resnet_kwargs = {}
        if self.rmid == 'nr':
            resnet_kwargs['no_relu'] = True
        if args.arch == 'resnet':
            if args.layers == 50:
                resnet = resnet50(pretrained=args.pretrained, **resnet_kwargs)
            else:
                resnet = resnet101(pretrained=args.pretrained, **resnet_kwargs)
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

        if args.get('dist', 'dot') == 'dot':
            self.classifier = nn.Conv2d(self.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False)
        elif args.get('dist') == 'cos':
            self.classifier = CosCls(in_dim=self.bottleneck_dim, n_classes=args.num_classes_tr, WeightNormR=False, cls_type=args.cls_type)
        elif args.get('dist') == 'cosN':  # adaptive weight norm
            self.classifier = CosCls(in_dim=self.bottleneck_dim, n_classes=args.num_classes_tr, WeightNormR=True, cls_type=args.cls_type)

        self.gamma = nn.Parameter(torch.tensor(0.2))

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
        if self.rmid == 'nr':
            x_4, x4_nr = self.layer4(x_3)
        else:
            x_4 = self.layer4(x_3)

        if self.m_scale:
            x = torch.cat([x_2, x_3], dim=1)
        else:
            x = x_4
        x = self.ppm(x)
        x = self.bottleneck(x)

        if self.rmid == 'mid' or self.rmid in [3,4]:
            return x, [x_2, x_3, x_4]
        elif self.rmid == 'nr':
            return x, [x4_nr]
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

    def outer_forward(self, f_q, f_s, fq_fea, fs_fea, s_label, q_label=None, pd_q0=None, pd_s=None, ret_curr=False):
        # f_q/f_s:[1,512,h,w],  fq_fea/fs_fea:[1,2048,h,w],  s_label: [1,H,w]
        bs, C, height, width = f_q.size()

        # 基于attention, refine f_q, 并对query img做prediction
        proj_q = fq_fea.view(bs, -1, height * width).permute(0, 2, 1)  # [1, 2048, hw] -> [1, hw, 2048]
        proj_k = fs_fea.view(bs, -1, height * width)  # [1, 2048, hw]
        proj_v = f_s.view(bs, -1, height * width)

        # normalize q and k
        proj_q = F.normalize(proj_q, dim=-1)
        proj_k = F.normalize(proj_k, dim=-2)
        sim = torch.bmm(proj_q, proj_k)  # [1, 3600 (q_hw), 3600(k_hw)]
        corr = sim.reshape(bs, height, width, height, width)                                     # return Corr

        # ignore misleading points
        pd_q_mask0 = pd_q0.argmax(dim=1)
        q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=f_q.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
        qf_mask = (q_mask != 255.0) * (pd_q_mask0 == 1)  # predicted qry FG
        qb_mask = (q_mask != 255.0) * (pd_q_mask0 == 0)  # predicted qry BG
        qf_mask = qf_mask.view(qf_mask.shape[0], -1, 1).expand(sim.shape)  # 保留query predicted FG
        qb_mask = qb_mask.view(qb_mask.shape[0], -1, 1).expand(sim.shape)  # 保留query predicted BG

        sim_qf = sim[qf_mask].reshape(1, -1, 3600)
        if sim_qf.numel() > 0:
            th_qf = torch.quantile(sim_qf.flatten(), 0.8)
            sim_qf = torch.mean(sim_qf, dim=1)  # 取平均 对应support img 与Q前景相关 所有pixel
            qf_mask = sim_qf
        else:
            print('------ pred qf mask is empty! ------')
            qf_mask = torch.zeros([1, 3600], dtype=torch.float).cuda()

        sim_qb = sim[qb_mask].reshape(1, -1, 3600)
        if sim_qb.numel() > 0:
            th_qb = torch.quantile(sim_qb.flatten(), 0.8)
            sim_qb = torch.mean(sim_qb, dim=1)  # 取平均 对应support img 与Q背景相关 所有pixel
            qb_mask = sim_qb
        else:
            print('------ pred qb mask is empty! ------')
            qb_mask = torch.zeros([1, 3600], dtype=torch.float).cuda()

        sf_mask = pd_s.argmax(dim=1).view(1, 3600)
        null_mask = torch.zeros([1, 3600], dtype=torch.bool)
        null_mask = null_mask.cuda() if torch.cuda.is_available() else null_mask
        ig_mask1 = (sim_qf > th_qf) & (sf_mask == 0) if sim_qf.numel() > 0 else null_mask
        ig_mask3 = (sim_qb > th_qb) & (sf_mask == 1) if sim_qb.numel() > 0 else null_mask
        ig_mask2 = (sim_qf > th_qf) & (sim_qb > th_qb) if sim_qf.numel() > 0 and sim_qb.numel() > 0 else null_mask
        ig_mask = ig_mask1 | ig_mask2 | ig_mask3

        ig_mask = ig_mask.unsqueeze(1).expand(sim.shape)
        sim[ig_mask == True] = 0.00001

        # mask ignored support pixels
        s_mask = F.interpolate(s_label.unsqueeze(1).float(), size=f_s.shape[-2:], mode='nearest')  # [1,1,h,w]
        s_mask = (s_mask > 1).view(s_mask.shape[0], 1, -1)  # [n_shot, 1, hw]
        s_mask = s_mask.expand(sim.shape)  # [1, q_hw, hw]
        sim[s_mask == True] = 0.00001

        if self.args.get('dist','dot')=='cos':
            proj_v = F.normalize(proj_v, dim=1)
            f_q = F.normalize(f_q, dim=1)

        attention = F.softmax(sim * self.args.temp, dim=-1)
        weighted_v = torch.bmm(proj_v, attention.permute(0, 2, 1))  # [1, 512, hw_k] * [1, hw_k, hw_q] -> [1, 512, hw_q]
        weighted_v = weighted_v.view(bs, C, height, width)

        out = (weighted_v * self.gamma + f_q)/(1+self.gamma)
        pred_q_label = self.classifier(out)

        if ret_curr == 'cr':
            return pred_q_label, [corr, weighted_v]
        elif ret_curr == 'cr_mk':
            return pred_q_label, [corr, weighted_v], [qf_mask, qb_mask]
        else:
            return pred_q_label

    def sampling(self, fq_fea, fs_fea, s_label, q_label=None, pd_q0=None, pd_s = None):
        bs, C, height, width = fq_fea.size()

        # 基于attention, refine f_q, 并对query img做prediction
        proj_q = F.normalize(fq_fea.view(bs, -1, height * width), dim=-2)  # [1, 2048, hw]
        proj_k = F.normalize(fs_fea.view(bs, -1, height * width), dim=-2)  # [1, 2048, hw]
        proj_q = proj_q.permute(0, 2, 1)  # [1, 2048, hw] -> [1, hw, 2048]
        sim = torch.bmm(proj_q, proj_k)  # [1, 3600 (q_hw), 3600(k_hw)]

        # mask ignored pixels
        s_mask = F.interpolate(s_label.unsqueeze(1).float(), size=fq_fea.shape[-2:], mode='nearest')  # [1,1,h,w]
        s_mask = (s_mask > 1).view(s_mask.shape[0], 1, -1)  # [n_shot, 1, hw]
        s_mask = s_mask.expand(sim.shape)  # [1, q_hw, hw]
        sim[s_mask == True] = 0.00001

        # ignore misleading points
        pd_q_mask0 = pd_q0.argmax(dim=1)
        q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=fq_fea.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
        qf_mask = (q_mask != 255.0) * (pd_q_mask0 == 1)  # predicted qry FG
        qb_mask = (q_mask != 255.0) * (pd_q_mask0 == 0)  # predicted qry BG
        qf_mask = qf_mask.view(qf_mask.shape[0], -1, 1).expand(sim.shape)  # 保留query predicted FG
        qb_mask = qb_mask.view(qb_mask.shape[0], -1, 1).expand(sim.shape)  # 保留query predicted BG

        sim_qf = sim[qf_mask].reshape(1, -1, 3600)
        if sim_qf.numel() > 0:
            th_qf = torch.quantile(sim_qf.flatten(), 0.8)
            sim_qf = torch.mean(sim_qf, dim=1)  # 取平均 对应support img 与Q前景相关 所有pixel
        else:
            print('------ pred qf mask is empty! ------')
        sim_qb = sim[qb_mask].reshape(1, -1, 3600)
        if sim_qb.numel() > 0:
            th_qb = torch.quantile(sim_qb.flatten(), 0.8)
            sim_qb = torch.mean(sim_qb, dim=1)  # 取平均 对应support img 与Q背景相关 所有pixel
        else:
            print('------ pred qb mask is empty! ------')
        sf_mask = pd_s.argmax(dim=1).view(1, 3600)

        null_mask = torch.zeros([1, 3600], dtype=torch.bool)
        null_mask = null_mask.cuda() if torch.cuda.is_available() else null_mask
        ig_mask1 = (sim_qf > th_qf) & (sf_mask == 0) if sim_qf.numel() > 0 else null_mask
        ig_mask3 = (sim_qb > th_qb) & (sf_mask == 1) if sim_qb.numel() > 0 else null_mask
        ig_mask2 = (sim_qf > th_qf) & (sim_qb > th_qb) if sim_qf.numel() > 0 and sim_qb.numel() > 0 else null_mask
        ig_mask = ig_mask1 | ig_mask2 | ig_mask3 | s_mask[:,1,:]

        return ig_mask


class CosCls(nn.Module):
    def __init__(self, in_dim=512, n_classes=2, WeightNormR=False, cls_type = '000'):
        super(CosCls, self).__init__()
        self.weight_norm, self.bias, self.temp = parse_param_coscls(cls_type)
        self.WeightNormR = WeightNormR
        self.cls = nn.Conv2d(in_dim, n_classes, kernel_size=1, bias=self.bias)
        if self.WeightNormR:
            WeightNorm.apply(self.cls, 'weight', dim=0) #split the weight update component to direction and norm
        if self.temp:
            self.scale_factor = nn.Parameter(torch.tensor(2.0))
        else:
            self.scale_factor = 2.0

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1, eps=0.00001)  # [B, ch, h, w]
        if self.weight_norm:
            self.cls.weight.data = F.normalize(self.cls.weight.data, p=2, dim=1, eps=0.00001)

        cos_dist = self.cls(x_norm)   #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist)
        return scores

    def reset_parameters(self):   # 与torch自己的method同名
        self.cls.reset_parameters()



def parse_param_coscls(cls_type):
    weight_norm_dt = {'n': True, '0': False, 'o': False}
    bias_dt = {'b': True, '0': False, 'o': False}
    temp_dt = {'t': True, '0': False, 'o': False}
    print('weight norm {}, bias {}, temp {}'.format( weight_norm_dt[cls_type[0]], bias_dt[cls_type[1]], temp_dt[cls_type[2]] ))
    return weight_norm_dt[cls_type[0]], bias_dt[cls_type[1]], temp_dt[cls_type[2]]


def get_classifier(args, num_classes=None):
    if num_classes is None:
        num_classes = args.num_classes_tr
    in_dim = args.bottleneck_dim

    if args.get('dist', 'dot') == 'dot':
        return nn.Conv2d(in_dim, num_classes, kernel_size=1, bias=False)
    elif args.get('dist') == 'cos':
        return CosCls(in_dim=in_dim, n_classes=num_classes, WeightNormR=False, cls_type=args.cls_type)
    elif args.get('dist') == 'cosN':  # adaptive weight norm
        return CosCls(in_dim=in_dim, n_classes=num_classes, WeightNormR=True,  cls_type=args.cls_type)
