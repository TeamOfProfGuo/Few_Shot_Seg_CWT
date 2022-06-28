# encoding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from .match import MatchNet
from .msm import MSBlock, WeightAverage


class MMN(nn.Module):
    def __init__(self, args, inner_channel=32, sem=True, wa=False):
        super().__init__()
        self.args = args     # rmid
        self.sem = sem
        self.wa = wa

        if self.args.layers == 50:
            self.nbottlenecks = [3, 4, 6, 3]
            self.feature_channels = [256, 512, 1024, 2048]

        self.bids = [int(num) for num in list(args.rmid[1:])]

        for i, b in enumerate(self.bids):
            c_in = self.feature_channels[b-1]
            if self.sem:
                setattr(self, "msblock"+str(b), MSBlock(c_in, rate=4, c_out=inner_channel))
                c_in = inner_channel

            if self.wa:
                setattr(self, "wa_"+str(b), WeightAverage(c_in))

        self.corr_net = MatchNet(temp=args.temp, cv_type='red', sce=False, cyc=False, sym_mode=True)

    def forward(self, fq_lst, fs_lst, f_q, f_s, padding_mask=None, s_padding_mask=None):
        fq1, fq2, fq3, fq4 = fq_lst
        fs1, fs2, fs3, fs4 = fs_lst

        if self.sem:
            fq4 = self.msblock4(fq4)   # [B, 32, 60, 60]
            fs4 = self.msblock4(fs4)   # [B, 32, 60, 60]
        if self.wa:
            fq4 = self.wa_4(fq4)
            fs4 = self.wa_4(fs4)

        att_fq4 = self.corr_net(fq4, fs4, f_s, s_mask=None, ig_mask=None, ret_corr=False, use_cyc=False, ret_cyc=False)

        fq = F.normalize(f_q, p=2, dim=1) + F.normalize(att_fq4, p=2, dim=1) * self.args.att_wt

        return fq, att_fq4

    @staticmethod
    def compute_loss(pred, label, loss_type='ce'):   # loss_type: ['ce', 'wt_ce',
        count = torch.bincount(label.view(-1))
        weight = torch.tensor([1.0, count[0]/count[1]])
        if torch.cuda.is_available():
            weight = weight.cuda()
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=255)
        return criterion(pred, label)


class SegLoss(nn.Module):
    def __init__(self, weighted_val: float = 1.0,  reduction: str = "sum",):
        super().__init__()
        self.weighted_val = weighted_val
        self.reduction = reduction

    def forward(self, prediction, target_seg,):
        return weighted_dice_loss(prediction, target_seg, self.weighted_val, self.reduction,)


def weighted_dice_loss(
        prediction,
        target_seg,
        weighted_val: float = 1.0,
        reduction: str = "sum",
        eps: float = 1e-8,
):
    """
    Weighted version of Dice Loss

    Args:
        prediction: prediction
        target_seg: segmentation target
        weighted_val: values of k positives,
        reduction: 'none' | 'mean' | 'sum'
        eps: the minimum eps,
    """
    target_seg_fg = target_seg == 1
    target_seg_bg = target_seg == 0
    target_seg = torch.stack([target_seg_bg, target_seg_fg], dim=1).float()

    n, _, h, w = target_seg.shape

    prediction = prediction.reshape(-1, h, w)  # [B*2, h, w]
    target_seg = target_seg.reshape(-1, h, w)  # [B*2, h, w]
    prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(-1, h * w)  # [B*2, h*w]
    target_seg = target_seg.reshape(-1, h * w)  # [B*2, h*w]

    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)  # [B*2]
    # normalize the loss
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum() / n
    elif reduction == "mean":
        loss = loss.mean()
    return loss