# encoding:utf-8

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class SegLoss(nn.Module):
    def __init__(self, loss_type='wt_ce', num_cls=2, fg_idx=1):   # loss_type ['wt_ce', 'wt_dc']
        super().__init__()
        self.loss_type = loss_type
        self.num_cls = num_cls
        self.fg_idx = fg_idx

    def forward(self, prediction, target_seg, input_type='lg'):
        """whether prediction has already processed by CrossEntropy: 'pb':prob, 'lt':lg"""
        if self.loss_type=='wt_dc' or self.loss_type=='dc':
            return weighted_dice_loss(prediction, target_seg, reduction='sum', input_type=input_type)
        elif self.loss_type == 'ce':
            criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
            return criterion_standard(prediction, target_seg)
        else:
            return weighted_ce_loss(prediction, target_seg, ignore_index=255, num_cls=self.num_cls, fg_idx=self.fg_idx)


def weighted_ce_loss(pred, label, ignore_index=255, num_cls=2, fg_idx=1):
    count = torch.bincount(label.view(-1))
    fg_cnt = count[fg_idx]
    bg_cnt = (torch.sum(count)-fg_cnt) if len(count)<=255 else (torch.sum(count)-count[255] -fg_cnt) # all pixels not belonging to current cls CONSIDERED as BG
    # bg_cnt = count[0]
    weight = torch.tensor([1.0]*num_cls)
    weight[fg_idx] = bg_cnt/fg_cnt
    if torch.cuda.is_available():
        weight = weight.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    return criterion(pred, label)


def weighted_dice_loss(prediction, target_seg, weighted_val=1.0, reduction="sum", input_type='lg', eps=1e-8,):
    """
    Weighted version of Dice Loss
    Args:
        prediction: prediction
        target_seg: segmentation target
        weighted_val: values of k positives, float
        reduction: 'none' | 'mean' | 'sum'
        eps: the minimum eps,
    """
    target_seg_fg = target_seg == 1  # [B, h, w]
    target_seg_bg = target_seg == 0  # [B, h, w]
    target_seg = torch.stack([target_seg_bg, target_seg_fg], dim=1).float()  # [B, 2, h, w] get rid of ignore pixels 255

    n, _, h, w = target_seg.shape

    prediction = prediction.reshape(-1, h, w)  # [B*2, h, w]
    target_seg = target_seg.reshape(-1, h, w)  # [B*2, h, w]
    if input_type in ['lg', 'lt']:
        prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(-1, h * w)  # [B*2, h*w]
    target_seg = target_seg.reshape(-1, h * w)  # [B*2, h*w]

    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)    # [B*2]
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)  # [B*2]
    # normalize the loss
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum() / n
    elif reduction == "mean":
        loss = loss.mean()
    return loss


class Adapt_SegLoss(nn.Module):
    def __init__(self, num_cls=2, fg_idx=1, tp=1.0):   # loss_type ['wt_ce', 'wt_dc']
        super().__init__()
        self.num_cls = num_cls
        self.fg_idx = fg_idx
        self.tp = tp

    def forward(self, prediction, target_seg, ):
        """whether prediction has already processed by CrossEntropy: 'pb':prob, 'lt':lg"""
        return weighted_adpt_ce_loss(prediction, target_seg, ignore_index=255, num_cls=self.num_cls, fg_idx=self.fg_idx, tp=self.tp)


def weighted_adpt_ce_loss(pred, label, ignore_index=255, num_cls=2, fg_idx=1, tp=1.0):
    count = torch.bincount(label.view(-1))
    fg_cnt = count[fg_idx]
    bg_cnt = (torch.sum(count) - fg_cnt) if len(count) <= 255 else (torch.sum(count) - count[255] - fg_cnt)  # all pixels not belonging to current cls CONSIDERED as BG
    weight = torch.tensor([1.0] * num_cls)
    weight[fg_idx] = (bg_cnt / fg_cnt)**tp
    if torch.cuda.is_available():
        weight = weight.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    return criterion(pred, label)



def get_corr(q, k):
    bs, ch, height, width = q.shape
    proj_q = q.view(bs, ch, height * width).permute(0, 2, 1)  # [1, ch, hw] -> [1, hw, ch]
    proj_k = k.view(bs, -1, height * width)  # [1, ch, hw]

    proj_q = F.normalize(proj_q, dim=-1)
    proj_k = F.normalize(proj_k, dim=-2)
    sim = torch.bmm(proj_q, proj_k)  # [1, 3600 (q_hw), 3600(k_hw)]
    return sim


def reset_cls_wt(cls_module, pre_cls_wt, num_classes_tr, idx_cls):
    B, ch, _, _ = cls_module.weight.shape
    cls_module.weight.data[:num_classes_tr] = pre_cls_wt
    std = 1.0 / np.sqrt(ch)
    cls_module.weight.data[idx_cls] = torch.empty([ch], device=cls_module.weight.device).uniform_(-std, std).reshape((1, ch, 1, 1))


def reset_spt_label(s_label, pred, idx_cls):
    """modify s_label, BG will be replaced with the pseudo label from base cls, FG idx updated to idx for multi-way"""
    pred[:, idx_cls, :, :] = -1000.0  # [1, 16/17, 473, 473]
    pred_mask = pred.argmax(1)        # [1, 473, 473]

    idx_bg = torch.where(s_label == 0)
    s_label[idx_bg] = pred_mask[idx_bg]
    s_label[torch.where(s_label == 1)] = idx_cls
    return s_label


def adapt_reset_spt_label(s_label, pred, pre_cls_wt, num_classes_tr, sub_cls=None):
    """modify s_label, BG will be replaced with the pseudo label from base cls, FG idx updated to idx for multi-way
    adapt: num of classes 根据spt图片中cls数量决定
    """
    pred_mask = pred.argmax(1)        # [1, 473, 473]

    if sub_cls is not None and sub_cls > 0:                   # ONLY for meta train stage
        pred_mask[ pred_mask == sub_cls ] = 0

    s_label[s_label==1] = num_classes_tr   # FG to a temporary idx avoid conflict with base class id
    idx_bg = torch.where(s_label == 0)
    s_label[idx_bg] = pred_mask[idx_bg]

    num_cls = 2
    cls_init_wt = []
    id_freq = torch.bincount(s_label.flatten())  # id: cls_idx val: freq  当前cls的id为16/num_classes_tr
    for i in range(1, min(len(id_freq), num_classes_tr)):
        if 0 < id_freq[i] <= 300 * len(s_label):
            s_label[ s_label == i] = 0
        elif id_freq[i] > 300 * len(s_label) and 0< i <num_classes_tr:  # 加一个新的cls
            s_label[ s_label ==i ] = num_cls
            num_cls += 1
            cls_init_wt.append(pre_cls_wt[i])  # 存其 initial weight

    s_label[s_label==num_classes_tr] = 1
    return s_label, cls_init_wt, num_cls


def compress_pred(pred, idx_cls, input_type='lg'):
    """combine all BG with Base class to compute binary loss. input_type: 'lg', 'pb' """
    if input_type in ['lg', 'lt']:
        pred = F.softmax(pred)  # [B, nC, h, w]
    B, C, h, w = pred.shape
    new = torch.zeros((B, 2, h, w), device=pred.device)
    new[:, 1, :, :] = pred[:, idx_cls, :, :]
    new[:, 0, :, :] = 1.0 - new[:, 1, :, :]
    return new

def pred2bmask(pred, idx_cls=1):
    pred = pred.argmax(dim=1)
    for idx in pred.unique():
        if idx>0 and idx!= idx_cls:
            pred[pred==idx] = 0

    pred[pred==idx_cls] =1
    return pred


def get_ig_mask(sim, s_label, q_label, pd_q0, pd_s): # sim [1, q_hw, k_hw]
    B, _, h, w = pd_q0.shape
    if sim.dim() == 5:
        sim = sim.reshape(B, h*w, h*w)

    # mask ignored support pixels
    s_mask = F.interpolate(s_label.unsqueeze(1).float(), size=(h,w), mode='nearest')  # [1,1,h,w]
    s_mask = (s_mask > 1).view(s_mask.shape[0], -1)  # [n_shot, hw]

    # ignore misleading points
    pd_q_mask0 = pd_q0.argmax(dim=1)
    q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=(h,w), mode='nearest').squeeze(1)  # [1,1,h,w]
    qf_mask = (q_mask != 255.0) * (pd_q_mask0 == 1)  # predicted qry FG
    qb_mask = (q_mask != 255.0) * (pd_q_mask0 == 0)  # predicted qry BG
    qf_mask = qf_mask.view(qf_mask.shape[0], -1, 1).expand(sim.shape)  # 保留query predicted FG
    qb_mask = qb_mask.view(qb_mask.shape[0], -1, 1).expand(sim.shape)  # 保留query predicted BG

    sim_qf = sim[qf_mask].reshape(1, -1, 3600)
    if sim_qf.numel() > 0:
        th_qf = torch.quantile(sim_qf.flatten(), 0.8)
        sim_qf = torch.mean(sim_qf, dim=1)  # 取平均 对应support img 与Q前景相关 所有pixel  [B, 3600_s]
        qf_mask = sim_qf
    else:
        print('------ pred qf mask is empty! ------')
        qf_mask = torch.zeros([1, 3600], dtype=torch.float).cuda()

    sim_qb = sim[qb_mask].reshape(1, -1, 3600)
    if sim_qb.numel() > 0:
        th_qb = torch.quantile(sim_qb.flatten(), 0.8)
        sim_qb = torch.mean(sim_qb, dim=1)  # 取平均 对应support img 与Q背景相关 所有pixel, [B, 3600_s]
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
    ig_mask = ig_mask1 | ig_mask2 | ig_mask3 | s_mask

    return ig_mask  # [B, hw_s]


def att_weighted_out(sim, v, temp=20.0, ig_mask=None):
    B, d_v, h, w = v.shape
    if sim.dim() == 5:
        sim = sim.reshape(B, h*w, h*w)  # [1, hw_q, hw_s]

    if ig_mask is not None:  # not None / False   [B, hw_s]
        ig_mask = ig_mask.unsqueeze(1).expand(sim.shape)
        sim[ig_mask == True] = 0.00001

    attn = F.softmax(sim * temp, dim=-1)
    weighted_v = torch.bmm(v.view(B, d_v, h*w), attn.permute(0, 2, 1))  # [B, 512, N_s] * [B, N_s, N_q] -> [1, 512, N_q]
    weighted_v = weighted_v.view(B, -1, h, w)
    return weighted_v

