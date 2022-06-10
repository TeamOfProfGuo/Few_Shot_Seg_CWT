# encoding:utf-8
import pdb

import os
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from collections import defaultdict
from .model import *
from .model.pspnet import get_model
from .model.transformer import CrossAttention, MHA, AttentionBlock, DynamicFusion
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, AverageMeter, get_wt_loss, get_aux_loss
from .util import load_cfg_from_cfg_file, merge_cfg_from_list, ensure_path, set_log_path, log
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Training classifier weight transformer')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main(args: argparse.Namespace) -> None:

    sv_path = 'fuse_{}/{}{}/shot{}_split{}/{}'.format(
        args.train_name, args.arch, args.layers, args.train_split, args.shot, args.exp_name)
    sv_path = os.path.join('./results', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))

    log(args)

    if args.manual_seed is not None:
        cudnn.benchmark = False  # 为True的话可以对网络结构固定、网络的输入形状不变的 模型提速
        cudnn.deterministic = True
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    # ====== Model + Optimizer ======
    model = get_model(args).cuda()

    if args.resume_weights:
        fname = args.resume_weights + args.train_name + '/' + \
                'split={}/pspnet_{}{}/best.pth'.format(args.train_split, args.arch, args.layers)
        if os.path.isfile(fname):
            log("=> loading weight '{}'".format(fname))
            pre_weight = torch.load(fname)['state_dict']
            pre_dict = model.state_dict()

            for index, key in enumerate(pre_dict.keys()):
                if 'classifier' not in key and 'gamma' not in key:
                    if pre_dict[key].shape == pre_weight['module.' + key].shape:
                        pre_dict[key] = pre_weight['module.' + key]
                    else:
                        log('Pre-trained shape and model shape for {}: {}, {}'.format(
                            key, pre_weight['module.' + key].shape, pre_dict[key].shape))
                        continue

            model.load_state_dict(pre_dict, strict=True)
            log("=> loaded weight '{}'".format(fname))
        else:
            log("=> no weight found at '{}'".format(fname))

        # Fix the backbone layers
        for param in model.layer0.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = False
        for param in model.ppm.parameters():
            param.requires_grad = False
        for param in model.bottleneck.parameters():
            param.requires_grad = False

    # ========= Data  ==========
    train_loader, train_sampler = get_train_loader(args)
    episodic_val_loader, _ = get_val_loader(args)

    # ======= Transformer =======
    FusionNet = FuseNet(im_size=30, mid_dim=256).cuda()
    optimizer_meta = get_optimizer(args, [dict(params=FusionNet.parameters(), lr=args.trans_lr * args.scale_lr)])
    scheduler = get_scheduler(args, optimizer_meta, len(train_loader))

    # ====== Metrics initialization ======
    max_val_mIoU = 0.

    # ====== Training  ======
    log('==> Start training')
    for epoch in range(1, args.epochs+1):

        train_loss_meter = AverageMeter()
        train_iou_meter = AverageMeter()
        train_loss_meter0 = AverageMeter()
        train_iou_meter0 = AverageMeter()

        iterable_train_loader = iter(train_loader)
        for i in range(1, len(train_loader)+1):
            qry_img, q_label, spt_imgs, s_label, subcls, _, _ = iterable_train_loader.next()  # q: [1, 3, 473, 473], s: [1, 1, 3, 473, 473]

            if torch.cuda.is_available():
                spt_imgs = spt_imgs.cuda()  # [1, 1, 3, h, w]
                s_label = s_label.cuda()  # [1, 1, h, w]
                q_label = q_label.cuda()  # [1, h, w]
                qry_img = qry_img.cuda()  # [1, 3, h, w]

            # ====== Phase 1: Train the binary classifier on support samples ======

            spt_imgs = spt_imgs.squeeze(0)       # [n_shots, 3, img_size, img_size]
            s_label = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]

            # fine-tune classifier
            model.eval()
            with torch.no_grad():
                f_s, fs_lst = model.extract_features(spt_imgs)  # f_s为ppm之后的feat, fs_lst为mid_feat
            model.inner_loop(f_s, s_label)

            # ====== Phase 2: Train the attention to update query score  ======
            model.eval()
            with torch.no_grad():
                f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
                pd_q0 = model.classifier(f_q)
                pd_s  = model.classifier(f_s)
                pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

            # filter out ignore pixels
            fs_fea = fs_lst[-1]  # [1, 2048, 60, 60]
            fq_fea = fq_lst[-1]  # [1, 2048, 60, 60]
            pd_q1, ret_corr, ret_mk = model.outer_forward(f_q, f_s, fq_fea, fs_fea, s_label, q_label, pd_q0, pd_s, ret_curr='cr_mk')

            corr, weighted_v = ret_corr                   # [B, h, w, h_s, w_s], [B, 512, h, w]
            sim_qf, sim_qb = ret_mk                       # [1, 3600]
            pd_mask0 = pd_q0.argmax(dim=1).unsqueeze(1)   # [B, 1, h, w]

            s_mask = torch.clone(s_label)
            s_mask[s_mask == 255] = 0
            s_mask = F.interpolate(s_mask.unsqueeze(1).float(), (30, 30), mode='bilinear', align_corners=True)
            sim_qf = F.interpolate(sim_qf.view(corr.shape[:3]).unsqueeze(1), (30, 30), mode='bilinear', align_corners=True)
            sim_qb = F.interpolate(sim_qb.view(corr.shape[:3]).unsqueeze(1), (30, 30), mode='bilinear', align_corners=True)

            wt = FusionNet(corr, pd_mask0, corr_fg=sim_qf, corr_bg=sim_qb, s_mask=s_mask)
            out = weighted_v * wt + f_q * (1-wt)
            pd_q = model.classifier(out)
            pred_q1= F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)
            pred_q = F.interpolate(pd_q , size=q_label.shape[-2:], mode='bilinear', align_corners=True)

            # Loss function: Dynamic class weights used for query image only during training
            q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
            q_back_pix = np.where(q_label_arr == 0)
            q_target_pix = np.where(q_label_arr == 1)
            loss_weight = torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)]).cuda()
            criterion = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
            q_loss = criterion(pred_q, q_label.long())
            q_loss0 = criterion(pred_q0, q_label.long())

            # auxiliary loss
            wt_loss = 0
            loss = q_loss

            optimizer_meta.zero_grad()
            loss.backward()
            optimizer_meta.step()
            if args.scheduler == 'cosine':
                scheduler.step()

            # Print loss and mIoU
            intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, args.num_classes_tr, 255)
            IoUf, IoUb = (intersection / (union + 1e-10)).cpu().numpy()  # mean of BG and FG
            train_loss_meter.update(q_loss.item() / args.batch_size, 1)
            train_iou_meter.update((IoUf+IoUb)/2, 1)

            intersection0, union0, target0 = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, args.num_classes_tr, 255)
            IoUf0, IoUb0 = (intersection0 / (union0 + 1e-10)).cpu().numpy()  # mean of BG and FG
            train_loss_meter0.update(q_loss0.item() / args.batch_size, 1)
            train_iou_meter0.update((IoUf0+IoUb0)/2, 1)

            intersection1, union1, target1 = intersectionAndUnionGPU(pred_q1.argmax(1), q_label, args.num_classes_tr, 255)
            IoUf1, IoUb1 = (intersection1 / (union1 + 1e-10)).cpu().numpy()  # mean of BG and FG

            if i%100==0 or (epoch==1 and i%10==0):
                log('Epoch{} Iter{} IoUf0 {:.2f} IoUb0 {:.2f} IoUf {:.2f} IoUb {:.2f} IoUf1 {:.2f} IoUb1 {:.2f} '
                    'q_loss {:.2f} wt_loss {:.2f} avg_wt {:.2f} std_wt {:.2f} lr {:.4f}'.format(
                    epoch, i, IoUf0, IoUb0, IoUf, IoUb, IoUf1, IoUb1,
                    q_loss, wt_loss, torch.mean(wt), torch.std(wt), optimizer_meta.param_groups[0]['lr']))
            if i%900==0:
                val_Iou, val_loss = validate_epoch(args=args, val_loader=episodic_val_loader, model=model, Net=FusionNet)

                # Model selection
                if val_Iou.item() > max_val_mIoU:
                    max_val_mIoU = val_Iou.item()

                    filename_transformer = os.path.join(sv_path, f'best.pth')

                    if args.save_models:
                        log('Saving checkpoint to: ' + filename_transformer)
                        torch.save({'epoch': epoch,
                                    'state_dict': model.state_dict(),
                                    'optimizer': optimizer_meta.state_dict()},
                                   filename_transformer)

                log("=> Max_mIoU = {:.3f}".format(max_val_mIoU))

        log('===========Epoch {}===========: The mIoU0 {:.2f}, mIoU {:.2f}, loss0 {:.2f}, loss {:.2f}==========='.format(
            epoch, train_iou_meter0.avg, train_iou_meter.avg, train_loss_meter0.avg,
            train_loss_meter.avg))
        train_iou_meter.reset()
        train_loss_meter.reset()


                # For debugging
                # if i%50 == 0:
                #     os.makedirs(trans_save_dir, exist_ok=True)
                #     filename_transformer = os.path.join(trans_save_dir, 'debug_iter{}.pth'.format(i))
                #
                #     if args.save_models:
                #         print('Saving checkpoint to: ' + filename_transformer)
                #         torch.save({'epoch': epoch,
                #                     'state_dict': model.state_dict(),
                #                     'optimizer': optimizer_meta.state_dict()},
                #                    filename_transformer)
                #     print('save ckpt to {}'.format(filename_transformer))

    if args.save_models:  # 所有跑完，存last epoch
        filename_transformer = os.path.join(sv_path, 'final.pth')
        torch.save(
            {'epoch': args.epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer_meta.state_dict()},
            filename_transformer)


def validate_epoch(args, val_loader, model, Net):
    log('==> Start testing')

    iter_num = 0
    start_time = time.time()
    loss_meter = AverageMeter()
    cls_intersection = defaultdict(int)  # Default value is 0
    cls_union = defaultdict(int)
    IoU = defaultdict(float)

    cls_intersection0 = defaultdict(int)  # Default value is 0
    cls_union0 = defaultdict(int)
    IoU0 = defaultdict(float)

    for e in range(args.test_num):

        iter_num += 1
        try:
            qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        except:
            iter_loader = iter(val_loader)
            qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        if torch.cuda.is_available():
            spt_imgs = spt_imgs.cuda()
            s_label = s_label.cuda()
            q_label = q_label.cuda()
            qry_img = qry_img.cuda()

        # ====== Phase 1: Train a new binary classifier on support samples. ======
        spt_imgs = spt_imgs.squeeze(0)   # [n_shots, 3, img_size, img_size]
        s_label = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]

        # fine-tune classifier
        model.eval()
        with torch.no_grad():
            f_s, fs_lst = model.extract_features(spt_imgs)
        model.inner_loop(f_s, s_label)

        # ====== Phase 2: Update query score using attention. ======
        model.eval()
        with torch.no_grad():
            f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
            pd_q0 = model.classifier(f_q)
            pd_s  = model.classifier(f_s)
            pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

        fs_fea = fs_lst[-1]  # [2, 2048, 60, 60]
        fq_fea = fq_lst[-1]  # [1, 2048, 60, 60]
        pd_q1, ret_corr, ret_mk = model.outer_forward(f_q, f_s, fq_fea, fs_fea, s_label, q_label, pd_q0, pd_s, ret_curr='cr_mk')

        corr, weighted_v = ret_corr  # [B, h, w, h_s, w_s], [B, 512, h, w]
        sim_qf, sim_qb = ret_mk  # [1, 3600]
        pd_mask0 = pd_q0.argmax(dim=1).unsqueeze(1)  # [B, 1, h, w]

        s_mask = torch.clone(s_label)
        s_mask[s_mask == 255] = 0
        s_mask = F.interpolate(s_mask.unsqueeze(1).float(), (30, 30), mode='bilinear', align_corners=True)
        sim_qf = F.interpolate(sim_qf.view(corr.shape[:3]).unsqueeze(1), (30, 30), mode='bilinear', align_corners=True)
        sim_qb = F.interpolate(sim_qb.view(corr.shape[:3]).unsqueeze(1), (30, 30), mode='bilinear', align_corners=True)

        wt = Net(corr, pd_mask0, corr_fg=sim_qf, corr_bg=sim_qb, s_mask=s_mask)
        out = weighted_v * wt + f_q * (1 - wt)
        pd_q = model.classifier(out)
        pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)
        pred_q = F.interpolate(pd_q, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

        # IoU and loss
        curr_cls = subcls[0].item()  # 当前episode所关注的cls
        intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, 2, 255)
        intersection, union = intersection.cpu(), union.cpu()
        cls_intersection[curr_cls] += intersection[1]  # only consider the FG
        cls_union[curr_cls] += union[1]                # only consider the FG
        IoU[curr_cls] = cls_intersection[curr_cls] / (cls_union[curr_cls] + 1e-10)   # cls wise IoU

        intersection0, union0, target0 = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, 2, 255)
        intersection0, union0 = intersection0.cpu(), union0.cpu()
        cls_intersection0[curr_cls] += intersection0[1]  # only consider the FG
        cls_union0[curr_cls] += union0[1]  # only consider the FG
        IoU0[curr_cls] = cls_intersection0[curr_cls] / (cls_union0[curr_cls] + 1e-10)  # cls wise IoU

        criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
        loss = criterion_standard(pred_q, q_label)
        loss_meter.update(loss.item())

        if (iter_num % 200 == 0):
            mIoU = np.mean([IoU[i] for i in IoU])                                  # mIoU across cls
            mIoU0 = np.mean([IoU0[i] for i in IoU0])
            log('Test: [{}/{}] mIoU0 {:.4f} mIoU {:.4f} Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                iter_num, args.test_num, mIoU0, mIoU, loss_meter=loss_meter))

    runtime = time.time() - start_time
    mIoU = np.mean(list(IoU.values()))  # IoU: dict{cls: cls-wise IoU}
    log('mIoU---Val result: mIoU0 {:.4f}, mIoU {:.4f} | time used {:.1f}m.'.format(mIoU0, mIoU, runtime/60))
    for class_ in cls_union:
        log("Class {} : {:.4f}".format(class_, IoU[class_]))
    log('\n')

    return mIoU, loss_meter.avg


if __name__ == "__main__":
    args = parse_args()

    main(args)