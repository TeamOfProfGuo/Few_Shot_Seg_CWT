# encoding:utf-8

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
from .model.pspnet import get_model
from .model.transformer import MultiHeadAttentionOne
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, get_model_dir, AverageMeter, get_model_dir_trans
from .test import validate_transformer
import argparse
from typing import Tuple
from .util import load_cfg_from_cfg_file, merge_cfg_from_list


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
    print(args)

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
            print("=> loading weight '{}'".format(fname))
            pre_weight = torch.load(fname)['state_dict']
            pre_dict = model.state_dict()

            for index, key in enumerate(pre_dict.keys()):
                if 'classifier' not in key and 'gamma' not in key:
                    if pre_dict[key].shape == pre_weight['module.' + key].shape:
                        pre_dict[key] = pre_weight['module.' + key]
                    else:
                        print('Pre-trained shape and model shape for {}: {}, {}'.format(
                            key, pre_weight['module.' + key].shape, pre_dict[key].shape))
                        continue

            model.load_state_dict(pre_dict, strict=True)
            print("=> loaded weight '{}'".format(fname))
        else:
            print("=> no weight found at '{}'".format(fname))

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

    # ======= Transformer =======
    param_list = [model.gamma]
    optimizer_meta = get_optimizer(args,[dict(params=param_list, lr=args.trans_lr * args.scale_lr)])
    trans_save_dir = os.path.join(args.model_dir,args.train_name,f'split={args.train_split}',f'shot_{args.shot}',f'{args.arch}{args.layers}')

    # ========= Data  ==========
    train_loader, train_sampler = get_train_loader(args)
    episodic_val_loader, _ = get_val_loader(args)

    # ====== Metrics initialization ======
    max_val_mIoU = 0.
    iter_per_epoch = args.iter_per_epoch if args.iter_per_epoch <= len(train_loader) else len(train_loader)

    # ====== Training  ======
    print('==> Start training')
    for epoch in range(args.epochs):

        train_loss_meter = AverageMeter()
        train_iou_meter = AverageMeter()
        train_loss_meter0 = AverageMeter()
        train_iou_meter0 = AverageMeter()

        iterable_train_loader = iter(train_loader)
        for i in range(iter_per_epoch):
            qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iterable_train_loader.next()  # q: [1, 3, 473, 473], s: [1, 1, 3, 473, 473]

            if torch.cuda.is_available():
                spprt_imgs = spprt_imgs.cuda()  # [1, 1, 3, h, w]
                s_label = s_label.cuda()  # [1, 1, h, w]
                q_label = q_label.cuda()  # [1, h, w]
                qry_img = qry_img.cuda()  # [1, 3, h, w]

            # ====== Phase 1: Train the binary classifier on support samples ======

            if spprt_imgs.shape[1] == 1:
                spprt_imgs_reshape = spprt_imgs.squeeze(0).expand(2, 3, args.image_size, args.image_size)
                s_label_reshape = s_label.squeeze(0).expand(2, args.image_size, args.image_size).long()
            else:
                spprt_imgs_reshape = spprt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
                s_label_reshape = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]

            # fine-tune classifier
            model.train()
            f_s, fs_lst = model.extract_features(spprt_imgs_reshape)
            model.inner_loop(f_s, s_label_reshape)

            # ====== Phase 2: Train the attention to update query score  ======
            # query score: baseline model vs. attention based
            model.eval()
            with torch.no_grad():
                f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
                pred_q0 = model.classifier(f_q)
                pred_q0 = F.interpolate(pred_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)
            # 基于attention refine pred_q
            fs_fea = fs_lst[-1]  # [2, 2048, 60, 60]
            fq_fea = fq_lst[-1]  # [1, 2048, 60, 60]
            pred_q = model.outer_forward(f_q, f_s[0:1], fq_fea, fs_fea[0:1], s_label_reshape[0:1])
            pred_q = F.interpolate(pred_q, size=q_label.shape[1:], mode='bilinear', align_corners=True)

            # Loss function: Dynamic class weights used for query image only during training
            q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
            q_back_pix = np.where(q_label_arr == 0)
            q_target_pix = np.where(q_label_arr == 1)
            loss_weight = torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)]).cuda()
            criterion = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
            q_loss = criterion(pred_q, q_label.long())
            q_loss0 = criterion(pred_q0, q_label.long())

            optimizer_meta.zero_grad()
            q_loss.backward()
            optimizer_meta.step()

            # Print loss and mIoU
            intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, args.num_classes_tr, 255)
            mIoU = (intersection / (union + 1e-10)).mean()  # mean of BG and FG
            train_loss_meter.update(q_loss.item() / args.batch_size, 1)
            train_iou_meter.update(mIoU, 1)
            intersection0, union0, target0 = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, args.num_classes_tr, 255)
            mIoU0 = (intersection0 / (union0 + 1e-10)).mean()  # mean of BG and FG
            train_loss_meter0.update(q_loss0.item() / args.batch_size, 1)
            train_iou_meter0.update(mIoU0, 1)
            print('Epoch {} Iter {} mIoU0 {:.2f} mIoU {:.2f}'.format(epoch+1, i, mIoU0, mIoU))

            if i % 1000 == 0:
                print('Epoch {}: The mIoU0 {:.2f}, mIoU {:.2f}, loss0 {:.2f}, loss {:.2f}, gamma {:.4f}'.format(
                    epoch + 1, train_iou_meter0.avg, train_iou_meter.avg, train_loss_meter0.avg,
                    train_loss_meter.avg, model.gamma.item()))
                train_iou_meter.reset()
                train_loss_meter.reset()

                val_Iou, val_loss = validate_epoch(args=args, val_loader=episodic_val_loader, model=model)

                # Model selection
                if val_Iou.item() > max_val_mIoU:
                    max_val_mIoU = val_Iou.item()

                    os.makedirs(trans_save_dir, exist_ok=True)
                    filename_transformer = os.path.join(trans_save_dir, f'best.pth')

                    if args.save_models:
                        print('Saving checkpoint to: ' + filename_transformer)
                        torch.save( {'epoch': epoch,
                                     'state_dict': model.state_dict(),
                                     'optimizer': optimizer_meta.state_dict()},
                                    filename_transformer)

                print("=> Max_mIoU = {:.3f}".format(max_val_mIoU))

    if args.save_models:  # 所有跑完，存last epoch
        filename_transformer = os.path.join(trans_save_dir, 'final.pth')
        torch.save(
            {'epoch': args.epochs,
             'state_dict': model.state_dict(),
             'optimizer': optimizer_meta.state_dict()},
            filename_transformer)


def validate_epoch(args, val_loader, model):
    print('==> Start testing')

    iter_num = 0
    start_time = time.time()
    loss_meter = AverageMeter()
    cls_intersection = defaultdict(int)  # Default value is 0
    cls_union = defaultdict(int)
    IoU = defaultdict(float)

    for e in range(args.test_num):

        iter_num += 1
        try:
            qry_img, q_label, spprt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        except:
            iter_loader = iter(val_loader)
            qry_img, q_label, spprt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        if torch.cuda.is_available():
            spprt_imgs = spprt_imgs.cuda()
            s_label = s_label.cuda()
            q_label = q_label.cuda()
            qry_img = qry_img.cuda()

        # ====== Phase 1: Train a new binary classifier on support samples. ======
        spprt_imgs = spprt_imgs.squeeze(0)   # [n_shots, 3, img_size, img_size]
        s_label = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]

        # fine-tune classifier
        model.eval()
        with torch.no_grad():
            f_s, fs_lst = model.extract_features(spprt_imgs)
        model.inner_loop(f_s, s_label)

        # ====== Phase 2: Update query score using attention. ======
        model.eval()
        with torch.no_grad():
            f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
            pred_q0 = model.classifier(f_q)
            pred_q0 = F.interpolate(pred_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)
        # 用layer4 的output来做attention
        fs_fea = fs_lst[-1]   # [1, 2048, 60, 60]
        fq_fea = fq_lst[-1]  # [1, 2048, 60, 60]
        pred_q = model.outer_forward(f_q, f_s, fq_fea, fs_fea, s_label)
        pred_q = F.interpolate(pred_q, size=q_label.shape[1:], mode='bilinear', align_corners=True)

        # IoU and loss
        curr_cls = subcls[0].item()  # 当前episode所关注的cls
        intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, 2, 255)
        intersection, union = intersection.cpu(), union.cpu()
        cls_intersection[curr_cls] += intersection[1]  # only consider the FG
        cls_union[curr_cls] += union[1]                # only consider the FG
        IoU[curr_cls] = cls_intersection[curr_cls] / (cls_union[curr_cls] + 1e-10)   # cls wise IoU

        criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
        loss = criterion_standard(pred_q, q_label)
        loss_meter.update(loss.item())

        if (iter_num % 200 == 0):
            mIoU = np.mean([IoU[i] for i in IoU])                                  # mIoU across cls
            print('Test: [{}/{}] mIoU {:.4f} Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                iter_num, args.test_num, mIoU, loss_meter=loss_meter))

    runtime = time.time() - start_time
    mIoU = np.mean(list(IoU.values()))  # IoU: dict{cls: cls-wise IoU}
    print('mIoU---Val result: mIoU {:.4f} | time used {:.1f}m.'.format(mIoU, runtime/60))
    for class_ in cls_union:
        print("Class {} : {:.4f}".format(class_, IoU[class_]))

    return mIoU, loss_meter.avg


if __name__ == "__main__":
    args = parse_args()

    main(args)