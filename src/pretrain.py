# encoding:utf-8

# This code is to pretrain model backbone on the base train data

import os
import time
import yaml
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from tensorboardX import SummaryWriter
from collections import defaultdict
from typing import Tuple, Dict
from torch import Tensor
from .model.pspnet import get_model

from .test import standard_validate, episodic_validate
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, get_model_dir, AverageMeter, get_model_dir_trans
from .util import load_cfg_from_cfg_file, merge_cfg_from_list
from .util import ensure_path, set_log_path, log
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

    sv_path = 'pretrain_{}/{}{}/shot{}_split{}/{}'.format(
        args.train_name, args.arch, args.layers, args.train_split, args.shot, args.exp_name)
    sv_path = os.path.join('./results', sv_path)
    ensure_path(sv_path)
    set_log_path(path=sv_path)
    log('save_path {}'.format(sv_path))
    yaml.dump(args, open(os.path.join(sv_path, 'config.yaml'), 'w'))

    log(args)
    writer = SummaryWriter(os.path.join(sv_path, 'model'))

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
    modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
    modules_new = [model.ppm, model.bottleneck, model.classifier]

    params_list = []
    for module in modules_ori:
        params_list.append(dict(params=module.parameters(), lr=args.lr))
    for module in modules_new:
        params_list.append(dict(params=module.parameters(), lr=args.lr * args.scale_lr))
    optimizer = get_optimizer(args, params_list)

    # ========== Validation ==================
    validate_fn = episodic_validate if args.episodic_val else standard_validate

    # ========= Data  ==========
    train_loader, train_sampler = get_train_loader(args, episodic=False)
    val_loader, _ = get_val_loader(args)  # mode='train' means that we will validate on images from validation set, but with the bases classes

    # ========== Scheduler  ================
    scheduler = get_scheduler(args, optimizer, len(train_loader))

    # ====== Metrics initialization ======
    max_val_mIoU = 0.
    iter_per_epoch = len(train_loader)
    log_iter = int(iter_per_epoch / args.log_freq) + 1

    metrics: Dict[str, Tensor] = {"val_mIou": torch.zeros((args.epochs, 1)).type(torch.float32),
                                  "val_loss": torch.zeros((args.epochs, 1)).type(torch.float32),
                                  "train_mIou": torch.zeros((args.epochs, log_iter)).type(torch.float32),
                                  "train_loss": torch.zeros((args.epochs, log_iter)).type(torch.float32),
                                  }

    # ====== Training  ======
    log('==> Start training')
    for epoch in range(args.epochs):

        loss_meter = AverageMeter()
        iterable_train_loader = iter(train_loader)

        for i in range(1, iter_per_epoch+1):
            model.train()

            images, gt = iterable_train_loader.next()  # q: [1, 3, 473, 473], s: [1, 1, 3, 473, 473]
            if torch.cuda.is_available():
                images = images.cuda()  # [1, 1, 3, h, w]
                gt = gt.cuda()  # [1, 1, h, w]

            loss = compute_loss(args=args, model=model, images=images, targets=gt.long(), num_classes=args.num_classes_tr)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.scheduler == 'cosine':
                scheduler.step()

            if i % args.log_freq == 0:
                model.eval()
                logits = model(images)
                intersection, union, target = intersectionAndUnionGPU(logits.argmax(1), gt, args.num_classes_tr, 255)

                allAcc = (intersection.sum() / (target.sum() + 1e-10))  # scalar
                mAcc = (intersection / (target + 1e-10)).mean()
                mIoU = (intersection / (union + 1e-10)).mean()
                loss_meter.update(loss.item())

                log('iter {}/{}: loss {:.2f}, running loss {:.2f}, Acc {:.4f}, mAcc {:.4f}, mIoU {:.4f}'.format(
                    i, epoch, loss.item(), loss_meter.avg, allAcc, mAcc, mIoU))

        log('============ Epoch {}=============: running loss {:.2f}'.format(epoch, loss_meter.avg))
        writer.add_scalar('train_loss', loss_meter.avg, epoch)
        writer.add_scalar("mean_iou/train", mIoU, epoch)
        writer.add_scalar("pixel accuracy/train", allAcc, epoch)
        loss_meter.reset()

        val_Iou, val_loss = validate_fn(args=args, val_loader=val_loader, model=model, use_callback=False)
        writer.add_scalar("mean_iou/val", val_Iou, epoch)
        writer.add_scalar("pixel accuracy/val", val_loss, epoch)

        # Model selection
        if val_Iou.item() > max_val_mIoU:
            max_val_mIoU = val_Iou.item()

            filename = os.path.join(sv_path, f'best.pth')
            if args.save_models:
                log('=> Max_mIoU = {:.3f}, Saving checkpoint to: {}'.format(max_val_mIoU, filename))
                torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, filename)

    if args.save_models:  # 所有跑完，存last epoch
        filename = os.path.join(sv_path, 'final.pth')
        torch.save( {'epoch': args.epochs, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}, filename )


def cross_entropy(logits: torch.tensor, one_hot: torch.tensor, targets: torch.tensor, mean_reduce: bool = True,
                  ignore_index: int = 255) -> torch.tensor:
    """
    inputs: one_hot  : shape [batch_size, num_classes, h, w]
            logits : shape [batch_size, num_classes, h, w]
            targets : shape [batch_size, h, w]
    returns:loss: shape [batch_size] or [] depending on mean_reduce
    """
    assert logits.size() == one_hot.size()
    log_prb = F.log_softmax(logits, dim=1)
    non_pad_mask = targets.ne(ignore_index)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask)
    if mean_reduce:
        return loss.mean()  # average later
    else:
        return loss


def compute_loss(args, model, images, targets, num_classes):
    """
    inputs:  images  : shape [batch_size, C, h, w]
             logits : shape [batch_size, num_classes, h, w]
             targets : shape [batch_size, h, w]
    returns: loss: shape []
             logits: shape [batch_size]
    """
    batch, h, w = targets.size()
    one_hot_mask = torch.zeros(batch, num_classes, h, w).cuda()
    new_target = targets.clone().unsqueeze(1)
    new_target[new_target == 255] = 0

    one_hot_mask.scatter_(1, new_target, 1).long()
    if args.smoothing:
        eps = 0.1
        one_hot = one_hot_mask * (1 - eps) + (1 - one_hot_mask) * eps / (num_classes - 1)
    else:
        one_hot = one_hot_mask  # [batch_size, num_classes, h, w]

    if args.mixup:
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        rand_index = torch.randperm(images.size()[0]).cuda()
        one_hot_a = one_hot
        targets_a = targets

        one_hot_b = one_hot[rand_index]
        target_b = targets[rand_index]
        mixed_images = lam * images + (1 - lam) * images[rand_index]

        logits = model(mixed_images)
        loss = cross_entropy(logits, one_hot_a, targets_a) * lam  \
            + cross_entropy(logits, one_hot_b, target_b) * (1. - lam)
    else:
        logits = model(images)
        loss = cross_entropy(logits, one_hot, targets)
    return loss


def validate_epoch(args, val_loader, model):
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
            pd_q0 = model.classifier(f_q)
            pd_s  = model.classifier(f_s)
            pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)
        # 用layer4 的output来做attention
        fea_idx = args.rmid-2 if args.rmid in [3, 4] else -1
        fs_fea = fs_lst[fea_idx]  # [1, 2048, 60, 60]
        fq_fea = fq_lst[fea_idx]  # [1, 2048, 60, 60]
        pred_q = model.outer_forward(f_q, f_s, fq_fea, fs_fea, s_label, q_label, pd_q0, pd_s)
        # cross attention: (f_q, f_s, fq_fea, fs_fea, s_label), self att: (f_q, f_q, fq_fea, fq_fea, q_label)
        pred_q = F.interpolate(pred_q, size=q_label.shape[1:], mode='bilinear', align_corners=True)

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