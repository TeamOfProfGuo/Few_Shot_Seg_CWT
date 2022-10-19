# encoding:utf-8

# This code is to pretrain model backbone on the base train data

import os
import yaml
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from tensorboardX import SummaryWriter
from typing import Dict
from torch import Tensor
from src.model import get_model

from .test import episodic_validate
from .loss import ContrastCELoss
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, AverageMeter
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

    sv_path = 'pretrain_{}/{}{}/split{}_shot{}/{}'.format(
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
    log(f"==> Using {'episodic' if args.episodic_val else 'standard'} validation\n")

    # ========= Data  ==========
    train_loader, train_sampler = get_train_loader(args, episodic=False)
    val_loader, _ = get_val_loader(args, episodic=args.episodic_val)  # mode='train' means that we will validate on images from validation set, but with the bases classes

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
    criterion = ContrastCELoss(args = args)
    for epoch in range(args.epochs):

        loss_meter = AverageMeter()
        iterable_train_loader = iter(train_loader)

        for i in range(1, iter_per_epoch+1):
            model.train()

            images, gt = iterable_train_loader.next()  # q: [1, 3, 473, 473], s: [1, 1, 3, 473, 473]
            if torch.cuda.is_available():
                images = images.cuda()  # [1, 1, 3, h, w]
                gt = gt.cuda()  # [1, 1, h, w]

            logits, embedding = model(images)
            loss = criterion(logits=logits, target=gt.long(), embedding=embedding, with_embed=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.scheduler == 'cosine':
                scheduler.step()

            if i % args.log_freq == 0:
                model.eval()
                logits = model(images)
                intersection, union, target = intersectionAndUnionGPU(logits.argmax(1), gt, args.num_classes_tr, 255)  # gt [B, 473, 473]

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

        # pdb.set_trace()
        if epoch%2==0 or epoch>=50:
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


def standard_validate(args, val_loader, model, use_callback):

    loss_meter = AverageMeter()
    intersections = torch.zeros(args.num_classes_tr).cuda()
    unions = torch.zeros(args.num_classes_tr).cuda()

    model.eval()
    for i, (images, gt) in enumerate(val_loader):

        if torch.cuda.is_available():
            images = images.cuda()  # [1, 1, 3, h, w]
            gt = gt.cuda()          # [1, 1, h, w]

        with torch.no_grad():
            logits = model(images).detach()
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255)
            loss = loss_fn(logits, gt)
            loss_meter.update(loss.item())

        intersection, union, target = intersectionAndUnionGPU(logits.argmax(1), gt, args.num_classes_tr, 255)  # gt [B, 473, 473]
        intersections += intersection  # [16]
        unions += union                # [16]

    mIoU = (intersections / (unions + 1e-10)).mean()
    acc = intersections.sum() / unions.sum()
    log(f'Testing results: running loss {loss_meter.avg:.2f}, Acc {acc:.4f}, mIoU {mIoU:.4f}\n')

    return mIoU, loss_meter.avg


if __name__ == "__main__":
    args = parse_args()

    world_size = len(args.gpus)
    args.distributed = (world_size > 1)
    main(args)