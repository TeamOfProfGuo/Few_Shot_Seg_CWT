# encoding:utf-8

import os
import random
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from collections import defaultdict
from .dataset.dataset import get_val_loader
from .util import AverageMeter, batch_intersectionAndUnionGPU, get_model_dir, get_model_dir_trans
from .util import find_free_port, setup, cleanup, to_one_hot, intersectionAndUnionGPU
from .model.pspnet import get_model, get_classifier
from .model.transformer import MultiHeadAttentionOne
import torch.distributed as dist
from tqdm import tqdm
from .util import load_cfg_from_cfg_file, merge_cfg_from_list, log
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time
from typing import Tuple


def parse_args() -> None:
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--config', type=str, required=True, help='config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def main_worker(args: argparse.Namespace) -> None:

    if args.manual_seed is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        random.seed(args.manual_seed)

    # ====== Model  ======
    model = get_model(args)

    trans_dim = args.bottleneck_dim
    transformer = MultiHeadAttentionOne(args.heads, trans_dim, trans_dim, trans_dim, dropout=0.5)
    
    root_trans = get_model_dir_trans(args)

    if args.resume_weights:
        if os.path.isfile(args.resume_weights):
            print("=> loading weight '{}'".format(args.resume_weights))

            pre_weight = torch.load(args.resume_weights)['state_dict']

            pre_dict = model.state_dict()
            for index, (key1, key2) in enumerate(zip(pre_dict.keys(), pre_weight.keys())):
                if 'classifier' not in key1 and index < len(pre_dict.keys()):
                    if pre_dict[key1].shape == pre_weight[key2].shape:
                        pre_dict[key1] = pre_weight[key2]
                    else:
                        print('Pre-trained {} shape and model {} shape: {}, {}'.
                              format(key2, key1, pre_weight[key2].shape, pre_dict[key1].shape))
                        continue

            model.load_state_dict(pre_dict, strict=True)

            print("=> loaded weight '{}'".format(args.resume_weights))
        else:
            print("=> no weight found at '{}'".format(args.resume_weights))

    if args.ckpt_used is not None:
        filepath = os.path.join(root_trans, f'{args.ckpt_used}.pth')
        assert os.path.isfile(filepath), filepath
        print("=> loading transformer weight '{}'".format(filepath))
        checkpoint = torch.load(filepath)
        transformer.load_state_dict(checkpoint['state_dict'])
        print("=> loaded transformer weight '{}'".format(filepath))
    else:
        print("=> Not loading anything")

    # ====== Data  ======
    episodic_val_loader, _ = get_val_loader(args)

    # ====== Test  ======
    val_Iou, val_loss = validate_transformer(
        args=args, val_loader=episodic_val_loader,
        model=model, transformer=transformer
    )


def validate_transformer(args: argparse.Namespace,
                         val_loader: torch.utils.data.DataLoader,
                         model: DDP,
                         transformer: DDP) -> Tuple[torch.tensor, torch.tensor]:

    print('==> Start testing')

    model.eval()
    transformer.eval()
    nb_episodes = int(args.test_num / args.batch_size_val)

    # ====== Metrics initialization  ======
    H, W = args.image_size, args.image_size
    if args.image_size == 473:
        h, w = 60, 60
    else:
        h, w = model.feature_res  # (53, 53)

    runtimes = torch.zeros(args.n_runs)  # args.n_runs=1
    val_IoUs = np.zeros(args.n_runs)
    val_losses = np.zeros(args.n_runs)

    # ====== Perform the runs  ======
    for run in range(args.n_runs):

        # ====== Initialize the metric dictionaries ======
        loss_meter = AverageMeter()
        iter_num, runtime = 0, 0
        cls_intersection = defaultdict(int)  # Default value is 0
        cls_union = defaultdict(int)
        cls_intersection0 = defaultdict(int)  # Default value is 0
        cls_union0 = defaultdict(int)
        IoU = defaultdict(int)
        IoU0 = defaultdict(int)

        for e in range(nb_episodes):
            t0 = time.time()
            logits_q = torch.zeros(args.batch_size_val, 1, 2, h, w)
            logits_q0 = torch.zeros(args.batch_size_val, 1, 2, h, w)
            gt_q = 255 * torch.ones(args.batch_size_val, 1, args.image_size,args.image_size).long()
            classes = []  # All classes considered in the tasks

            # ====== Process each task separately ======
            # Batch size val is 50 here.

            for i in range(args.batch_size_val):
                try:
                    qry_img, q_label, spprt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
                except:
                    iter_loader = iter(val_loader)
                    qry_img, q_label, spprt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
                iter_num += 1

                if torch.cuda.is_available():
                    spprt_imgs = spprt_imgs.cuda()
                    s_label = s_label.cuda()
                    q_label = q_label.cuda()
                    qry_img = qry_img.cuda()

                # ====== Phase 1: Train a new binary classifier on support samples. ======

                binary_classifier = nn.Conv2d(args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False).cuda()

                optimizer = optim.SGD(binary_classifier.parameters(), lr=args.cls_lr)

                # Dynamic class weights
                s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
                back_pix = np.where(s_label_arr == 0)
                target_pix = np.where(s_label_arr == 1)

                criterion = nn.CrossEntropyLoss(
                    weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
                    ignore_index=255)

                with torch.no_grad():
                    f_s, _ = model.extract_features(spprt_imgs.squeeze(0))  # [n_task, n_shots, c, h, w]

                for index in range(args.adapt_iter):
                    output_support = binary_classifier(f_s)
                    output_support = F.interpolate(output_support, size=s_label.size()[2:],
                                                   mode='bilinear', align_corners=True)
                    s_loss = criterion(output_support, s_label.squeeze(0))
                    optimizer.zero_grad()
                    s_loss.backward()
                    optimizer.step()

                # ====== Phase 2: Update classifier's weights with old weights and query features. ======
                with torch.no_grad():
                    f_q, _ = model.extract_features(qry_img)  # [n_task, c, h, w]
                    pred_q0 = binary_classifier(f_q)

                    f_q = F.normalize(f_q, dim=1)
                    weights_cls = binary_classifier.weight.data  # [2, c, 1, 1]
                    weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(f_q.shape[0], 2, 512)  # [1, 2, c]
                    updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [1, 2, c]

                    # Build a temporary new classifier for prediction
                    Pseudo_cls = nn.Conv2d(args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False).cuda()
                    # Initialize the weights with updated ones
                    Pseudo_cls.weight.data = torch.as_tensor(updated_weights_cls.squeeze(0).unsqueeze(2).unsqueeze(3))

                    pred_q = Pseudo_cls(f_q)   # [1, 2, 60, 60] 没有expand到2个

                logits_q[i] = pred_q.detach()  # [1 batch_size, 2 channel, 60, 60] 其实一个batch只有一个obs, i=0
                logits_q0[i] =pred_q0.detach()
                gt_q[i, 0] = q_label           # [1 batch_size, 1 channel, 473, 473]
                classes.append([class_.item() for class_ in subcls])

            t1 = time.time()
            runtime += t1 - t0

            logits = F.interpolate(logits_q.squeeze(1), size=(H, W),mode='bilinear', align_corners=True).detach()
            logits0 = F.interpolate(logits_q0.squeeze(1), size=(H, W),mode='bilinear', align_corners=True).detach()
            intersection, union, _ = batch_intersectionAndUnionGPU(logits.unsqueeze(1), gt_q, 2)
            intersection, union = intersection.cpu(), union.cpu()
            intersection0, union0, _ = batch_intersectionAndUnionGPU(logits0.unsqueeze(1), gt_q, 2)
            intersection0, union0 = intersection0.cpu(), union0.cpu()

            # ====== Log metrics ======
            criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
            loss = criterion_standard(logits, gt_q.squeeze(1))
            loss_meter.update(loss.item())
            for i, task_classes in enumerate(classes):  # classes list of list/ each sublist corresponds to nb_episodes
                for j, class_ in enumerate(task_classes):
                    cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
                    cls_union[class_] += union[i, 0, j + 1]
                    cls_intersection0[class_] += intersection0[i, 0, j + 1]  # Do not count background
                    cls_union0[class_] += union0[i, 0, j + 1]

            for class_ in cls_union:
                IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)   # cls wise IoU
                IoU0[class_] = cls_intersection0[class_] / (cls_union0[class_] + 1e-10)  # cls wise IoU

            if (iter_num % 200 == 0):
                mIoU = np.mean([IoU[i] for i in IoU])                                  # mIoU across cls
                mIoU0 = np.mean([IoU0[i] for i in IoU0])
                print('Test: [{}/{}] mIoU {:.4f} mIoU0 {:.4f} Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                    iter_num, args.test_num, mIoU, mIoU0, loss_meter=loss_meter))

        runtimes[run] = runtime
        mIoU = np.mean(list(IoU.values()))  # IoU: dict{cls: cls-wise IoU}
        print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        for class_ in cls_union:
            print("Class {} : {:.4f}".format(class_, IoU[class_]))

        val_IoUs[run] = mIoU
        val_losses[run] = loss_meter.avg

    print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs.mean()))
    print('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs.mean(), val_losses.mean()


def episodic_validate(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader,
                      model: DDP,  use_callback: bool,) -> Tuple[torch.tensor, torch.tensor]:

    log('==> Start testing')

    model.eval()
    nb_episodes = int(args.test_num / args.batch_size_val)

    # ========== Metrics initialization  ==========

    H, W = args.image_size, args.image_size
    h, w = (60, 60) # model.feature_res # (53, 53)

    runtimes = torch.zeros(args.n_runs)
    val_IoUs = np.zeros(args.n_runs)
    val_losses = np.zeros(args.n_runs)

    # ========== Perform the runs  ==========
    for run in range(args.n_runs):

        # =============== Initialize the metric dictionaries ===============
        loss_meter = AverageMeter()
        iter_num, runtime = 0, 0
        cls_intersection = defaultdict(int)  # Default value is 0
        cls_union = defaultdict(int)
        IoU = defaultdict(int)

        # =============== episode = group of tasks ===============
        for e in range(nb_episodes):
            logits_q = torch.zeros(args.batch_size_val, 1, 2, h, w)
            gt_q = 255 * torch.ones(args.batch_size_val, 1, args.image_size, args.image_size).long()
            classes = []  # All classes considered in the tasks

            # =========== Generate tasks and extract features for each task ===============
            for i in range(args.batch_size_val):
                try:
                    qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iter_loader.next()
                except:
                    iter_loader = iter(val_loader)
                    qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iter_loader.next()
                iter_num += 1

                if torch.cuda.is_available():
                    q_label = q_label.cuda()
                    spprt_imgs = spprt_imgs.cuda()
                    s_label = s_label.cuda()
                    qry_img = qry_img.cuda()

                with torch.no_grad():
                    f_s, _ = model.extract_features(spprt_imgs.squeeze(0))

                # ====== Phase 1: Train a new binary classifier on support samples. ======
                binary_classifier = get_classifier(args, num_classes=2).cuda()
                optimizer = optim.SGD(binary_classifier.parameters(), lr=args.cls_lr)

                # Dynamic class weights
                s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
                back_pix = np.where(s_label_arr == 0)
                target_pix = np.where(s_label_arr == 1)

                criterion = nn.CrossEntropyLoss(
                    weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]).cuda(),
                    ignore_index=255)

                for index in range(args.adapt_iter):
                    output_support = binary_classifier(f_s)
                    output_support = F.interpolate(output_support, size=s_label.size()[2:], mode='bilinear', align_corners=True)
                    s_loss = criterion(output_support, s_label.squeeze(0))
                    optimizer.zero_grad()
                    s_loss.backward()
                    optimizer.step()

                # ====== Phase 2: run the model on query set. ======
                with torch.no_grad():
                    f_q, _ = model.extract_features(qry_img)  # [n_task, c, h, w]
                    pd_q = binary_classifier(f_q)

                logits_q[i] = pd_q.detach()  # [1 batch_size, 2 channel, 60, 60] 其实一个batch只有一个obs, i=0
                gt_q[i, 0] = q_label  # [1 batch_size, 1 channel, 473, 473]
                classes.append([class_.item() for class_ in subcls])

            # ================== metrics ==================
            logits = F.interpolate(logits_q.squeeze(1), size=(H, W), mode='bilinear', align_corners=True).detach()
            intersection, union, _ = batch_intersectionAndUnionGPU(logits.unsqueeze(1), gt_q, 2)
            intersection, union = intersection.cpu(), union.cpu()

            criterion_standard = nn.CrossEntropyLoss(ignore_index=255)
            loss = criterion_standard(logits, gt_q.squeeze(1))
            loss_meter.update(loss.item())
            for i, task_classes in enumerate(classes):
                for j, class_ in enumerate(task_classes):
                    cls_intersection[class_] += intersection[i, 0, j + 1]  # Do not count background
                    cls_union[class_] += union[i, 0, j + 1]

            for class_ in cls_union:
                IoU[class_] = cls_intersection[class_] / (cls_union[class_] + 1e-10)

            if (iter_num % 200 == 0):
                mIoU = np.mean([IoU[i] for i in IoU])  # mIoU across cls
                log('Test: [{}/{}] mIoU {:.4f} Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                    iter_num, args.test_num, mIoU, loss_meter=loss_meter))

        # ================== summarize each run ==================
        mIoU = np.mean(list(IoU.values()))
        log('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        for class_ in cls_union:
            log("Class {} : {:.4f}".format(class_, IoU[class_]))

        val_IoUs[run] = mIoU
        val_losses[run] = loss_meter.avg

    log('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs.mean()))
    log('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs.mean(), val_losses.mean()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)

    if args.debug:
        args.test_num = 500
        args.n_runs = 2

    main_worker(args)