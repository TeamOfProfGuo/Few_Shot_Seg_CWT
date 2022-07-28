# encoding:utf-8
import pdb

import os
import time
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from .model import MMN, SegLoss
from .model.pspnet import get_model
from .optimizer import get_optimizer, get_scheduler
from .dataset.dataset import get_val_loader, get_train_loader
from .util import intersectionAndUnionGPU, AverageMeter, CompareMeter
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

    sv_path = 'aug_{}/{}{}/split{}_shot{}/{}'.format(
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
            model_dict = model.state_dict()

            for index, key in enumerate(model_dict.keys()):
                if 'classifier' not in key and 'gamma' not in key:
                    if model_dict[key].shape == pre_weight[key].shape:
                        model_dict[key] = pre_weight[key]
                    else:
                        log( 'Pre-trained shape and model shape dismatch for {}'.format(key) )

            model.load_state_dict(model_dict, strict=True)
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

    # ======= Transformer ======= args, inner_channel=32, sem=True, wa=False
    Trans = MMN(args, agg=args.agg, wa=args.wa, red_dim=args.red_dim).cuda()
    optimizer_meta = get_optimizer(args, [dict(params=Trans.parameters(), lr=args.trans_lr * args.scale_lr)])
    scheduler = get_scheduler(args, optimizer_meta, len(train_loader))

    # ====== Summarize Testing FG / All  ======


    iter_num = 0
    fg_num = defaultdict(int)  # Default value is 0
    fb_num = defaultdict(int)
    fb_ratio = defaultdict(AverageMeter)

    for e in range(args.test_num):
        iter_num += 1
        try:
            qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        except:
            iter_loader = iter(episodic_val_loader)
            qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
        if torch.cuda.is_available():
            spt_imgs = spt_imgs.cuda()
            s_label = s_label.cuda()
            q_label = q_label.cuda()
            qry_img = qry_img.cuda()


        freq = torch.bincount(q_label.flatten())
        fg_num[subcls[0].item()] += freq[1].item()
        fb_num[subcls[0].item()] += torch.sum(freq).item()
        fb_ratio[subcls[0].item()].update( (freq[1]/torch.sum(freq)).item() )

    if e%10==0:
        fb_ratio0 = {k: round(fg_num[k]/fb_num[k], 3) for k in fg_num}
        print('iter {} {}'.format(e, fb_ratio0))
        for k, v in fb_ratio:
            print('-- k {} : v {}'.format(k, v.avg))


if __name__ == "__main__":
    args = parse_args()

    world_size = len(args.gpus)
    args.distributed = (world_size > 1)
    main(args)