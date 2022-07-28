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

    sv_path = 'count_{}/{}{}/split{}_shot{}/{}'.format(
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

    # ========= Data  ==========
    episodic_val_loader, _ = get_val_loader(args)

    # ====== Summarize Testing FG / All  ======
    args.test_num=2000

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
            for k, v in fb_ratio.items():
                print('-- k {} : v {}'.format(k, v.avg))


if __name__ == "__main__":
    args = parse_args()

    world_size = len(args.gpus)
    args.distributed = (world_size > 1)
    main(args)