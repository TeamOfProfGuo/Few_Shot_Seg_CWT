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
from src.model.pspnet import get_model
from src.model.transformer import MultiHeadAttentionOne
from src.optimizer import get_optimizer, get_scheduler
from src.dataset.dataset import get_val_loader, get_train_loader
from src.util import intersectionAndUnionGPU, get_model_dir, AverageMeter, get_model_dir_trans
from src.util import setup, cleanup, to_one_hot, batch_intersectionAndUnionGPU, find_free_port
import argparse
from src.util import load_cfg_from_cfg_file, merge_cfg_from_list

# =================== get config
"""  
DATA= pascal 
SPLIT= 0
GPU = [0]
LAYERS= 50 
SHOT= 1                         
"""

arg_input = ' --config config_files/pascal_asy.yaml   \
  --opts train_split 0   layers 50   gpus [0]   shot 1   trans_lr 0.001   heads 4   cls_lr 0.1    batch_size 1  \
  batch_size_val 1   epochs 20     test_num 1000 '

parser = argparse.ArgumentParser(description='Training classifier weight transformer')
parser.add_argument('--config', type=str, required=True, help='config_files/pascal_asy.yaml')
parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args(arg_input.split())

assert args.config is not None
cfg = load_cfg_from_cfg_file(args.config)
if args.opts is not None:
    cfg = merge_cfg_from_list(cfg, args.opts)
args = cfg

# ====================================  main ================================================

print(args)
if args.manual_seed is not None:
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if torch.cuda.is_available():
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

# ====== Model + Optimizer ======
model = get_model(args)

args.resume_weights = './pretrained_models/pascal/split=0/pspnet_resnet50/best.pth'
if args.resume_weights:
    if os.path.isfile(args.resume_weights):
        print("=> loading weight '{}'".format(args.resume_weights))

        pre_weight = torch.load(args.resume_weights, map_location=lambda storage, location: storage)['state_dict']

        pre_dict = model.state_dict()
        for index, key in enumerate(pre_dict.keys()):
            if 'classifier' not in key and 'gamma' not in key:
                if pre_dict[key].shape == pre_weight['module.'+key].shape:
                    pre_dict[key] = pre_weight['module.'+key]
                else:
                    print('Pre-trained shape and model shape for {}: {}, {}'.format(
                        key, pre_weight['module.'+key].shape, pre_dict[key].shape))
                    continue

        model.load_state_dict(pre_dict, strict=True)
        print("=> loaded weight '{}'".format(args.resume_weights))
    else:
        print("=> no weight found at '{}'".format(args.resume_weights))

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

# ====== Transformer ======
    param_list = [model.gamma]
    optimizer_meta = get_optimizer(args, [dict(params=param_list, lr=args.trans_lr * args.scale_lr)])
    trans_save_dir = get_model_dir_trans(args)

# ====== Data  ======
args.workers = 0
train_loader, train_sampler = get_train_loader(args)   # split 0: len 4760, cls[6,~20]在pascal train中对应4760个图片， 先随机选图片，再根据图片选cls
episodic_val_loader, _ = get_val_loader(args)          # split 0: len 364， 对应cls[1,~5],在pascal val中对应364个图片

# ====== Metrics initialization ======
max_val_mIoU = 0.
iter_per_epoch = args.iter_per_epoch if args.iter_per_epoch <= len(train_loader) else len(train_loader)

# ================================================ Training ================================================

epoch = 1
args=args
train_loader=train_loader
iter_per_epoch=iter_per_epoch
model=model
epoch=epoch

train_loss_meter = AverageMeter()
train_iou_meter = AverageMeter()
# loss_meter = AverageMeter()
# train_losses = torch.zeros(iter_per_epoch)
# train_Ious = torch.zeros(iter_per_epoch)

iterable_train_loader = iter(train_loader)
qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iterable_train_loader.next()

# ====== Phase 1: Train the binary classifier on support samples ======

# Keep the batch size as 1.   spt_imgs: [1(eps), 1(shot), 3, 473, 473]
if spprt_imgs.shape[1] == 1:
    spprt_imgs_reshape = spprt_imgs.squeeze(0).expand(2, 3, args.image_size, args.image_size)    # [2, 3, h, w]
    s_label_reshape = s_label.squeeze(0).expand(2, args.image_size, args.image_size).long()      # [2, 473, 473]
else:
    spprt_imgs_reshape = spprt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
    s_label_reshape = s_label.squeeze(0).long() # [n_shots, img_size, img_size]


w1 = model.classifier.weight.data
model.train()
f_s, fs_lst = model.extract_features(spprt_imgs_reshape)
model.inner_loop(f_s, s_label_reshape)
w2 = model.classifier.weight

# ====== Phase 2: Train the transformer to update the classifier's weights ======

model.eval()
with torch.no_grad():
    f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
    pred_q0 = model.classifier(f_q)
    pred_q0 = F.interpolate(pred_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)
# 用layer4 的output来做attention
fs_fea = fs_lst[-1][0:1]   # [1, 2048, 60, 60]
fq_fea = fq_lst[-1]        # [1, 2048, 60, 60]
f_s = f_s[0:1]
pred_q = model.outer_forward(f_q, f_s, fq_fea, fs_fea)

def outer_forward(f_q, f_s, fq_fea, fs_fea):
    bs, C, height, width = f_q.size()
    proj_q = fq_fea.view(bs, -1, height*width).permute(0, 2, 1)   #  [1, 2048, hw] -> [1, hw, 2048]
    proj_k = fs_fea.view(bs, -1, height*width)                    #  [1, 2048, hw]
    proj_v = f_s.view(bs, -1, height*width)
    sim = torch.bmm(proj_q, proj_k)                               #  [1, 3600 (q_hw), 3600(k_hw)]
    attention = F.softmax(sim,dim=-1)
    weighted_v = torch.bmm(proj_v, attention.permute(0, 2, 1))    # [1, 512, hw_k] * [1, hw_k, hw_q] -> [1, 512, hw_q]
    weighted_v = weighted_v.view(bs, C, height, width)

    out = weighted_v * model.gamma + f_q
    out = model.classifier(out)


# Dynamic class weights used for query image only during training
q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
q_back_pix = np.where(q_label_arr == 0)
q_target_pix = np.where(q_label_arr == 1)
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)]),       # cuda   dim 0: 对应背景， dim1 对应前景
    ignore_index=255)
pred_q = F.interpolate(pred_q, size=q_label.shape[1:],mode='bilinear', align_corners=True)
q_loss = criterion(pred_q, q_label.long())
q_loss0 = criterion(pred_q0, q_label.long())

optimizer_meta.zero_grad()
q_loss.backward()
optimizer_meta.step()

# Print loss and mIoU
intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, args.num_classes_tr, 255)
mIoU = (intersection / (union + 1e-10)).mean()
train_loss_meter.update(q_loss.item() / args.batch_size)
train_iou_meter.update(mIoU)

# ================================================ validation ================================================

# val_Iou, val_loss = validate_transformer(args=args,val_loader=episodic_val_loader,model=model,transformer=transformer)
args=args
val_loader=episodic_val_loader
model=model

iter_num = 0
start_time = time.time()
loss_meter = AverageMeter()
cls_intersection = defaultdict(int)  # Default value is 0
cls_union = defaultdict(int)
IoU = defaultdict(float)

# ====== iteration starts  ======
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
# f_s = f_s[0:1]
pred_q = model.outer_forward(f_q, f_s, fq_fea, fs_fea)
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