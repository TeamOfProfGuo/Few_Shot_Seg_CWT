import os
import cv2
import time
import random
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from collections import defaultdict
from src.model import conv4d
from src.model.pspnet import get_model
from src.model.transformer import MultiHeadAttentionOne, DynamicFusion
from src.optimizer import get_optimizer, get_scheduler
from src.dataset.dataset import get_val_loader, get_train_loader, EpisodicData
from src.util import intersectionAndUnionGPU, get_model_dir, AverageMeter, get_model_dir_trans
import argparse
from src.util import load_cfg_from_cfg_file, merge_cfg_from_list

# =================== get config ===================
"""  
DATA= pascal 
SPLIT= 0
GPU = [0]
LAYERS= 50 
SHOT= 1                         
"""

arg_input = ' --config config_files/pascal_asy.yaml   \
  --opts train_split 0   layers 50    shot 1   trans_lr 0.001   cls_lr 0.1    batch_size 1  \
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
print(args)

args.dist = 'cosN'
# ====================================  main ================================================
random.seed(args.manual_seed)
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

# ====== Model + Optimizer ======
model = get_model(args)

if args.resume_weights:
    fname = args.resume_weights + args.train_name + '/' + \
            'split={}/pspnet_{}{}/best.pth'.format(args.train_split, args.arch, args.layers)
    if os.path.isfile(fname):
        print("=> loading weight '{}'".format(fname))
        pre_weight = torch.load(fname, map_location=lambda storage, location: storage)['state_dict']
        pre_dict = model.state_dict()

        for index, key in enumerate(pre_dict.keys()):
            if 'classifier' not in key and 'gamma' not in key:
                if pre_dict[key].shape == pre_weight['module.' + key].shape:
                    pre_dict[key] = pre_weight['module.' + key]
                    print('load ' + key)
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

# ====== Transformer ======
param_list = [model.gamma]
optimizer_meta = get_optimizer(args, [dict(params=param_list, lr=args.trans_lr * args.scale_lr)])
trans_save_dir = get_model_dir_trans(args)

# ====== Data  ======
args.workers = 0
args.augmentations = ['hor_flip', 'resize_np']

train_loader, train_sampler = get_train_loader(args)   # split 0: len 4760, cls[6,~20]在pascal train中对应4760个图片， 先随机选图片，再根据图片选cls
episodic_val_loader, _ = get_val_loader(args)          # split 0: len 364， 对应cls[1,~5],在pascal val中对应364个图片

# ====== Metrics initialization ======
max_val_mIoU = 0.
iter_per_epoch = args.iter_per_epoch if args.iter_per_epoch <= len(train_loader) else len(train_loader)

# ================================================ Training ================================================
epoch = 1
train_loss_meter = AverageMeter()
train_iou_meter = AverageMeter()
iterable_train_loader = iter(train_loader)

# ====== iteration starts
qry_img, q_label, spt_imgs, s_label, subcls, sl, ql = iterable_train_loader.next()
spt_imgs = spt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
s_label = s_label.squeeze(0).long() # [n_shots, img_size, img_size]

# ====== Phase 1: Train the binary classifier on support samples ======
model.eval()
with torch.no_grad():
    f_s, fs_lst = model.extract_features(spt_imgs)
model.inner_loop(f_s, s_label)

# ====== Phase 2: Train the transformer to update the classifier's weights ======
model.eval()
with torch.no_grad():
    f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
    pd_q0 = model.classifier(f_q)
    pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)
    pd_q_mask0 = pd_q0.argmax(dim=1)
    pred_q_mask0 = pred_q0.argmax(dim=1)

    pd_s = model.classifier(f_s[0:1])
    pred_s = F.interpolate(pd_s, size=q_label.shape[1:], mode='bilinear', align_corners=True)
    pred_s_mask = pred_s.argmax(dim=1)

# ====== 可视化图片 ======
invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                               ])
inv_s = invTrans(spt_imgs[0])
# for i in range(1, 473+1, 8):
#     for j in range(1, 473+1, 8):
#         inv_s[:, i-1, j-1] = torch.tensor([0, 1.0, 0])
plt.imshow(inv_s.permute(1, 2, 0))

inv_q = invTrans(qry_img[0])
inv_q[:, (30-1)*8, (30-1)*8] = torch.tensor([1.0, 0, 0])
plt.imshow(inv_q.permute(1, 2, 0))

# ================================== 改进attention机制  ==================================
TH = 0.8

# 用layer4 的output来做attention
fs_fea = fs_lst[-1]    # [2, 2048, 60, 60]
fq_fea = fq_lst[-1]    # [1, 2048, 60, 60]

# ==== attention 开始 ====
self = model
bs, C, height, width = f_q.size()

# 基于attention, refine f_q, 并对query img做prediction
proj_q = fq_fea.view(bs, -1, height * width).permute(0, 2, 1)  # [1, 2048, hw] -> [1, hw, 2048]
proj_k = fs_fea.view(bs, -1, height * width)  # [1, 2048, hw]
proj_v = f_s.view(bs, -1, height * width)

# normalize q and k
proj_q = F.normalize(proj_q, dim=-1)   # [B, N_q, 2048}
proj_k = F.normalize(proj_k, dim=-2)   # [B, 2048, N_s]
sim = torch.bmm(proj_q, proj_k)  # [1, 3600 (N_q), 3600(N_s)]


# ============ to calculate fusion weight
FusionNet = DynamicFusion(im_size=30, mid_dim=256)
optimizer_meta = get_optimizer(args, [dict(params=FusionNet.parameters(), lr=args.trans_lr * args.scale_lr)])

corr = sim.reshape(bs, height, width, height, width)  # [B, h, w, h_s, w_s]
s_mask_corr = F.interpolate(s_label.unsqueeze(1).float(), size=f_s.shape[-2:], mode='nearest')  # [1,1,h,w]
s_mask_corr[s_mask_corr==255] = 0
wt = FusionNet(corr, s_mask_corr)

# ============ end

# mask ignored pixels
s_mask = F.interpolate(s_label.unsqueeze(1).float(), size=f_s.shape[-2:], mode='nearest')  # [1,1,h,w]
s_mask = (s_mask > 1).view(s_mask.shape[0], 1, -1)  # [n_shot, 1, hw]
s_mask = s_mask.expand(sim.shape)  # [1, q_hw, hw]
sim[s_mask == True] = 0.00001

# ignore misleading points
q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=f_q.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
qf_mask = (q_mask != 255.0) * (pd_q_mask0==1)   # predicted qry FG
qb_mask = (q_mask != 255.0) * (pd_q_mask0==0)   # predicted qry BG
qf_mask = qf_mask.view(qf_mask.shape[0], -1, 1).expand(sim.shape)   # 保留query predicted FG
qb_mask = qb_mask.view(qb_mask.shape[0], -1, 1).expand(sim.shape)   # 保留query predicted BG

sim_qf = sim[qf_mask].reshape(1, -1, 3600)
th_qf = torch.quantile(sim_qf.flatten(), TH)
sim_qb = sim[qb_mask].reshape(1, -1, 3600)
th_qb = torch.quantile(sim_qb.flatten(), TH)
sim_qf = torch.mean(sim_qf, dim=1)          # 取平均 对应support img 与Q前景相关 所有pixel
sim_qb = torch.mean(sim_qb, dim=1)          # 取平均 对应support img 与Q背景相关 所有pixel
sf_mask = pd_s.argmax(dim=1).view(1, 3600)

ig_mask1 = (sim_qf>th_qf) & (sf_mask==0)   # query的前景attend到support背景
ig_mask2 = (sim_qf>th_qf) & (sim_qb>th_qb)    # query的前景与query的背景
ig_mask3 = (sim_qb>th_qb) & (sf_mask==1)   # query的背景attend到support前景
ig_mask = ig_mask1 | ig_mask2 | ig_mask3 | s_mask[:,1,:]
plt.imshow(ig_mask.view(60,60).long(), cmap='gray')

ig_mask = ig_mask.unsqueeze(1).expand(sim.shape)
sim[ig_mask == True] =0.00001

attention = F.softmax(sim * 20.0, dim=-1)
weighted_v = torch.bmm(proj_v, attention.permute(0, 2, 1))  # [1, 512, hw_k] * [1, hw_k, hw_q] -> [1, 512, hw_q]
weighted_v = weighted_v.view(bs, C, height, width)

out = (weighted_v * self.gamma + f_q) / (1 + self.gamma)
pred_q = self.classifier(out)
pred_q = F.interpolate(pred_q, size=q_label.shape[1:],mode='bilinear', align_corners=True)

# ===== Dynamic Fusion
out = weighted_v * wt + f_q * (1-wt)
pred_q = self.classifier(out)
pred_q = F.interpolate(pred_q, size=q_label.shape[1:],mode='bilinear', align_corners=True)

# ===== loss function
q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
q_back_pix = np.where(q_label_arr == 0)
q_target_pix = np.where(q_label_arr == 1)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)]), ignore_index=255)

loss_q = criterion(pred_q, q_label.long())
optimizer_meta.zero_grad()
loss_q.backward()
optimizer_meta.step()

FusionNet.conv4d.conv1.weight.grad
torch.max(FusionNet.att[0].weight.grad)
torch.max(FusionNet.att[2].weight.grad)



# ==== 比较结果 base vs att ====
intersection0, union0, target0 = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, args.num_classes_tr, 255)
print(intersection0 / (union0 + 1e-10))

intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, args.num_classes_tr, 255)
print(intersection / (union + 1e-10))


# =================== mask query  =====================
q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=f_q.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
q_mask = (q_mask != 255.0)

q_mask = q_mask * (pd_q_mask0==1)             # query 根据 pred_mask 选取前景或背景
q_mask = q_mask.view(q_mask.shape[0], -1, 1)  # [n_shot, 1, hw]
q_mask = q_mask.expand(sim.shape)             # [1, q_hw, hw]

sim0 = attention[:,(23-1)*60+45-1]         # 只看query image 单点的 att score的分布
sim0 = sim[:,(23-1)*60+45-1]               # 只看query image 单点的 sim score的分布

sim0 = sim[q_mask].reshape(1, -1, 3600)    # query image 前景或背景 overall sim score 分布
sim0 = torch.mean(sim0, dim=1)

a = sim0.numpy().reshape(60,60)
print('max {:.4f}, mean {:.4f}, min {:.4f}'.format(np.max(a), np.mean(a), np.min(a)))
a = np.uint8((a - np.min(a)) / (np.max(a) - np.min(a)) * 255)
heatmap = cv2.applyColorMap(cv2.resize(a, (473, 473)), cv2.COLORMAP_JET)
img = inv_s.permute(1, 2, 0).numpy()*255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = heatmap * 0.3 + img * 0.7
cv2.imwrite('CAM.jpg', result)
cv2.imwrite('heatmap.jpg', heatmap)

# support image pred mask (reflect quality of f_s)
mask = np.uint8(200 * pred_s_mask.squeeze().numpy())
heatmap = cv2.applyColorMap(cv2.resize(mask, (473, 473)), cv2.COLORMAP_JET)

s_label_mask = torch.clone(s_label)
mask = np.uint8(200 * s_label_mask.squeeze().numpy())
heatmap = cv2.applyColorMap(cv2.resize(mask, (473, 473)), cv2.COLORMAP_JET)

ig_mask = F.interpolate( (ig_mask1 | ig_mask2 | ig_mask3).view(1, 1, 60, 60).float(), size=(473, 473), mode='nearest')
ig_mask = np.uint8(200 * ig_mask.squeeze().numpy())
heatmap = cv2.applyColorMap(cv2.resize(ig_mask, (473, 473)), cv2.COLORMAP_JET)

# =================== mask support ===================

# query image pred mask (reflect quality of f_s)
mask = np.uint8(200 * pred_q0.argmax(1).squeeze().numpy())
heatmap = cv2.applyColorMap(cv2.resize(mask, (473, 473)), cv2.COLORMAP_JET)

mask = np.uint8(200 * pred_q.argmax(1).squeeze().numpy())
heatmap = cv2.applyColorMap(cv2.resize(mask, (473, 473)), cv2.COLORMAP_JET)

q_label_mask = torch.clone(q_label)
q_label_mask[q_label==255] = 0
mask = np.uint8(200 * q_label_mask.squeeze().numpy())
heatmap = cv2.applyColorMap(cv2.resize(mask, (473, 473)), cv2.COLORMAP_JET)


img = inv_q.permute(1, 2, 0).numpy()*255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = heatmap * 0.3 + img * 0.7
cv2.imwrite('CAM.jpg', result)
cv2.imwrite('heatmap.jpg', heatmap)