
import os
import cv2
import random
from torchvision import transforms
import matplotlib.pyplot as plt
from src.model import *
from src.model.nets.pspnet import get_model
from src.optimizer import get_optimizer, get_scheduler
from src.dataset.dataset import get_val_loader, get_train_loader
from src.util import intersectionAndUnionGPU, AverageMeter
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

arg_input = ' --config config_files/pascal_match.yaml   \
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
args.rmid = 'mid4'
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

# ====== Data  ======
args.workers = 0
args.augmentations = ['hor_flip', 'resize_np']

train_loader, train_sampler = get_train_loader(args)   # split 0: len 4760, cls[6,~20]在pascal train中对应4760个图片， 先随机选图片，再根据图片选cls
episodic_val_loader, _ = get_val_loader(args)          # split 0: len 364， 对应cls[1,~5],在pascal val中对应364个图片

# ====== Transformer ======
if args.crm_type == 'chm':
    FusionNet = CHMLearner(ktype='psi', feat_dim=2048, temp=args.temp)
    print('model setting kernel type {}, input feature dim {}'.format('psi', 2048))
else:
    FusionNet = MatchNet(temp=args.temp, cv_type='red', sym_mode=True)
optimizer_meta = get_optimizer(args, [dict(params=FusionNet.parameters(), lr=args.trans_lr * args.scale_lr)])
scheduler = get_scheduler(args, optimizer_meta, len(train_loader))

fname = './results/fuse_pascal/resnet50/split0_shot1/match_l4_nig/best1.pth'
pre_weight = torch.load(fname, map_location=lambda storage, location: storage)['state_dict']
FusionNet.load_state_dict(pre_weight, strict=True)

# ====== Metrics initialization ======
max_val_mIoU = 0.
# ================================================ Training ================================================
epoch = 1
train_loss_meter = AverageMeter()
train_iou_meter = AverageMeter()
iterable_train_loader = iter(episodic_val_loader)

# ====== iteration starts
qry_img, q_label, spt_imgs, s_label, subcls, sl, ql = iterable_train_loader.next()
spt_imgs = spt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
s_label = s_label.squeeze(0).long() # [n_shots, img_size, img_size]

# ====== 可视化图片 ======
invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                               ])
inv_s = invTrans(spt_imgs[0])
for i in range(1, 473+1, 8):
    for j in range(1, 473+1, 8):
        inv_s[:, i-1, j-1] = torch.tensor([0, 1.0, 0])
plt.imshow(inv_s.permute(1, 2, 0))

inv_q = invTrans(qry_img[0])
for i in range(1, 473+1, 8):
    for j in range(1, 473+1, 8):
        inv_q[:, i-1, j-1] = torch.tensor([0, 1.0, 0])
inv_q[:, (34-1)*8, (37-1)*8] = torch.tensor([1.0, 0, 0])
plt.imshow(inv_q.permute(1, 2, 0))

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

# ================================== Dynamic Fusion  ==================================
if args.rmid == 'nr':
    idx = -1
elif args.rmid in ['mid2', 'mid3', 'mid4']:
    idx = int(args.rmid[-1]) - 2
fs_fea = fs_lst[idx]  # [1, 2048, 60, 60]
fq_fea = fq_lst[idx]  # [1, 2048, 60, 60]
with torch.no_grad():
    pd_q_b, ret_corr, ig_mask = model.outer_forward(f_q, f_s, fq_fea, fs_fea, s_label, q_label, pd_q0, pd_s, ret_corr='cr_ig')
    corr, weighted_v_b = ret_corr
    pd_q1_b = model.classifier(weighted_v_b)
    pred_q1_b = F.interpolate(pd_q1_b, size=q_label.shape[-2:], mode='bilinear', align_corners=True)     # weighted_v based on original corr matrix
    pred_q_b = F.interpolate(pd_q_b, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

if not args.ignore:
    ig_mask = None
if args.crm_type == 'chm':
    fs_fea = F.interpolate(fs_fea, scale_factor=0.5, mode='bilinear', align_corners=True)
    fq_fea = F.interpolate(fq_fea, scale_factor=0.5, mode='bilinear', align_corners=True)
    weighted_v = FusionNet(fq_fea, fs_fea, v=f_s.view(f_s.shape[:2] + (-1,)), ig_mask=ig_mask, ret_corr=False)
else:
    weighted_v = FusionNet(corr=corr, v=f_s.view(f_s.shape[:2] + (-1,)), ig_mask=ig_mask)

pd_q1 = model.classifier(weighted_v)
pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

for wt in [0.2, 0.3, 0.4]:
    with torch.no_grad():
        out = weighted_v * wt + f_q * (1-wt)
        pd_q = model.classifier(out)
        pred_q = F.interpolate(pd_q, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

    # ==== 比较结果 base vs att ====
    print('====', wt)
    intersection0, union0, target0 = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, args.num_classes_tr, 255)
    print('pred 0', np.round( (intersection0 / (union0 + 1e-10)).numpy() , 3) )

    intersection1, union1, target1 = intersectionAndUnionGPU(pred_q1.argmax(1), q_label, args.num_classes_tr, 255)
    print('pred 1', np.round( (intersection1 / (union1 + 1e-10)).numpy(), 3))

    intersection1b, union1b, target1b = intersectionAndUnionGPU(pred_q1_b.argmax(1), q_label, args.num_classes_tr, 255)
    print('pred 1b', np.round( (intersection1b / (union1b + 1e-10)).numpy(), 3))

    intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, args.num_classes_tr, 255)
    print('pred', np.round( (intersection / (union + 1e-10)).numpy(), 3))

    intersection, union, target = intersectionAndUnionGPU(pred_q_b.argmax(1), q_label, args.num_classes_tr, 255)
    print('pred b', np.round( (intersection / (union + 1e-10)).numpy(), 3))

# ========= Loss function: Dynamic class weights used for query image only during training   =========
q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
q_back_pix = np.where(q_label_arr == 0)
q_target_pix = np.where(q_label_arr == 1)
loss_weight = torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)])
criterion = nn.CrossEntropyLoss(weight=loss_weight, ignore_index=255)
q_loss1 = criterion(pred_q1, q_label.long())
q_loss0 = criterion(pred_q0, q_label.long())

optimizer_meta.zero_grad()
q_loss1.backward()
optimizer_meta.step()
if args.scheduler == 'cosine':
    scheduler.step()

torch.max(FusionNet.NeighConsensus.conv[0].conv1.weight.grad)

# =================== mask query  =====================
sim = corr


q_mask = F.interpolate(q_label.unsqueeze(1).float(), size=f_q.shape[-2:], mode='nearest').squeeze(1)  # [1,1,h,w]
q_mask = (q_mask != 255.0)

q_mask = q_mask * (pd_q_mask0==1)             # query 根据 pred_mask 选取前景或背景
q_mask = q_mask.view(q_mask.shape[0], -1, 1)  # [n_shot, 1, hw]
q_mask = q_mask.expand(sim.shape)             # [1, q_hw, hw]

sim0 = sim[q_mask].reshape(1, -1, 3600)    # query image 前景或背景 overall sim score 分布
sim0 = torch.mean(sim0, dim=1)

sim0 = sim[:,(34-1)*60+37-1]               # 只看query image 单点的 sim score的分布

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

mask = np.uint8(200 * pred_q1_b.argmax(1).squeeze().numpy())
heatmap = cv2.applyColorMap(cv2.resize(mask, (473, 473)), cv2.COLORMAP_JET)

soft_mask = np.uint8(200* F.softmax(pred_q1, dim=1)[0,1].numpy())
heatmap = cv2.applyColorMap(cv2.resize(soft_mask, (473, 473)), cv2.COLORMAP_JET)

q_label_mask = torch.clone(q_label)
q_label_mask[q_label==255] = 0
mask = np.uint8(200 * q_label_mask.squeeze().numpy())
heatmap = cv2.applyColorMap(cv2.resize(mask, (473, 473)), cv2.COLORMAP_JET)

img = inv_q.permute(1, 2, 0).numpy()*255
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = heatmap * 0.3 + img * 0.7
cv2.imwrite('CAM.jpg', result)
cv2.imwrite('heatmap.jpg', heatmap)