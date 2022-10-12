import random
from torchvision import transforms
import matplotlib.pyplot as plt
from src.model import *
from src.model.nets.pspnet import get_model
from src.optimizer import get_optimizer
from src.dataset.dataset import get_val_loader, get_train_loader
from src.util import intersectionAndUnionGPU
import argparse
from src.util import load_cfg_from_cfg_file, merge_cfg_from_list

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from mask import Masker
masker = Masker()
# =================== get config ===================
"""  
DATA= pascal 
SPLIT= 0
GPU = [0]
LAYERS= 50 
SHOT= 1                         
"""

arg_input = ' --config config_files/pascal_mmn.yaml   \
  --opts  layers 50   trans_lr 0.001   cls_lr 0.1    batch_size 1   batch_size_val 1   epochs 20   test_num 1000 '

parser = argparse.ArgumentParser(description='Training classifier weight transformer')
parser.add_argument('--config', type=str, required=True, help='config_files/pascal_mmn.yaml')
parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args(arg_input.split())

assert args.config is not None
cfg = load_cfg_from_cfg_file(args.config)
if args.opts is not None:
    cfg = merge_cfg_from_list(cfg, args.opts)
args = cfg
print(args)

args.shot = 1
args.train_split = 3
args.cls_type = 'ooooo'

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
        model_dict = model.state_dict()
        pre_cls_wt = pre_weight['classifier.weight']  # [16, 512, 1, 1]

        for index, key in enumerate(model_dict.keys()):
            if 'classifier' not in key and 'gamma' not in key:
                if model_dict[key].shape == pre_weight[key].shape:
                    model_dict[key] = pre_weight[key]
                else:
                    print('Pre-trained shape and model shape dismatch for {}'.format(key))

        model.load_state_dict(model_dict, strict=True)
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
args.distributed = False

train_loader, train_sampler = get_train_loader(args)   # split 0: len 4760, cls[6,~20]在pascal train中对应4760个图片， 先随机选图片，再根据图片选cls
episodic_val_loader, _ = get_val_loader(args)          # split 0: len 364， 对应cls[1,~5],在pascal val中对应364个图片

# ====== Transformer ======

Trans = MMN(args, agg=args.agg, wa=args.wa, red_dim=args.red_dim)
optimizer_meta = get_optimizer(args, [dict(params=Trans.parameters(), lr=args.trans_lr * args.scale_lr)])
# scheduler = get_scheduler(args, optimizer_meta, len(train_loader))

fname = './results/msc_pascal/resnet50/split0_shot1/dotr_wa/best1.pth'
pre_weight = torch.load(fname, map_location=lambda storage, location: storage)['state_dict']
Trans.load_state_dict(pre_weight, strict=True)


# ====== Metrics initialization ======
max_val_mIoU = 0.
# ================================================ Training ================================================
iterable_train_loader = iter(train_loader,)

# ====== iteration starts
qry_img, q_label, spt_imgs, s_label, subcls, sl, ql = iterable_train_loader.next()
spt_imgs = spt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
s_label = s_label.squeeze(0).long() # [n_shots, img_size, img_size]


# ====== 可视化图片 ======
invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                               ])
inv_s = invTrans(spt_imgs[0])
plt.imshow(inv_s.permute(1, 2, 0))

inv_q = invTrans(qry_img[0])
plt.imshow(inv_q.permute(1, 2, 0))

# ====== Phase 1: Train the binary classifier on support samples ======
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ================== inner loop
model.eval()
with torch.no_grad():
    f_s, fs_lst = model.extract_features(spt_imgs)
model.inner_loop(f_s, s_label)

# ====== Phase 2: Train the transformer to update the classifier's weights ======
with torch.no_grad():
    f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
    pred_q0 = model.classifier(f_q)
    pred_q0 = F.interpolate(pred_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

mask = masker.mask_to_rgb(pred_q0.argmax(1).squeeze())
plt.imshow(mask)
plt.imshow(F.softmax(pred_q0, dim=1)[0,1])
plt.imshow(pred_q0.argmax(1).squeeze())

intersection0, union0, target0 = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, args.num_classes_tr, 255)
print('pred 0', np.round( (intersection0 / (union0 + 1e-10)).numpy() , 3) )


attn, _, att_out = Trans(fq_lst, fs_lst, f_q, f_s, ret_attn=True)
pd_q1 = model.classifier(att_out)
pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

confidence = (pred_q1.argmax(1) == q_label)|(q_label==255)
plt.imshow(confidence.squeeze().float())

va, idx = torch.max(attn, dim=-1)
va = torch.mean(attn, dim=-1)
plt.imshow(va.data.view(60, 60).numpy())


plt.imshow(F.softmax(pred_q1, dim=1).data[0,1])
plt.imshow(pred_q1.argmax(1).squeeze())


intersection1, union1, target1 = intersectionAndUnionGPU(pred_q1.argmax(1), q_label, args.num_classes_tr, 255)
print('pred 1', np.round( (intersection1 / (union1 + 1e-10)).numpy() , 3) )


# ====================================================== 对比实验 ======================================================
new_classifier = nn.Conv2d(model.bottleneck_dim, 2, kernel_size=1, bias=True)

# optimizer and loss function
optimizer = torch.optim.SGD(new_classifier.parameters(), lr=args.cls_lr)

criterion = SegLoss(loss_type=args.inner_loss_type, num_cls=2, fg_idx=1)


s_label_copy = s_label.clone()
s_label_copy[s_label_copy==7] = 1
s_label_copy[ s_label_copy==13] =0

# inner loop 学习 classifier的params
for index in range(args.adapt_iter):
    pred_s_label = new_classifier(f_s)  # [n_shot, 2(cls), 60, 60]
    pred_s_label = F.interpolate(pred_s_label, size=s_label_copy.size()[1:],mode='bilinear', align_corners=True)
    s_loss = criterion(pred_s_label, s_label_copy)  # pred_label: [n_shot, 2, 473, 473], label [n_shot, 473, 473]
    optimizer.zero_grad()
    s_loss.backward()
    optimizer.step()

# ====== Phase 2: Train the transformer to update the classifier's weights ======
model.eval()
with torch.no_grad():

    pd_q0 = new_classifier(f_q)
    pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)
    pred_q_mask0 = pred_q0.argmax(dim=1)

out_mask = masker.mask_to_rgb(pred_q_mask0.squeeze())
plt.imshow(out_mask)




# ================================== Dynamic Fusion  ==================================

Trans.train()
fq, att_fq = Trans(fq_lst, fs_lst, f_q, f_s)
pd_q1 = model.classifier(att_fq)
pred_q1 = F.interpolate(pd_q1, size=q_label.shape[-2:], mode='bilinear', align_corners=True)

criterion = SegLoss(loss_type=args.loss_type)
q_loss1 = criterion(pred_q1, q_label.long())


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

# ===================================== evaluation  ===============================================================


iter_loader = iter(episodic_val_loader,)

qry_img, q_label, spt_imgs, s_label, subcls, spprt_oris, qry_oris = iter_loader.next()
spt_imgs = spt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
s_label = s_label.squeeze(0).long()  # [n_shots, img_size, img_size]
s_label_copy = s_label.clone()


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
inv_q[:, (51-1)*8, (49-1)*8] = torch.tensor([1.0, 0, 0])
plt.imshow(inv_q.permute(1, 2, 0))


# =======
reset_cls_wt(model.val_classifier, pre_cls_wt, args.num_classes_tr, idx_cls=args.num_classes_tr)
model.eval()
with torch.no_grad():
    f_s, fs_lst = model.extract_features(spt_imgs)
    out = model.val_classifier(f_s)  # [1, 17, 60, 60]
    out = F.interpolate(out, size=(473, 473), mode='bilinear', align_corners=True)

s_label = reset_spt_label(s_label, pred=out, idx_cls=args.num_classes_tr)
mask = masker.mask_to_rgb(s_label.squeeze())
plt.imshow(mask)
model.increment_inner_loop(f_s, s_label, cls_idx=args.num_classes_tr, meta_train=False)

with torch.no_grad():
    f_q, fq_lst = model.extract_features(qry_img)  # [n_task, c, h, w]
    pred_q0 = model.val_classifier(f_q)
    pred_q0 = F.interpolate(pred_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)

mask = masker.mask_to_rgb(pred_q0.argmax(1).squeeze())
plt.imshow(mask)


pred_q0 = compress_pred(pred_q0, idx_cls=args.num_classes_tr, input_type='lg')  #

intersection0, union0, target0 = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, 2, 255)
print('pred 0', np.round( (intersection0 / (union0 + 1e-10)).numpy() , 3) )


# ====================================================== 对比实验 ======================================================
new_classifier = nn.Conv2d(model.bottleneck_dim, 2, kernel_size=1, bias=True)

# optimizer and loss function
optimizer = torch.optim.SGD(new_classifier.parameters(), lr=self.args.cls_lr)

criterion = SegLoss(loss_type=args.inner_loss_type, num_cls=2, fg_idx=1)


# inner loop 学习 classifier的params
for index in range(args.adapt_iter):
    pred_s_label = new_classifier(f_s)  # [n_shot, 2(cls), 60, 60]
    pred_s_label = F.interpolate(pred_s_label, size=s_label_copy.size()[1:],mode='bilinear', align_corners=True)
    s_loss = criterion(pred_s_label, s_label_copy)  # pred_label: [n_shot, 2, 473, 473], label [n_shot, 473, 473]
    optimizer.zero_grad()
    s_loss.backward()
    optimizer.step()

# ====== Phase 2: Train the transformer to update the classifier's weights ======
model.eval()
with torch.no_grad():

    pd_q0 = new_classifier(f_q)
    pred_q0 = F.interpolate(pd_q0, size=q_label.shape[1:], mode='bilinear', align_corners=True)
    pred_q_mask0 = pred_q0.argmax(dim=1)

out_mask = masker.mask_to_rgb(pred_q_mask0.squeeze())
plt.imshow(out_mask)

intersection0, union0, target0 = intersectionAndUnionGPU(pred_q0.argmax(1), q_label, 2, 255)
print('pred 0', np.round( (intersection0 / (union0 + 1e-10)).numpy() , 3) )


out_mask = masker.mask_to_rgb(q_label.squeeze())
plt.imshow(out_mask)