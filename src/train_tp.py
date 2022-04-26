import os
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
from tqdm import tqdm
from src.test import validate_transformer
from typing import Dict
import torch.distributed as dist
import argparse
from typing import Tuple
from src.util import load_cfg_from_cfg_file, merge_cfg_from_list

# =================== get config
"""  
DATA= pascal 
SPLIT= 0
GPU = [0]
LAYERS= 50 
SHOT= 1                         
"""

arg_input = ' --config config_files/pascal.yaml   \
  --opts train_split 0   layers 50   gpus [0]   shot 1   trans_lr 0.001   heads 4   cls_lr 0.1    batch_size 1  \
  batch_size_val 1   epochs 20     test_num 1000 '

parser = argparse.ArgumentParser(description='Training classifier weight transformer')
parser.add_argument('--config', type=str, required=True, help='config_files/pascal.yaml')
parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args(arg_input.split())

assert args.config is not None
cfg = load_cfg_from_cfg_file(args.config)
if args.opts is not None:
    cfg = merge_cfg_from_list(cfg, args.opts)
args = cfg

world_size = len(args.gpus)
distributed = world_size > 1
args.distributed = distributed
args.port = find_free_port()
# mp.spawn(main_worker, args=(world_size, args), nprocs=world_size, join=True)

# ====================================  main ================================================

# for distributed learning
# rank=1
# print(f"==> Running process rank {rank}.")
# setup(args, rank, world_size)

print(args)
rank =1
if args.manual_seed is not None:
    random.seed(args.manual_seed + rank)
    np.random.seed(args.manual_seed + rank)
    torch.manual_seed(args.manual_seed + rank)
    if torch.cuda.is_available():
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(args.manual_seed + rank)
        torch.cuda.manual_seed_all(args.manual_seed + rank)

# ====== Model + Optimizer ======
model = get_model(args)


args.resume_weights = './pretrained_models/pascal/split=0/pspnet_resnet50/best.pth'
if args.resume_weights:
    if os.path.isfile(args.resume_weights):
        print("=> loading weight '{}'".format(args.resume_weights))

        pre_weight = torch.load(args.resume_weights, map_location=lambda storage, location: storage)['state_dict']

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

#model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
#model = DDP(model, device_ids=[rank])

# ====== Transformer ======
trans_dim = args.bottleneck_dim

transformer = MultiHeadAttentionOne(args.heads, trans_dim, trans_dim, trans_dim, dropout=0.5)

optimizer_transformer = get_optimizer(args,
                                      [dict(params=transformer.parameters(), lr=args.trans_lr * args.scale_lr)])

trans_save_dir = get_model_dir_trans(args)

# ====== Data  ======
args.workers = 0
train_loader, train_sampler = get_train_loader(args)
episodic_val_loader, _ = get_val_loader(args)


# ====== Metrics initialization ======
max_val_mIoU = 0.
if args.debug:
    iter_per_epoch = 5
else:
    iter_per_epoch = args.iter_per_epoch if args.iter_per_epoch <= len(train_loader) else len(train_loader)

log_iter = iter_per_epoch

# ================================================ Training ================================================

# ================ train epoch
epoch = 1

args=args
train_loader=train_loader
iter_per_epoch=iter_per_epoch
model=model
transformer=transformer
optimizer_trans=optimizer_transformer
epoch=epoch
log_iter=log_iter

loss_meter = AverageMeter()
train_losses = torch.zeros(log_iter)
train_Ious = torch.zeros(log_iter)



model.train()
transformer.train()


# os.environ["OMP_NUM_THREADS"] = "1"
iterable_train_loader = iter(train_loader)
qry_img, q_label, spprt_imgs, s_label, subcls, _, _ = iterable_train_loader.next()

spprt_imgs = spprt_imgs   # [1, 1, 3, h, w]
s_label = s_label         # [1, 1, h, w]
q_label = q_label         # [1, h, w]
qry_img = qry_img         # [1, 3, 473, 473]

# ====== Phase 1: Train the binary classifier on support samples ======

# Keep the batch size as 1.
if spprt_imgs.shape[1] == 1:
    spprt_imgs_reshape = spprt_imgs.squeeze(0).expand(2, 3, args.image_size, args.image_size)    # [2, 3, h, w]
    s_label_reshape = s_label.squeeze(0).expand(2, args.image_size, args.image_size).long()      # [2, 473, 473]
else:
    spprt_imgs_reshape = spprt_imgs.squeeze(0)  # [n_shots, 3, img_size, img_size]
    s_label_reshape = s_label.squeeze(0).long() # [n_shots, img_size, img_size]

binary_cls = nn.Conv2d(args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False)
if torch.cuda.is_available():
    binary_cls = binary_cls.cuda()

optimizer = optim.SGD(binary_cls.parameters(), lr=args.cls_lr)

# Dynamic class weights
s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
back_pix = np.where(s_label_arr == 0)
target_pix = np.where(s_label_arr == 1)
weight = torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])])
if torch.cuda.is_available():
    weight = weight.cuda()

criterion = nn.CrossEntropyLoss(
    weight=weight,
    ignore_index=255
)

with torch.no_grad():
    f_s = model.extract_features(spprt_imgs_reshape)  # [n_task, c, h, w]   [2, 512, 60, 60]
    # f_s = model.module.extract_features(spprt_imgs_reshape)  # [n_task, c, h, w]

args.adapt_iter = 2
for index in range(args.adapt_iter):
    output_support = binary_cls(f_s)  # [2, 2, 60, 60]
    output_support = F.interpolate(
        output_support, size=s_label.size()[2:],
        mode='bilinear', align_corners=True
    )
    s_loss = criterion(output_support, s_label_reshape)
    optimizer.zero_grad()
    s_loss.backward()
    optimizer.step()

# ====== Phase 2: Train the transformer to update the classifier's weights ======
# Inputs of the transformer: weights of classifier trained on support sets, features of the query sample.

# Dynamic class weights used for query image only during training
q_label_arr = q_label.cpu().numpy().copy()  # [n_task, img_size, img_size]
q_back_pix = np.where(q_label_arr == 0)
q_target_pix = np.where(q_label_arr == 1)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, len(q_back_pix[0]) / (len(q_target_pix[0]) + 1e-12)]),                                    # cuda
    ignore_index=255
)

model.eval()
with torch.no_grad():
    f_q = model.extract_features(qry_img)  # [n_task, c, h, w]
    f_q = F.normalize(f_q, dim=1)

# Weights of the classifier.
weights_cls = binary_cls.weight.data

weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(
    args.batch_size, 2, weights_cls.shape[1]
)  # [n_task, 2, c]

# Update the classifier's weights with transformer
updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [n_task, 2, c]

f_q_reshape = f_q.view(args.batch_size, args.bottleneck_dim, -1)  # [n_task, c, hw]

pred_q = torch.matmul(updated_weights_cls, f_q_reshape).view(
    args.batch_size, 2, f_q.shape[-2], f_q.shape[-1]
)  # # [n_task, 2, h, w]

pred_q = F.interpolate(
    pred_q, size=q_label.shape[1:],
    mode='bilinear', align_corners=True
)

loss_q = criterion(pred_q, q_label.long())

optimizer_trans.zero_grad()
loss_q.backward()
optimizer_trans.step()

# Print loss and mIoU
intersection, union, target = intersectionAndUnionGPU(pred_q.argmax(1), q_label, args.num_classes_tr, 255)

mIoU = (intersection / (union + 1e-10)).mean()
loss_meter.update(loss_q.item() / args.batch_size)

# ================================================ validation ================================================

# val_Iou, val_loss = validate_transformer(args=args,val_loader=episodic_val_loader,model=model,transformer=transformer)
args=args
val_loader=episodic_val_loader
model=model
transformer=transformer

model.eval()
transformer.eval()
nb_episodes = int(args.test_num / args.batch_size_val)

# ====== Metrics initialization  ======
H, W = args.image_size, args.image_size
if args.image_size == 473:
    h, w = 60, 60
else:
    h, w = model.feature_res  # (53, 53)

runtimes = torch.zeros(args.n_runs)
val_IoUs = np.zeros(args.n_runs)
val_losses = np.zeros(args.n_runs)

# ====== Perform the runs  ======
# for run in range(args.n_runs):

# ====== Initialize the metric dictionaries ======
loss_meter = AverageMeter()
iter_num = 0
cls_intersection = defaultdict(int)  # Default value is 0
cls_union = defaultdict(int)
IoU = defaultdict(int)
runtime = 0

e = 0 # 共跑 np_episodes 次
import time
t0 = time.time()
logits_q = torch.zeros(args.batch_size_val, 1, args.num_classes_tr, h, w)
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
binary_classifier = nn.Conv2d(
    args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
)

optimizer = optim.SGD(binary_classifier.parameters(), lr=args.cls_lr)

# Dynamic class weights
s_label_arr = s_label.cpu().numpy().copy()  # [n_task, n_shots, img_size, img_size]
back_pix = np.where(s_label_arr == 0)
target_pix = np.where(s_label_arr == 1)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor([1.0, len(back_pix[0]) / len(target_pix[0])]),
    ignore_index=255
)

with torch.no_grad():
    f_s = model.extract_features(spprt_imgs.squeeze(0))  # [n_task, n_shots, c, h, w]

for index in range(args.adapt_iter):
    output_support = binary_classifier(f_s)
    output_support = F.interpolate(
        output_support, size=s_label.size()[2:],
        mode='bilinear', align_corners=True
    )
    s_loss = criterion(output_support, s_label.squeeze(0))
    optimizer.zero_grad()
    s_loss.backward()
    optimizer.step()

# ====== Phase 2: Update classifier's weights with old weights and query features. ======
with torch.no_grad():
    f_q = model.extract_features(qry_img)  # [n_task, c, h, w]
    f_q = F.normalize(f_q, dim=1)

    weights_cls = binary_classifier.weight.data  # [2, c, 1, 1]

    weights_cls_reshape = weights_cls.squeeze().unsqueeze(0).expand(
        f_q.shape[0], 2, 512
    )  # [1, 2, c]

    updated_weights_cls = transformer(weights_cls_reshape, f_q, f_q)  # [1, 2, c]

    # Build a temporary new classifier for prediction
    Pseudo_cls = nn.Conv2d(
        args.bottleneck_dim, args.num_classes_tr, kernel_size=1, bias=False
    )

    # Initialize the weights with updated ones
    Pseudo_cls.weight.data = torch.as_tensor(
        updated_weights_cls.squeeze(0).unsqueeze(2).unsqueeze(3)
    )

    pred_q = Pseudo_cls(f_q)

logits_q[i] = pred_q.detach()
gt_q[i, 0] = q_label
classes.append([class_.item() for class_ in subcls])

            t1 = time.time()
            runtime += t1 - t0

            logits = F.interpolate(
                logits_q.squeeze(1), size=(H, W),
                mode='bilinear', align_corners=True
            ).detach()

            intersection, union, _ = batch_intersectionAndUnionGPU(
                logits.unsqueeze(1), gt_q, 2
            )

            intersection, union = intersection.cpu(), union.cpu()

            # ====== Log metrics ======
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
                mIoU = np.mean([IoU[i] for i in IoU])
                print('Test: [{}/{}] '
                      'mIoU {:.4f} '
                      'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '.format(
                    iter_num, args.test_num, mIoU, loss_meter=loss_meter
                ))

        runtimes[run] = runtime
        mIoU = np.mean(list(IoU.values()))
        print('mIoU---Val result: mIoU {:.4f}.'.format(mIoU))
        for class_ in cls_union:
            print("Class {} : {:.4f}".format(class_, IoU[class_]))

        val_IoUs[run] = mIoU
        val_losses[run] = loss_meter.avg

    print('Average mIoU over {} runs --- {:.4f}.'.format(args.n_runs, val_IoUs.mean()))
    print('Average runtime / run --- {:.4f}.'.format(runtimes.mean()))

    return val_IoUs.mean(), val_losses.mean()










#======


    if args.distributed:
        dist.all_reduce(val_Iou), dist.all_reduce(val_loss)
        val_Iou /= world_size
        val_loss /= world_size

    if main_process(args):
        # Model selection
        if val_Iou.item() > max_val_mIoU:
            max_val_mIoU = val_Iou.item()

            os.makedirs(trans_save_dir, exist_ok=True)
            filename_transformer = os.path.join(trans_save_dir, f'best.pth')

            if args.save_models:
                print('Saving checkpoint to: ' + filename_transformer)

                torch.save(
                    {
                        'epoch': epoch,
                        'state_dict': transformer.state_dict(),
                        'optimizer': optimizer_transformer.state_dict()
                    },
                    filename_transformer
                )

        print("=> Max_mIoU = {:.3f}".format(max_val_mIoU))

if args.save_models and main_process(args):
    filename_transformer = os.path.join(trans_save_dir, 'final.pth')
    torch.save(
        {
            'epoch': args.epochs,
            'state_dict': transformer.state_dict(),
            'optimizer': optimizer_transformer.state_dict()
         },
        filename_transformer
    )

cleanup()




# ====== From classes in query image, chose one randomly ======
class_chosen = np.random.choice(label_class)
new_label = np.zeros_like(label)
ignore_pix = np.where(label == 255)
target_pix = np.where(label == class_chosen)
new_label[ignore_pix] = 255
new_label[target_pix] = 1
label = new_label

file_class_chosen = self.sub_class_file_list[class_chosen]
num_file = len(file_class_chosen)

# ====== Build support ======
# First, randomly choose indexes of support images
support_image_path_list = []
support_label_path_list = []
support_idx_list = []

if self.random_shot:
    shot = random.randint(1, self.shot)
else:
    shot = self.shot

for k in range(shot):
    support_idx = random.randint(1, num_file) - 1
    support_image_path = image_path
    support_label_path = label_path
    while((support_image_path == image_path and support_label_path == label_path)
          or support_idx in support_idx_list):
        support_idx = random.randint(1, num_file) - 1
        support_image_path, support_label_path = file_class_chosen[support_idx]
    support_idx_list.append(support_idx)
    support_image_path_list.append(support_image_path)
    support_label_path_list.append(support_label_path)

support_image_list = []
support_label_list = []
subcls_list = [self.class_list.index(class_chosen) + 1]

# Second, read support images and masks
for k in range(shot):
    support_image_path = support_image_path_list[k]
    support_label_path = support_label_path_list[k]
    support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
    support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
    support_image = np.float32(support_image)
    support_label = cv2.imread(support_label_path, cv2.IMREAD_GRAYSCALE)
    target_pix = np.where(support_label == class_chosen)
    ignore_pix = np.where(support_label == 255)
    support_label[:, :] = 0
    support_label[target_pix[0], target_pix[1]] = 1
    support_label[ignore_pix[0], ignore_pix[1]] = 255
    if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
        raise (
            RuntimeError("Support Image & label shape mismatch: "
                         + support_image_path + " " + support_label_path + "\n")
        )
    support_image_list.append(support_image)
    support_label_list.append(support_label)
assert len(support_label_list) == shot and len(support_image_list) == shot

# Original support images and labels
support_images = support_image_list.copy()
support_labels = support_label_list.copy()

# Forward images through transforms
if self.transform is not None:
    qry_img, target = self.transform(image, label)
    for k in range(shot):
        support_image_list[k], support_label_list[k] = self.transform(support_image_list[k], support_label_list[k])
        support_image_list[k] = support_image_list[k].unsqueeze(0)
        support_label_list[k] = support_label_list[k].unsqueeze(0)

# Reshape properly
spprt_imgs = torch.cat(support_image_list, 0)
spprt_labels = torch.cat(support_label_list, 0)

return qry_img, target, spprt_imgs, spprt_labels, subcls_list, \
       [support_image_path_list, support_labels], [image_path, label]