import random
from torchvision import transforms
import matplotlib.pyplot as plt
from src.model import *
from src.model.nets import get_model
from src.optimizer import get_optimizer, get_scheduler
from src.dataset.dataset import get_val_loader, get_train_loader
from src.util import intersectionAndUnionGPU
import argparse
from src.util import load_cfg_from_cfg_file, merge_cfg_from_list
from src.test import episodic_validate
from src.loss import ContrastCELoss

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

arg_input = ' --config config_files/pascal_pretrain.yaml   \
  --opts  layers 50  epochs 20   test_num 1000 '

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

args.cls_type = 'ooooo'
args.contrast = False

# ====================================  main ================================================
random.seed(args.manual_seed)
np.random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)

# ====== Model + Optimizer ======
model = get_model(args)
modules_ori = [model.layer0, model.layer1, model.layer2, model.layer3, model.layer4]
modules_new = [model.ppm, model.bottleneck, model.classifier]
if args.contrast:
    modules_new.append(model.proj_head)


params_list = []
for module in modules_ori:
    params_list.append(dict(params=module.parameters(), lr=args.lr))
for module in modules_new:
    params_list.append(dict(params=module.parameters(), lr=args.lr * args.scale_lr))
optimizer = get_optimizer(args, params_list)

validate_fn = episodic_validate if args.episodic_val else standard_validate

# ========= Data  ==========
args.workers = 0
args.batch_size=2
args.distributed = False
train_loader, train_sampler = get_train_loader(args, episodic=False)
val_loader, _ = get_val_loader(args,  episodic=args.episodic_val)  # mode='train' means that we will validate on images from validation set, but with the bases classes

# ========== Scheduler  ================
scheduler = get_scheduler(args, optimizer, len(train_loader))

# ====== Metrics initialization ======
max_val_mIoU = 0.
iter_per_epoch = len(train_loader)
log_iter = int(iter_per_epoch / args.log_freq) + 1

# ================================================ Training ================================================
iterable_train_loader = iter(train_loader,)
model.train()

images, gt = iterable_train_loader.next()  # q: [1, 3, 473, 473], s: [1, 1, 3, 473, 473]
criterion = ContrastCELoss(args=args)
logits, embedding = model(images)


loss = criterion(logits=logits, target=gt.long(), embedding=embedding, with_embed=True)

sv_path = 'pretrain_{}'.format(args.train_name) + ('_contrast' if args.contrast else '_no') + \
          '/{}{}/split{}_shot{}/{}'.format(args.arch, args.layers, args.train_split, args.shot, args.exp_name)