
import torch
from collections import OrderedDict

fpath = './pretrained_models/pascal/split=2/pspnet_resnet50/'
pre_weight = torch.load(fpath+'best0.pth')    # ['epoch', 'state_dict', 'optimizer']

wt = pre_weight['state_dict']
new_wt = OrderedDict()
for k, v in wt.items():
    new_wt[k.replace('module.', '')] = v

pre_weight['state_dict'] = new_wt

torch.save(pre_weight, fpath+'best1.pth')

