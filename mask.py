import os
import sys
import time
import pickle
import numpy as np
from addict import Dict

from PIL import Image

import torch
import torch.nn as nn
from torch.utils import data

import torchvision.transforms as transform

# Default Work Dir: /scratch/[NetID]/SSeg/
BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)
# config: sys.argv[1]
MODEL_DIR = os.path.join(BASE_DIR, 'mask', sys.argv[1])
OUTPUT_DIR = os.path.join(MODEL_DIR, '%s_pred' % sys.argv[1])



class Masker():
    def __init__(self,):
        self.colors = self.generate_colors()

    def generate_colors(self):
        colors = []
        t = 255 * 0.2
        for i in range(1, 5):
            for j in range(1, 5):
                for k in range(1, 5):
                    colors.append(np.array([t * i, t * j, t * k], dtype=np.uint8))
        while len(colors) <= 256:
            colors.append(np.array([0, 0, 0], dtype=np.uint8))
        return colors

    def mask_to_rgb(self, t):
        assert len(t.shape) == 2
        t = t.numpy().astype(np.uint8)
        rgb = np.zeros((t.shape[0], t.shape[1], 3), dtype=np.uint8)
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                rgb[i, j, :] = self.colors[t[i, j]]
        return rgb  # Image.fromarray(rgb)

    def denormalize(self, input_image, mean, std, imtype=np.uint8):
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # if it's torch.Tensor, then convert
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))

            image_numpy = image_numpy * 255  # [0,1] to [0,255]
            image_numpy = np.transpose(image_numpy, (1, 2, 0))  # chw to hwc
        else:
            image_numpy = input_image
        return image_numpy.astype(imtype)

    def gen_mask(self):

        with open(os.path.join(OUTPUT_DIR, 'colors.txt'), 'w') as f:
            for i in range(self.nclass):
                # print('%s %s' % (self.testset.classes[i], tuple(self.colors[i])))
                f.write('%s|%s\n' % (self.testset.classes[i], self.colors[i]))

        self.model.eval()
        for i, (image, dep, target) in enumerate(self.valloader):

            last_file = os.path.join(OUTPUT_DIR, 'comb_%03d.jpg' % (self.args.batch_size * (i + 1) - 1))
            if os.path.isfile(last_file):
                print('Pred Step %03d skipped.' % i)
                continue

            if self.args.cuda:
                image, dep, target = image.to(self.device), dep.to(self.device), target.to(self.device)

            pred = torch.argmax(self.model(image, dep)[0], dim=1)
            if self.args.cuda:
                pred.cpu()

            for j in range(self.args.batch_size):
                curr_rgb = self.denormalize(image[j], mean=[.485, .456, .406], std=[.229, .224, .225])
                dep_size = dep[j].size()
                curr_dep = self.denormalize(dep[j].expand(3, dep_size[1], dep_size[2]), mean=[0.2798], std=[0.1387])
                mask_gt = self.mask_to_rgb(target[j])
                mask_pred = self.mask_to_rgb(pred[j])

                part1 = np.concatenate((curr_rgb, curr_dep), axis=1)
                part2 = np.concatenate((mask_gt, mask_pred), axis=1)
                res = np.concatenate((part1, part2), axis=0)
                comb_path = os.path.join(OUTPUT_DIR, 'comb_%03d.jpg' % (self.args.batch_size * i + j))
                img = Image.fromarray(res)
                img.save(comb_path)

                curr_dir = os.path.join(OUTPUT_DIR, '%03d' % (self.args.batch_size * i + j))
                if not os.path.isdir(curr_dir):
                    os.makedirs(curr_dir)
                else:
                    continue

                gt_path = os.path.join(curr_dir, 'gt.jpg')
                img = Image.fromarray(mask_gt)
                img.save(gt_path)

                rgb_path = os.path.join(curr_dir, 'rgb.jpg')
                img = Image.fromarray(curr_rgb)
                img.save(rgb_path)

                dep_path = os.path.join(curr_dir, 'dep.jpg')
                img = Image.fromarray(curr_dep)
                img.save(dep_path)

                pred_path = os.path.join(curr_dir, 'pred_path.jpg')
                img = Image.fromarray(mask_pred)
                img.save(pred_path)

                print('Pred Step %03d-%d' % (i, j))


if __name__ == "__main__":
    print('[Model Info]:', sys.argv[1])
    print("-------mark program start----------")
    # configuration
    args = Dict(get_config(sys.argv[1]))
    # args.training.cuda = (args.training.use_cuda and torch.cuda.is_available())
    args.training.cuda = False
    if len(sys.argv) > 3:
        print('debugging mode...')
        args.training.workers = 1
    masker = Masker(args)