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