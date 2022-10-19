from .pspnet import PSPNet
from torch import nn

def get_model(args) -> nn.Module:
    return PSPNet(args, zoom_factor=8, use_ppm=True)