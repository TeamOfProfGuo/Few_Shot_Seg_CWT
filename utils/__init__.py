from .metrics import BinaryIOU
from .metrics import FullIOU
from .operations import NormConv2d, generate_location_features, decode_seg_map, pca, get_binary_logits
from .operations import replace_array_ele_as_dict, sort_dict_by

import os
import shutil

def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove or (basename.startswith('_') or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)
