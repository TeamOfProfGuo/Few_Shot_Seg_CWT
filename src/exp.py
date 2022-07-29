
LOSS_ID = {'c': 'ce', 'd': 'wdc', 'w': 'wce'}
RATIO_ID = {'a': 0.0, 'b': 0.1, 'c': 0.3, 'd': 0.5, 'e': 1.0}

def modify_config(args, date, group, info):

    # init dict for modified args
    exp_dict = {}

    # skip as needed
    if date is None:
        return args, exp_dict

    # [bca]: base model with color augmentation
    if group == "bca":
        assert len(info) == 4
        exp_dict['brightness'] = int(info[0]) / 10
        exp_dict['contrast'] = int(info[1]) / 10
        exp_dict['saturation'] = int(info[2]) / 10
        exp_dict['hue'] = int(info[3]) / 10
        exp_dict['augmentations'] = ['color_aug', 'hor_flip', 'resize_np']

    # apply experiment settings
    for k, v in exp_dict.items():
        args.__setattr__(k, v)

    return args, exp_dict
