
import torch
import numpy as np
import torch.nn.functional as F


def cross_entropy(logits: torch.tensor, one_hot: torch.tensor, targets: torch.tensor, mean_reduce: bool = True,
                  ignore_index: int = 255) -> torch.tensor:
    """
    inputs: one_hot  : shape [batch_size, num_classes, h, w]
            logits : shape [batch_size, num_classes, h, w]
            targets : shape [batch_size, h, w]
    returns:loss: shape [batch_size] or [] depending on mean_reduce
    """
    assert logits.size() == one_hot.size()
    log_prb = F.log_softmax(logits, dim=1)
    non_pad_mask = targets.ne(ignore_index)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.masked_select(non_pad_mask)
    if mean_reduce:
        return loss.mean()  # average later
    else:
        return loss


def CE_loss(args, logits, targets, num_classes):
    """
    inputs:  images  : shape [batch_size, C, h, w]
             logits : shape [batch_size, num_classes, h, w]
             targets : shape [batch_size, h, w]
    returns: loss: shape []
             logits: shape [batch_size]
             logits = model(images)
    """
    batch, h, w = targets.size()
    one_hot_mask = torch.zeros(batch, num_classes, h, w, device=targets.device)
    new_target = targets.clone().unsqueeze(1)
    new_target[new_target == 255] = 0

    one_hot_mask.scatter_(1, new_target, 1).long()
    if args.smoothing:
        eps = 0.1
        one_hot = one_hot_mask * (1 - eps) + (1 - one_hot_mask) * eps / (num_classes - 1)
    else:
        one_hot = one_hot_mask  # [batch_size, num_classes, h, w]

    loss = cross_entropy(logits, one_hot, targets)
    return loss


