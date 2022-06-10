
import torch
import torch.nn.functional as F

def get_corr(q, k):
    bs, ch, height, width = q.shape
    proj_q = q.view(bs, ch, height * width).permute(0, 2, 1)  # [1, ch, hw] -> [1, hw, ch]
    proj_k = k.view(bs, -1, height * width)  # [1, ch, hw]

    proj_q = F.normalize(proj_q, dim=-1)
    proj_k = F.normalize(proj_k, dim=-2)
    sim = torch.bmm(proj_q, proj_k)  # [1, 3600 (q_hw), 3600(k_hw)]
    return sim

