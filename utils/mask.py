import torch

def get_yawzi_downward_mask():
    H = 560
    W = 994
    mask = torch.ones((H, W))
    mask[-72:, 145:241] = 0       # robot leg
    mask[-8:, 369:609] = 0        # some distortion artifact
    return mask.unsqueeze(0).unsqueeze(-1) # [1, H, W, 1], can multiply with [B, H, W, 3] image
