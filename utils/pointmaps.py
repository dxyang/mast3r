import torch

def transform_pointmap(pointmap_wrt_b, T_a_b):
    # pointmap: W x H x 3
    # T_a_b: 4 x 4
    # return: W x H x 3
    dev = pointmap_wrt_b.device
    pcd_wrt_b = pointmap_wrt_b.view(-1, 3)
    hpcd_wrt_b = torch.cat([pcd_wrt_b, torch.ones(pcd_wrt_b.shape[0], 1).to(dev)], dim=1) # N x 4
    hpcd_wrt_a = torch.matmul(T_a_b, hpcd_wrt_b.T).T # N x 4
    pcd_wrt_a = hpcd_wrt_a[:, :3] # N x 3
    pointmap_wrt_a = pcd_wrt_a.reshape(pointmap_wrt_b.shape)
    return pointmap_wrt_a

