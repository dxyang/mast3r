import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


from gsplat.optimizers import SelectiveAdam
from gsplat.strategy.ops import _update_param_with_optimizer

from utils.gsplat_utils import knn, rgb_to_sh

'''
let's make some assumptions
* you have your typical COLMAP output (camera poses, intrinsics, sfm pointcloud)
* we want to incrementally build our gaussian splatting map
* at every timestep, we will have 
    * a new RGB camera image
    * sparse depth (from the pointcloud)
    * camera pose

* eventually, we can relax the our assumptions
'''

def create_splats_with_optimizers(
    init_type: str = "sfm",
    points: np.ndarray = None,    # N x 3
    rgb: np.ndarray = None,       # uint8
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 0,           # default to just modelling color in each gaussian
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    isotropic: bool = False,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    '''
    adapted and modified from gsplat code
    '''
    if init_type == "sfm":
        points = torch.from_numpy(points).float()
        rgbs = torch.from_numpy(rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    if isotropic:
        scales = torch.log(dist_avg * init_scale)  # [N,]
    else:
        scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    import pdb; pdb.set_trace()

    N = points.shape[0]

    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]
    if not isotropic:
        params.append(("quats", torch.nn.Parameter(quats), 1e-3))

    # color is SH coefficients.
    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
    if sh_degree > 0:
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers

def initialize_first_timestep():
    '''
    given a few images, initialize the splats?
    maybe uses dust3r output + hardcoded scaling?
    tentatively could use sfm pointcloud from metashape
    '''
    pass

def add_new_frame(
    rgb_image: np.ndarray = None,     # uint8
    pcd_points: np.ndarray = None,    # N x 3
    pcd_rgb: np.ndarray = None,       # uint8
    T_world_camera: np.ndarray = None,    # 4 x 4
    params: torch.nn.ParameterDict = None,
    optimizers: Dict[str, torch.optim.Optimizer] = None,
):
    is_isotropic = "quats" in params
    use_sh = "shN" in params
    init_opacity = 0.1
    init_scale = 1.0
    sh_degree = 0

    new_points = torch.from_numpy(pcd_points).float()
    rgbs = torch.from_numpy(pcd_rgb / 255.0).float()
    N = new_points.shape[0]

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(new_points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    if is_isotropic:
        new_scales = torch.log(dist_avg * init_scale)  # [N,]
    else:
        new_scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    new_quats = torch.rand((N, 4))  # [N, 4]
    new_opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
    colors[:, 0, :] = rgb_to_sh(rgbs)
    new_sh0 = torch.nn.Parameter(colors[:, :1, :])
    if use_sh:
        new_shN = torch.nn.Parameter(colors[:, 1:, :])

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "means":
            new_p = torch.cat([p, new_points])
        elif name == "scales":
            new_p = torch.cat([p, new_scales])
        elif name == "opacities":
            new_p = torch.cat([p, new_opacities])
        elif name == "quats":
            new_p = torch.cat([p, new_quats])
        elif name == "sh0":
            new_p = torch.cat([p, new_sh0])
        elif name == "shN":
            new_p = torch.cat([p, new_shN])
        else:
            assert False

        return torch.nn.Parameter(new_p, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((N, *v.shape[1:]))])

    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    # not sure yet if I need to worry about the "state" referenced in many of the gsplat docs
