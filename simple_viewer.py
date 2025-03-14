"""A simple example to render a (large-scale) Gaussian Splats

```bash
python examples/simple_viewer.py --scene_grid 13
```
"""

import argparse
import math
import os
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import viser

from gsplat._helper import load_test_data
from gsplat.distributed import cli
from gsplat.rendering import rasterization


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)

    means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []
    for ckpt_path in args.ckpt:
        ckpt = torch.load(ckpt_path, map_location=device)["splats"]
        means.append(ckpt["means"])
        quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
        ckpt_scales = torch.exp(ckpt["scales"])
        if len(ckpt_scales.shape) == 1:
            ckpt_scales = ckpt_scales.unsqueeze(1).repeat(1, 3)
        scales.append(ckpt_scales)
        opacities.append(torch.sigmoid(ckpt["opacities"]))
        sh0.append(ckpt["sh0"])
        # shN.append(ckpt["shN"])
    means = torch.cat(means, dim=0)
    quats = torch.cat(quats, dim=0)
    scales = torch.cat(scales, dim=0)
    opacities = torch.cat(opacities, dim=0)
    sh0 = torch.cat(sh0, dim=0)
    # shN = torch.cat(shN, dim=0)
    colors = sh0
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)

    # # crop
    # aabb = torch.tensor((-1.0, -1.0, -1.0, 1.0, 1.0, 0.7), device=device)
    # edges = aabb[3:] - aabb[:3]
    # sel = ((means >= aabb[:3]) & (means <= aabb[3:])).all(dim=-1)
    # sel = torch.where(sel)[0]
    # means, quats, scales, colors, opacities = (
    #     means[sel],
    #     quats[sel],
    #     scales[sel],
    #     colors[sel],
    #     opacities[sel],
    # )

    # # repeat the scene into a grid (to mimic a large-scale setting)
    # repeats = args.scene_grid
    # gridx, gridy = torch.meshgrid(
    #     [
    #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
    #         torch.arange(-(repeats // 2), repeats // 2 + 1, device=device),
    #     ],
    #     indexing="ij",
    # )
    # grid = torch.stack([gridx, gridy, torch.zeros_like(gridx)], dim=-1).reshape(
    #     -1, 3
    # )
    # means = means[None, :, :] + grid[:, None, :] * edges[None, None, :]
    # means = means.reshape(-1, 3)
    # quats = quats.repeat(repeats**2, 1)
    # scales = scales.repeat(repeats**2, 1)
    # colors = colors.repeat(repeats**2, 1, 1)
    # opacities = opacities.repeat(repeats**2)
    print("Number of Gaussians:", len(means))

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        if args.backend == "gsplat":
            rasterization_fn = rasterization
        elif args.backend == "inria":
            from gsplat import rasterization_inria_wrapper

            rasterization_fn = rasterization_inria_wrapper
        else:
            raise ValueError

        render_colors, render_alphas, meta = rasterization_fn(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            # this is to speedup large-scale rendering by skipping far-away Gaussians.
            radius_clip=3,
        )
        render_rgbs = render_colors[0, ..., 0:3].cpu().numpy()
        return render_rgbs

    server = viser.ViserServer(port=args.port, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    """
    # Use single GPU to view the scene
    CUDA_VISIBLE_DEVICES=0 python simple_viewer.py \
        --ckpt results/garden/ckpts/ckpt_3499_rank0.pt results/garden/ckpts/ckpt_3499_rank1.pt \
        --port 8081
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument(
        "--ckpt", type=str, nargs="+", default=None, help="path to the .pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument("--backend", type=str, default="gsplat", help="gsplat, inria")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
