from datetime import datetime
import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser, SpatialDataset
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam

from seasplat.losses import SmoothDepthLoss
from splat.model import create_splats_with_optimizers, add_new_frame
from utils.gsplat_utils import set_random_seed, seed_worker, AffineExposureOptModule
from utils.mask import get_yawzi_downward_mask
@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "/srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/metashape_export"
    # Downsample factor for the dataset
    data_factor: int = 2
    # Experiment name to save results to
    exp_name: str = "scratch"
    # Every N images there is a test image
    test_every: int = 1e8
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = False
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # incremental relevant parameters
    # number of initial images to use for initialization
    num_init_images: int = 10
    # number of initial optimization steps
    num_init_steps: int = 1000
    # number of optimization steps for every new image
    num_add_steps: int = 100
    # sliding window size for optimizing
    sliding_window: int = 5

    full_splat_opt: bool = False
    full_splat_opt_every: int = 100
    full_splat_opt_steps: int = 1000

    apply_robo_mask: bool = False

    smooth_depth_loss: bool = False
    smooth_depth_lambda: float = 2.0

    # use isotropic gaussians
    isotropic: bool = True

    exposure_optimization: bool = False
    exposure_lr_init: float = 0.001


    # debugging
    start_image_idx: int = 0
    num_total_images: int = 1e10

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 0 # originally 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.device = "cuda"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="all",
            load_depths=True,
        )
        print(f"Dataset length: {len(self.trainset)}")

        # Obtain init pointcloud from parser
        global_pcd_indices = set([])
        for idx in range(cfg.num_init_images):
            data = self.trainset[cfg.start_image_idx + idx]
            global_pcd_indices = global_pcd_indices.union(data["point_indices"])
        global_pcd_indices = list(global_pcd_indices)

        sfm_init_points = self.parser.points[global_pcd_indices]
        sfm_init_colors = self.parser.points_rgb[global_pcd_indices]

        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        if cfg.init_type == "sfm":
            self.splats, self.optimizers = create_splats_with_optimizers(
                init_type=cfg.init_type,
                points=sfm_init_points,
                rgb=sfm_init_colors,
                visible_adam=cfg.visible_adam,
                isotropic=cfg.isotropic,
            )
        elif cfg.init_type == "random":
            self.splats, self.optimizers = create_splats_with_optimizers(
                init_type=cfg.init_type,
                init_num_pts=cfg.init_num_pts,
                init_extent=cfg.init_extent,
                visible_adam=cfg.visible_adam,
                isotropic=cfg.isotropic,
            )
        else:
            assert False
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.exposure_optimizers = []
        if cfg.exposure_optimization:
            self.exposure = AffineExposureOptModule(len(self.trainset)).to(self.device)
            self.exposure_optimizers = [
                torch.optim.Adam(
                    self.exposure.parameters(),
                    lr=cfg.exposure_lr_init,
                )
            ]
            print(f"Exposure optimization enabled for {len(self.trainset)} cameras")

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Losses and Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        self.smooth_depth_criterion = SmoothDepthLoss()

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        # misc
        if cfg.apply_robo_mask:
            self.robo_mask = get_yawzi_downward_mask().to(self.device)

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]

        scales = torch.exp(self.splats["scales"])  # [N, 3]
        # makes scales N,3 if isotropic
        if len(scales.shape) == 1:
            scales = scales.unsqueeze(1).repeat(1, 3)

        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if "shN" in self.splats:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
        else:
            colors = self.splats["sh0"]  # [N, 1, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device

        # Dump cfg.
        with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
            yaml.dump(vars(cfg), f)

        self.max_steps = cfg.max_steps
        init_step = 0

        self.schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / self.max_steps)
            ),
        ]
        if cfg.exposure_optimization:
            # exposure optimization has a learning rate schedule
            self.schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.exposure_optimizers[0], gamma=0.01 ** (1.0 / self.max_steps)
                )
            )

        # Iterative training loop
        self.global_tic = time.time()
        self.global_step = 0
        pbar = tqdm.tqdm(range(len(self.trainset)))
        for idx in pbar:
            if idx < cfg.start_image_idx + cfg.num_init_images:
                continue
            elif idx == cfg.start_image_idx + cfg.num_init_images:
                self.splat_optimization(pbar, cfg.num_init_steps, [i for i in range(cfg.start_image_idx, cfg.start_image_idx + cfg.num_init_images)])
            elif idx == cfg.start_image_idx + cfg.num_init_images + cfg.num_total_images:
                print(f"Reached {cfg.start_image_idx + cfg.num_init_images + cfg.num_total_images} images, stopping training")
                break

            # add a new image to the gaussian model
            data = self.trainset[idx]
            add_new_frame(
                rgb_image=data["image"].to(self.device),
                pcd_points=data["points_xyz"].to(self.device),
                pcd_rgb=data["points_rgb"].to(self.device),
                T_world_camera=data["camtoworld"].to(self.device),
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
            )
            self.splat_optimization(pbar, cfg.num_add_steps, [i for i in range(idx - cfg.sliding_window + 1, idx + 1)])

            if idx % cfg.full_splat_opt_every == 0 and cfg.full_splat_opt:
                self.splat_optimization(pbar, cfg.full_splat_opt_steps, [i for i in range(idx)])

        print(f"Finished incrementally adding the whole dataset in {self.global_step} steps in {time.time() - self.global_tic} seconds")
        print(f"Saving model")

        data = {"step": self.global_step, "splats": self.splats.state_dict()}
        torch.save(
            data, f"{self.ckpt_dir}/ckpt_{self.global_step}_rank0.pt"
        )
        self.render_traj(self.global_step)
        print(f"Optimizing the whole scene for {self.max_steps - self.global_step} steps")
        self.splat_optimization(pbar, self.max_steps - self.global_step + 1, [i for i in range(len(self.trainset))])

        data = {"step": self.global_step, "splats": self.splats.state_dict()}
        torch.save(
            data, f"{self.ckpt_dir}/ckpt_{self.global_step}_rank0.pt"
        )
        self.render_traj(self.global_step)

    def splat_optimization(self, pbar, num_steps, select_idxs, dbg=False):
        for step in range(num_steps):
            if not self.cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            dset_idx = np.random.choice(select_idxs)
            data = self.trainset[dset_idx]

            # parse data
            camtoworlds = camtoworlds_gt = data["camtoworld"].to(self.device).unsqueeze(0)  # [1, 4, 4]
            Ks = data["K"].to(self.device).unsqueeze(0)  # [1, 3, 3]
            pixels = data["image"].to(self.device).unsqueeze(0) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(self.device).unsqueeze(0)
            masks = data["mask"].to(self.device).unsqueeze(0) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(self.device).unsqueeze(0)  # [1, M, 2]
                depths_gt = data["depths"].to(self.device).unsqueeze(0)  # [1, M]

            height, width = pixels.shape[1:3]
            sh_degree_to_use = cfg.sh_degree

            # forward pass
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.smooth_depth_loss else "RGB", # ED = expected depth (D / alpha)
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=self.device)
                colors = colors + bkgd * (1.0 - alphas)

            if cfg.exposure_optimization:
                colors = self.exposure(colors, torch.from_numpy(np.array([dset_idx])).to(self.device))

            # strategy pre backward
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.global_step,
                info=info,
            )

            # calculate losses and backward
            if cfg.apply_robo_mask:
                l1loss = F.l1_loss(colors * self.robo_mask, pixels * self.robo_mask)
            else:
                l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda

            if cfg.smooth_depth_loss:
                smooth_depth_loss = self.smooth_depth_criterion(
                    colors.permute(0, 3, 1, 2),
                    depths.permute(0, 3, 1, 2),
                    mask=self.robo_mask.permute(0, 3, 1, 2) if cfg.apply_robo_mask else None
                )
                loss += cfg.smooth_depth_lambda * smooth_depth_loss

            loss.backward()

            # a bunch of logging
            desc = f"loss={loss.item():.3f}| " f"num gs: {len(self.splats['means'])}| " f"step={self.global_step}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            pbar.set_description(desc)

            if cfg.tb_every > 0 and self.global_step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), self.global_step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), self.global_step)
                if cfg.smooth_depth_loss:
                    self.writer.add_scalar("train/smoothdepthloss", smooth_depth_loss.item(), self.global_step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), self.global_step)
                self.writer.add_scalar("train/mem", mem, self.global_step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), self.global_step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, self.global_step)
                self.writer.flush()

            # save checkpoint before updating the model
            if self.global_step in [i - 1 for i in cfg.save_steps] or self.global_step == self.max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - self.global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", self.global_step, stats)
                with open(
                    f"{self.stats_dir}/train_step{self.global_step:04d}_rank0.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": self.global_step, "splats": self.splats.state_dict()}
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{self.global_step}_rank0.pt"
                )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                visibility_mask = (info["radii"] > 0).any(0)

            # Optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.exposure_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in self.schedulers:
                scheduler.step()

            # strategy post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=self.global_step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=self.global_step,
                    info=info,
                    lr=self.schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # bookkeeping
            self.global_step += 1

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(self.global_step, num_train_rays_per_step)


    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()

    @torch.no_grad()
    def render_traj(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all = self.parser.camtoworlds[5:-5]
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # save to video
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/traj_{step}.mp4", fps=30)
        for i in tqdm.trange(len(camtoworlds_all), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            depths = renders[..., 3:4]  # [1, H, W, 1]
            depths = (depths - depths.min()) / (depths.max() - depths.min())
            canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

            # write images
            canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)
            writer.append_data(canvas)
        writer.close()
        print(f"Video saved to {video_dir}/traj_{step}.mp4")


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    CUDA_VISIBLE_DEVICES=0 python simple_incremental_trainer.py default \
    --steps_scaler 10 \
    --exp_name scratch \
    --num_init_images 10 \
    --num_init_steps 1000 \
    --num_add_steps 100
    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(
                    verbose=True,
                    grow_grad2d=0.0002, #0.0008,
                    grow_scale3d=0.01, # / 15.0,
                    prune_scale3d=0.1, # / 15.0,
                    refine_start_iter = 250, # probably similar to num_init_steps
                    refine_stop_iter = 500_000,
                    reset_every = 250, # this is not used / commented out /shrug
                    refine_every = 100, # probably similar to num_add_steps,
                    absgrad=False, # if absgrad is True, should set grow_grad2d = 0.0008
                    revised_opacity=False, # default False
                    do_opacity_reset=False,
                ),
                visible_adam=True,
                num_init_images=10,
                num_init_steps=500,
                num_add_steps=50,
                sliding_window=5,
                max_steps=500_000,
                save_steps=[7_000, 50_000, 100_000, 150_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000],
                full_splat_opt = False,
                full_splat_opt_every = 100,
                full_splat_opt_steps = 100,
                far_plane=10.0
            ),
        ),
        "original_default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)
    now = datetime.now()
    today = now.strftime("%m%d%Y")
    cfg.result_dir = f"experiments/{today}/{cfg.exp_name}"

    runner = Runner(0, 0, 1, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
