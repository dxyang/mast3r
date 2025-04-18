import os
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch import Tensor
from torchvision.utils import save_image
from tqdm import tqdm

import viser

from gsplat.rendering import rasterization

from datasets.colmap import Dataset, Parser

from utils.gsplat_utils import AffineExposureOptModule, CameraOptModule
from utils.pytorch3d_utils import so3_relative_angle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG_SH_DEGREE = 0
CFG_NEAR_PLANE = 0.01
CFG_FAR_PLANE = 10.0
CFG_CAMERA_MODEL = "pinhole"

def load_splats(model_path: str):
    data = torch.load(model_path)

    print(f"Loaded model from {model_path}")
    print(f"Step: {data['step']}")
    print(f"Num gaussians: {data['splats']['means'].shape[0]}")
    print(f"Exposure optimization: {'yes' if 'exposure_opt' in data else 'no'}")
    print(f"Pose optimization: {'yes' if 'pose_adjust' in data else 'no'}")

    return data

def rasterize_splats(
    splats: torch.nn.ParameterDict,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    masks: Optional[Tensor] = None,
    **kwargs
):
    means = splats["means"]  # [N, 3]
    # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
    # rasterization does normalization internally
    quats = splats["quats"]  # [N, 4]

    scales = torch.exp(splats["scales"])  # [N, 3]
    # makes scales N,3 if isotropic
    if len(scales.shape) == 1:
        scales = scales.unsqueeze(1).repeat(1, 3)

    opacities = torch.sigmoid(splats["opacities"])  # [N,]

    if "shN" in splats:
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]
    else:
        colors = splats["sh0"]  # [N, 1, 3]

    rasterize_mode = "classic"

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
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode=rasterize_mode,
        distributed=False,
        camera_model=CFG_CAMERA_MODEL,
        **kwargs,
    )
    if masks is not None:
        render_colors[~masks] = 0
    return render_colors, render_alphas, info

def estimate_camera_pose(
    rgb_image: Tensor,
    K: Tensor,
    T_world_camInit: Tensor,
    splats: torch.nn.ParameterDict,
    do_expo_opt: bool = False,
    max_iter: int = 100,
    lr: float =1e-4,
    dbg: bool = False,
    alpha_threshold: float = 0.0,
    scene: viser.SceneApi = None,
    scene_str: str = None,
    visible: bool = True
) -> Tensor:
    # Initialize the camera optimization module with the initial guess
    cam_opt_module = CameraOptModule(1).to(device)
    cam_opt_module.zero_init()

    # Exposure optimization
    if do_expo_opt:
        exposure_opt_module = AffineExposureOptModule(1).to(device)
        expo_optimizer = torch.optim.Adam(exposure_opt_module.parameters(), lr=1e-4)

    # Define the optimizer
    pose_optimizer = torch.optim.Adam(cam_opt_module.parameters(), lr=lr)

    # Define the loss function
    criterion = torch.nn.L1Loss()

    # do once things
    Ks = K.unsqueeze(0)
    width, height = rgb_image.shape[1], rgb_image.shape[0]
    image_id = torch.tensor([0], device=device)
    rgb_01_batch = rgb_image.unsqueeze(0) / 255.0

    # Iterate to refine the camera pose
    if dbg:
        pbar = tqdm(range(max_iter))
    else:
        pbar = range(max_iter)
    for iter in pbar:  # Number of iterations can be adjusted
        pose_optimizer.zero_grad()
        if do_expo_opt:
            expo_optimizer.zero_grad()

        # Apply the transformation to the camera pose
        T_world_cam = cam_opt_module(T_world_camInit.unsqueeze(0), image_id)

        # Render the scene with the current camera pose
        renders, alphas, info = rasterize_splats(
            splats=splats,
            camtoworlds=T_world_cam,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=CFG_SH_DEGREE,
            near_plane=CFG_NEAR_PLANE,
            far_plane=CFG_FAR_PLANE,
            render_mode="RGB"
        )
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        # Compute the loss between the rendered image and the input image
        # only on pixels above some alpha confidence threshold (if 0.0, use all pixels)
        masked_colors = torch.where(alphas >= alpha_threshold, colors, 0.0)
        masked_rgb = torch.where(alphas >= alpha_threshold, rgb_01_batch, 0.0)
        loss = criterion(masked_colors, masked_rgb)

        # Backpropagate the loss and update the camera pose
        loss.backward()
        pose_optimizer.step()
        if do_expo_opt:
            expo_optimizer.step()

        # dbg = Path("/srv/warplab/dxy/mast3r_experiments/03242025/yawzi_ss_hgs_params_10x_re100_poseopt1e4/pose_estimate/image_hats")
        # save_image(colors.permute(0, 3, 1, 2), dbg / f"img_{iter}.png")

        if scene is not None:
            # update the scene with the current camera pose
            with torch.no_grad():
                T_world_cam = cam_opt_module(T_world_camInit.unsqueeze(0), image_id).squeeze().cpu().numpy()
                rot_wxyz = R.from_matrix(T_world_cam[:3, :3]).as_quat(scalar_first=True)
                pos_xyz = T_world_cam[:3, 3]
                scene.add_frame(
                    name=scene_str,
                    wxyz=rot_wxyz,
                    position=pos_xyz,
                    axes_length=0.1,
                    axes_radius=0.005,
                    visible=visible,
                )

        # logging
        if dbg:
            desc = f"loss={loss.item():.3f} "
            pbar.set_description(desc)

    # Return the refined camera pose
    T_world_cam = cam_opt_module(T_world_camInit.unsqueeze(0), image_id)
    return T_world_cam.detach().squeeze(), loss.item(), colors.detach().squeeze(), alphas[0].detach()


if __name__ == "__main__":
    # functions
    estimate_pose = True
    render_images = True

    # arguments
    data_dir = Path("/srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/metashape_export")
    data_factor = 2
    normalize_world_space = False
    test_every = 1e8
    center_crop = True

    start_offset = 1
    num_opt_iter = 500
    print(f"Starting pose estimation at offset {start_offset}")
    print(f"Number of optimization iterations: {num_opt_iter}")
    skip_save_img = True

    model_path = Path("/srv/warplab/dxy/mast3r_experiments/03242025/yawzi_ss_hgs_params_10x_re100_poseopt1e4/ckpts/ckpt_300000_rank0.pt")

    # output
    exp_path = model_path.parent.parent
    output_dir = exp_path / "pose_estimate"
    img_output_dir = output_dir / f"image_hats_{start_offset}_{num_opt_iter}"
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir, exist_ok=True)
    output_fp = output_dir / f"pose_estimate_{start_offset}_{num_opt_iter}_data.pkl"

    # load the model
    # TODO: make the splats not require_grad?
    splat_data = load_splats(str(model_path))
    splats = splat_data["splats"]

    # load the dataset
    parser = Parser(
        str(data_dir),
        factor=2,
        normalize=normalize_world_space,
        test_every=test_every,
        center_crop=center_crop,
    )
    trainset = Dataset(
        parser,
        split="all",
        load_depths=False,
    )

    if estimate_pose:
        # for each image in the trainset...
        # intialize the camera pose with the last camera pose in the sequence
        t_errors = []
        rot_errors = []
        l1_errors = []
        T_world_camEstimates = []
        T_world_camInits = []
        T_world_camGts = []
        for i in tqdm(range(len(trainset))):
            data = trainset[i]
            img_fp = Path(parser.image_paths[data['image_id']])
            img_hat_fp = img_output_dir / img_fp.name

            rgb_image = data["image"].to(device) # HWC
            T_world_camGt = data["camtoworld"].to(device) # 4x4
            if i < start_offset:
                T_world_camInit = T_world_camGt.clone()
            else:
                T_world_camInit = trainset[i - start_offset]["camtoworld"].to(device)
            K = data["K"].to(device) # 3x3

            # estimate the camera pose
            T_world_camEstimate, recon_loss, rgb_hat, alpha_hat = estimate_camera_pose(rgb_image, K, T_world_camInit, splats=splats, max_iter=num_opt_iter, do_expo_opt=False)

            # calculate l2 distance of translation error
            t_error = torch.norm(T_world_camEstimate[:3, 3] - T_world_camGt[:3, 3], p=2)
            t_errors.append(t_error.cpu().numpy())

            # calculate rotation error
            rot_error_rads = so3_relative_angle(T_world_camEstimate[:3, :3].unsqueeze(0), T_world_camGt[:3, :3].unsqueeze(0)).squeeze()
            rot_errors.append(rot_error_rads.cpu().numpy())

            # calculate pose l1 error
            l1_error = torch.norm(T_world_camEstimate - T_world_camGt, p=1)
            l1_errors.append(l1_error.cpu().numpy())

            # bookkeeping
            T_world_camEstimates.append(T_world_camEstimate.cpu().numpy())
            T_world_camInits.append(T_world_camInit.cpu().numpy())
            T_world_camGts.append(T_world_camGt.cpu().numpy())

            # save rendered image for later viz debug
            if not skip_save_img:
                save_image(rgb_hat.permute(2, 0, 1).unsqueeze(0), img_hat_fp)

        output_data = {
            "t_errors": np.stack(t_errors),
            "rot_errors": np.stack(rot_errors),
            "l1_errors": np.stack(l1_errors),
            "T_world_camEstimates": np.stack(T_world_camEstimates),
            "T_world_camInits": np.stack(T_world_camInits),
            "T_world_camGts": np.stack(T_world_camGts)
        }
        with open(output_fp, "wb") as f:
            pickle.dump(output_data, f)
        print(f"Saved pose estimation data to {output_fp}")

    if render_images:
        print(f"Rendering images....")
        with torch.no_grad():
            # load the results pickle
            results = pickle.load(open(output_fp, "rb"))
            print(f"Loaded results from {output_fp}")

            # do once set up
            rgb_image = trainset[0]["image"].to(device) # HWC
            width, height = rgb_image.shape[1], rgb_image.shape[0]
            K = trainset[0]["K"].to(device) # 3x3
            Ks = K.unsqueeze(0)

            # render image for each estimated camera pose
            # assert len(trainset) == len(results["T_world_camEstimates"])
            T_world_camEstimates = results["T_world_camEstimates"]
            for i in tqdm(range(len(trainset))):
                img_fp = Path(parser.image_paths[trainset[i]['image_id']])
                img_hat_fp = img_output_dir / img_fp.name

                T_world_camEst = torch.from_numpy(T_world_camEstimates[i]).unsqueeze(0).to(device)

                renders, _, _ = rasterize_splats(
                    splats=splats,
                    camtoworlds=T_world_camEst,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=CFG_SH_DEGREE,
                    near_plane=CFG_NEAR_PLANE,
                    far_plane=CFG_FAR_PLANE,
                    render_mode="RGB"
                )
                if renders.shape[-1] == 4:
                    colors, depths = renders[..., 0:3], renders[..., 3:4]
                else:
                    colors, depths = renders, None

                save_image(colors.squeeze().permute(2, 0, 1).unsqueeze(0), img_hat_fp)

        print(f"Done rendering images....")
