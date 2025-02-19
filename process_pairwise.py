import os
from datetime import datetime
import glob
from pathlib import Path

import cv2
import math
import numpy as np
import PIL.Image
import torch
import tqdm

from mast3r.model import AsymmetricMASt3R
from dust3r.model import AsymmetricCroCo3DStereo

from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import inv
from dust3r.utils.image import load_images

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment

from mast3r_helpers import get_mast3r_output, scale_intrinsics

from tap import Tap

'''
# original images
--image_dir "/srv/warplab/tektite/jobs/Yawzi_2024_11_14/rosbag_extracted/RAW"

# center cropped images
--image_dir "/srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/raw_downward"

CUDA_VISIBLE_DEVICES=0 \
python process_pairwise.py \
--exp_name "yawzi_mast3r_nocrop_ss3_window2" \
--image_dir "/srv/warplab/tektite/jobs/Yawzi_2024_11_14/rosbag_extracted/RAW" \
--step_size 3 \
--window_size 2 \
--use_rosbag_raw \
--use_mast3r

CUDA_VISIBLE_DEVICES=1 \
python process_pairwise.py \
--exp_name "yawzi_mast3r_midcrop_ss3_window2" \
--image_dir "/srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/raw_downward" \
--step_size 3 \
--window_size 2 \
--use_rosbag_raw \
--use_mast3r
'''


class ArgParser(Tap):
    exp_name: str
    image_dir: str = os.path.expanduser("~/localdata/usvi_nov_2024/yawzi/rectified_imgs_with_caminfo")
    crop_middle: bool = False # take the middle 1024x768 or 2048x1536 instead of the full 1920x1080 or 3840x2160 image
    step_size: int = 1
    num_images: int = 0

    window_size: int = 2

    use_mast3r: bool = False
    mast3r_v1: bool = False
    mast3r_v2: bool = False
    mast3r_v3: bool = False

    use_rosbag_raw: bool = False
    use_7scenes: bool = False

if __name__ == '__main__':
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    args = ArgParser().parse_args()

    now = datetime.now()
    today = now.strftime("%m%d%Y")
    exp_dir = f"experiments/{today}/{args.exp_name}"
    os.makedirs(exp_dir, exist_ok=True)

    if args.use_7scenes:
        sorted_image_list = sorted([Path(fp).name for fp in glob.glob(f"{args.image_dir}/*color.png")])
    elif args.use_rosbag_raw:
        sorted_image_list = sorted([Path(fp).name for fp in glob.glob(f"{args.image_dir}/*.png")])
    else:
        sorted_image_list = sorted(os.listdir(args.image_dir), key=lambda x: float(x.split('_')[1]))

    # culling the number of images we're processing
    if args.step_size != 1:
        sorted_image_list = sorted_image_list[::args.step_size]
    if args.num_images > 0:
        sorted_image_list = sorted_image_list[:args.num_images]

    image_fp_list = [f"{args.image_dir}/{image_fp}" for image_fp in sorted_image_list]

    # you can put the path to a local checkpoint in model_name if needed
    if args.use_mast3r:
        model_name = os.path.expanduser("~/code/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        model = AsymmetricMASt3R.from_pretrained(model_name).to(device)
    else:
        model_name = os.path.expanduser("~/code/mast3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)

    T_world_cam1 = torch.eye(4)
    T_world_cams = [T_world_cam1.unsqueeze(0)]

    if args.window_size == 1:
        num_batches = 1
        step_size = len(image_fp_list)
    else:
        step_size = math.floor(args.window_size / 2)
        num_batches = len([i for i in range(0, len(image_fp_list) - args.window_size + 1, step_size)])

    # detect what the middle crop size should be
    if args.crop_middle:
        img = PIL.Image.open(image_fp_list[0]).convert("RGB")
        h, w = np.array(img).shape[:2]
        if (h, w) == (1080, 1920):
            center_crop = (768, 1024)
        elif (h, w) == (2160, 3840):
            center_crop = (1536, 2048)
        else:
            raise ValueError(f"image is not 1920x1080 or 3840x2160")

    # process the images in batches
    for batch_idx, start_idx in enumerate(tqdm.tqdm(range(0, len(image_fp_list) - args.window_size + 1, step_size))):
        if args.use_mast3r:
            assert args.window_size == 2, "MASt3R code currently only supports pairwise processing"

            # approach 1
            if args.mast3r_v1:
                images = load_images(image_fp_list[start_idx:start_idx+args.window_size], size=512, verbose=False)
                pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
                output = inference(pairs, model, device, batch_size=1, verbose=False)

                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=False, min_conf_thr=1.0) # default min_conf_thr is 3.0
                poses = [p.detach().cpu() for p in scene.get_im_poses()]
                # pts3d = scene.get_pts3d()
                # confidence_masks = scene.get_masks()

                # get relative image poses
                if torch.equal(poses[0], torch.eye(4)):
                    # camera 1 is the reference
                    T_cam1_cam2 = poses[1]
                if torch.equal(poses[1], torch.eye(4)):
                    # camera 2 is the reference
                    T_cam2_cam1 = poses[0]
                    T_cam1_cam2 = torch.linalg.inv(T_cam2_cam1)

                if torch.equal(poses[0], torch.eye(4)) and torch.equal(poses[1], torch.eye(4)):
                    print(f"Warning: both poses are identity matrices for batch: {image_fp_list[start_idx:start_idx+args.window_size]}")

            # approach 2
            elif args.mast3r_v2:
                optim_level = "refine"
                lr1, lr2 = 0.07, 0.014
                niter1, niter2 = 200, 80 #500, 200
                matching_conf_thr = 5.0
                shared_intrinsics = True
                verbose = False

                scene = sparse_global_alignment(image_fp_list[start_idx:start_idx+args.window_size], pairs, "/tmp/mast3r",
                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                    matching_conf_thr=matching_conf_thr, verbose=verbose
                )

                poses = [p.detach().cpu() for p in scene.get_im_poses()]

                mast3r_T_world_cam1 = poses[0]
                mast3r_T_world_cam2 = poses[1]

                T_cam1_cam2 = torch.matmul(torch.linalg.inv(mast3r_T_world_cam1), mast3r_T_world_cam2)

            # approach 3
            elif args.mast3r_v3:
                matches_im0, matches_im1, \
                pts3d_im0, pts3d_im1, \
                conf_im0, conf_im1, \
                desc_conf_im0, desc_conf_im1, \
                K0, K1 = get_mast3r_output(image_fp_list[start_idx:start_idx+args.window_size])

                # this gives transform to bring object points (image 0 points) in image 1 space I think
                retval, rvec, tvec = cv2.solvePnP(
                        objectPoints=pts3d_im0[matches_im0[:, 1], matches_im0[:, 0], :],
                        imagePoints=matches_im1.astype(np.float32),    # ensure same datatype for opencv
                        cameraMatrix=K1,
                        distCoeffs=np.zeros((0,)),
                        flags=cv2.SOLVEPNP_EPNP
                )
                R = cv2.Rodrigues(rvec)[0]  # world to cam
                pose = inv(np.r_[np.c_[R, tvec], [(0, 0, 0, 1)]])  # cam to world
                T_cam2_cam1 = torch.from_numpy(pose.astype(np.float32))
                T_cam1_cam2 = torch.linalg.inv(T_cam2_cam1)
            else:
                assert False
        else:
            if args.window_size == 1:
                images = load_images(image_fp_list, size=512, verbose=False, crop=center_crop if args.crop_middle else None)
            elif args.window_size >= 2:
                images = load_images(image_fp_list[start_idx:start_idx+args.window_size], size=512, verbose=False, crop=center_crop if args.crop_middle else None)

            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            # dxy: I think a larger batch size should be ok here?
            output = inference(pairs, model, device, batch_size=1, verbose=False)

            if args.window_size == 2:
                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=False) # min_conf_thr=2  or whatever if you wanted to play with it
            else:
                scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False)
                loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

            # retrieve useful values from scene:
            # imgs = scene.imgs
            # focals = scene.get_focals()
            poses = [p.detach().cpu() for p in scene.get_im_poses()]
            # TODO: figure out how to use the 3d points and get them all in the same frame to post process plot
            # pts3d = scene.get_pts3d()
            # confidence_masks = scene.get_masks()

            if args.window_size == 2:
                # get relative image poses
                if torch.equal(poses[0], torch.eye(4)):
                    # camera 1 is the reference
                    T_cam1_cam2 = poses[1]
                if torch.equal(poses[1], torch.eye(4)):
                    # camera 2 is the reference
                    T_cam2_cam1 = poses[0]
                    T_cam1_cam2 = torch.linalg.inv(T_cam2_cam1)
                if torch.equal(poses[0], torch.eye(4)) and torch.equal(poses[1], torch.eye(4)):
                    print(f"Warning: both poses are identity matrices for batch: {image_fp_list[start_idx:start_idx+args.window_size]}")
            else:
                # poses are all T_world_cam but we want relative poses from the last cam
                T_world_lastcam = torch.eye(4)
                T_lastcam_currcam_list = []
                for pose_idx, T_world_cam in enumerate(poses):
                    T_lastcam_cam = torch.matmul(torch.linalg.inv(T_world_lastcam), T_world_cam)
                    T_lastcam_currcam_list.append(T_lastcam_cam)
                    T_world_lastcam = T_world_cam

        if args.window_size == 2:
            # get the current cameras in world frame
            T_world_cam2 = torch.matmul(T_world_cam1, T_cam1_cam2)
            T_world_cams.append(T_world_cam2.unsqueeze(0))

            # update last camera reference
            T_world_cam1 = T_world_cam2
        else:
            if batch_idx == 0:
                chain_pose_list_from_idx = 1
            else:
                # window is 6, start from 3, step size is 3
                # window is 5, start from 3, step size is 2
                # window is 4, start from 2, step size is 2
                # window is 3, start from 2, step size is 1
                chain_pose_list_from_idx = math.ceil(args.window_size / 2)

            for T_lastcam_currcam in T_lastcam_currcam_list[chain_pose_list_from_idx:]:
                T_world_currcam = torch.matmul(T_world_cam1, T_lastcam_currcam)
                T_world_cams.append(T_world_currcam.unsqueeze(0))

                # update last camera reference
                T_world_cam1 = T_world_currcam

        if batch_idx % 100 == 0:
            save_T_world_cams = torch.cat(T_world_cams, dim=0)
            torch.save(save_T_world_cams, f"{exp_dir}/poses.pt")

    save_T_world_cams = torch.cat(T_world_cams, dim=0)
    torch.save(save_T_world_cams, f"{exp_dir}/poses.pt")
    print(f"Saved {len(T_world_cams)} poses to {exp_dir}/poses.pt")
