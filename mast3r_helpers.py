from typing import List, Dict, Any

import numpy as np
import torch

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import AsymmetricMASt3R

DEVICE = 'cuda'
MODEL_NAME = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
BORDER = 3

def scale_intrinsics(K: np.ndarray, prev_w: float, prev_h: float) -> np.ndarray:
    """Scale the intrinsics matrix by a given factor .

    Args:
        K (np.ndarray): 3x3 intrinsics matrix
        scale (float): Scale factor

    Returns:
        np.ndarray: Scaled intrinsics matrix
    """
    assert K.shape == (3, 3), f"Expected (3, 3), but got {K.shape=}"

    scale_w = 512.0 / prev_w  # sizes of the images in the Mast3r dataset
    scale_h = 384.0 / prev_h  # sizes of the images in the Mast3r dataset

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_w
    K_scaled[0, 2] *= scale_w
    K_scaled[1, 1] *= scale_h
    K_scaled[1, 2] *= scale_h

    return K_scaled

def get_mast3r_output(image_fp_list: List[str]) -> Dict[str, Any]:
    # Load model, run inference
    model = AsymmetricMASt3R.from_pretrained(MODEL_NAME).to(DEVICE)
    images = load_images(image_fp_list, size=512)
    output = inference([tuple(images)], model, DEVICE,
                       batch_size=1, verbose=False)

    # raw predictions
    # view - dict_keys(['img', 'true_shape', 'idx', 'instance'])
    # pred - dict_keys(['pts3d', 'conf', 'desc', 'desc_conf'])
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=DEVICE, dist='dot', block_size=2**13)

    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= BORDER) & \
                        (matches_im0[:, 0] < int(W0) - BORDER) & \
                        (matches_im0[:, 1] >= BORDER) & \
                        (matches_im0[:, 1] < int(H0) - BORDER)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= BORDER) & \
                        (matches_im1[:, 0] < int(W1) - BORDER) & \
                        (matches_im1[:, 1] >= BORDER) & \
                        (matches_im1[:, 1] < int(H1) - BORDER)

    valid_matches = valid_matches_im0 & valid_matches_im1

    # matches are Nx2 image coordinates.
    matches_im0 = matches_im0[valid_matches]
    matches_im1 = matches_im1[valid_matches]

    # Convert the other outputs to numpy arrays
    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()

    conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy()
    conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()

    desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()

    # Get camera Ks
    pts3d_im0_for_cal = pred1['pts3d'].squeeze(0).detach().cpu()
    pp = torch.tensor((W0/2, H0/2))
    focal = float(estimate_focal_knowing_depth(pts3d_im0_for_cal[None], pp, focal_mode='weiszfeld'))
    K0 = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

    pts3d_im1_for_cal = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu()
    pp = torch.tensor((W1/2, H1/2))
    focal = float(estimate_focal_knowing_depth(pts3d_im1_for_cal[None], pp, focal_mode='weiszfeld'))
    K1 = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

    return matches_im0, matches_im1, pts3d_im0, pts3d_im1, conf_im0, conf_im1, desc_conf_im0, desc_conf_im1, K0, K1
