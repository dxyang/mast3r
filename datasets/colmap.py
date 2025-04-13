import os
from pathlib import Path
import json
from typing import Any, Dict, List, Optional
from typing_extensions import assert_never

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.stats import gaussian_kde
import torch
from pycolmap import SceneManager

from utils.csv_odom import read_csv_apriltag, read_csv_odom, slerp_closets_odomcam, K_T_COLMAPWORLD_GTSAMWORLD
from utils.depth_image import depth_image_to_pcd

from .dvl_data import DvlDataset

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        center_crop: bool = False, # adjust camera K to center crop (so 1/2 resolution)
        monodepth_key: str = "None"
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        self.manager = manager
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # HACK
        if center_crop:
            assert factor == 2, "Only support 2x center crop"

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = manager.cameras[camera_id]
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            if center_crop:
                assert cx == cam.width // factor, "Only support center crop for now"
                assert cy == cam.height // factor, "Only support center crop for now"
                K[0, 2] -= cam.width // (factor * 2)
                K[1, 2] -= cam.height // (factor * 2)
            else:
                K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            type_ = cam.camera_type
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                params = np.array([cam.k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective" or camtype == "fisheye"
            ), f"Only perspective and fisheye cameras are supported, got {type_}"

            params_dict[camera_id] = params
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        # Load bounds if possible (only used in forward facing scenes).
        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # Load images.
        if center_crop:
            image_dir_suffix = "_cc"
        elif factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        self.image_dir = image_dir

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        # colmap_to_image = dict(zip(colmap_files, image_files))
        # image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        # dxy: let's assume the downsampled images have the same name
        image_paths = [os.path.join(image_dir, f) for f in image_names]

        # monodepth files
        depth_dir = os.path.join(data_dir, "depth" + image_dir_suffix)
        depth_dir = os.path.join(depth_dir, monodepth_key)
        if os.path.exists(depth_dir):
            print(f"Have access to monodepth from {depth_dir}")
            self.monodepth_dir = depth_dir
            self.monodepth_paths = [os.path.join(depth_dir, f"{Path(f).stem}.npy") for f in image_names] # List[str], (num_images,)

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width

        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = fx * x1 * r + width // 2
                mapy = fy * y1 * r + height // 2

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        load_monodepths: bool = False,
        use_dvl_data: bool = False,
        use_odom_csv: bool = False,
        use_apriltag_csv: bool = False
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        elif split == "all":
            self.indices = indices
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

        self.load_monodepths = load_monodepths
        self.use_dvl_data = use_dvl_data
        if load_monodepths:
            assert hasattr(self.parser, "monodepth_paths"), "Monodepth paths not found"
            assert len(self.parser.monodepth_paths) == len(self.parser.image_paths), "Monodepth paths not found"
            print(f"Will be utilizing monodepths like {self.parser.monodepth_paths[0]}")

            if use_dvl_data:
                dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
                self.dvl_dataset = DvlDataset(str(dir_path / "dvl_data.csv"))

        # get the timestamps of the images
        self.ros_ts_list = []
        for image_name in self.parser.image_names:
            ros_t_sec, ros_t_ns = image_name.split('.')[0].split('_')[-1].split('-')
            ros_ts = int(int(ros_t_sec) * 1e9 + int(ros_t_ns)) # nanoseconds
            self.ros_ts_list.append(ros_ts)
        self.ros_ts_list = np.array(self.ros_ts_list)

        self.use_odom_csv = use_odom_csv
        if use_odom_csv:
            dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
            csv_odom_fp = str(dir_path / "02212025_compare_trajectories.csv")
            T_world_gtsamCams_dict, odom_timestamps = read_csv_odom(csv_odom_fp)
            odom_timestamps = (odom_timestamps * 1e9).astype(np.int64) # nanoseconds
            koi = "optimized_odom_visodom_tag"
            T_gtsamWorld_odomCams = slerp_closets_odomcam(self.ros_ts_list, odom_timestamps, T_world_gtsamCams_dict[koi])
            self.T_world_odomCams = [K_T_COLMAPWORLD_GTSAMWORLD @ T_gtsamWorld_odomCam for T_gtsamWorld_odomCam in T_gtsamWorld_odomCams]

        self.use_apriltag_csv = use_apriltag_csv
        self.apriltag_seen_idxs = []
        if self.use_apriltag_csv:
            dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
            apriltag_csv_fp = str(dir_path / "04102025_tag_poses.csv")
            T_cam_apriltags, apriltag_timestamps_s = read_csv_apriltag(apriltag_csv_fp)
            apriltag_timestamp_ns = (apriltag_timestamps_s * 1e9).astype(np.int64) # nanoseconds
            for idx, ros_ts_ns in enumerate(self.ros_ts_list):
                if ros_ts_ns in apriltag_timestamp_ns:
                    self.apriltag_seen_idxs.append(idx)

        self.updated_poses = {}

    def __len__(self):
        return len(self.indices)

    def update_T_world_cam(self, idx: int, T_world_cam: np.ndarray):
        """Update the T_world_cam for a specific index."""
        self.updated_poses[idx] = T_world_cam # 4x4 array

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        assert index == item # I think is the same always lol
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index] # 4x4
        mask = self.parser.mask_dict[camera_id]

        # if we're using odom cams, overwrite T_world_cam
        if self.use_odom_csv:
            camtoworlds = self.T_world_odomCams[item].numpy()

        # if an updated pose has been provided for this camera, use that
        if item in self.updated_poses:
            camtoworlds = self.updated_poses[item]

        # extract ros timestamp of image
        ros_ts = self.ros_ts_list[item]

        if len(params) > 0:
            assert False # "Distorted images are not supported yet."
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_fp": self.parser.image_paths[index],
            "image_id": torch.from_numpy(np.array([item])).int(),  # the index of the image in the dataset
            "ros_ts": ros_ts # nanoseconds as an int
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_monodepths:
            subsample = 100

            # unproject monodepth into pointcloud
            monodepth = np.load(self.parser.monodepth_paths[index])
            pcd_wrt_cam = depth_image_to_pcd(monodepth[:, :, None], K) # 3 x N

            if self.use_dvl_data:
                avg_range, stddev_range = self.dvl_dataset.get_range_at_timestamp(ros_ts)
                scale = avg_range / np.mean(pcd_wrt_cam)
                pcd_wrt_cam *= scale

            pcd_wrt_world = (camtoworlds[:3, :3] @ pcd_wrt_cam + camtoworlds[:3, 3:4]).T # N x 3
            pcd_rgb = image.reshape(-1, 3) # N x 3
            data["md_xyz_wrt_world"] = torch.from_numpy(pcd_wrt_world).float()[::subsample]
            data["md_rgb"] = torch.from_numpy(pcd_rgb).float()[::subsample]

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_errs = self.parser.points_err[point_indices] # (M,)
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            point_indices = point_indices[selector]
            data["points"] = torch.from_numpy(points).float() # projected onto image plane (?)
            data["points_errs"] = torch.from_numpy(points_errs).float()
            data["depths"] = torch.from_numpy(depths).float()
            data["points_rgb"] = torch.from_numpy(self.parser.points_rgb[point_indices]).float()
            data["points_xyz"] = torch.from_numpy(self.parser.points[point_indices]).float() # wrt world
            data["point_indices"] = point_indices.tolist()

        return data

class SpatialDataset:
    """A dataset wrapper class that returns a spatially sampled image."""

    def __init__(
        self,
        dataset: Dataset,
        dbg_plot: bool = False
    ):
        self.dataset = dataset

        if dbg_plot:
            import plotly.express as px

        xy_worlds = []
        # get xy bounds of the dataset
        for idx in self.dataset.indices:
            c2w = self.dataset.parser.camtoworlds[idx]
            xy_world = c2w[:2, 3]
            xy_worlds.append(xy_world)
        self.xy_worlds = np.array(xy_worlds)
        idxs = [i for i in range(len(self.xy_worlds))]

        if dbg_plot:
            # plot random points
            plot_xys = []
            for _ in range(2000):
                res = np.random.choice(idxs)
                plot_xys.append(self.xy_worlds[res])
            plot_xys = np.array(plot_xys)

            fig = px.scatter(x=plot_xys[:, 0], y=plot_xys[:, 1])
            fig.write_image("z_random.png")

        # fit a KDE
        kde = gaussian_kde(self.xy_worlds.T)

        # reweight the points
        weights = 1.0 / kde(self.xy_worlds.T)
        normalized_weights = weights / np.sum(weights)

        if dbg_plot:
            # plot reweighted points
            plot_xys = []
            for _ in range(2000):
                res = np.random.choice(idxs, p=normalized_weights)
                plot_xys.append(self.xy_worlds[res])
            plot_xys = np.array(plot_xys)

            fig = px.scatter(x=plot_xys[:, 0], y=plot_xys[:, 1])
            fig.write_image("z_resampled.png")

        self.idxs = idxs
        self.normalized_weights = normalized_weights

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        idx = np.random.choice(self.idxs, p=self.normalized_weights)
        return self.dataset.__getitem__(idx)

    def get_weights_for_subset(self, subset: List[int]) -> np.ndarray:
        kde = gaussian_kde(self.xy_worlds[subset].T)
        weights = 1.0 / kde(self.xy_worlds[subset].T)
        normalized_weights = weights / np.sum(weights)
        return normalized_weights

    def get_weights_for_full(self):
        return self.normalized_weights


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    argparser.add_argument("--factor", type=int, default=2)
    argparser.add_argument("--center_crop", action='store_true')
    argparser.add_argument("--test_every", type=int, default=1e6) # don't have any images in the test
    args = argparser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir,
        factor=args.factor,
        normalize=False,
        test_every=args.test_every,
        center_crop=args.center_crop,
        monodepth_key="None"
    )
    dataset = Dataset(
        parser,
        split="all",
        load_depths=False,
        load_monodepths=False,
        use_odom_csv=True,
        use_apriltag_csv=True
    )
    print(f"Dataset: {len(dataset)} images.")
    test = dataset[0]
    import pdb; pdb.set_trace()
    for idx, data in enumerate(tqdm.tqdm(dataset)):
        if args.factor == 2:
            try:
                assert data["image"].shape == (560, 994, 3)
            except:
                print(data["image"].shape)
                print(parser.image_paths[idx])
        else:
            assert data["image"].shape == (560 * 2, 994 * 2, 3)

    # writer = imageio.get_writer("results/points.mp4", fps=30)
    # for data in tqdm.tqdm(dataset, desc="Plotting points"):
    #     image = data["image"].numpy().astype(np.uint8)
    #     points = data["points"].numpy()
    #     depths = data["depths"].numpy()
    #     for x, y in points:
    #         cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
    #     writer.append_data(image)
    # writer.close()
