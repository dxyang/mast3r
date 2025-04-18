import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import gaussian_kde
import torch

from datasets.colmap import Parser

from utils.csv_odom import read_csv_odom, slerp_closets_odomcam, K_T_COLMAPWORLD_GTSAMWORLD

class CsvCenterDataset():
    """A dataset sampler that provides dataset indices ordered by the distance from the center of the scene"""

    def __init__(
        self,
        parser: Parser,
        csv_fp: str = None,
        drop_last_150_frames: bool = True
    ):
        if csv_fp is None:
            dataset_file_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
            self.csv_fp = str(dataset_file_dir_path / "02212025_compare_trajectories.csv")
        else:
            self.csv_fp = csv_fp

        # get the timestamps of the images
        self.parser = parser
        self.ros_ts_list = []
        for image_name in self.parser.image_names:
            ros_t_sec, ros_t_ns = image_name.split('.')[0].split('_')[-1].split('-')
            ros_ts = int(int(ros_t_sec) * 1e9 + int(ros_t_ns)) # nanoseconds
            self.ros_ts_list.append(ros_ts)
        self.ros_ts_list = np.array(self.ros_ts_list)
        if drop_last_150_frames:
            self.ros_ts_list = self.ros_ts_list[:-150]

        # Parse CSV odom data
        T_world_gtsamCams_dict, odom_timestamps = read_csv_odom(self.csv_fp)
        koi = "optimized_odom_visodom_tag"
        self.odom_timestamps = (odom_timestamps * 1e9).astype(np.int64) # nanoseconds
        self.T_gtsamWorld_odomCams = slerp_closets_odomcam(self.ros_ts_list, self.odom_timestamps, T_world_gtsamCams_dict[koi])

        self.num_images = len(self.T_gtsamWorld_odomCams)

        # extract camera xyzs
        cam_xyzs = []
        for T_world_cam in self.T_gtsamWorld_odomCams:
            cam_xyz = T_world_cam[:3, 3]
            cam_xyzs.append(cam_xyz)
        self.cam_xyzs = torch.stack(cam_xyzs) # N, 3
        self.cam_xys = self.cam_xyzs[:, :2] # N, 2

        # get center
        self.center = self.cam_xyzs.mean(dim=0)

        # get distances
        self.distances = torch.norm(self.cam_xyzs - self.center, dim=1)
        self.min_distance = self.distances.min()
        self.max_distance = self.distances.max()

        # sort distances
        self.idxs = torch.argsort(self.distances)

        # kde for spatial sampler
        kde = gaussian_kde(self.cam_xys.T)
        weights = 1.0 / kde(self.cam_xys.T)
        self.normalized_weights = weights / np.sum(weights)

    def __len__(self):
        return self.num_images

    def get_weights_for_subset(self, subset: List[int]) -> np.ndarray:
        kde = gaussian_kde(self.cam_xys[subset].T)
        weights = 1.0 / kde(self.cam_xys[subset].T)
        normalized_weights = weights / np.sum(weights)
        return normalized_weights

    def get_weights_for_full(self):
        return self.normalized_weights

    def get_indices_within_dist(self, max_dist: float):
        """Get indices of images within a certain distance from the center"""
        return torch.where(self.distances < max_dist)[0].tolist()

    def get_indices_within_range(self, min_dist: float, max_dist: float):
        """Get indices of images within a certain distance from the center"""
        return torch.where(torch.logical_and(self.distances <= max_dist, self.distances >= min_dist))[0].tolist()

if __name__ == "__main__":
    import os
    from pathlib import Path

    data_dir = "/srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/metashape_export/"
    factor = 2
    center_crop = False
    test_every = 1e10

    parser = Parser(
        data_dir=data_dir,
        factor=factor,
        normalize=False,
        test_every=test_every,
        center_crop=center_crop,
        monodepth_key="None"
    )


    dataset_file_dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parent / "datasets"
    csv_odom_fp = str(dataset_file_dir_path / "02212025_compare_trajectories.csv")
    apriltag_csv_fp = str(dataset_file_dir_path / "04102025_tag_poses.csv")

    center_sampler = CsvCenterDataset(parser=parser, csv_fp=csv_odom_fp)

    buckets = {}
    last_dist = 0.0
    for dist_idx, dist in enumerate(np.arange(1.0, 13.0, 1.0)):
        idxs = center_sampler.get_indices_within_range(last_dist, dist)
        weights = center_sampler.get_weights_for_subset(idxs)

        last_dist = dist

        plot_xyzs = []
        for _ in range(1000):
            res = np.random.choice(idxs, p=weights)
            plot_xyzs.append(center_sampler.cam_xyzs[res])
        plot_xyzs = np.array(plot_xyzs)

        buckets[dist_idx] = plot_xyzs

    import pandas as pd
    import plotly.express as px

    all_points = []
    all_colors = []

    for dist_idx, plot_xyzs in buckets.items():
        for point in plot_xyzs:
            all_points.append(point)
            all_colors.append(dist_idx)  # Use dist_idx as the color category

    # Convert to a DataFrame for easier plotting with Plotly Express
    df = pd.DataFrame(all_points, columns=["x", "y", "z"])
    df["distance_threshold"] = all_colors

    # Plot using Plotly Express
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="distance_threshold",
        title="Point Clouds Colored by Distance Threshold",
        labels={"distance_threshold": "Distance Threshold"}
    )

    fig.write_html('plotly_3dscatter.html')
