import csv
import os
import glob
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch



from utils.plotly_viz_utils import PlotlyScene, plot_transform, plot_points, plot_points_sequence

from tap import Tap

class ArgParser(Tap):
    exp_name: str
    subsample: int = 1
    plot_transforms: bool = False

    #
    seven_scenes_dir: str = os.path.expanduser("~/localdata/7scenes")

if __name__ == '__main__':
    device = 'cuda'
    args = ArgParser().parse_args()

    exp_dir = f"experiments/{args.exp_name}"

    pose_fp = f"{exp_dir}/poses.pt"
    T_world_cams = torch.load(pose_fp)

    if 'mast3r' in args.exp_name:
        xmin, xmax = -60, 60
        ymin, ymax = -60, 60
        zmin, zmax = -60, 60
    elif '7scenes' in args.exp_name or 'test' in args.exp_name:
        xmin, xmax = -2, 2
        ymin, ymax = -2, 2
        zmin, zmax = -2, 2
    else:
        xmin, xmax = -10, 10
        ymin, ymax = -10, 10
        zmin, zmax = -5, 15

    tf_scene = PlotlyScene(
        size=(800, 800), x_range=(xmin, xmax), y_range=(ymin, ymax), z_range=(zmin, zmax)
    )
    pcd_scene = PlotlyScene(
        size=(800, 800), x_range=(xmin, xmax), y_range=(ymin, ymax), z_range=(zmin, zmax)
    )
    gt_pcd_scene = PlotlyScene(
        size=(800, 800), x_range=(xmin, xmax), y_range=(ymin, ymax), z_range=(zmin, zmax)
    )

    if "7scenes" in args.exp_name:
        # try to read in the ground truth data and plot over top
        if "chess" in args.exp_name:
            dset_subset = "chess"
        elif "stairs" in args.exp_name:
            dset_subset = "stairs"
        elif "office" in args.exp_name:
            dset_subset = "office"
        elif "heads" in args.exp_name:
            dset_subset = "heads"
        else:
            assert False

        # python process_pairwise.py --exp_name 7scenes_stairs_seq01 --use_7scenes --image_dir /home/dayang/localdata/7scenes/stairs/seq-01
        # python process_pairwise.py --exp_name 7scenes_office_seq01 --use_7scenes --image_dir /home/dayang/localdata/7scenes/office/seq-01
        # python process_pairwise.py --exp_name 7scenes_heads_seq01 --use_7scenes --image_dir /home/dayang/localdata/7scenes/heads/seq-01

        if "seq01" in args.exp_name:
            dset_subset += "/seq-01"
        else:
            assert False

        pose_fp_list = sorted([Path(fp) for fp in glob.glob(f"{args.seven_scenes_dir}/{dset_subset}/*pose.txt")])
        cam_7scenes_pcd = []
        for pose_fp in pose_fp_list:
            T_world_cam = np.loadtxt(pose_fp)
            cam_7scenes_pcd.append(T_world_cam[:3, 3])
            plot_transform(tf_scene.figure, T_world_cam, label=pose_fp.name, linelength=0.1, linewidth=5)
        cam_7scenes_pcd = np.stack(cam_7scenes_pcd, axis=1)
        plot_points_sequence(gt_pcd_scene.figure, cam_7scenes_pcd, size=10, name='7scenes_poses')

    pcd_xyz = []

    for idx, T_world_cam in enumerate(T_world_cams[::args.subsample]):
        if args.plot_transforms:
            plot_transform(tf_scene.figure, T_world_cam.cpu().numpy(), label=f"image_{idx}", linelength=0.1, linewidth=5)

        pcd_xyz.append(T_world_cam[:3, 3])

    pcd_xyz = torch.stack(pcd_xyz, dim=1).cpu().numpy()

    plot_points_sequence(pcd_scene.figure, pcd_xyz, size=10, name='camera_poses')

    if args.plot_transforms:
        tf_scene.plot_scene_to_html(f"{exp_dir}/T_world_cam")
    pcd_scene.plot_scene_to_html(f"{exp_dir}/cam_points")

    if "7scenes" in args.exp_name:
        gt_pcd_scene.plot_scene_to_html(f"{exp_dir}/gt_poses")


    # export poses to csv format
    # timestamp x y z qx qy qz qw
    # HACK: assume what the original image directories are and copypasta logic from process_pairwise.py
    assert 'yawzi' in args.exp_name
    if 'midcrop' in args.exp_name:
        image_dir = "/srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/raw_downward"
    elif 'nocrop' in args.exp_name:
        image_dir = "/srv/warplab/tektite/jobs/Yawzi_2024_11_14/rosbag_extracted/RAW"
    else:
        raise ValueError("invalid exp_name")

    sorted_image_list = sorted([Path(fp).name for fp in glob.glob(f"{image_dir}/*.png")])
    assert 'ss3' in args.exp_name
    assert 'window2' in args.exp_name
    sorted_image_list = sorted_image_list[::3]
    assert len(sorted_image_list) == len(T_world_cams)

    # convert each T_world_cam into a row of the csv
    csv_rows = []
    for idx, (T_world_cam, image_fp) in enumerate(zip(T_world_cams, sorted_image_list)):
        T_world_cam = T_world_cam.cpu().numpy()
        timestamp = image_fp.split('.')[0]
        ts_s, ts_ns = float(timestamp.split('-')[0]), float(timestamp.split('-')[1]) / 1e9
        combined_ts = ts_s + ts_ns
        combined_ts_string = "{:.9f}".format(combined_ts)

        tx, ty, tz = T_world_cam[:3, 3]
        rotation = R.from_matrix(T_world_cam[:3, :3])
        qx, qy, qz, qw = rotation.as_quat()

        csv_rows.append([combined_ts_string, tx, ty, tz, qx, qy, qz, qw])

    csv_fp = f"{exp_dir}/poses.csv"
    with open(csv_fp, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
