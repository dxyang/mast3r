import csv
import os
import glob
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import torch
from tqdm import tqdm
import open3d as o3d
import open3d.core as o3c

from utils.plotly_viz_utils import PlotlyScene, plot_transform, plot_points, plot_points_sequence
from utils.csv_odom import find_closest_timestamps, slerp_closets_odomcam, read_csv_odom, read_csv_apriltag
from tap import Tap

from utils.pointmaps import transform_pointmap

'''
python plot_odom_with_pointmaps.py \
--csv 02212025_compare_trajectories.csv \
--do_midcrop \
--o3d_ss 10 \
--use_slerp_poses
--num_images 1000


python plot_odom_with_pointmaps.py \
--csv 02212025_compare_trajectories.csv \
--do_midcrop \
--o3d_ss 10 \
--use_slerp_poses
--num_images 1000

'''

class ArgParser(Tap):
    csv: str
    koi: str = "optimized_odom_visodom_tag"

    apriltag_csv: str = "yawzi_apriltag_poses"

    plot_plotly: bool = False

    do_midcrop: bool = False
    o3d_ss: int = 10

    num_images: int = -1

    use_slerp_poses: bool = False

    use_dust3r_poses: bool = False

if __name__ == '__main__':
    device = 'cuda'
    args = ArgParser().parse_args()

    xmin, xmax = -20, 20
    ymin, ymax = -20, 20
    zmin, zmax = -15, 15

    dust3r_scene = PlotlyScene(
        size=(800, 800), x_range=(xmin, xmax), y_range=(ymin, ymax), z_range=(zmin, zmax)
    )
    scaled_dust3r_scene = PlotlyScene(
        size=(800, 800), x_range=(xmin, xmax), y_range=(ymin, ymax), z_range=(zmin, zmax)
    )

    # parse csv
    T_world_gtsamCams_dict, odom_timestamps = read_csv_odom(args.csv)
    koi = args.koi

    # parse apriltag csv
    T_cam_apriltags, apriltag_timestamps = read_csv_apriltag(args.apriltag_csv)

    # get data from dust3r
    if args.do_midcrop:
        exp_dir = "experiments/02202025/yawzi_dust3r_dynamicconf_midcrop_ss3_window2_saveimgpts"
        img_dir = "/srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/raw_downward"
        cropstr = 'midcrop'
    else:
        exp_dir = "experiments/02202025/yawzi_dust3r_dynamicconf_nocrop_ss3_window2_saveimgpts"
        img_dir = "/srv/warplab/tektite/jobs/Yawzi_2024_11_14/rosbag_extracted/RAW"
        cropstr = 'nocrop'
    # make sure we're going to assign image name timestamps properly
    assert 'yawzi' in exp_dir
    assert 'ss3' in exp_dir
    assert 'window2' in exp_dir
    saved_imgs = np.load(f"{exp_dir}/imgs.npy")
    saved_pointmaps = np.load(f"{exp_dir}/pointmaps.npy")
    saved_intrinsics = np.load(f"{exp_dir}/intrinsics.npy")
    T_dust3rWorld_cams = torch.load(f"{exp_dir}/poses.pt")
    sorted_image_list = sorted([Path(fp).name for fp in glob.glob(f"{img_dir}/*.png")])
    sorted_image_list = sorted_image_list[::3]
    assert len(sorted_image_list) == len(saved_imgs)

    # we have timestamps for each image, but the odometry poses have different timestamps
    dust3r_image_timestamps = []
    for img_name in tqdm(sorted_image_list):
        ts_s, ts_ns = float(img_name.split('.')[0].split('-')[0]), float(img_name.split('.')[0].split('-')[1])
        aggregate_ts = ts_s + ts_ns / 1e9
        dust3r_image_timestamps.append(aggregate_ts)

    T_world_odomCams = []
    if not args.use_slerp_poses:
        # approach 1: for each dust3r image, find the odometry pose with the closest timestamp
        T_world_odomCams = find_closest_timestamps(dust3r_image_timestamps, odom_timestamps, T_world_gtsamCams_dict[koi])
    else:
        # approach 2: for each dust3r image, slerp between the odometry poses with the closest timestamps
        T_world_odomCams = slerp_closets_odomcam(dust3r_image_timestamps, odom_timestamps, T_world_gtsamCams_dict[koi])

    # plot things
    full_pointmaps_wrt_world = []
    o3d_colors = []
    num_images = len(dust3r_image_timestamps) if args.num_images == -1 else args.num_images

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for idx in tqdm(range(num_images)):
        viz_img = saved_imgs[idx]
        flattened = viz_img.reshape(-1, 3)
        img_color = [f"rgb({r},{g},{b})" for r,g,b in flattened]

        # need to determine scaling for pointmap
        scaling = 1.0
        do_scaling = True
        if do_scaling:
            if idx == 0:
                # get translation of dust3r cams
                T_lastDust3rCam_currDust3rCam = torch.matmul(torch.linalg.inv(T_dust3rWorld_cams[0]), T_dust3rWorld_cams[1])

                # get translation of odometry cams
                T_lastOdomCam_currOdomCam = torch.matmul(torch.linalg.inv(T_world_odomCams[0]), T_world_odomCams[1])
            else:
                T_lastDust3rCam_currDust3rCam = torch.matmul(torch.linalg.inv(T_dust3rWorld_cams[idx-1]), T_dust3rWorld_cams[idx])
                T_lastOdomCam_currOdomCam = torch.matmul(torch.linalg.inv(T_world_odomCams[idx-1]), T_world_odomCams[idx])

            dust3r_translation = torch.linalg.norm(T_lastDust3rCam_currDust3rCam[:3, 3])
            odom_translation = torch.linalg.norm(T_lastOdomCam_currOdomCam[:3, 3])
            scaling =  odom_translation / dust3r_translation
            # print(dust3r_translation, odom_translation, scaling)

        pointmap_wrt_dust3rWorld = torch.from_numpy(saved_pointmaps[idx])
        T_dust3rWorld_cam = T_dust3rWorld_cams[idx]
        T_cam_dust3rWorld = torch.linalg.inv(T_dust3rWorld_cam)

        pointmap_wrt_cam = transform_pointmap(pointmap_wrt_dust3rWorld, T_cam_dust3rWorld)
        pointmap_wrt_cam *= scaling
        pointmap_wrt_world = transform_pointmap(pointmap_wrt_cam, T_world_odomCams[idx])

        # generate open3d assets
        full_pointmaps_wrt_world.append(pointmap_wrt_world.reshape(-1, 3)[::args.o3d_ss].cpu().numpy()) # N x 3
        o3d_colors.append(flattened[::args.o3d_ss])

        depth_img = np.ascontiguousarray(pointmap_wrt_cam[:, :, 2].cpu().numpy())
        rgb_img = saved_imgs[idx]

        o3d_d = o3d.geometry.Image(depth_img)
        o3d_rgb = o3d.geometry.Image(rgb_img)
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(512, 384, saved_intrinsics[idx][0, 0], saved_intrinsics[idx][1, 1], saved_intrinsics[idx][0, 2], saved_intrinsics[idx][1, 2])
        o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_rgb, o3d_d, depth_scale=1.0, depth_trunc=100.0, convert_rgb_to_intensity=False)
        volume.integrate(
            o3d_rgbd,
            o3d_intrinsics,
            torch.linalg.inv(T_world_odomCams[idx]).cpu().numpy())

        # plotting
        if args.plot_plotly:
            plot_transform(dust3r_scene.figure, T_world_odomCams[idx], label=f"image_{idx}", linelength=0.5, linewidth=10)
            plot_points(dust3r_scene.figure, pointmap_wrt_world.reshape(-1, 3)[::args.o3d_ss].T, size=1, name=f'pointmap_{idx}', color=img_color[::args.o3d_ss])

    if args.plot_plotly:
        dust3r_scene.plot_scene_to_html(f"new_odom_poses")
        print(f"Saved to odom_poses.html")


    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(f"ply_output/{cropstr}/mesh_scene_{num_images}_{'slerp' if args.use_slerp_poses else 'noslerp'}_ss{args.o3d_ss}.ply", mesh, print_progress=True)
    print(f"Saved mesh_scene_{num_images}_{'slerp' if args.use_slerp_poses else 'noslerp'}_ss{args.o3d_ss}.ply")

    full_pointmaps_wrt_world = np.concatenate(full_pointmaps_wrt_world, axis=0)
    o3d_colors = np.clip(np.concatenate(o3d_colors, axis=0).astype(np.float32) / 255.0, 0.0, 1.0)

    pcd = o3d.t.geometry.PointCloud(full_pointmaps_wrt_world)
    pcd.point.colors = o3d_colors
    print(pcd, "\n")
    print(full_pointmaps_wrt_world.shape)
    print(o3d_colors.shape)
    o3d.io.write_point_cloud(f"ply_output/{cropstr}/pointmap_scene_{num_images}_{'slerp' if args.use_slerp_poses else 'noslerp'}_ss{args.o3d_ss}.ply", pcd.to_legacy(), print_progress=True)
    print(f"Saved pointmap_scene_{num_images}_{'slerp' if args.use_slerp_poses else 'noslerp'}_ss{args.o3d_ss}.ply")

    ''''
    let's try tripling the translation between poses and seeing how the pointmaps change
    '''
    # ss_pts3d = 100
    # T_world_lastCam = torch.eye(4)
    # T_world_lastScaledCam = torch.eye(4)
    # for idx in tqdm(range(50)):
    #     viz_img = saved_imgs[idx]
    #     flattened = viz_img.reshape(-1, 3)
    #     img_color = [f"rgb({r},{g},{b})" for r,g,b in flattened]

    #     scaling = 3.0

    #     # need to calculate the new position of this camera
    #     T_world_currCam = T_dust3rWorld_cams[idx]
    #     T_lastCam_currCam = torch.matmul(torch.linalg.inv(T_world_lastCam), T_world_currCam)
    #     T_lastCam_newCurrCam = T_lastCam_currCam
    #     T_lastCam_newCurrCam[:3, 3] *= scaling
    #     T_world_newCurrCam = torch.matmul(T_world_lastScaledCam, T_lastCam_newCurrCam)

    #     # bookeeping
    #     T_world_lastCam = T_world_currCam
    #     T_world_lastScaledCam = T_world_newCurrCam

    #     # pointmap
    #     pointmap_wrt_dust3rWorld = torch.from_numpy(saved_pointmaps[idx])
    #     T_dust3rWorld_cam = T_dust3rWorld_cams[idx]
    #     T_cam_dust3rWorld = torch.linalg.inv(T_dust3rWorld_cam)

    #     pointmap_wrt_cam = transform_pointmap(pointmap_wrt_dust3rWorld, T_cam_dust3rWorld)
    #     scaled_pointmap_wrt_cam = pointmap_wrt_cam * scaling
    #     scaled_pointmap_wrt_world = transform_pointmap(scaled_pointmap_wrt_cam, T_world_newCurrCam)
    #     pointmap_wrt_world = transform_pointmap(pointmap_wrt_cam, T_world_currCam)

    #     # original trajectory
    #     plot_transform(dust3r_scene.figure, T_world_currCam, label=f"og_{idx}", linelength=0.5, linewidth=10)
    #     plot_points(dust3r_scene.figure, pointmap_wrt_world.reshape(-1, 3)[::ss_pts3d].T, size=1, name=f'og_pts_{idx}', color=img_color[::ss_pts3d])

    #     # scaled trajectory and pointmaps
    #     plot_transform(scaled_dust3r_scene.figure, T_world_newCurrCam, label=f"scaled_{idx}", linelength=0.5, linewidth=10)
    #     plot_points(scaled_dust3r_scene.figure, scaled_pointmap_wrt_world.reshape(-1, 3)[::ss_pts3d].T, size=1, name=f'scaled_pts_{idx}', color=img_color[::ss_pts3d])

    # dust3r_scene.plot_scene_to_html(f"odom_poses")
    # scaled_dust3r_scene.plot_scene_to_html(f"odom_poses_scaled")
    # print(f"Saved to odom_poses.html")


