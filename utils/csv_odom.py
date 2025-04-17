import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch

'''
CSV has body in the world frame
this is the transformation from cam to robot body (i.e. the camera in body frame)
'''
K_T_ROBOT_CAM = torch.eye(4)
K_T_ROBOT_CAM[:3, :3] = torch.from_numpy(R.from_quat([0.707, -0.707, 0.000, 0.000]).as_matrix()) # x, y, z, w
K_T_ROBOT_CAM[:3, 3] = torch.Tensor([0.113, 0.060, -0.192])


'''
GTSAM world to COLMAP world (as outputted by evo)
'''
K_T_COLMAPWORLD_GTSAMWORLD = torch.eye(4)
K_T_COLMAPWORLD_GTSAMWORLD[:3, :3] = 4.2241796838262795 * torch.from_numpy(np.array([
    [ 0.99770068,  0.06698315,  0.01032523],
    [-0.05258566,  0.86118415, -0.50556563],
    [-0.04275631,  0.50386021,  0.86272637],
]))
K_T_COLMAPWORLD_GTSAMWORLD[:3, 3] = torch.from_numpy(np.array(
    [-0.13231239, -17.37355454,  32.39187592],
))

'''
COLMAP world to GTSAM world (as outputted by evo)
'''
K_T_GTSAMWORLD_COLMAPWORLD = torch.eye(4)
K_T_GTSAMWORLD_COLMAPWORLD[:3, :3] = 0.2362260583547413 * torch.from_numpy(np.array([
    [ 0.99770068, -0.05258566, -0.04275631],
    [ 0.06698315,  0.86118415,  0.50386021],
    [ 0.01032523, -0.50556563,  0.86272637],
]))
K_T_GTSAMWORLD_COLMAPWORLD[:3, 3] = torch.from_numpy(np.array(
    [0.14217289, -0.31996015, -8.69227534],
))

def get_apriltag_pose(t_s, apriltag_timestamps_s, T_cam_apriltags, max_delta_t_s=0.05):
    timestamp_idx = 0
    apriltag_timestamp = apriltag_timestamps_s[timestamp_idx]

    while t_s > apriltag_timestamp and timestamp_idx < len(apriltag_timestamps_s) - 1:
        timestamp_idx += 1
        apriltag_timestamp = apriltag_timestamps_s[timestamp_idx]

    if timestamp_idx == 0:
        # we are pretty far behind the first apriltag detection
        return False, torch.eye(4)
    elif timestamp_idx == len(apriltag_timestamps_s) - 1:
        # we are pretty far ahead of the last odometry timestamp
        return False, torch.eye(4)
    else:
        # calculate slerp alpha
        t0 = apriltag_timestamps_s[timestamp_idx - 1]
        t1 = apriltag_timestamps_s[timestamp_idx]

        # if we aren't completely inbetween detections, then don't interpolate
        if abs(t - t0) > max_delta_t_s or abs(t - t1) > max_delta_t_s:
            return False, torch.eye(4)

        alpha = (t_s - t0) / (t1 - t0)

        # apply slerp
        T_cam_apriltag0 = T_cam_apriltags[timestamp_idx - 1]
        T_cam_apriltag1 = T_cam_apriltags[timestamp_idx]

        T_cam_apriltag = slerp_poses(T_cam_apriltag0, T_cam_apriltag1, alpha)

        return True, T_cam_apriltag


def get_closest_apriltag_pose(t_s, apriltag_timestamps_s, T_cam_apriltags, max_delta_t_s=0.05):
    closest_idx = np.argmin(np.abs(apriltag_timestamps_s - t_s))
    delta = np.abs(apriltag_timestamps_s[closest_idx] - t_s)
    if delta < max_delta_t_s:
        return True, T_cam_apriltags[closest_idx]
    else:
        return False, torch.eye(4)

def read_csv_apriltag(csv_fp):
    T_cam_apriltags = []
    timestamps = []

    with open(csv_fp, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    # ignore header
                    continue
                # timestamp, x, y, z, qx, qy, qz, qw, mean_error = row
                timestamp, x, y, z, qx, qy, qz, qw = row

                translation = np.array([float(x), float(y), float(z)])
                rotation = R.from_quat([float(qx), float(qy), float(qz), float(qw)])
                T_cam_apriltag = np.eye(4)
                T_cam_apriltag[:3, :3] = rotation.as_matrix()
                T_cam_apriltag[:3, 3] = translation

                T_cam_apriltag = torch.from_numpy(T_cam_apriltag).float()

                T_cam_apriltags.append(T_cam_apriltag)
                timestamps.append(float(timestamp))

    timestamps_s = np.array(timestamps)
    T_cam_apriltags = torch.stack(T_cam_apriltags)

    return T_cam_apriltags, timestamps_s

def read_csv_odom(csv_fp):
    T_world_gtsamCams_dict = {}
    timestamps_dict = {}

    with open(csv_fp, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row_idx, row in enumerate(reader):
                if row_idx == 0:
                    # ignore header
                    continue
                timestamp, pose_type, x, y, z, qx, qy, qz, qw = row
                if "landmark" in pose_type:
                    continue

                translation = np.array([float(x), float(y), float(z)])
                rotation = R.from_quat([float(qx), float(qy), float(qz), float(qw)])
                T_world_robot = np.eye(4)
                T_world_robot[:3, :3] = rotation.as_matrix()
                T_world_robot[:3, 3] = translation

                T_world_cam = torch.from_numpy(np.matmul(T_world_robot, K_T_ROBOT_CAM.numpy())).float()

                if pose_type not in T_world_gtsamCams_dict:
                    T_world_gtsamCams_dict[pose_type] = []
                    timestamps_dict[pose_type] = []
                T_world_gtsamCams_dict[pose_type].append(T_world_cam)
                timestamps_dict[pose_type].append(float(timestamp))

    for k, v in timestamps_dict.items():
        timestamps_dict[k] = np.array(v)
    for k, v in T_world_gtsamCams_dict.items():
        T_world_gtsamCams_dict[k] = torch.stack(v)

    assert (np.allclose(timestamps_dict['odom'], timestamps_dict['optimized_odom_tag'], rtol=1e-8, atol=1e-8) and \
            np.allclose(timestamps_dict['odom'], timestamps_dict['optimized_odom_visodom_tag'], rtol=1e-8, atol=1e-8) and \
            np.allclose(timestamps_dict['optimized_odom_visodom_tag'], timestamps_dict['optimized_odom_tag'], rtol=1e-8, atol=1e-8))
    timestamps = timestamps_dict['odom']

    return T_world_gtsamCams_dict, timestamps

def find_closest_timestamps(timestamps, odom_timestamps, T_world_odomCams):
    closest_T_world_odomCams = []
    closest_idxs = []
    for dust3r_ts in timestamps:
        closest_idx = np.argmin(np.abs(odom_timestamps - dust3r_ts))
        if closest_idx in closest_idxs:
            print(f"Warning: closest_idx {closest_idx} already found for dust3r_ts {dust3r_ts}")
        closest_T_world_odomCams.append(T_world_odomCams[closest_idx])
        closest_idxs.append(closest_idx)

    return closest_T_world_odomCams


def slerp_poses(T_world_cam1, T_world_cam2, alpha):
    t_slerp = T_world_cam1[:3, 3] * (1 - alpha) + T_world_cam2[:3, 3] * alpha
    R0 = R.from_matrix(T_world_cam1[:3, :3].cpu().numpy())
    R1 = R.from_matrix(T_world_cam2[:3, :3].cpu().numpy())
    slerp = Slerp([0, 1], R.concatenate([R0, R1]))
    R_slerp = slerp(alpha).as_matrix()
    T_world_slerpedCam = torch.eye(4)
    T_world_slerpedCam[:3, :3] = torch.from_numpy(R_slerp)
    T_world_slerpedCam[:3, 3] = t_slerp

    return T_world_slerpedCam

def slerp_closets_odomcam(timestamps, odom_timestamps, T_world_odomCams):
    slerped_T_world_odomCams = []

    for dust3r_ts in timestamps:
        odometry_timestamp_idx = 0
        odometry_timestamp = odom_timestamps[odometry_timestamp_idx]

        while dust3r_ts > odometry_timestamp and odometry_timestamp_idx < len(odom_timestamps) - 1:
            odometry_timestamp_idx += 1
            odometry_timestamp = odom_timestamps[odometry_timestamp_idx]

        if odometry_timestamp_idx == 0:
            # we are pretty far behind the first odometry timestamp
            T_world_odomCamDust3r = T_world_odomCams[0]
        elif odometry_timestamp_idx == len(odom_timestamps) - 1:
            # we are pretty far ahead of the last odometry timestamp
            T_world_odomCamDust3r = T_world_odomCams[-1]
        else:
            # calculate slerp alpha
            t0 = odom_timestamps[odometry_timestamp_idx - 1]
            t1 = odom_timestamps[odometry_timestamp_idx]
            alpha = (dust3r_ts - t0) / (t1 - t0)

            # apply slerp
            T_world_odomCam0 = T_world_odomCams[odometry_timestamp_idx - 1]
            T_world_odomCam1 = T_world_odomCams[odometry_timestamp_idx]

            T_world_odomCamDust3r = slerp_poses(T_world_odomCam0, T_world_odomCam1, alpha)

        slerped_T_world_odomCams.append(T_world_odomCamDust3r)

    return slerped_T_world_odomCams

if __name__ == "__main__":
    '''
    most of the code below sanity checks transforming between metashape and colmap
    '''
    import os
    from pathlib import Path

    from datasets.colmap import Parser, Dataset
    from utils.plotly_viz_utils import PlotlyScene, plot_points_sequence, plot_transform

    data_dir = "/srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/metashape_export/"
    factor = 2
    center_crop = True
    test_every = 1e10

    dataset_file_dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parent / "datasets"
    csv_odom_fp = str(dataset_file_dir_path / "02212025_compare_trajectories.csv")
    apriltag_csv_fp = str(dataset_file_dir_path / "04102025_tag_poses.csv")

    # Parse CSV odom data
    T_world_gtsamCams_dict, odom_timestamps = read_csv_odom(csv_odom_fp)
    odom_timestamps = (odom_timestamps * 1e9).astype(np.int64) # nanoseconds
    koi = "optimized_odom_visodom_tag"
    T_world_gtsamCams = T_world_gtsamCams_dict[koi]

    # Parse CSV apriltag data
    T_cam_apriltags, apriltag_timestamps_s = read_csv_apriltag(apriltag_csv_fp)

    # Parse COLMAP data.
    parser = Parser(
        data_dir=data_dir,
        factor=factor,
        normalize=False,
        test_every=test_every,
        center_crop=center_crop,
        monodepth_key="None"
    )
    T_world_colmapCams = [c2w for c2w in parser.camtoworlds]
    T_world_colmapCams = torch.from_numpy(np.stack(T_world_colmapCams)).float()

    # Align the two sets of poses
    gtsamPcd_gtsamWorld = T_world_gtsamCams[:, :3, 3]
    colmapPcd_colmapWorld = T_world_colmapCams[:, :3, 3]

    # Transform between spaces
    T_colmapWorld_gtsamCams = [K_T_COLMAPWORLD_GTSAMWORLD @ T_world_cam for T_world_cam in T_world_gtsamCams]
    T_colmapWorld_gtsamCams = torch.stack(T_colmapWorld_gtsamCams)
    gtsamPcd_colmapWorld = T_colmapWorld_gtsamCams[:, :3, 3]

    T_gtsamWorld_colmapCams = [K_T_GTSAMWORLD_COLMAPWORLD @ T_world_cam for T_world_cam in T_world_colmapCams]
    T_gtsamWorld_colmapCams = torch.stack(T_gtsamWorld_colmapCams)
    colmapPcd_gtsamWorld = T_gtsamWorld_colmapCams[:, :3, 3]

    # Plot and verify alignment
    xmin, xmax = -50, 50
    ymin, ymax = -50, 50
    zmin, zmax = -50, 50
    pcd_scene = PlotlyScene(
        size=(800, 800), x_range=(xmin, xmax), y_range=(ymin, ymax), z_range=(zmin, zmax)
    )
    plot_points_sequence(pcd_scene.figure, colmapPcd_colmapWorld.T, size=3, name='colmap')
    plot_points_sequence(pcd_scene.figure, gtsamPcd_gtsamWorld.T, size=3, name='gtsam')
    # plot_points_sequence(pcd_scene.figure, colmapPcd_gtsamWorld.T, size=3, name='colmap_gtsam')
    # plot_points_sequence(pcd_scene.figure, gtsamPcd_colmapWorld.T, size=3, name='gtsam_colmap')

    # subsample = 100
    # for idx, T_world_cam in enumerate(T_world_colmapCams[::subsample]):
    #     plot_transform(pcd_scene.figure, T_world_cam.cpu().numpy(), label=f"colmap_{idx}", linelength=0.5, linewidth=10)
    # for idx, T_world_cam in enumerate(T_world_gtsamCams[::subsample]):
    #     plot_transform(pcd_scene.figure, T_world_cam.cpu().numpy(), label=f"gtsam_{idx}", linelength=0.5, linewidth=10)
    # for idx, T_world_cam in enumerate(T_gtsamWorld_colmapCams[::subsample]):
    #     plot_transform(pcd_scene.figure, T_world_cam.cpu().numpy(), label=f"colmap_hat_{idx}", linelength=0.5, linewidth=10)
    # for idx, T_world_cam in enumerate(T_colmapWorld_gtsamCams[::subsample]):
    #     plot_transform(pcd_scene.figure, T_world_cam.cpu().numpy(), label=f"gtsam_hat_{idx}", linelength=0.5, linewidth=10)

    # For all the image timestamps, find when we have apriltag pose
    # estimates and collect that set of images to train a splat
    ros_ts_list = []
    for image_name in parser.image_names:
        ros_t_sec, ros_t_ns = image_name.split('.')[0].split('_')[-1].split('-')
        ros_ts = int(int(ros_t_sec) * 1e9 + int(ros_t_ns)) # nanoseconds
        ros_ts_list.append(ros_ts)
    ros_ts_list = np.array(ros_ts_list)

    # T_colmapWorld_tags = []
    # T_world_colmapTagCams = []
    # for idx, t_s in enumerate(ros_ts_list / 1e9):
    #     res, T_cam_tag = get_closest_apriltag_pose(t_s, apriltag_timestamps_s, T_cam_apriltags)
    #     T_world_colmapCam = T_world_colmapCams[idx]
    #     if res:
    #         T_colmapWorld_tag = T_world_colmapCam @ T_cam_tag
    #         T_colmapWorld_tags.append(T_colmapWorld_tag)
    #         T_world_colmapTagCams.append(T_world_colmapCam)

    T_gtsamWorld_tags = []
    T_world_gtsamTagCams = []
    for idx, t_s in enumerate(odom_timestamps / 1e9):
        res, T_cam_tag = get_closest_apriltag_pose(t_s, apriltag_timestamps_s, T_cam_apriltags)
        T_world_gtsamCam = T_world_gtsamCams[idx]
        if res:
            T_gtsamWorld_tag = T_world_gtsamCam @ T_cam_tag
            T_gtsamWorld_tags.append(T_gtsamWorld_tag)
            T_world_gtsamTagCams.append(T_world_gtsamCam)

    T_colmapWorld_tags = []
    for T_world_tag in T_gtsamWorld_tags:
        T_colmapWorld_tag = K_T_COLMAPWORLD_GTSAMWORLD @ T_world_tag
        T_colmapWorld_tags.append(T_colmapWorld_tag)

    # for every image, get what odometry thinks the pose should be
    T_gtsamWorld_odomCams = slerp_closets_odomcam(ros_ts_list, odom_timestamps, T_world_gtsamCams)
    T_colmapWorld_odomCams = []
    for T_world_cam in T_gtsamWorld_odomCams:
        T_colmapWorld_odomCam = K_T_COLMAPWORLD_GTSAMWORLD @ T_world_cam
        T_colmapWorld_odomCams.append(T_colmapWorld_odomCam)
    T_colmapWorld_odomCams = torch.stack(T_colmapWorld_odomCams)
    odomCamPcd_colmapWorld = T_colmapWorld_odomCams[:, :3, 3]

    # for every image, which ones can see apriltags? where does it think the apriltag is?
    T_colmapWorld_odomCamSeesTags = []
    T_colmapWorld_tagSeens = []
    for idx, t_s in enumerate(ros_ts_list / 1e9):
        res, T_cam_tag = get_closest_apriltag_pose(t_s, apriltag_timestamps_s, T_cam_apriltags)
        if res:
            # apriltag_t_s = apriltag_timestamps_s[np.argmin(np.abs(apriltag_timestamps_s - t_s))]
            # print(f"image: {t_s}, apriltag: {apriltag_t_s}, delta: {abs(apriltag_t_s - t_s)}")
            T_colmapWorld_odomCam = T_colmapWorld_odomCams[idx]
            T_colmapWorld_odomCamSeesTags.append(T_colmapWorld_odomCam)
            T_colmapWorld_tagSeen = T_colmapWorld_odomCam @ T_cam_tag
            T_colmapWorld_tagSeens.append(T_colmapWorld_tagSeen)


    # # Create a viz scene
    # xmin, xmax = -50, 50
    # ymin, ymax = -50, 50
    # zmin, zmax = -50, 50
    # apriltag_scene = PlotlyScene(
    #     size=(800, 800), x_range=(xmin, xmax), y_range=(ymin, ymax), z_range=(zmin, zmax)
    # )
    # for idx, T_world_tag in enumerate(T_gtsamWorld_tags[::10]):
    #     plot_transform(pcd_scene.figure, T_world_tag.cpu().numpy(), label=f"tag_{idx}", linelength=0.1, linewidth=10)
    # for idx, T_world_tag in enumerate(T_colmapWorld_tags[::10]):
    #     plot_transform(pcd_scene.figure, T_world_tag.cpu().numpy(), label=f"tag_{idx}", linelength=0.1, linewidth=10)

    plot_points_sequence(pcd_scene.figure, odomCamPcd_colmapWorld.T, size=3, name='odomCams')
    # for idx, T_world_cam in enumerate(T_colmapWorld_odomCams[::50]):
    #     plot_transform(pcd_scene.figure, T_world_cam.cpu().numpy(), label=f"odomCam{idx}", linelength=0.1, linewidth=10)
    # for idx, T_world_cam in enumerate(T_colmapWorld_odomCamSeesTags):
    #     plot_transform(pcd_scene.figure, T_world_cam.cpu().numpy(), label=f"camSeesTag_{idx}", linelength=0.1, linewidth=10)
    # for idx, T_world_tag in enumerate(T_colmapWorld_tagSeens):
    #     plot_transform(pcd_scene.figure, T_world_tag.cpu().numpy(), label=f"seenTag_{idx}", linelength=0.1, linewidth=10)

    pcd_scene.plot_scene_to_html(f"yawzi_gtsam_metashape_scene")

    import pdb; pdb.set_trace()