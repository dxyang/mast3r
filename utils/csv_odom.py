import csv
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch

K_T_ROBOT_CAM =np.eye(4)
K_T_ROBOT_CAM[:3, :3] = R.from_euler('xyz', [180, 0, -90], degrees=True).as_matrix()

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

                T_world_cam = torch.from_numpy(np.matmul(T_world_robot, K_T_ROBOT_CAM)).float()

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
        elif odometry_timestamp_idx == len(timestamps) - 1:
            # we are pretty far ahead of the last odometry timestamp
            T_world_odomCamDust3r = T_world_odomCams[-1]
        else:
            # calculate slerp alpha
            t0 = timestamps[odometry_timestamp_idx - 1]
            t1 = timestamps[odometry_timestamp_idx]
            alpha = (dust3r_ts - t0) / (t1 - t0)
            # apply slerp
            T_world_odomCam0 = T_world_odomCams[odometry_timestamp_idx - 1]
            T_world_odomCam1 = T_world_odomCams[odometry_timestamp_idx]

            T_world_odomCamDust3r = slerp_poses(T_world_odomCam0, T_world_odomCam1, alpha)

        slerped_T_world_odomCams.append(T_world_odomCamDust3r)

    return slerped_T_world_odomCams
