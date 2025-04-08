import numpy as np

def depth_coords_to_camera_points(
    coords: np.ndarray, z_vals: np.ndarray, camera_k: np.ndarray
):
    """Transform a set of depth-valued coordinates in the image frame into a PointCloud.

    Args:
        coords: Coordinates in the form [u, v]
        z_vals: Depth values.
        camera_k: Intrinsics.
        frame: the frame enum for the camera.

    Returns:
        A PointCloud representing the depth image projected into the camera frame.
    """
    assert coords.shape[0] == 2 and z_vals.shape[0] == 1, "{} {}".format(
        coords.shape[0], z_vals.shape[0]
    )
    assert coords.shape[1] == z_vals.shape[1]

    # Invert K.
    k_inv = np.linalg.inv(camera_k)

    # Homogenize the coordinats in form [u, v, 1].
    homogenous_uvs = np.concatenate((coords, np.ones((1, coords.shape[1]))))

    # Get the unscaled position for each of the points in the image frame.
    unscaled_points = k_inv @ homogenous_uvs

    # Scale the points by their depth values.
    scaled_points = np.multiply(unscaled_points, z_vals)

    return scaled_points


def depth_image_to_pcd(
    depth_image: np.ndarray,
    camera_k: np.ndarray,
):
    """Convert a depth image (HWC) to a pointmap.

    Args:
        depth_image: Depth as an array (HWC).
        camera_k: Camera intrinsics. 

    Returns:
        pointcloud with respect to the camera frame.
    """
    H, W, C = depth_image.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    pixels = np.stack([u * depth_image.squeeze(), v * depth_image.squeeze(), depth_image.squeeze()], axis=-1).reshape(-1, 3).T  # shape (3, N)
    points_wrt_cam = np.linalg.inv(camera_k) @ pixels  # shape (3, N)

    return points_wrt_cam


def depth_image_to_pointcloud3d(
    depth_image: np.ndarray,
    camera_k: np.ndarray,
    mask: np.ndarray = None,
    subsample: int = 1,
):
    """Convert a depth image (HWC) to a pointcloud.

    Args:
        depth_image: Depth as an array (HWC). Units should be consistent with camera_k
        camera_k: Camera intrinsics. Must be the same units as the depth image
        [optional] mask: NxMx1 mask to apply to the depth image before extracting a pointcloud.
        [optional] subsample: Factor by which to subsample points.

    Returns:
        PointCloud3D of the projected depth image realtive to the camera frame in the same units
        as camera_k and the depth image
    """
    v_size, u_size, _ = depth_image.shape

    # Extract lists of coordinates.
    u_img_range = np.arange(0, u_size)
    v_img_range = np.arange(0, v_size)
    u_grid, v_grid = np.meshgrid(u_img_range, v_img_range)
    u_img, v_img, d = (
        u_grid.ravel(),
        v_grid.ravel(),
        depth_image[v_grid, u_grid].ravel(),
    )

    # Apply mask.
    if mask is not None:
        v_grid, u_grid = np.where(mask.squeeze())
        u_img, v_img, d = (
            u_grid.ravel(),
            v_grid.ravel(),
            depth_image[v_grid, u_grid].ravel(),
        )

    # Subsample, potentially.
    if subsample != 1:
        u_img, v_img, d = u_img[::subsample], v_img[::subsample], d[::subsample]

    # Convert to camera frame.
    pc = depth_coords_to_camera_points(
        np.stack((u_img, v_img)), np.expand_dims(d, axis=0), camera_k
    )

    # ignore points that projected to 0
    d_zero_mask = pc[2, :] != 0
    pc = pc[:, d_zero_mask]

    return pc