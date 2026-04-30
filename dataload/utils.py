import math
import numpy as np
import torch

# Numerical constants.
NUMERICAL_EPS = 1e-12

# Reference-pose construction.
RELATIVE_POSE_GROUP_SIZE = 4

# History-frame selection hyperparameters.
FOV_SPHERE_SAMPLE_COUNT = 20_000
FOV_SPHERE_RADIUS = 1.0
OVERLAP_RATIO_EPS = 1e-6
DISTANCE_RANGE_EPS = 1e-10
DISTANCE_PENALTY_SCALE = 0.02
MIN_SELECTION_SCORE = 0.8
HIGH_OVERLAP_THRESHOLD = 0.95
MIN_DISTANCE_THRESHOLD = 0.5

def orthonormalize_R(R):
    """Project a near-rotation matrix back to SO(3) with SVD."""
    U, _, Vt = np.linalg.svd(R)
    R_o = U @ Vt
    if np.linalg.det(R_o) < 0:
        U[:, -1] *= -1
        R_o = U @ Vt
    return R_o

def mat3_to_quat_wxyz(R):
    """Convert a 3x3 rotation matrix to a unit quaternion."""
    R = orthonormalize_R(R)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    else:
        if (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + NUMERICAL_EPS
    return q

def quat_wxyz_to_mat3(q):
    """Convert a unit quaternion to a 3x3 rotation matrix."""
    q = q / (np.linalg.norm(q) + NUMERICAL_EPS)
    w, x, y, z = q
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    return orthonormalize_R(R)

def average_quats_markley(qs, weights=None):
    """Average quaternions with the Markley method."""
    qs = np.asarray(qs, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(qs), dtype=np.float64)
    A = np.zeros((4, 4), dtype=np.float64)
    for q, w in zip(qs, weights):
        q = q / (np.linalg.norm(q) + NUMERICAL_EPS)
        A += w * np.outer(q, q)
    eigvals, eigvecs = np.linalg.eigh(A)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    if np.dot(q_mean, qs[0]) < 0:
        q_mean = -q_mean
    return q_mean / (np.linalg.norm(q_mean) + NUMERICAL_EPS)

def average_c2w(c2w_list, weights=None):
    """
    Average camera-to-world transforms in world coordinates.

    Translation is averaged directly from camera centers. Rotation is
    averaged with the Markley quaternion method.
    """
    assert len(c2w_list) > 0
    centers = np.stack([T[:3, 3] for T in c2w_list], axis=0)
    if weights is None:
        C_bar = centers.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + NUMERICAL_EPS)
        C_bar = (centers * w[:, None]).sum(axis=0)

    quats = [mat3_to_quat_wxyz(T[:3, :3]) for T in c2w_list]
    q_bar = average_quats_markley(quats, weights=weights)
    R_bar = quat_wxyz_to_mat3(q_bar)

    T_bar = np.eye(4, dtype=np.float64)
    T_bar[:3, :3] = R_bar
    T_bar[:3, 3] = C_bar
    return T_bar

def compute_rel_poses_letent(abs_c2ws):
    """Build latent reference poses from the first frame and fixed-size groups."""
    ref_c2ws = [abs_c2ws[0].astype(np.float64)]

    num_groups = (abs_c2ws.shape[0] - 1) // RELATIVE_POSE_GROUP_SIZE
    for group_idx in range(num_groups):
        start = 1 + RELATIVE_POSE_GROUP_SIZE * group_idx
        end = start + RELATIVE_POSE_GROUP_SIZE
        c2w_avg = average_c2w([abs_c2ws[i] for i in range(start, end)])
        ref_c2ws.append(c2w_avg)

    return np.array(ref_c2ws)

def c2w_to_xyz_yaw_pitch(c2w, degrees=True):
    """Extract pitch and yaw from camera-to-world rotations using +Z as forward."""
    c2w = np.asarray(c2w, dtype=np.float64)
    if c2w.ndim == 2:
        c2w = c2w[None, ...]

    R = c2w
    f = np.matmul(R, np.array([0.0, 0.0, 1.0]))
    f = f / (np.linalg.norm(f, axis=-1, keepdims=True) + NUMERICAL_EPS)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    yaw = np.arctan2(fx, fz)
    pitch = np.arctan2(fy, np.sqrt(fx * fx + fz * fz))

    if degrees:
        yaw = np.degrees(yaw)
        pitch = np.degrees(pitch)

    return pitch, yaw

def calculate_half_fov(fx, fy):
    """Compute half horizontal and vertical FOV angles in degrees."""
    fx = torch.as_tensor(fx, dtype=torch.float32)
    fy = torch.as_tensor(fy, dtype=torch.float32)

    fov_x = 2 * torch.atan(1 / (2 * fx))
    fov_y = 2 * torch.atan(1 / (2 * fy))

    fov_x_deg = fov_x * 180 / math.pi
    fov_y_deg = fov_y * 180 / math.pi

    return fov_x_deg / 2, fov_y_deg / 2

def is_inside_fov_3d_hv(points, center, center_pitch, center_yaw, fov_half_h, fov_half_v):
    """
    Check whether points are within a given 3D field of view (FOV) 
    with separately defined horizontal and vertical ranges.

    The center view direction is specified by pitch and yaw (in degrees).

    :param points: (N, B, 3) Sample point coordinates
    :param center: (3,) Center coordinates of the FOV
    :param center_pitch: Pitch angle of the center view (in degrees)
    :param center_yaw: Yaw angle of the center view (in degrees)
    :param fov_half_h: Horizontal half-FOV angle (in degrees)
    :param fov_half_v: Vertical half-FOV angle (in degrees)
    :return: Boolean tensor (N, B), indicating whether each point is inside the FOV
    """
    # Compute vectors relative to the center
    vectors = points - center  # shape (N, B, 3)
    x = vectors[..., 0]
    y = vectors[..., 1]
    z = vectors[..., 2]

    # Compute horizontal angle (yaw): measured with respect to the z-axis as the forward direction,
    # and the x-axis as left-right, resulting in a range of -180 to 180 degrees.
    azimuth = torch.atan2(x, z) * (180 / math.pi)

    # Compute vertical angle (pitch): measured with respect to the horizontal plane,
    # resulting in a range of -90 to 90 degrees.
    elevation = torch.atan2(y, torch.sqrt(x**2 + z**2)) * (180 / math.pi)

    # Compute the angular difference from the center view (handling circular angle wrap-around)
    diff_azimuth = (azimuth - center_yaw).abs() % 360
    diff_elevation = (elevation - center_pitch).abs() % 360

    # Adjust values greater than 180 degrees to the shorter angular difference
    diff_azimuth = torch.where(diff_azimuth > 180, 360 - diff_azimuth, diff_azimuth)
    diff_elevation = torch.where(diff_elevation > 180, 360 - diff_elevation, diff_elevation)

    # Check if both horizontal and vertical angles are within their respective FOV limits
    return (diff_azimuth < fov_half_h) & (diff_elevation < fov_half_v)

def generate_points_in_sphere(n_points, radius):
    """Sample points uniformly inside a sphere."""
    samples_r = torch.rand(n_points)
    samples_phi = torch.rand(n_points)
    samples_u = torch.rand(n_points)

    r = radius * torch.pow(samples_r, 1 / 3)
    phi = 2 * math.pi * samples_phi
    theta = torch.acos(1 - 2 * samples_u)

    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)

    points = torch.stack((x, y, z), dim=1)
    return points

def generate_condition_indices(
    curr_frame_idx,
    memory_condition_length,
    pose_conditions,
    fov_x_deg,
    fov_y_deg,
):
    """Select history-frame indices based on FOV overlap and distance."""
    points = generate_points_in_sphere(FOV_SPHERE_SAMPLE_COUNT, FOV_SPHERE_RADIUS)
    points = points.to(device=pose_conditions.device, dtype=pose_conditions.dtype)
    points = points[:, None].repeat(1, pose_conditions.shape[1], 1)
    fov_half_h = fov_x_deg
    fov_half_v = fov_y_deg

    ref_masks = []
    ref_points = []
    ref_centers = []
    for ref_idx in range(curr_frame_idx, pose_conditions.shape[0]):
        points_ref = points + pose_conditions[ref_idx, :, :3, 3][None]
        ref_points.append(points_ref)
        center_pitch, center_yaw = c2w_to_xyz_yaw_pitch(pose_conditions[ref_idx, :, :3, :3])
        center = pose_conditions[ref_idx, :, :3, 3]
        ref_centers.append(center)
        ref_masks.append(
            is_inside_fov_3d_hv(points_ref, center, center_pitch, center_yaw, fov_half_h, fov_half_v)
        )

    ref_in_fov_list = []
    dist_list = []
    for points_ref, center_ref in zip(ref_points, ref_centers):
        per_ref_in_fov_list = []
        per_ref_dist_list = []
        for pc in pose_conditions[:curr_frame_idx]:
            center_pitch, center_yaw = c2w_to_xyz_yaw_pitch(pc[:, :3, :3])
            per_ref_in_fov_list.append(
                is_inside_fov_3d_hv(points_ref, pc[:, :3, 3], center_pitch, center_yaw, fov_half_h, fov_half_v)
            )
            per_ref_dist_list.append(torch.linalg.norm(center_ref - pc[:, :3, 3], dim=-1))

        ref_in_fov_list.append(torch.stack(per_ref_in_fov_list))
        dist_list.append(torch.stack(per_ref_dist_list))

    idx_list = []
    for in_fov1, in_fov_list, per_ref_dist_list in zip(ref_masks, ref_in_fov_list, dist_list):
        overlap_ratio = ((in_fov1.bool() & in_fov_list).sum(1)) / in_fov1.sum().clamp(min=OVERLAP_RATIO_EPS)
        dist_range = (per_ref_dist_list.max() - per_ref_dist_list.min()).clamp(min=DISTANCE_RANGE_EPS)
        distance_penalty = (
            (per_ref_dist_list - per_ref_dist_list.min()) / dist_range * DISTANCE_PENALTY_SCALE
        )
        confidence = overlap_ratio - distance_penalty

        top_vals, r_idx = torch.topk(confidence, k=memory_condition_length, dim=0)
        invalid_mask = top_vals < MIN_SELECTION_SCORE

        selected_overlap = overlap_ratio[r_idx, 0]
        selected_dist = per_ref_dist_list[r_idx, 0]
        extra_mask = (selected_overlap > HIGH_OVERLAP_THRESHOLD) & (
            selected_dist.abs() > MIN_DISTANCE_THRESHOLD
        )

        r_idx = r_idx.clone()
        r_idx[invalid_mask | extra_mask] = -1
        idx_list.append(r_idx.squeeze(-1))
    return idx_list


