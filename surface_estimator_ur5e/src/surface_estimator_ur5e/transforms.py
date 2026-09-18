"""Rigid transforms and UR poses (metres and rotation vectors in radians).

Local TCP offsets are applied in the starting pose's frame. Euler-angle helper
arguments explicitly use degrees; all returned position values use metres.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def ur_pose_to_transform(pose_ur: np.ndarray) -> np.ndarray:
    """Convert UR ``[x, y, z, rx, ry, rz]`` pose to a homogeneous transform."""

    pose = np.asarray(pose_ur, dtype=float)
    if pose.shape != (6,):
        raise ValueError(f"Expected a six-dimensional UR pose, got {pose.shape}.")
    return make_transform(Rotation.from_rotvec(pose[3:]).as_matrix(), pose[:3])


def transform_to_ur_pose(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to UR position plus rotation vector."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}.")
    pose = np.empty(6, dtype=float)
    pose[:3] = matrix[:3, 3]
    pose[3:] = Rotation.from_matrix(matrix[:3, :3]).as_rotvec()
    return pose


def translation_transform(offset: np.ndarray | list[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(offset, dtype=float)
    return transform


def rotation_transform(axis: str, angle_deg: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(axis, float(angle_deg), degrees=True).as_matrix()
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ np.asarray(points, dtype=float).T).T + transform[:3, 3]


def marker_relative_transform(
    translation_m: np.ndarray,
    rpy_deg: np.ndarray,
    local_yaw_deg: float,
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(translation_m, dtype=float)
    marker_r_obj = Rotation.from_euler(
        "xyz", np.asarray(rpy_deg, dtype=float), degrees=True
    ).as_matrix()
    obj_r_yawed = Rotation.from_euler("z", float(local_yaw_deg), degrees=True).as_matrix()
    transform[:3, :3] = marker_r_obj @ obj_r_yawed
    return transform


def pose_translated_in_tcp_frame(
    current_tcp_pose_ur: np.ndarray, offset_tcp_m: np.ndarray
) -> np.ndarray:
    pose = np.asarray(current_tcp_pose_ur, dtype=float).copy()
    if pose.shape != (6,):
        raise ValueError(f"Expected 6D TCP pose, got {pose.shape}.")
    offset = np.asarray(offset_tcp_m, dtype=float)
    if offset.shape != (3,):
        raise ValueError(f"Expected 3D TCP offset, got {offset.shape}.")
    rotation = Rotation.from_rotvec(pose[3:6]).as_matrix()
    pose[:3] += rotation @ offset
    return pose


def pose_rotated_in_tcp_frame(
    current_tcp_pose_ur: np.ndarray,
    axis: str,
    angle_deg: float,
) -> np.ndarray:
    pose = np.asarray(current_tcp_pose_ur, dtype=float).copy()
    current_rotation = Rotation.from_rotvec(pose[3:6])
    local_rotation = Rotation.from_euler(axis, float(angle_deg), degrees=True)
    pose[3:6] = (current_rotation * local_rotation).as_rotvec()
    return pose


def rotation_delta_deg(current_tcp_pose_ur: np.ndarray, target_tcp_pose_ur: np.ndarray) -> float:
    current_rotation = Rotation.from_rotvec(current_tcp_pose_ur[3:6]).as_matrix()
    target_rotation = Rotation.from_rotvec(target_tcp_pose_ur[3:6]).as_matrix()
    return float(np.rad2deg(Rotation.from_matrix(target_rotation @ current_rotation.T).magnitude()))


def pose_with_local_xz_adjustment(
    pose: np.ndarray, axis: str, angle_deg: float, x_offset_mm: float, z_offset_mm: float
) -> np.ndarray:
    """Translate in the original TCP frame, then rotate about its local axis."""
    offset_m = np.array([x_offset_mm, 0.0, z_offset_mm], dtype=float) / 1000.0
    return pose_rotated_in_tcp_frame(pose_translated_in_tcp_frame(pose, offset_m), axis, angle_deg)
