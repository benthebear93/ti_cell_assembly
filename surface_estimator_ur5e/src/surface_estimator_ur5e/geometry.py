"""Geometry utilities for contact point extraction and plane fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

NormalDirection = Literal["auto", "positive_z", "negative_z"]


@dataclass(frozen=True)
class PlaneEstimate:
    """Estimated plane and fitting diagnostics."""

    centroid: np.ndarray
    normal: np.ndarray
    d: float
    signed_distances: np.ndarray
    rms_error: float


def quaternion_xyzw_to_matrix(q: np.ndarray | list[float]) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""

    quat = np.asarray(q, dtype=float)
    if quat.shape != (4,):
        raise ValueError(f"Expected quaternion with shape (4,), got {quat.shape}.")
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be non-zero.")
    return Rotation.from_quat(quat / norm).as_matrix()


def compute_contact_point(
    tcp_position: np.ndarray | list[float],
    tcp_quat_xyzw: np.ndarray | list[float],
    contact_offset: np.ndarray | list[float],
) -> np.ndarray:
    """Compute physical contact point from TCP pose and TCP-frame offset."""

    position = np.asarray(tcp_position, dtype=float)
    offset = np.asarray(contact_offset, dtype=float)
    if position.shape != (3,):
        raise ValueError(f"Expected TCP position with shape (3,), got {position.shape}.")
    if offset.shape != (3,):
        raise ValueError(f"Expected contact offset with shape (3,), got {offset.shape}.")
    return position + quaternion_xyzw_to_matrix(tcp_quat_xyzw) @ offset


def estimate_plane_from_points(
    points: np.ndarray | list[list[float]],
    normal_hint: NormalDirection | np.ndarray | list[float] | None = None,
    flip: bool = False,
) -> PlaneEstimate:
    """Estimate a plane from 3 or more points.

    Exactly three points use a cross product. More than three points use SVD/PCA.
    The returned plane equation is ``normal.T @ x + d = 0``.
    """

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {pts.shape}.")
    if pts.shape[0] < 3:
        raise ValueError("At least three contact points are required.")

    centroid = pts.mean(axis=0)
    if pts.shape[0] == 3:
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-10:
            raise ValueError("The three contact points are collinear.")
        normal = normal / normal_norm
    else:
        centered = pts - centroid
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        if singular_values[1] < 1e-10:
            raise ValueError("Contact points are collinear or nearly collinear.")
        normal = vh[-1]
        normal = normal / np.linalg.norm(normal)

    normal = _orient_normal(normal, normal_hint)
    if flip:
        normal = -normal

    d = -float(normal @ centroid)
    signed_distances = point_plane_signed_distances(pts, normal, d)
    rms_error = float(np.sqrt(np.mean(signed_distances**2)))
    return PlaneEstimate(
        centroid=centroid,
        normal=normal,
        d=d,
        signed_distances=signed_distances,
        rms_error=rms_error,
    )


def point_plane_signed_distances(
    points: np.ndarray | list[list[float]],
    normal: np.ndarray | list[float],
    d: float,
) -> np.ndarray:
    """Compute signed distances from points to a normalized plane."""

    pts = np.asarray(points, dtype=float)
    n = np.asarray(normal, dtype=float)
    n_norm = np.linalg.norm(n)
    if n_norm <= 0.0:
        raise ValueError("Plane normal norm must be non-zero.")
    return (pts @ (n / n_norm)) + float(d) / n_norm


def make_surface_frame(centroid: np.ndarray | list[float], normal: np.ndarray | list[float]) -> np.ndarray:
    """Create a 4x4 transform whose z-axis is the plane normal."""

    origin = np.asarray(centroid, dtype=float)
    z_axis = _unit(np.asarray(normal, dtype=float))
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ z_axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    x_axis = _unit(np.cross(reference, z_axis))
    y_axis = _unit(np.cross(z_axis, x_axis))

    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    transform[:3, 3] = origin
    return transform


def _orient_normal(
    normal: np.ndarray,
    normal_hint: NormalDirection | np.ndarray | list[float] | None,
) -> np.ndarray:
    if normal_hint is None:
        return normal
    if isinstance(normal_hint, str) and normal_hint == "auto":
        return normal
    if isinstance(normal_hint, str):
        if normal_hint == "positive_z":
            hint = np.array([0.0, 0.0, 1.0])
        elif normal_hint == "negative_z":
            hint = np.array([0.0, 0.0, -1.0])
        else:
            raise ValueError(
                "normal_hint must be 'auto', 'positive_z', 'negative_z', or a 3D vector."
            )
    else:
        hint = _unit(np.asarray(normal_hint, dtype=float))
    return normal if float(normal @ hint) >= 0.0 else -normal


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm
