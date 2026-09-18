"""Shared holder/tray geometry for execution and visualization.

The ceiling-mounted UR base has +Z pointing down. Tray geometry uses 4x4
transforms; rotation targets and lifts use six-element UR poses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from surface_estimator_ur5e.transforms import (
    rotation_transform,
    transform_points,
    translation_transform,
)

DEFAULT_MARKER_POSE = Path("data/markers/aruco_364_in_base.yaml")
DEFAULT_TI_ASSEMBLY_OBJ = Path(
    "assets/ti_assembly_no_presser_cathode_tray/ti_assembly_no_presser_cathode_tray.obj"
)
DEFAULT_ASSEMBLY_ORIGIN_X_MM = -60.0
DEFAULT_ASSEMBLY_ORIGIN_Y_MM = -18.5
DEFAULT_ASSEMBLY_ORIGIN_Z_MM = 0.0
DEFAULT_ASSEMBLY_RPY_DEG = (180.0, 0.0, 0.0)
DEFAULT_ASSEMBLY_LOCAL_YAW_DEG = -270.0
FOUR_PIN_FRAME_LOCAL_RX_DEG = 90.0
PIN_MATERIAL_NAME = "0.913725_0.647059_0.329412_0.000000_0.000000"
DEFAULT_VIRTUAL_TRAY_OBJ = Path("assets/ti_tray/ti_tray.obj")
DEFAULT_VIRTUAL_TRAY_TCP_Z_MM = 0.0
DEFAULT_VIRTUAL_TRAY_LOCAL_RX_DEG = -90.0
DEFAULT_VIRTUAL_TRAY_HANDLE_ROOT_Y_MM = -37.5
DEFAULT_VIRTUAL_TRAY_HANDLE_SCALE = 0.55
TI_TRAY_PROTRUSION_CENTER_M = np.array([0.0, -0.05150, 0.02750], dtype=float)
TI_TRAY_FOUR_PIN_HOLE_CENTER_M = np.array([0.0, 0.0, 0.01030], dtype=float)
BASE_Z_DOWN = np.array([0.0, 0.0, 1.0], dtype=float)
DEFAULT_TCP_OFFSET_UR = [0.0, 0.0, 0.158, -1.5707, 0.0, 0.0]
DEFAULT_INITIAL_Q_DEG = (44.85, -15.96, -81.60, 7.72, 89.63, 135.01)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


def floor_constrained_marker_transform(base_t_marker_raw: np.ndarray) -> np.ndarray:
    """Return a marker frame constrained to a horizontal floor.

    The UR base is ceiling-mounted in this workcell, so base +Z is treated as the
    physical downward direction. OpenCV/ArUco marker +Z is also the marker backside
    direction for a floor marker seen from above, so a TCP 100 mm above the marker is
    a -Z offset in this constrained marker frame.
    """

    z_axis = BASE_Z_DOWN.copy()
    x_raw = np.asarray(base_t_marker_raw[:3, 0], dtype=float)
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        y_raw = np.asarray(base_t_marker_raw[:3, 1], dtype=float)
        x_axis = np.cross(y_raw, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        raise ValueError("Could not project marker x-axis onto the floor plane.")
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    constrained = np.eye(4)
    constrained[:3, 0] = x_axis
    constrained[:3, 1] = y_axis
    constrained[:3, 2] = z_axis
    constrained[:3, 3] = base_t_marker_raw[:3, 3]
    return constrained


def make_four_pin_rotation_target(
    current_tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    tcp_rotation_offset_rpy_deg: tuple[float, float, float] | list[float],
) -> np.ndarray:
    pin_r_tcp_target = Rotation.from_euler(
        "xyz",
        np.asarray(tcp_rotation_offset_rpy_deg, dtype=float),
        degrees=True,
    ).as_matrix()
    target_tcp_pose = np.asarray(current_tcp_pose_ur, dtype=float).copy()
    target_tcp_pose[3:6] = Rotation.from_matrix(base_t_pin[:3, :3] @ pin_r_tcp_target).as_rotvec()
    return target_tcp_pose


def with_base_z_lift(pose_ur: np.ndarray, lift_m: float) -> np.ndarray:
    lifted = np.asarray(pose_ur, dtype=float).copy()
    # Lifting away from the floor is base -Z because the UR base is ceiling-mounted.
    lifted[2] -= float(lift_m)
    return lifted


def load_tray_geometry(
    obj_path: Path, handle_root_y_mm: float, handle_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return shortened tray vertices and its grasp point, both in CAD metres."""
    import trimesh

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    shorten_lower_tray_handle(mesh, handle_root_y_mm / 1000.0, handle_scale)
    attachment = shortened_lower_handle_point(
        TI_TRAY_PROTRUSION_CENTER_M, handle_root_y_mm / 1000.0, handle_scale
    )
    return np.asarray(mesh.vertices, dtype=float), attachment


def four_pin_feature_frame(obj_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pin_centers = load_pin_centers(obj_path)
    x_groups = cluster_sorted_indices(pin_centers[:, 0], max_gap=0.010)
    if len(x_groups) < 3:
        raise ValueError(f"Expected at least 3 pin columns, found {len(x_groups)}.")

    x_groups = sorted(x_groups, key=lambda indices: float(np.mean(pin_centers[indices, 0])))
    row_split_z = float(np.median(pin_centers[:, 2]))
    grid: dict[tuple[int, str], np.ndarray] = {}
    for column_index, indices in enumerate(x_groups):
        for row_name, selector in (
            ("lower", pin_centers[indices, 2] <= row_split_z),
            ("upper", pin_centers[indices, 2] > row_split_z),
        ):
            row_indices = np.asarray(indices, dtype=int)[selector]
            if len(row_indices) == 0:
                raise ValueError(f"Could not find {row_name} row for pin column {column_index}.")
            grid[(column_index, row_name)] = np.mean(pin_centers[row_indices], axis=0)

    # The four-pin datum is the two columns opposite the adjacent two-pin column.
    selected = np.asarray(
        [
            grid[(0, "lower")],
            grid[(1, "lower")],
            grid[(1, "upper")],
            grid[(0, "upper")],
        ],
        dtype=float,
    )
    adjacent = np.asarray([grid[(2, "lower")], grid[(2, "upper")]], dtype=float)
    left_column = np.mean([grid[(0, "lower")], grid[(0, "upper")]], axis=0)
    right_column = np.mean([grid[(1, "lower")], grid[(1, "upper")]], axis=0)
    lower_row = np.mean([grid[(0, "lower")], grid[(1, "lower")]], axis=0)
    upper_row = np.mean([grid[(0, "upper")], grid[(1, "upper")]], axis=0)

    x_axis = normalized(right_column - left_column)
    y_raw = upper_row - lower_row
    z_axis = normalized(np.cross(x_axis, y_raw))
    y_axis = normalized(np.cross(z_axis, x_axis))

    frame = np.eye(4)
    frame[:3, 0] = x_axis
    frame[:3, 1] = y_axis
    frame[:3, 2] = z_axis
    frame[:3, 3] = np.mean(selected, axis=0)
    frame[:3, :3] = (
        frame[:3, :3]
        @ Rotation.from_euler(
            "x",
            FOUR_PIN_FRAME_LOCAL_RX_DEG,
            degrees=True,
        ).as_matrix()
    )
    return frame, selected, adjacent


def load_pin_centers(obj_path: Path) -> np.ndarray:
    import trimesh

    loaded = trimesh.load(obj_path, process=False)
    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"Expected {obj_path} to load as a trimesh.Scene.")

    pin_mesh = loaded.geometry.get(PIN_MATERIAL_NAME)
    if pin_mesh is None:
        available = ", ".join(str(name) for name in loaded.geometry)
        raise ValueError(f"Could not find pin material {PIN_MATERIAL_NAME}. Available: {available}")

    components = pin_mesh.split(only_watertight=False)
    if len(components) < 6:
        raise ValueError(f"Expected pin mesh to split into multiple parts, got {len(components)}.")
    component_centers = np.asarray(
        [0.5 * (component.bounds[0] + component.bounds[1]) for component in components],
        dtype=float,
    )

    x_groups = cluster_sorted_indices(component_centers[:, 0], max_gap=0.010)
    row_split_z = float(np.median(component_centers[:, 2]))
    pin_centers = []
    for indices in x_groups:
        indices_array = np.asarray(indices, dtype=int)
        for selector in (
            component_centers[indices_array, 2] <= row_split_z,
            component_centers[indices_array, 2] > row_split_z,
        ):
            part_indices = indices_array[selector]
            if len(part_indices) > 0:
                pin_centers.append(np.mean(component_centers[part_indices], axis=0))
    return np.asarray(pin_centers, dtype=float)


def cluster_sorted_indices(values: np.ndarray, max_gap: float) -> list[list[int]]:
    order = np.argsort(np.asarray(values, dtype=float))
    groups: list[list[int]] = [[int(order[0])]]
    for previous_index, current_index in zip(order[:-1], order[1:], strict=True):
        if float(values[current_index] - values[previous_index]) > max_gap:
            groups.append([])
        groups[-1].append(int(current_index))
    return groups


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return np.asarray(vector, dtype=float) / norm


def shorten_lower_tray_handle(mesh: object, handle_root_y_m: float, handle_scale: float) -> None:
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    lower = vertices[:, 1] < handle_root_y_m
    vertices[lower, 1] = handle_root_y_m + (vertices[lower, 1] - handle_root_y_m) * handle_scale
    mesh.vertices = vertices


def shortened_lower_handle_point(
    point: np.ndarray,
    handle_root_y_m: float,
    handle_scale: float,
) -> np.ndarray:
    shortened = np.asarray(point, dtype=float).copy()
    if shortened[1] < handle_root_y_m:
        shortened[1] = handle_root_y_m + (shortened[1] - handle_root_y_m) * handle_scale
    return shortened


def tray_vertices_in_base(
    tcp_transform: np.ndarray,
    tray_vertices: np.ndarray,
    tray_attachment_local: np.ndarray,
    tray_center_tcp_z_mm: float,
    tray_local_rx_deg: float,
) -> np.ndarray:
    """Place CAD vertices using a base-to-TCP 4x4 transform."""
    base_t_tray = (
        np.asarray(tcp_transform, dtype=float)
        @ translation_transform([0.0, 0.0, tray_center_tcp_z_mm / 1000.0])
        @ rotation_transform("x", tray_local_rx_deg)
        @ translation_transform(-tray_attachment_local)
    )
    return transform_points(base_t_tray, tray_vertices)


def tray_floor_clearance_m(
    tcp_transform: np.ndarray,
    tray_vertices: np.ndarray,
    tray_attachment_local: np.ndarray,
    floor_z_m: float,
    tray_center_tcp_z_mm: float,
    tray_local_rx_deg: float,
) -> float:
    """Signed clearance: positive above the floor, negative through it."""
    vertices_base = tray_vertices_in_base(
        tcp_transform,
        tray_vertices,
        tray_attachment_local,
        tray_center_tcp_z_mm=tray_center_tcp_z_mm,
        tray_local_rx_deg=tray_local_rx_deg,
    )
    return floor_z_m - float(np.max(vertices_base[:, 2]))
