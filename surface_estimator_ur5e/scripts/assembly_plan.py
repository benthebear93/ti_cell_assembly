"""Marker/tray geometry and motion planning (no robot motion commands)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from assembly_config import (
    IK_ORIENTATION_TOLERANCE_RAD,
    IK_POSITION_TOLERANCE_M,
    LIFT_WAYPOINT_STEP_MM,
    MAX_IK_JOINT_STEP_DEG,
    MIN_Q3_SINGULARITY_MARGIN_DEG,
    MIN_Q5_SINGULARITY_MARGIN_DEG,
    POST_GRASP_DWELL_S,
    ROTATION_LIFT_CANDIDATES_MM,
    ROTATION_WAYPOINT_STEP_DEG,
    effective_pin_image_alignment_metadata,
    effective_post_two_pin_close_image_alignment_metadata,
)
from scipy.spatial.transform import Rotation, Slerp

from surface_estimator_ur5e.transforms import (
    marker_relative_transform,
    pose_rotated_in_tcp_frame,
    pose_translated_in_tcp_frame,
    rotation_delta_deg,
    ur_pose_to_transform,
)
from surface_estimator_ur5e.workcell_geometry import (
    TI_TRAY_FOUR_PIN_HOLE_CENTER_M,
    TI_TRAY_PROTRUSION_CENTER_M,
    four_pin_feature_frame,
    load_tray_geometry,
    make_four_pin_rotation_target,
    resolve_project_path,
    shortened_lower_handle_point,
    tray_floor_clearance_m,
    with_base_z_lift,
)

Targets = list[tuple[str, np.ndarray]]


def tcp_shift(pose: np.ndarray, *, x: float = 0, y: float = 0, z: float = 0) -> np.ndarray:
    """Translate by millimetres in the current TCP frame; keep its orientation."""
    return pose_translated_in_tcp_frame(pose, np.array([x, y, z], dtype=float) / 1000.0)


def report_targets(targets: Targets) -> Targets:
    for name, pose in targets:
        print(f"  {name}: {fmt(pose)}")
    return targets


def check_joint_margins(q: np.ndarray, motion_name: str) -> tuple[float, float]:
    margins = (distance_to_nearest_pi_multiple_deg(q[2]), distance_to_nearest_pi_multiple_deg(q[4]))
    for joint, margin, minimum in zip(
        (3, 5), margins, (MIN_Q3_SINGULARITY_MARGIN_DEG, MIN_Q5_SINGULARITY_MARGIN_DEG), strict=True
    ):
        if margin < minimum:
            raise RuntimeError(
                f"q{joint} singularity margin is {margin:.1f} deg at {motion_name}; no further motion was sent."
            )
    return margins


def marker_offset_target_position(
    base_t_marker: np.ndarray, offset_marker_m: np.ndarray
) -> np.ndarray:
    offset_homogeneous = np.ones(4, dtype=float)
    offset_homogeneous[:3] = offset_marker_m
    return (base_t_marker @ offset_homogeneous)[:3]


def marker_target_offset(args: argparse.Namespace) -> np.ndarray:
    if args.marker_frame_mode == "raw":
        z_mm = args.offset_z_mm
        if z_mm is None:
            z_mm = args.height_above_marker_mm
        return np.array([args.offset_x_mm, args.offset_y_mm, z_mm], dtype=float) / 1000.0

    return (
        np.array(
            [args.offset_x_mm, args.offset_y_mm, -args.height_above_marker_mm],
            dtype=float,
        )
        / 1000.0
    )


def make_target_pose_ur(
    base_t_marker: np.ndarray,
    offset_marker_m: np.ndarray,
    current_tcp_pose_ur: np.ndarray,
    orientation: str,
) -> np.ndarray:
    target_position = marker_offset_target_position(base_t_marker, offset_marker_m)
    if orientation == "current":
        target_rotvec = current_tcp_pose_ur[3:6]
    elif orientation == "marker":
        target_rotvec = Rotation.from_matrix(base_t_marker[:3, :3]).as_rotvec()
    else:
        raise ValueError(f"Unknown orientation mode: {orientation}")
    return np.concatenate([target_position, target_rotvec])


def save_target_yaml(
    path: Path,
    marker_id: int | None,
    marker_pose_path: Path,
    base_t_marker: np.ndarray,
    offset_marker_m: np.ndarray,
    current_tcp_pose_ur: np.ndarray,
    target_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
) -> None:
    payload = {
        "marker_id": marker_id,
        "marker_pose_file": str(marker_pose_path),
        "marker_frame_mode": args.marker_frame_mode,
        "offset_marker_m": round_list(offset_marker_m),
        "height_above_marker_m": round(float(args.height_above_marker_mm / 1000.0), 10),
        "orientation_mode": args.orientation,
        "target_tcp_pose_ur": round_list(target_tcp_pose_ur),
        "target_tcp_position_m": round_list(target_tcp_pose_ur[:3]),
        "target_tcp_rotation_vector_rad": round_list(target_tcp_pose_ur[3:]),
        "current_tcp_pose_ur_at_plan_time": round_list(current_tcp_pose_ur),
        "distance_from_current_tcp_mm": round(
            float(np.linalg.norm(target_tcp_pose_ur[:3] - current_tcp_pose_ur[:3]) * 1000.0),
            3,
        ),
        "base_T_marker": matrix_list(base_t_marker),
        "robot_ip": args.robot_ip,
        "tcp_offset_ur": round_list(args.tcp_offset_ur),
        "set_tcp_before_plan": not args.no_set_tcp,
        "start_from_initial_pose": bool(args.start_from_initial_pose),
        "initial_q_deg": round_list(np.asarray(args.initial_q_deg)),
        "close_gripper_after_move": not args.skip_gripper_close,
        "gripper_delay_s": round(float(args.gripper_delay_s), 4),
        "post_grasp_dwell_s": POST_GRASP_DWELL_S,
        "gripper_close_percent": round(float(args.gripper_close_percent), 4),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def compute_base_t_four_pin_frame(
    args: argparse.Namespace, base_t_marker: np.ndarray
) -> np.ndarray:
    marker_t_assembly = marker_relative_transform(
        np.asarray(
            [
                args.assembly_origin_x_mm,
                args.assembly_origin_y_mm,
                args.assembly_origin_z_mm,
            ],
            dtype=float,
        )
        / 1000.0,
        np.asarray(args.assembly_rpy_deg, dtype=float),
        args.assembly_local_yaw_deg,
    )
    assembly_t_pin, _selected_pin_centers, _adjacent_pin_centers = four_pin_feature_frame(
        resolve_project_path(args.assembly_obj)
    )
    return base_t_marker @ marker_t_assembly @ assembly_t_pin


def adjacent_two_pin_center_offset_in_pin(args: argparse.Namespace) -> tuple[np.ndarray, float]:
    assembly_t_pin, selected_pin_centers, adjacent_pin_centers = four_pin_feature_frame(
        resolve_project_path(args.assembly_obj)
    )
    four_pin_center = np.mean(selected_pin_centers, axis=0)
    adjacent_two_pin_center = np.mean(adjacent_pin_centers, axis=0)
    offset_in_pin = assembly_t_pin[:3, :3].T @ (adjacent_two_pin_center - four_pin_center)
    distance_mm = float(np.linalg.norm(adjacent_two_pin_center - four_pin_center) * 1000.0)
    return offset_in_pin, distance_mm


def adjacent_two_pin_center_shift_x_mm(args: argparse.Namespace) -> float:
    offset_in_pin, distance_mm = adjacent_two_pin_center_offset_in_pin(args)
    x_distance_mm = abs(float(offset_in_pin[0] * 1000.0))
    lateral_residual_mm = float(np.linalg.norm(offset_in_pin[1:]) * 1000.0)
    if x_distance_mm < 1e-6:
        raise RuntimeError("Could not compute adjacent two-pin +X shift from holder OBJ.")
    if lateral_residual_mm > 1.0:
        print(
            "warning: adjacent two-pin center is not purely along holder pin-frame X; "
            f"using X component {x_distance_mm:.3f} mm from total distance {distance_mm:.3f} mm."
        )
    return x_distance_mm


def adjacent_two_pin_center_base(args: argparse.Namespace, base_t_pin: np.ndarray) -> np.ndarray:
    offset_in_pin, _distance_mm = adjacent_two_pin_center_offset_in_pin(args)
    return (
        np.asarray(base_t_pin[:3, 3], dtype=float)
        + np.asarray(base_t_pin[:3, :3], dtype=float) @ offset_in_pin
    )


def effective_post_insert_shift_x_mm(args: argparse.Namespace) -> float:
    if args.post_insert_shift_x_mm is not None:
        return float(args.post_insert_shift_x_mm)
    return adjacent_two_pin_center_shift_x_mm(args)


def tray_attachment_local(args: argparse.Namespace) -> np.ndarray:
    return shortened_lower_handle_point(
        TI_TRAY_PROTRUSION_CENTER_M,
        args.tray_handle_root_y_mm / 1000.0,
        args.tray_handle_scale,
    )


def tray_four_hole_center_offset_tcp(args: argparse.Namespace) -> np.ndarray:
    tray_r_reference = Rotation.from_euler("x", args.tray_local_rx_deg, degrees=True).as_matrix()
    return np.asarray(
        [0.0, 0.0, args.tray_center_tcp_z_mm / 1000.0], dtype=float
    ) + tray_r_reference @ (TI_TRAY_FOUR_PIN_HOLE_CENTER_M - tray_attachment_local(args))


def tray_four_hole_center_position(tcp_pose_ur: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    pose = np.asarray(tcp_pose_ur, dtype=float)
    rotation = Rotation.from_rotvec(pose[3:6]).as_matrix()
    return pose[:3] + rotation @ tray_four_hole_center_offset_tcp(args)


def move_target_tray_center_toward_pin(
    target_tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
    approach_mm: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    target_pose = np.asarray(target_tcp_pose_ur, dtype=float).copy()
    requested_approach_m = float(approach_mm) / 1000.0
    if requested_approach_m <= 0.0:
        return target_pose, np.zeros(3, dtype=float), 0.0, 0.0

    pin_axis = np.asarray(base_t_pin[:3, 2], dtype=float)
    pin_axis_norm = float(np.linalg.norm(pin_axis))
    if pin_axis_norm < 1e-9:
        raise ValueError("Four-pin frame has a near-zero insertion axis.")
    pin_axis = pin_axis / pin_axis_norm

    tray_center = tray_four_hole_center_position(target_pose, args)
    pin_center = np.asarray(base_t_pin[:3, 3], dtype=float)
    pin_to_tray = pin_center - tray_center
    lateral_delta = pin_to_tray - np.dot(pin_to_tray, pin_axis) * pin_axis
    lateral_distance_m = float(np.linalg.norm(lateral_delta))
    if lateral_distance_m < 1e-9:
        return target_pose, np.zeros(3, dtype=float), 0.0, 0.0

    approach_m = min(requested_approach_m, lateral_distance_m)
    approach_vector = lateral_delta / lateral_distance_m * approach_m
    target_pose[:3] += approach_vector
    return target_pose, approach_vector, lateral_distance_m, lateral_distance_m - approach_m


def pin_target(
    pose: np.ndarray,
    pin: np.ndarray,
    args: argparse.Namespace,
    coordinates: tuple[float | None, ...],
):
    """Position the tray holes in the pin frame; None preserves that coordinate."""
    pose = np.asarray(pose, dtype=float)
    current_reference = tray_four_hole_center_position(pose, args)
    rotation, origin = pin[:3, :3], pin[:3, 3]
    current_in_pin = rotation.T @ (current_reference - origin)
    target_in_pin = current_in_pin.copy()
    for axis, value in enumerate(coordinates):
        if value is not None:
            target_in_pin[axis] = value
    target_reference = origin + rotation @ target_in_pin
    translation = target_reference - current_reference
    target = pose.copy()
    target[:3] += translation
    return target, current_reference, target_reference, translation, current_in_pin, target_in_pin


def pin_approach_target_pose(
    pose: np.ndarray, pin: np.ndarray, args: argparse.Namespace, clearance_mm: float
):
    return pin_target(pose, pin, args, (0, float(clearance_mm) / 1000, 0))[:4]


def pin_centering_target_pose(pose: np.ndarray, pin: np.ndarray, args: argparse.Namespace):
    return pin_target(pose, pin, args, (0, None, 0))[:4]


def load_pin_image_alignment_correction(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    axis = str(data.get("axis", "y")).lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"{path} has invalid alignment axis {axis!r}.")

    angle_deg = float(data.get("angle_deg", 0.0))
    if "tcp_local_x_offset_mm" in data or "tcp_local_z_offset_mm" in data:
        x_offset_mm = float(data.get("tcp_local_x_offset_mm", 0.0))
        z_offset_mm = float(data.get("tcp_local_z_offset_mm", 0.0))
    else:
        start_pose = np.asarray(data["start_tcp_pose_ur"], dtype=float).reshape(6)
        target_pose = np.asarray(data["target_tcp_pose_ur"], dtype=float).reshape(6)
        start_rotation = Rotation.from_rotvec(start_pose[3:6]).as_matrix()
        local_delta_mm = start_rotation.T @ (target_pose[:3] - start_pose[:3]) * 1000.0
        x_offset_mm = float(local_delta_mm[0])
        z_offset_mm = float(local_delta_mm[2])

    values = np.asarray([angle_deg, x_offset_mm, z_offset_mm], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains non-finite image alignment correction values.")

    return {
        "path": str(path),
        "label": data.get("label"),
        "axis": axis,
        "angle_deg": angle_deg,
        "tcp_local_x_offset_mm": x_offset_mm,
        "tcp_local_z_offset_mm": z_offset_mm,
    }


def pose_with_tcp_local_alignment_correction(
    pose: np.ndarray, correction: dict[str, Any]
) -> np.ndarray:
    shifted = tcp_shift(
        pose, x=correction["tcp_local_x_offset_mm"], z=correction["tcp_local_z_offset_mm"]
    )
    return pose_rotated_in_tcp_frame(shifted, correction["axis"], correction["angle_deg"])


def print_pin_image_alignment_correction_plan(
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
    title: str = "Post-approach image alignment correction",
    metadata_path: Path | None = None,
) -> np.ndarray | None:
    metadata_path = metadata_path or effective_pin_image_alignment_metadata(args)
    if metadata_path is None:
        return None
    correction = load_pin_image_alignment_correction(metadata_path)
    translation_mm = float(
        np.linalg.norm([correction["tcp_local_x_offset_mm"], correction["tcp_local_z_offset_mm"]])
    )
    if translation_mm > args.max_pin_image_correction_mm:
        raise RuntimeError(
            f"Image correction {translation_mm:.3f} mm exceeds max_pin_image_correction_mm."
        )
    if abs(correction["angle_deg"]) > args.max_pin_image_correction_deg:
        raise RuntimeError("Image rotation exceeds max_pin_image_correction_deg.")
    target = pose_with_tcp_local_alignment_correction(start_tcp_pose_ur, correction)
    report_targets([(title, target)])
    return target


def print_post_two_pin_close_image_alignment_plan(
    start: np.ndarray, args: argparse.Namespace
) -> np.ndarray | None:
    if not args.post_two_pin_close_image_align or args.auto_pin_image_align:
        return None
    return print_pin_image_alignment_correction_plan(
        start,
        args,
        title="Post-two-pin-close image alignment correction",
        metadata_path=effective_post_two_pin_close_image_alignment_metadata(args),
    )


def post_two_pin_aligned_insert_release_target_poses(
    start: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, ...]:
    insert = tcp_shift(start, y=-args.post_two_pin_aligned_insert_y_mm)
    retreat = tcp_shift(insert, y=args.post_two_pin_release_retreat_y_mm)
    shift_z = tcp_shift(retreat, z=args.post_two_pin_release_shift_z_mm)
    shift_xy = tcp_shift(
        shift_z, x=args.post_two_pin_release_final_x_mm, y=args.post_two_pin_release_final_y_mm
    )
    rotate = pose_rotated_in_tcp_frame(shift_xy, "x", args.post_two_pin_release_rotate_x_deg)
    return insert, retreat, shift_z, shift_xy, rotate


def print_post_two_pin_aligned_insert_release_plan(
    start: np.ndarray, args: argparse.Namespace
) -> Targets:
    if not args.post_two_pin_aligned_insert_release:
        return []
    names = (
        "post-two-pin aligned TCP -Y insertion",
        "post-two-pin release TCP +Y retreat",
        "post-two-pin release TCP Z shift",
        "post-two-pin release TCP final X/Y shift",
        "post-two-pin release TCP final X rotation",
    )
    return report_targets(
        list(zip(names, post_two_pin_aligned_insert_release_target_poses(start, args), strict=True))
    )


def post_two_pin_release_after_rotation_target_poses(
    start: np.ndarray, args: argparse.Namespace
) -> Targets:
    shift_y = tcp_shift(start, y=-40.0)
    shift_z = tcp_shift(shift_y, z=10.0)
    return [
        ("post-two-pin release after-rotation TCP Y shift", shift_y),
        ("post-two-pin release after-rotation TCP Z shift", shift_z),
    ]


def print_post_two_pin_release_after_rotation_plan(
    start: np.ndarray, args: argparse.Namespace
) -> Targets:
    return report_targets(post_two_pin_release_after_rotation_target_poses(start, args))


def post_two_pin_after_close_target_poses(start: np.ndarray, args: argparse.Namespace) -> Targets:
    pre_shift = tcp_shift(
        start,
        y=-3.0,
        z=20.0,
    )
    shift = tcp_shift(
        pre_shift, y=args.post_two_pin_after_close_y_mm, z=args.post_two_pin_after_close_final_z_mm
    )
    rotate = pose_rotated_in_tcp_frame(shift, "x", -5.0)
    return [
        ("post-two-pin after-close pre-final TCP Y/Z shift", pre_shift),
        ("post-two-pin after-close TCP Y/Z shift", shift),
        ("post-two-pin after-close TCP X rotation", rotate),
    ]


def print_post_two_pin_after_close_plan(start: np.ndarray, args: argparse.Namespace) -> Targets:
    return report_targets(post_two_pin_after_close_target_poses(start, args))


def post_two_pin_after_close_tail_target_poses(
    start: np.ndarray, args: argparse.Namespace
) -> Targets:
    tail = tcp_shift(
        start,
        y=-60.0,
        z=15.0,
    )
    return [
        ("post-two-pin after-close tail TCP Y/Z shift", tail),
        *post_two_pin_after_close_tail_next_target_poses(tail, args),
    ]


def post_two_pin_after_close_tail_next_target_poses(
    start: np.ndarray, args: argparse.Namespace
) -> Targets:
    rotate = pose_rotated_in_tcp_frame(start, "x", 30.0)
    shift = tcp_shift(
        rotate,
        y=-35.0,
        z=-2.0,
    )
    return [
        ("post-two-pin after-close tail next TCP X rotation", rotate),
        ("post-two-pin after-close tail next TCP Y/Z shift", shift),
    ]


def pin_insertion_target_pose(
    pose: np.ndarray, pin: np.ndarray, args: argparse.Namespace, target_y_mm: float
):
    return pin_target(pose, pin, args, (None, float(target_y_mm) / 1000, None))


def print_pin_insertion_plan(
    start: np.ndarray, base_t_pin: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    target, _, _, translation, _, _ = pin_insertion_target_pose(
        start, base_t_pin, args, args.pin_insertion_target_y_mm
    )
    distance_mm = float(np.linalg.norm(translation) * 1000.0)
    if distance_mm > args.max_pin_insertion_mm:
        raise RuntimeError(
            f"Insertion {distance_mm:.1f} mm exceeds max_pin_insertion_mm {args.max_pin_insertion_mm:.1f}."
        )
    report_targets([("final pin insertion descend", target)])
    return target


def post_insert_retreat_target_poses(
    start: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, float]:
    shift_x_mm = effective_post_insert_shift_x_mm(args)
    retreat = tcp_shift(start, y=args.post_insert_retreat_y_mm)
    return retreat, tcp_shift(retreat, x=shift_x_mm), shift_x_mm


def adjacent_two_pin_tcp_target_pose(
    current_tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_pose = np.asarray(current_tcp_pose_ur, dtype=float).copy()
    adjacent_center = adjacent_two_pin_center_base(args, base_t_pin)
    target_position = adjacent_center + np.asarray(base_t_pin[:3, 1], dtype=float) * (
        args.post_insert_two_pin_target_y_mm / 1000.0
    )
    translation = target_position - target_pose[:3]
    target_pose[:3] = target_position
    return target_pose, adjacent_center, target_position, translation


def print_post_insert_release_retreat_plan(
    start: np.ndarray,
    args: argparse.Namespace,
    base_t_pin: np.ndarray | None = None,
) -> Targets:
    if not args.post_insert_release_retreat:
        return []
    retreat, shift, _ = post_insert_retreat_target_poses(start, args)
    targets = [
        ("post-insert TCP +Y retreat", retreat),
        ("post-insert TCP +X adjacent-pin shift", shift),
    ]
    if args.post_insert_move_to_two_pin:
        if base_t_pin is None:
            raise RuntimeError("Adjacent two-pin target needs the holder four-pin frame.")
        target, _, _, _ = adjacent_two_pin_tcp_target_pose(shift, base_t_pin, args)
        targets.append(("post-insert move TCP to adjacent two-pin center", target))
    return report_targets(targets)


def post_two_pin_close_retreat_target_poses(
    start: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    retreat = tcp_shift(start, y=args.post_two_pin_close_retreat_y_mm)
    return retreat, tcp_shift(retreat, x=-args.post_two_pin_close_shift_negative_x_mm)


def print_post_two_pin_close_retreat_plan(start: np.ndarray, args: argparse.Namespace) -> Targets:
    if not args.post_two_pin_close_retreat:
        return []
    retreat, shift = post_two_pin_close_retreat_target_poses(start, args)
    return report_targets(
        [
            ("post-two-pin-close TCP +Y retreat", retreat),
            ("post-two-pin-close TCP -X shift", shift),
        ]
    )


def four_pin_alignment_plan(
    pose: np.ndarray,
    pin: np.ndarray,
    floor_z_m: float,
    args: argparse.Namespace,
    minimum_lift_m: float = 0.0,
) -> dict[str, Any]:
    vertices, attachment = load_tray_geometry(
        resolve_project_path(args.tray_obj), args.tray_handle_root_y_mm, args.tray_handle_scale
    )
    target = make_four_pin_rotation_target(pose, pin, args.tcp_rotation_offset_rpy_deg)
    target, approach, before, after = move_target_tray_center_toward_pin(
        target, pin, args, args.rotation_approach_mm
    )
    current_clearance, target_clearance = (
        tray_floor_clearance_m(
            ur_pose_to_transform(p),
            vertices,
            attachment,
            floor_z_m,
            args.tray_center_tcp_z_mm,
            args.tray_local_rx_deg,
        )
        for p in (pose, target)
    )
    requested = args.safe_floor_clearance_mm / 1000.0
    lift = max(
        args.post_grasp_lift_mm / 1000.0,
        minimum_lift_m,
        requested - current_clearance,
        requested - target_clearance,
        0.0,
    )
    return {
        "target_tcp_pose_ur": target,
        "lifted_current_tcp_pose_ur": with_base_z_lift(pose, lift),
        "lifted_target_tcp_pose_ur": with_base_z_lift(target, lift),
        "angle_delta_deg": rotation_delta_deg(pose, target),
        "current_clearance_m": current_clearance,
        "target_clearance_m": target_clearance,
        "safety_lift_m": lift,
        "rotation_approach_vector_m": approach,
        "pre_approach_lateral_distance_m": before,
        "post_approach_lateral_distance_m": after,
    }


def pose_waypoints(
    start_pose_ur: np.ndarray,
    target_pose_ur: np.ndarray,
    translation_step_m: float,
    rotation_step_deg: float,
) -> list[np.ndarray]:
    start_pose = np.asarray(start_pose_ur, dtype=float)
    target_pose = np.asarray(target_pose_ur, dtype=float)
    translation_distance_m = float(np.linalg.norm(target_pose[:3] - start_pose[:3]))
    rotation_distance_deg = rotation_delta_deg(start_pose, target_pose)
    segments = max(
        1,
        int(np.ceil(translation_distance_m / translation_step_m)),
        int(np.ceil(rotation_distance_deg / rotation_step_deg)),
    )
    fractions = np.linspace(0.0, 1.0, segments + 1)
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_rotvec(np.vstack([start_pose[3:6], target_pose[3:6]])),
    )(fractions)

    waypoints: list[np.ndarray] = []
    for fraction, rotation in zip(fractions[1:], rotations[1:], strict=True):
        pose = np.empty(6, dtype=float)
        pose[:3] = (1.0 - fraction) * start_pose[:3] + fraction * target_pose[:3]
        pose[3:6] = rotation.as_rotvec()
        waypoints.append(pose)
    return waypoints


def alignment_waypoints(
    grasped_tcp_pose_ur: np.ndarray,
    plan: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    lift_waypoints = pose_waypoints(
        grasped_tcp_pose_ur,
        plan["lifted_current_tcp_pose_ur"],
        LIFT_WAYPOINT_STEP_MM / 1000.0,
        ROTATION_WAYPOINT_STEP_DEG,
    )
    rotation_waypoints = pose_waypoints(
        plan["lifted_current_tcp_pose_ur"],
        plan["lifted_target_tcp_pose_ur"],
        LIFT_WAYPOINT_STEP_MM / 1000.0,
        ROTATION_WAYPOINT_STEP_DEG,
    )
    return lift_waypoints, rotation_waypoints


def distance_to_nearest_pi_multiple_deg(angle_rad: float) -> float:
    wrapped = (float(angle_rad) + np.pi / 2.0) % np.pi - np.pi / 2.0
    return abs(float(np.rad2deg(wrapped)))


def unwrap_joints_near(q_rad: np.ndarray, q_near_rad: np.ndarray) -> np.ndarray:
    q = np.asarray(q_rad, dtype=float)
    q_near = np.asarray(q_near_rad, dtype=float)
    return q + 2.0 * np.pi * np.round((q_near - q) / (2.0 * np.pi))


def preflight_alignment_ik(
    rtde_control: Any,
    q_start_rad: np.ndarray,
    lift_waypoints: list[np.ndarray],
    rotation_waypoints: list[np.ndarray],
) -> dict[str, Any]:
    q_near = np.asarray(q_start_rad, dtype=float)
    q_path = []
    min_q3 = min_q5 = float("inf")
    max_step = 0.0
    for index, pose in enumerate(lift_waypoints + rotation_waypoints, start=1):
        name = f"alignment waypoint {index}"
        q = unwrap_joints_near(require_ik_solution(rtde_control, q_near, pose, name), q_near)
        step = float(np.max(np.abs(np.rad2deg(q - q_near))))
        if step > MAX_IK_JOINT_STEP_DEG:
            raise RuntimeError(f"IK branch jump of {step:.1f} deg at {name}.")
        q3, q5 = check_joint_margins(q, name)
        min_q3, min_q5, max_step = min(min_q3, q3), min(min_q5, q5), max(max_step, step)
        q_path.append(q)
        q_near = q
    return dict(
        q_path=q_path,
        minimum_q3_margin_deg=min_q3,
        minimum_q5_margin_deg=min_q5,
        maximum_joint_step_deg=max_step,
    )


def select_safe_alignment_plan(
    rtde_control: Any,
    grasped_tcp_pose_ur: np.ndarray,
    q_start_rad: np.ndarray,
    base_t_pin: np.ndarray,
    floor_z_m: float,
    args: argparse.Namespace,
):
    candidates, failures = [], []
    for lift_mm in ROTATION_LIFT_CANDIDATES_MM:
        plan = four_pin_alignment_plan(
            grasped_tcp_pose_ur, base_t_pin, floor_z_m, args, minimum_lift_m=lift_mm / 1000.0
        )
        if plan["safety_lift_m"] > args.max_safety_lift_mm / 1000.0:
            failures.append(f"{1000 * plan['safety_lift_m']:.1f} mm lift exceeds safety limit")
            continue
        lift, rotation = alignment_waypoints(grasped_tcp_pose_ur, plan)
        try:
            report = preflight_alignment_ik(rtde_control, q_start_rad, lift, rotation)
            candidates.append((plan, lift, rotation, report))
        except RuntimeError as exc:
            failures.append(f"{lift_mm:.1f} mm lift: {exc}")
    if not candidates:
        raise RuntimeError("No safe four-pin rotation path was found: " + "; ".join(failures))
    return min(
        candidates,
        key=lambda c: (
            c[0]["safety_lift_m"],
            -c[3]["minimum_q3_margin_deg"],
            -c[3]["minimum_q5_margin_deg"],
        ),
    )


def print_four_pin_alignment_plan(
    plan: dict[str, Any], base_t_pin: np.ndarray, args: argparse.Namespace
) -> None:
    print(
        f"Four-pin alignment: rotation={plan['angle_delta_deg']:.1f} deg, lift={1000 * plan['safety_lift_m']:.1f} mm"
    )
    print(
        f"  floor clearance before/after rotation: {1000 * plan['current_clearance_m']:.1f}/{1000 * plan['target_clearance_m']:.1f} mm"
    )
    report_targets([("lifted four-pin target", plan["lifted_target_tcp_pose_ur"])])


def print_pin_approach_plan(
    start: np.ndarray, base_t_pin: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray]:
    centering, _, _, _ = pin_centering_target_pose(start, base_t_pin, args)
    approach, _, _, _ = pin_approach_target_pose(
        centering, base_t_pin, args, args.pin_approach_clearance_mm
    )
    report_targets([("pin centering", centering), ("pin-axis approach", approach)])
    return centering, approach


def post_two_pin_release_resume_target_poses(
    start: np.ndarray, args: argparse.Namespace
) -> Targets:
    shift_z = tcp_shift(start, z=args.post_two_pin_release_shift_z_mm)
    shift_xy = tcp_shift(
        shift_z, x=args.post_two_pin_release_final_x_mm, y=args.post_two_pin_release_final_y_mm
    )
    rotate = pose_rotated_in_tcp_frame(shift_xy, "x", args.post_two_pin_release_rotate_x_deg)
    return [
        ("resume post-two-pin release TCP Z shift", shift_z),
        ("resume post-two-pin release TCP final X/Y shift", shift_xy),
        ("resume post-two-pin release TCP final X rotation", rotate),
    ]


def print_post_two_pin_release_resume_plan(start: np.ndarray, args: argparse.Namespace) -> Targets:
    return report_targets(post_two_pin_release_resume_target_poses(start, args))


def round_list(values: np.ndarray, decimals: int = 10) -> list[float]:
    return [round(float(value), decimals) for value in np.asarray(values, dtype=float).reshape(-1)]


def matrix_list(matrix: np.ndarray, decimals: int = 10) -> list[list[float]]:
    return [[round(float(value), decimals) for value in row] for row in np.asarray(matrix)]


def fmt(values: np.ndarray, decimals: int = 6) -> str:
    return "[" + ", ".join(f"{float(value): .{decimals}f}" for value in values) + "]"


@dataclass
class PostInsertPlan:
    """Named groups preserve the gripper/camera boundaries in the sequence."""

    retreat: list[tuple[str, np.ndarray]] = field(default_factory=list)
    transfer: list[tuple[str, np.ndarray]] = field(default_factory=list)
    correction: np.ndarray | None = None
    aligned_release: list[tuple[str, np.ndarray]] = field(default_factory=list)
    after_rotation: list[tuple[str, np.ndarray]] = field(default_factory=list)
    align_image: bool = False

    def targets(self) -> list[tuple[str, np.ndarray]]:
        targets = [*self.retreat, *self.transfer]
        if self.correction is not None:
            targets.append(("post-two-pin-close image alignment correction", self.correction))
        return [*targets, *self.aligned_release, *self.after_rotation]


def plan_post_insert(
    start_pose: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> PostInsertPlan:
    """Plan release, two-pin pickup and transfer in their execution order."""
    plan = PostInsertPlan()
    plan.retreat = print_post_insert_release_retreat_plan(start_pose, args, base_t_pin)
    if not plan.retreat or not args.post_insert_move_to_two_pin:
        return plan
    plan.transfer = print_post_two_pin_close_retreat_plan(plan.retreat[-1][1], args)
    if not plan.transfer:
        return plan
    plan.correction = print_post_two_pin_close_image_alignment_plan(plan.transfer[-1][1], args)
    plan.align_image = args.post_two_pin_close_image_align and (
        args.auto_pin_image_align
        or effective_post_two_pin_close_image_alignment_metadata(args) is not None
    )
    if not plan.align_image or not args.post_two_pin_aligned_insert_release:
        return plan
    aligned_pose = plan.correction if plan.correction is not None else plan.transfer[-1][1]
    plan.aligned_release = print_post_two_pin_aligned_insert_release_plan(aligned_pose, args)
    if not args.stop_after_post_two_pin_release_rotation:
        plan.after_rotation = print_post_two_pin_release_after_rotation_plan(
            plan.aligned_release[-1][1],
            args,
        )
    return plan


def plan_pin_sequence(
    start_pose: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray]]:
    """Shared dry-run and execution preflight, including the approach distance limit."""
    centering, approach = print_pin_approach_plan(start_pose, base_t_pin, args)
    distance_mm = 1000.0 * (
        np.linalg.norm(centering[:3] - start_pose[:3])
        + np.linalg.norm(approach[:3] - centering[:3])
    )
    if distance_mm > args.max_pin_approach_mm:
        raise RuntimeError(
            f"Pin approach is {distance_mm:.1f} mm, above max_pin_approach_mm "
            f"{args.max_pin_approach_mm:.1f}. Re-check the plan."
        )
    targets = [
        ("post-rotation pin centering", centering),
        ("post-rotation pin-axis approach", approach),
    ]
    correction = print_pin_image_alignment_correction_plan(approach, args)
    if correction is not None:
        targets.append(("post-approach image alignment correction", correction))
    if args.auto_pin_image_align:
        print("Live image alignment runs during execution; later targets will be recalculated.")
    if args.insert_after_pin_approach:
        insertion = print_pin_insertion_plan(targets[-1][1], base_t_pin, args)
        targets.append(("final pin insertion descend", insertion))
        targets.extend(plan_post_insert(insertion, base_t_pin, args).targets())
    return targets


def require_ik_solution(
    rtde_control: Any, q_near: np.ndarray, pose_ur: np.ndarray, motion_name: str
) -> np.ndarray:
    parameters = (
        pose_ur.tolist(),
        q_near.tolist(),
        IK_POSITION_TOLERANCE_M,
        IK_ORIENTATION_TOLERANCE_RAD,
    )
    if not rtde_control.getInverseKinematicsHasSolution(*parameters):
        raise RuntimeError(f"No IK solution exists for {motion_name}.")
    q = np.asarray(rtde_control.getInverseKinematics(*parameters), dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise RuntimeError(f"Invalid IK result for {motion_name}.")
    return q
