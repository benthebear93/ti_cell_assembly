#!/usr/bin/env python3
"""Visualize a saved ArUco marker frame with the UR5e mesh in viser."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import assembly_plan as motion
import numpy as np
from assembly_config import MotionConfig
from assembly_sequence import AssemblySequence, MotionEvent, build_assembly_sequence, pin_motion_events
from scipy.spatial.transform import Rotation, Slerp

from surface_estimator_ur5e.calibration import load_marker_transform
from surface_estimator_ur5e.io import DEFAULT_JOINT_ORDER
from surface_estimator_ur5e.live_robot import DEFAULT_ROBOT_IP
from surface_estimator_ur5e.robot_io import connect_rtde_control, set_tcp_offset
from surface_estimator_ur5e.robot_model import SimpleUR5eVisualizer, URDFRobotVisualizer
from surface_estimator_ur5e.transforms import (
    marker_relative_transform,
    rotation_transform,
    transform_points,
    transform_to_ur_pose,
    translation_transform,
    ur_pose_to_transform,
)
from surface_estimator_ur5e.visualization import _ceiling_mount_display_transform
from surface_estimator_ur5e.workcell_geometry import (
    DEFAULT_ASSEMBLY_LOCAL_YAW_DEG,
    DEFAULT_ASSEMBLY_ORIGIN_X_MM,
    DEFAULT_ASSEMBLY_ORIGIN_Y_MM,
    DEFAULT_ASSEMBLY_ORIGIN_Z_MM,
    DEFAULT_ASSEMBLY_RPY_DEG,
    DEFAULT_INITIAL_Q_DEG,
    DEFAULT_MARKER_POSE,
    DEFAULT_TCP_OFFSET_UR,
    DEFAULT_TI_ASSEMBLY_OBJ,
    DEFAULT_VIRTUAL_TRAY_HANDLE_ROOT_Y_MM,
    DEFAULT_VIRTUAL_TRAY_HANDLE_SCALE,
    DEFAULT_VIRTUAL_TRAY_LOCAL_RX_DEG,
    DEFAULT_VIRTUAL_TRAY_OBJ,
    DEFAULT_VIRTUAL_TRAY_TCP_Z_MM,
    TI_TRAY_FOUR_PIN_HOLE_CENTER_M,
    TI_TRAY_PROTRUSION_CENTER_M,
    floor_constrained_marker_transform,
    four_pin_feature_frame,
    shorten_lower_tray_handle,
    shortened_lower_handle_point,
)

MOTION_DEFAULTS = MotionConfig()
STAGE_COLORS = (
    (40, 120, 250), (50, 210, 100), (240, 130, 30), (120, 70, 230),
    (210, 50, 210), (40, 190, 190), (235, 60, 60),
)

# Keep the existing static viewer's tail geometry separate from execution settings.
DEFAULT_POST_TWO_PIN_RELEASE_FINAL_Y_MM = -150.0
DEFAULT_POST_TWO_PIN_RELEASE_ROTATE_X_DEG = -45.0
DEFAULT_POST_TWO_PIN_AFTER_CLOSE_Y_MM = -40.0
DEFAULT_POST_TWO_PIN_AFTER_CLOSE_FINAL_Z_MM = 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the saved marker frame, marker-relative target, and UR5e in viser."
    )
    parser.add_argument("--marker-pose", type=Path, default=DEFAULT_MARKER_POSE)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--offset-x-mm", type=float, default=MOTION_DEFAULTS.offset_x_mm)
    parser.add_argument("--offset-y-mm", type=float, default=MOTION_DEFAULTS.offset_y_mm)
    parser.add_argument(
        "--height-above-marker-mm",
        type=float,
        default=MOTION_DEFAULTS.height_above_marker_mm,
        help="Physical height above the floor marker, toward the ceiling.",
    )
    parser.add_argument(
        "--marker-frame-mode",
        choices=("floor", "raw"),
        default="floor",
        help="Frame used for the target. The floor-constrained marker frame is shown by default.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Also draw the raw ArUco frame and square for debugging.",
    )
    parser.add_argument(
        "--offset-z-mm",
        type=float,
        default=None,
        help="Raw marker z offset, only used with --marker-frame-mode raw.",
    )
    parser.add_argument(
        "--target-orientation",
        choices=("current", "marker"),
        default="current",
        help="Frame orientation to draw at the target point.",
    )
    parser.add_argument(
        "--show-plan-frames",
        "--show-coordinate-frames",
        dest="show_plan_frames",
        action="store_true",
        help="Draw the holder/tray frames used by marker_based_motion.",
    )
    parser.add_argument(
        "--show-assembly",
        "--show-holder",
        dest="show_assembly",
        action="store_true",
        help="Draw the holder OBJ in the marker frame.",
    )
    parser.add_argument("--assembly-obj", "--holder-obj", dest="assembly_obj", type=Path, default=DEFAULT_TI_ASSEMBLY_OBJ)
    parser.add_argument(
        "--assembly-origin-x-mm",
        "--holder-origin-x-mm",
        dest="assembly_origin_x_mm",
        type=float,
        default=DEFAULT_ASSEMBLY_ORIGIN_X_MM,
    )
    parser.add_argument(
        "--assembly-origin-y-mm",
        "--holder-origin-y-mm",
        dest="assembly_origin_y_mm",
        type=float,
        default=DEFAULT_ASSEMBLY_ORIGIN_Y_MM,
    )
    parser.add_argument(
        "--assembly-origin-z-mm",
        "--holder-origin-z-mm",
        dest="assembly_origin_z_mm",
        type=float,
        default=DEFAULT_ASSEMBLY_ORIGIN_Z_MM,
    )
    parser.add_argument(
        "--assembly-rpy-deg",
        "--holder-rpy-deg",
        dest="assembly_rpy_deg",
        nargs=3,
        type=float,
        default=DEFAULT_ASSEMBLY_RPY_DEG,
        metavar=("ROLL", "PITCH", "YAW"),
        help=(
            "OBJ orientation relative to the selected marker frame. The default maps CAD +Z "
            "to marker -Z so the OBJ z=0 bottom sits on the floor marker plane."
        ),
    )
    parser.add_argument(
        "--assembly-local-yaw-deg",
        "--holder-local-yaw-deg",
        dest="assembly_local_yaw_deg",
        type=float,
        default=DEFAULT_ASSEMBLY_LOCAL_YAW_DEG,
        help="Additional yaw about the OBJ local +Z axis after the upright floor alignment.",
    )
    parser.add_argument(
        "--show-virtual-tray",
        "--show-tray",
        dest="show_virtual_tray",
        action="store_true",
        help="Draw a grasped tray at a TCP-relative offset.",
    )
    parser.add_argument("--virtual-tray-obj", "--tray-obj", dest="virtual_tray_obj", type=Path, default=DEFAULT_VIRTUAL_TRAY_OBJ)
    parser.add_argument(
        "--virtual-tray-tcp-z-mm",
        "--tray-tcp-z-mm",
        dest="virtual_tray_tcp_z_mm",
        type=float,
        default=DEFAULT_VIRTUAL_TRAY_TCP_Z_MM,
        help="Tray center position along the current TCP local Z axis.",
    )
    parser.add_argument(
        "--virtual-tray-local-rx-deg",
        "--tray-local-rx-deg",
        dest="virtual_tray_local_rx_deg",
        type=float,
        default=DEFAULT_VIRTUAL_TRAY_LOCAL_RX_DEG,
        help="Rotation about the tray grasp frame local X axis.",
    )
    parser.add_argument(
        "--virtual-tray-handle-root-y-mm",
        "--tray-handle-root-y-mm",
        dest="virtual_tray_handle_root_y_mm",
        type=float,
        default=DEFAULT_VIRTUAL_TRAY_HANDLE_ROOT_Y_MM,
        help="Local tray Y location where the shortened lower handle starts.",
    )
    parser.add_argument(
        "--virtual-tray-handle-scale",
        "--tray-handle-scale",
        dest="virtual_tray_handle_scale",
        type=float,
        default=DEFAULT_VIRTUAL_TRAY_HANDLE_SCALE,
        help="Scale applied to tray vertices below the lower-handle root.",
    )
    parser.add_argument(
        "--tcp-rotation-offset-rpy-deg",
        nargs=3,
        type=float,
        default=MOTION_DEFAULTS.tcp_rotation_offset_rpy_deg,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Local offset from the holder four-pin frame to the desired TCP frame.",
    )
    parser.add_argument(
        "--rotation-approach-mm",
        type=float,
        default=MOTION_DEFAULTS.rotation_approach_mm,
        help="Motion-plan lateral approach during four-pin rotation.",
    )
    parser.add_argument(
        "--pin-approach-clearance-mm",
        type=float,
        default=MOTION_DEFAULTS.pin_approach_clearance_mm,
        help="Tray-reference distance along holder four-pin frame +Y after rotation.",
    )
    parser.add_argument(
        "--pin-image-alignment-metadata",
        type=Path,
        default=None,
        help=(
            "Draw the saved tcp_image_alignment_sweep correction after pin approach. "
            "When --insert-after-pin-approach is set and this is omitted, the same "
            "hardcoded metadata default as marker_based_motion is used."
        ),
    )
    parser.add_argument(
        "--post-two-pin-close-image-alignment-metadata",
        type=Path,
        default=None,
        help=(
            "Saved tcp_image_alignment_sweep metadata to draw after the final "
            "post-close TCP-local move. If omitted, uses the latest calibrated "
            "post-close metadata."
        ),
    )
    parser.add_argument(
        "--max-pin-image-correction-mm",
        type=float,
        default=MOTION_DEFAULTS.max_pin_image_correction_mm,
        help="Abort preview if the metadata translation correction exceeds this value.",
    )
    parser.add_argument(
        "--max-pin-image-correction-deg",
        type=float,
        default=MOTION_DEFAULTS.max_pin_image_correction_deg,
        help="Abort preview if the metadata rotation correction exceeds this value.",
    )
    parser.add_argument(
        "--insert-after-pin-approach",
        action="store_true",
        help="Draw the final insertion descend after optional image correction.",
    )
    parser.add_argument(
        "--pin-insertion-target-y-mm",
        type=float,
        default=MOTION_DEFAULTS.pin_insertion_target_y_mm,
        help="Final tray 4-hole center Y coordinate in the holder four-pin frame.",
    )
    parser.add_argument(
        "--max-pin-insertion-mm",
        type=float,
        default=MOTION_DEFAULTS.max_pin_insertion_mm,
        help="Abort preview if the final insertion translation exceeds this distance.",
    )
    parser.set_defaults(post_insert_release_retreat=True)
    parser.add_argument(
        "--post-insert-release-retreat",
        dest="post_insert_release_retreat",
        action="store_true",
        help="Draw the post-insertion gripper release and TCP-local retreat path.",
    )
    parser.add_argument(
        "--no-post-insert-release-retreat",
        dest="post_insert_release_retreat",
        action="store_false",
        help="Do not draw the post-insertion release/retreat path.",
    )
    parser.add_argument(
        "--post-insert-retreat-y-mm",
        type=float,
        default=MOTION_DEFAULTS.post_insert_retreat_y_mm,
        help="TCP-local +Y retreat after final insertion.",
    )
    parser.add_argument(
        "--post-insert-shift-x-mm",
        type=float,
        default=None,
        help=(
            "TCP-local +X shift after the +Y retreat. If omitted, compute the distance "
            "from the holder four-pin center to the adjacent two-pin center from the OBJ."
        ),
    )
    parser.set_defaults(post_insert_move_to_two_pin=True)
    parser.add_argument(
        "--post-insert-move-to-two-pin",
        dest="post_insert_move_to_two_pin",
        action="store_true",
        help="Draw the final TCP move into the holder adjacent two-pin target.",
    )
    parser.add_argument(
        "--no-post-insert-move-to-two-pin",
        dest="post_insert_move_to_two_pin",
        action="store_false",
        help="Do not draw the final TCP move into the adjacent two-pin target.",
    )
    parser.add_argument(
        "--post-insert-two-pin-target-y-mm",
        type=float,
        default=MOTION_DEFAULTS.post_insert_two_pin_target_y_mm,
        help="Holder pin-frame +Y offset applied to the final adjacent two-pin TCP target.",
    )
    parser.set_defaults(post_two_pin_close_retreat=True)
    parser.add_argument(
        "--post-two-pin-close-retreat",
        dest="post_two_pin_close_retreat",
        action="store_true",
        help="Draw TCP local +Y and -X moves after closing at the adjacent two-pin target.",
    )
    parser.add_argument(
        "--no-post-two-pin-close-retreat",
        dest="post_two_pin_close_retreat",
        action="store_false",
        help="Do not draw TCP-local moves after closing at the adjacent two-pin target.",
    )
    parser.add_argument(
        "--post-two-pin-close-retreat-y-mm",
        type=float,
        default=MOTION_DEFAULTS.post_two_pin_close_retreat_y_mm,
        help="TCP-local +Y move after closing at the adjacent two-pin target.",
    )
    parser.add_argument(
        "--post-two-pin-close-shift-negative-x-mm",
        type=float,
        default=MOTION_DEFAULTS.post_two_pin_close_shift_negative_x_mm,
        help="TCP-local -X move after the post-close +Y retreat.",
    )
    parser.set_defaults(post_two_pin_close_image_align=True)
    parser.add_argument(
        "--post-two-pin-close-image-align",
        dest="post_two_pin_close_image_align",
        action="store_true",
        help="Draw image alignment again after the final post-close TCP-local move. This is the default.",
    )
    parser.add_argument(
        "--no-post-two-pin-close-image-align",
        dest="post_two_pin_close_image_align",
        action="store_false",
        help="Stop preview after the final post-close TCP-local move without image alignment.",
    )
    parser.set_defaults(post_two_pin_aligned_insert_release=True)
    parser.add_argument(
        "--post-two-pin-aligned-insert-release",
        dest="post_two_pin_aligned_insert_release",
        action="store_true",
        help="Draw the final post-alignment TCP local -Y insertion, gripper release, +Y retreat, and Z shift.",
    )
    parser.add_argument(
        "--no-post-two-pin-aligned-insert-release",
        dest="post_two_pin_aligned_insert_release",
        action="store_false",
        help="Do not draw the final post-alignment insert/release retreat.",
    )
    parser.add_argument(
        "--post-two-pin-aligned-insert-y-mm",
        type=float,
        default=MOTION_DEFAULTS.post_two_pin_aligned_insert_y_mm,
        help="TCP-local -Y insertion after final post-two-pin image alignment.",
    )
    parser.add_argument(
        "--post-two-pin-release-retreat-y-mm",
        type=float,
        default=MOTION_DEFAULTS.post_two_pin_release_retreat_y_mm,
        help="TCP-local +Y retreat after opening at the final post-two-pin insertion.",
    )
    parser.add_argument(
        "--post-two-pin-release-shift-z-mm",
        type=float,
        default=MOTION_DEFAULTS.post_two_pin_release_shift_z_mm,
        help="TCP-local Z move after the final +Y release retreat.",
    )
    parser.add_argument(
        "--post-two-pin-release-final-y-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_RELEASE_FINAL_Y_MM,
        help="TCP-local Y move after the final release Z shift.",
    )
    parser.add_argument(
        "--post-two-pin-release-final-x-mm",
        type=float,
        default=MOTION_DEFAULTS.post_two_pin_release_final_x_mm,
        help="TCP-local X move sent together with the final release Y move.",
    )
    parser.add_argument(
        "--post-two-pin-release-rotate-x-deg",
        type=float,
        default=DEFAULT_POST_TWO_PIN_RELEASE_ROTATE_X_DEG,
        help="TCP-local X-axis rotation after the final release X/Y translation.",
    )
    parser.add_argument(
        "--post-two-pin-release-final-close-percent",
        type=float,
        default=MOTION_DEFAULTS.post_two_pin_release_final_close_percent,
        help="Gripper close percent after the final release X-axis rotation.",
    )
    parser.add_argument(
        "--post-two-pin-after-close-y-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_AFTER_CLOSE_Y_MM,
        help="TCP-local Y move after the final post-two-pin gripper close.",
    )
    parser.add_argument(
        "--post-two-pin-after-close-final-z-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_AFTER_CLOSE_FINAL_Z_MM,
        help="TCP-local Z move sent together with the final after-close Y move.",
    )
    parser.add_argument(
        "--show-motion-trajectory",
        "--show-traj",
        dest="show_motion_trajectory",
        action="store_true",
        help="Draw the same marker grasp, lift, rotation, and pin approach path used by marker_based_motion.",
    )
    parser.set_defaults(trajectory_start_from_initial_pose=True)
    parser.add_argument(
        "--trajectory-start-from-initial-pose",
        dest="trajectory_start_from_initial_pose",
        action="store_true",
        help=(
            "Use the saved marker_based_motion initial joint pose as the trajectory start. "
            "This calls RTDE getForwardKinematics only; it does not move the robot. This is the default."
        ),
    )
    parser.add_argument(
        "--trajectory-no-start-from-initial-pose",
        "--trajectory-current-start",
        dest="trajectory_start_from_initial_pose",
        action="store_false",
        help="Preview the trajectory from the current TCP instead of the saved initial joint pose.",
    )
    parser.add_argument(
        "--allow-approx-trajectory",
        action="store_true",
        help=(
            "Allow local URDF/DH FK fallback if controller FK is unavailable. "
            "Without this, trajectory preview fails rather than showing an approximate path."
        ),
    )
    parser.add_argument(
        "--initial-q-deg",
        nargs=6,
        type=float,
        default=DEFAULT_INITIAL_Q_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Initial joint pose used when --trajectory-start-from-initial-pose is set.",
    )
    parser.add_argument(
        "--post-grasp-lift-mm",
        type=float,
        default=MOTION_DEFAULTS.post_grasp_lift_mm,
        help="Trajectory preview lift along base -Z after grasp.",
    )
    parser.add_argument(
        "--safe-floor-clearance-mm",
        type=float,
        default=MOTION_DEFAULTS.safe_floor_clearance_mm,
        help="Trajectory preview tray bounding-box clearance above the marker floor.",
    )
    parser.add_argument(
        "--trajectory-step-mm",
        type=float,
        default=5.0,
        help="Maximum displayed trajectory sample spacing for translation.",
    )
    parser.add_argument(
        "--trajectory-rotation-step-deg",
        type=float,
        default=5.0,
        help="Maximum displayed trajectory sample spacing for rotation.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--tcp-offset-ur",
        nargs=6,
        type=float,
        default=DEFAULT_TCP_OFFSET_UR,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Active UR TCP offset used during hand-eye calibration.",
    )
    parser.add_argument(
        "--no-set-tcp",
        action="store_true",
        help="Do not call RTDEControl.setTcp before reading current robot state.",
    )
    parser.add_argument(
        "--q-deg",
        nargs=6,
        type=float,
        metavar=("BASE", "SHOULDER", "ELBOW", "WRIST1", "WRIST2", "WRIST3"),
        help="Offline joint angles in degrees. If omitted, reads current q from RTDE.",
    )
    parser.add_argument(
        "--tcp-pose-ur",
        nargs=6,
        type=float,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Offline TCP pose. If omitted, reads current TCP pose from RTDE.",
    )
    return parser.parse_args()


def read_robot_state(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.q_deg is not None:
        q_rad = np.deg2rad(np.asarray(args.q_deg, dtype=float))
        if args.tcp_pose_ur is None:
            tcp_pose_ur = np.zeros(6, dtype=float)
        else:
            tcp_pose_ur = np.asarray(args.tcp_pose_ur, dtype=float)
        return q_rad, tcp_pose_ur

    rtde_control = None
    try:
        if not args.no_set_tcp:
            rtde_control = connect_rtde_control(args.robot_ip)
            set_tcp_offset(rtde_control, args.tcp_offset_ur)

        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError("ur_rtde is not installed. Run 'uv sync' from the repo root.") from exc

    rtde_receive = RTDEReceiveInterface(args.robot_ip)
    try:
        q_rad = np.asarray(rtde_receive.getActualQ(), dtype=float)
        tcp_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    finally:
        if hasattr(rtde_receive, "disconnect"):
            rtde_receive.disconnect()
        if rtde_control is not None and hasattr(rtde_control, "stopScript"):
            rtde_control.stopScript()
        if rtde_control is not None and hasattr(rtde_control, "disconnect"):
            rtde_control.disconnect()

    if q_rad.shape != (6,):
        raise RuntimeError(f"Expected 6D joint vector from RTDE, got {q_rad.shape}.")
    if tcp_pose_ur.shape != (6,):
        raise RuntimeError(f"Expected 6D TCP pose from RTDE, got {tcp_pose_ur.shape}.")
    return q_rad, tcp_pose_ur


def initial_tcp_pose_from_rtde_fk(args: argparse.Namespace) -> np.ndarray:
    rtde_control = None
    try:
        rtde_control = connect_rtde_control(args.robot_ip)
    except Exception as exc:
        if args.allow_approx_trajectory:
            print(
                "warning: RTDEControl FK for initial pose unavailable "
                f"({exc}); using local URDF FK fallback because --allow-approx-trajectory was set."
            )
            return initial_tcp_pose_from_local_fk(args)
        raise RuntimeError(
            "Exact trajectory preview needs RTDEControl.getForwardKinematics for the saved "
            "initial joint pose, but RTDEControl could not connect. Re-run without "
            "--trajectory-start-from-initial-pose after moving the robot to the intended start, "
            "or pass --allow-approx-trajectory deliberately for a non-exact preview."
        ) from exc

    try:
        pose = np.asarray(
            rtde_control.getForwardKinematics(
                np.deg2rad(np.asarray(args.initial_q_deg, dtype=float)).tolist(),
                [float(value) for value in args.tcp_offset_ur],
            ),
            dtype=float,
        )
    except Exception as exc:
        if args.allow_approx_trajectory:
            print(
                "warning: RTDEControl FK for initial pose failed "
                f"({exc}); using local URDF FK fallback because --allow-approx-trajectory was set."
            )
            return initial_tcp_pose_from_local_fk(args)
        raise RuntimeError(
            "Exact trajectory preview needs RTDEControl.getForwardKinematics for the saved "
            "initial joint pose, but the FK call failed. Re-run after the robot control "
            "interface is available, or pass --allow-approx-trajectory deliberately for a "
            "non-exact preview."
        ) from exc
    finally:
        if rtde_control is not None and hasattr(rtde_control, "stopScript"):
            rtde_control.stopScript()
        if rtde_control is not None and hasattr(rtde_control, "disconnect"):
            rtde_control.disconnect()
    if pose.shape != (6,) or not np.all(np.isfinite(pose)):
        raise RuntimeError(f"Expected finite 6D FK pose for initial q, got {pose}.")
    return pose


def initial_tcp_pose_from_local_fk(args: argparse.Namespace) -> np.ndarray:
    q_rad = np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
    try:
        base_t_tool0 = URDFRobotVisualizer().link_transform_in_robot_base(
            q_rad,
            link_name="tool0",
            joint_order=list(DEFAULT_JOINT_ORDER),
        )
    except Exception as exc:
        print(f"warning: URDF FK unavailable ({exc}); using approximate DH FK fallback.")
        base_t_tool0 = SimpleUR5eVisualizer().forward_kinematics(q_rad)[-1]
    base_t_tcp = base_t_tool0 @ ur_pose_to_transform(np.asarray(args.tcp_offset_ur, dtype=float))
    return transform_to_ur_pose(base_t_tcp)


def connect_exact_preview_control(args: argparse.Namespace) -> Any:
    rtde_control = connect_rtde_control(args.robot_ip)
    if not args.no_set_tcp:
        set_tcp_offset(rtde_control, args.tcp_offset_ur)
    return rtde_control


def marker_target_offset(args: argparse.Namespace) -> np.ndarray:
    if args.marker_frame_mode == "raw":
        z_mm = args.offset_z_mm
        if z_mm is None:
            z_mm = args.height_above_marker_mm
        return np.asarray([args.offset_x_mm, args.offset_y_mm, z_mm], dtype=float) / 1000.0
    return np.asarray(
        [args.offset_x_mm, args.offset_y_mm, -args.height_above_marker_mm],
        dtype=float,
    ) / 1000.0


def target_transform(
    base_t_marker: np.ndarray,
    offset_marker_m: np.ndarray,
    current_tcp_pose_ur: np.ndarray,
    orientation: str,
) -> np.ndarray:
    marker_target = np.ones(4, dtype=float)
    marker_target[:3] = offset_marker_m

    base_t_target = np.eye(4)
    base_t_target[:3, 3] = (base_t_marker @ marker_target)[:3]
    if orientation == "marker":
        base_t_target[:3, :3] = base_t_marker[:3, :3]
    else:
        base_t_target[:3, :3] = Rotation.from_rotvec(current_tcp_pose_ur[3:6]).as_matrix()
    return base_t_target


def add_marker_square(server: object, name: str, transform: np.ndarray, side_m: float) -> None:
    scene = scene_handle(server)
    half = side_m / 2.0
    local_vertices = np.asarray(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=float,
    )
    vertices = (transform[:3, :3] @ local_vertices.T).T + transform[:3, 3]
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    if hasattr(scene, "add_mesh_simple"):
        try:
            scene.add_mesh_simple(
                name,
                vertices=vertices,
                faces=faces,
                color=(45, 45, 45),
                opacity=0.85,
                side="double",
            )
        except TypeError:
            scene.add_mesh_simple(name, vertices=vertices, faces=faces, color=(45, 45, 45))


def add_offset_segments(
    server: object,
    name: str,
    marker_transform: np.ndarray,
    offset_marker_m: np.ndarray,
) -> None:
    scene = scene_handle(server)
    if not hasattr(scene, "add_line_segments"):
        return

    points_marker = np.asarray(
        [
            [[0.0, 0.0, 0.0], [offset_marker_m[0], 0.0, 0.0]],
            [[offset_marker_m[0], 0.0, 0.0], [offset_marker_m[0], offset_marker_m[1], 0.0]],
            [
                [offset_marker_m[0], offset_marker_m[1], 0.0],
                [offset_marker_m[0], offset_marker_m[1], offset_marker_m[2]],
            ],
            [[0.0, 0.0, 0.0], offset_marker_m],
        ],
        dtype=float,
    )
    points_world = transform_points(marker_transform, points_marker.reshape(-1, 3)).reshape(-1, 2, 3)
    colors = np.asarray(
        [
            [[230, 60, 60], [230, 60, 60]],
            [[40, 190, 80], [40, 190, 80]],
            [[60, 120, 240], [60, 120, 240]],
            [[245, 180, 40], [245, 180, 40]],
        ],
        dtype=np.uint8,
    )
    scene.add_line_segments(name, points=points_world, colors=colors, line_width=4.0)


def add_point(server: object, name: str, point: np.ndarray, color: tuple[int, int, int], radius: float) -> None:
    scene = scene_handle(server)
    if hasattr(scene, "add_icosphere"):
        scene.add_icosphere(name, position=point, radius=radius, color=color)
    elif hasattr(scene, "add_point_cloud"):
        scene.add_point_cloud(
            name,
            points=np.asarray([point]),
            colors=np.asarray([color], dtype=np.uint8),
            point_size=2.0 * radius,
        )


def add_ti_assembly_mesh(
    server: object,
    obj_path: Path,
    transform: np.ndarray,
    *,
    frame_scale: float = 1.0,
    show_bounds: bool = True,
) -> np.ndarray:
    import trimesh

    if not obj_path.exists():
        raise FileNotFoundError(f"Holder OBJ not found: {obj_path}")

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    mesh.apply_transform(transform)
    scene = scene_handle(server)
    scene.add_mesh_trimesh("/ti_assembly/no_presser_cathode_tray", mesh=mesh)
    add_frame(
        server,
        "/frames/ti_assembly_obj_origin",
        transform,
        axes_length=0.08 * frame_scale,
    )
    add_point(
        server,
        "/ti_assembly/red_marked_origin",
        transform[:3, 3],
        (255, 20, 20),
        0.012 * frame_scale,
    )
    if show_bounds:
        add_mesh_bounds(server, "/ti_assembly/bounds", mesh.bounds)
    pin_frame, selected_pin_centers, adjacent_pin_centers = four_pin_feature_frame(obj_path)
    display_t_pin_frame = transform @ pin_frame
    add_frame(
        server,
        "/frames/four_pin_center",
        display_t_pin_frame,
        axes_length=0.04 * frame_scale,
    )
    add_pin_center_markers(
        server,
        transform_points(transform, selected_pin_centers),
        transform_points(transform, adjacent_pin_centers),
    )
    return bounds_corners(mesh.bounds)


def add_pin_center_markers(
    server: object,
    selected_pin_centers: np.ndarray,
    adjacent_pin_centers: np.ndarray,
) -> None:
    for index, center in enumerate(selected_pin_centers, start=1):
        add_point(server, f"/ti_assembly/four_pin_centers/pin_{index}", center, (40, 220, 90), 0.004)
    for index, center in enumerate(adjacent_pin_centers, start=1):
        add_point(server, f"/ti_assembly/adjacent_pin_centers/pin_{index}", center, (250, 160, 40), 0.004)

    scene = scene_handle(server)
    if hasattr(scene, "add_line_segments"):
        closed = np.vstack([selected_pin_centers, selected_pin_centers[0]])
        scene.add_line_segments(
            "/ti_assembly/four_pin_rectangle",
            points=np.stack([closed[:-1], closed[1:]], axis=1),
            colors=np.tile(np.asarray([40, 220, 90], dtype=np.uint8), (4, 2, 1)),
            line_width=3.0,
        )


def add_virtual_tray(
    server: object,
    obj_path: Path,
    display_t_tcp: np.ndarray,
    tcp_z_mm: float,
    local_rx_deg: float,
    handle_root_y_mm: float,
    handle_scale: float,
) -> np.ndarray:
    import trimesh

    if not obj_path.exists():
        raise FileNotFoundError(f"Virtual tray OBJ not found: {obj_path}")
    if not 0.0 < handle_scale <= 1.0:
        raise ValueError("--virtual-tray-handle-scale must be > 0 and <= 1.")

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    handle_root_y_m = handle_root_y_mm / 1000.0
    shorten_lower_tray_handle(mesh, handle_root_y_m, handle_scale)
    tray_attachment_local = shortened_lower_handle_point(
        TI_TRAY_PROTRUSION_CENTER_M,
        handle_root_y_m,
        handle_scale,
    )

    display_t_center = display_t_tcp @ translation_transform([0.0, 0.0, tcp_z_mm / 1000.0])
    center_t_tray = rotation_transform("x", local_rx_deg) @ translation_transform(
        -tray_attachment_local
    )
    mesh.apply_transform(center_t_tray)
    mesh.visual.face_colors = np.tile(np.array([205, 230, 145, 210], dtype=np.uint8), (len(mesh.faces), 1))

    scene = scene_handle(server)
    center_wxyz = Rotation.from_matrix(display_t_center[:3, :3]).as_quat()[[3, 0, 1, 2]]
    controls = scene.add_transform_controls(
        "/virtual_tray/adjustment",
        scale=0.08,
        line_width=3.0,
        position=display_t_center[:3, 3],
        wxyz=center_wxyz,
    )
    scene.add_mesh_trimesh("/virtual_tray/adjustment/short_handle_tray", mesh=mesh)
    scene.add_frame(
        "/virtual_tray/adjustment/center_frame",
        axes_length=0.04,
        axes_radius=0.0012,
    )
    tray_wxyz = Rotation.from_matrix(center_t_tray[:3, :3]).as_quat()[[3, 0, 1, 2]]
    scene.add_frame(
        "/virtual_tray/adjustment/obj_frame",
        position=center_t_tray[:3, 3],
        wxyz=tray_wxyz,
        axes_length=0.035,
        axes_radius=0.00105,
    )
    scene.add_icosphere(
        "/virtual_tray/adjustment/center",
        position=(0.0, 0.0, 0.0),
        radius=0.006,
        color=(245, 70, 210),
    )
    tray_four_hole_center_in_adjustment = (
        center_t_tray @ np.asarray([*TI_TRAY_FOUR_PIN_HOLE_CENTER_M, 1.0], dtype=float)
    )[:3]
    scene.add_icosphere(
        "/virtual_tray/adjustment/four_hole_center",
        position=tray_four_hole_center_in_adjustment,
        radius=0.007,
        color=(30, 220, 120),
    )
    add_point(server, "/virtual_tray/tcp_reference", display_t_tcp[:3, 3], (40, 120, 250), 0.005)
    add_mesh_bounds(server, "/virtual_tray/adjustment/bounds", mesh.bounds)

    @controls.on_drag_end
    def _(_) -> None:
        adjusted_t_center = np.eye(4)
        adjusted_t_center[:3, 3] = np.asarray(controls.position, dtype=float)
        adjusted_xyzw = np.asarray(controls.wxyz, dtype=float)[[1, 2, 3, 0]]
        adjusted_t_center[:3, :3] = Rotation.from_quat(adjusted_xyzw).as_matrix()
        initial_center_t_adjusted = np.linalg.inv(display_t_center) @ adjusted_t_center
        adjustment_rpy_deg = Rotation.from_matrix(
            initial_center_t_adjusted[:3, :3]
        ).as_euler("xyz", degrees=True)
        print(
            "Virtual tray visual adjustment relative to initial TCP attachment: "
            f"translation_mm={fmt(initial_center_t_adjusted[:3, 3] * 1000.0, decimals=3)}, "
            f"rpy_deg={fmt(adjustment_rpy_deg, decimals=3)}"
        )

    return transform_points(display_t_center, bounds_corners(mesh.bounds))


def add_line_segment(
    server: object,
    name: str,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    scene = scene_handle(server)
    if not hasattr(scene, "add_line_segments"):
        return
    scene.add_line_segments(
        name,
        points=np.asarray([[start, end]], dtype=float),
        colors=np.asarray([[color, color]], dtype=np.uint8),
        line_width=2.0,
    )


def add_polyline(
    server: object,
    name: str,
    points: np.ndarray,
    color: tuple[int, int, int],
    line_width: float = 2.0,
) -> None:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return
    scene = scene_handle(server)
    if hasattr(scene, "add_line_segments"):
        scene.add_line_segments(
            name,
            points=np.stack([points[:-1], points[1:]], axis=1),
            colors=np.tile(np.asarray(color, dtype=np.uint8), (len(points) - 1, 2, 1)),
            line_width=line_width,
        )
    if hasattr(scene, "add_point_cloud"):
        scene.add_point_cloud(
            f"{name}_samples",
            points=points,
            colors=np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1)),
            point_size=0.006,
        )


def transform_rotation_delta_deg(start: np.ndarray, target: np.ndarray) -> float:
    delta = Rotation.from_matrix(target[:3, :3] @ start[:3, :3].T)
    return float(np.rad2deg(delta.magnitude()))


def interpolate_transforms(
    start: np.ndarray,
    target: np.ndarray,
    translation_step_m: float,
    rotation_step_deg: float,
) -> list[np.ndarray]:
    start_t = np.asarray(start, dtype=float)
    target_t = np.asarray(target, dtype=float)
    translation_distance_m = float(np.linalg.norm(target_t[:3, 3] - start_t[:3, 3]))
    rotation_distance_deg = transform_rotation_delta_deg(start_t, target_t)
    segments = max(
        1,
        int(np.ceil(translation_distance_m / max(translation_step_m, 1e-6))),
        int(np.ceil(rotation_distance_deg / max(rotation_step_deg, 1e-6))),
    )
    fractions = np.linspace(0.0, 1.0, segments + 1)
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack([start_t[:3, :3], target_t[:3, :3]], axis=0)),
    )(fractions)

    samples: list[np.ndarray] = []
    for fraction, rotation in zip(fractions, rotations, strict=True):
        transform = np.eye(4)
        transform[:3, 3] = (1.0 - fraction) * start_t[:3, 3] + fraction * target_t[:3, 3]
        transform[:3, :3] = rotation.as_matrix()
        samples.append(transform)
    return samples


def marker_motion_args_from_visualization(args: argparse.Namespace) -> argparse.Namespace:
    """Use motion defaults, with the viewer's explicit settings and geometry names."""
    settings = asdict(MOTION_DEFAULTS)
    settings.update(vars(args))
    settings.update(
        start_from_initial_pose=args.trajectory_start_from_initial_pose,
        orientation=args.target_orientation,
        auto_pin_image_align=False,
        tray_obj=args.virtual_tray_obj,
        tray_center_tcp_z_mm=args.virtual_tray_tcp_z_mm,
        tray_local_rx_deg=args.virtual_tray_local_rx_deg,
        tray_handle_root_y_mm=args.virtual_tray_handle_root_y_mm,
        tray_handle_scale=args.virtual_tray_handle_scale,
    )
    return argparse.Namespace(**settings)


def preview_moves(events: list[MotionEvent], *, tray_attached: bool = False):
    """Track attachment from gripper events instead of a list of excluded stages."""
    for event in events:
        if event.tray_action is not None:
            tray_attached = event.tray_action == "attach"
        if event.kind == "move":
            yield event, tray_attached


def add_preview_endpoint(
    server: object, name: str, transform: np.ndarray, color: tuple[int, int, int]
) -> None:
    add_frame(server, name, transform, axes_length=0.055)
    add_point(server, name + "/tcp", transform[:3, 3], color, radius=0.005)


def preview_stage_name(index: int, name: str) -> str:
    # Some moves intentionally share a description; the index keeps scene paths unique.
    return f"{index:02d}_{name.replace(' ', '_').replace('/', '_')}"


def add_motion_plan_frames(
    server: object,
    display_transform: np.ndarray,
    base_t_marker: np.ndarray,
    base_t_tcp: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    """Draw rotation and pin targets from the current TCP, without calling IK."""
    motion_args = marker_motion_args_from_visualization(args)
    current_pose = transform_to_ur_pose(base_t_tcp)
    base_t_pin = motion.compute_base_t_four_pin_frame(motion_args, base_t_marker)
    rotation = motion.make_four_pin_rotation_target(
        current_pose,
        base_t_pin,
        motion_args.tcp_rotation_offset_rpy_deg,
    )
    rotation, approach_vector, pre_lateral, post_lateral = (
        motion.move_target_tray_center_toward_pin(
            rotation,
            base_t_pin,
            motion_args,
            motion_args.rotation_approach_mm,
        )
    )
    events = [MotionEvent("move", "four-pin rotation", rotation)]
    events.extend(pin_motion_events(rotation, base_t_pin, motion_args))

    add_frame(server, "/frames/holder_four_pin_motion_frame", display_transform @ base_t_pin, 0.07)
    current_reference = motion.tray_four_hole_center_position(current_pose, motion_args)
    last_reference = transform_points(display_transform, current_reference[None, :])[0]
    last_tcp = (display_transform @ base_t_tcp)[:3, 3]
    camera_points = [last_tcp, last_reference, (display_transform @ base_t_pin)[:3, 3]]
    add_point(server, "/motion_plan/current_tray_reference", last_reference, (210, 50, 210), 0.006)
    for index, (event, attached) in enumerate(preview_moves(events, tray_attached=True), start=1):
        name = "/motion_plan/" + preview_stage_name(index, event.name)
        color = STAGE_COLORS[(index - 1) % len(STAGE_COLORS)]
        display_target = display_transform @ ur_pose_to_transform(event.target_pose_ur)
        add_preview_endpoint(server, name, display_target, color)
        add_line_segment(server, name + "/tcp_move", last_tcp, display_target[:3, 3], color)
        last_tcp = display_target[:3, 3]
        camera_points.append(last_tcp)
        if attached:
            reference = motion.tray_four_hole_center_position(event.target_pose_ur, motion_args)
            display_reference = transform_points(display_transform, reference[None, :])[0]
            add_point(server, name + "/tray_reference", display_reference, color, 0.006)
            add_line_segment(server, name + "/tray_move", last_reference, display_reference, color)
            last_reference = display_reference
            camera_points.append(display_reference)
        print(f"  {index:02d} {event.name}: TCP base m {fmt(event.target_pose_ur[:3])}")

    if (
        args.insert_after_pin_approach
        and args.post_insert_release_retreat
        and args.post_insert_move_to_two_pin
    ):
        center = motion.adjacent_two_pin_center_base(motion_args, base_t_pin)
        display_center = transform_points(display_transform, center[None, :])[0]
        add_point(
            server, "/motion_plan/adjacent_two_pin_center", display_center, (30, 30, 30), 0.008
        )
        camera_points.append(display_center)
    print(f"  rotation approach base mm: {fmt(approach_vector * 1000, 3)}")
    print(
        f"  lateral distance before/after approach mm: {pre_lateral * 1000:.3f}/{post_lateral * 1000:.3f}"
    )
    return np.asarray(camera_points)


@dataclass(frozen=True)
class TrajectoryStage:
    name: str
    samples: list[np.ndarray]
    color: tuple[int, int, int]


@dataclass(frozen=True)
class MotionTrajectory:
    sequence: AssemblySequence
    stages: list[TrajectoryStage]
    tcp_points: np.ndarray
    tray_reference_points: np.ndarray


def build_motion_trajectory(
    base_t_marker: np.ndarray,
    start_tcp_pose_ur: np.ndarray,
    start_q_rad: np.ndarray,
    rtde_control: Any,
    args: argparse.Namespace,
) -> MotionTrajectory:
    motion_args = marker_motion_args_from_visualization(args)
    sequence = build_assembly_sequence(
        motion_args,
        rtde_control,
        base_t_marker,
        np.asarray(start_tcp_pose_ur, dtype=float),
        np.asarray(start_q_rad, dtype=float),
    )
    stages = []
    tcp_samples = []
    grasped_samples = []
    start = ur_pose_to_transform(start_tcp_pose_ur)
    q_near = sequence.grasp_q_rad
    for index, (event, attached) in enumerate(preview_moves(sequence.events)):
        # The grasp and lift/rotation have already passed their IK preflight.
        if index >= 3:
            q_near = motion.require_ik_solution(
                rtde_control, q_near, event.target_pose_ur, event.name
            )
        target = ur_pose_to_transform(event.target_pose_ur)
        samples = interpolate_transforms(
            start,
            target,
            args.trajectory_step_mm / 1000.0,
            args.trajectory_rotation_step_deg,
        )
        stages.append(TrajectoryStage(event.name, samples, STAGE_COLORS[index % len(STAGE_COLORS)]))
        tcp_samples.extend(samples if not tcp_samples else samples[1:])
        if attached:
            grasped_samples.extend(samples if not grasped_samples else samples[1:])
        start = target
    tray_reference_points = np.asarray(
        [
            motion.tray_four_hole_center_position(transform_to_ur_pose(sample), motion_args)
            for sample in grasped_samples
        ]
    ).reshape(-1, 3)
    return MotionTrajectory(
        sequence,
        stages,
        np.asarray([sample[:3, 3] for sample in tcp_samples]),
        tray_reference_points,
    )


def add_motion_trajectory(
    server: object,
    display_transform: np.ndarray,
    base_t_marker: np.ndarray,
    start_tcp_pose_ur: np.ndarray,
    start_q_rad: np.ndarray,
    rtde_control: Any,
    args: argparse.Namespace,
) -> np.ndarray:
    plan = build_motion_trajectory(
        base_t_marker, start_tcp_pose_ur, start_q_rad, rtde_control, args
    )
    tcp_points = transform_points(display_transform, plan.tcp_points)
    tray_points = transform_points(display_transform, plan.tray_reference_points)
    add_polyline(server, "/motion_trajectory/tcp_path", tcp_points, (40, 120, 250), line_width=3.0)
    add_polyline(
        server,
        "/motion_trajectory/tray_reference_path",
        tray_points,
        (210, 50, 210),
        line_width=4.0,
    )
    add_preview_endpoint(
        server,
        "/motion_trajectory/frames/00_start_tcp",
        display_transform @ ur_pose_to_transform(start_tcp_pose_ur),
        (40, 120, 250),
    )
    for index, stage in enumerate(plan.stages, start=1):
        name = preview_stage_name(index, stage.name)
        points = transform_points(
            display_transform, np.asarray([sample[:3, 3] for sample in stage.samples])
        )
        add_polyline(
            server, f"/motion_trajectory/stages/{name}", points, stage.color, line_width=2.5
        )
        add_preview_endpoint(
            server,
            f"/motion_trajectory/frames/{name}",
            display_transform @ stage.samples[-1],
            stage.color,
        )
        print(f"  {index:02d} {stage.name}: TCP base m {fmt(stage.samples[-1][:3, 3])}")
    alignment = plan.sequence.alignment
    print(f"  safety lift mm: {alignment['safety_lift_m'] * 1000:.3f}")
    print(
        "  tray floor clearance before/after rotation mm: "
        f"{alignment['current_clearance_m'] * 1000:.3f}/{alignment['target_clearance_m'] * 1000:.3f}"
    )
    report = plan.sequence.ik_report
    print(
        f"  IK minimum Q3/Q5 margin deg: {report['minimum_q3_margin_deg']:.3f}/"
        f"{report['minimum_q5_margin_deg']:.3f}; maximum joint step deg: {report['maximum_joint_step_deg']:.3f}"
    )
    return np.vstack(
        [tcp_points, tray_points, (display_transform @ plan.sequence.base_t_pin)[:3, 3]]
    )


def add_mesh_bounds(server: object, name: str, bounds: np.ndarray) -> None:
    scene = scene_handle(server)
    if not hasattr(scene, "add_line_segments"):
        return
    corners = bounds_corners(bounds)
    edge_indices = np.asarray(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=int,
    )
    segments = corners[edge_indices]
    scene.add_line_segments(
        name,
        points=segments,
        colors=np.tile(np.asarray([30, 30, 30], dtype=np.uint8), (len(segments), 2, 1)),
        line_width=1.5,
    )


def bounds_corners(bounds: np.ndarray) -> np.ndarray:
    lo, hi = np.asarray(bounds, dtype=float)
    return np.asarray(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=float,
    )


def add_frame(server: object, name: str, transform: np.ndarray, axes_length: float) -> None:
    scene = scene_handle(server)
    if not hasattr(scene, "add_frame"):
        return
    wxyz = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    scene.add_frame(
        name,
        wxyz=wxyz,
        position=transform[:3, 3],
        axes_length=axes_length,
        axes_radius=axes_length * 0.03,
    )


def add_base_plane(server: object, display_transform: np.ndarray, size_m: float = 1.2) -> None:
    scene = scene_handle(server)
    if not hasattr(scene, "add_mesh_simple"):
        return
    half = size_m / 2.0
    base_vertices = np.asarray(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=float,
    )
    vertices = transform_points(display_transform, base_vertices)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    try:
        scene.add_mesh_simple(
            "/workcell/base_z0_plane",
            vertices=vertices,
            faces=faces,
            color=(205, 210, 215),
            opacity=0.22,
            side="double",
        )
    except TypeError:
        scene.add_mesh_simple(
            "/workcell/base_z0_plane",
            vertices=vertices,
            faces=faces,
            color=(205, 210, 215),
        )


def set_camera(server: object, points: np.ndarray) -> None:
    center = np.mean(points, axis=0)
    extent = max(0.5, float(np.max(np.linalg.norm(points - center, axis=1))))
    position = center + np.asarray([0.7, -1.0, 0.55], dtype=float) * max(1.2, 2.2 * extent)

    if hasattr(server, "initial_camera"):
        server.initial_camera.position = tuple(position)
        server.initial_camera.look_at = tuple(center)
        if hasattr(server.initial_camera, "up"):
            server.initial_camera.up = (0.0, 0.0, 1.0)

    if hasattr(server, "on_client_connect"):
        @server.on_client_connect
        def _(client: object) -> None:
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(center)
            if hasattr(client.camera, "up"):
                client.camera.up = (0.0, 0.0, 1.0)


def scene_handle(server: object) -> object:
    return getattr(server, "scene", server)


def fmt(values: np.ndarray, decimals: int = 6) -> str:
    return "[" + ", ".join(f"{float(value): .{decimals}f}" for value in np.asarray(values).reshape(-1)) + "]"


def main() -> None:
    args = parse_args()
    marker_id, marker_length_m, base_t_marker_raw, marker_data = load_marker_transform(args.marker_pose)
    q_rad, current_tcp_pose_ur = read_robot_state(args)
    trajectory_start_tcp_pose_ur = (
        initial_tcp_pose_from_rtde_fk(args)
        if args.show_motion_trajectory and args.trajectory_start_from_initial_pose
        else current_tcp_pose_ur
    )
    trajectory_start_q_rad = (
        np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
        if args.show_motion_trajectory and args.trajectory_start_from_initial_pose
        else q_rad
    )
    trajectory_rtde_control = None

    base_t_marker_floor = floor_constrained_marker_transform(base_t_marker_raw)
    base_t_marker = base_t_marker_floor if args.marker_frame_mode == "floor" else base_t_marker_raw
    offset_marker_m = marker_target_offset(args)
    base_t_tcp = ur_pose_to_transform(current_tcp_pose_ur)
    base_t_trajectory_start_tcp = ur_pose_to_transform(trajectory_start_tcp_pose_ur)
    target_source_tcp_pose_ur = (
        trajectory_start_tcp_pose_ur
        if args.show_motion_trajectory
        else current_tcp_pose_ur
    )
    base_t_target = target_transform(
        base_t_marker,
        offset_marker_m,
        target_source_tcp_pose_ur,
        args.target_orientation,
    )

    display_transform = _ceiling_mount_display_transform()
    display_t_marker_raw = display_transform @ base_t_marker_raw
    display_t_marker_floor = display_transform @ base_t_marker_floor
    display_t_marker = display_transform @ base_t_marker
    display_t_tcp = display_transform @ base_t_tcp
    display_t_trajectory_start_tcp = display_transform @ base_t_trajectory_start_tcp
    display_t_target = display_transform @ base_t_target
    assembly_camera_points = np.empty((0, 3), dtype=float)
    virtual_tray_camera_points = np.empty((0, 3), dtype=float)
    motion_plan_camera_points = np.empty((0, 3), dtype=float)
    motion_trajectory_camera_points = np.empty((0, 3), dtype=float)

    import viser

    server = viser.ViserServer(port=args.port)
    add_base_plane(server, display_transform)
    add_frame(server, "/frames/robot_base", display_transform, axes_length=0.18)
    if args.show_raw:
        add_frame(server, f"/frames/aruco_{marker_id}_raw", display_t_marker_raw, axes_length=0.10)
    add_frame(server, f"/frames/aruco_{marker_id}_floor", display_t_marker_floor, axes_length=0.12)
    add_frame(server, "/frames/current_tcp", display_t_tcp, axes_length=0.08)
    if args.show_motion_trajectory:
        add_frame(
            server,
            "/frames/trajectory_start_tcp",
            display_t_trajectory_start_tcp,
            axes_length=0.07,
        )
    add_frame(server, "/frames/marker_offset_target", display_t_target, axes_length=0.08)
    if args.show_raw:
        add_marker_square(server, f"/markers/aruco_{marker_id}_raw_square", display_t_marker_raw, marker_length_m)
    add_marker_square(server, f"/markers/aruco_{marker_id}_floor_square", display_t_marker_floor, marker_length_m)
    add_offset_segments(server, "/markers/offset_segments", display_t_marker, offset_marker_m)
    add_point(server, "/markers/marker_origin", display_t_marker[:3, 3], (40, 40, 40), 0.012)
    add_point(server, "/targets/marker_offset_tcp", display_t_target[:3, 3], (250, 185, 40), 0.018)
    add_point(server, "/targets/current_tcp", display_t_tcp[:3, 3], (40, 120, 250), 0.014)
    if args.show_assembly:
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
        display_t_assembly = display_t_marker @ marker_t_assembly
        assembly_camera_points = add_ti_assembly_mesh(server, args.assembly_obj, display_t_assembly)
    if args.show_virtual_tray:
        virtual_tray_camera_points = add_virtual_tray(
            server,
            args.virtual_tray_obj,
            display_t_tcp,
            args.virtual_tray_tcp_z_mm,
            args.virtual_tray_local_rx_deg,
            args.virtual_tray_handle_root_y_mm,
            args.virtual_tray_handle_scale,
        )
    if args.show_plan_frames:
        motion_plan_camera_points = add_motion_plan_frames(
            server,
            display_transform,
            base_t_marker,
            base_t_tcp,
            args,
        )
    if args.show_motion_trajectory:
        try:
            trajectory_rtde_control = connect_exact_preview_control(args)
            motion_trajectory_camera_points = add_motion_trajectory(
                server,
                display_transform,
                base_t_marker,
                trajectory_start_tcp_pose_ur,
                trajectory_start_q_rad,
                trajectory_rtde_control,
                args,
            )
        finally:
            if trajectory_rtde_control is not None and hasattr(trajectory_rtde_control, "stopScript"):
                trajectory_rtde_control.stopScript()
            if trajectory_rtde_control is not None and hasattr(trajectory_rtde_control, "disconnect"):
                trajectory_rtde_control.disconnect()

    urdf_visualizer = URDFRobotVisualizer()
    try:
        robot_root_transform = urdf_visualizer.root_transform_for_robot_base(display_transform)
        urdf_visualizer.add_to_scene(
            server,
            q_rad,
            root_name="/ur5e_mesh",
            joint_order=list(DEFAULT_JOINT_ORDER),
            root_transform=robot_root_transform,
        )
        print("Loaded UR5e mesh from robot_descriptions/yourdfpy.")
    except Exception as exc:
        print(f"warning: URDF mesh visualization unavailable ({exc}); using simple UR5e sketch.")
        SimpleUR5eVisualizer().add_to_scene(
            server,
            q_rad,
            root_name="/ur5e_simple",
            joint_order=list(DEFAULT_JOINT_ORDER),
            root_transform=display_transform,
        )

    camera_points = np.vstack(
        [
            np.zeros(3),
            display_t_marker[:3, 3],
            display_t_target[:3, 3],
            display_t_tcp[:3, 3],
            assembly_camera_points,
            virtual_tray_camera_points,
            motion_plan_camera_points,
            motion_trajectory_camera_points,
        ]
    )
    set_camera(server, camera_points)

    capture_quality = marker_data.get("capture_quality", {})
    print("Marker frame visualization:")
    print(f"  marker pose: {args.marker_pose}")
    print(f"  marker_id: {marker_id}")
    print(f"  marker translation in base m: {fmt(base_t_marker[:3, 3])}")
    print(f"  marker length m: {marker_length_m:.6f}")
    print(f"  capture quality: {capture_quality}")
    print(f"  marker_frame_mode for target: {args.marker_frame_mode}")
    print(f"  tcp offset UR: {args.tcp_offset_ur}")
    print(f"  set tcp before visualization: {not args.no_set_tcp}")
    print(f"  offset marker mm: {fmt(offset_marker_m * 1000.0, decimals=3)}")
    print(f"  target position in base m: {fmt(base_t_target[:3, 3])}")
    print(f"  current TCP in base m: {fmt(base_t_tcp[:3, 3])}")
    if args.show_motion_trajectory:
        print(f"  trajectory start TCP in base m: {fmt(base_t_trajectory_start_tcp[:3, 3])}")
    if args.show_assembly:
        print(f"  holder OBJ: {args.assembly_obj}")
        print(
            "  holder OBJ origin in marker mm: "
            f"{fmt(np.asarray([args.assembly_origin_x_mm, args.assembly_origin_y_mm, args.assembly_origin_z_mm]), decimals=3)}"
        )
        print(f"  holder RPY in marker deg: {fmt(np.asarray(args.assembly_rpy_deg), decimals=3)}")
        print(f"  holder local yaw deg: {args.assembly_local_yaw_deg:.3f}")
    if args.show_virtual_tray:
        print(f"  tray OBJ: {args.virtual_tray_obj}")
        print(f"  tray adjustment frame in TCP z mm: {args.virtual_tray_tcp_z_mm:.3f}")
        print(f"  tray local rx deg: {args.virtual_tray_local_rx_deg:.3f}")
        print(
            "  tray shortened handle: "
            f"root_y={args.virtual_tray_handle_root_y_mm:.3f} mm, "
            f"scale={args.virtual_tray_handle_scale:.3f}"
        )
    if args.show_plan_frames:
        print(f"  motion frame rotation approach mm: {args.rotation_approach_mm:.3f}")
        print(f"  motion frame pin approach clearance mm: {args.pin_approach_clearance_mm:.3f}")
    print(f"Viser server is running on requested port {args.port}. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping visualization.")


if __name__ == "__main__":
    main()
