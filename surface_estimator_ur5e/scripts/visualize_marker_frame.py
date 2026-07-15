#!/usr/bin/env python3
"""Visualize a saved ArUco marker frame with the UR5e mesh in viser."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from surface_estimator_ur5e.io import DEFAULT_JOINT_ORDER
from surface_estimator_ur5e.live_robot import DEFAULT_ROBOT_IP
from surface_estimator_ur5e.robot_model import SimpleUR5eVisualizer, URDFRobotVisualizer
from surface_estimator_ur5e.visualization import _ceiling_mount_display_transform


DEFAULT_MARKER_POSE = Path("data/markers/aruco_364_in_base.yaml")
DEFAULT_OFFSET_X_MM = 144.0
DEFAULT_OFFSET_Y_MM = -50.5
DEFAULT_HEIGHT_ABOVE_MARKER_MM = 15.0
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
DEFAULT_ROTATION_APPROACH_MM = 50.0
DEFAULT_PIN_APPROACH_CLEARANCE_MM = 50.0
DEFAULT_MAX_PIN_IMAGE_CORRECTION_MM = 10.0
DEFAULT_MAX_PIN_IMAGE_CORRECTION_DEG = 10.0
DEFAULT_PIN_INSERTION_TARGET_Y_MM = 0.0
DEFAULT_MAX_PIN_INSERTION_MM = 80.0
DEFAULT_POST_INSERT_RETREAT_Y_MM = 50.0
DEFAULT_POST_INSERT_TWO_PIN_TARGET_Y_MM = 19.0
DEFAULT_POST_TWO_PIN_CLOSE_RETREAT_Y_MM = 30.0
DEFAULT_POST_TWO_PIN_CLOSE_SHIFT_NEGATIVE_X_MM = 58.0
DEFAULT_POST_TWO_PIN_ALIGNED_INSERT_Y_MM = 17.0
DEFAULT_POST_TWO_PIN_RELEASE_RETREAT_Y_MM = 150.0
DEFAULT_POST_TWO_PIN_RELEASE_SHIFT_Z_MM = -170.0
DEFAULT_POST_TWO_PIN_RELEASE_FINAL_X_MM = 20.0
DEFAULT_POST_TWO_PIN_RELEASE_FINAL_Y_MM = -150.0
DEFAULT_POST_TWO_PIN_RELEASE_ROTATE_X_DEG = -45.0
DEFAULT_POST_TWO_PIN_RELEASE_FINAL_CLOSE_PERCENT = 50.0
DEFAULT_POST_TWO_PIN_AFTER_CLOSE_Y_MM = -40.0
DEFAULT_POST_TWO_PIN_AFTER_CLOSE_FINAL_Y_MM = 30.0
DEFAULT_POST_TWO_PIN_AFTER_CLOSE_FINAL_Z_MM = 100.0
DEFAULT_TCP_ROTATION_OFFSET_RPY_DEG = (0.0, 0.0, 0.0)
DEFAULT_INITIAL_Q_DEG = [44.85, -15.96, -81.60, 7.72, 89.63, 135.01]
DEFAULT_POST_GRASP_LIFT_MM = 15.0
DEFAULT_SAFE_FLOOR_CLEARANCE_MM = 15.0
TI_TRAY_PROTRUSION_CENTER_M = np.array([0.0, -0.05150, 0.02750], dtype=float)
TI_TRAY_FOUR_PIN_HOLE_CENTER_M = np.array([0.0, 0.0, 0.01030], dtype=float)
BASE_Z_DOWN = np.array([0.0, 0.0, 1.0], dtype=float)
DEFAULT_TCP_OFFSET_UR = [0.0, 0.0, 0.158, -1.5707, 0.0, 0.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show the saved marker frame, marker-relative target, and UR5e in viser."
    )
    parser.add_argument("--marker-pose", type=Path, default=DEFAULT_MARKER_POSE)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--offset-x-mm", type=float, default=DEFAULT_OFFSET_X_MM)
    parser.add_argument("--offset-y-mm", type=float, default=DEFAULT_OFFSET_Y_MM)
    parser.add_argument(
        "--height-above-marker-mm",
        type=float,
        default=DEFAULT_HEIGHT_ABOVE_MARKER_MM,
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
        default=DEFAULT_TCP_ROTATION_OFFSET_RPY_DEG,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Local offset from the holder four-pin frame to the desired TCP frame.",
    )
    parser.add_argument(
        "--rotation-approach-mm",
        type=float,
        default=DEFAULT_ROTATION_APPROACH_MM,
        help="Motion-plan lateral approach during four-pin rotation.",
    )
    parser.add_argument(
        "--pin-approach-clearance-mm",
        type=float,
        default=DEFAULT_PIN_APPROACH_CLEARANCE_MM,
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
        default=DEFAULT_MAX_PIN_IMAGE_CORRECTION_MM,
        help="Abort preview if the metadata translation correction exceeds this value.",
    )
    parser.add_argument(
        "--max-pin-image-correction-deg",
        type=float,
        default=DEFAULT_MAX_PIN_IMAGE_CORRECTION_DEG,
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
        default=DEFAULT_PIN_INSERTION_TARGET_Y_MM,
        help="Final tray 4-hole center Y coordinate in the holder four-pin frame.",
    )
    parser.add_argument(
        "--max-pin-insertion-mm",
        type=float,
        default=DEFAULT_MAX_PIN_INSERTION_MM,
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
        default=DEFAULT_POST_INSERT_RETREAT_Y_MM,
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
        default=DEFAULT_POST_INSERT_TWO_PIN_TARGET_Y_MM,
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
        default=DEFAULT_POST_TWO_PIN_CLOSE_RETREAT_Y_MM,
        help="TCP-local +Y move after closing at the adjacent two-pin target.",
    )
    parser.add_argument(
        "--post-two-pin-close-shift-negative-x-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_CLOSE_SHIFT_NEGATIVE_X_MM,
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
        default=DEFAULT_POST_TWO_PIN_ALIGNED_INSERT_Y_MM,
        help="TCP-local -Y insertion after final post-two-pin image alignment.",
    )
    parser.add_argument(
        "--post-two-pin-release-retreat-y-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_RELEASE_RETREAT_Y_MM,
        help="TCP-local +Y retreat after opening at the final post-two-pin insertion.",
    )
    parser.add_argument(
        "--post-two-pin-release-shift-z-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_RELEASE_SHIFT_Z_MM,
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
        default=DEFAULT_POST_TWO_PIN_RELEASE_FINAL_X_MM,
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
        default=DEFAULT_POST_TWO_PIN_RELEASE_FINAL_CLOSE_PERCENT,
        help="Gripper close percent after the final release X-axis rotation.",
    )
    parser.add_argument(
        "--post-two-pin-after-close-y-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_AFTER_CLOSE_Y_MM,
        help="TCP-local Y move after the final post-two-pin gripper close.",
    )
    parser.add_argument(
        "--post-two-pin-after-close-final-y-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_AFTER_CLOSE_FINAL_Y_MM,
        help="TCP-local Y move sent together with the final after-close Z move.",
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
        default=DEFAULT_POST_GRASP_LIFT_MM,
        help="Trajectory preview lift along base -Z after grasp.",
    )
    parser.add_argument(
        "--safe-floor-clearance-mm",
        type=float,
        default=DEFAULT_SAFE_FLOOR_CLEARANCE_MM,
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


def load_marker_transform(path: Path) -> tuple[int | None, float, np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Marker pose file not found: {path}. Create it first with: "
            "uv run python scripts/read_aruco_marker_to_base.py "
            "--marker-length-mm 25.4 --marker-id 364 --save --save-samples 50"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    pose = data.get("pose_in_base")
    if not isinstance(pose, dict):
        raise ValueError(f"{path} must contain pose_in_base.")

    if "transform_matrix" in pose:
        base_t_marker = np.asarray(pose["transform_matrix"], dtype=float)
    else:
        translation = np.asarray(pose.get("translation_m"), dtype=float).reshape(3)
        if "rotation_quaternion_xyzw" in pose:
            rotation = Rotation.from_quat(pose["rotation_quaternion_xyzw"]).as_matrix()
        elif "rotation_matrix" in pose:
            rotation = np.asarray(pose["rotation_matrix"], dtype=float).reshape(3, 3)
        else:
            raise ValueError(f"{path} pose_in_base must include marker orientation.")
        base_t_marker = np.eye(4)
        base_t_marker[:3, :3] = rotation
        base_t_marker[:3, 3] = translation

    if base_t_marker.shape != (4, 4):
        raise ValueError(f"{path} transform_matrix must be 4x4.")
    if not np.all(np.isfinite(base_t_marker)):
        raise ValueError(f"{path} transform_matrix contains non-finite values.")

    marker_id = data.get("marker_id")
    marker_length_m = float(data.get("marker_length_m", 0.0254))
    return int(marker_id) if marker_id is not None else None, marker_length_m, base_t_marker, data


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


def ur_pose_to_transform(pose_ur: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(pose_ur[:3], dtype=float)
    transform[:3, :3] = Rotation.from_rotvec(pose_ur[3:6]).as_matrix()
    return transform


def transform_to_ur_pose(transform: np.ndarray) -> np.ndarray:
    pose = np.empty(6, dtype=float)
    pose[:3] = np.asarray(transform[:3, 3], dtype=float)
    pose[3:6] = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    return pose


def connect_rtde_control(robot_ip: str) -> Any:
    try:
        from rtde_control import RTDEControlInterface
    except ImportError as exc:
        raise RuntimeError(
            "ur_rtde control module is not installed. Run 'uv sync' from the repo root."
        ) from exc
    return RTDEControlInterface(robot_ip)


def set_tcp_offset(rtde_control: Any, tcp_offset_ur: list[float]) -> None:
    tcp_offset = [float(value) for value in tcp_offset_ur]
    if not rtde_control.setTcp(tcp_offset):
        raise RuntimeError(f"Failed to set active TCP offset: {tcp_offset}")


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


def marker_motion_args_from_visualization(args: argparse.Namespace) -> argparse.Namespace:
    import marker_based_motion as motion

    return argparse.Namespace(
        marker_pose=args.marker_pose,
        robot_ip=args.robot_ip,
        start_from_initial_pose=args.trajectory_start_from_initial_pose,
        initial_q_deg=args.initial_q_deg,
        offset_x_mm=args.offset_x_mm,
        offset_y_mm=args.offset_y_mm,
        height_above_marker_mm=args.height_above_marker_mm,
        marker_frame_mode=args.marker_frame_mode,
        offset_z_mm=args.offset_z_mm,
        orientation=args.target_orientation,
        assembly_obj=args.assembly_obj,
        assembly_origin_x_mm=args.assembly_origin_x_mm,
        assembly_origin_y_mm=args.assembly_origin_y_mm,
        assembly_origin_z_mm=args.assembly_origin_z_mm,
        assembly_rpy_deg=args.assembly_rpy_deg,
        assembly_local_yaw_deg=args.assembly_local_yaw_deg,
        tcp_rotation_offset_rpy_deg=args.tcp_rotation_offset_rpy_deg,
        post_grasp_lift_mm=args.post_grasp_lift_mm,
        safe_floor_clearance_mm=args.safe_floor_clearance_mm,
        max_safety_lift_mm=motion.DEFAULT_MAX_SAFETY_LIFT_MM,
        rotation_approach_mm=args.rotation_approach_mm,
        pin_approach_clearance_mm=args.pin_approach_clearance_mm,
        max_pin_approach_mm=motion.DEFAULT_MAX_PIN_APPROACH_MM,
        pin_image_alignment_metadata=args.pin_image_alignment_metadata,
        auto_pin_image_align=False,
        max_pin_image_correction_mm=args.max_pin_image_correction_mm,
        max_pin_image_correction_deg=args.max_pin_image_correction_deg,
        insert_after_pin_approach=args.insert_after_pin_approach,
        pin_insertion_target_y_mm=args.pin_insertion_target_y_mm,
        max_pin_insertion_mm=args.max_pin_insertion_mm,
        pin_insertion_speed_m_s=motion.DEFAULT_PIN_INSERTION_SPEED_M_S,
        pin_insertion_acceleration_m_s2=motion.DEFAULT_PIN_INSERTION_ACCELERATION_M_S2,
        post_insert_release_retreat=args.post_insert_release_retreat,
        post_insert_retreat_y_mm=args.post_insert_retreat_y_mm,
        post_insert_shift_x_mm=args.post_insert_shift_x_mm,
        post_insert_move_to_two_pin=args.post_insert_move_to_two_pin,
        post_insert_two_pin_target_y_mm=args.post_insert_two_pin_target_y_mm,
        post_two_pin_close_retreat=args.post_two_pin_close_retreat,
        post_two_pin_close_retreat_y_mm=args.post_two_pin_close_retreat_y_mm,
        post_two_pin_close_shift_negative_x_mm=args.post_two_pin_close_shift_negative_x_mm,
        post_two_pin_close_image_align=args.post_two_pin_close_image_align,
        post_two_pin_close_image_alignment_metadata=args.post_two_pin_close_image_alignment_metadata,
        post_two_pin_aligned_insert_release=args.post_two_pin_aligned_insert_release,
        post_two_pin_aligned_insert_y_mm=args.post_two_pin_aligned_insert_y_mm,
        post_two_pin_release_retreat_y_mm=args.post_two_pin_release_retreat_y_mm,
        post_two_pin_release_shift_z_mm=args.post_two_pin_release_shift_z_mm,
        post_two_pin_release_final_x_mm=args.post_two_pin_release_final_x_mm,
        post_two_pin_release_final_y_mm=args.post_two_pin_release_final_y_mm,
        post_two_pin_release_rotate_x_deg=args.post_two_pin_release_rotate_x_deg,
        post_two_pin_release_final_close_percent=args.post_two_pin_release_final_close_percent,
        post_two_pin_after_close_y_mm=args.post_two_pin_after_close_y_mm,
        post_two_pin_after_close_final_y_mm=args.post_two_pin_after_close_final_y_mm,
        post_two_pin_after_close_final_z_mm=args.post_two_pin_after_close_final_z_mm,
        tray_obj=args.virtual_tray_obj,
        tray_center_tcp_z_mm=args.virtual_tray_tcp_z_mm,
        tray_local_rx_deg=args.virtual_tray_local_rx_deg,
        tray_handle_root_y_mm=args.virtual_tray_handle_root_y_mm,
        tray_handle_scale=args.virtual_tray_handle_scale,
        tcp_offset_ur=args.tcp_offset_ur,
        no_set_tcp=args.no_set_tcp,
    )


def image_alignment_correction_pose_from_metadata(
    start_tcp_pose_ur: np.ndarray,
    motion_args: argparse.Namespace,
    metadata_path: Path | None = None,
) -> tuple[np.ndarray | None, Path | None]:
    import marker_based_motion as motion

    if metadata_path is None:
        metadata_path = motion.effective_pin_image_alignment_metadata(motion_args)
    if metadata_path is None:
        return None, None

    correction = motion.load_pin_image_alignment_correction(metadata_path)
    correction_translation_mm = float(
        np.linalg.norm(
            [
                correction["tcp_local_x_offset_mm"],
                correction["tcp_local_z_offset_mm"],
            ]
        )
    )
    if correction_translation_mm > motion_args.max_pin_image_correction_mm:
        raise RuntimeError(
            f"Image alignment correction is {correction_translation_mm:.3f} mm, above "
            f"--max-pin-image-correction-mm {motion_args.max_pin_image_correction_mm:.3f}."
        )
    if abs(float(correction["angle_deg"])) > motion_args.max_pin_image_correction_deg:
        raise RuntimeError(
            f"Image alignment correction is {correction['angle_deg']:.3f} deg, above "
            f"--max-pin-image-correction-deg {motion_args.max_pin_image_correction_deg:.3f}."
        )
    return motion.pose_with_tcp_local_alignment_correction(start_tcp_pose_ur, correction), metadata_path


def connect_exact_preview_control(args: argparse.Namespace) -> Any:
    rtde_control = connect_rtde_control(args.robot_ip)
    if not args.no_set_tcp:
        set_tcp_offset(rtde_control, args.tcp_offset_ur)
    return rtde_control


def floor_constrained_marker_transform(base_t_marker_raw: np.ndarray) -> np.ndarray:
    z_axis = BASE_Z_DOWN.copy()
    x_raw = np.asarray(base_t_marker_raw[:3, 0], dtype=float)
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-6:
        y_raw = np.asarray(base_t_marker_raw[:3, 1], dtype=float)
        x_axis = np.cross(y_raw, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    transform = np.eye(4)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis
    transform[:3, 3] = base_t_marker_raw[:3, 3]
    return transform


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


def marker_relative_transform(
    translation_m: np.ndarray,
    rpy_deg: np.ndarray,
    local_yaw_deg: float,
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(translation_m, dtype=float)
    marker_r_obj = Rotation.from_euler("xyz", np.asarray(rpy_deg, dtype=float), degrees=True).as_matrix()
    obj_r_yawed = Rotation.from_euler("z", float(local_yaw_deg), degrees=True).as_matrix()
    transform[:3, :3] = marker_r_obj @ obj_r_yawed
    return transform


def add_ti_assembly_mesh(server: object, obj_path: Path, transform: np.ndarray) -> np.ndarray:
    import trimesh

    if not obj_path.exists():
        raise FileNotFoundError(f"Holder OBJ not found: {obj_path}")

    mesh = trimesh.load(obj_path, force="mesh", process=False)
    mesh.apply_transform(transform)
    scene = scene_handle(server)
    scene.add_mesh_trimesh("/ti_assembly/no_presser_cathode_tray", mesh=mesh)
    add_frame(server, "/frames/ti_assembly_obj_origin", transform, axes_length=0.08)
    add_point(server, "/ti_assembly/red_marked_origin", transform[:3, 3], (255, 20, 20), 0.012)
    add_mesh_bounds(server, "/ti_assembly/bounds", mesh.bounds)
    pin_frame, selected_pin_centers, adjacent_pin_centers = four_pin_feature_frame(obj_path)
    display_t_pin_frame = transform @ pin_frame
    add_frame(server, "/frames/four_pin_center", display_t_pin_frame, axes_length=0.04)
    add_pin_center_markers(
        server,
        transform_points(transform, selected_pin_centers),
        transform_points(transform, adjacent_pin_centers),
    )
    return bounds_corners(mesh.bounds)


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
    frame[:3, :3] = frame[:3, :3] @ Rotation.from_euler(
        "x",
        FOUR_PIN_FRAME_LOCAL_RX_DEG,
        degrees=True,
    ).as_matrix()
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


def tray_center_position(tcp_transform: np.ndarray, tray_center_tcp_z_mm: float) -> np.ndarray:
    return (tcp_transform @ np.asarray([0.0, 0.0, tray_center_tcp_z_mm / 1000.0, 1.0]))[:3]


def tray_vertices_in_base(
    tcp_transform: np.ndarray,
    tray_vertices: np.ndarray,
    tray_attachment_local: np.ndarray,
    tray_center_tcp_z_mm: float,
    tray_local_rx_deg: float,
) -> np.ndarray:
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
    vertices_base = tray_vertices_in_base(
        tcp_transform,
        tray_vertices,
        tray_attachment_local,
        tray_center_tcp_z_mm=tray_center_tcp_z_mm,
        tray_local_rx_deg=tray_local_rx_deg,
    )
    return floor_z_m - float(np.max(vertices_base[:, 2]))


def holder_four_pin_transform(args: argparse.Namespace, base_t_marker: np.ndarray) -> np.ndarray:
    marker_t_holder = marker_relative_transform(
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
    holder_t_pin, _selected_pin_centers, _adjacent_pin_centers = four_pin_feature_frame(
        args.assembly_obj
    )
    return base_t_marker @ marker_t_holder @ holder_t_pin


def rotation_target_transform(
    base_t_tcp: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    pin_r_tcp_target = Rotation.from_euler(
        "xyz",
        np.asarray(args.tcp_rotation_offset_rpy_deg, dtype=float),
        degrees=True,
    ).as_matrix()

    target = np.asarray(base_t_tcp, dtype=float).copy()
    target[:3, :3] = base_t_pin[:3, :3] @ pin_r_tcp_target
    return move_target_tray_center_toward_pin(
        target,
        base_t_pin,
        args.virtual_tray_tcp_z_mm,
        args.rotation_approach_mm,
    )


def move_target_tray_center_toward_pin(
    target_t_tcp: np.ndarray,
    base_t_pin: np.ndarray,
    tray_center_tcp_z_mm: float,
    approach_mm: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    target = np.asarray(target_t_tcp, dtype=float).copy()
    requested_approach_m = float(approach_mm) / 1000.0
    if requested_approach_m <= 0.0:
        return target, np.zeros(3, dtype=float), 0.0, 0.0

    pin_axis = normalized(np.asarray(base_t_pin[:3, 2], dtype=float))
    tray_center = tray_center_position(target, tray_center_tcp_z_mm)
    lateral_delta = np.asarray(base_t_pin[:3, 3], dtype=float) - tray_center
    lateral_delta = lateral_delta - np.dot(lateral_delta, pin_axis) * pin_axis
    lateral_distance_m = float(np.linalg.norm(lateral_delta))
    if lateral_distance_m < 1e-9:
        return target, np.zeros(3, dtype=float), 0.0, 0.0

    approach_m = min(requested_approach_m, lateral_distance_m)
    approach_vector = lateral_delta / lateral_distance_m * approach_m
    target[:3, 3] += approach_vector
    return target, approach_vector, lateral_distance_m, lateral_distance_m - approach_m


def pin_approach_target_transform(
    start_t_tcp: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start = np.asarray(start_t_tcp, dtype=float)
    current_tray_center = tray_center_position(start, args.virtual_tray_tcp_z_mm)
    pin_rotation = np.asarray(base_t_pin[:3, :3], dtype=float)
    pin_origin = np.asarray(base_t_pin[:3, 3], dtype=float)
    target_tray_center_in_pin = np.asarray(
        [0.0, args.pin_approach_clearance_mm / 1000.0, 0.0],
        dtype=float,
    )
    target_tray_center = pin_origin + pin_rotation @ target_tray_center_in_pin
    translation = target_tray_center - current_tray_center
    target = start.copy()
    target[:3, 3] += translation
    return target, current_tray_center, target_tray_center, translation


def add_motion_plan_frames(
    server: object,
    display_transform: np.ndarray,
    base_t_marker: np.ndarray,
    base_t_tcp: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    import marker_based_motion as motion

    motion_args = marker_motion_args_from_visualization(args)
    current_tcp_pose_ur = transform_to_ur_pose(base_t_tcp)
    base_t_pin = motion.compute_base_t_four_pin_frame(motion_args, base_t_marker)
    rotation_pose = motion.make_four_pin_rotation_target(
        current_tcp_pose_ur,
        base_t_pin,
        motion_args.tcp_rotation_offset_rpy_deg,
    )
    (
        rotation_pose,
        rotation_approach_vector,
        pre_lateral_m,
        post_lateral_m,
    ) = motion.move_target_tray_center_toward_pin(
        rotation_pose,
        base_t_pin,
        motion_args,
        motion_args.rotation_approach_mm,
    )
    pin_centering_pose, rotation_tray_reference, pin_centering_reference, pin_centering_vector = (
        motion.pin_centering_target_pose(
            rotation_pose,
            base_t_pin,
            motion_args,
        )
    )
    pin_approach_pose, _pin_centered_reference, pin_target_reference, pin_approach_vector = (
        motion.pin_approach_target_pose(
            pin_centering_pose,
            base_t_pin,
            motion_args,
            motion_args.pin_approach_clearance_mm,
        )
    )
    planned_current_pose = pin_approach_pose
    image_correction_pose, pin_image_alignment_metadata = image_alignment_correction_pose_from_metadata(
        planned_current_pose,
        motion_args,
    )
    if image_correction_pose is not None:
        planned_current_pose = image_correction_pose

    insertion_pose = None
    insertion_target_reference = None
    insertion_vector = None
    insertion_target_reference_in_pin = None
    if motion_args.insert_after_pin_approach:
        (
            insertion_pose,
            _insertion_current_reference,
            insertion_target_reference,
            insertion_vector,
            _insertion_current_reference_in_pin,
            insertion_target_reference_in_pin,
        ) = motion.pin_insertion_target_pose(
            planned_current_pose,
            base_t_pin,
            motion_args,
            motion_args.pin_insertion_target_y_mm,
        )
        insertion_distance_mm = float(np.linalg.norm(insertion_vector) * 1000.0)
        if insertion_distance_mm > motion_args.max_pin_insertion_mm:
            raise RuntimeError(
                f"Motion-frame preview failed: final insertion is {insertion_distance_mm:.1f} mm, "
                f"above --max-pin-insertion-mm {motion_args.max_pin_insertion_mm:.1f}."
            )

    post_insert_retreat_y_pose = None
    post_insert_shift_x_pose = None
    post_insert_two_pin_pose = None
    post_insert_shift_x_mm = None
    adjacent_two_pin_center = None
    adjacent_two_pin_target = None
    adjacent_two_pin_translation = None
    post_two_pin_close_retreat_y_pose = None
    post_two_pin_close_shift_negative_x_pose = None
    post_two_pin_close_image_correction_pose = None
    post_two_pin_close_image_alignment_metadata = None
    post_two_pin_aligned_insert_pose = None
    post_two_pin_release_retreat_pose = None
    post_two_pin_release_shift_z_pose = None
    post_two_pin_release_final_y_pose = None
    post_two_pin_release_rotate_x_pose = None
    post_two_pin_after_close_y_pose = None
    post_two_pin_after_close_yz_pose = None
    if insertion_pose is not None and motion_args.post_insert_release_retreat:
        (
            post_insert_retreat_y_pose,
            post_insert_shift_x_pose,
            post_insert_shift_x_mm,
        ) = motion.post_insert_retreat_target_poses(insertion_pose, motion_args)
        if motion_args.post_insert_move_to_two_pin:
            (
                post_insert_two_pin_pose,
                adjacent_two_pin_center,
                adjacent_two_pin_target,
                adjacent_two_pin_translation,
            ) = motion.adjacent_two_pin_tcp_target_pose(
                post_insert_shift_x_pose,
                base_t_pin,
                motion_args,
            )
            if motion_args.post_two_pin_close_retreat:
                (
                    post_two_pin_close_retreat_y_pose,
                    post_two_pin_close_shift_negative_x_pose,
                ) = motion.post_two_pin_close_retreat_target_poses(
                    post_insert_two_pin_pose,
                    motion_args,
                )
                if motion_args.post_two_pin_close_image_align:
                    (
                        post_two_pin_close_image_correction_pose,
                        post_two_pin_close_image_alignment_metadata,
                    ) = image_alignment_correction_pose_from_metadata(
                        post_two_pin_close_shift_negative_x_pose,
                        motion_args,
                        metadata_path=motion.effective_post_two_pin_close_image_alignment_metadata(
                            motion_args
                        ),
                    )
                    if (
                        post_two_pin_close_image_correction_pose is not None
                        and motion_args.post_two_pin_aligned_insert_release
                    ):
                        (
                            post_two_pin_aligned_insert_pose,
                            post_two_pin_release_retreat_pose,
                            post_two_pin_release_shift_z_pose,
                            post_two_pin_release_final_y_pose,
                            post_two_pin_release_rotate_x_pose,
                        ) = motion.post_two_pin_aligned_insert_release_target_poses(
                            post_two_pin_close_image_correction_pose,
                            motion_args,
                        )
                        (
                            (_, post_two_pin_after_close_y_pose),
                            (_, post_two_pin_after_close_yz_pose),
                        ) = motion.post_two_pin_after_close_target_poses(
                            post_two_pin_release_rotate_x_pose,
                            motion_args,
                        )

    base_t_rotation = ur_pose_to_transform(rotation_pose)
    base_t_pin_centering = ur_pose_to_transform(pin_centering_pose)
    base_t_pin_approach = ur_pose_to_transform(pin_approach_pose)
    base_t_image_correction = (
        ur_pose_to_transform(image_correction_pose) if image_correction_pose is not None else None
    )
    base_t_insertion = ur_pose_to_transform(insertion_pose) if insertion_pose is not None else None
    base_t_post_insert_retreat_y = (
        ur_pose_to_transform(post_insert_retreat_y_pose)
        if post_insert_retreat_y_pose is not None
        else None
    )
    base_t_post_insert_shift_x = (
        ur_pose_to_transform(post_insert_shift_x_pose)
        if post_insert_shift_x_pose is not None
        else None
    )
    base_t_post_insert_two_pin = (
        ur_pose_to_transform(post_insert_two_pin_pose)
        if post_insert_two_pin_pose is not None
        else None
    )
    base_t_post_two_pin_close_retreat_y = (
        ur_pose_to_transform(post_two_pin_close_retreat_y_pose)
        if post_two_pin_close_retreat_y_pose is not None
        else None
    )
    base_t_post_two_pin_close_shift_negative_x = (
        ur_pose_to_transform(post_two_pin_close_shift_negative_x_pose)
        if post_two_pin_close_shift_negative_x_pose is not None
        else None
    )
    base_t_post_two_pin_close_image_correction = (
        ur_pose_to_transform(post_two_pin_close_image_correction_pose)
        if post_two_pin_close_image_correction_pose is not None
        else None
    )
    base_t_post_two_pin_aligned_insert = (
        ur_pose_to_transform(post_two_pin_aligned_insert_pose)
        if post_two_pin_aligned_insert_pose is not None
        else None
    )
    base_t_post_two_pin_release_retreat = (
        ur_pose_to_transform(post_two_pin_release_retreat_pose)
        if post_two_pin_release_retreat_pose is not None
        else None
    )
    base_t_post_two_pin_release_shift_z = (
        ur_pose_to_transform(post_two_pin_release_shift_z_pose)
        if post_two_pin_release_shift_z_pose is not None
        else None
    )
    base_t_post_two_pin_release_final_y = (
        ur_pose_to_transform(post_two_pin_release_final_y_pose)
        if post_two_pin_release_final_y_pose is not None
        else None
    )
    base_t_post_two_pin_release_rotate_x = (
        ur_pose_to_transform(post_two_pin_release_rotate_x_pose)
        if post_two_pin_release_rotate_x_pose is not None
        else None
    )
    base_t_post_two_pin_after_close_y = (
        ur_pose_to_transform(post_two_pin_after_close_y_pose)
        if post_two_pin_after_close_y_pose is not None
        else None
    )
    base_t_post_two_pin_after_close_yz = (
        ur_pose_to_transform(post_two_pin_after_close_yz_pose)
        if post_two_pin_after_close_yz_pose is not None
        else None
    )

    display_t_pin = display_transform @ base_t_pin
    display_t_rotation = display_transform @ base_t_rotation
    display_t_pin_centering = display_transform @ base_t_pin_centering
    display_t_pin_approach = display_transform @ base_t_pin_approach
    display_t_image_correction = (
        display_transform @ base_t_image_correction if base_t_image_correction is not None else None
    )
    display_t_insertion = display_transform @ base_t_insertion if base_t_insertion is not None else None
    display_t_post_insert_retreat_y = (
        display_transform @ base_t_post_insert_retreat_y
        if base_t_post_insert_retreat_y is not None
        else None
    )
    display_t_post_insert_shift_x = (
        display_transform @ base_t_post_insert_shift_x
        if base_t_post_insert_shift_x is not None
        else None
    )
    display_t_post_insert_two_pin = (
        display_transform @ base_t_post_insert_two_pin
        if base_t_post_insert_two_pin is not None
        else None
    )
    display_t_post_two_pin_close_retreat_y = (
        display_transform @ base_t_post_two_pin_close_retreat_y
        if base_t_post_two_pin_close_retreat_y is not None
        else None
    )
    display_t_post_two_pin_close_shift_negative_x = (
        display_transform @ base_t_post_two_pin_close_shift_negative_x
        if base_t_post_two_pin_close_shift_negative_x is not None
        else None
    )
    display_t_post_two_pin_close_image_correction = (
        display_transform @ base_t_post_two_pin_close_image_correction
        if base_t_post_two_pin_close_image_correction is not None
        else None
    )
    display_t_post_two_pin_aligned_insert = (
        display_transform @ base_t_post_two_pin_aligned_insert
        if base_t_post_two_pin_aligned_insert is not None
        else None
    )
    display_t_post_two_pin_release_retreat = (
        display_transform @ base_t_post_two_pin_release_retreat
        if base_t_post_two_pin_release_retreat is not None
        else None
    )
    display_t_post_two_pin_release_shift_z = (
        display_transform @ base_t_post_two_pin_release_shift_z
        if base_t_post_two_pin_release_shift_z is not None
        else None
    )
    display_t_post_two_pin_release_final_y = (
        display_transform @ base_t_post_two_pin_release_final_y
        if base_t_post_two_pin_release_final_y is not None
        else None
    )
    display_t_post_two_pin_release_rotate_x = (
        display_transform @ base_t_post_two_pin_release_rotate_x
        if base_t_post_two_pin_release_rotate_x is not None
        else None
    )
    display_t_post_two_pin_after_close_y = (
        display_transform @ base_t_post_two_pin_after_close_y
        if base_t_post_two_pin_after_close_y is not None
        else None
    )
    display_t_post_two_pin_after_close_yz = (
        display_transform @ base_t_post_two_pin_after_close_yz
        if base_t_post_two_pin_after_close_yz is not None
        else None
    )

    current_tray_reference = motion.tray_four_hole_center_position(
        current_tcp_pose_ur,
        motion_args,
    )
    pin_approach_tray_reference = motion.tray_four_hole_center_position(
        pin_approach_pose,
        motion_args,
    )
    image_correction_reference = (
        motion.tray_four_hole_center_position(image_correction_pose, motion_args)
        if image_correction_pose is not None
        else None
    )
    final_tray_reference = (
        insertion_target_reference
        if insertion_target_reference is not None
        else image_correction_reference
        if image_correction_reference is not None
        else pin_approach_tray_reference
    )
    display_current_tray_reference = transform_points(display_transform, np.asarray([current_tray_reference]))[0]
    display_rotation_tray_reference = transform_points(display_transform, np.asarray([rotation_tray_reference]))[0]
    display_pin_centering_reference = transform_points(display_transform, np.asarray([pin_centering_reference]))[0]
    display_pin_target_reference = transform_points(display_transform, np.asarray([pin_target_reference]))[0]
    display_pin_approach_tray_reference = transform_points(
        display_transform,
        np.asarray([pin_approach_tray_reference]),
    )[0]
    display_image_correction_reference = (
        transform_points(display_transform, np.asarray([image_correction_reference]))[0]
        if image_correction_reference is not None
        else None
    )
    display_final_tray_reference = transform_points(display_transform, np.asarray([final_tray_reference]))[0]
    display_adjacent_two_pin_center = (
        transform_points(display_transform, np.asarray([adjacent_two_pin_center]))[0]
        if adjacent_two_pin_center is not None
        else None
    )
    display_adjacent_two_pin_target = (
        transform_points(display_transform, np.asarray([adjacent_two_pin_target]))[0]
        if adjacent_two_pin_target is not None
        else None
    )

    add_frame(server, "/frames/holder_four_pin_motion_frame", display_t_pin, axes_length=0.07)
    add_frame(server, "/frames/rotation_approach_tcp_target", display_t_rotation, axes_length=0.065)
    add_frame(server, "/frames/pin_centering_tcp_target", display_t_pin_centering, axes_length=0.055)
    add_frame(server, "/frames/pin_approach_tcp_target", display_t_pin_approach, axes_length=0.065)
    if display_t_image_correction is not None:
        add_frame(
            server,
            "/frames/image_alignment_correction_tcp_target",
            display_t_image_correction,
            axes_length=0.055,
        )
    if display_t_insertion is not None:
        add_frame(server, "/frames/final_insertion_tcp_target", display_t_insertion, axes_length=0.055)
    if display_t_post_insert_retreat_y is not None:
        add_frame(server, "/frames/post_insert_y_retreat_tcp_target", display_t_post_insert_retreat_y, axes_length=0.05)
    if display_t_post_insert_shift_x is not None:
        add_frame(server, "/frames/post_insert_x_shift_tcp_target", display_t_post_insert_shift_x, axes_length=0.05)
    if display_t_post_insert_two_pin is not None:
        add_frame(server, "/frames/adjacent_two_pin_tcp_target", display_t_post_insert_two_pin, axes_length=0.055)
    if display_t_post_two_pin_close_retreat_y is not None:
        add_frame(
            server,
            "/frames/post_two_pin_close_y_retreat_tcp_target",
            display_t_post_two_pin_close_retreat_y,
            axes_length=0.05,
        )
    if display_t_post_two_pin_close_shift_negative_x is not None:
        add_frame(
            server,
            "/frames/post_two_pin_close_negative_x_shift_tcp_target",
            display_t_post_two_pin_close_shift_negative_x,
            axes_length=0.05,
        )
    if display_t_post_two_pin_close_image_correction is not None:
        add_frame(
            server,
            "/frames/post_two_pin_close_image_alignment_tcp_target",
            display_t_post_two_pin_close_image_correction,
            axes_length=0.05,
        )
    if display_t_post_two_pin_aligned_insert is not None:
        add_frame(
            server,
            "/frames/post_two_pin_aligned_insert_tcp_target",
            display_t_post_two_pin_aligned_insert,
            axes_length=0.05,
        )
    if display_t_post_two_pin_release_retreat is not None:
        add_frame(
            server,
            "/frames/post_two_pin_release_retreat_tcp_target",
            display_t_post_two_pin_release_retreat,
            axes_length=0.05,
        )
    if display_t_post_two_pin_release_shift_z is not None:
        add_frame(
            server,
            "/frames/post_two_pin_release_shift_z_tcp_target",
            display_t_post_two_pin_release_shift_z,
            axes_length=0.05,
        )
    if display_t_post_two_pin_release_final_y is not None:
        add_frame(
            server,
            "/frames/post_two_pin_release_final_y_tcp_target",
            display_t_post_two_pin_release_final_y,
            axes_length=0.05,
        )
    if display_t_post_two_pin_release_rotate_x is not None:
        add_frame(
            server,
            "/frames/post_two_pin_release_rotate_x_tcp_target",
            display_t_post_two_pin_release_rotate_x,
            axes_length=0.05,
        )
    if display_t_post_two_pin_after_close_y is not None:
        add_frame(
            server,
            "/frames/post_two_pin_after_close_y_tcp_target",
            display_t_post_two_pin_after_close_y,
            axes_length=0.05,
        )
    if display_t_post_two_pin_after_close_yz is not None:
        add_frame(
            server,
            "/frames/post_two_pin_after_close_yz_tcp_target",
            display_t_post_two_pin_after_close_yz,
            axes_length=0.05,
        )

    add_point(server, "/motion_frames/current_tray_4hole_center", display_current_tray_reference, (40, 120, 250), 0.008)
    add_point(server, "/motion_frames/rotation_tray_4hole_center", display_rotation_tray_reference, (240, 130, 30), 0.008)
    add_point(server, "/motion_frames/pin_centering_tray_4hole_center", display_pin_centering_reference, (120, 70, 230), 0.008)
    add_point(server, "/motion_frames/pin_target_tray_4hole_center", display_pin_target_reference, (50, 210, 100), 0.008)
    add_point(server, "/motion_frames/pin_approach_tray_4hole_center", display_pin_approach_tray_reference, (210, 50, 210), 0.008)
    if display_image_correction_reference is not None:
        add_point(
            server,
            "/motion_frames/image_corrected_tray_4hole_center",
            display_image_correction_reference,
            (40, 190, 190),
            0.008,
        )
    if display_adjacent_two_pin_target is not None:
        add_point(
            server,
            "/motion_frames/adjacent_two_pin_target_y_offset",
            display_adjacent_two_pin_target,
            (0, 0, 0),
            0.009,
        )
    add_point(server, "/motion_frames/final_tray_4hole_center", display_final_tray_reference, (235, 60, 60), 0.008)
    if display_adjacent_two_pin_center is not None:
        add_point(
            server,
            "/motion_frames/adjacent_two_pin_center",
            display_adjacent_two_pin_center,
            (20, 20, 20),
            0.008,
        )
    add_line_segment(
        server,
        "/motion_frames/current_to_rotation_tray_4hole_center",
        display_current_tray_reference,
        display_rotation_tray_reference,
        (240, 130, 30),
    )
    add_line_segment(
        server,
        "/motion_frames/rotation_to_pin_centering_tray_4hole_center",
        display_rotation_tray_reference,
        display_pin_centering_reference,
        (120, 70, 230),
    )
    add_line_segment(
        server,
        "/motion_frames/pin_centering_to_pin_approach_tray_4hole_center",
        display_pin_centering_reference,
        display_pin_approach_tray_reference,
        (210, 50, 210),
    )
    if display_image_correction_reference is not None:
        add_line_segment(
            server,
            "/motion_frames/pin_approach_to_image_corrected_tray_4hole_center",
            display_pin_approach_tray_reference,
            display_image_correction_reference,
            (40, 190, 190),
        )
    if insertion_target_reference is not None:
        add_line_segment(
            server,
            "/motion_frames/image_corrected_to_final_inserted_tray_4hole_center",
            display_image_correction_reference
            if display_image_correction_reference is not None
            else display_pin_approach_tray_reference,
            display_final_tray_reference,
            (235, 60, 60),
        )
    add_line_segment(
        server,
        "/motion_frames/pin_center_to_target_clearance",
        display_t_pin[:3, 3],
        display_pin_target_reference,
        (50, 210, 100),
    )
    if display_t_post_insert_retreat_y is not None and display_t_insertion is not None:
        add_line_segment(
            server,
            "/motion_frames/post_insert_tcp_to_y_retreat",
            display_t_insertion[:3, 3],
            display_t_post_insert_retreat_y[:3, 3],
            (90, 90, 90),
        )
    if display_t_post_insert_shift_x is not None and display_t_post_insert_retreat_y is not None:
        add_line_segment(
            server,
            "/motion_frames/post_insert_y_retreat_to_x_shift",
            display_t_post_insert_retreat_y[:3, 3],
            display_t_post_insert_shift_x[:3, 3],
            (70, 70, 70),
        )
    if display_t_post_insert_two_pin is not None and display_t_post_insert_shift_x is not None:
        add_line_segment(
            server,
            "/motion_frames/post_insert_x_shift_to_adjacent_two_pin",
            display_t_post_insert_shift_x[:3, 3],
            display_t_post_insert_two_pin[:3, 3],
            (20, 20, 20),
        )
    if display_t_post_two_pin_close_retreat_y is not None and display_t_post_insert_two_pin is not None:
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_close_tcp_to_y_retreat",
            display_t_post_insert_two_pin[:3, 3],
            display_t_post_two_pin_close_retreat_y[:3, 3],
            (80, 80, 80),
        )
    if (
        display_t_post_two_pin_close_shift_negative_x is not None
        and display_t_post_two_pin_close_retreat_y is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_close_y_retreat_to_negative_x_shift",
            display_t_post_two_pin_close_retreat_y[:3, 3],
            display_t_post_two_pin_close_shift_negative_x[:3, 3],
            (30, 30, 30),
        )
    if (
        display_t_post_two_pin_close_image_correction is not None
        and display_t_post_two_pin_close_shift_negative_x is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_close_negative_x_shift_to_image_alignment",
            display_t_post_two_pin_close_shift_negative_x[:3, 3],
            display_t_post_two_pin_close_image_correction[:3, 3],
            (40, 190, 190),
        )
    if (
        display_t_post_two_pin_aligned_insert is not None
        and display_t_post_two_pin_close_image_correction is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_close_image_alignment_to_aligned_insert",
            display_t_post_two_pin_close_image_correction[:3, 3],
            display_t_post_two_pin_aligned_insert[:3, 3],
            (235, 60, 60),
        )
    if (
        display_t_post_two_pin_release_retreat is not None
        and display_t_post_two_pin_aligned_insert is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_aligned_insert_to_release_retreat",
            display_t_post_two_pin_aligned_insert[:3, 3],
            display_t_post_two_pin_release_retreat[:3, 3],
            (80, 80, 80),
        )
    if (
        display_t_post_two_pin_release_shift_z is not None
        and display_t_post_two_pin_release_retreat is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_release_retreat_to_shift_z",
            display_t_post_two_pin_release_retreat[:3, 3],
            display_t_post_two_pin_release_shift_z[:3, 3],
            (90, 90, 90),
        )
    if (
        display_t_post_two_pin_release_final_y is not None
        and display_t_post_two_pin_release_shift_z is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_release_shift_z_to_final_y",
            display_t_post_two_pin_release_shift_z[:3, 3],
            display_t_post_two_pin_release_final_y[:3, 3],
            (70, 70, 70),
        )
    if (
        display_t_post_two_pin_release_rotate_x is not None
        and display_t_post_two_pin_release_final_y is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_release_final_y_to_rotate_x",
            display_t_post_two_pin_release_final_y[:3, 3],
            display_t_post_two_pin_release_rotate_x[:3, 3],
            (120, 120, 120),
        )
    if (
        display_t_post_two_pin_after_close_y is not None
        and display_t_post_two_pin_release_rotate_x is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_release_rotate_x_to_after_close_y",
            display_t_post_two_pin_release_rotate_x[:3, 3],
            display_t_post_two_pin_after_close_y[:3, 3],
            (80, 120, 120),
        )
    if (
        display_t_post_two_pin_after_close_yz is not None
        and display_t_post_two_pin_after_close_y is not None
    ):
        add_line_segment(
            server,
            "/motion_frames/post_two_pin_after_close_y_to_yz",
            display_t_post_two_pin_after_close_y[:3, 3],
            display_t_post_two_pin_after_close_yz[:3, 3],
            (60, 120, 160),
        )

    print("Motion-plan coordinate frames:")
    print(f"  holder four-pin frame base position m: {fmt(base_t_pin[:3, 3])}")
    print(f"  current tray 4-hole center base m: {fmt(current_tray_reference)}")
    print(
        "  current tray 4-hole center in pin frame mm: "
        f"{fmt(base_t_pin[:3, :3].T @ (current_tray_reference - base_t_pin[:3, 3]) * 1000.0, decimals=3)}"
    )
    print(
        "  rotation approach vector base mm: "
        f"{fmt(rotation_approach_vector * 1000.0, decimals=3)}"
    )
    print(
        "  tray 4-hole center lateral distance to four-pin center mm: "
        f"{pre_lateral_m * 1000.0:.3f} -> {post_lateral_m * 1000.0:.3f}"
    )
    print(f"  rotation target TCP pose base position m: {fmt(base_t_rotation[:3, 3])}")
    print(
        "  pin centering vector base mm: "
        f"{fmt(pin_centering_vector * 1000.0, decimals=3)}"
    )
    print(
        "  pin approach tray 4-hole target in pin frame mm: "
        f"{fmt(base_t_pin[:3, :3].T @ (pin_target_reference - base_t_pin[:3, 3]) * 1000.0, decimals=3)}"
    )
    print(f"  pin approach tray 4-hole target base m: {fmt(pin_target_reference)}")
    print(
        "  pin approach vector base mm: "
        f"{fmt(pin_approach_vector * 1000.0, decimals=3)}"
    )
    if pin_image_alignment_metadata is not None:
        print(f"  image alignment metadata: {pin_image_alignment_metadata}")
    if image_correction_reference is not None:
        print(
            "  image-corrected tray 4-hole center in pin frame mm: "
            f"{fmt(base_t_pin[:3, :3].T @ (image_correction_reference - base_t_pin[:3, 3]) * 1000.0, decimals=3)}"
        )
    if insertion_vector is not None:
        print(
            "  final insertion vector base mm: "
            f"{fmt(insertion_vector * 1000.0, decimals=3)}"
        )
        print(
            "  final inserted tray 4-hole center in pin frame mm: "
            f"{fmt(insertion_target_reference_in_pin * 1000.0, decimals=3)}"
        )
    if post_insert_shift_x_mm is not None:
        print(f"  post-insert TCP local +X shift mm: {post_insert_shift_x_mm:.3f}")
    if adjacent_two_pin_center is not None:
        adjacent_two_pin_center_in_pin = base_t_pin[:3, :3].T @ (
            adjacent_two_pin_center - base_t_pin[:3, 3]
        )
        adjacent_two_pin_target_in_pin = base_t_pin[:3, :3].T @ (
            adjacent_two_pin_target - base_t_pin[:3, 3]
        )
        print(
            "  adjacent two-pin center in pin frame mm: "
            f"{fmt(1000.0 * adjacent_two_pin_center_in_pin, decimals=3)}"
        )
        print(
            "  adjacent two-pin target in pin frame mm: "
            f"{fmt(1000.0 * adjacent_two_pin_target_in_pin, decimals=3)}"
        )
        print(f"  adjacent two-pin target +Y offset mm: {args.post_insert_two_pin_target_y_mm:.3f}")
        print(f"  adjacent two-pin center base m: {fmt(adjacent_two_pin_center)}")
        print(f"  adjacent two-pin TCP target base m: {fmt(adjacent_two_pin_target)}")
        print(
            "  adjacent two-pin move translation base mm: "
            f"{fmt(1000.0 * adjacent_two_pin_translation, decimals=3)}"
        )
    if base_t_post_two_pin_close_retreat_y is not None:
        print(f"  post-two-pin-close TCP local +Y retreat mm: {args.post_two_pin_close_retreat_y_mm:.3f}")
        print(
            "  post-two-pin-close +Y TCP target base m: "
            f"{fmt(base_t_post_two_pin_close_retreat_y[:3, 3])}"
        )
    if base_t_post_two_pin_close_shift_negative_x is not None:
        print(
            "  post-two-pin-close TCP local -X shift mm: "
            f"{args.post_two_pin_close_shift_negative_x_mm:.3f}"
        )
        print(
            "  post-two-pin-close -X TCP target base m: "
            f"{fmt(base_t_post_two_pin_close_shift_negative_x[:3, 3])}"
        )
    if base_t_post_two_pin_close_image_correction is not None:
        print(
            "  post-two-pin-close image alignment metadata: "
            f"{post_two_pin_close_image_alignment_metadata}"
        )
        print(
            "  post-two-pin-close image alignment TCP target base m: "
            f"{fmt(base_t_post_two_pin_close_image_correction[:3, 3])}"
        )
    if base_t_post_two_pin_aligned_insert is not None:
        print(
            "  post-two-pin aligned TCP local -Y insertion mm: "
            f"{args.post_two_pin_aligned_insert_y_mm:.3f}"
        )
        print(
            "  post-two-pin aligned insertion TCP target base m: "
            f"{fmt(base_t_post_two_pin_aligned_insert[:3, 3])}"
        )
    if base_t_post_two_pin_release_retreat is not None:
        print(
            "  post-two-pin release TCP local +Y retreat mm: "
            f"{args.post_two_pin_release_retreat_y_mm:.3f}"
        )
        print(
            "  post-two-pin release retreat TCP target base m: "
            f"{fmt(base_t_post_two_pin_release_retreat[:3, 3])}"
        )
    if base_t_post_two_pin_release_shift_z is not None:
        print(
            "  post-two-pin release TCP local Z shift mm: "
            f"{args.post_two_pin_release_shift_z_mm:.3f}"
        )
        print(
            "  post-two-pin release Z-shift TCP target base m: "
            f"{fmt(base_t_post_two_pin_release_shift_z[:3, 3])}"
        )
    if base_t_post_two_pin_release_final_y is not None:
        print(
            "  post-two-pin release TCP local final X/Y shift mm: "
            f"{args.post_two_pin_release_final_x_mm:.3f}/"
            f"{args.post_two_pin_release_final_y_mm:.3f}"
        )
        print(
            "  post-two-pin release final-X/Y TCP target base m: "
            f"{fmt(base_t_post_two_pin_release_final_y[:3, 3])}"
        )
    if base_t_post_two_pin_release_rotate_x is not None:
        print(
            "  post-two-pin release TCP local X rotation deg: "
            f"{args.post_two_pin_release_rotate_x_deg:.3f}"
        )
        print(
            "  post-two-pin release rotate-X TCP target base m: "
            f"{fmt(base_t_post_two_pin_release_rotate_x[:3, 3])}"
        )
    if base_t_post_two_pin_after_close_y is not None:
        print(
            "  post-two-pin after-close TCP local Y shift mm: "
            f"{args.post_two_pin_after_close_y_mm:.3f}"
        )
        print(
            "  post-two-pin after-close Y-shift TCP target base m: "
            f"{fmt(base_t_post_two_pin_after_close_y[:3, 3])}"
        )
    if base_t_post_two_pin_after_close_yz is not None:
        print(
            "  post-two-pin after-close TCP local final Y/Z shift mm: "
            f"{args.post_two_pin_after_close_final_y_mm:.3f}/"
            f"{args.post_two_pin_after_close_final_z_mm:.3f}"
        )
        print(
            "  post-two-pin after-close final-Y/Z TCP target base m: "
            f"{fmt(base_t_post_two_pin_after_close_yz[:3, 3])}"
        )
    print(f"  pin approach TCP target base position m: {fmt(base_t_pin_approach[:3, 3])}")

    camera_points = [
        base_t_pin[:3, 3],
        current_tray_reference,
        rotation_tray_reference,
        pin_centering_reference,
        pin_target_reference,
        pin_approach_tray_reference,
        final_tray_reference,
        base_t_rotation[:3, 3],
        base_t_pin_centering[:3, 3],
        base_t_pin_approach[:3, 3],
    ]
    if image_correction_reference is not None:
        camera_points.append(image_correction_reference)
    if base_t_image_correction is not None:
        camera_points.append(base_t_image_correction[:3, 3])
    if base_t_insertion is not None:
        camera_points.append(base_t_insertion[:3, 3])
    if base_t_post_insert_retreat_y is not None:
        camera_points.append(base_t_post_insert_retreat_y[:3, 3])
    if base_t_post_insert_shift_x is not None:
        camera_points.append(base_t_post_insert_shift_x[:3, 3])
    if adjacent_two_pin_center is not None:
        camera_points.append(adjacent_two_pin_center)
    if adjacent_two_pin_target is not None:
        camera_points.append(adjacent_two_pin_target)
    if base_t_post_insert_two_pin is not None:
        camera_points.append(base_t_post_insert_two_pin[:3, 3])
    if base_t_post_two_pin_close_retreat_y is not None:
        camera_points.append(base_t_post_two_pin_close_retreat_y[:3, 3])
    if base_t_post_two_pin_close_shift_negative_x is not None:
        camera_points.append(base_t_post_two_pin_close_shift_negative_x[:3, 3])
    if base_t_post_two_pin_close_image_correction is not None:
        camera_points.append(base_t_post_two_pin_close_image_correction[:3, 3])
    if base_t_post_two_pin_aligned_insert is not None:
        camera_points.append(base_t_post_two_pin_aligned_insert[:3, 3])
    if base_t_post_two_pin_release_retreat is not None:
        camera_points.append(base_t_post_two_pin_release_retreat[:3, 3])
    if base_t_post_two_pin_release_shift_z is not None:
        camera_points.append(base_t_post_two_pin_release_shift_z[:3, 3])
    if base_t_post_two_pin_release_final_y is not None:
        camera_points.append(base_t_post_two_pin_release_final_y[:3, 3])
    if base_t_post_two_pin_release_rotate_x is not None:
        camera_points.append(base_t_post_two_pin_release_rotate_x[:3, 3])
    if base_t_post_two_pin_after_close_y is not None:
        camera_points.append(base_t_post_two_pin_after_close_y[:3, 3])
    if base_t_post_two_pin_after_close_yz is not None:
        camera_points.append(base_t_post_two_pin_after_close_yz[:3, 3])
    return np.vstack(camera_points)


def load_preview_tray_geometry(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(args.virtual_tray_obj, force="mesh", process=False)
    handle_root_y_m = args.virtual_tray_handle_root_y_mm / 1000.0
    shorten_lower_tray_handle(mesh, handle_root_y_m, args.virtual_tray_handle_scale)
    tray_attachment_local = shortened_lower_handle_point(
        TI_TRAY_PROTRUSION_CENTER_M,
        handle_root_y_m,
        args.virtual_tray_handle_scale,
    )
    return np.asarray(mesh.vertices, dtype=float), tray_attachment_local


def with_base_z_lift(transform: np.ndarray, lift_m: float) -> np.ndarray:
    lifted = np.asarray(transform, dtype=float).copy()
    lifted[2, 3] -= float(lift_m)
    return lifted


def marker_motion_grasp_transform(
    base_t_marker: np.ndarray,
    start_t_tcp: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    offset_marker_m = marker_target_offset(args)
    marker_target = np.ones(4, dtype=float)
    marker_target[:3] = offset_marker_m

    grasp = np.eye(4)
    grasp[:3, 3] = (base_t_marker @ marker_target)[:3]
    if args.target_orientation == "marker":
        grasp[:3, :3] = base_t_marker[:3, :3]
    else:
        grasp[:3, :3] = start_t_tcp[:3, :3]
    return grasp


def build_motion_trajectory(
    base_t_marker: np.ndarray,
    start_tcp_pose_ur: np.ndarray,
    start_q_rad: np.ndarray,
    rtde_control: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import marker_based_motion as motion

    motion_args = marker_motion_args_from_visualization(args)
    offset_marker_m = motion.marker_target_offset(motion_args)
    start_tcp_pose = np.asarray(start_tcp_pose_ur, dtype=float)
    base_t_start = ur_pose_to_transform(start_tcp_pose)
    base_t_pin = motion.compute_base_t_four_pin_frame(motion_args, base_t_marker)
    grasp_tcp_pose = motion.make_target_pose_ur(
        base_t_marker,
        offset_marker_m,
        start_tcp_pose,
        motion_args.orientation,
    )

    q_start = np.asarray(start_q_rad, dtype=float)
    if not rtde_control.getInverseKinematicsHasSolution(
        grasp_tcp_pose.tolist(),
        q_start.tolist(),
        motion.IK_POSITION_TOLERANCE_M,
        motion.IK_ORIENTATION_TOLERANCE_RAD,
    ):
        raise RuntimeError("Exact trajectory preview failed: no IK solution for marker grasp pose.")
    grasp_q_rad = np.asarray(
        rtde_control.getInverseKinematics(
            grasp_tcp_pose.tolist(),
            q_start.tolist(),
            motion.IK_POSITION_TOLERANCE_M,
            motion.IK_ORIENTATION_TOLERANCE_RAD,
        ),
        dtype=float,
    )
    grasp_q_rad = motion.unwrap_joints_near(grasp_q_rad, q_start)

    safe_plan, lift_waypoints, rotation_waypoints, ik_report = motion.select_safe_alignment_plan(
        rtde_control,
        grasp_tcp_pose,
        grasp_q_rad,
        base_t_pin,
        float(base_t_marker[2, 3]),
        motion_args,
    )
    pin_centering_pose, rotation_tray_reference, pin_centering_reference, pin_centering_vector = (
        motion.pin_centering_target_pose(
            safe_plan["lifted_target_tcp_pose_ur"],
            base_t_pin,
            motion_args,
        )
    )
    pin_approach_pose, _pin_centered_reference, pin_target_reference, pin_approach_vector = (
        motion.pin_approach_target_pose(
            pin_centering_pose,
            base_t_pin,
            motion_args,
            motion_args.pin_approach_clearance_mm,
        )
    )
    planned_current_pose = pin_approach_pose
    image_correction_pose, pin_image_alignment_metadata = image_alignment_correction_pose_from_metadata(
        planned_current_pose,
        motion_args,
    )
    if image_correction_pose is not None:
        planned_current_pose = image_correction_pose

    insertion_pose = None
    insertion_current_reference = None
    insertion_target_reference = None
    insertion_vector = None
    insertion_current_reference_in_pin = None
    insertion_target_reference_in_pin = None
    if motion_args.insert_after_pin_approach:
        (
            insertion_pose,
            insertion_current_reference,
            insertion_target_reference,
            insertion_vector,
            insertion_current_reference_in_pin,
            insertion_target_reference_in_pin,
        ) = motion.pin_insertion_target_pose(
            planned_current_pose,
            base_t_pin,
            motion_args,
            motion_args.pin_insertion_target_y_mm,
        )
        insertion_distance_mm = float(np.linalg.norm(insertion_vector) * 1000.0)
        if insertion_distance_mm > motion_args.max_pin_insertion_mm:
            raise RuntimeError(
                f"Exact trajectory preview failed: final insertion is {insertion_distance_mm:.1f} mm, "
                f"above --max-pin-insertion-mm {motion_args.max_pin_insertion_mm:.1f}."
            )

    post_insert_retreat_y_pose = None
    post_insert_shift_x_pose = None
    post_insert_two_pin_pose = None
    post_insert_shift_x_mm = None
    adjacent_two_pin_center = None
    adjacent_two_pin_target = None
    adjacent_two_pin_translation = None
    post_two_pin_close_retreat_y_pose = None
    post_two_pin_close_shift_negative_x_pose = None
    post_two_pin_close_image_correction_pose = None
    post_two_pin_close_image_alignment_metadata = None
    post_two_pin_aligned_insert_pose = None
    post_two_pin_release_retreat_pose = None
    post_two_pin_release_shift_z_pose = None
    post_two_pin_release_final_y_pose = None
    post_two_pin_release_rotate_x_pose = None
    post_two_pin_after_close_y_pose = None
    post_two_pin_after_close_yz_pose = None
    if insertion_pose is not None and motion_args.post_insert_release_retreat:
        (
            post_insert_retreat_y_pose,
            post_insert_shift_x_pose,
            post_insert_shift_x_mm,
        ) = motion.post_insert_retreat_target_poses(insertion_pose, motion_args)
        if motion_args.post_insert_move_to_two_pin:
            (
                post_insert_two_pin_pose,
                adjacent_two_pin_center,
                adjacent_two_pin_target,
                adjacent_two_pin_translation,
            ) = motion.adjacent_two_pin_tcp_target_pose(
                post_insert_shift_x_pose,
                base_t_pin,
                motion_args,
            )
            if motion_args.post_two_pin_close_retreat:
                (
                    post_two_pin_close_retreat_y_pose,
                    post_two_pin_close_shift_negative_x_pose,
                ) = motion.post_two_pin_close_retreat_target_poses(
                    post_insert_two_pin_pose,
                    motion_args,
                )
                if motion_args.post_two_pin_close_image_align:
                    (
                        post_two_pin_close_image_correction_pose,
                        post_two_pin_close_image_alignment_metadata,
                    ) = image_alignment_correction_pose_from_metadata(
                        post_two_pin_close_shift_negative_x_pose,
                        motion_args,
                        metadata_path=motion.effective_post_two_pin_close_image_alignment_metadata(
                            motion_args
                        ),
                    )
                    if (
                        post_two_pin_close_image_correction_pose is not None
                        and motion_args.post_two_pin_aligned_insert_release
                    ):
                        (
                            post_two_pin_aligned_insert_pose,
                            post_two_pin_release_retreat_pose,
                            post_two_pin_release_shift_z_pose,
                            post_two_pin_release_final_y_pose,
                            post_two_pin_release_rotate_x_pose,
                        ) = motion.post_two_pin_aligned_insert_release_target_poses(
                            post_two_pin_close_image_correction_pose,
                            motion_args,
                        )
                        (
                            (_, post_two_pin_after_close_y_pose),
                            (_, post_two_pin_after_close_yz_pose),
                        ) = motion.post_two_pin_after_close_target_poses(
                            post_two_pin_release_rotate_x_pose,
                            motion_args,
                        )

    q_near = (
        np.asarray(ik_report["q_path"][-1], dtype=float)
        if ik_report["q_path"]
        else grasp_q_rad
    )
    planned_poses = [
        ("pin centering", pin_centering_pose),
        ("pin-y approach", pin_approach_pose),
    ]
    if image_correction_pose is not None:
        planned_poses.append(("post-approach image alignment correction", image_correction_pose))
    if insertion_pose is not None:
        planned_poses.append(("final pin insertion descend", insertion_pose))
    if post_insert_retreat_y_pose is not None:
        planned_poses.append(("post-insert TCP +Y retreat", post_insert_retreat_y_pose))
    if post_insert_shift_x_pose is not None:
        planned_poses.append(("post-insert TCP +X adjacent-pin shift", post_insert_shift_x_pose))
    if post_insert_two_pin_pose is not None:
        planned_poses.append(("post-insert move TCP to adjacent two-pin center", post_insert_two_pin_pose))
    if post_two_pin_close_retreat_y_pose is not None:
        planned_poses.append(("post-two-pin-close TCP +Y retreat", post_two_pin_close_retreat_y_pose))
    if post_two_pin_close_shift_negative_x_pose is not None:
        planned_poses.append(("post-two-pin-close TCP -X shift", post_two_pin_close_shift_negative_x_pose))
    if post_two_pin_close_image_correction_pose is not None:
        planned_poses.append(
            (
                "post-two-pin-close image alignment correction",
                post_two_pin_close_image_correction_pose,
            )
        )
    if post_two_pin_aligned_insert_pose is not None:
        planned_poses.append(("post-two-pin aligned TCP -Y insertion", post_two_pin_aligned_insert_pose))
    if post_two_pin_release_retreat_pose is not None:
        planned_poses.append(("post-two-pin release TCP +Y retreat", post_two_pin_release_retreat_pose))
    if post_two_pin_release_shift_z_pose is not None:
        planned_poses.append(("post-two-pin release TCP Z shift", post_two_pin_release_shift_z_pose))
    if post_two_pin_release_final_y_pose is not None:
        planned_poses.append(("post-two-pin release TCP final X/Y shift", post_two_pin_release_final_y_pose))
    if post_two_pin_release_rotate_x_pose is not None:
        planned_poses.append(("post-two-pin release TCP final X rotation", post_two_pin_release_rotate_x_pose))
    if post_two_pin_after_close_y_pose is not None:
        planned_poses.append(("post-two-pin after-close TCP Y shift", post_two_pin_after_close_y_pose))
    if post_two_pin_after_close_yz_pose is not None:
        planned_poses.append(("post-two-pin after-close TCP Y/Z shift", post_two_pin_after_close_yz_pose))
    for name, pose in planned_poses:
        if not rtde_control.getInverseKinematicsHasSolution(
            pose.tolist(),
            q_near.tolist(),
            motion.IK_POSITION_TOLERANCE_M,
            motion.IK_ORIENTATION_TOLERANCE_RAD,
        ):
            raise RuntimeError(f"Exact trajectory preview failed: no IK solution for {name} pose.")
        q_near = np.asarray(
            rtde_control.getInverseKinematics(
                pose.tolist(),
                q_near.tolist(),
                motion.IK_POSITION_TOLERANCE_M,
                motion.IK_ORIENTATION_TOLERANCE_RAD,
            ),
            dtype=float,
        )

    base_t_grasp = ur_pose_to_transform(grasp_tcp_pose)
    base_t_lifted_grasp = ur_pose_to_transform(safe_plan["lifted_current_tcp_pose_ur"])
    base_t_lifted_rotation = ur_pose_to_transform(safe_plan["lifted_target_tcp_pose_ur"])
    base_t_pin_centering = ur_pose_to_transform(pin_centering_pose)
    base_t_pin_approach = ur_pose_to_transform(pin_approach_pose)
    base_t_image_correction = (
        ur_pose_to_transform(image_correction_pose) if image_correction_pose is not None else None
    )
    base_t_insertion = ur_pose_to_transform(insertion_pose) if insertion_pose is not None else None
    base_t_post_insert_retreat_y = (
        ur_pose_to_transform(post_insert_retreat_y_pose)
        if post_insert_retreat_y_pose is not None
        else None
    )
    base_t_post_insert_shift_x = (
        ur_pose_to_transform(post_insert_shift_x_pose)
        if post_insert_shift_x_pose is not None
        else None
    )
    base_t_post_insert_two_pin = (
        ur_pose_to_transform(post_insert_two_pin_pose)
        if post_insert_two_pin_pose is not None
        else None
    )
    base_t_post_two_pin_close_retreat_y = (
        ur_pose_to_transform(post_two_pin_close_retreat_y_pose)
        if post_two_pin_close_retreat_y_pose is not None
        else None
    )
    base_t_post_two_pin_close_shift_negative_x = (
        ur_pose_to_transform(post_two_pin_close_shift_negative_x_pose)
        if post_two_pin_close_shift_negative_x_pose is not None
        else None
    )
    base_t_post_two_pin_close_image_correction = (
        ur_pose_to_transform(post_two_pin_close_image_correction_pose)
        if post_two_pin_close_image_correction_pose is not None
        else None
    )
    base_t_post_two_pin_aligned_insert = (
        ur_pose_to_transform(post_two_pin_aligned_insert_pose)
        if post_two_pin_aligned_insert_pose is not None
        else None
    )
    base_t_post_two_pin_release_retreat = (
        ur_pose_to_transform(post_two_pin_release_retreat_pose)
        if post_two_pin_release_retreat_pose is not None
        else None
    )
    base_t_post_two_pin_release_shift_z = (
        ur_pose_to_transform(post_two_pin_release_shift_z_pose)
        if post_two_pin_release_shift_z_pose is not None
        else None
    )
    base_t_post_two_pin_release_final_y = (
        ur_pose_to_transform(post_two_pin_release_final_y_pose)
        if post_two_pin_release_final_y_pose is not None
        else None
    )
    base_t_post_two_pin_release_rotate_x = (
        ur_pose_to_transform(post_two_pin_release_rotate_x_pose)
        if post_two_pin_release_rotate_x_pose is not None
        else None
    )
    base_t_post_two_pin_after_close_y = (
        ur_pose_to_transform(post_two_pin_after_close_y_pose)
        if post_two_pin_after_close_y_pose is not None
        else None
    )
    base_t_post_two_pin_after_close_yz = (
        ur_pose_to_transform(post_two_pin_after_close_yz_pose)
        if post_two_pin_after_close_yz_pose is not None
        else None
    )

    step_m = args.trajectory_step_mm / 1000.0
    step_deg = args.trajectory_rotation_step_deg
    stages = [
        ("marker_approach", base_t_start, base_t_grasp, (40, 120, 250)),
        ("post_grasp_lift", base_t_grasp, base_t_lifted_grasp, (50, 210, 100)),
        ("four_pin_rotation", base_t_lifted_grasp, base_t_lifted_rotation, (240, 130, 30)),
        ("pin_centering", base_t_lifted_rotation, base_t_pin_centering, (120, 70, 230)),
        ("pin_y_approach", base_t_pin_centering, base_t_pin_approach, (210, 50, 210)),
    ]
    stage_start = base_t_pin_approach
    if base_t_image_correction is not None:
        stages.append(
            (
                "image_alignment_correction",
                stage_start,
                base_t_image_correction,
                (40, 190, 190),
            )
        )
        stage_start = base_t_image_correction
    if base_t_insertion is not None:
        stages.append(
            (
                "final_pin_insertion",
                stage_start,
                base_t_insertion,
                (235, 60, 60),
            )
        )
        stage_start = base_t_insertion
    if base_t_post_insert_retreat_y is not None:
        stages.append(
            (
                "post_insert_y_retreat",
                stage_start,
                base_t_post_insert_retreat_y,
                (90, 90, 90),
            )
        )
        stage_start = base_t_post_insert_retreat_y
    if base_t_post_insert_shift_x is not None:
        stages.append(
            (
                "post_insert_x_shift",
                stage_start,
                base_t_post_insert_shift_x,
                (20, 20, 20),
            )
        )
        stage_start = base_t_post_insert_shift_x
    if base_t_post_insert_two_pin is not None:
        stages.append(
            (
                "post_insert_two_pin_center",
                stage_start,
                base_t_post_insert_two_pin,
                (0, 0, 0),
            )
        )
        stage_start = base_t_post_insert_two_pin
    if base_t_post_two_pin_close_retreat_y is not None:
        stages.append(
            (
                "post_two_pin_close_y_retreat",
                stage_start,
                base_t_post_two_pin_close_retreat_y,
                (80, 80, 80),
            )
        )
        stage_start = base_t_post_two_pin_close_retreat_y
    if base_t_post_two_pin_close_shift_negative_x is not None:
        stages.append(
            (
                "post_two_pin_close_negative_x_shift",
                stage_start,
                base_t_post_two_pin_close_shift_negative_x,
                (30, 30, 30),
            )
        )
        stage_start = base_t_post_two_pin_close_shift_negative_x
    if base_t_post_two_pin_close_image_correction is not None:
        stages.append(
            (
                "post_two_pin_close_image_alignment",
                stage_start,
                base_t_post_two_pin_close_image_correction,
                (40, 190, 190),
            )
        )
        stage_start = base_t_post_two_pin_close_image_correction
    if base_t_post_two_pin_aligned_insert is not None:
        stages.append(
            (
                "post_two_pin_aligned_insert",
                stage_start,
                base_t_post_two_pin_aligned_insert,
                (235, 60, 60),
            )
        )
        stage_start = base_t_post_two_pin_aligned_insert
    if base_t_post_two_pin_release_retreat is not None:
        stages.append(
            (
                "post_two_pin_release_retreat",
                stage_start,
                base_t_post_two_pin_release_retreat,
                (80, 80, 80),
            )
        )
        stage_start = base_t_post_two_pin_release_retreat
    if base_t_post_two_pin_release_shift_z is not None:
        stages.append(
            (
                "post_two_pin_release_shift_z",
                stage_start,
                base_t_post_two_pin_release_shift_z,
                (90, 90, 90),
            )
        )
        stage_start = base_t_post_two_pin_release_shift_z
    if base_t_post_two_pin_release_final_y is not None:
        stages.append(
            (
                "post_two_pin_release_final_y",
                stage_start,
                base_t_post_two_pin_release_final_y,
                (70, 70, 70),
            )
        )
        stage_start = base_t_post_two_pin_release_final_y
    if base_t_post_two_pin_release_rotate_x is not None:
        stages.append(
            (
                "post_two_pin_release_rotate_x",
                stage_start,
                base_t_post_two_pin_release_rotate_x,
                (120, 120, 120),
            )
        )
        stage_start = base_t_post_two_pin_release_rotate_x
    if base_t_post_two_pin_after_close_y is not None:
        stages.append(
            (
                "post_two_pin_after_close_y_shift",
                stage_start,
                base_t_post_two_pin_after_close_y,
                (80, 120, 120),
            )
        )
        stage_start = base_t_post_two_pin_after_close_y
    if base_t_post_two_pin_after_close_yz is not None:
        stages.append(
            (
                "post_two_pin_after_close_yz_shift",
                stage_start,
                base_t_post_two_pin_after_close_yz,
                (60, 120, 160),
            )
        )

    stage_samples: list[tuple[str, list[np.ndarray], tuple[int, int, int]]] = []
    tcp_samples: list[np.ndarray] = []
    for stage_name, stage_start, stage_target, color in stages:
        samples = interpolate_transforms(stage_start, stage_target, step_m, step_deg)
        stage_samples.append((stage_name, samples, color))
        tcp_samples.extend(samples if not tcp_samples else samples[1:])

    grasped_samples: list[np.ndarray] = []
    for stage_name, samples, _color in stage_samples:
        if stage_name in {
            "marker_approach",
            "post_insert_y_retreat",
            "post_insert_x_shift",
            "post_insert_two_pin_center",
            "post_two_pin_close_y_retreat",
            "post_two_pin_close_negative_x_shift",
            "post_two_pin_close_image_alignment",
            "post_two_pin_aligned_insert",
            "post_two_pin_release_retreat",
            "post_two_pin_release_shift_z",
            "post_two_pin_release_final_y",
            "post_two_pin_release_rotate_x",
            "post_two_pin_after_close_y_shift",
            "post_two_pin_after_close_yz_shift",
        }:
            continue
        grasped_samples.extend(samples if not grasped_samples else samples[1:])
    tray_reference_samples = np.asarray(
        [
            motion.tray_four_hole_center_position(
                transform_to_ur_pose(sample),
                motion_args,
            )
            for sample in grasped_samples
        ],
        dtype=float,
    )
    tcp_points = np.asarray([sample[:3, 3] for sample in tcp_samples], dtype=float)
    final_tcp_pose = (
        insertion_pose
        if insertion_pose is not None
        else image_correction_pose
        if image_correction_pose is not None
        else pin_approach_pose
    )
    final_tray_reference = motion.tray_four_hole_center_position(final_tcp_pose, motion_args)

    return {
        "base_t_pin": base_t_pin,
        "base_t_grasp": base_t_grasp,
        "base_t_lifted_grasp": base_t_lifted_grasp,
        "base_t_lifted_rotation": base_t_lifted_rotation,
        "base_t_pin_centering": base_t_pin_centering,
        "base_t_pin_approach": base_t_pin_approach,
        "base_t_image_correction": base_t_image_correction,
        "base_t_insertion": base_t_insertion,
        "base_t_post_insert_retreat_y": base_t_post_insert_retreat_y,
        "base_t_post_insert_shift_x": base_t_post_insert_shift_x,
        "base_t_post_insert_two_pin": base_t_post_insert_two_pin,
        "base_t_post_two_pin_close_retreat_y": base_t_post_two_pin_close_retreat_y,
        "base_t_post_two_pin_close_shift_negative_x": base_t_post_two_pin_close_shift_negative_x,
        "base_t_post_two_pin_close_image_correction": base_t_post_two_pin_close_image_correction,
        "base_t_post_two_pin_aligned_insert": base_t_post_two_pin_aligned_insert,
        "base_t_post_two_pin_release_retreat": base_t_post_two_pin_release_retreat,
        "base_t_post_two_pin_release_shift_z": base_t_post_two_pin_release_shift_z,
        "base_t_post_two_pin_release_final_y": base_t_post_two_pin_release_final_y,
        "base_t_post_two_pin_release_rotate_x": base_t_post_two_pin_release_rotate_x,
        "base_t_post_two_pin_after_close_y": base_t_post_two_pin_after_close_y,
        "base_t_post_two_pin_after_close_yz": base_t_post_two_pin_after_close_yz,
        "rotation_approach_vector": safe_plan["rotation_approach_vector_m"],
        "pin_centering_vector": pin_centering_vector,
        "pin_approach_vector": pin_approach_vector,
        "pin_image_alignment_metadata": pin_image_alignment_metadata,
        "post_two_pin_close_image_alignment_metadata": post_two_pin_close_image_alignment_metadata,
        "image_correction_pose": image_correction_pose,
        "insertion_vector": insertion_vector,
        "insertion_current_reference": insertion_current_reference,
        "insertion_target_reference": insertion_target_reference,
        "insertion_current_reference_in_pin": insertion_current_reference_in_pin,
        "insertion_target_reference_in_pin": insertion_target_reference_in_pin,
        "post_insert_shift_x_mm": post_insert_shift_x_mm,
        "adjacent_two_pin_center": adjacent_two_pin_center,
        "adjacent_two_pin_target": adjacent_two_pin_target,
        "adjacent_two_pin_translation": adjacent_two_pin_translation,
        "pin_centering_reference": pin_centering_reference,
        "pin_target_reference": pin_target_reference,
        "final_tray_reference": final_tray_reference,
        "rotation_tray_reference": rotation_tray_reference,
        "pre_lateral_m": safe_plan["pre_approach_lateral_distance_m"],
        "post_lateral_m": safe_plan["post_approach_lateral_distance_m"],
        "current_clearance_m": safe_plan["current_clearance_m"],
        "target_clearance_m": safe_plan["target_clearance_m"],
        "safety_lift_m": safe_plan["safety_lift_m"],
        "lift_waypoint_count": len(lift_waypoints),
        "rotation_waypoint_count": len(rotation_waypoints),
        "ik_report": ik_report,
        "stage_samples": stage_samples,
        "tcp_points": tcp_points,
        "tray_reference_points": tray_reference_samples,
    }


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
        base_t_marker,
        start_tcp_pose_ur,
        start_q_rad,
        rtde_control,
        args,
    )

    display_tcp_points = transform_points(display_transform, plan["tcp_points"])
    display_tray_reference_points = transform_points(
        display_transform,
        plan["tray_reference_points"],
    )
    add_polyline(
        server,
        "/motion_trajectory/tcp_path",
        display_tcp_points,
        (40, 120, 250),
        line_width=3.0,
    )
    add_polyline(
        server,
        "/motion_trajectory/tray_reference_path",
        display_tray_reference_points,
        (210, 50, 210),
        line_width=4.0,
    )

    for stage_name, samples, color in plan["stage_samples"]:
        display_stage_points = transform_points(
            display_transform,
            np.asarray([sample[:3, 3] for sample in samples], dtype=float),
        )
        add_polyline(
            server,
            f"/motion_trajectory/stages/{stage_name}",
            display_stage_points,
            color,
            line_width=2.5,
        )

    endpoint_frames = [
        ("00_start_tcp", ur_pose_to_transform(start_tcp_pose_ur)),
        ("01_marker_grasp", plan["base_t_grasp"]),
        ("02_lifted_grasp", plan["base_t_lifted_grasp"]),
        ("03_lifted_rotation", plan["base_t_lifted_rotation"]),
        ("04_pin_centering", plan["base_t_pin_centering"]),
        ("05_pin_y_approach", plan["base_t_pin_approach"]),
    ]
    if plan["base_t_image_correction"] is not None:
        endpoint_frames.append(("06_image_alignment_correction", plan["base_t_image_correction"]))
    if plan["base_t_insertion"] is not None:
        endpoint_frames.append(("07_final_pin_insertion", plan["base_t_insertion"]))
    if plan["base_t_post_insert_retreat_y"] is not None:
        endpoint_frames.append(("08_post_insert_y_retreat", plan["base_t_post_insert_retreat_y"]))
    if plan["base_t_post_insert_shift_x"] is not None:
        endpoint_frames.append(("09_post_insert_x_shift", plan["base_t_post_insert_shift_x"]))
    if plan["base_t_post_insert_two_pin"] is not None:
        endpoint_frames.append(("10_adjacent_two_pin_tcp", plan["base_t_post_insert_two_pin"]))
    if plan["base_t_post_two_pin_close_retreat_y"] is not None:
        endpoint_frames.append(("11_post_two_pin_close_y_retreat", plan["base_t_post_two_pin_close_retreat_y"]))
    if plan["base_t_post_two_pin_close_shift_negative_x"] is not None:
        endpoint_frames.append(
            (
                "12_post_two_pin_close_negative_x_shift",
                plan["base_t_post_two_pin_close_shift_negative_x"],
            )
        )
    if plan["base_t_post_two_pin_close_image_correction"] is not None:
        endpoint_frames.append(
            (
                "13_post_two_pin_close_image_alignment",
                plan["base_t_post_two_pin_close_image_correction"],
            )
        )
    if plan["base_t_post_two_pin_aligned_insert"] is not None:
        endpoint_frames.append(
            (
                "14_post_two_pin_aligned_insert",
                plan["base_t_post_two_pin_aligned_insert"],
            )
        )
    if plan["base_t_post_two_pin_release_retreat"] is not None:
        endpoint_frames.append(
            (
                "15_post_two_pin_release_retreat",
                plan["base_t_post_two_pin_release_retreat"],
            )
        )
    if plan["base_t_post_two_pin_release_shift_z"] is not None:
        endpoint_frames.append(
            (
                "16_post_two_pin_release_shift_z",
                plan["base_t_post_two_pin_release_shift_z"],
            )
        )
    if plan["base_t_post_two_pin_release_final_y"] is not None:
        endpoint_frames.append(
            (
                "17_post_two_pin_release_final_y",
                plan["base_t_post_two_pin_release_final_y"],
            )
        )
    if plan["base_t_post_two_pin_release_rotate_x"] is not None:
        endpoint_frames.append(
            (
                "18_post_two_pin_release_rotate_x",
                plan["base_t_post_two_pin_release_rotate_x"],
            )
        )
    if plan["base_t_post_two_pin_after_close_y"] is not None:
        endpoint_frames.append(
            (
                "19_post_two_pin_after_close_y",
                plan["base_t_post_two_pin_after_close_y"],
            )
        )
    if plan["base_t_post_two_pin_after_close_yz"] is not None:
        endpoint_frames.append(
            (
                "20_post_two_pin_after_close_yz",
                plan["base_t_post_two_pin_after_close_yz"],
            )
        )
    for name, transform in endpoint_frames:
        add_frame(
            server,
            f"/motion_trajectory/frames/{name}",
            display_transform @ transform,
            axes_length=0.045,
        )
        add_point(
            server,
            f"/motion_trajectory/points/{name}",
            (display_transform @ transform)[:3, 3],
            (20, 20, 20),
            0.005,
        )

    base_t_pin = plan["base_t_pin"]
    pin_centering_reference = plan["pin_centering_reference"]
    pin_centering_reference_in_pin = (
        base_t_pin[:3, :3].T @ (pin_centering_reference - base_t_pin[:3, 3])
    )
    pin_target_reference = plan["pin_target_reference"]
    pin_target_reference_in_pin = (
        base_t_pin[:3, :3].T @ (pin_target_reference - base_t_pin[:3, 3])
    )
    final_reference = plan["final_tray_reference"]
    final_reference_in_pin = base_t_pin[:3, :3].T @ (final_reference - base_t_pin[:3, 3])

    print("Marker-motion trajectory preview:")
    print(f"  start source: {'initial pose FK' if args.trajectory_start_from_initial_pose else 'current TCP'}")
    print(f"  tcp samples: {len(plan['tcp_points'])}")
    print(f"  tray reference samples after grasp: {len(plan['tray_reference_points'])}")
    print(f"  marker grasp TCP base m: {fmt(plan['base_t_grasp'][:3, 3])}")
    print(f"  safety lift along base -Z mm: {1000.0 * plan['safety_lift_m']:.3f}")
    print(
        "  exact IK preflight: "
        f"lift waypoints={plan['lift_waypoint_count']}, "
        f"rotation waypoints={plan['rotation_waypoint_count']}, "
        f"minimum q3 margin={plan['ik_report']['minimum_q3_margin_deg']:.1f} deg, "
        f"minimum q5 margin={plan['ik_report']['minimum_q5_margin_deg']:.1f} deg"
    )
    print(
        "  rotation approach vector base mm: "
        f"{fmt(1000.0 * plan['rotation_approach_vector'], decimals=3)}"
    )
    print(
        "  tray reference lateral distance to four-pin center mm: "
        f"{1000.0 * plan['pre_lateral_m']:.3f} -> {1000.0 * plan['post_lateral_m']:.3f}"
    )
    print(
        "  pin centering vector base mm: "
        f"{fmt(1000.0 * plan['pin_centering_vector'], decimals=3)}"
    )
    print(
        "  pin-y approach vector base mm: "
        f"{fmt(1000.0 * plan['pin_approach_vector'], decimals=3)}"
    )
    if plan["pin_image_alignment_metadata"] is not None:
        print(f"  image alignment metadata: {plan['pin_image_alignment_metadata']}")
    if plan["image_correction_pose"] is not None:
        image_correction_reference = (
            plan["insertion_current_reference"]
            if plan["base_t_insertion"] is not None
            else plan["final_tray_reference"]
        )
        image_correction_reference_in_pin = (
            base_t_pin[:3, :3].T @ (image_correction_reference - base_t_pin[:3, 3])
        )
        print(
            "  after image correction tray 4-hole center in pin frame mm: "
            f"{fmt(1000.0 * image_correction_reference_in_pin, decimals=3)}"
        )
    if plan["insertion_vector"] is not None:
        print(
            "  final insertion vector base mm: "
            f"{fmt(1000.0 * plan['insertion_vector'], decimals=3)}"
        )
        print(
            "  insertion start tray 4-hole center in pin frame mm: "
            f"{fmt(1000.0 * plan['insertion_current_reference_in_pin'], decimals=3)}"
        )
        print(
            "  insertion target tray 4-hole center in pin frame mm: "
            f"{fmt(1000.0 * plan['insertion_target_reference_in_pin'], decimals=3)}"
        )
    if plan["base_t_post_insert_retreat_y"] is not None:
        print(f"  post-insert TCP local +Y retreat mm: {args.post_insert_retreat_y_mm:.3f}")
    if plan["base_t_post_insert_shift_x"] is not None:
        print(f"  post-insert TCP local +X shift mm: {plan['post_insert_shift_x_mm']:.3f}")
    if plan["adjacent_two_pin_center"] is not None:
        adjacent_two_pin_center_in_pin = base_t_pin[:3, :3].T @ (
            plan["adjacent_two_pin_center"] - base_t_pin[:3, 3]
        )
        adjacent_two_pin_target_in_pin = base_t_pin[:3, :3].T @ (
            plan["adjacent_two_pin_target"] - base_t_pin[:3, 3]
        )
        print(
            "  adjacent two-pin center in pin frame mm: "
            f"{fmt(1000.0 * adjacent_two_pin_center_in_pin, decimals=3)}"
        )
        print(
            "  adjacent two-pin target in pin frame mm: "
            f"{fmt(1000.0 * adjacent_two_pin_target_in_pin, decimals=3)}"
        )
        print(f"  adjacent two-pin target +Y offset mm: {args.post_insert_two_pin_target_y_mm:.3f}")
        print(
            "  adjacent two-pin move translation base mm: "
            f"{fmt(1000.0 * plan['adjacent_two_pin_translation'], decimals=3)}"
        )
    if plan["base_t_post_two_pin_close_retreat_y"] is not None:
        print(f"  post-two-pin-close TCP local +Y retreat mm: {args.post_two_pin_close_retreat_y_mm:.3f}")
    if plan["base_t_post_two_pin_close_shift_negative_x"] is not None:
        print(
            "  post-two-pin-close TCP local -X shift mm: "
            f"{args.post_two_pin_close_shift_negative_x_mm:.3f}"
        )
    if plan["base_t_post_two_pin_close_image_correction"] is not None:
        print(
            "  post-two-pin-close image alignment metadata: "
            f"{plan['post_two_pin_close_image_alignment_metadata']}"
        )
    if plan["base_t_post_two_pin_aligned_insert"] is not None:
        print(
            "  post-two-pin aligned TCP local -Y insertion mm: "
            f"{args.post_two_pin_aligned_insert_y_mm:.3f}"
        )
    if plan["base_t_post_two_pin_release_retreat"] is not None:
        print(
            "  post-two-pin release TCP local +Y retreat mm: "
            f"{args.post_two_pin_release_retreat_y_mm:.3f}"
        )
    if plan["base_t_post_two_pin_release_shift_z"] is not None:
        print(
            "  post-two-pin release TCP local Z shift mm: "
            f"{args.post_two_pin_release_shift_z_mm:.3f}"
        )
    if plan["base_t_post_two_pin_release_final_y"] is not None:
        print(
            "  post-two-pin release TCP local final X/Y shift mm: "
            f"{args.post_two_pin_release_final_x_mm:.3f}/"
            f"{args.post_two_pin_release_final_y_mm:.3f}"
        )
    if plan["base_t_post_two_pin_release_rotate_x"] is not None:
        print(
            "  post-two-pin release TCP local X rotation deg: "
            f"{args.post_two_pin_release_rotate_x_deg:.3f}"
        )
    if plan["base_t_post_two_pin_after_close_y"] is not None:
        print(
            "  post-two-pin after-close TCP local Y shift mm: "
            f"{args.post_two_pin_after_close_y_mm:.3f}"
        )
    if plan["base_t_post_two_pin_after_close_yz"] is not None:
        print(
            "  post-two-pin after-close TCP local final Y/Z shift mm: "
            f"{args.post_two_pin_after_close_final_y_mm:.3f}/"
            f"{args.post_two_pin_after_close_final_z_mm:.3f}"
        )
    print(
        "  pin centering target tray reference in pin frame mm: "
        f"{fmt(1000.0 * pin_centering_reference_in_pin, decimals=3)}"
    )
    print(
        "  pin-y target tray reference in pin frame mm: "
        f"{fmt(1000.0 * pin_target_reference_in_pin, decimals=3)}"
    )
    print(
        "  final tray reference in pin frame mm: "
        f"{fmt(1000.0 * final_reference_in_pin, decimals=3)}"
    )
    print(f"  current tray floor clearance at grasp mm: {1000.0 * plan['current_clearance_m']:.3f}")
    print(f"  target tray floor clearance before lift mm: {1000.0 * plan['target_clearance_m']:.3f}")

    return np.vstack(
        [
            display_tcp_points,
            display_tray_reference_points,
            transform_points(
                display_transform,
                np.asarray(
                    [
                        plan["base_t_pin"][:3, 3],
                        pin_target_reference,
                        final_reference,
                        *(
                            [plan["adjacent_two_pin_center"], plan["adjacent_two_pin_target"]]
                            if plan["adjacent_two_pin_target"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_close_retreat_y"][:3, 3]]
                            if plan["base_t_post_two_pin_close_retreat_y"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_close_shift_negative_x"][:3, 3]]
                            if plan["base_t_post_two_pin_close_shift_negative_x"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_close_image_correction"][:3, 3]]
                            if plan["base_t_post_two_pin_close_image_correction"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_aligned_insert"][:3, 3]]
                            if plan["base_t_post_two_pin_aligned_insert"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_release_retreat"][:3, 3]]
                            if plan["base_t_post_two_pin_release_retreat"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_release_shift_z"][:3, 3]]
                            if plan["base_t_post_two_pin_release_shift_z"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_release_final_y"][:3, 3]]
                            if plan["base_t_post_two_pin_release_final_y"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_release_rotate_x"][:3, 3]]
                            if plan["base_t_post_two_pin_release_rotate_x"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_after_close_y"][:3, 3]]
                            if plan["base_t_post_two_pin_after_close_y"] is not None
                            else []
                        ),
                        *(
                            [plan["base_t_post_two_pin_after_close_yz"][:3, 3]]
                            if plan["base_t_post_two_pin_after_close_yz"] is not None
                            else []
                        ),
                    ],
                    dtype=float,
                ),
            ),
        ]
    )


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


def translation_transform(offset: np.ndarray | list[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(offset, dtype=float)
    return transform


def rotation_transform(axis: str, angle_deg: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(axis, float(angle_deg), degrees=True).as_matrix()
    return transform


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


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ np.asarray(points, dtype=float).T).T + transform[:3, 3]


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
