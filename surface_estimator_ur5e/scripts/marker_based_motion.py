#!/usr/bin/env python3
"""Move the UR5e TCP to a pose defined by an offset from a saved marker frame."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import yaml

from surface_estimator_ur5e.live_robot import DEFAULT_ROBOT_IP
from surface_estimator_ur5e.motion_sequence import (
    DEFAULT_GRIPPER_PORT,
    GripperCommand,
    RobotiqHandEGripper,
)

from visualize_marker_frame import (
    DEFAULT_ASSEMBLY_LOCAL_YAW_DEG,
    DEFAULT_ASSEMBLY_ORIGIN_X_MM,
    DEFAULT_ASSEMBLY_ORIGIN_Y_MM,
    DEFAULT_ASSEMBLY_ORIGIN_Z_MM,
    DEFAULT_ASSEMBLY_RPY_DEG,
    DEFAULT_TI_ASSEMBLY_OBJ,
    DEFAULT_VIRTUAL_TRAY_HANDLE_ROOT_Y_MM,
    DEFAULT_VIRTUAL_TRAY_HANDLE_SCALE,
    DEFAULT_VIRTUAL_TRAY_LOCAL_RX_DEG,
    DEFAULT_VIRTUAL_TRAY_OBJ,
    DEFAULT_VIRTUAL_TRAY_TCP_Z_MM,
    TI_TRAY_FOUR_PIN_HOLE_CENTER_M,
    TI_TRAY_PROTRUSION_CENTER_M,
    four_pin_feature_frame,
    marker_relative_transform,
    rotation_transform,
    shortened_lower_handle_point,
    shorten_lower_tray_handle,
    translation_transform,
)


DEFAULT_MARKER_POSE = Path("data/markers/aruco_364_in_base.yaml")
DEFAULT_TARGET_OUTPUT = Path("data/markers/aruco_364_tcp_target.yaml")
DEFAULT_OFFSET_X_MM = 144.0
DEFAULT_OFFSET_Y_MM = -50.5
DEFAULT_HEIGHT_ABOVE_MARKER_MM = 15.0
BASE_Z_DOWN = np.array([0.0, 0.0, 1.0], dtype=float)
DEFAULT_TCP_OFFSET_UR = [0.0, 0.0, 0.158, -1.5707, 0.0, 0.0]
DEFAULT_INITIAL_Q_DEG = [44.85, -15.96, -81.60, 7.72, 89.63, 135.01]
DEFAULT_SPEED_M_S = 0.02
DEFAULT_ACCELERATION_M_S2 = 0.04
DEFAULT_POST_GRASP_LIFT_MM = 15.0
DEFAULT_SAFE_FLOOR_CLEARANCE_MM = 15.0
DEFAULT_MAX_SAFETY_LIFT_MM = 80.0
DEFAULT_MAX_ROTATION_DEG = 120.0
DEFAULT_ROTATION_APPROACH_MM = 50.0
DEFAULT_PIN_APPROACH_CLEARANCE_MM = 50.0
DEFAULT_MAX_PIN_APPROACH_MM = 300.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIN_IMAGE_ALIGNMENT_METADATA = (
    PROJECT_ROOT
    / "images"
    / "tcp_image_alignment_20260715_114251"
    / "aligned_034_y_m03p750deg_xp00p000mm_zm04p750mm_metadata.yaml"
)
DEFAULT_POST_TWO_PIN_CLOSE_IMAGE_ALIGNMENT_METADATA = (
    PROJECT_ROOT
    / "images"
    / "tcp_image_alignment_20260715_133958"
    / "aligned_017_y_p02p500deg_xp00p500mm_zp01p250mm_metadata.yaml"
)
DEFAULT_MAX_PIN_IMAGE_CORRECTION_MM = 10.0
DEFAULT_MAX_PIN_IMAGE_CORRECTION_DEG = 10.0
DEFAULT_PIN_INSERTION_TARGET_Y_MM = 0.0
DEFAULT_MAX_PIN_INSERTION_MM = 80.0
DEFAULT_PIN_INSERTION_SPEED_M_S = 0.005
DEFAULT_PIN_INSERTION_ACCELERATION_M_S2 = 0.01
DEFAULT_POST_INSERT_RETREAT_Y_MM = 50.0
DEFAULT_POST_INSERT_TWO_PIN_TARGET_Y_MM = 19.0
DEFAULT_POST_TWO_PIN_CLOSE_RETREAT_Y_MM = 30.0
DEFAULT_POST_TWO_PIN_CLOSE_SHIFT_NEGATIVE_X_MM = 58.0
DEFAULT_POST_TWO_PIN_ALIGNED_INSERT_Y_MM = 20.0
DEFAULT_POST_TWO_PIN_RELEASE_RETREAT_Y_MM = 30.0
DEFAULT_PIN_IMAGE_SERIAL = "261322073147"
DEFAULT_PIN_IMAGE_WIDTH = 1280
DEFAULT_PIN_IMAGE_HEIGHT = 720
DEFAULT_PIN_IMAGE_FPS = 30
DEFAULT_PIN_IMAGE_WARMUP_FRAMES = 10
DEFAULT_PIN_IMAGE_TIMEOUT_MS = 10000
DEFAULT_AUTO_PIN_IMAGE_SEARCH_STEP_MM = 0.25
DEFAULT_AUTO_PIN_IMAGE_SEARCH_STEP_DEG = 0.25
DEFAULT_AUTO_PIN_IMAGE_SEARCH_ITERATIONS = 2
DEFAULT_AUTO_PIN_IMAGE_SEARCH_MAX_MM = 3.0
DEFAULT_AUTO_PIN_IMAGE_SEARCH_MAX_DEG = 3.0
DEFAULT_AUTO_PIN_IMAGE_SETTLE_S = 0.2
DEFAULT_INITIAL_SPEED_RAD_S = 0.05
DEFAULT_INITIAL_ACCELERATION_RAD_S2 = 0.05
DEFAULT_INITIAL_DWELL_S = 0.5
POST_GRASP_DWELL_S = 3.0
POST_TWO_PIN_CLOSE_DWELL_S = 3.0
ROTATION_LIFT_CANDIDATES_MM = (0.0, 30.0, 50.0, 70.0)
LIFT_WAYPOINT_STEP_MM = 10.0
ROTATION_WAYPOINT_STEP_DEG = 5.0
MIN_Q3_SINGULARITY_MARGIN_DEG = 25.0
MIN_Q5_SINGULARITY_MARGIN_DEG = 10.0
MAX_IK_JOINT_STEP_DEG = 20.0
IK_POSITION_TOLERANCE_M = 1e-5
IK_ORIENTATION_TOLERANCE_RAD = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute a UR TCP target from a saved marker pose. "
            "Default behavior is dry-run only; add --execute to move the robot."
        )
    )
    parser.add_argument("--marker-pose", type=Path, default=DEFAULT_MARKER_POSE)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.set_defaults(start_from_initial_pose=True)
    parser.add_argument(
        "--start-from-initial-pose",
        dest="start_from_initial_pose",
        action="store_true",
        help="MoveJ to the saved initial joint pose before computing the marker target. This is the default.",
    )
    parser.add_argument(
        "--no-start-from-initial-pose",
        "--skip-initial-pose",
        dest="start_from_initial_pose",
        action="store_false",
        help="Start from the current robot pose instead of first moving to the saved initial joint pose.",
    )
    parser.add_argument(
        "--initial-q-deg",
        nargs=6,
        type=float,
        default=DEFAULT_INITIAL_Q_DEG,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    )
    parser.add_argument("--initial-speed-rad-s", type=float, default=DEFAULT_INITIAL_SPEED_RAD_S)
    parser.add_argument(
        "--initial-acceleration-rad-s2",
        type=float,
        default=DEFAULT_INITIAL_ACCELERATION_RAD_S2,
    )
    parser.add_argument("--initial-dwell-s", type=float, default=DEFAULT_INITIAL_DWELL_S)
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
        help=(
            "'floor' projects marker x/y onto the horizontal floor and uses base +Z as down. "
            "'raw' uses the saved ArUco 6D pose directly."
        ),
    )
    parser.add_argument(
        "--offset-z-mm",
        type=float,
        default=None,
        help="Raw marker z offset, only used with --marker-frame-mode raw.",
    )
    parser.add_argument(
        "--orientation",
        choices=("current", "marker"),
        default="current",
        help="TCP orientation to use at the target. 'current' keeps the current TCP rotation.",
    )
    parser.add_argument("--speed-m-s", type=float, default=DEFAULT_SPEED_M_S)
    parser.add_argument("--acceleration-m-s2", type=float, default=DEFAULT_ACCELERATION_M_S2)
    parser.add_argument(
        "--max-distance-mm",
        type=float,
        default=300.0,
        help="Abort --execute if the TCP target is farther than this from the current TCP.",
    )
    parser.add_argument("--allow-large-move", action="store_true")
    parser.add_argument(
        "--gripper-delay-s",
        type=float,
        default=0.5,
        help="Seconds to wait after moveL before closing the gripper.",
    )
    parser.add_argument("--gripper-close-percent", type=float, default=100.0)
    parser.add_argument("--gripper-speed", type=int, default=80)
    parser.add_argument("--gripper-force", type=int, default=100)
    parser.add_argument("--gripper-port", type=int, default=DEFAULT_GRIPPER_PORT)
    parser.add_argument(
        "--skip-gripper-close",
        action="store_true",
        help="Move only; do not close the gripper after reaching the target.",
    )
    parser.add_argument(
        "--align-to-four-pin-frame",
        action="store_true",
        help="After closing the gripper, lift the tray and rotate TCP to the four-pin frame.",
    )
    parser.add_argument(
        "--segmented-alignment",
        action="store_true",
        help=(
            "Execute the post-grasp lift and four-pin rotation waypoint-by-waypoint. "
            "By default execution uses one continuous moveL per stage after IK preflight."
        ),
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
    )
    parser.add_argument(
        "--assembly-local-yaw-deg",
        "--holder-local-yaw-deg",
        dest="assembly_local_yaw_deg",
        type=float,
        default=DEFAULT_ASSEMBLY_LOCAL_YAW_DEG,
    )
    parser.add_argument(
        "--tcp-rotation-offset-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Optional local offset from the four-pin frame to the desired TCP frame.",
    )
    parser.add_argument(
        "--post-grasp-lift-mm",
        type=float,
        default=DEFAULT_POST_GRASP_LIFT_MM,
        help="Minimum lift along base -Z after grasping and before rotation.",
    )
    parser.add_argument(
        "--safe-floor-clearance-mm",
        type=float,
        default=DEFAULT_SAFE_FLOOR_CLEARANCE_MM,
        help="Minimum tray bounding-box clearance above marker floor during rotation.",
    )
    parser.add_argument(
        "--max-safety-lift-mm",
        type=float,
        default=DEFAULT_MAX_SAFETY_LIFT_MM,
        help="Abort execute if the required pre-rotation lift exceeds this value.",
    )
    parser.add_argument("--max-rotation-deg", type=float, default=DEFAULT_MAX_ROTATION_DEG)
    parser.add_argument("--allow-large-rotation", action="store_true")
    parser.add_argument(
        "--rotation-approach-mm",
        type=float,
        default=DEFAULT_ROTATION_APPROACH_MM,
        help=(
            "Translate during four-pin rotation so the held tray reference moves this far "
            "toward the holder four-pin center, projected perpendicular to the pin axis."
        ),
    )
    parser.set_defaults(approach_pin_after_rotation=True)
    parser.add_argument(
        "--approach-pin-after-rotation",
        dest="approach_pin_after_rotation",
        action="store_true",
        help="After four-pin rotation, move the held tray reference to holder pin +Y. This is the default.",
    )
    parser.add_argument(
        "--no-approach-pin-after-rotation",
        "--skip-pin-approach",
        dest="approach_pin_after_rotation",
        action="store_false",
        help="Skip the post-rotation move toward the holder pins.",
    )
    parser.add_argument(
        "--continue-pin-sequence-from-current",
        action="store_true",
        help=(
            "Assume the tray is already grasped and four-pin rotation alignment is complete; "
            "run only the pin approach, optional image correction, and optional insertion "
            "from the current TCP pose."
        ),
    )
    parser.add_argument(
        "--pin-approach-clearance-mm",
        type=float,
        default=DEFAULT_PIN_APPROACH_CLEARANCE_MM,
        help="Target tray-reference distance along holder four-pin frame +Y after rotation.",
    )
    parser.add_argument(
        "--max-pin-approach-mm",
        type=float,
        default=DEFAULT_MAX_PIN_APPROACH_MM,
        help="Abort execute if the post-rotation pin approach translation exceeds this distance.",
    )
    parser.add_argument(
        "--pin-image-alignment-metadata",
        type=Path,
        default=None,
        help=(
            "Apply a saved tcp_image_alignment_sweep aligned metadata correction after "
            "the post-rotation pin approach."
        ),
    )
    parser.add_argument(
        "--auto-pin-image-align",
        action="store_true",
        help=(
            "After the post-rotation pin approach, automatically search small TCP local "
            "X/Z/rotation offsets that best match --pin-image-reference."
        ),
    )
    parser.add_argument(
        "--pin-image-reference",
        type=Path,
        default=None,
        help="Reference aligned RGB image used by --auto-pin-image-align.",
    )
    parser.add_argument(
        "--pin-image-roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional image ROI for automatic alignment scoring.",
    )
    parser.add_argument("--pin-image-serial", default=DEFAULT_PIN_IMAGE_SERIAL)
    parser.add_argument("--pin-image-width", type=int, default=DEFAULT_PIN_IMAGE_WIDTH)
    parser.add_argument("--pin-image-height", type=int, default=DEFAULT_PIN_IMAGE_HEIGHT)
    parser.add_argument("--pin-image-fps", type=int, default=DEFAULT_PIN_IMAGE_FPS)
    parser.add_argument(
        "--pin-image-warmup-frames",
        type=int,
        default=DEFAULT_PIN_IMAGE_WARMUP_FRAMES,
    )
    parser.add_argument(
        "--pin-image-timeout-ms",
        type=int,
        default=DEFAULT_PIN_IMAGE_TIMEOUT_MS,
    )
    parser.add_argument(
        "--pin-image-output-dir",
        type=Path,
        default=None,
        help="Directory for automatic pin image alignment captures.",
    )
    parser.add_argument(
        "--auto-pin-image-search-step-mm",
        type=float,
        default=DEFAULT_AUTO_PIN_IMAGE_SEARCH_STEP_MM,
    )
    parser.add_argument(
        "--auto-pin-image-search-step-deg",
        type=float,
        default=DEFAULT_AUTO_PIN_IMAGE_SEARCH_STEP_DEG,
    )
    parser.add_argument(
        "--auto-pin-image-search-iterations",
        type=int,
        default=DEFAULT_AUTO_PIN_IMAGE_SEARCH_ITERATIONS,
    )
    parser.add_argument(
        "--auto-pin-image-search-max-mm",
        type=float,
        default=DEFAULT_AUTO_PIN_IMAGE_SEARCH_MAX_MM,
    )
    parser.add_argument(
        "--auto-pin-image-search-max-deg",
        type=float,
        default=DEFAULT_AUTO_PIN_IMAGE_SEARCH_MAX_DEG,
    )
    parser.add_argument(
        "--auto-pin-image-settle-s",
        type=float,
        default=DEFAULT_AUTO_PIN_IMAGE_SETTLE_S,
    )
    parser.add_argument(
        "--max-pin-image-correction-mm",
        type=float,
        default=DEFAULT_MAX_PIN_IMAGE_CORRECTION_MM,
        help="Abort if the image-based TCP local X/Z correction exceeds this value.",
    )
    parser.add_argument(
        "--max-pin-image-correction-deg",
        type=float,
        default=DEFAULT_MAX_PIN_IMAGE_CORRECTION_DEG,
        help="Abort if the image-based TCP local rotation correction exceeds this value.",
    )
    parser.add_argument(
        "--insert-after-pin-approach",
        action="store_true",
        help=(
            "After optional image correction, move only along holder four-pin frame Y "
            "to --pin-insertion-target-y-mm."
        ),
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
        help="Abort if the final insertion translation exceeds this distance.",
    )
    parser.add_argument(
        "--pin-insertion-speed-m-s",
        type=float,
        default=DEFAULT_PIN_INSERTION_SPEED_M_S,
        help="Linear speed used only for the final pin insertion descend.",
    )
    parser.add_argument(
        "--pin-insertion-acceleration-m-s2",
        type=float,
        default=DEFAULT_PIN_INSERTION_ACCELERATION_M_S2,
        help="Linear acceleration used only for the final pin insertion descend.",
    )
    parser.set_defaults(post_insert_release_retreat=True)
    parser.add_argument(
        "--post-insert-release-retreat",
        dest="post_insert_release_retreat",
        action="store_true",
        help=(
            "After final insertion, fully open the gripper, move TCP local +Y, "
            "then move TCP local +X. This is the default when insertion runs."
        ),
    )
    parser.add_argument(
        "--no-post-insert-release-retreat",
        dest="post_insert_release_retreat",
        action="store_false",
        help="Skip the gripper-open and TCP-local retreat after final insertion.",
    )
    parser.add_argument(
        "--post-insert-retreat-y-mm",
        type=float,
        default=DEFAULT_POST_INSERT_RETREAT_Y_MM,
        help="TCP-local +Y move after opening the gripper at the inserted pose.",
    )
    parser.add_argument(
        "--post-insert-shift-x-mm",
        type=float,
        default=None,
        help=(
            "TCP-local +X move after the +Y retreat. If omitted, compute the distance "
            "from the holder four-pin center to the adjacent two-pin center from the OBJ."
        ),
    )
    parser.set_defaults(post_insert_move_to_two_pin=True)
    parser.add_argument(
        "--post-insert-move-to-two-pin",
        dest="post_insert_move_to_two_pin",
        action="store_true",
        help=(
            "After the post-insert +Y/+X retreat, move the TCP origin to the holder "
            "adjacent two-pin target and close the gripper. This is the default."
        ),
    )
    parser.add_argument(
        "--no-post-insert-move-to-two-pin",
        dest="post_insert_move_to_two_pin",
        action="store_false",
        help="Skip the final move to the adjacent two-pin target and gripper close.",
    )
    parser.add_argument(
        "--post-insert-two-pin-target-y-mm",
        type=float,
        default=DEFAULT_POST_INSERT_TWO_PIN_TARGET_Y_MM,
        help=(
            "Holder pin-frame +Y offset applied to the final adjacent two-pin TCP target."
        ),
    )
    parser.set_defaults(post_two_pin_close_retreat=True)
    parser.add_argument(
        "--post-two-pin-close-retreat",
        dest="post_two_pin_close_retreat",
        action="store_true",
        help=(
            "After closing at the adjacent two-pin target, move TCP local +Y then "
            "TCP local -X. This is the default."
        ),
    )
    parser.add_argument(
        "--no-post-two-pin-close-retreat",
        dest="post_two_pin_close_retreat",
        action="store_false",
        help="Skip the TCP-local moves after closing at the adjacent two-pin target.",
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
        help=(
            "Apply image alignment again after the final post-close TCP-local move. "
            "This is the default."
        ),
    )
    parser.add_argument(
        "--no-post-two-pin-close-image-align",
        dest="post_two_pin_close_image_align",
        action="store_false",
        help="Stop after the final post-close TCP-local move without image alignment.",
    )
    parser.add_argument(
        "--post-two-pin-close-image-alignment-metadata",
        type=Path,
        default=None,
        help=(
            "Saved tcp_image_alignment_sweep metadata to apply after the final "
            "post-close TCP-local move. If omitted, uses the latest calibrated "
            "post-close metadata."
        ),
    )
    parser.set_defaults(post_two_pin_aligned_insert_release=True)
    parser.add_argument(
        "--post-two-pin-aligned-insert-release",
        dest="post_two_pin_aligned_insert_release",
        action="store_true",
        help=(
            "After final post-close image alignment, move TCP local -Y, fully open "
            "the gripper, then retreat TCP local +Y. This is the default."
        ),
    )
    parser.add_argument(
        "--no-post-two-pin-aligned-insert-release",
        dest="post_two_pin_aligned_insert_release",
        action="store_false",
        help="Skip the final aligned -Y insertion and gripper release retreat.",
    )
    parser.add_argument(
        "--post-two-pin-aligned-insert-y-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_ALIGNED_INSERT_Y_MM,
        help=(
            "TCP-local -Y move after final post-close image alignment. This move uses "
            "the final insertion speed/acceleration."
        ),
    )
    parser.add_argument(
        "--post-two-pin-release-retreat-y-mm",
        type=float,
        default=DEFAULT_POST_TWO_PIN_RELEASE_RETREAT_Y_MM,
        help="TCP-local +Y retreat after fully opening the gripper at the final aligned insertion.",
    )
    parser.add_argument("--tray-obj", type=Path, default=DEFAULT_VIRTUAL_TRAY_OBJ)
    parser.add_argument(
        "--tray-center-tcp-z-mm",
        "--tray-reference-tcp-z-mm",
        dest="tray_center_tcp_z_mm",
        type=float,
        default=DEFAULT_VIRTUAL_TRAY_TCP_Z_MM,
        help=(
            "TCP-local Z offset to the held tray reference frame. "
            "Default 0 means virtual_tray/adjustment is the TCP reference."
        ),
    )
    parser.add_argument("--tray-local-rx-deg", type=float, default=DEFAULT_VIRTUAL_TRAY_LOCAL_RX_DEG)
    parser.add_argument("--tray-handle-root-y-mm", type=float, default=DEFAULT_VIRTUAL_TRAY_HANDLE_ROOT_Y_MM)
    parser.add_argument("--tray-handle-scale", type=float, default=DEFAULT_VIRTUAL_TRAY_HANDLE_SCALE)
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
        help="Do not call RTDEControl.setTcp before reading or moving.",
    )
    parser.add_argument(
        "--save-target",
        nargs="?",
        const=str(DEFAULT_TARGET_OUTPUT),
        help="Save the computed target pose YAML. If no path is given, uses data/markers.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def load_marker_transform(path: Path) -> tuple[int | None, np.ndarray, dict[str, Any]]:
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
        transform = np.asarray(pose["transform_matrix"], dtype=float)
    else:
        translation = np.asarray(pose.get("translation_m"), dtype=float).reshape(3)
        if "rotation_quaternion_xyzw" in pose:
            rotation = Rotation.from_quat(pose["rotation_quaternion_xyzw"]).as_matrix()
        elif "rotation_matrix" in pose:
            rotation = np.asarray(pose["rotation_matrix"], dtype=float).reshape(3, 3)
        else:
            raise ValueError(
                f"{path} pose_in_base must contain transform_matrix, "
                "rotation_quaternion_xyzw, or rotation_matrix."
            )
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

    if transform.shape != (4, 4):
        raise ValueError(f"{path} transform_matrix must be 4x4.")
    if not np.all(np.isfinite(transform)):
        raise ValueError(f"{path} transform_matrix contains non-finite values.")

    marker_id = data.get("marker_id")
    return int(marker_id) if marker_id is not None else None, transform, data


def marker_offset_target_position(base_t_marker: np.ndarray, offset_marker_m: np.ndarray) -> np.ndarray:
    offset_homogeneous = np.ones(4, dtype=float)
    offset_homogeneous[:3] = offset_marker_m
    return (base_t_marker @ offset_homogeneous)[:3]


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


def marker_target_offset(args: argparse.Namespace) -> np.ndarray:
    if args.marker_frame_mode == "raw":
        z_mm = args.offset_z_mm
        if z_mm is None:
            z_mm = args.height_above_marker_mm
        return np.array([args.offset_x_mm, args.offset_y_mm, z_mm], dtype=float) / 1000.0

    return np.array(
        [args.offset_x_mm, args.offset_y_mm, -args.height_above_marker_mm],
        dtype=float,
    ) / 1000.0


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


def effective_pin_image_alignment_metadata(args: argparse.Namespace) -> Path | None:
    if args.auto_pin_image_align:
        return None
    if args.pin_image_alignment_metadata is not None:
        return args.pin_image_alignment_metadata
    if args.insert_after_pin_approach:
        return DEFAULT_PIN_IMAGE_ALIGNMENT_METADATA
    return None


def effective_post_two_pin_close_image_alignment_metadata(args: argparse.Namespace) -> Path | None:
    if not args.post_two_pin_close_image_align or args.auto_pin_image_align:
        return None
    if args.post_two_pin_close_image_alignment_metadata is not None:
        return args.post_two_pin_close_image_alignment_metadata
    return DEFAULT_POST_TWO_PIN_CLOSE_IMAGE_ALIGNMENT_METADATA


def validate_args(args: argparse.Namespace) -> None:
    numeric_values = [
        args.offset_x_mm,
        args.offset_y_mm,
        args.height_above_marker_mm,
        args.initial_speed_rad_s,
        args.initial_acceleration_rad_s2,
        args.initial_dwell_s,
        args.speed_m_s,
        args.acceleration_m_s2,
        args.max_distance_mm,
        args.gripper_delay_s,
        args.assembly_origin_x_mm,
        args.assembly_origin_y_mm,
        args.assembly_origin_z_mm,
        args.assembly_local_yaw_deg,
        args.post_grasp_lift_mm,
        args.safe_floor_clearance_mm,
        args.max_safety_lift_mm,
        args.max_rotation_deg,
        args.rotation_approach_mm,
        args.pin_approach_clearance_mm,
        args.max_pin_approach_mm,
        args.max_pin_image_correction_mm,
        args.max_pin_image_correction_deg,
        args.pin_insertion_target_y_mm,
        args.max_pin_insertion_mm,
        args.pin_image_width,
        args.pin_image_height,
        args.pin_image_fps,
        args.pin_image_warmup_frames,
        args.pin_image_timeout_ms,
        args.auto_pin_image_search_step_mm,
        args.auto_pin_image_search_step_deg,
        args.auto_pin_image_search_iterations,
        args.auto_pin_image_search_max_mm,
        args.auto_pin_image_search_max_deg,
        args.auto_pin_image_settle_s,
        args.pin_insertion_speed_m_s,
        args.pin_insertion_acceleration_m_s2,
        args.post_insert_retreat_y_mm,
        args.post_insert_two_pin_target_y_mm,
        args.post_two_pin_close_retreat_y_mm,
        args.post_two_pin_close_shift_negative_x_mm,
        args.post_two_pin_aligned_insert_y_mm,
        args.post_two_pin_release_retreat_y_mm,
        args.tray_center_tcp_z_mm,
        args.tray_local_rx_deg,
        args.tray_handle_root_y_mm,
        args.tray_handle_scale,
    ]
    if args.post_insert_shift_x_mm is not None:
        numeric_values.append(args.post_insert_shift_x_mm)
    numeric_values.extend(args.assembly_rpy_deg)
    numeric_values.extend(args.tcp_rotation_offset_rpy_deg)
    numeric_values.extend(args.tcp_offset_ur)
    numeric_values.extend(args.initial_q_deg)
    if not np.all(np.isfinite(numeric_values)):
        raise ValueError("All numeric arguments must be finite.")
    if args.initial_speed_rad_s <= 0.0:
        raise ValueError("--initial-speed-rad-s must be positive.")
    if args.initial_acceleration_rad_s2 <= 0.0:
        raise ValueError("--initial-acceleration-rad-s2 must be positive.")
    if args.initial_dwell_s < 0.0:
        raise ValueError("--initial-dwell-s must be non-negative.")
    if args.speed_m_s <= 0.0:
        raise ValueError("--speed-m-s must be positive.")
    if args.acceleration_m_s2 <= 0.0:
        raise ValueError("--acceleration-m-s2 must be positive.")
    if args.max_distance_mm < 0.0:
        raise ValueError("--max-distance-mm must be non-negative.")
    if args.gripper_delay_s < 0.0:
        raise ValueError("--gripper-delay-s must be non-negative.")
    if args.post_grasp_lift_mm < 0.0:
        raise ValueError("--post-grasp-lift-mm must be non-negative.")
    if args.safe_floor_clearance_mm < 0.0:
        raise ValueError("--safe-floor-clearance-mm must be non-negative.")
    if args.max_safety_lift_mm < 0.0:
        raise ValueError("--max-safety-lift-mm must be non-negative.")
    if args.max_rotation_deg < 0.0:
        raise ValueError("--max-rotation-deg must be non-negative.")
    if args.rotation_approach_mm < 0.0:
        raise ValueError("--rotation-approach-mm must be non-negative.")
    if args.pin_approach_clearance_mm < 0.0:
        raise ValueError("--pin-approach-clearance-mm must be non-negative.")
    if args.max_pin_approach_mm < 0.0:
        raise ValueError("--max-pin-approach-mm must be non-negative.")
    image_alignment_metadata_paths = [
        ("Pin image alignment metadata", effective_pin_image_alignment_metadata(args)),
        (
            "Post-two-pin-close image alignment metadata",
            effective_post_two_pin_close_image_alignment_metadata(args),
        ),
    ]
    for metadata_label, pin_image_alignment_metadata in image_alignment_metadata_paths:
        if pin_image_alignment_metadata is None:
            continue
        if not pin_image_alignment_metadata.exists():
            raise FileNotFoundError(
                f"{metadata_label} not found: {pin_image_alignment_metadata}"
            )
        if not pin_image_alignment_metadata.is_file():
            raise FileNotFoundError(
                f"{metadata_label} must be a YAML file, got directory or "
                f"non-file path: {pin_image_alignment_metadata}"
            )
    if args.auto_pin_image_align:
        if args.pin_image_reference is None:
            raise ValueError("--auto-pin-image-align requires --pin-image-reference.")
        if not args.pin_image_reference.exists():
            raise FileNotFoundError(f"--pin-image-reference not found: {args.pin_image_reference}")
        if not args.pin_image_reference.is_file():
            raise FileNotFoundError(
                f"--pin-image-reference must be an image file, got: {args.pin_image_reference}"
            )
        if args.pin_image_alignment_metadata is not None:
            raise ValueError(
                "Use either --auto-pin-image-align or --pin-image-alignment-metadata, not both."
            )
    if args.pin_image_roi is not None:
        if args.pin_image_roi[2] <= 0 or args.pin_image_roi[3] <= 0:
            raise ValueError("--pin-image-roi width and height must be positive.")
        if args.pin_image_roi[0] < 0 or args.pin_image_roi[1] < 0:
            raise ValueError("--pin-image-roi x and y must be non-negative.")
    if args.pin_image_width <= 0 or args.pin_image_height <= 0:
        raise ValueError("--pin-image-width and --pin-image-height must be positive.")
    if args.pin_image_fps <= 0:
        raise ValueError("--pin-image-fps must be positive.")
    if args.pin_image_warmup_frames < 0:
        raise ValueError("--pin-image-warmup-frames must be non-negative.")
    if args.pin_image_timeout_ms <= 0:
        raise ValueError("--pin-image-timeout-ms must be positive.")
    if args.auto_pin_image_search_step_mm < 0.0:
        raise ValueError("--auto-pin-image-search-step-mm must be non-negative.")
    if args.auto_pin_image_search_step_deg < 0.0:
        raise ValueError("--auto-pin-image-search-step-deg must be non-negative.")
    if args.auto_pin_image_search_iterations < 1:
        raise ValueError("--auto-pin-image-search-iterations must be at least 1.")
    if args.auto_pin_image_search_max_mm < 0.0:
        raise ValueError("--auto-pin-image-search-max-mm must be non-negative.")
    if args.auto_pin_image_search_max_deg < 0.0:
        raise ValueError("--auto-pin-image-search-max-deg must be non-negative.")
    if args.auto_pin_image_settle_s < 0.0:
        raise ValueError("--auto-pin-image-settle-s must be non-negative.")
    if args.max_pin_image_correction_mm < 0.0:
        raise ValueError("--max-pin-image-correction-mm must be non-negative.")
    if args.max_pin_image_correction_deg < 0.0:
        raise ValueError("--max-pin-image-correction-deg must be non-negative.")
    if args.pin_insertion_speed_m_s <= 0.0:
        raise ValueError("--pin-insertion-speed-m-s must be positive.")
    if args.pin_insertion_acceleration_m_s2 <= 0.0:
        raise ValueError("--pin-insertion-acceleration-m-s2 must be positive.")
    if args.max_pin_insertion_mm < 0.0:
        raise ValueError("--max-pin-insertion-mm must be non-negative.")
    if args.post_insert_retreat_y_mm < 0.0:
        raise ValueError("--post-insert-retreat-y-mm must be non-negative.")
    if args.post_insert_shift_x_mm is not None and args.post_insert_shift_x_mm < 0.0:
        raise ValueError("--post-insert-shift-x-mm must be non-negative.")
    if args.post_two_pin_close_retreat_y_mm < 0.0:
        raise ValueError("--post-two-pin-close-retreat-y-mm must be non-negative.")
    if args.post_two_pin_close_shift_negative_x_mm < 0.0:
        raise ValueError("--post-two-pin-close-shift-negative-x-mm must be non-negative.")
    if args.post_two_pin_aligned_insert_y_mm < 0.0:
        raise ValueError("--post-two-pin-aligned-insert-y-mm must be non-negative.")
    if args.post_two_pin_release_retreat_y_mm < 0.0:
        raise ValueError("--post-two-pin-release-retreat-y-mm must be non-negative.")
    if not 0.0 < args.tray_handle_scale <= 1.0:
        raise ValueError("--tray-handle-scale must be > 0 and <= 1.")
    if not 0.0 <= args.gripper_close_percent <= 100.0:
        raise ValueError("--gripper-close-percent must be between 0 and 100.")
    if not 0 <= args.gripper_speed <= 255:
        raise ValueError("--gripper-speed must be between 0 and 255.")
    if not 0 <= args.gripper_force <= 255:
        raise ValueError("--gripper-force must be between 0 and 255.")
    if not 1 <= args.gripper_port <= 65535:
        raise ValueError("--gripper-port must be between 1 and 65535.")
    if args.offset_z_mm is not None and args.marker_frame_mode != "raw":
        raise ValueError("--offset-z-mm is only valid with --marker-frame-mode raw.")
    if args.tcp_offset_ur is not None:
        tcp_offset = np.asarray(args.tcp_offset_ur, dtype=float)
        if tcp_offset.shape != (6,) or not np.all(np.isfinite(tcp_offset)):
            raise ValueError("--tcp-offset-ur must contain six finite values.")
    if args.align_to_four_pin_frame and args.skip_gripper_close and not args.continue_pin_sequence_from_current:
        raise ValueError("--align-to-four-pin-frame requires the marker motion to close the gripper.")
    if args.continue_pin_sequence_from_current and not args.approach_pin_after_rotation:
        raise ValueError("--continue-pin-sequence-from-current requires pin approach to be enabled.")


def print_motion_speed_config(args: argparse.Namespace) -> None:
    print("Motion speed config:")
    print(
        "  initial moveJ: "
        f"speed={args.initial_speed_rad_s:.4f} rad/s, "
        f"acceleration={args.initial_acceleration_rad_s2:.4f} rad/s^2"
    )
    print(
        "  normal moveL: "
        f"speed={args.speed_m_s:.4f} m/s, "
        f"acceleration={args.acceleration_m_s2:.4f} m/s^2"
    )
    if args.insert_after_pin_approach:
        print(
            "  final insertion moveL: "
            f"speed={args.pin_insertion_speed_m_s:.4f} m/s, "
            f"acceleration={args.pin_insertion_acceleration_m_s2:.4f} m/s^2"
        )


def read_current_tcp_pose(robot_ip: str) -> tuple[Any, np.ndarray]:
    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError("ur_rtde is not installed. Run 'uv sync' from the repo root.") from exc

    rtde_receive = RTDEReceiveInterface(robot_ip)
    tcp_pose = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    if tcp_pose.shape != (6,):
        raise RuntimeError(f"Expected 6D TCP pose from RTDE, got {tcp_pose.shape}.")
    return rtde_receive, tcp_pose


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
        raise RuntimeError(f"Failed to set active TCP offset to {fmt(np.asarray(tcp_offset))}.")


def move_to_initial_pose(rtde_receive: Any, rtde_control: Any, args: argparse.Namespace) -> None:
    target_q_rad = np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
    current_q_rad = np.asarray(rtde_receive.getActualQ(), dtype=float)
    print("Moving to saved marker-motion initial pose:")
    print(f"  current q_deg: {fmt(np.rad2deg(current_q_rad), decimals=3)}")
    print(f"  initial q_deg: {fmt(np.asarray(args.initial_q_deg), decimals=3)}")
    print(
        f"  speed={args.initial_speed_rad_s:.4f} rad/s, "
        f"acceleration={args.initial_acceleration_rad_s2:.4f} rad/s^2"
    )
    success = rtde_control.moveJ(
        target_q_rad.tolist(),
        args.initial_speed_rad_s,
        args.initial_acceleration_rad_s2,
        False,
    )
    if not success:
        raise RuntimeError("Initial pose moveJ returned false.")
    if args.initial_dwell_s > 0.0:
        time.sleep(args.initial_dwell_s)


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def compute_base_t_four_pin_frame(args: argparse.Namespace, base_t_marker: np.ndarray) -> np.ndarray:
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
    return np.asarray(base_t_pin[:3, 3], dtype=float) + np.asarray(base_t_pin[:3, :3], dtype=float) @ offset_in_pin


def effective_post_insert_shift_x_mm(args: argparse.Namespace) -> float:
    if args.post_insert_shift_x_mm is not None:
        return float(args.post_insert_shift_x_mm)
    return adjacent_two_pin_center_shift_x_mm(args)


def pose_translated_in_tcp_frame(current_tcp_pose_ur: np.ndarray, offset_tcp_m: np.ndarray) -> np.ndarray:
    pose = np.asarray(current_tcp_pose_ur, dtype=float).copy()
    rotation = Rotation.from_rotvec(pose[3:6]).as_matrix()
    pose[:3] += rotation @ np.asarray(offset_tcp_m, dtype=float)
    return pose


def pose_vector_to_transform(pose_ur: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(pose_ur[:3], dtype=float)
    transform[:3, :3] = Rotation.from_rotvec(pose_ur[3:6]).as_matrix()
    return transform


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ np.asarray(points, dtype=float).T).T + transform[:3, 3]


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return np.asarray(vector, dtype=float) / norm


def load_virtual_tray_geometry(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load(resolve_project_path(args.tray_obj), force="mesh", process=False)
    handle_root_y_m = args.tray_handle_root_y_mm / 1000.0
    shorten_lower_tray_handle(
        mesh,
        handle_root_y_m,
        args.tray_handle_scale,
    )
    tray_attachment_local = shortened_lower_handle_point(
        TI_TRAY_PROTRUSION_CENTER_M,
        handle_root_y_m,
        args.tray_handle_scale,
    )
    return np.asarray(mesh.vertices, dtype=float), tray_attachment_local


def tray_vertices_in_base(
    tcp_pose_ur: np.ndarray,
    tray_vertices: np.ndarray,
    tray_attachment_local: np.ndarray,
    tray_center_tcp_z_mm: float,
    tray_local_rx_deg: float,
) -> np.ndarray:
    base_t_tray = (
        pose_vector_to_transform(tcp_pose_ur)
        @ translation_transform([0.0, 0.0, tray_center_tcp_z_mm / 1000.0])
        @ rotation_transform("x", tray_local_rx_deg)
        @ translation_transform(-tray_attachment_local)
    )
    return transform_points(base_t_tray, tray_vertices)


def tray_reference_position(tcp_pose_ur: np.ndarray, tray_reference_tcp_z_mm: float) -> np.ndarray:
    pose = np.asarray(tcp_pose_ur, dtype=float)
    rotation = Rotation.from_rotvec(pose[3:6]).as_matrix()
    return pose[:3] + rotation @ np.asarray([0.0, 0.0, tray_reference_tcp_z_mm / 1000.0])


def tray_center_position(tcp_pose_ur: np.ndarray, tray_center_tcp_z_mm: float) -> np.ndarray:
    return tray_reference_position(tcp_pose_ur, tray_center_tcp_z_mm)


def tray_attachment_local(args: argparse.Namespace) -> np.ndarray:
    return shortened_lower_handle_point(
        TI_TRAY_PROTRUSION_CENTER_M,
        args.tray_handle_root_y_mm / 1000.0,
        args.tray_handle_scale,
    )


def tray_four_hole_center_offset_tcp(args: argparse.Namespace) -> np.ndarray:
    tray_r_reference = Rotation.from_euler("x", args.tray_local_rx_deg, degrees=True).as_matrix()
    return (
        np.asarray([0.0, 0.0, args.tray_center_tcp_z_mm / 1000.0], dtype=float)
        + tray_r_reference @ (TI_TRAY_FOUR_PIN_HOLE_CENTER_M - tray_attachment_local(args))
    )


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


def pin_approach_target_pose(
    current_tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
    clearance_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_pose = np.asarray(current_tcp_pose_ur, dtype=float)
    current_tray_reference = tray_four_hole_center_position(current_pose, args)
    pin_rotation = np.asarray(base_t_pin[:3, :3], dtype=float)
    pin_origin = np.asarray(base_t_pin[:3, 3], dtype=float)
    target_reference_in_pin = np.asarray(
        [0.0, float(clearance_mm) / 1000.0, 0.0],
        dtype=float,
    )
    target_tray_reference = pin_origin + pin_rotation @ target_reference_in_pin
    translation = target_tray_reference - current_tray_reference
    target_pose = current_pose.copy()
    target_pose[:3] += translation
    return target_pose, current_tray_reference, target_tray_reference, translation


def pin_centering_target_pose(
    current_tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_pose = np.asarray(current_tcp_pose_ur, dtype=float)
    current_tray_reference = tray_four_hole_center_position(current_pose, args)
    pin_rotation = np.asarray(base_t_pin[:3, :3], dtype=float)
    pin_origin = np.asarray(base_t_pin[:3, 3], dtype=float)
    current_reference_in_pin = pin_rotation.T @ (current_tray_reference - pin_origin)
    target_reference_in_pin = np.asarray(
        [0.0, current_reference_in_pin[1], 0.0],
        dtype=float,
    )
    target_tray_reference = pin_origin + pin_rotation @ target_reference_in_pin
    translation = target_tray_reference - current_tray_reference
    target_pose = current_pose.copy()
    target_pose[:3] += translation
    return target_pose, current_tray_reference, target_tray_reference, translation


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
    current_tcp_pose_ur: np.ndarray,
    correction: dict[str, Any],
) -> np.ndarray:
    current_pose = np.asarray(current_tcp_pose_ur, dtype=float)
    current_rotation = Rotation.from_rotvec(current_pose[3:6])
    local_translation_m = np.asarray(
        [
            float(correction["tcp_local_x_offset_mm"]) / 1000.0,
            0.0,
            float(correction["tcp_local_z_offset_mm"]) / 1000.0,
        ],
        dtype=float,
    )
    local_rotation = Rotation.from_euler(
        str(correction["axis"]),
        float(correction["angle_deg"]),
        degrees=True,
    )
    target_pose = current_pose.copy()
    target_pose[:3] = current_pose[:3] + current_rotation.as_matrix() @ local_translation_m
    target_pose[3:6] = (current_rotation * local_rotation).as_rotvec()
    return target_pose


def print_pin_image_alignment_correction_plan(
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
    title: str = "Post-approach image alignment correction",
    metadata_path: Path | None = None,
) -> np.ndarray | None:
    if metadata_path is None:
        metadata_path = effective_pin_image_alignment_metadata(args)
    if metadata_path is None:
        return None

    correction = load_pin_image_alignment_correction(metadata_path)
    correction_xy_mm = float(
        np.linalg.norm(
            [
                correction["tcp_local_x_offset_mm"],
                correction["tcp_local_z_offset_mm"],
            ]
        )
    )
    if correction_xy_mm > args.max_pin_image_correction_mm:
        raise RuntimeError(
            f"Image alignment correction is {correction_xy_mm:.3f} mm, above "
            f"--max-pin-image-correction-mm {args.max_pin_image_correction_mm:.3f}."
        )
    if abs(float(correction["angle_deg"])) > args.max_pin_image_correction_deg:
        raise RuntimeError(
            f"Image alignment correction is {correction['angle_deg']:.3f} deg, above "
            f"--max-pin-image-correction-deg {args.max_pin_image_correction_deg:.3f}."
        )

    target_pose = pose_with_tcp_local_alignment_correction(start_tcp_pose_ur, correction)
    translation_mm = (target_pose[:3] - np.asarray(start_tcp_pose_ur, dtype=float)[:3]) * 1000.0
    print(f"{title}:")
    print(f"  metadata: {metadata_path}")
    print(f"  source label: {correction.get('label')}")
    print(f"  local rotation axis: TCP {correction['axis'].upper()}")
    print(f"  local rotation deg: {correction['angle_deg']:.3f}")
    print(
        "  TCP local translation mm: "
        f"[x={correction['tcp_local_x_offset_mm']:.3f}, "
        f"y=0.000, z={correction['tcp_local_z_offset_mm']:.3f}]"
    )
    print(f"  base translation mm: {fmt(translation_mm, decimals=3)}")
    print(f"  target_tcp_pose_ur: {fmt(target_pose)}")
    return target_pose


def print_post_two_pin_close_image_alignment_plan(
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray | None:
    if not args.post_two_pin_close_image_align:
        print("Post-two-pin-close image alignment: disabled.")
        return None
    if args.auto_pin_image_align:
        print(
            "Post-two-pin-close image alignment: automatic search will run during "
            "--execute after the final TCP-local move; dry-run cannot know the final "
            "searched correction."
        )
        return None
    return print_pin_image_alignment_correction_plan(
        start_tcp_pose_ur,
        args,
        title="Post-two-pin-close image alignment correction",
        metadata_path=effective_post_two_pin_close_image_alignment_metadata(args),
    )


def post_two_pin_aligned_insert_release_target_poses(
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    insert_pose = pose_translated_in_tcp_frame(
        start_tcp_pose_ur,
        np.asarray([0.0, -args.post_two_pin_aligned_insert_y_mm / 1000.0, 0.0], dtype=float),
    )
    release_retreat_pose = pose_translated_in_tcp_frame(
        insert_pose,
        np.asarray([0.0, args.post_two_pin_release_retreat_y_mm / 1000.0, 0.0], dtype=float),
    )
    return insert_pose, release_retreat_pose


def print_post_two_pin_aligned_insert_release_plan(
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray]]:
    if not args.post_two_pin_aligned_insert_release:
        print("Post-two-pin aligned insert/release: disabled.")
        return []

    insert_pose, release_retreat_pose = post_two_pin_aligned_insert_release_target_poses(
        start_tcp_pose_ur,
        args,
    )
    print("Post-two-pin aligned insert/release:")
    print("  sequence: TCP local -Y insertion -> fully open gripper -> TCP local +Y retreat")
    print(f"  TCP local -Y insertion mm: {args.post_two_pin_aligned_insert_y_mm:.3f}")
    print(
        "  insertion speed/acceleration: "
        f"{args.pin_insertion_speed_m_s:.4f} m/s, "
        f"{args.pin_insertion_acceleration_m_s2:.4f} m/s^2"
    )
    print(f"  TCP local +Y release retreat mm: {args.post_two_pin_release_retreat_y_mm:.3f}")
    print(f"  aligned_insert_tcp_pose_ur: {fmt(insert_pose)}")
    print(f"  release_retreat_tcp_pose_ur: {fmt(release_retreat_pose)}")
    return [
        ("post-two-pin aligned TCP -Y insertion", insert_pose),
        ("post-two-pin release TCP +Y retreat", release_retreat_pose),
    ]


def pin_insertion_target_pose(
    current_tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
    target_y_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current_pose = np.asarray(current_tcp_pose_ur, dtype=float)
    current_tray_reference = tray_four_hole_center_position(current_pose, args)
    pin_rotation = np.asarray(base_t_pin[:3, :3], dtype=float)
    pin_origin = np.asarray(base_t_pin[:3, 3], dtype=float)
    current_reference_in_pin = pin_rotation.T @ (current_tray_reference - pin_origin)
    target_reference_in_pin = current_reference_in_pin.copy()
    target_reference_in_pin[1] = float(target_y_mm) / 1000.0
    target_tray_reference = pin_origin + pin_rotation @ target_reference_in_pin
    translation = target_tray_reference - current_tray_reference
    target_pose = current_pose.copy()
    target_pose[:3] += translation
    return (
        target_pose,
        current_tray_reference,
        target_tray_reference,
        translation,
        current_reference_in_pin,
        target_reference_in_pin,
    )


def print_pin_insertion_plan(
    start_tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    (
        target_pose,
        current_tray_reference,
        target_tray_reference,
        translation,
        current_reference_in_pin,
        target_reference_in_pin,
    ) = pin_insertion_target_pose(
        start_tcp_pose_ur,
        base_t_pin,
        args,
        args.pin_insertion_target_y_mm,
    )
    insertion_distance_mm = float(np.linalg.norm(translation) * 1000.0)
    if insertion_distance_mm > args.max_pin_insertion_mm:
        raise RuntimeError(
            f"Final pin insertion is {insertion_distance_mm:.1f} mm, above "
            f"--max-pin-insertion-mm {args.max_pin_insertion_mm:.1f}."
        )

    print("Final pin insertion descend:")
    print("  preserves current tray 4-hole center X/Z in the holder four-pin frame")
    print(
        "  tray 4-hole center in pin frame before insertion mm: "
        f"{fmt(1000.0 * current_reference_in_pin, decimals=3)}"
    )
    print(
        "  tray 4-hole center in pin frame target mm: "
        f"{fmt(1000.0 * target_reference_in_pin, decimals=3)}"
    )
    print(f"  target pin-frame Y mm: {args.pin_insertion_target_y_mm:.3f}")
    print(f"  tray 4-hole center before insertion base m: {fmt(current_tray_reference)}")
    print(f"  target tray 4-hole center base m: {fmt(target_tray_reference)}")
    print(f"  insertion translation base mm: {fmt(1000.0 * translation, decimals=3)}")
    print(f"  insertion distance mm: {insertion_distance_mm:.3f}")
    print(
        "  insertion speed/acceleration: "
        f"{args.pin_insertion_speed_m_s:.4f} m/s, "
        f"{args.pin_insertion_acceleration_m_s2:.4f} m/s^2"
    )
    print(f"  target_tcp_pose_ur: {fmt(target_pose)}")
    return target_pose


def post_insert_retreat_target_poses(
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, float]:
    shift_x_mm = effective_post_insert_shift_x_mm(args)
    retreat_y_pose = pose_translated_in_tcp_frame(
        start_tcp_pose_ur,
        np.asarray([0.0, args.post_insert_retreat_y_mm / 1000.0, 0.0], dtype=float),
    )
    shift_x_pose = pose_translated_in_tcp_frame(
        retreat_y_pose,
        np.asarray([shift_x_mm / 1000.0, 0.0, 0.0], dtype=float),
    )
    return retreat_y_pose, shift_x_pose, shift_x_mm


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
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
    base_t_pin: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    if not args.post_insert_release_retreat:
        print("Post-insert release/retreat: disabled.")
        return []

    retreat_y_pose, shift_x_pose, shift_x_mm = post_insert_retreat_target_poses(
        start_tcp_pose_ur,
        args,
    )
    print("Post-insert release/retreat:")
    print("  sequence: fully open gripper -> TCP local +Y retreat -> TCP local +X shift")
    print(f"  TCP local +Y retreat mm: {args.post_insert_retreat_y_mm:.3f}")
    if args.post_insert_shift_x_mm is None:
        print(
            "  TCP local +X shift mm: "
            f"{shift_x_mm:.3f} (computed from holder adjacent two-pin center)"
        )
    else:
        print(f"  TCP local +X shift mm: {shift_x_mm:.3f} (CLI override)")
    print(f"  retreat_y_tcp_pose_ur: {fmt(retreat_y_pose)}")
    print(f"  shift_x_tcp_pose_ur: {fmt(shift_x_pose)}")
    targets = [
        ("post-insert TCP +Y retreat", retreat_y_pose),
        ("post-insert TCP +X adjacent-pin shift", shift_x_pose),
    ]
    if args.post_insert_move_to_two_pin:
        if base_t_pin is None:
            raise RuntimeError("Adjacent two-pin TCP target needs the holder four-pin frame.")
        two_pin_pose, adjacent_center, target_position, translation = adjacent_two_pin_tcp_target_pose(
            shift_x_pose,
            base_t_pin,
            args,
        )
        adjacent_in_pin = base_t_pin[:3, :3].T @ (adjacent_center - base_t_pin[:3, 3])
        target_in_pin = base_t_pin[:3, :3].T @ (target_position - base_t_pin[:3, 3])
        print("  then move TCP origin to adjacent two-pin target and close gripper")
        print(
            "  adjacent two-pin center in holder pin frame mm: "
            f"{fmt(1000.0 * adjacent_in_pin, decimals=3)}"
        )
        print(
            "  adjacent two-pin target in holder pin frame mm: "
            f"{fmt(1000.0 * target_in_pin, decimals=3)}"
        )
        print(f"  adjacent two-pin target +Y offset mm: {args.post_insert_two_pin_target_y_mm:.3f}")
        print(f"  adjacent two-pin center base m: {fmt(adjacent_center)}")
        print(f"  adjacent two-pin target base m: {fmt(target_position)}")
        print(f"  move-to-two-pin translation base mm: {fmt(1000.0 * translation, decimals=3)}")
        print(f"  two_pin_tcp_pose_ur: {fmt(two_pin_pose)}")
        targets.append(("post-insert move TCP to adjacent two-pin center", two_pin_pose))
    else:
        print("  move to adjacent two-pin center: disabled")
    return targets


def post_two_pin_close_retreat_target_poses(
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    retreat_y_pose = pose_translated_in_tcp_frame(
        start_tcp_pose_ur,
        np.asarray([0.0, args.post_two_pin_close_retreat_y_mm / 1000.0, 0.0], dtype=float),
    )
    shift_negative_x_pose = pose_translated_in_tcp_frame(
        retreat_y_pose,
        np.asarray(
            [-args.post_two_pin_close_shift_negative_x_mm / 1000.0, 0.0, 0.0],
            dtype=float,
        ),
    )
    return retreat_y_pose, shift_negative_x_pose


def print_post_two_pin_close_retreat_plan(
    start_tcp_pose_ur: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray]]:
    if not args.post_two_pin_close_retreat:
        print("Post-two-pin-close retreat: disabled.")
        return []

    retreat_y_pose, shift_negative_x_pose = post_two_pin_close_retreat_target_poses(
        start_tcp_pose_ur,
        args,
    )
    print("Post-two-pin-close retreat:")
    print("  sequence: TCP local +Y retreat -> TCP local -X shift")
    print(f"  TCP local +Y retreat mm: {args.post_two_pin_close_retreat_y_mm:.3f}")
    print(f"  TCP local -X shift mm: {args.post_two_pin_close_shift_negative_x_mm:.3f}")
    print(f"  post_close_retreat_y_tcp_pose_ur: {fmt(retreat_y_pose)}")
    print(f"  post_close_shift_negative_x_tcp_pose_ur: {fmt(shift_negative_x_pose)}")
    return [
        ("post-two-pin-close TCP +Y retreat", retreat_y_pose),
        ("post-two-pin-close TCP -X shift", shift_negative_x_pose),
    ]


def pose_with_tcp_local_xz_rotation_adjustment(
    start_tcp_pose_ur: np.ndarray,
    axis: str,
    angle_deg: float,
    x_offset_mm: float,
    z_offset_mm: float,
) -> np.ndarray:
    start_pose = np.asarray(start_tcp_pose_ur, dtype=float)
    start_rotation = Rotation.from_rotvec(start_pose[3:6])
    local_translation_m = np.asarray([x_offset_mm, 0.0, z_offset_mm], dtype=float) / 1000.0
    local_rotation = Rotation.from_euler(axis, angle_deg, degrees=True)
    target_pose = start_pose.copy()
    target_pose[:3] = start_pose[:3] + start_rotation.as_matrix() @ local_translation_m
    target_pose[3:6] = (start_rotation * local_rotation).as_rotvec()
    return target_pose


def normalize_alignment_gray(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float32)
    image = image - float(np.mean(image))
    norm = float(np.linalg.norm(image))
    if norm < 1e-9:
        return np.zeros_like(image, dtype=np.float32)
    return image / norm


def crop_alignment_roi(image: np.ndarray, roi: list[int] | tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return image
    x, y, width, height = [int(value) for value in roi]
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError(
            f"--pin-image-roi {roi} exceeds image size {image.shape[1]}x{image.shape[0]}."
        )
    return image[y : y + height, x : x + width]


def load_reference_alignment_image(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    if args.pin_image_reference is None:
        raise RuntimeError("--pin-image-reference is required for automatic image alignment.")
    reference_bgr = cv2.imread(str(args.pin_image_reference), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        raise RuntimeError(f"Failed to read reference image: {args.pin_image_reference}")
    if reference_bgr.shape[:2] != (args.pin_image_height, args.pin_image_width):
        raise RuntimeError(
            f"Reference image is {reference_bgr.shape[1]}x{reference_bgr.shape[0]}, but capture "
            f"is configured for {args.pin_image_width}x{args.pin_image_height}."
        )
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    reference_roi = crop_alignment_roi(reference_gray, args.pin_image_roi)
    reference_roi = cv2.equalizeHist(reference_roi)
    return reference_bgr, normalize_alignment_gray(reference_roi)


def score_alignment_image(
    image_bgr: np.ndarray,
    reference_gray_normalized: np.ndarray,
    args: argparse.Namespace,
) -> float:
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = crop_alignment_roi(gray, args.pin_image_roi)
    roi = cv2.equalizeHist(roi)
    roi_normalized = normalize_alignment_gray(roi)
    if roi_normalized.shape != reference_gray_normalized.shape:
        raise RuntimeError(
            f"Alignment ROI shape mismatch: {roi_normalized.shape} vs "
            f"{reference_gray_normalized.shape}."
        )
    return float(np.sum(reference_gray_normalized * roi_normalized))


def start_pin_image_pipeline(args: argparse.Namespace) -> Any:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if args.pin_image_serial:
        config.enable_device(args.pin_image_serial)
    config.enable_stream(
        rs.stream.color,
        args.pin_image_width,
        args.pin_image_height,
        rs.format.bgr8,
        args.pin_image_fps,
    )
    pipeline.start(config)
    return pipeline


def warmup_pin_image_pipeline(pipeline: Any, args: argparse.Namespace) -> None:
    for _ in range(args.pin_image_warmup_frames):
        pipeline.wait_for_frames(args.pin_image_timeout_ms)


def capture_pin_image(pipeline: Any, args: argparse.Namespace) -> np.ndarray:
    frames = pipeline.wait_for_frames(args.pin_image_timeout_ms)
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("Timed out before receiving a RealSense color frame.")
    return np.asanyarray(color_frame.get_data())


def auto_alignment_output_dir(args: argparse.Namespace) -> Path:
    if args.pin_image_output_dir is not None:
        output_dir = args.pin_image_output_dir
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"images/auto_pin_image_alignment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_auto_alignment_capture(
    output_dir: Path,
    image_bgr: np.ndarray,
    iteration: int,
    candidate_index: int,
    state: tuple[float, float, float],
    score: float,
    pose_ur: np.ndarray,
    args: argparse.Namespace,
    label: str = "candidate",
) -> tuple[Path, Path]:
    import cv2

    angle_deg, x_offset_mm, z_offset_mm = state
    safe_angle = f"{angle_deg:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    safe_x = f"{x_offset_mm:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    safe_z = f"{z_offset_mm:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    prefix = f"{label}_i{iteration:02d}_c{candidate_index:02d}_{safe_angle}deg_x{safe_x}mm_z{safe_z}mm"
    color_path = output_dir / f"{prefix}_color.png"
    metadata_path = output_dir / f"{prefix}_metadata.yaml"
    latest_path = output_dir / "latest_color.png"
    cv2.imwrite(str(color_path), image_bgr)
    cv2.imwrite(str(latest_path), image_bgr)
    metadata = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": label,
        "iteration": int(iteration),
        "candidate_index": int(candidate_index),
        "score": float(score),
        "axis": args.axis if hasattr(args, "axis") else "y",
        "angle_deg": float(angle_deg),
        "tcp_local_x_offset_mm": float(x_offset_mm),
        "tcp_local_z_offset_mm": float(z_offset_mm),
        "target_tcp_pose_ur": round_list(pose_ur),
        "reference_image": str(args.pin_image_reference),
        "roi": list(args.pin_image_roi) if args.pin_image_roi is not None else None,
        "files": {
            "color": str(color_path),
            "latest_color": str(latest_path),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)
    return color_path, metadata_path


def unique_alignment_candidates(
    center: tuple[float, float, float],
    step_deg: float,
    step_mm: float,
    max_deg: float,
    max_mm: float,
) -> list[tuple[float, float, float]]:
    angle_deg, x_mm, z_mm = center
    candidates = [center]
    if step_deg > 0.0:
        candidates.extend(
            [
                (angle_deg + step_deg, x_mm, z_mm),
                (angle_deg - step_deg, x_mm, z_mm),
            ]
        )
    if step_mm > 0.0:
        candidates.extend(
            [
                (angle_deg, x_mm + step_mm, z_mm),
                (angle_deg, x_mm - step_mm, z_mm),
                (angle_deg, x_mm, z_mm + step_mm),
                (angle_deg, x_mm, z_mm - step_mm),
            ]
        )
    unique: list[tuple[float, float, float]] = []
    seen: set[tuple[float, float, float]] = set()
    for candidate in candidates:
        key = tuple(round(float(value), 6) for value in candidate)
        if key in seen:
            continue
        if abs(candidate[0]) > max_deg + 1e-9:
            continue
        if abs(candidate[1]) > max_mm + 1e-9 or abs(candidate[2]) > max_mm + 1e-9:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def execute_auto_pin_image_alignment(
    rtde_receive: Any,
    rtde_control: Any,
    args: argparse.Namespace,
    title: str = "Automatic post-approach image alignment",
) -> np.ndarray:
    if not args.auto_pin_image_align:
        return np.asarray(rtde_receive.getActualTCPPose(), dtype=float)

    _reference_bgr, reference_gray = load_reference_alignment_image(args)
    output_dir = auto_alignment_output_dir(args)
    start_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    start_q_rad = np.asarray(rtde_receive.getActualQ(), dtype=float)
    best_state = (0.0, 0.0, 0.0)
    best_pose = start_pose_ur.copy()
    best_score = -float("inf")
    best_image = None
    step_mm = float(args.auto_pin_image_search_step_mm)
    step_deg = float(args.auto_pin_image_search_step_deg)

    print(f"{title}:")
    print(f"  reference image: {args.pin_image_reference}")
    print(f"  output_dir: {output_dir}")
    print(f"  ROI: {args.pin_image_roi}")
    print(
        f"  search bounds: x/z <= {args.auto_pin_image_search_max_mm:.3f} mm, "
        f"rotation <= {args.auto_pin_image_search_max_deg:.3f} deg"
    )
    print(f"  initial steps: {step_mm:.3f} mm, {step_deg:.3f} deg")

    pipeline = start_pin_image_pipeline(args)
    try:
        warmup_pin_image_pipeline(pipeline, args)
        for iteration in range(args.auto_pin_image_search_iterations):
            candidates = unique_alignment_candidates(
                best_state,
                step_deg,
                step_mm,
                args.auto_pin_image_search_max_deg,
                args.auto_pin_image_search_max_mm,
            )
            iteration_best_state = best_state
            iteration_best_pose = best_pose
            iteration_best_score = best_score
            print(
                f"  iteration {iteration + 1}/{args.auto_pin_image_search_iterations}: "
                f"{len(candidates)} candidates"
            )
            for candidate_index, candidate in enumerate(candidates):
                candidate_pose = pose_with_tcp_local_xz_rotation_adjustment(
                    start_pose_ur,
                    "y",
                    candidate[0],
                    candidate[1],
                    candidate[2],
                )
                require_ik_solution(
                    rtde_control,
                    start_q_rad,
                    candidate_pose,
                    "automatic image alignment candidate",
                )
                execute_continuous_movel(
                    rtde_receive,
                    rtde_control,
                    candidate_pose,
                    args,
                    motion_name=(
                        "auto image alignment candidate "
                        f"i{iteration + 1} c{candidate_index + 1}"
                    ),
                )
                if args.auto_pin_image_settle_s > 0.0:
                    time.sleep(args.auto_pin_image_settle_s)
                image_bgr = capture_pin_image(pipeline, args)
                score = score_alignment_image(image_bgr, reference_gray, args)
                save_auto_alignment_capture(
                    output_dir,
                    image_bgr,
                    iteration + 1,
                    candidate_index + 1,
                    candidate,
                    score,
                    candidate_pose,
                    args,
                )
                print(
                    "    candidate "
                    f"angle={candidate[0]:+.3f} deg, x={candidate[1]:+.3f} mm, "
                    f"z={candidate[2]:+.3f} mm, score={score:.6f}"
                )
                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best_state = candidate
                    iteration_best_pose = candidate_pose
                    best_image = image_bgr

            best_state = iteration_best_state
            best_pose = iteration_best_pose
            best_score = iteration_best_score
            print(
                f"  best after iteration {iteration + 1}: "
                f"angle={best_state[0]:+.3f} deg, x={best_state[1]:+.3f} mm, "
                f"z={best_state[2]:+.3f} mm, score={best_score:.6f}"
            )
            step_mm *= 0.5
            step_deg *= 0.5

        execute_continuous_movel(
            rtde_receive,
            rtde_control,
            best_pose,
            args,
            motion_name="automatic image alignment best pose",
        )
        if best_image is None:
            best_image = capture_pin_image(pipeline, args)
        best_color_path, best_metadata_path = save_auto_alignment_capture(
            output_dir,
            best_image,
            args.auto_pin_image_search_iterations,
            0,
            best_state,
            best_score,
            best_pose,
            args,
            label="best",
        )
        print(f"  automatic alignment best image: {best_color_path}")
        print(f"  automatic alignment metadata: {best_metadata_path}")
        return np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    finally:
        pipeline.stop()


def tray_floor_clearance_m(
    tcp_pose_ur: np.ndarray,
    tray_vertices: np.ndarray,
    tray_attachment_local: np.ndarray,
    floor_z_m: float,
    tray_center_tcp_z_mm: float,
    tray_local_rx_deg: float,
) -> float:
    vertices_base = tray_vertices_in_base(
        tcp_pose_ur,
        tray_vertices,
        tray_attachment_local,
        tray_center_tcp_z_mm,
        tray_local_rx_deg,
    )
    # In this workcell base +Z is physical down, so floor contact is max base z.
    return floor_z_m - float(np.max(vertices_base[:, 2]))


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


def rotation_delta_deg(current_tcp_pose_ur: np.ndarray, target_tcp_pose_ur: np.ndarray) -> float:
    current_rotation = Rotation.from_rotvec(current_tcp_pose_ur[3:6]).as_matrix()
    target_rotation = Rotation.from_rotvec(target_tcp_pose_ur[3:6]).as_matrix()
    return float(np.rad2deg(Rotation.from_matrix(target_rotation @ current_rotation.T).magnitude()))


def four_pin_alignment_plan(
    tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    floor_z_m: float,
    args: argparse.Namespace,
    minimum_lift_m: float = 0.0,
) -> dict[str, Any]:
    tray_vertices, tray_attachment_local = load_virtual_tray_geometry(args)
    target_tcp_pose_ur = make_four_pin_rotation_target(
        tcp_pose_ur,
        base_t_pin,
        args.tcp_rotation_offset_rpy_deg,
    )
    (
        target_tcp_pose_ur,
        rotation_approach_vector_m,
        pre_approach_lateral_distance_m,
        post_approach_lateral_distance_m,
    ) = move_target_tray_center_toward_pin(
        target_tcp_pose_ur,
        base_t_pin,
        args,
        args.rotation_approach_mm,
    )
    angle_delta_deg = rotation_delta_deg(tcp_pose_ur, target_tcp_pose_ur)
    current_clearance_m = tray_floor_clearance_m(
        tcp_pose_ur,
        tray_vertices,
        tray_attachment_local,
        floor_z_m,
        args.tray_center_tcp_z_mm,
        args.tray_local_rx_deg,
    )
    target_clearance_m = tray_floor_clearance_m(
        target_tcp_pose_ur,
        tray_vertices,
        tray_attachment_local,
        floor_z_m,
        args.tray_center_tcp_z_mm,
        args.tray_local_rx_deg,
    )
    requested_clearance_m = args.safe_floor_clearance_mm / 1000.0
    safety_lift_m = max(
        args.post_grasp_lift_mm / 1000.0,
        minimum_lift_m,
        requested_clearance_m - current_clearance_m,
        requested_clearance_m - target_clearance_m,
        0.0,
    )
    return {
        "target_tcp_pose_ur": target_tcp_pose_ur,
        "lifted_current_tcp_pose_ur": with_base_z_lift(tcp_pose_ur, safety_lift_m),
        "lifted_target_tcp_pose_ur": with_base_z_lift(target_tcp_pose_ur, safety_lift_m),
        "angle_delta_deg": angle_delta_deg,
        "current_clearance_m": current_clearance_m,
        "target_clearance_m": target_clearance_m,
        "safety_lift_m": safety_lift_m,
        "rotation_approach_vector_m": rotation_approach_vector_m,
        "pre_approach_lateral_distance_m": pre_approach_lateral_distance_m,
        "post_approach_lateral_distance_m": post_approach_lateral_distance_m,
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
    q_path: list[np.ndarray] = []
    minimum_q3_margin_deg = float("inf")
    minimum_q5_margin_deg = float("inf")
    maximum_joint_step_deg = 0.0

    all_waypoints = lift_waypoints + rotation_waypoints
    for waypoint_index, pose in enumerate(all_waypoints, start=1):
        pose_list = pose.tolist()
        has_solution = rtde_control.getInverseKinematicsHasSolution(
            pose_list,
            q_near.tolist(),
            IK_POSITION_TOLERANCE_M,
            IK_ORIENTATION_TOLERANCE_RAD,
        )
        if not has_solution:
            raise RuntimeError(f"No IK solution at alignment waypoint {waypoint_index}.")

        q_solution = np.asarray(
            rtde_control.getInverseKinematics(
                pose_list,
                q_near.tolist(),
                IK_POSITION_TOLERANCE_M,
                IK_ORIENTATION_TOLERANCE_RAD,
            ),
            dtype=float,
        )
        if q_solution.shape != (6,) or not np.all(np.isfinite(q_solution)):
            raise RuntimeError(f"Invalid IK result at alignment waypoint {waypoint_index}.")
        q_solution = unwrap_joints_near(q_solution, q_near)

        joint_step_deg = float(np.max(np.abs(np.rad2deg(q_solution - q_near))))
        q3_margin_deg = distance_to_nearest_pi_multiple_deg(q_solution[2])
        q5_margin_deg = distance_to_nearest_pi_multiple_deg(q_solution[4])
        maximum_joint_step_deg = max(maximum_joint_step_deg, joint_step_deg)
        minimum_q3_margin_deg = min(minimum_q3_margin_deg, q3_margin_deg)
        minimum_q5_margin_deg = min(minimum_q5_margin_deg, q5_margin_deg)

        if joint_step_deg > MAX_IK_JOINT_STEP_DEG:
            raise RuntimeError(
                f"IK branch jump of {joint_step_deg:.1f} deg at alignment waypoint "
                f"{waypoint_index}."
            )
        if q3_margin_deg < MIN_Q3_SINGULARITY_MARGIN_DEG:
            raise RuntimeError(
                f"q3 singularity margin is {q3_margin_deg:.1f} deg at alignment waypoint "
                f"{waypoint_index}."
            )
        if q5_margin_deg < MIN_Q5_SINGULARITY_MARGIN_DEG:
            raise RuntimeError(
                f"q5 singularity margin is {q5_margin_deg:.1f} deg at alignment waypoint "
                f"{waypoint_index}."
            )

        q_path.append(q_solution)
        q_near = q_solution

    return {
        "q_path": q_path,
        "minimum_q3_margin_deg": minimum_q3_margin_deg,
        "minimum_q5_margin_deg": minimum_q5_margin_deg,
        "maximum_joint_step_deg": maximum_joint_step_deg,
    }


def select_safe_alignment_plan(
    rtde_control: Any,
    grasped_tcp_pose_ur: np.ndarray,
    q_start_rad: np.ndarray,
    base_t_pin: np.ndarray,
    floor_z_m: float,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    failures: list[str] = []
    safe_candidates: list[
        tuple[dict[str, Any], list[np.ndarray], list[np.ndarray], dict[str, Any]]
    ] = []
    for lift_mm in ROTATION_LIFT_CANDIDATES_MM:
        plan = four_pin_alignment_plan(
            grasped_tcp_pose_ur,
            base_t_pin,
            floor_z_m,
            args,
            minimum_lift_m=lift_mm / 1000.0,
        )
        actual_lift_mm = 1000.0 * plan["safety_lift_m"]
        if plan["safety_lift_m"] > args.max_safety_lift_mm / 1000.0:
            failures.append(f"{actual_lift_mm:.1f} mm lift exceeds safety limit")
            continue

        lift_waypoints, rotation_waypoints = alignment_waypoints(grasped_tcp_pose_ur, plan)
        try:
            ik_report = preflight_alignment_ik(
                rtde_control,
                q_start_rad,
                lift_waypoints,
                rotation_waypoints,
            )
        except RuntimeError as exc:
            failures.append(f"{actual_lift_mm:.1f} mm lift: {exc}")
            continue

        print(
            f"Safe {actual_lift_mm:.1f} mm rotation-lift candidate: "
            f"min q3 margin={ik_report['minimum_q3_margin_deg']:.1f} deg, "
            f"min q5 margin={ik_report['minimum_q5_margin_deg']:.1f} deg."
        )
        safe_candidates.append((plan, lift_waypoints, rotation_waypoints, ik_report))

    if safe_candidates:
        selected = min(
            safe_candidates,
            key=lambda candidate: (
                candidate[0]["safety_lift_m"],
                -candidate[3]["minimum_q3_margin_deg"],
                -candidate[3]["minimum_q5_margin_deg"],
            ),
        )
        print(
            f"Selected {1000.0 * selected[0]['safety_lift_m']:.1f} mm rotation lift "
            "with the smallest safe lift."
        )
        return selected

    details = "\n  - ".join(failures)
    raise RuntimeError(
        "No safe four-pin rotation path was found. No alignment motion was sent."
        + (f"\n  - {details}" if details else "")
    )


def print_four_pin_alignment_plan(plan: dict[str, Any], base_t_pin: np.ndarray, args: argparse.Namespace) -> None:
    print("Four-pin rotation alignment after marker grasp:")
    print(f"  four_pin_center position base m: {fmt(base_t_pin[:3, 3])}")
    print(f"  target_rotation_tcp_pose_ur: {fmt(plan['target_tcp_pose_ur'])}")
    print(f"  rotation_delta_deg: {plan['angle_delta_deg']:.3f}")
    print(f"  tray adjustment frame offset in TCP z mm: {args.tray_center_tcp_z_mm:.3f}")
    print(
        "  tray 4-hole center offset in TCP mm: "
        f"{fmt(1000.0 * tray_four_hole_center_offset_tcp(args), decimals=3)}"
    )
    print(f"  tray local rx deg: {args.tray_local_rx_deg:.3f}")
    approach_vector_mm = np.asarray(plan["rotation_approach_vector_m"], dtype=float) * 1000.0
    print(f"  rotation approach requested mm: {args.rotation_approach_mm:.3f}")
    print(f"  rotation approach vector base mm: {fmt(approach_vector_mm, decimals=3)}")
    print(
        "  tray 4-hole center lateral distance to four-pin center mm: "
        f"{1000.0 * plan['pre_approach_lateral_distance_m']:.3f} -> "
        f"{1000.0 * plan['post_approach_lateral_distance_m']:.3f}"
    )
    print(f"  current tray floor clearance mm: {1000.0 * plan['current_clearance_m']:.3f}")
    print(f"  target tray floor clearance without lift mm: {1000.0 * plan['target_clearance_m']:.3f}")
    print(f"  requested safe floor clearance mm: {args.safe_floor_clearance_mm:.3f}")
    print(f"  post-grasp/minimum lift mm: {args.post_grasp_lift_mm:.3f}")
    print(f"  planned lift along base -Z mm: {1000.0 * plan['safety_lift_m']:.3f}")
    print(f"  lifted_target_tcp_pose_ur: {fmt(plan['lifted_target_tcp_pose_ur'])}")


def print_pin_approach_plan(
    start_tcp_pose_ur: np.ndarray,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    centering_pose, _centering_current_reference, centering_target_reference, centering_translation = (
        pin_centering_target_pose(
            start_tcp_pose_ur,
            base_t_pin,
            args,
        )
    )
    target_pose, current_tray_reference, target_tray_reference, translation = pin_approach_target_pose(
        centering_pose,
        base_t_pin,
        args,
        args.pin_approach_clearance_mm,
    )
    pin_y_axis_base = normalized(np.asarray(base_t_pin[:3, 1], dtype=float))
    pin_rotation = np.asarray(base_t_pin[:3, :3], dtype=float)
    pin_origin = np.asarray(base_t_pin[:3, 3], dtype=float)
    current_reference_in_pin = pin_rotation.T @ (current_tray_reference - pin_origin)
    target_reference_in_pin = pin_rotation.T @ (target_tray_reference - pin_origin)
    print("Post-rotation pin approach:")
    print(f"  enabled: {args.approach_pin_after_rotation}")
    print(f"  tray alignment reference: tray 4-hole center")
    print(
        "  tray 4-hole center offset in TCP mm: "
        f"{fmt(1000.0 * tray_four_hole_center_offset_tcp(args), decimals=3)}"
    )
    print(f"  holder four-pin +Y axis base: {fmt(pin_y_axis_base)}")
    print(
        "  requested target tray 4-hole center in pin frame mm: "
        f"[ 0.000, {args.pin_approach_clearance_mm: .3f},  0.000]"
    )
    print(
        "  tray 4-hole center in pin frame before approach mm: "
        f"{fmt(1000.0 * current_reference_in_pin, decimals=3)}"
    )
    print(
        "  centering target tray 4-hole center in pin frame mm: "
        f"{fmt(1000.0 * (pin_rotation.T @ (centering_target_reference - pin_origin)), decimals=3)}"
    )
    print(
        "  centering translation base mm: "
        f"{fmt(1000.0 * centering_translation, decimals=3)}"
    )
    print(
        "  tray 4-hole center in pin frame target mm: "
        f"{fmt(1000.0 * target_reference_in_pin, decimals=3)}"
    )
    print(f"  tray 4-hole center before approach base m: {fmt(current_tray_reference)}")
    print(f"  centering tray 4-hole center base m: {fmt(centering_target_reference)}")
    print(f"  target tray 4-hole center base m: {fmt(target_tray_reference)}")
    print(f"  final approach translation base mm: {fmt(1000.0 * translation, decimals=3)}")
    print(f"  centering distance mm: {1000.0 * float(np.linalg.norm(centering_translation)):.3f}")
    print(f"  final approach distance mm: {1000.0 * float(np.linalg.norm(translation)):.3f}")
    print(f"  target_tcp_pose_ur: {fmt(target_pose)}")
    return centering_pose, target_pose


def execute_checked_waypoints(
    rtde_receive: Any,
    rtde_control: Any,
    waypoints: list[np.ndarray],
    args: argparse.Namespace,
    motion_name: str,
) -> None:
    for waypoint_index, pose in enumerate(waypoints, start=1):
        success = rtde_control.moveL(
            pose.tolist(),
            args.speed_m_s,
            args.acceleration_m_s2,
            False,
        )
        if not success:
            raise RuntimeError(
                f"{motion_name} waypoint {waypoint_index}/{len(waypoints)} returned false."
            )

        actual_q_rad = np.asarray(rtde_receive.getActualQ(), dtype=float)
        q3_margin_deg = distance_to_nearest_pi_multiple_deg(actual_q_rad[2])
        q5_margin_deg = distance_to_nearest_pi_multiple_deg(actual_q_rad[4])
        print(
            f"  {motion_name} {waypoint_index}/{len(waypoints)}: "
            f"q3={np.rad2deg(actual_q_rad[2]):.1f} deg "
            f"(margin {q3_margin_deg:.1f}), "
            f"q5 margin={q5_margin_deg:.1f} deg"
        )
        if q3_margin_deg < MIN_Q3_SINGULARITY_MARGIN_DEG:
            raise RuntimeError(
                f"Actual q3 singularity margin fell to {q3_margin_deg:.1f} deg after "
                f"{motion_name} waypoint {waypoint_index}; no further motion was sent."
            )
        if q5_margin_deg < MIN_Q5_SINGULARITY_MARGIN_DEG:
            raise RuntimeError(
                f"Actual q5 singularity margin fell to {q5_margin_deg:.1f} deg after "
                f"{motion_name} waypoint {waypoint_index}; no further motion was sent."
            )


def check_actual_joint_margins(rtde_receive: Any, motion_name: str) -> None:
    actual_q_rad = np.asarray(rtde_receive.getActualQ(), dtype=float)
    q3_margin_deg = distance_to_nearest_pi_multiple_deg(actual_q_rad[2])
    q5_margin_deg = distance_to_nearest_pi_multiple_deg(actual_q_rad[4])
    print(
        f"  {motion_name} complete: "
        f"q3={np.rad2deg(actual_q_rad[2]):.1f} deg "
        f"(margin {q3_margin_deg:.1f}), "
        f"q5 margin={q5_margin_deg:.1f} deg"
    )
    if q3_margin_deg < MIN_Q3_SINGULARITY_MARGIN_DEG:
        raise RuntimeError(
            f"Actual q3 singularity margin fell to {q3_margin_deg:.1f} deg after "
            f"{motion_name}; no further motion was sent."
        )
    if q5_margin_deg < MIN_Q5_SINGULARITY_MARGIN_DEG:
        raise RuntimeError(
            f"Actual q5 singularity margin fell to {q5_margin_deg:.1f} deg after "
            f"{motion_name}; no further motion was sent."
        )


def execute_continuous_movel(
    rtde_receive: Any,
    rtde_control: Any,
    target_pose_ur: np.ndarray,
    args: argparse.Namespace,
    motion_name: str,
    speed_m_s: float | None = None,
    acceleration_m_s2: float | None = None,
) -> None:
    current_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    target_pose = np.asarray(target_pose_ur, dtype=float)
    motion_speed_m_s = args.speed_m_s if speed_m_s is None else speed_m_s
    motion_acceleration_m_s2 = (
        args.acceleration_m_s2 if acceleration_m_s2 is None else acceleration_m_s2
    )
    translation_mm = float(np.linalg.norm(target_pose[:3] - current_pose_ur[:3]) * 1000.0)
    rotation_deg = rotation_delta_deg(current_pose_ur, target_pose)
    if translation_mm < 0.01 and rotation_deg < 0.01:
        print(f"Skipping {motion_name}; current TCP is already at the target pose.")
        return

    print(
        f"Executing continuous {motion_name}: "
        f"translation={translation_mm:.2f} mm, rotation={rotation_deg:.2f} deg, "
        f"speed={motion_speed_m_s:.4f} m/s, "
        f"acceleration={motion_acceleration_m_s2:.4f} m/s^2."
    )
    success = rtde_control.moveL(
        target_pose.tolist(),
        motion_speed_m_s,
        motion_acceleration_m_s2,
        False,
    )
    if not success:
        raise RuntimeError(f"{motion_name} moveL returned false.")
    check_actual_joint_margins(rtde_receive, motion_name)


def require_ik_solution(
    rtde_control: Any,
    q_near: np.ndarray,
    pose_ur: np.ndarray,
    motion_name: str,
) -> np.ndarray:
    if not rtde_control.getInverseKinematicsHasSolution(
        pose_ur.tolist(),
        q_near.tolist(),
        IK_POSITION_TOLERANCE_M,
        IK_ORIENTATION_TOLERANCE_RAD,
    ):
        raise RuntimeError(f"No IK solution exists for the {motion_name}.")
    return np.asarray(
        rtde_control.getInverseKinematics(
            pose_ur.tolist(),
            q_near.tolist(),
            IK_POSITION_TOLERANCE_M,
            IK_ORIENTATION_TOLERANCE_RAD,
        ),
        dtype=float,
    )


def execute_pin_approach_after_rotation(
    rtde_receive: Any,
    rtde_control: Any,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> None:
    current_tcp_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    centering_pose, target_pose = print_pin_approach_plan(current_tcp_pose_ur, base_t_pin, args)
    planned_current_pose = target_pose
    planned_image_correction_pose = print_pin_image_alignment_correction_plan(
        planned_current_pose,
        args,
    )
    if planned_image_correction_pose is not None:
        planned_current_pose = planned_image_correction_pose
    planned_insertion_pose = None
    if args.insert_after_pin_approach:
        planned_insertion_pose = print_pin_insertion_plan(planned_current_pose, base_t_pin, args)
    planned_post_insert_targets: list[tuple[str, np.ndarray]] = []
    if planned_insertion_pose is not None:
        planned_post_insert_targets = print_post_insert_release_retreat_plan(
            planned_insertion_pose,
            args,
            base_t_pin,
        )
        if args.post_insert_move_to_two_pin and planned_post_insert_targets:
            planned_post_close_targets = print_post_two_pin_close_retreat_plan(
                planned_post_insert_targets[-1][1],
                args,
            )
            planned_post_insert_targets.extend(planned_post_close_targets)
            if planned_post_close_targets:
                planned_post_close_image_alignment_pose = print_post_two_pin_close_image_alignment_plan(
                    planned_post_close_targets[-1][1],
                    args,
                )
                if planned_post_close_image_alignment_pose is not None:
                    planned_post_insert_targets.append(
                        (
                            "post-two-pin-close image alignment correction",
                            planned_post_close_image_alignment_pose,
                        )
                    )
                    planned_post_insert_targets.extend(
                        print_post_two_pin_aligned_insert_release_plan(
                            planned_post_close_image_alignment_pose,
                            args,
                        )
                    )

    centering_distance_mm = float(
        np.linalg.norm(centering_pose[:3] - current_tcp_pose_ur[:3]) * 1000.0
    )
    approach_distance_mm = float(np.linalg.norm(target_pose[:3] - centering_pose[:3]) * 1000.0)
    total_distance_mm = centering_distance_mm + approach_distance_mm
    if total_distance_mm > args.max_pin_approach_mm:
        raise RuntimeError(
            f"Post-rotation pin approach is {total_distance_mm:.1f} mm total, above "
            f"--max-pin-approach-mm {args.max_pin_approach_mm:.1f}. Re-check the plan."
        )

    current_q_rad = np.asarray(rtde_receive.getActualQ(), dtype=float)
    q_near = current_q_rad
    planned_poses: list[tuple[str, np.ndarray]] = [
        ("post-rotation pin centering", centering_pose),
        ("post-rotation pin approach target", target_pose),
    ]
    if planned_image_correction_pose is not None:
        planned_poses.append(("post-approach image alignment correction", planned_image_correction_pose))
    if planned_insertion_pose is not None:
        planned_poses.append(("final pin insertion descend", planned_insertion_pose))
    planned_poses.extend(planned_post_insert_targets)
    for name, pose in planned_poses:
        q_near = require_ik_solution(rtde_control, q_near, pose, name)

    execute_continuous_movel(
        rtde_receive,
        rtde_control,
        centering_pose,
        args,
        motion_name="post-rotation pin centering",
    )
    execute_continuous_movel(
        rtde_receive,
        rtde_control,
        target_pose,
        args,
        motion_name="post-rotation pin-axis approach",
    )
    if args.auto_pin_image_align:
        execute_auto_pin_image_alignment(rtde_receive, rtde_control, args)
    elif effective_pin_image_alignment_metadata(args) is not None:
        current_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
        image_correction_pose = print_pin_image_alignment_correction_plan(current_pose_ur, args)
        if image_correction_pose is None:
            raise RuntimeError("Image alignment correction metadata was expected but not loaded.")
        require_ik_solution(
            rtde_control,
            np.asarray(rtde_receive.getActualQ(), dtype=float),
            image_correction_pose,
            "post-approach image alignment correction",
        )
        execute_continuous_movel(
            rtde_receive,
            rtde_control,
            image_correction_pose,
            args,
            motion_name="post-approach image alignment correction",
        )
    if args.insert_after_pin_approach:
        current_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
        insertion_pose = print_pin_insertion_plan(current_pose_ur, base_t_pin, args)
        require_ik_solution(
            rtde_control,
            np.asarray(rtde_receive.getActualQ(), dtype=float),
            insertion_pose,
            "final pin insertion descend",
        )
        execute_continuous_movel(
            rtde_receive,
            rtde_control,
            insertion_pose,
            args,
            motion_name="final pin insertion descend",
            speed_m_s=args.pin_insertion_speed_m_s,
            acceleration_m_s2=args.pin_insertion_acceleration_m_s2,
        )
        execute_post_insert_release_retreat(rtde_receive, rtde_control, base_t_pin, args)


def execute_four_pin_alignment(
    rtde_receive: Any,
    rtde_control: Any,
    base_t_pin: np.ndarray,
    floor_z_m: float,
    args: argparse.Namespace,
) -> None:
    grasped_tcp_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    grasped_q_rad = np.asarray(rtde_receive.getActualQ(), dtype=float)
    plan, lift_waypoints, rotation_waypoints, _ik_report = select_safe_alignment_plan(
        rtde_control,
        grasped_tcp_pose_ur,
        grasped_q_rad,
        base_t_pin,
        floor_z_m,
        args,
    )
    print_four_pin_alignment_plan(plan, base_t_pin, args)

    if plan["angle_delta_deg"] > args.max_rotation_deg and not args.allow_large_rotation:
        raise RuntimeError(
            f"Rotation delta is {plan['angle_delta_deg']:.1f} deg, above "
            f"--max-rotation-deg {args.max_rotation_deg:.1f}. Re-check the frame or add "
            "--allow-large-rotation deliberately."
        )
    if args.segmented_alignment:
        print(
            f"Executing segmented lift with {len(lift_waypoints)} waypoints at "
            f"speed={args.speed_m_s:.4f} m/s."
        )
        execute_checked_waypoints(
            rtde_receive,
            rtde_control,
            lift_waypoints,
            args,
            motion_name="post-grasp lift",
        )

        print(
            f"Executing segmented four-pin rotation with {len(rotation_waypoints)} waypoints "
            f"({ROTATION_WAYPOINT_STEP_DEG:.1f} deg maximum step)."
        )
        execute_checked_waypoints(
            rtde_receive,
            rtde_control,
            rotation_waypoints,
            args,
            motion_name="four-pin rotation",
        )
    else:
        print(
            "IK preflight used "
            f"{len(lift_waypoints)} lift waypoints and {len(rotation_waypoints)} rotation "
            "waypoints; execution sends one continuous moveL per stage."
        )
        execute_continuous_movel(
            rtde_receive,
            rtde_control,
            plan["lifted_current_tcp_pose_ur"],
            args,
            motion_name="post-grasp lift",
        )
        execute_continuous_movel(
            rtde_receive,
            rtde_control,
            plan["lifted_target_tcp_pose_ur"],
            args,
            motion_name="four-pin rotation",
        )
    final_tcp_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    print(f"final_aligned_tcp_pose_ur: {fmt(final_tcp_pose_ur)}")


def close_gripper_after_delay(args: argparse.Namespace) -> None:
    if args.gripper_delay_s > 0.0:
        print(f"Waiting {args.gripper_delay_s:.1f}s before closing gripper.")
        time.sleep(args.gripper_delay_s)

    command = GripperCommand(
        name="marker_target_close",
        position_percent=args.gripper_close_percent,
        speed=args.gripper_speed,
        force=args.gripper_force,
        wait=True,
        timeout_s=5.0,
        post_dwell_s=0.0,
    )
    gripper = RobotiqHandEGripper(args.robot_ip, port=args.gripper_port)
    try:
        gripper.connect()
        start_position = gripper.get_position()
        print(f"gripper raw position before close: {start_position} (0=open, 255=closed)")
        print(
            "Closing gripper: "
            f"{args.gripper_close_percent:.1f}% closed, "
            f"speed={args.gripper_speed}, force={args.gripper_force}"
        )
        gripper.move_to_percent(command)
        end_position = gripper.get_position()
        print(f"gripper raw position after close: {end_position}")
    finally:
        gripper.close()


def close_gripper_now(args: argparse.Namespace, command_name: str = "gripper_close") -> None:
    command = GripperCommand(
        name=command_name,
        position_percent=args.gripper_close_percent,
        speed=args.gripper_speed,
        force=args.gripper_force,
        wait=True,
        timeout_s=5.0,
        post_dwell_s=0.0,
    )
    gripper = RobotiqHandEGripper(args.robot_ip, port=args.gripper_port)
    try:
        gripper.connect()
        start_position = gripper.get_position()
        print(f"gripper raw position before close: {start_position} (0=open, 255=closed)")
        print(
            "Closing gripper: "
            f"{args.gripper_close_percent:.1f}% closed, "
            f"speed={args.gripper_speed}, force={args.gripper_force}"
        )
        gripper.move_to_percent(command)
        end_position = gripper.get_position()
        print(f"gripper raw position after close: {end_position}")
    finally:
        gripper.close()


def open_gripper_fully(
    args: argparse.Namespace,
    command_name: str = "post_insert_open",
    post_dwell_s: float = 0.5,
) -> None:
    command = GripperCommand(
        name=command_name,
        position_percent=0.0,
        speed=args.gripper_speed,
        force=args.gripper_force,
        wait=True,
        timeout_s=5.0,
        post_dwell_s=post_dwell_s,
    )
    gripper = RobotiqHandEGripper(args.robot_ip, port=args.gripper_port)
    try:
        gripper.connect()
        start_position = gripper.get_position()
        print(f"gripper raw position before open: {start_position} (0=open, 255=closed)")
        print(
            "Opening gripper fully: "
            f"0.0% closed, speed={args.gripper_speed}, force={args.gripper_force}, "
            f"post_dwell_s={post_dwell_s:.1f}"
        )
        gripper.move_to_percent(command)
        end_position = gripper.get_position()
        print(f"gripper raw position after open: {end_position}")
    finally:
        gripper.close()


def execute_post_insert_release_retreat(
    rtde_receive: Any,
    rtde_control: Any,
    base_t_pin: np.ndarray,
    args: argparse.Namespace,
) -> None:
    if not args.post_insert_release_retreat:
        print("Skipping post-insert release/retreat.")
        return

    current_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    post_insert_targets = print_post_insert_release_retreat_plan(current_pose_ur, args, base_t_pin)
    post_close_targets: list[tuple[str, np.ndarray]] = []
    if args.post_insert_move_to_two_pin and post_insert_targets:
        post_close_targets = print_post_two_pin_close_retreat_plan(
            post_insert_targets[-1][1],
            args,
        )
    post_close_image_alignment_pose = None
    post_close_image_alignment_metadata = effective_post_two_pin_close_image_alignment_metadata(args)
    if post_close_targets:
        post_close_image_alignment_pose = print_post_two_pin_close_image_alignment_plan(
            post_close_targets[-1][1],
            args,
        )
    post_close_alignment_will_run = bool(
        post_close_targets
        and args.post_two_pin_close_image_align
        and (
            args.auto_pin_image_align
            or post_close_image_alignment_metadata is not None
        )
    )
    if post_close_targets and not post_close_alignment_will_run:
        print("Post-two-pin-close image alignment will not run:")
        print(f"  post-two-pin-close image align enabled: {args.post_two_pin_close_image_align}")
        print(f"  auto image align enabled: {args.auto_pin_image_align}")
        print(f"  post-two-pin-close image metadata: {post_close_image_alignment_metadata}")
        print(f"  final aligned insert/release enabled: {args.post_two_pin_aligned_insert_release}")
    post_aligned_insert_release_targets: list[tuple[str, np.ndarray]] = []
    if post_close_alignment_will_run and args.post_two_pin_aligned_insert_release:
        post_aligned_start_pose = (
            post_close_image_alignment_pose
            if post_close_image_alignment_pose is not None
            else post_close_targets[-1][1]
        )
        post_aligned_insert_release_targets = print_post_two_pin_aligned_insert_release_plan(
            post_aligned_start_pose,
            args,
        )
    q_near = np.asarray(rtde_receive.getActualQ(), dtype=float)
    for name, pose in [*post_insert_targets, *post_close_targets]:
        q_near = require_ik_solution(rtde_control, q_near, pose, name)
    if post_close_image_alignment_pose is not None:
        q_near = require_ik_solution(
            rtde_control,
            q_near,
            post_close_image_alignment_pose,
            "post-two-pin-close image alignment correction",
        )
    for name, pose in post_aligned_insert_release_targets:
        q_near = require_ik_solution(rtde_control, q_near, pose, name)

    open_gripper_fully(args)
    for name, pose in post_insert_targets:
        execute_continuous_movel(
            rtde_receive,
            rtde_control,
            pose,
            args,
            motion_name=name,
        )
    if args.post_insert_move_to_two_pin:
        close_gripper_now(args, command_name="post_insert_two_pin_close")
        print(
            f"Holding still for {POST_TWO_PIN_CLOSE_DWELL_S:.1f}s after closing "
            "gripper at adjacent two-pin target."
        )
        time.sleep(POST_TWO_PIN_CLOSE_DWELL_S)
        for name, pose in post_close_targets:
            execute_continuous_movel(
                rtde_receive,
                rtde_control,
                pose,
                args,
                motion_name=name,
            )
        if post_close_alignment_will_run:
            if args.auto_pin_image_align:
                execute_auto_pin_image_alignment(
                    rtde_receive,
                    rtde_control,
                    args,
                    title="Automatic post-two-pin-close image alignment",
                )
            elif post_close_image_alignment_metadata is not None:
                current_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
                image_correction_pose = print_pin_image_alignment_correction_plan(
                    current_pose_ur,
                    args,
                    title="Post-two-pin-close image alignment correction",
                    metadata_path=post_close_image_alignment_metadata,
                )
                if image_correction_pose is None:
                    raise RuntimeError("Post-two-pin-close image correction metadata was expected but not loaded.")
                require_ik_solution(
                    rtde_control,
                    np.asarray(rtde_receive.getActualQ(), dtype=float),
                    image_correction_pose,
                    "post-two-pin-close image alignment correction",
                )
                execute_continuous_movel(
                    rtde_receive,
                    rtde_control,
                    image_correction_pose,
                    args,
                    motion_name="post-two-pin-close image alignment correction",
                )
            if args.post_two_pin_aligned_insert_release:
                current_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
                post_aligned_insert_release_targets = print_post_two_pin_aligned_insert_release_plan(
                    current_pose_ur,
                    args,
                )
                q_near = np.asarray(rtde_receive.getActualQ(), dtype=float)
                for name, pose in post_aligned_insert_release_targets:
                    q_near = require_ik_solution(rtde_control, q_near, pose, name)
                if len(post_aligned_insert_release_targets) != 2:
                    raise RuntimeError("Expected final aligned insert and release-retreat targets.")
                insert_name, insert_pose = post_aligned_insert_release_targets[0]
                execute_continuous_movel(
                    rtde_receive,
                    rtde_control,
                    insert_pose,
                    args,
                    motion_name=insert_name,
                    speed_m_s=args.pin_insertion_speed_m_s,
                    acceleration_m_s2=args.pin_insertion_acceleration_m_s2,
                )
                open_gripper_fully(
                    args,
                    command_name="post_two_pin_aligned_open",
                    post_dwell_s=2.0,
                )
                retreat_name, retreat_pose = post_aligned_insert_release_targets[1]
                execute_continuous_movel(
                    rtde_receive,
                    rtde_control,
                    retreat_pose,
                    args,
                    motion_name=retreat_name,
                )


def round_list(values: np.ndarray, decimals: int = 10) -> list[float]:
    return [round(float(value), decimals) for value in np.asarray(values, dtype=float).reshape(-1)]


def matrix_list(matrix: np.ndarray, decimals: int = 10) -> list[list[float]]:
    return [[round(float(value), decimals) for value in row] for row in np.asarray(matrix)]


def fmt(values: np.ndarray, decimals: int = 6) -> str:
    return "[" + ", ".join(f"{float(value): .{decimals}f}" for value in values) + "]"


def main() -> None:
    args = parse_args()
    validate_args(args)
    print_motion_speed_config(args)

    marker_id, base_t_marker_raw, _marker_data = load_marker_transform(args.marker_pose)
    base_t_marker = (
        floor_constrained_marker_transform(base_t_marker_raw)
        if args.marker_frame_mode == "floor"
        else base_t_marker_raw
    )
    offset_marker_m = marker_target_offset(args)
    base_t_pin = None
    if args.align_to_four_pin_frame or args.continue_pin_sequence_from_current:
        base_t_pin = compute_base_t_four_pin_frame(args, base_t_marker)

    rtde_receive = None
    rtde_control = None
    try:
        if args.execute or not args.no_set_tcp or args.start_from_initial_pose:
            rtde_control = connect_rtde_control(args.robot_ip)
        if not args.no_set_tcp:
            set_tcp_offset(rtde_control, args.tcp_offset_ur)

        rtde_receive, current_tcp_pose_ur = read_current_tcp_pose(args.robot_ip)
        if args.continue_pin_sequence_from_current:
            if base_t_pin is None:
                raise RuntimeError("Four-pin frame was not computed.")
            print("Continuing pin approach/correction/insertion from the current TCP pose.")
            print(f"  marker pose: {args.marker_pose}")
            print(f"  marker_frame_mode: {args.marker_frame_mode}")
            print(f"  tcp offset UR: {args.tcp_offset_ur}")
            print(f"  set tcp before plan: {not args.no_set_tcp}")
            print(f"  current_tcp_pose_ur: {fmt(current_tcp_pose_ur)}")
            print(f"  pin approach clearance mm: {args.pin_approach_clearance_mm:.3f}")
            print(f"  auto pin image align: {args.auto_pin_image_align}")
            print(
                "  pin image alignment metadata: "
                f"{effective_pin_image_alignment_metadata(args)}"
            )
            print(
                "  post-two-pin-close image alignment metadata: "
                f"{effective_post_two_pin_close_image_alignment_metadata(args)}"
            )
            print(f"  insert after pin approach: {args.insert_after_pin_approach}")
            if args.insert_after_pin_approach:
                print(f"  pin insertion target Y mm: {args.pin_insertion_target_y_mm:.3f}")
                print(f"  post-insert release/retreat: {args.post_insert_release_retreat}")
                if args.post_insert_release_retreat:
                    print(f"  post-insert TCP local +Y mm: {args.post_insert_retreat_y_mm:.3f}")
                    print(
                        "  post-insert TCP local +X mm: "
                        f"{args.post_insert_shift_x_mm if args.post_insert_shift_x_mm is not None else 'computed from OBJ'}"
                    )
                    print(f"  post-insert move to adjacent two-pin center: {args.post_insert_move_to_two_pin}")
                    if args.post_insert_move_to_two_pin:
                        print(
                            "  post-insert adjacent two-pin target +Y mm: "
                            f"{args.post_insert_two_pin_target_y_mm:.3f}"
                        )
                        print(f"  post-two-pin-close retreat: {args.post_two_pin_close_retreat}")
                        if args.post_two_pin_close_retreat:
                            print(
                                "  post-two-pin-close TCP local +Y/-X mm: "
                                f"{args.post_two_pin_close_retreat_y_mm:.3f}/"
                                f"{args.post_two_pin_close_shift_negative_x_mm:.3f}"
                            )
                            print(
                                "  post-two-pin aligned insert/release: "
                                f"{args.post_two_pin_aligned_insert_release}"
                            )
                            if args.post_two_pin_aligned_insert_release:
                                print(
                                    "  post-two-pin aligned TCP local -Y/+Y mm: "
                                    f"{args.post_two_pin_aligned_insert_y_mm:.3f}/"
                                    f"{args.post_two_pin_release_retreat_y_mm:.3f}"
                                )
            if not args.execute:
                centering_pose, target_pose = print_pin_approach_plan(
                    current_tcp_pose_ur,
                    base_t_pin,
                    args,
                )
                planned_current_pose = target_pose
                planned_image_correction_pose = print_pin_image_alignment_correction_plan(
                    planned_current_pose,
                    args,
                )
                if planned_image_correction_pose is not None:
                    planned_current_pose = planned_image_correction_pose
                if args.auto_pin_image_align:
                    print(
                        "Automatic image alignment will run during --execute after the "
                        "pin approach; dry-run cannot know the final searched correction."
                    )
                if args.insert_after_pin_approach:
                    planned_insertion_pose = print_pin_insertion_plan(
                        planned_current_pose,
                        base_t_pin,
                        args,
                    )
                    planned_post_insert_targets = print_post_insert_release_retreat_plan(
                        planned_insertion_pose,
                        args,
                        base_t_pin,
                    )
                    if args.post_insert_move_to_two_pin and planned_post_insert_targets:
                        planned_post_close_targets = print_post_two_pin_close_retreat_plan(
                            planned_post_insert_targets[-1][1],
                            args,
                        )
                        if planned_post_close_targets:
                            planned_post_close_image_alignment_pose = (
                                print_post_two_pin_close_image_alignment_plan(
                                    planned_post_close_targets[-1][1],
                                    args,
                                )
                            )
                            if planned_post_close_image_alignment_pose is not None:
                                print_post_two_pin_aligned_insert_release_plan(
                                    planned_post_close_image_alignment_pose,
                                    args,
                                )
                print("DRY RUN ONLY: add --execute to continue from current pose.")
                return
            execute_pin_approach_after_rotation(
                rtde_receive,
                rtde_control,
                base_t_pin,
                args,
            )
            print("Current-pose pin approach/correction/insertion sequence complete.")
            return
        if args.start_from_initial_pose:
            if args.execute:
                move_to_initial_pose(rtde_receive, rtde_control, args)
                current_tcp_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
            else:
                initial_q_rad = np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
                current_tcp_pose_ur = np.asarray(
                    rtde_control.getForwardKinematics(
                        initial_q_rad.tolist(),
                        args.tcp_offset_ur,
                    ),
                    dtype=float,
                )
                print(
                    "Initial pose move is enabled; dry-run uses calibrated FK without moveJ."
                )

        target_tcp_pose_ur = make_target_pose_ur(
            base_t_marker,
            offset_marker_m,
            current_tcp_pose_ur,
            args.orientation,
        )
        distance_mm = float(np.linalg.norm(target_tcp_pose_ur[:3] - current_tcp_pose_ur[:3]) * 1000.0)

        print("Marker-based TCP target:")
        print(f"  marker pose: {args.marker_pose}")
        print(f"  marker_id: {marker_id}")
        print(f"  marker_frame_mode: {args.marker_frame_mode}")
        print(f"  tcp offset UR: {args.tcp_offset_ur}")
        print(f"  set tcp before plan: {not args.no_set_tcp}")
        print(f"  start from initial pose: {args.start_from_initial_pose}")
        if args.start_from_initial_pose:
            print(f"  initial q_deg: {fmt(np.asarray(args.initial_q_deg), decimals=3)}")
        print(f"  offset_marker_mm: {fmt(offset_marker_m * 1000.0, decimals=3)}")
        if args.marker_frame_mode == "floor":
            print(f"  height_above_marker_mm: {args.height_above_marker_mm:.3f}")
            print("  note: floor mode uses base +Z as down, so height above marker is marker -Z.")
        print(f"  orientation: {args.orientation}")
        print(f"  current_tcp_pose_ur: {fmt(current_tcp_pose_ur)}")
        print(f"  target_tcp_pose_ur:  {fmt(target_tcp_pose_ur)}")
        print(f"  target_tcp_position_m: {fmt(target_tcp_pose_ur[:3])}")
        print(f"  distance_from_current_tcp: {distance_mm:.1f} mm")
        print(f"  close gripper after move: {not args.skip_gripper_close}")
        if not args.skip_gripper_close:
            print(f"  gripper delay: {args.gripper_delay_s:.1f} s")
            print(f"  post-grasp dwell: {POST_GRASP_DWELL_S:.1f} s")
            print(f"  gripper close percent: {args.gripper_close_percent:.1f}")
        print(f"  align to four-pin frame after grasp: {args.align_to_four_pin_frame}")
        if args.align_to_four_pin_frame:
            print(f"  approach pin after rotation: {args.approach_pin_after_rotation}")
            print(f"  pin approach clearance mm: {args.pin_approach_clearance_mm:.3f}")
            print(f"  auto pin image align: {args.auto_pin_image_align}")
            if args.auto_pin_image_align:
                print(f"  pin image reference: {args.pin_image_reference}")
                print(f"  pin image ROI: {args.pin_image_roi}")
                print(
                    "  auto pin image search: "
                    f"step={args.auto_pin_image_search_step_mm:.3f} mm/"
                    f"{args.auto_pin_image_search_step_deg:.3f} deg, "
                    f"iterations={args.auto_pin_image_search_iterations}, "
                    f"max={args.auto_pin_image_search_max_mm:.3f} mm/"
                    f"{args.auto_pin_image_search_max_deg:.3f} deg"
                )
            print(
                "  pin image alignment metadata: "
                f"{effective_pin_image_alignment_metadata(args)}"
            )
            print(
                "  post-two-pin-close image alignment metadata: "
                f"{effective_post_two_pin_close_image_alignment_metadata(args)}"
            )
            print(f"  insert after pin approach: {args.insert_after_pin_approach}")
            if args.insert_after_pin_approach:
                print(f"  pin insertion target Y mm: {args.pin_insertion_target_y_mm:.3f}")
                print(f"  post-insert release/retreat: {args.post_insert_release_retreat}")
                if args.post_insert_release_retreat:
                    print(f"  post-insert TCP local +Y mm: {args.post_insert_retreat_y_mm:.3f}")
                    print(
                        "  post-insert TCP local +X mm: "
                        f"{args.post_insert_shift_x_mm if args.post_insert_shift_x_mm is not None else 'computed from OBJ'}"
                    )
                    print(f"  post-insert move to adjacent two-pin center: {args.post_insert_move_to_two_pin}")
                    if args.post_insert_move_to_two_pin:
                        print(
                            "  post-insert adjacent two-pin target +Y mm: "
                            f"{args.post_insert_two_pin_target_y_mm:.3f}"
                        )
                        print(f"  post-two-pin-close retreat: {args.post_two_pin_close_retreat}")
                        if args.post_two_pin_close_retreat:
                            print(
                                "  post-two-pin-close TCP local +Y/-X mm: "
                                f"{args.post_two_pin_close_retreat_y_mm:.3f}/"
                                f"{args.post_two_pin_close_shift_negative_x_mm:.3f}"
                            )
        if args.align_to_four_pin_frame and base_t_pin is not None:
            plan = four_pin_alignment_plan(
                target_tcp_pose_ur,
                base_t_pin,
                float(base_t_marker[2, 3]),
                args,
            )
            print("  dry-run alignment plan below uses the planned marker target as the grasp pose.")
            print_four_pin_alignment_plan(plan, base_t_pin, args)
            if not args.execute:
                current_q_rad = (
                    np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
                    if args.start_from_initial_pose
                    else np.asarray(rtde_receive.getActualQ(), dtype=float)
                )
                if not rtde_control.getInverseKinematicsHasSolution(
                    target_tcp_pose_ur.tolist(),
                    current_q_rad.tolist(),
                    IK_POSITION_TOLERANCE_M,
                    IK_ORIENTATION_TOLERANCE_RAD,
                ):
                    raise RuntimeError("No IK solution exists for the planned marker grasp pose.")
                planned_grasp_q_rad = np.asarray(
                    rtde_control.getInverseKinematics(
                        target_tcp_pose_ur.tolist(),
                        current_q_rad.tolist(),
                        IK_POSITION_TOLERANCE_M,
                        IK_ORIENTATION_TOLERANCE_RAD,
                    ),
                    dtype=float,
                )
                safe_plan, lift_waypoints, rotation_waypoints, ik_report = (
                    select_safe_alignment_plan(
                        rtde_control,
                        target_tcp_pose_ur,
                        planned_grasp_q_rad,
                        base_t_pin,
                        float(base_t_marker[2, 3]),
                        args,
                    )
                )
                print(
                    "  exact IK preflight: "
                    f"lift={1000.0 * safe_plan['safety_lift_m']:.1f} mm, "
                    f"lift waypoints={len(lift_waypoints)}, "
                    f"rotation waypoints={len(rotation_waypoints)}, "
                    f"minimum q3 margin={ik_report['minimum_q3_margin_deg']:.1f} deg"
                )
                if args.approach_pin_after_rotation:
                    pin_centering_target, pin_approach_target = print_pin_approach_plan(
                        safe_plan["lifted_target_tcp_pose_ur"],
                        base_t_pin,
                        args,
                    )
                    pin_centering_distance_mm = float(
                        np.linalg.norm(
                            pin_centering_target[:3] - safe_plan["lifted_target_tcp_pose_ur"][:3]
                        )
                        * 1000.0
                    )
                    pin_approach_distance_mm = float(
                        np.linalg.norm(pin_approach_target[:3] - pin_centering_target[:3])
                        * 1000.0
                    )
                    total_pin_approach_distance_mm = (
                        pin_centering_distance_mm + pin_approach_distance_mm
                    )
                    if total_pin_approach_distance_mm > args.max_pin_approach_mm:
                        raise RuntimeError(
                            f"Post-rotation pin approach is {total_pin_approach_distance_mm:.1f} mm total, "
                            f"above --max-pin-approach-mm {args.max_pin_approach_mm:.1f}."
                        )
                    q_near = (
                        np.asarray(ik_report["q_path"][-1], dtype=float)
                        if ik_report["q_path"]
                        else planned_grasp_q_rad
                    )
                    planned_poses: list[tuple[str, np.ndarray]] = [
                        ("post-rotation pin centering", pin_centering_target),
                        ("post-rotation pin approach target", pin_approach_target),
                    ]
                    planned_current_pose = pin_approach_target
                    planned_image_correction_pose = print_pin_image_alignment_correction_plan(
                        planned_current_pose,
                        args,
                    )
                    if planned_image_correction_pose is not None:
                        planned_poses.append(
                            (
                                "post-approach image alignment correction",
                                planned_image_correction_pose,
                            )
                        )
                        planned_current_pose = planned_image_correction_pose
                    if args.auto_pin_image_align:
                        print(
                            "Automatic image alignment will run during --execute after the "
                            "pin approach; dry-run cannot know the final searched correction."
                        )
                    if args.insert_after_pin_approach:
                        planned_insertion_pose = print_pin_insertion_plan(
                            planned_current_pose,
                            base_t_pin,
                            args,
                        )
                        planned_poses.append(("final pin insertion descend", planned_insertion_pose))
                        planned_post_insert_targets = print_post_insert_release_retreat_plan(
                            planned_insertion_pose,
                            args,
                            base_t_pin,
                        )
                        planned_poses.extend(planned_post_insert_targets)
                        if args.post_insert_move_to_two_pin and planned_post_insert_targets:
                            planned_post_close_targets = print_post_two_pin_close_retreat_plan(
                                planned_post_insert_targets[-1][1],
                                args,
                            )
                            planned_poses.extend(planned_post_close_targets)
                            if planned_post_close_targets:
                                planned_post_close_image_alignment_pose = (
                                    print_post_two_pin_close_image_alignment_plan(
                                        planned_post_close_targets[-1][1],
                                        args,
                                    )
                                )
                                if planned_post_close_image_alignment_pose is not None:
                                    planned_poses.append(
                                        (
                                            "post-two-pin-close image alignment correction",
                                            planned_post_close_image_alignment_pose,
                                        )
                                    )
                                    planned_poses.extend(
                                        print_post_two_pin_aligned_insert_release_plan(
                                            planned_post_close_image_alignment_pose,
                                            args,
                                        )
                                    )
                    for name, pose in planned_poses:
                        q_near = require_ik_solution(rtde_control, q_near, pose, name)

        if args.save_target is not None:
            save_path = Path(args.save_target)
            save_target_yaml(
                save_path,
                marker_id,
                args.marker_pose,
                base_t_marker,
                offset_marker_m,
                current_tcp_pose_ur,
                target_tcp_pose_ur,
                args,
            )
            print(f"Saved target plan to {save_path}")

        if not args.execute:
            print("DRY RUN ONLY: add --execute to send moveL to the robot.")
            return

        if distance_mm > args.max_distance_mm and not args.allow_large_move:
            raise RuntimeError(
                f"Target is {distance_mm:.1f} mm from current TCP, above "
                f"--max-distance-mm {args.max_distance_mm:.1f}. Re-check the plan or add "
                "--allow-large-move deliberately."
            )

        print(
            f"Executing moveL at speed={args.speed_m_s:.4f} m/s, "
            f"acceleration={args.acceleration_m_s2:.4f} m/s^2"
        )
        success = rtde_control.moveL(
            target_tcp_pose_ur.tolist(),
            args.speed_m_s,
            args.acceleration_m_s2,
            False,
        )
        if not success:
            raise RuntimeError("moveL returned false.")
        final_tcp_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
        print(f"final_tcp_pose_ur: {fmt(final_tcp_pose_ur)}")
        print("Marker-based move complete.")
        if not args.skip_gripper_close:
            close_gripper_after_delay(args)
            print("Gripper close complete.")
            print(f"Holding still for {POST_GRASP_DWELL_S:.1f}s after closing gripper.")
            time.sleep(POST_GRASP_DWELL_S)
        if args.align_to_four_pin_frame:
            if base_t_pin is None:
                raise RuntimeError("Four-pin frame was not computed.")
            execute_four_pin_alignment(
                rtde_receive,
                rtde_control,
                base_t_pin,
                float(base_t_marker[2, 3]),
                args,
            )
            print("Marker-based grasp and four-pin rotation alignment complete.")
            if args.approach_pin_after_rotation:
                execute_pin_approach_after_rotation(
                    rtde_receive,
                    rtde_control,
                    base_t_pin,
                    args,
                )
                print("Post-rotation pin approach/correction/insertion sequence complete.")
    finally:
        if rtde_control is not None and hasattr(rtde_control, "stopScript"):
            rtde_control.stopScript()
        if rtde_receive is not None and hasattr(rtde_receive, "disconnect"):
            rtde_receive.disconnect()
        if rtde_control is not None and hasattr(rtde_control, "disconnect"):
            rtde_control.disconnect()


if __name__ == "__main__":
    main()
