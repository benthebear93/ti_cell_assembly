"""Assembly calibration and a small command-line interface.

MotionConfig contains tunable workcell values. Execution and resume selection
are CLI-only, so loading a calibration file cannot enable robot motion.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import numpy as np
import yaml

from surface_estimator_ur5e.live_robot import DEFAULT_ROBOT_IP
from surface_estimator_ur5e.motion_sequence import DEFAULT_GRIPPER_PORT
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
)

DEFAULT_TARGET_OUTPUT = Path("data/markers/aruco_364_tcp_target.yaml")
DEFAULT_MAX_SAFETY_LIFT_MM = 80.0
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
DEFAULT_PIN_INSERTION_SPEED_M_S = 0.005
DEFAULT_PIN_INSERTION_ACCELERATION_M_S2 = 0.01
DEFAULT_POST_TWO_PIN_RELEASE_AFTER_ROTATION_Z_MM = 40.0
POST_GRASP_DWELL_S = 1.5
POST_TWO_PIN_CLOSE_DWELL_S = 3.0
ROTATION_LIFT_CANDIDATES_MM = (0.0, 30.0, 50.0, 70.0)
LIFT_WAYPOINT_STEP_MM = 10.0
ROTATION_WAYPOINT_STEP_DEG = 5.0
MIN_Q3_SINGULARITY_MARGIN_DEG = 25.0
MIN_Q5_SINGULARITY_MARGIN_DEG = 10.0
MAX_IK_JOINT_STEP_DEG = 20.0
IK_POSITION_TOLERANCE_M = 1e-5
IK_ORIENTATION_TOLERANCE_RAD = 1e-5


def parameter(default, *, minimum=None, maximum=None, positive=False, length=None, choices=None):
    """Keep each calibration value and its allowed range in one place."""
    return field(
        default=default,
        metadata=dict(
            minimum=minimum,
            maximum=maximum,
            positive=positive,
            length=length,
            choices=choices,
        ),
    )


@dataclass
class MotionConfig:
    """Distances are millimetres unless the field explicitly uses metres."""

    # Initial pose and joint motion.
    initial_q_deg: tuple[float, ...] = parameter(DEFAULT_INITIAL_Q_DEG, length=6)
    initial_speed_rad_s: float = parameter(0.05, positive=True)
    initial_acceleration_rad_s2: float = parameter(0.05, positive=True)
    initial_dwell_s: float = parameter(0.5, minimum=0)

    # Marker-relative grasp target.
    offset_x_mm: float = 144.0
    offset_y_mm: float = -50.5
    height_above_marker_mm: float = 15.0
    marker_frame_mode: str = parameter("floor", choices=("floor", "raw"))
    offset_z_mm: float | None = None
    orientation: str = parameter("current", choices=("current", "marker"))

    # Cartesian motion and travel limit.
    speed_m_s: float = parameter(0.03, positive=True)
    acceleration_m_s2: float = parameter(0.06, positive=True)
    max_distance_mm: float = parameter(300.0, minimum=0)
    allow_large_move: bool = False

    # Hand-E gripper.
    gripper_delay_s: float = parameter(0.5, minimum=0)
    gripper_close_percent: float = parameter(100.0, minimum=0, maximum=100)
    gripper_speed: int = parameter(80, minimum=0, maximum=255)
    gripper_force: int = parameter(100, minimum=0, maximum=255)
    gripper_port: int = parameter(DEFAULT_GRIPPER_PORT, minimum=1, maximum=65535)
    skip_gripper_close: bool = False

    # Holder alignment and floor clearance.
    segmented_alignment: bool = False
    assembly_obj: Path = DEFAULT_TI_ASSEMBLY_OBJ
    assembly_origin_x_mm: float = DEFAULT_ASSEMBLY_ORIGIN_X_MM
    assembly_origin_y_mm: float = DEFAULT_ASSEMBLY_ORIGIN_Y_MM
    assembly_origin_z_mm: float = DEFAULT_ASSEMBLY_ORIGIN_Z_MM
    assembly_rpy_deg: tuple[float, ...] = parameter(DEFAULT_ASSEMBLY_RPY_DEG, length=3)
    assembly_local_yaw_deg: float = DEFAULT_ASSEMBLY_LOCAL_YAW_DEG
    tcp_rotation_offset_rpy_deg: tuple[float, ...] = parameter((0.0, 0.0, 0.0), length=3)
    post_grasp_lift_mm: float = parameter(15.0, minimum=0)
    safe_floor_clearance_mm: float = parameter(15.0, minimum=0)
    max_safety_lift_mm: float = parameter(DEFAULT_MAX_SAFETY_LIFT_MM, minimum=0)
    max_rotation_deg: float = parameter(120.0, minimum=0)
    allow_large_rotation: bool = False
    rotation_approach_mm: float = parameter(50.0, minimum=0)
    approach_pin_after_rotation: bool = True
    pin_approach_clearance_mm: float = parameter(50.0, minimum=0)
    max_pin_approach_mm: float = parameter(DEFAULT_MAX_PIN_APPROACH_MM, minimum=0)

    # Saved image correction or live camera search.
    pin_image_alignment_metadata: Path | None = None
    auto_pin_image_align: bool = False
    pin_image_reference: Path | None = None
    pin_image_roi: tuple[int, ...] | None = parameter(None, length=4)
    pin_image_serial: str = "261322073147"
    pin_image_width: int = parameter(1280, positive=True)
    pin_image_height: int = parameter(720, positive=True)
    pin_image_fps: int = parameter(30, positive=True)
    pin_image_warmup_frames: int = parameter(10, minimum=0)
    pin_image_timeout_ms: int = parameter(10000, positive=True)
    pin_image_output_dir: Path | None = None
    auto_pin_image_search_step_mm: float = parameter(0.25, minimum=0)
    auto_pin_image_search_step_deg: float = parameter(0.25, minimum=0)
    auto_pin_image_search_iterations: int = parameter(2, minimum=1)
    auto_pin_image_search_max_mm: float = parameter(3.0, minimum=0)
    auto_pin_image_search_max_deg: float = parameter(3.0, minimum=0)
    auto_pin_image_settle_s: float = parameter(0.2, minimum=0)
    max_pin_image_correction_mm: float = parameter(10.0, minimum=0)
    max_pin_image_correction_deg: float = parameter(10.0, minimum=0)

    # Four-pin insertion.
    pin_insertion_target_y_mm: float = 0.0
    max_pin_insertion_mm: float = parameter(80.0, minimum=0)
    pin_insertion_speed_m_s: float = parameter(DEFAULT_PIN_INSERTION_SPEED_M_S, positive=True)
    pin_insertion_acceleration_m_s2: float = parameter(
        DEFAULT_PIN_INSERTION_ACCELERATION_M_S2, positive=True
    )

    # Release and adjacent two-pin pickup.
    post_insert_retreat_y_mm: float = parameter(50.0, minimum=0)
    post_insert_shift_x_mm: float | None = parameter(None, minimum=0)
    post_insert_move_to_two_pin: bool = True
    post_insert_two_pin_target_y_mm: float = 19.0
    post_two_pin_close_retreat: bool = True
    post_two_pin_close_retreat_y_mm: float = parameter(30.0, minimum=0)
    post_two_pin_close_shift_negative_x_mm: float = parameter(58.0, minimum=0)

    # Second alignment, placement and release.
    post_two_pin_close_image_align: bool = True
    post_two_pin_close_image_alignment_metadata: Path | None = None
    post_two_pin_aligned_insert_release: bool = True
    post_two_pin_aligned_insert_y_mm: float = parameter(17.0, minimum=0)
    post_two_pin_release_retreat_y_mm: float = parameter(150.0, minimum=0)
    post_two_pin_release_shift_z_mm: float = -170.0
    post_two_pin_release_final_y_mm: float = -170.0
    post_two_pin_release_final_x_mm: float = 20.0
    post_two_pin_release_rotate_x_deg: float = -88.0
    post_two_pin_release_after_rotation_y_mm: float = 10.0
    post_two_pin_release_final_close_percent: float = parameter(50.0, minimum=0, maximum=100)
    stop_after_post_two_pin_release_rotation: bool = False

    # Final held-part moves.
    post_two_pin_after_close_y_mm: float = -10.0
    post_two_pin_after_close_final_z_mm: float = 15.0

    # Tray geometry and calibrated TCP.
    tray_obj: Path = DEFAULT_VIRTUAL_TRAY_OBJ
    tray_center_tcp_z_mm: float = DEFAULT_VIRTUAL_TRAY_TCP_Z_MM
    tray_local_rx_deg: float = DEFAULT_VIRTUAL_TRAY_LOCAL_RX_DEG
    tray_handle_root_y_mm: float = DEFAULT_VIRTUAL_TRAY_HANDLE_ROOT_Y_MM
    tray_handle_scale: float = parameter(
        DEFAULT_VIRTUAL_TRAY_HANDLE_SCALE, positive=True, maximum=1
    )
    tcp_offset_ur: tuple[float, ...] = parameter(tuple(DEFAULT_TCP_OFFSET_UR), length=6)
    no_set_tcp: bool = False


def build_arg_parser(default_task: str = "pick") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan marker-based assembly; add --execute to move the robot.",
        epilog="Calibration, gripper, camera and safety settings live in MotionConfig / --config YAML.",
    )
    parser.add_argument("--config", type=Path, help="YAML overrides for MotionConfig fields.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--marker-pose", type=Path, default=DEFAULT_MARKER_POSE)
    parser.add_argument(
        "--task",
        choices=("pick", "align", "insert", "assembly"),
        default=default_task,
        help="pick: grasp; align: grasp + align/approach; insert: stop at insertion; assembly: full sequence.",
    )
    parser.add_argument(
        "--resume",
        choices=("pins", "release", "after-close", "tail", "tail-next"),
        help="Run a selected section from the current TCP pose, without initial pose or marker grasp.",
    )
    parser.add_argument("--from-current", action="store_true", help="Skip the initial joint pose.")
    parser.add_argument("--speed-m-s", type=float, default=None)
    parser.add_argument("--acceleration-m-s2", type=float, default=None)
    parser.add_argument("--save-target", type=Path, nargs="?", const=DEFAULT_TARGET_OUTPUT)
    parser.add_argument("--execute", action="store_true")
    return parser


def _config_value(name: str, value: object, annotation: object) -> object:
    """Reject malformed YAML before opening robot or camera connections."""
    types = get_args(annotation)
    if type(None) in types:
        if value is None:
            return None
        annotation = next(t for t in types if t is not type(None))
    if get_origin(annotation) is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{name} must be a list.")
        return tuple(_config_value(name, v, get_args(annotation)[0]) for v in value)
    if annotation is Path:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a path string.")
        return Path(value)
    valid = isinstance(value, (float, int)) if annotation is float else type(value) is annotation
    if not valid or (annotation in (float, int) and isinstance(value, bool)):
        raise ValueError(f"{name} must have type {annotation.__name__}.")
    return float(value) if annotation is float else value


def load_config(path: Path | None) -> MotionConfig:
    if path is None:
        return MotionConfig()
    with path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping of MotionConfig fields.")
    annotations = get_type_hints(MotionConfig)
    unknown = data.keys() - annotations.keys()
    if unknown:
        raise ValueError(f"Unknown motion settings: {', '.join(sorted(map(str, unknown)))}")
    return MotionConfig(
        **{name: _config_value(name, value, annotations[name]) for name, value in data.items()}
    )


def parse_motion_args(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    args = parser.parse_args(argv)
    try:
        settings = asdict(load_config(args.config))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    for name in ("speed_m_s", "acceleration_m_s2"):
        if getattr(args, name) is not None:
            settings[name] = getattr(args, name)
    vars(args).update(settings)
    args.start_from_initial_pose = not (args.from_current or args.resume)
    args.align_to_four_pin_frame = args.task != "pick"
    args.insert_after_pin_approach = args.task in ("insert", "assembly")
    args.post_insert_release_retreat = args.task == "assembly"
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return parse_motion_args(build_arg_parser(), argv)


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
    for name, value in vars(args).items():
        if isinstance(value, (int, float, list, tuple, np.ndarray)) and not np.all(
            np.isfinite(value)
        ):
            raise ValueError(f"{name} must contain finite values.")
    for spec in fields(MotionConfig):
        value, rule = getattr(args, spec.name), spec.metadata
        if value is None:
            continue
        if rule.get("length") is not None and np.shape(value) != (rule["length"],):
            raise ValueError(f"{spec.name} must contain {rule['length']} values.")
        if rule.get("choices") is not None and value not in rule["choices"]:
            raise ValueError(f"{spec.name} must be one of {rule['choices']}.")
        if rule.get("positive") and value <= 0:
            raise ValueError(f"{spec.name} must be positive.")
        if rule.get("minimum") is not None and value < rule["minimum"]:
            raise ValueError(f"{spec.name} must be >= {rule['minimum']}.")
        if rule.get("maximum") is not None and value > rule["maximum"]:
            raise ValueError(f"{spec.name} must be <= {rule['maximum']}.")
    paths = [
        effective_pin_image_alignment_metadata(args),
        effective_post_two_pin_close_image_alignment_metadata(args),
    ]
    if args.auto_pin_image_align:
        if args.pin_image_reference is None:
            raise ValueError("auto_pin_image_align requires pin_image_reference.")
        if args.pin_image_alignment_metadata is not None:
            raise ValueError("Choose automatic image alignment or saved metadata, not both.")
        paths.append(args.pin_image_reference)
    for path in paths:
        if path is not None and not path.is_file():
            raise FileNotFoundError(f"Alignment file not found: {path}")
    if args.pin_image_roi is not None:
        x, y, width, height = args.pin_image_roi
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("pin_image_roi requires non-negative x/y and positive width/height.")
    if args.offset_z_mm is not None and args.marker_frame_mode != "raw":
        raise ValueError("offset_z_mm is only valid with marker_frame_mode raw.")
    if args.align_to_four_pin_frame and args.skip_gripper_close and args.resume != "pins":
        raise ValueError("Alignment tasks require skip_gripper_close: false.")
    if args.resume == "pins" and not args.approach_pin_after_rotation:
        raise ValueError("--resume pins requires approach_pin_after_rotation: true.")
    if args.insert_after_pin_approach and not args.approach_pin_after_rotation:
        raise ValueError("Insertion tasks require approach_pin_after_rotation: true.")
