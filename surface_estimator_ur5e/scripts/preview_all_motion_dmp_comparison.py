#!/usr/bin/env python3
"""Roll out all saved assembly DMPs for nominal and offset tasks in one viewer."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, replace
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

import generate_tray_dmp_dataset as dmpgen
import marker_based_motion as motion
import preview_assembly_motion as preview
import preview_tray_dmp_comparison as tray_sequence_viewer
from surface_estimator_ur5e.io import DEFAULT_JOINT_ORDER
from surface_estimator_ur5e.pyroki_ik import (
    PyrokiRTDEControlAdapter,
    transform_to_ur_pose,
    ur_pose_to_transform,
)
from surface_estimator_ur5e.robot_model import URDFRobotVisualizer
from surface_estimator_ur5e.visualization import _ceiling_mount_display_transform
from visualize_marker_frame import (
    add_base_plane,
    add_frame,
    add_marker_square,
    transform_points,
    translation_transform,
)


DEFAULT_MODEL = Path("data/dmp/all_assembly_motion_dmps.npz")
DEFAULT_OFFSET_MARKER_MM = np.asarray([20.0, -200.0, 0.0], dtype=float)


@dataclass(frozen=True)
class AllMotionWeights:
    """Saved parameters needed to roll out every move event without training."""

    event_kind: tuple[str, ...]
    event_name: tuple[str, ...]
    event_primitive_index: np.ndarray
    position_weights: np.ndarray
    orientation_weights: np.ndarray
    primitive_start_pose_ur: np.ndarray
    primitive_goal_pose_ur: np.ndarray
    basis_functions: int
    alpha: float
    beta: float
    cs_alpha: float
    internal_samples: int


@dataclass(frozen=True)
class TaskRollout:
    """One complete 35-event DMP task and its visual workpiece trajectories."""

    args: argparse.Namespace
    events: tuple[preview.MotionEvent, ...]
    frames: tuple[preview.AnimationFrame, ...]
    tray_transforms: tuple[np.ndarray, ...]
    tray_states: tuple[str, ...]
    cathode_transforms: tuple[np.ndarray, ...]
    cathode_states: tuple[str, ...]
    tray_fit_rms_m: float
    cathode_initial_fit_rms_m: float
    cathode_final_fit_rms_m: float


@dataclass(frozen=True)
class CombinedRollout:
    """Nominal task, return bridge, reset, and full offset task."""

    frames: tuple[preview.AnimationFrame, ...]
    nominal_tray_transforms: tuple[np.ndarray, ...]
    nominal_tray_states: tuple[str, ...]
    nominal_cathode_transforms: tuple[np.ndarray, ...]
    nominal_cathode_states: tuple[str, ...]
    offset_tray_transforms: tuple[np.ndarray, ...]
    offset_tray_states: tuple[str, ...]
    offset_cathode_transforms: tuple[np.ndarray, ...]
    offset_cathode_states: tuple[str, ...]
    offset_objects_visible: tuple[bool, ...]
    second_offset_tray_transforms: tuple[np.ndarray, ...]
    second_offset_tray_states: tuple[str, ...]
    second_offset_cathode_transforms: tuple[np.ndarray, ...]
    second_offset_cathode_states: tuple[str, ...]
    second_offset_objects_visible: tuple[bool, ...]
    nominal_frame_count: int
    transition_frame_count: int
    reset_frame_count: int
    offset_frame_count: int
    second_transition_frame_count: int
    second_reset_frame_count: int


def parse_args() -> argparse.Namespace:
    """Build the full-motion, read-only DMP comparison CLI."""

    parser = motion.build_arg_parser()
    parser.description = (
        "Use the saved 27 moveL DMPs and eight discrete events to animate one full "
        "nominal assembly task followed by one full marker-offset task. The robot, "
        "tray, cathode plate, and holders retain their original materials."
    )
    parser.set_defaults(
        align_to_four_pin_frame=True,
        insert_after_pin_approach=True,
        no_set_tcp=True,
        execute=False,
    )
    parser.add_argument("--dmp-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--simple-dmp-root", type=Path, default=dmpgen.DEFAULT_SIMPLE_DMP_ROOT)
    parser.add_argument(
        "--offset-marker-mm",
        type=float,
        nargs=3,
        default=DEFAULT_OFFSET_MARKER_MM.copy(),
        metavar=("DX", "DY", "DZ"),
    )
    parser.add_argument(
        "--offset-yaw-deg",
        type=float,
        default=0.0,
        help="Additional offset-holder rotation about marker-frame +Z.",
    )
    parser.add_argument(
        "--second-offset-marker-mm",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Optional third task's additional holder translation from nominal.",
    )
    parser.add_argument(
        "--second-offset-yaw-deg",
        type=float,
        default=0.0,
        help="Third task's additional holder rotation about marker-frame +Z.",
    )
    parser.add_argument(
        "--second-offset-recenter-release-tail",
        action="store_true",
        help=(
            "After the third task releases both parts, reuse the first offset's "
            "empty-gripper tail targets. Disabled by default."
        ),
    )
    parser.add_argument(
        "--offset-equivalent-grasp-y-flip",
        action="store_true",
        help=(
            "Use the Hand-E's equivalent 180-degree local-Y grasp for every offset "
            "move target. This is visualization-only and leaves the workpiece goal fixed."
        ),
    )
    parser.add_argument(
        "--offset-grasp-shift-z-mm",
        type=float,
        default=0.0,
        help=(
            "Shift the offset TCP grasp point along its post-flip local Z while keeping "
            "the visual workpiece goal fixed."
        ),
    )
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument("--playback-hz", type=float, default=20.0)
    parser.add_argument("--time-scale", type=float, default=4.0)
    parser.add_argument("--trajectory-step-mm", type=float, default=5.0)
    parser.add_argument("--trajectory-rotation-step-deg", type=float, default=3.0)
    parser.add_argument("--rotation-speed-deg-s", type=float, default=30.0)
    parser.add_argument("--return-joint-step-deg", type=float, default=1.0)
    parser.add_argument("--reset-frames", type=int, default=10)
    parser.add_argument(
        "--visual-tray-stl",
        type=Path,
        default=preview.DEFAULT_VISUAL_TRAY_STL,
    )
    parser.add_argument(
        "--visual-tray-handle-root-y-mm",
        type=float,
        default=preview.DEFAULT_VISUAL_TRAY_HANDLE_ROOT_Y_MM,
    )
    parser.add_argument(
        "--visual-tray-handle-scale",
        type=float,
        default=preview.DEFAULT_VISUAL_TRAY_HANDLE_SCALE,
    )
    parser.add_argument(
        "--visual-cathode-plate-stl",
        type=Path,
        default=preview.DEFAULT_VISUAL_CATHODE_PLATE_STL,
    )
    parser.add_argument(
        "--visual-flange-adapter-stl",
        type=Path,
        default=preview.DEFAULT_VISUAL_FLANGE_ADAPTER_STL,
    )
    args = parser.parse_args()
    if args.execute:
        parser.error("The full-motion DMP viewer is offline; --execute is forbidden.")
    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    for name in (
        "playback_hz",
        "time_scale",
        "trajectory_step_mm",
        "trajectory_rotation_step_deg",
        "rotation_speed_deg_s",
        "return_joint_step_deg",
    ):
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.reset_frames < 1:
        parser.error("--reset-frames must be at least one.")
    if not np.all(np.isfinite(args.offset_marker_mm)):
        parser.error("--offset-marker-mm must contain three finite values.")
    if not np.isfinite(args.offset_yaw_deg):
        parser.error("--offset-yaw-deg must be finite.")
    if args.second_offset_marker_mm is not None and not np.all(
        np.isfinite(args.second_offset_marker_mm)
    ):
        parser.error("--second-offset-marker-mm must contain three finite values.")
    if not np.isfinite(args.second_offset_yaw_deg):
        parser.error("--second-offset-yaw-deg must be finite.")
    if not np.isfinite(args.offset_grasp_shift_z_mm):
        parser.error("--offset-grasp-shift-z-mm must be finite.")
    if not 0.0 < args.visual_tray_handle_scale <= 1.0:
        parser.error("--visual-tray-handle-scale must be > 0 and <= 1.")
    continuation_flags = (
        args.continue_pin_sequence_from_current,
        args.continue_post_two_pin_release_from_current,
        args.continue_post_two_pin_after_close_from_current,
        args.continue_post_two_pin_tail_from_current,
        args.continue_post_two_pin_tail_next_from_current,
        args.continue_post_two_pin_tail_extra_from_current,
    )
    if any(continuation_flags):
        parser.error("Continuation modes cannot represent the complete task.")
    return args


def load_all_motion_weights(model_path: Path) -> AllMotionWeights:
    """Load and validate the reusable all-motion DMP artifact."""

    resolved = motion.resolve_project_path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(f"All-motion DMP model not found: {resolved}")
    required = (
        "event_kind",
        "event_name",
        "event_primitive_index",
        "position_weights",
        "orientation_weights",
        "primitive_start_pose_ur",
        "primitive_goal_pose_ur",
        "dmp_basis_functions",
        "dmp_alpha",
        "dmp_beta",
        "dmp_cs_alpha",
        "dmp_internal_samples",
    )
    with np.load(resolved) as loaded:
        missing = [name for name in required if name not in loaded]
        if missing:
            raise ValueError(f"All-motion DMP model is missing arrays: {missing}")
        arrays = {name: np.asarray(loaded[name]).copy() for name in required}
    event_kind = tuple(str(value) for value in arrays["event_kind"])
    event_name = tuple(str(value) for value in arrays["event_name"])
    event_primitive_index = np.asarray(arrays["event_primitive_index"], dtype=np.int32)
    position_weights = np.asarray(arrays["position_weights"], dtype=float)
    orientation_weights = np.asarray(arrays["orientation_weights"], dtype=float)
    primitive_start_pose_ur = np.asarray(arrays["primitive_start_pose_ur"], dtype=float)
    primitive_goal_pose_ur = np.asarray(arrays["primitive_goal_pose_ur"], dtype=float)
    if len(event_kind) != 35 or len(event_name) != 35:
        raise ValueError(f"Expected 35 saved events, got {len(event_kind)}.")
    if event_primitive_index.shape != (35,):
        raise ValueError("event_primitive_index must have shape (35,).")
    move_count = int(np.count_nonzero(event_primitive_index >= 0))
    if move_count != 27:
        raise ValueError(f"Expected 27 saved move primitives, got {move_count}.")
    if position_weights.shape != orientation_weights.shape:
        raise ValueError("Position and orientation weight shapes differ.")
    if position_weights.shape[0] != 27 or position_weights.shape[1] != 3:
        raise ValueError(f"Unexpected DMP weight shape {position_weights.shape}.")
    if primitive_start_pose_ur.shape != (27, 6) or primitive_goal_pose_ur.shape != (27, 6):
        raise ValueError("Saved primitive start/goal poses must both have shape (27, 6).")
    return AllMotionWeights(
        event_kind=event_kind,
        event_name=event_name,
        event_primitive_index=event_primitive_index,
        position_weights=position_weights,
        orientation_weights=orientation_weights,
        primitive_start_pose_ur=primitive_start_pose_ur,
        primitive_goal_pose_ur=primitive_goal_pose_ur,
        basis_functions=int(arrays["dmp_basis_functions"]),
        alpha=float(arrays["dmp_alpha"]),
        beta=float(arrays["dmp_beta"]),
        cs_alpha=float(arrays["dmp_cs_alpha"]),
        internal_samples=int(arrays["dmp_internal_samples"]),
    )


def prefixed_frame(frame: preview.AnimationFrame, prefix: str) -> preview.AnimationFrame:
    """Copy a frame while making its task half explicit in the status panel."""

    return preview.AnimationFrame(
        q_rad=np.asarray(frame.q_rad, dtype=float).copy(),
        tcp_pose_ur=np.asarray(frame.tcp_pose_ur, dtype=float).copy(),
        stage_name=f"{prefix} / {frame.stage_name}",
        gripper_percent=float(frame.gripper_percent),
        tray_attached=bool(frame.tray_attached),
        base_t_tray_center=np.asarray(frame.base_t_tray_center, dtype=float).copy(),
    )


def rollout_move_dmp(
    weights: AllMotionWeights,
    primitive_index: int,
    start_pose_ur: np.ndarray,
    goal_pose_ur: np.ndarray,
    sample_count: int,
    cartesian_dmp_class: type,
) -> np.ndarray:
    """Roll one saved DMP with new endpoints without calling train()."""

    start = np.asarray(start_pose_ur, dtype=float)
    goal = np.asarray(goal_pose_ur, dtype=float)
    nominal_start = weights.primitive_start_pose_ur[primitive_index]
    nominal_goal = weights.primitive_goal_pose_ur[primitive_index]
    model = cartesian_dmp_class(
        n_bfs=weights.basis_functions,
        alpha=weights.alpha,
        beta=weights.beta,
        cs_alpha=weights.cs_alpha,
    )
    model.position_dmp.w = weights.position_weights[primitive_index].copy()
    model.position_dmp.p0 = nominal_start[:3].copy()
    model.position_dmp.gp = nominal_goal[:3].copy()
    endpoint_quaternions = dmpgen.rotation_to_numpy_quaternion(
        Rotation.from_rotvec(np.stack([nominal_start[3:], nominal_goal[3:]]))
    )
    model.quaternion_dmp.w = weights.orientation_weights[primitive_index].copy()
    model.quaternion_dmp.q0 = endpoint_quaternions[0]
    model.quaternion_dmp.go = endpoint_quaternions[1]
    normalized_time = np.linspace(0.0, 1.0, weights.internal_samples)
    position, _velocity, _acceleration, orientation, _omega, _d_omega = model.rollout(
        normalized_time,
        1.0,
    )
    poses = np.column_stack(
        [position, dmpgen.numpy_quaternion_to_rotation(orientation).as_rotvec()]
    )
    poses = dmpgen.enforce_pose_endpoints(poses, nominal_start, nominal_goal)
    start_correction = ur_pose_to_transform(start) @ np.linalg.inv(
        ur_pose_to_transform(nominal_start)
    )
    goal_correction = ur_pose_to_transform(goal) @ np.linalg.inv(ur_pose_to_transform(nominal_goal))
    retargeted = np.empty_like(poses)
    for index, (fraction, nominal_pose) in enumerate(
        zip(np.linspace(0.0, 1.0, len(poses)), poses, strict=True)
    ):
        correction = preview.interpolate_transform(
            start_correction,
            goal_correction,
            float(fraction),
        )
        retargeted[index] = transform_to_ur_pose(correction @ ur_pose_to_transform(nominal_pose))
    retargeted = dmpgen.enforce_pose_endpoints(retargeted, start, goal)
    return dmpgen.resample_pose_sequence(retargeted, max(2, int(sample_count)))


def solve_pose(
    ik: PyrokiRTDEControlAdapter,
    pose_ur: np.ndarray,
    seed_q_rad: np.ndarray,
    stage_name: str,
) -> np.ndarray:
    """Solve one strict continuous IK sample."""

    pose = np.asarray(pose_ur, dtype=float)
    seed = np.asarray(seed_q_rad, dtype=float)
    if not ik.getInverseKinematicsHasSolution(
        pose.tolist(),
        seed.tolist(),
        motion.IK_POSITION_TOLERANCE_M,
        motion.IK_ORIENTATION_TOLERANCE_RAD,
    ):
        raise RuntimeError(f"No strict PyRoki IK solution during {stage_name!r}.")
    q_rad = np.asarray(
        ik.getInverseKinematics(
            pose.tolist(),
            seed.tolist(),
            motion.IK_POSITION_TOLERANCE_M,
            motion.IK_ORIENTATION_TOLERANCE_RAD,
        ),
        dtype=float,
    )
    joint_step_deg = float(np.max(np.abs(np.rad2deg(q_rad - seed))))
    if joint_step_deg > motion.MAX_IK_JOINT_STEP_DEG:
        joint_index = int(np.argmax(np.abs(q_rad - seed)))
        raise RuntimeError(
            f"IK branch jump of {joint_step_deg:.1f} deg on q{joint_index + 1} "
            f"during {stage_name!r}."
        )
    return q_rad


def blend_pose_sequences(
    linear_poses: np.ndarray,
    dmp_poses: np.ndarray,
    dmp_fraction: float,
) -> np.ndarray:
    """Blend a DMP primitive toward its endpoint-equivalent moveL path."""

    linear = np.asarray(linear_poses, dtype=float)
    dmp = np.asarray(dmp_poses, dtype=float)
    if linear.shape != dmp.shape or linear.ndim != 2 or linear.shape[1] != 6:
        raise ValueError("Linear and DMP pose sequences must have matching Nx6 shapes.")
    blended = np.empty_like(dmp)
    for index, (linear_pose, dmp_pose) in enumerate(zip(linear, dmp, strict=True)):
        transform = preview.interpolate_transform(
            ur_pose_to_transform(linear_pose),
            ur_pose_to_transform(dmp_pose),
            float(dmp_fraction),
        )
        blended[index] = transform_to_ur_pose(transform)
    blended[0] = dmp[0]
    blended[-1] = dmp[-1]
    return blended


def verify_event_program(
    events: list[preview.MotionEvent],
    weights: AllMotionWeights,
) -> None:
    """Ensure the rebuilt real program still matches the trained event program."""

    if len(events) != len(weights.event_name):
        raise ValueError(f"Expected {len(weights.event_name)} events, rebuilt {len(events)}.")
    for index, event in enumerate(events):
        if event.kind != weights.event_kind[index] or event.name != weights.event_name[index]:
            raise ValueError(
                f"Event {index} differs from the trained program: "
                f"{event.kind}/{event.name!r} vs "
                f"{weights.event_kind[index]}/{weights.event_name[index]!r}."
            )


def equivalent_offset_grasp_events(
    events: list[preview.MotionEvent],
    flip_local_y: bool,
    shift_local_z_mm: float,
) -> list[preview.MotionEvent]:
    """Change only the offset TCP grasp gauge, never the workpiece goal."""

    if not flip_local_y and abs(float(shift_local_z_mm)) < 1e-12:
        return events
    local_y_flip = Rotation.from_rotvec([0.0, np.pi, 0.0]).as_matrix()
    transformed: list[preview.MotionEvent] = []
    for event in events:
        if event.kind != "move":
            transformed.append(event)
            continue
        assert event.target_pose_ur is not None
        base_t_tcp = ur_pose_to_transform(event.target_pose_ur)
        if flip_local_y:
            base_t_tcp[:3, :3] = base_t_tcp[:3, :3] @ local_y_flip
        base_t_tcp[:3, 3] += base_t_tcp[:3, 2] * float(shift_local_z_mm) / 1000.0
        transformed.append(replace(event, target_pose_ur=transform_to_ur_pose(base_t_tcp)))
    return transformed


def recenter_post_release_events(
    events: list[preview.MotionEvent],
    reference_events: tuple[preview.MotionEvent, ...],
) -> list[preview.MotionEvent]:
    """Reuse a reachable empty-gripper tail after the final part release."""

    if len(events) != len(reference_events):
        raise ValueError("Task and post-release reference programs differ in length.")
    first_tail_name = "post-two-pin release TCP +Y retreat"
    first_tail_index = next(
        (index for index, event in enumerate(events) if event.name == first_tail_name),
        None,
    )
    if first_tail_index is None:
        raise ValueError(f"Could not find post-release tail event {first_tail_name!r}.")
    recentered = list(events)
    for index in range(first_tail_index, len(events)):
        event = events[index]
        reference = reference_events[index]
        if event.kind != reference.kind or event.name != reference.name:
            raise ValueError(f"Post-release reference event {index} does not match.")
        if event.kind == "move":
            assert reference.target_pose_ur is not None
            recentered[index] = replace(
                event,
                target_pose_ur=np.asarray(reference.target_pose_ur, dtype=float).copy(),
            )
    return recentered


def rollout_event_program(
    args: argparse.Namespace,
    ik: PyrokiRTDEControlAdapter,
    initial_q_rad: np.ndarray,
    initial_tcp_pose_ur: np.ndarray,
    events: list[preview.MotionEvent],
    weights: AllMotionWeights,
    cartesian_dmp_class: type,
) -> list[preview.AnimationFrame]:
    """Roll all 27 DMPs and preserve the eight real discrete events."""

    verify_event_program(events, weights)
    current_q = np.asarray(initial_q_rad, dtype=float).copy()
    current_pose = np.asarray(initial_tcp_pose_ur, dtype=float).copy()
    gripper_percent = 0.0
    tray_attached = False
    frames = [
        preview.AnimationFrame(
            current_q.copy(),
            current_pose.copy(),
            "initial joint pose",
            gripper_percent,
            tray_attached,
            np.eye(4),
        )
    ]
    for event_index, event in enumerate(events):
        if event.kind == "move":
            assert event.target_pose_ur is not None
            assert event.speed_m_s is not None
            primitive_index = int(weights.event_primitive_index[event_index])
            if primitive_index < 0:
                raise ValueError(f"Move event {event_index} has no saved primitive.")
            linear_samples = preview.interpolate_move(
                current_pose,
                event.target_pose_ur,
                event.speed_m_s,
                args,
            )
            poses = rollout_move_dmp(
                weights,
                primitive_index,
                current_pose,
                np.asarray(event.target_pose_ur, dtype=float),
                len(linear_samples) + 1,
                cartesian_dmp_class,
            )
            linear_poses = np.vstack([current_pose, np.asarray(linear_samples, dtype=float)])
            solved_poses: np.ndarray | None = None
            solved_q: list[np.ndarray] | None = None
            last_error: RuntimeError | None = None
            for dmp_fraction in (1.0, 0.90, 0.75, 0.50, 0.25, 0.10, 0.05, 0.0):
                candidate_poses = (
                    poses
                    if dmp_fraction == 1.0
                    else blend_pose_sequences(linear_poses, poses, dmp_fraction)
                )
                candidate_q: list[np.ndarray] = []
                q_near = current_q.copy()
                try:
                    for pose in candidate_poses[1:]:
                        q_near = solve_pose(ik, pose, q_near, event.name)
                        candidate_q.append(q_near)
                except RuntimeError as exc:
                    last_error = exc
                    continue
                solved_poses = candidate_poses
                solved_q = candidate_q
                if dmp_fraction < 1.0:
                    print(
                        f"IK-safe DMP repair for {event.name!r}: "
                        f"retained {100.0 * dmp_fraction:.0f}% DMP shape."
                    )
                break
            if solved_poses is None or solved_q is None:
                assert last_error is not None
                raise RuntimeError(
                    f"No strict IK path for DMP primitive {event.name!r}, even after "
                    "moveL projection."
                ) from last_error
            for pose, q_rad in zip(solved_poses[1:], solved_q, strict=True):
                current_q = q_rad
                current_pose = np.asarray(pose, dtype=float)
                frames.append(
                    preview.AnimationFrame(
                        current_q.copy(),
                        current_pose.copy(),
                        event.name,
                        gripper_percent,
                        tray_attached,
                        np.eye(4),
                    )
                )
        elif event.kind == "gripper":
            assert event.gripper_percent is not None
            previous_percent = gripper_percent
            pause_frames = max(2, int(np.ceil(0.5 * args.playback_hz / args.time_scale)))
            for fraction in np.linspace(0.0, 1.0, pause_frames + 1)[1:]:
                gripper_percent = float(
                    (1.0 - fraction) * previous_percent + fraction * event.gripper_percent
                )
                frames.append(
                    preview.AnimationFrame(
                        current_q.copy(),
                        current_pose.copy(),
                        event.name,
                        gripper_percent,
                        tray_attached,
                        np.eye(4),
                    )
                )
            if event.tray_action == "attach":
                tray_attached = True
            elif event.tray_action == "detach":
                tray_attached = False
            frames.append(
                preview.AnimationFrame(
                    current_q.copy(),
                    current_pose.copy(),
                    event.name,
                    gripper_percent,
                    tray_attached,
                    np.eye(4),
                )
            )
        elif event.kind == "dwell":
            pause_frames = max(
                1,
                int(np.ceil(event.dwell_s * args.playback_hz / args.time_scale)),
            )
            frames.extend(
                preview.AnimationFrame(
                    current_q.copy(),
                    current_pose.copy(),
                    event.name,
                    gripper_percent,
                    tray_attached,
                    np.eye(4),
                )
                for _ in range(pause_frames)
            )
        else:
            raise AssertionError(f"Unhandled event kind: {event.kind}")
    return frames


def tray_states(frames: list[preview.AnimationFrame]) -> tuple[str, ...]:
    """Label waiting, attached, and seated tray states."""

    ever_attached = False
    states: list[str] = []
    for frame in frames:
        ever_attached = ever_attached or frame.tray_attached
        if frame.tray_attached:
            states.append("attached to TCP")
        elif ever_attached:
            states.append("seated on holder pins")
        else:
            states.append("waiting at marker")
    return tuple(states)


def cathode_states(
    frame_count: int,
    animation: preview.CathodePlateAnimation,
) -> tuple[str, ...]:
    """Label the cathode plate's two-pin, attached, and four-pin states."""

    states: list[str] = []
    for index in range(frame_count):
        if index <= animation.attach_frame:
            states.append("seated on adjacent two pins")
        elif index < animation.detach_frame:
            states.append("attached to TCP")
        else:
            states.append("seated on four pins")
    return tuple(states)


def build_task_rollout(
    task_args: argparse.Namespace,
    ik: PyrokiRTDEControlAdapter,
    weights: AllMotionWeights,
    cartesian_dmp_class: type,
    tray_mesh: object,
    tray_hole_centers: np.ndarray,
    cathode_mesh: object,
    cathode_hole_centers: np.ndarray,
    equivalent_grasp_y_flip: bool = False,
    grasp_shift_z_mm: float = 0.0,
    post_release_reference_events: tuple[preview.MotionEvent, ...] | None = None,
) -> tuple[TaskRollout, np.ndarray, np.ndarray]:
    """Rebuild real targets, roll every DMP, and compute workpiece animation."""

    initial_q, initial_tcp, base_t_marker, events = preview.build_actual_motion_events(
        task_args,
        ik,
    )
    if equivalent_grasp_y_flip:
        initial_q = np.asarray(initial_q, dtype=float).copy()
        initial_q[-1] -= 2.0 * np.pi
    if post_release_reference_events is not None:
        events = recenter_post_release_events(events, post_release_reference_events)
    events = equivalent_offset_grasp_events(
        events,
        equivalent_grasp_y_flip,
        grasp_shift_z_mm,
    )
    frames = rollout_event_program(
        task_args,
        ik,
        initial_q,
        initial_tcp,
        events,
        weights,
        cartesian_dmp_class,
    )
    (
        tray_transforms,
        _base_pin_centers,
        tray_fit_rms_m,
        _tray_axial_shift_m,
        _marker_plane_error_m,
    ) = preview.visual_tray_transforms(
        task_args,
        base_t_marker,
        frames,
        np.asarray(tray_mesh.vertices, dtype=float),
        tray_hole_centers,
    )
    cathode_animation = preview.visual_cathode_plate_animation(
        task_args,
        base_t_marker,
        frames,
        np.asarray(cathode_mesh.vertices, dtype=float),
        cathode_hole_centers,
    )
    return (
        TaskRollout(
            args=task_args,
            events=tuple(events),
            frames=tuple(frames),
            tray_transforms=tuple(np.asarray(value, dtype=float) for value in tray_transforms),
            tray_states=tray_states(frames),
            cathode_transforms=tuple(
                np.asarray(value, dtype=float) for value in cathode_animation.transforms
            ),
            cathode_states=cathode_states(len(frames), cathode_animation),
            tray_fit_rms_m=float(tray_fit_rms_m),
            cathode_initial_fit_rms_m=float(cathode_animation.initial_two_pin_rms_m),
            cathode_final_fit_rms_m=float(cathode_animation.final_four_pin_lateral_rms_m),
        ),
        np.asarray(initial_q, dtype=float),
        np.asarray(base_t_marker, dtype=float),
    )


def combine_rollouts(
    nominal: TaskRollout,
    offset: TaskRollout,
    second_offset: TaskRollout | None,
    initial_q_rad: np.ndarray,
    ik: PyrokiRTDEControlAdapter,
    args: argparse.Namespace,
) -> CombinedRollout:
    """Join nominal and up to two offset tasks with returns and scene resets."""

    frames = [prefixed_frame(frame, "nominal") for frame in nominal.frames]
    second_seed = offset if second_offset is None else second_offset
    nominal_tray_transforms = [value.copy() for value in nominal.tray_transforms]
    nominal_tray_states = list(nominal.tray_states)
    nominal_cathode_transforms = [value.copy() for value in nominal.cathode_transforms]
    nominal_cathode_states = list(nominal.cathode_states)
    offset_tray_transforms = [offset.tray_transforms[0].copy() for _ in nominal.frames]
    offset_tray_states = ["not introduced" for _ in nominal.frames]
    offset_cathode_transforms = [offset.cathode_transforms[0].copy() for _ in nominal.frames]
    offset_cathode_states = ["not introduced" for _ in nominal.frames]
    offset_objects_visible = [False for _ in nominal.frames]
    second_offset_tray_transforms = [second_seed.tray_transforms[0].copy() for _ in nominal.frames]
    second_offset_tray_states = ["not introduced" for _ in nominal.frames]
    second_offset_cathode_transforms = [
        second_seed.cathode_transforms[0].copy() for _ in nominal.frames
    ]
    second_offset_cathode_states = ["not introduced" for _ in nominal.frames]
    second_offset_objects_visible = [False for _ in nominal.frames]

    last_frame = nominal.frames[-1]
    q_start = np.asarray(last_frame.q_rad, dtype=float)
    q_goal = np.asarray(initial_q_rad, dtype=float)
    max_delta_deg = float(np.max(np.abs(np.rad2deg(q_goal - q_start))))
    transition_count = max(2, int(np.ceil(max_delta_deg / args.return_joint_step_deg)))
    for fraction in np.linspace(0.0, 1.0, transition_count + 1)[1:]:
        q_rad = (1.0 - fraction) * q_start + fraction * q_goal
        tcp_pose_ur = np.asarray(ik.getForwardKinematics(q_rad.tolist()), dtype=float)
        gripper_percent = (1.0 - fraction) * last_frame.gripper_percent
        frames.append(
            preview.AnimationFrame(
                q_rad=q_rad,
                tcp_pose_ur=tcp_pose_ur,
                stage_name="transition / return robot to shared task start",
                gripper_percent=float(gripper_percent),
                tray_attached=False,
                base_t_tray_center=np.eye(4),
            )
        )
        nominal_tray_transforms.append(nominal.tray_transforms[-1].copy())
        nominal_tray_states.append(nominal.tray_states[-1])
        nominal_cathode_transforms.append(nominal.cathode_transforms[-1].copy())
        nominal_cathode_states.append(nominal.cathode_states[-1])
        offset_tray_transforms.append(offset.tray_transforms[0].copy())
        offset_tray_states.append("not introduced")
        offset_cathode_transforms.append(offset.cathode_transforms[0].copy())
        offset_cathode_states.append("not introduced")
        offset_objects_visible.append(False)
        second_offset_tray_transforms.append(second_seed.tray_transforms[0].copy())
        second_offset_tray_states.append("not introduced")
        second_offset_cathode_transforms.append(second_seed.cathode_transforms[0].copy())
        second_offset_cathode_states.append("not introduced")
        second_offset_objects_visible.append(False)

    for _ in range(args.reset_frames):
        source = offset.frames[0]
        frames.append(
            preview.AnimationFrame(
                q_rad=np.asarray(source.q_rad, dtype=float).copy(),
                tcp_pose_ur=np.asarray(source.tcp_pose_ur, dtype=float).copy(),
                stage_name="transition / reset workpieces at shared pickup and offset holder",
                gripper_percent=float(source.gripper_percent),
                tray_attached=False,
                base_t_tray_center=np.eye(4),
            )
        )
        nominal_tray_transforms.append(nominal.tray_transforms[-1].copy())
        nominal_tray_states.append(nominal.tray_states[-1])
        nominal_cathode_transforms.append(nominal.cathode_transforms[-1].copy())
        nominal_cathode_states.append(nominal.cathode_states[-1])
        offset_tray_transforms.append(offset.tray_transforms[0].copy())
        offset_tray_states.append(offset.tray_states[0])
        offset_cathode_transforms.append(offset.cathode_transforms[0].copy())
        offset_cathode_states.append(offset.cathode_states[0])
        offset_objects_visible.append(True)
        second_offset_tray_transforms.append(second_seed.tray_transforms[0].copy())
        second_offset_tray_states.append("not introduced")
        second_offset_cathode_transforms.append(second_seed.cathode_transforms[0].copy())
        second_offset_cathode_states.append("not introduced")
        second_offset_objects_visible.append(False)

    frames.extend(prefixed_frame(frame, "offset-1") for frame in offset.frames)
    nominal_tray_transforms.extend(nominal.tray_transforms[-1].copy() for _ in offset.frames)
    nominal_tray_states.extend(nominal.tray_states[-1] for _ in offset.frames)
    nominal_cathode_transforms.extend(nominal.cathode_transforms[-1].copy() for _ in offset.frames)
    nominal_cathode_states.extend(nominal.cathode_states[-1] for _ in offset.frames)
    offset_tray_transforms.extend(value.copy() for value in offset.tray_transforms)
    offset_tray_states.extend(offset.tray_states)
    offset_cathode_transforms.extend(value.copy() for value in offset.cathode_transforms)
    offset_cathode_states.extend(offset.cathode_states)
    offset_objects_visible.extend(True for _ in offset.frames)
    second_offset_tray_transforms.extend(
        second_seed.tray_transforms[0].copy() for _ in offset.frames
    )
    second_offset_tray_states.extend("not introduced" for _ in offset.frames)
    second_offset_cathode_transforms.extend(
        second_seed.cathode_transforms[0].copy() for _ in offset.frames
    )
    second_offset_cathode_states.extend("not introduced" for _ in offset.frames)
    second_offset_objects_visible.extend(False for _ in offset.frames)

    second_transition_count = 0
    second_reset_count = 0
    if second_offset is not None:
        last_frame = offset.frames[-1]
        q_start = np.asarray(last_frame.q_rad, dtype=float)
        q_goal = np.asarray(initial_q_rad, dtype=float)
        max_delta_deg = float(np.max(np.abs(np.rad2deg(q_goal - q_start))))
        second_transition_count = max(2, int(np.ceil(max_delta_deg / args.return_joint_step_deg)))
        for fraction in np.linspace(0.0, 1.0, second_transition_count + 1)[1:]:
            q_rad = (1.0 - fraction) * q_start + fraction * q_goal
            tcp_pose_ur = np.asarray(ik.getForwardKinematics(q_rad.tolist()), dtype=float)
            gripper_percent = (1.0 - fraction) * last_frame.gripper_percent
            frames.append(
                preview.AnimationFrame(
                    q_rad=q_rad,
                    tcp_pose_ur=tcp_pose_ur,
                    stage_name="transition / return robot before offset-2",
                    gripper_percent=float(gripper_percent),
                    tray_attached=False,
                    base_t_tray_center=np.eye(4),
                )
            )
            nominal_tray_transforms.append(nominal.tray_transforms[-1].copy())
            nominal_tray_states.append(nominal.tray_states[-1])
            nominal_cathode_transforms.append(nominal.cathode_transforms[-1].copy())
            nominal_cathode_states.append(nominal.cathode_states[-1])
            offset_tray_transforms.append(offset.tray_transforms[-1].copy())
            offset_tray_states.append(offset.tray_states[-1])
            offset_cathode_transforms.append(offset.cathode_transforms[-1].copy())
            offset_cathode_states.append(offset.cathode_states[-1])
            offset_objects_visible.append(True)
            second_offset_tray_transforms.append(second_offset.tray_transforms[0].copy())
            second_offset_tray_states.append("not introduced")
            second_offset_cathode_transforms.append(second_offset.cathode_transforms[0].copy())
            second_offset_cathode_states.append("not introduced")
            second_offset_objects_visible.append(False)

        second_reset_count = args.reset_frames
        for _ in range(second_reset_count):
            source = second_offset.frames[0]
            frames.append(
                preview.AnimationFrame(
                    q_rad=np.asarray(source.q_rad, dtype=float).copy(),
                    tcp_pose_ur=np.asarray(source.tcp_pose_ur, dtype=float).copy(),
                    stage_name="transition / introduce offset-2 workpieces",
                    gripper_percent=float(source.gripper_percent),
                    tray_attached=False,
                    base_t_tray_center=np.eye(4),
                )
            )
            nominal_tray_transforms.append(nominal.tray_transforms[-1].copy())
            nominal_tray_states.append(nominal.tray_states[-1])
            nominal_cathode_transforms.append(nominal.cathode_transforms[-1].copy())
            nominal_cathode_states.append(nominal.cathode_states[-1])
            offset_tray_transforms.append(offset.tray_transforms[-1].copy())
            offset_tray_states.append(offset.tray_states[-1])
            offset_cathode_transforms.append(offset.cathode_transforms[-1].copy())
            offset_cathode_states.append(offset.cathode_states[-1])
            offset_objects_visible.append(True)
            second_offset_tray_transforms.append(second_offset.tray_transforms[0].copy())
            second_offset_tray_states.append(second_offset.tray_states[0])
            second_offset_cathode_transforms.append(second_offset.cathode_transforms[0].copy())
            second_offset_cathode_states.append(second_offset.cathode_states[0])
            second_offset_objects_visible.append(True)

        frames.extend(prefixed_frame(frame, "offset-2") for frame in second_offset.frames)
        nominal_tray_transforms.extend(
            nominal.tray_transforms[-1].copy() for _ in second_offset.frames
        )
        nominal_tray_states.extend(nominal.tray_states[-1] for _ in second_offset.frames)
        nominal_cathode_transforms.extend(
            nominal.cathode_transforms[-1].copy() for _ in second_offset.frames
        )
        nominal_cathode_states.extend(nominal.cathode_states[-1] for _ in second_offset.frames)
        offset_tray_transforms.extend(
            offset.tray_transforms[-1].copy() for _ in second_offset.frames
        )
        offset_tray_states.extend(offset.tray_states[-1] for _ in second_offset.frames)
        offset_cathode_transforms.extend(
            offset.cathode_transforms[-1].copy() for _ in second_offset.frames
        )
        offset_cathode_states.extend(offset.cathode_states[-1] for _ in second_offset.frames)
        offset_objects_visible.extend(True for _ in second_offset.frames)
        second_offset_tray_transforms.extend(
            value.copy() for value in second_offset.tray_transforms
        )
        second_offset_tray_states.extend(second_offset.tray_states)
        second_offset_cathode_transforms.extend(
            value.copy() for value in second_offset.cathode_transforms
        )
        second_offset_cathode_states.extend(second_offset.cathode_states)
        second_offset_objects_visible.extend(True for _ in second_offset.frames)

    expected_length = len(frames)
    for values in (
        nominal_tray_transforms,
        nominal_tray_states,
        nominal_cathode_transforms,
        nominal_cathode_states,
        offset_tray_transforms,
        offset_tray_states,
        offset_cathode_transforms,
        offset_cathode_states,
        offset_objects_visible,
        second_offset_tray_transforms,
        second_offset_tray_states,
        second_offset_cathode_transforms,
        second_offset_cathode_states,
        second_offset_objects_visible,
    ):
        if len(values) != expected_length:
            raise AssertionError("Combined full-task visualization arrays differ in length.")
    return CombinedRollout(
        frames=tuple(frames),
        nominal_tray_transforms=tuple(nominal_tray_transforms),
        nominal_tray_states=tuple(nominal_tray_states),
        nominal_cathode_transforms=tuple(nominal_cathode_transforms),
        nominal_cathode_states=tuple(nominal_cathode_states),
        offset_tray_transforms=tuple(offset_tray_transforms),
        offset_tray_states=tuple(offset_tray_states),
        offset_cathode_transforms=tuple(offset_cathode_transforms),
        offset_cathode_states=tuple(offset_cathode_states),
        offset_objects_visible=tuple(offset_objects_visible),
        second_offset_tray_transforms=tuple(second_offset_tray_transforms),
        second_offset_tray_states=tuple(second_offset_tray_states),
        second_offset_cathode_transforms=tuple(second_offset_cathode_transforms),
        second_offset_cathode_states=tuple(second_offset_cathode_states),
        second_offset_objects_visible=tuple(second_offset_objects_visible),
        nominal_frame_count=len(nominal.frames),
        transition_frame_count=transition_count,
        reset_frame_count=args.reset_frames,
        offset_frame_count=len(offset.frames),
        second_transition_frame_count=second_transition_count,
        second_reset_frame_count=second_reset_count,
    )


def launch_viewer(
    args: argparse.Namespace,
    ik: PyrokiRTDEControlAdapter,
    base_t_marker: np.ndarray,
    nominal: TaskRollout,
    offset: TaskRollout,
    second_offset: TaskRollout | None,
    combined: CombinedRollout,
    tray_mesh: object,
    tray_hole_centers: np.ndarray,
    resolved_tray_path: Path,
    cathode_mesh: object,
    cathode_hole_centers: np.ndarray,
    resolved_cathode_path: Path,
) -> None:
    """Render one normally colored robot and both full task locations."""

    import viser
    from viser.extras import ViserUrdf

    display_transform = _ceiling_mount_display_transform()
    display_t_marker = display_transform @ base_t_marker
    server = viser.ViserServer(port=args.port)
    print("Viser server created; adding scene geometry.", flush=True)
    scene = server.scene
    add_base_plane(server, display_transform)
    add_frame(server, "/frames/robot_base", display_transform, axes_length=0.04)
    add_frame(server, "/frames/marker_floor", display_t_marker, axes_length=0.020)
    add_marker_square(server, "/marker/floor_square", display_t_marker, 0.0254)
    nominal_holder_points = tray_sequence_viewer.add_original_holder(
        scene,
        Path(args.assembly_obj),
        tray_sequence_viewer.assembly_display_transform(nominal.args, display_t_marker),
        "/holders/nominal",
    )
    offset_holder_points = tray_sequence_viewer.add_original_holder(
        scene,
        Path(args.assembly_obj),
        tray_sequence_viewer.assembly_display_transform(offset.args, display_t_marker),
        "/holders/offset",
    )
    second_offset_holder_points = np.empty((0, 3), dtype=float)
    if second_offset is not None:
        second_offset_holder_points = tray_sequence_viewer.add_original_holder(
            scene,
            Path(args.assembly_obj),
            tray_sequence_viewer.assembly_display_transform(second_offset.args, display_t_marker),
            "/holders/second_offset",
        )
    print("Holder geometry added.", flush=True)

    visualizer = URDFRobotVisualizer()
    robot_root_transform = visualizer.root_transform_for_robot_base(display_transform)
    scene.add_frame(
        "/animated_robot",
        position=robot_root_transform[:3, 3],
        wxyz=Rotation.from_matrix(robot_root_transform[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    viser_urdf = ViserUrdf(server, ik.urdf, root_node_name="/animated_robot")
    print("Robot URDF added.", flush=True)

    first_frame = combined.frames[0]
    first_display_t_tcp = display_transform @ ur_pose_to_transform(first_frame.tcp_pose_ur)
    tcp_handle = scene.add_frame(
        "/animation/tcp",
        position=first_display_t_tcp[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_tcp[:3, :3]).as_quat()[[3, 0, 1, 2]],
        axes_length=0.022,
        axes_radius=0.0007,
    )
    tcp_marker = scene.add_icosphere(
        "/animation/current_tcp_point",
        position=first_display_t_tcp[:3, 3],
        radius=0.004,
        color=(30, 120, 250),
    )

    first_base_t_adapter = preview.base_t_hande_mount(first_frame.tcp_pose_ur, args.tcp_offset_ur)
    first_display_t_adapter = display_transform @ first_base_t_adapter
    adapter_root = scene.add_frame(
        "/animation/flange_camera_adapter",
        position=first_display_t_adapter[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_adapter[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    adapter_mesh, resolved_adapter_path = preview.load_preview_flange_adapter_mesh(
        Path(args.visual_flange_adapter_stl)
    )
    scene.add_mesh_trimesh("/animation/flange_camera_adapter/mesh", adapter_mesh)
    first_base_t_hande = first_base_t_adapter @ translation_transform(
        [0.0, 0.0, preview.FLANGE_ADAPTER_STACK_HEIGHT_M]
    )
    first_display_t_hande = display_transform @ first_base_t_hande
    hande_root = scene.add_frame(
        "/animation/hande",
        position=first_display_t_hande[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_hande[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    hande_mesh_dir = motion.resolve_project_path(preview.HANDE_MESH_DIR)
    coupler_mesh = preview.load_preview_hande_mesh(
        hande_mesh_dir / "io_coupler.obj", 600, (150, 155, 160, 255)
    )
    body_mesh = preview.load_preview_hande_mesh(
        hande_mesh_dir / "hande.obj", 1200, (45, 48, 52, 255)
    )
    finger_mesh = preview.load_preview_hande_mesh(
        hande_mesh_dir / "finger.obj", 500, (70, 72, 70, 255)
    )
    scene.add_mesh_trimesh("/animation/hande/io_coupler", coupler_mesh)
    scene.add_mesh_trimesh(
        "/animation/hande/body",
        body_mesh,
        position=(0.0, 0.0, preview.HANDE_COUPLER_HEIGHT_M),
    )
    left_finger = scene.add_frame(
        "/animation/hande/left_finger",
        position=(
            preview.HANDE_FINGER_OFFSET_M,
            0.0,
            preview.HANDE_COUPLER_HEIGHT_M + preview.HANDE_BODY_HEIGHT_M,
        ),
        show_axes=False,
    )
    scene.add_mesh_trimesh("/animation/hande/left_finger/mesh", finger_mesh)
    right_finger = scene.add_frame(
        "/animation/hande/right_finger",
        position=(
            -preview.HANDE_FINGER_OFFSET_M,
            0.0,
            preview.HANDE_COUPLER_HEIGHT_M + preview.HANDE_BODY_HEIGHT_M,
        ),
        wxyz=(0.0, 0.0, 0.0, 1.0),
        show_axes=False,
    )
    scene.add_mesh_trimesh("/animation/hande/right_finger/mesh", finger_mesh)
    print("End-effector geometry added.", flush=True)

    first_display_t_nominal_tray = display_transform @ combined.nominal_tray_transforms[0]
    nominal_tray_root = scene.add_frame(
        "/nominal_tray",
        position=first_display_t_nominal_tray[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_nominal_tray[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    scene.add_mesh_trimesh("/nominal_tray/mesh", tray_mesh.copy())
    for index, hole_center in enumerate(tray_hole_centers):
        scene.add_icosphere(
            f"/nominal_tray/hole_centers/{index}",
            position=hole_center,
            radius=0.0015,
            color=(25, 170, 255),
        )

    first_display_t_offset_tray = display_transform @ combined.offset_tray_transforms[0]
    offset_tray_root = scene.add_frame(
        "/offset_tray",
        position=first_display_t_offset_tray[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_offset_tray[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
        visible=False,
    )
    scene.add_mesh_trimesh("/offset_tray/mesh", tray_mesh.copy())
    for index, hole_center in enumerate(tray_hole_centers):
        scene.add_icosphere(
            f"/offset_tray/hole_centers/{index}",
            position=hole_center,
            radius=0.0015,
            color=(25, 170, 255),
        )

    first_display_t_nominal_cathode = display_transform @ combined.nominal_cathode_transforms[0]
    nominal_cathode_root = scene.add_frame(
        "/nominal_cathode_plate",
        position=first_display_t_nominal_cathode[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_nominal_cathode[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    scene.add_mesh_trimesh("/nominal_cathode_plate/mesh", cathode_mesh.copy())
    handle_holes = set(np.argsort(cathode_hole_centers[:, 0])[:2].tolist())
    for index, hole_center in enumerate(cathode_hole_centers):
        scene.add_icosphere(
            f"/nominal_cathode_plate/hole_centers/{index}",
            position=hole_center,
            radius=0.0015,
            color=(255, 145, 35) if index in handle_holes else (180, 60, 235),
        )

    second_offset_tray_root = None
    second_offset_cathode_root = None
    if second_offset is not None:
        first_display_t_second_offset_tray = (
            display_transform @ combined.second_offset_tray_transforms[0]
        )
        second_offset_tray_root = scene.add_frame(
            "/second_offset_tray",
            position=first_display_t_second_offset_tray[:3, 3],
            wxyz=Rotation.from_matrix(first_display_t_second_offset_tray[:3, :3]).as_quat()[
                [3, 0, 1, 2]
            ],
            show_axes=False,
            visible=False,
        )
        scene.add_mesh_trimesh("/second_offset_tray/mesh", tray_mesh.copy())
        for index, hole_center in enumerate(tray_hole_centers):
            scene.add_icosphere(
                f"/second_offset_tray/hole_centers/{index}",
                position=hole_center,
                radius=0.0015,
                color=(25, 170, 255),
            )

        first_display_t_second_offset_cathode = (
            display_transform @ combined.second_offset_cathode_transforms[0]
        )
        second_offset_cathode_root = scene.add_frame(
            "/second_offset_cathode_plate",
            position=first_display_t_second_offset_cathode[:3, 3],
            wxyz=Rotation.from_matrix(first_display_t_second_offset_cathode[:3, :3]).as_quat()[
                [3, 0, 1, 2]
            ],
            show_axes=False,
            visible=False,
        )
        scene.add_mesh_trimesh("/second_offset_cathode_plate/mesh", cathode_mesh.copy())
        for index, hole_center in enumerate(cathode_hole_centers):
            scene.add_icosphere(
                f"/second_offset_cathode_plate/hole_centers/{index}",
                position=hole_center,
                radius=0.0015,
                color=(255, 145, 35) if index in handle_holes else (180, 60, 235),
            )

    first_display_t_offset_cathode = display_transform @ combined.offset_cathode_transforms[0]
    offset_cathode_root = scene.add_frame(
        "/offset_cathode_plate",
        position=first_display_t_offset_cathode[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_offset_cathode[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
        visible=False,
    )
    scene.add_mesh_trimesh("/offset_cathode_plate/mesh", cathode_mesh.copy())
    for index, hole_center in enumerate(cathode_hole_centers):
        scene.add_icosphere(
            f"/offset_cathode_plate/hole_centers/{index}",
            position=hole_center,
            radius=0.0015,
            color=(255, 145, 35) if index in handle_holes else (180, 60, 235),
        )

    print("Workpiece geometry added.", flush=True)

    motion_paths = preview.add_solved_paths(scene, display_transform, list(combined.frames))
    print("Motion paths added.", flush=True)
    tcp_points = transform_points(
        display_transform,
        np.asarray([frame.tcp_pose_ur[:3] for frame in combined.frames]),
    )
    tray_points = np.asarray(
        [
            (display_transform @ transform)[:3, 3]
            for transforms in (
                combined.nominal_tray_transforms,
                combined.offset_tray_transforms,
            )
            for transform in transforms
        ]
    )
    cathode_points = np.asarray(
        [
            (display_transform @ transform)[:3, 3]
            for transforms in (
                combined.nominal_cathode_transforms,
                combined.offset_cathode_transforms,
            )
            for transform in transforms
        ]
    )
    camera_point_groups = [
        nominal_holder_points,
        offset_holder_points,
        tcp_points,
        tray_points,
        cathode_points,
    ]
    if second_offset is not None:
        camera_point_groups.extend(
            [
                second_offset_holder_points,
                np.asarray(
                    [
                        (display_transform @ transform)[:3, 3]
                        for transform in combined.second_offset_tray_transforms
                    ]
                ),
                np.asarray(
                    [
                        (display_transform @ transform)[:3, 3]
                        for transform in combined.second_offset_cathode_transforms
                    ]
                ),
            ]
        )
    preview.set_camera(
        server,
        np.vstack(camera_point_groups),
    )

    play = server.gui.add_checkbox("Play", initial_value=True)
    loop = server.gui.add_checkbox("Loop", initial_value=not args.no_loop)
    show_path = server.gui.add_checkbox("Motion path", initial_value=True)

    @show_path.on_update
    def _toggle_path(_event: object) -> None:
        with server.atomic():
            for handle in motion_paths:
                handle.visible = bool(show_path.value)

    speed = server.gui.add_slider(
        "Playback multiplier",
        min=0.25,
        max=4.0,
        step=0.25,
        initial_value=1.0,
    )
    frame_slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=len(combined.frames) - 1,
        step=1,
        initial_value=0,
    )
    stage_status = server.gui.add_text("Motion stage", initial_value=first_frame.stage_name)
    gripper_status = server.gui.add_text(
        "Gripper", initial_value=f"{first_frame.gripper_percent:.1f}% closed"
    )
    nominal_tray_status = server.gui.add_text(
        "Nominal tray", initial_value=combined.nominal_tray_states[0]
    )
    nominal_cathode_status = server.gui.add_text(
        "Nominal cathode", initial_value=combined.nominal_cathode_states[0]
    )
    offset_tray_status = server.gui.add_text(
        "Offset 1 tray", initial_value=combined.offset_tray_states[0]
    )
    offset_cathode_status = server.gui.add_text(
        "Offset 1 cathode", initial_value=combined.offset_cathode_states[0]
    )
    second_offset_tray_status = None
    second_offset_cathode_status = None
    if second_offset is not None:
        second_offset_tray_status = server.gui.add_text(
            "Offset 2 tray", initial_value=combined.second_offset_tray_states[0]
        )
        second_offset_cathode_status = server.gui.add_text(
            "Offset 2 cathode",
            initial_value=combined.second_offset_cathode_states[0],
        )

    actual_port = server.get_port()
    print(f"All-motion nominal/offset DMP viewer ready: http://localhost:{actual_port}")
    sequence_summary = (
        f"Full sequence: nominal {combined.nominal_frame_count} + return "
        f"{combined.transition_frame_count} + reset {combined.reset_frame_count} + "
        f"offset-1 {combined.offset_frame_count}"
    )
    if second_offset is not None:
        sequence_summary += (
            f" + return {combined.second_transition_frame_count} + reset "
            f"{combined.second_reset_frame_count} + offset-2 "
            f"{len(second_offset.frames)}"
        )
    print(f"{sequence_summary} = {len(combined.frames)} frames.")
    print("DMP program per task: 27 move primitives + 8 discrete events.")
    print(f"Offset marker translation: {np.asarray(args.offset_marker_mm).tolist()} mm")
    print(f"Offset marker-frame yaw: {args.offset_yaw_deg:+.3f} deg")
    if second_offset is not None:
        print(
            "Second offset marker translation: "
            f"{np.asarray(args.second_offset_marker_mm).tolist()} mm"
        )
        print(f"Second offset marker-frame yaw: {args.second_offset_yaw_deg:+.3f} deg")
        print(
            "Second offset empty-gripper release tail: "
            + (
                "recentered to offset-1."
                if args.second_offset_recenter_release_tail
                else "kept at offset-2."
            )
        )
    print(
        "Offset equivalent grasp: local-Y flip "
        f"{'enabled' if args.offset_equivalent_grasp_y_flip else 'disabled'}, "
        f"local-Z shift {args.offset_grasp_shift_z_mm:+.3f} mm"
    )
    print(
        "Nominal fits: tray "
        f"{nominal.tray_fit_rms_m * 1000.0:.6f} mm, cathode initial/final "
        f"{nominal.cathode_initial_fit_rms_m * 1000.0:.6f}/"
        f"{nominal.cathode_final_fit_rms_m * 1000.0:.6f} mm RMS."
    )
    print(
        "Offset fits: tray "
        f"{offset.tray_fit_rms_m * 1000.0:.6f} mm, cathode initial/final "
        f"{offset.cathode_initial_fit_rms_m * 1000.0:.6f}/"
        f"{offset.cathode_final_fit_rms_m * 1000.0:.6f} mm RMS."
    )
    if second_offset is not None:
        print(
            "Second offset fits: tray "
            f"{second_offset.tray_fit_rms_m * 1000.0:.6f} mm, cathode initial/final "
            f"{second_offset.cathode_initial_fit_rms_m * 1000.0:.6f}/"
            f"{second_offset.cathode_final_fit_rms_m * 1000.0:.6f} mm RMS."
        )
    print(f"Visual tray: {resolved_tray_path}")
    print(f"Visual cathode plate: {resolved_cathode_path}")
    print(f"Visual flange adapter: {resolved_adapter_path}")
    print("No RTDE or gripper connection was opened. Press Ctrl+C to stop.")

    frame_index = 0
    try:
        while True:
            if not play.value:
                frame_index = min(int(frame_slider.value), len(combined.frames) - 1)
            frame = combined.frames[frame_index]
            cfg = visualizer._configuration_for_urdf_object(
                ik.urdf,
                frame.q_rad,
                list(DEFAULT_JOINT_ORDER),
            )
            display_t_tcp = display_transform @ ur_pose_to_transform(frame.tcp_pose_ur)
            base_t_adapter = preview.base_t_hande_mount(frame.tcp_pose_ur, args.tcp_offset_ur)
            display_t_adapter = display_transform @ base_t_adapter
            base_t_hande = base_t_adapter @ translation_transform(
                [0.0, 0.0, preview.FLANGE_ADAPTER_STACK_HEIGHT_M]
            )
            display_t_hande = display_transform @ base_t_hande
            display_t_nominal_tray = (
                display_transform @ combined.nominal_tray_transforms[frame_index]
            )
            display_t_nominal_cathode = (
                display_transform @ combined.nominal_cathode_transforms[frame_index]
            )
            display_t_offset_tray = display_transform @ combined.offset_tray_transforms[frame_index]
            display_t_offset_cathode = (
                display_transform @ combined.offset_cathode_transforms[frame_index]
            )
            display_t_second_offset_tray = (
                display_transform @ combined.second_offset_tray_transforms[frame_index]
            )
            display_t_second_offset_cathode = (
                display_transform @ combined.second_offset_cathode_transforms[frame_index]
            )
            finger_x_m = preview.HANDE_FINGER_OFFSET_M - (
                preview.HANDE_FINGER_RANGE_M * frame.gripper_percent / 100.0
            )
            with server.atomic():
                viser_urdf.update_cfg(cfg)
                tcp_handle.position = display_t_tcp[:3, 3]
                tcp_handle.wxyz = Rotation.from_matrix(display_t_tcp[:3, :3]).as_quat()[
                    [3, 0, 1, 2]
                ]
                tcp_marker.position = display_t_tcp[:3, 3]
                adapter_root.position = display_t_adapter[:3, 3]
                adapter_root.wxyz = Rotation.from_matrix(display_t_adapter[:3, :3]).as_quat()[
                    [3, 0, 1, 2]
                ]
                hande_root.position = display_t_hande[:3, 3]
                hande_root.wxyz = Rotation.from_matrix(display_t_hande[:3, :3]).as_quat()[
                    [3, 0, 1, 2]
                ]
                nominal_tray_root.position = display_t_nominal_tray[:3, 3]
                nominal_tray_root.wxyz = Rotation.from_matrix(
                    display_t_nominal_tray[:3, :3]
                ).as_quat()[[3, 0, 1, 2]]
                nominal_cathode_root.position = display_t_nominal_cathode[:3, 3]
                nominal_cathode_root.wxyz = Rotation.from_matrix(
                    display_t_nominal_cathode[:3, :3]
                ).as_quat()[[3, 0, 1, 2]]
                offset_tray_root.position = display_t_offset_tray[:3, 3]
                offset_tray_root.wxyz = Rotation.from_matrix(
                    display_t_offset_tray[:3, :3]
                ).as_quat()[[3, 0, 1, 2]]
                offset_cathode_root.position = display_t_offset_cathode[:3, 3]
                offset_cathode_root.wxyz = Rotation.from_matrix(
                    display_t_offset_cathode[:3, :3]
                ).as_quat()[[3, 0, 1, 2]]
                offset_tray_root.visible = combined.offset_objects_visible[frame_index]
                offset_cathode_root.visible = combined.offset_objects_visible[frame_index]
                if second_offset_tray_root is not None and second_offset_cathode_root is not None:
                    second_offset_tray_root.position = display_t_second_offset_tray[:3, 3]
                    second_offset_tray_root.wxyz = Rotation.from_matrix(
                        display_t_second_offset_tray[:3, :3]
                    ).as_quat()[[3, 0, 1, 2]]
                    second_offset_cathode_root.position = display_t_second_offset_cathode[:3, 3]
                    second_offset_cathode_root.wxyz = Rotation.from_matrix(
                        display_t_second_offset_cathode[:3, :3]
                    ).as_quat()[[3, 0, 1, 2]]
                    second_offset_tray_root.visible = combined.second_offset_objects_visible[
                        frame_index
                    ]
                    second_offset_cathode_root.visible = combined.second_offset_objects_visible[
                        frame_index
                    ]
                left_finger.position = (
                    finger_x_m,
                    0.0,
                    preview.HANDE_COUPLER_HEIGHT_M + preview.HANDE_BODY_HEIGHT_M,
                )
                right_finger.position = (
                    -finger_x_m,
                    0.0,
                    preview.HANDE_COUPLER_HEIGHT_M + preview.HANDE_BODY_HEIGHT_M,
                )
                stage_status.value = frame.stage_name
                gripper_status.value = f"{frame.gripper_percent:.1f}% closed"
                nominal_tray_status.value = combined.nominal_tray_states[frame_index]
                nominal_cathode_status.value = combined.nominal_cathode_states[frame_index]
                offset_tray_status.value = combined.offset_tray_states[frame_index]
                offset_cathode_status.value = combined.offset_cathode_states[frame_index]
                if (
                    second_offset_tray_status is not None
                    and second_offset_cathode_status is not None
                ):
                    second_offset_tray_status.value = combined.second_offset_tray_states[
                        frame_index
                    ]
                    second_offset_cathode_status.value = combined.second_offset_cathode_states[
                        frame_index
                    ]
                if play.value:
                    frame_slider.value = frame_index

            time.sleep(1.0 / (args.playback_hz * float(speed.value)))
            if play.value:
                if frame_index + 1 < len(combined.frames):
                    frame_index += 1
                elif loop.value:
                    frame_index = 0
                else:
                    play.value = False
    except KeyboardInterrupt:
        print("Stopping all-motion nominal/offset DMP viewer.")


def main() -> None:
    args = parse_args()
    motion.validate_args(args)
    weights = load_all_motion_weights(args.dmp_model)
    cartesian_dmp_class, _simple_dmp_root = dmpgen.load_cartesian_dmp_class(args.simple_dmp_root)
    initial_q_rad = np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
    print("Initializing offline PyRoki solver; no RTDE connection will be opened...", flush=True)
    ik = PyrokiRTDEControlAdapter(
        initial_q_rad=initial_q_rad,
        tcp_offset_ur=np.asarray(args.tcp_offset_ur, dtype=float),
    )
    tray_mesh, tray_hole_centers, resolved_tray_path = preview.load_visual_tray_mesh(
        Path(args.visual_tray_stl)
    )
    preview.shorten_lower_tray_handle(
        tray_mesh,
        args.visual_tray_handle_root_y_mm / 1000.0,
        args.visual_tray_handle_scale,
    )
    cathode_mesh, cathode_hole_centers, resolved_cathode_path = (
        preview.load_visual_cathode_plate_mesh(Path(args.visual_cathode_plate_stl))
    )

    nominal_args = copy.deepcopy(args)
    nominal, initial_q_rad, base_t_marker = build_task_rollout(
        nominal_args,
        ik,
        weights,
        cartesian_dmp_class,
        tray_mesh,
        tray_hole_centers,
        cathode_mesh,
        cathode_hole_centers,
    )
    offset_args = copy.deepcopy(args)
    offset_args.assembly_origin_x_mm += float(args.offset_marker_mm[0])
    offset_args.assembly_origin_y_mm += float(args.offset_marker_mm[1])
    offset_args.assembly_origin_z_mm += float(args.offset_marker_mm[2])
    offset_args.assembly_local_yaw_deg += float(args.offset_yaw_deg)
    offset, offset_initial_q, offset_base_t_marker = build_task_rollout(
        offset_args,
        ik,
        weights,
        cartesian_dmp_class,
        tray_mesh,
        tray_hole_centers,
        cathode_mesh,
        cathode_hole_centers,
        equivalent_grasp_y_flip=args.offset_equivalent_grasp_y_flip,
        grasp_shift_z_mm=args.offset_grasp_shift_z_mm,
    )
    second_offset: TaskRollout | None = None
    second_offset_initial_q: np.ndarray | None = None
    second_offset_base_t_marker: np.ndarray | None = None
    if args.second_offset_marker_mm is not None:
        second_offset_args = copy.deepcopy(args)
        second_offset_args.assembly_origin_x_mm += float(args.second_offset_marker_mm[0])
        second_offset_args.assembly_origin_y_mm += float(args.second_offset_marker_mm[1])
        second_offset_args.assembly_origin_z_mm += float(args.second_offset_marker_mm[2])
        second_offset_args.assembly_local_yaw_deg += float(args.second_offset_yaw_deg)
        (
            second_offset,
            second_offset_initial_q,
            second_offset_base_t_marker,
        ) = build_task_rollout(
            second_offset_args,
            ik,
            weights,
            cartesian_dmp_class,
            tray_mesh,
            tray_hole_centers,
            cathode_mesh,
            cathode_hole_centers,
            post_release_reference_events=(
                offset.events if args.second_offset_recenter_release_tail else None
            ),
        )
    print("All requested DMP task rollouts passed strict IK.", flush=True)
    nominal_initial_tcp = np.asarray(
        ik.getForwardKinematics(initial_q_rad.tolist(), args.tcp_offset_ur),
        dtype=float,
    )
    offset_initial_tcp = np.asarray(
        ik.getForwardKinematics(offset_initial_q.tolist(), args.tcp_offset_ur),
        dtype=float,
    )
    np.testing.assert_allclose(offset_initial_tcp, nominal_initial_tcp, atol=1e-6)
    np.testing.assert_allclose(offset_base_t_marker, base_t_marker, atol=1e-12)
    if second_offset is not None:
        assert second_offset_initial_q is not None
        assert second_offset_base_t_marker is not None
        second_offset_initial_tcp = np.asarray(
            ik.getForwardKinematics(second_offset_initial_q.tolist(), args.tcp_offset_ur),
            dtype=float,
        )
        np.testing.assert_allclose(second_offset_initial_tcp, nominal_initial_tcp, atol=1e-6)
        np.testing.assert_allclose(second_offset_base_t_marker, base_t_marker, atol=1e-12)
    print("Equivalent offset start pose and marker frame verified.", flush=True)
    combined = combine_rollouts(nominal, offset, second_offset, initial_q_rad, ik, args)
    print("Combined playback sequence built; initializing viser.", flush=True)
    launch_viewer(
        args,
        ik,
        base_t_marker,
        nominal,
        offset,
        second_offset,
        combined,
        tray_mesh,
        tray_hole_centers,
        resolved_tray_path,
        cathode_mesh,
        cathode_hole_centers,
        resolved_cathode_path,
    )


if __name__ == "__main__":
    main()
