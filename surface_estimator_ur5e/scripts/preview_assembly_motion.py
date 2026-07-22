#!/usr/bin/env python3
"""Animate the real marker-based TI cell assembly program fully offline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import marker_based_motion as motion
from surface_estimator_ur5e.io import DEFAULT_JOINT_ORDER
from surface_estimator_ur5e.pyroki_ik import (
    PyrokiRTDEControlAdapter,
    ur_pose_to_transform,
)
from surface_estimator_ur5e.robot_model import URDFRobotVisualizer
from surface_estimator_ur5e.visualization import _ceiling_mount_display_transform
from visualize_marker_frame import (
    PIN_MATERIAL_NAME,
    add_base_plane,
    add_frame,
    add_marker_square,
    add_ti_assembly_mesh,
    four_pin_feature_frame,
    marker_relative_transform,
    set_camera,
    shorten_lower_tray_handle,
    transform_points,
    translation_transform,
)


EventKind = Literal["move", "gripper", "dwell"]
TrayAction = Literal["attach", "detach"]
DEFAULT_VISUAL_TRAY_STL = Path("assets/ti_tray_short.stl")
DEFAULT_VISUAL_CATHODE_PLATE_STL = Path("assets/CATHODE_PLATE_w_handle.stl")
DEFAULT_VISUAL_TRAY_HANDLE_ROOT_Y_MM = -37.5
DEFAULT_VISUAL_TRAY_HANDLE_SCALE = 0.25
HANDE_MESH_DIR = Path("assets/robotiq_hande_description/meshes")
HANDE_COUPLER_HEIGHT_M = 0.011
HANDE_BODY_HEIGHT_M = 0.099
HANDE_FINGER_OFFSET_M = 0.038
HANDE_FINGER_RANGE_M = 0.030


@dataclass(frozen=True)
class MotionEvent:
    """One command-equivalent event from the real assembly program."""

    kind: EventKind
    name: str
    target_pose_ur: np.ndarray | None = None
    speed_m_s: float | None = None
    gripper_percent: float | None = None
    tray_action: TrayAction | None = None
    dwell_s: float = 0.0


@dataclass(frozen=True)
class AnimationFrame:
    """One solved visualization frame."""

    q_rad: np.ndarray
    tcp_pose_ur: np.ndarray
    stage_name: str
    gripper_percent: float
    tray_attached: bool
    base_t_tray_center: np.ndarray


@dataclass(frozen=True)
class CathodePlateAnimation:
    """Visualization-only poses and fit diagnostics for the second handled part."""

    transforms: list[np.ndarray]
    attach_frame: int
    detach_frame: int
    initial_two_pin_rms_m: float
    initial_support_shift_m: float
    final_four_pin_lateral_rms_m: float
    final_pin_axis_offset_m: float
    attachment_correction_m: float
    attachment_correction_deg: float


def parse_args() -> argparse.Namespace:
    """Extend the real motion CLI with offline playback controls."""

    parser = motion.build_arg_parser()
    parser.description = (
        "Offline PyRoki/viser animation of the real marker_based_motion execution path. "
        "This command never connects to or commands the robot."
    )
    parser.set_defaults(
        align_to_four_pin_frame=True,
        insert_after_pin_approach=True,
        no_set_tcp=True,
        execute=False,
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--visual-tray-stl",
        type=Path,
        default=DEFAULT_VISUAL_TRAY_STL,
        help=(
            "Visualization-only tray mesh. This never replaces --tray-obj in the "
            "real motion geometry calculations."
        ),
    )
    parser.add_argument(
        "--visual-cathode-plate-stl",
        type=Path,
        default=DEFAULT_VISUAL_CATHODE_PLATE_STL,
        help="Visualization-only handled cathode plate used by the two-pin sequence.",
    )
    parser.add_argument(
        "--visual-tray-handle-root-y-mm",
        type=float,
        default=DEFAULT_VISUAL_TRAY_HANDLE_ROOT_Y_MM,
        help="Local Y coordinate where preview-only tray handle shortening starts.",
    )
    parser.add_argument(
        "--visual-tray-handle-scale",
        type=float,
        default=DEFAULT_VISUAL_TRAY_HANDLE_SCALE,
        help=(
            "Preview-only lower-handle length scale about its root "
            "(default: 0.25, or 20 mm to 5 mm)."
        ),
    )
    parser.add_argument("--playback-hz", type=float, default=20.0)
    parser.add_argument(
        "--time-scale",
        type=float,
        default=4.0,
        help="Playback speed relative to the commanded robot speed (default: 4x).",
    )
    parser.add_argument(
        "--trajectory-step-mm",
        type=float,
        default=5.0,
        help="Maximum Cartesian translation spacing between solved IK frames.",
    )
    parser.add_argument(
        "--trajectory-rotation-step-deg",
        type=float,
        default=3.0,
        help="Maximum orientation spacing between solved IK frames.",
    )
    parser.add_argument(
        "--rotation-speed-deg-s",
        type=float,
        default=30.0,
        help="Preview-only angular speed used to time pure-orientation moveL stages.",
    )
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Solve the full offline trajectory and print its summary without opening viser.",
    )
    args = parser.parse_args()

    if args.execute:
        parser.error("This is an offline preview; --execute is intentionally forbidden.")
    if not args.start_from_initial_pose:
        parser.error("Offline full-sequence preview requires --start-from-initial-pose.")
    if args.auto_pin_image_align:
        parser.error(
            "Offline preview cannot perform a live camera search; use saved alignment metadata."
        )
    continuation_flags = (
        args.continue_pin_sequence_from_current,
        args.continue_post_two_pin_release_from_current,
        args.continue_post_two_pin_after_close_from_current,
        args.continue_post_two_pin_tail_from_current,
        args.continue_post_two_pin_tail_next_from_current,
        args.continue_post_two_pin_tail_extra_from_current,
    )
    if any(continuation_flags):
        parser.error("Current-pose continuation modes are not valid for full offline preview.")
    if args.port <= 0 or args.port > 65535:
        parser.error("--port must be between 1 and 65535.")
    for name in (
        "playback_hz",
        "time_scale",
        "trajectory_step_mm",
        "trajectory_rotation_step_deg",
        "rotation_speed_deg_s",
    ):
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 0.0 < args.visual_tray_handle_scale <= 1.0:
        parser.error("--visual-tray-handle-scale must be > 0 and <= 1.")
    return args


def build_actual_motion_events(
    args: argparse.Namespace,
    ik: PyrokiRTDEControlAdapter,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[MotionEvent]]:
    """Build the same target/action order used by ``marker_based_motion.main``."""

    motion.validate_args(args)
    initial_q_rad = np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
    initial_tcp_pose_ur = np.asarray(
        ik.getForwardKinematics(initial_q_rad.tolist(), args.tcp_offset_ur),
        dtype=float,
    )
    _marker_id, base_t_marker_raw, _marker_data = motion.load_marker_transform(args.marker_pose)
    base_t_marker = (
        motion.floor_constrained_marker_transform(base_t_marker_raw)
        if args.marker_frame_mode == "floor"
        else base_t_marker_raw
    )
    base_t_pin = motion.compute_base_t_four_pin_frame(args, base_t_marker)
    marker_target_pose = motion.make_target_pose_ur(
        base_t_marker,
        motion.marker_target_offset(args),
        initial_tcp_pose_ur,
        args.orientation,
    )
    grasp_q_rad = motion.require_ik_solution(
        ik,
        initial_q_rad,
        marker_target_pose,
        "marker grasp",
    )
    safe_plan, _lift_waypoints, _rotation_waypoints, _ik_report = motion.select_safe_alignment_plan(
        ik,
        marker_target_pose,
        grasp_q_rad,
        base_t_pin,
        float(base_t_marker[2, 3]),
        args,
    )

    events: list[MotionEvent] = [
        MotionEvent("gripper", "pre-motion open", gripper_percent=0.0),
        MotionEvent(
            "move",
            "marker grasp moveL",
            marker_target_pose,
            args.speed_m_s,
        ),
        MotionEvent(
            "gripper",
            "marker target close",
            gripper_percent=args.gripper_close_percent,
            tray_action="attach",
        ),
        MotionEvent("dwell", "post-grasp dwell", dwell_s=motion.POST_GRASP_DWELL_S),
        MotionEvent(
            "move",
            "post-grasp lift",
            np.asarray(safe_plan["lifted_current_tcp_pose_ur"], dtype=float),
            args.speed_m_s,
        ),
        MotionEvent(
            "move",
            "four-pin rotation",
            np.asarray(safe_plan["lifted_target_tcp_pose_ur"], dtype=float),
            args.speed_m_s,
        ),
    ]
    if not args.approach_pin_after_rotation:
        return initial_q_rad, initial_tcp_pose_ur, base_t_marker, events

    centering_pose, _centering_current, _centering_target, _centering_vector = (
        motion.pin_centering_target_pose(
            safe_plan["lifted_target_tcp_pose_ur"],
            base_t_pin,
            args,
        )
    )
    approach_pose, _approach_current, _approach_target, _approach_vector = (
        motion.pin_approach_target_pose(
            centering_pose,
            base_t_pin,
            args,
            args.pin_approach_clearance_mm,
        )
    )
    events.extend(
        [
            MotionEvent(
                "move",
                "post-rotation pin centering",
                centering_pose,
                args.speed_m_s,
            ),
            MotionEvent(
                "move",
                "post-rotation pin-axis approach",
                approach_pose,
                args.speed_m_s,
            ),
        ]
    )
    current_pose = approach_pose
    image_correction_pose = motion.print_pin_image_alignment_correction_plan(
        current_pose,
        args,
    )
    if image_correction_pose is not None:
        events.append(
            MotionEvent(
                "move",
                "post-approach image alignment correction",
                image_correction_pose,
                args.speed_m_s,
            )
        )
        current_pose = image_correction_pose
    if not args.insert_after_pin_approach:
        return initial_q_rad, initial_tcp_pose_ur, base_t_marker, events

    insertion_pose, _current_ref, _target_ref, insertion_vector, _a, _b = (
        motion.pin_insertion_target_pose(
            current_pose,
            base_t_pin,
            args,
            args.pin_insertion_target_y_mm,
        )
    )
    insertion_distance_mm = float(np.linalg.norm(insertion_vector) * 1000.0)
    if insertion_distance_mm > args.max_pin_insertion_mm:
        raise RuntimeError(
            f"Final insertion is {insertion_distance_mm:.1f} mm, above "
            f"the {args.max_pin_insertion_mm:.1f} mm safety limit."
        )
    events.append(
        MotionEvent(
            "move",
            "final pin insertion descend",
            insertion_pose,
            args.pin_insertion_speed_m_s,
        )
    )
    if not args.post_insert_release_retreat:
        return initial_q_rad, initial_tcp_pose_ur, base_t_marker, events

    events.append(
        MotionEvent(
            "gripper",
            "post-insert open",
            gripper_percent=0.0,
            tray_action="detach",
        )
    )
    retreat_y_pose, shift_x_pose, _shift_x_mm = motion.post_insert_retreat_target_poses(
        insertion_pose,
        args,
    )
    events.extend(
        [
            MotionEvent(
                "move",
                "post-insert TCP +Y retreat",
                retreat_y_pose,
                args.speed_m_s,
            ),
            MotionEvent(
                "move",
                "post-insert TCP +X adjacent-pin shift",
                shift_x_pose,
                args.speed_m_s,
            ),
        ]
    )
    if not args.post_insert_move_to_two_pin:
        return initial_q_rad, initial_tcp_pose_ur, base_t_marker, events

    two_pin_pose, _adjacent_center, _target_position, _translation = (
        motion.adjacent_two_pin_tcp_target_pose(shift_x_pose, base_t_pin, args)
    )
    events.extend(
        [
            MotionEvent(
                "move",
                "post-insert move TCP to adjacent two-pin center",
                two_pin_pose,
                args.speed_m_s,
            ),
            MotionEvent(
                "gripper",
                "post-insert two-pin close",
                gripper_percent=args.gripper_close_percent,
            ),
            MotionEvent(
                "dwell",
                "post-two-pin close dwell",
                dwell_s=motion.POST_TWO_PIN_CLOSE_DWELL_S,
            ),
        ]
    )
    if not args.post_two_pin_close_retreat:
        return initial_q_rad, initial_tcp_pose_ur, base_t_marker, events

    close_retreat_pose, close_shift_pose = motion.post_two_pin_close_retreat_target_poses(
        two_pin_pose,
        args,
    )
    events.extend(
        [
            MotionEvent(
                "move",
                "post-two-pin-close TCP +Y retreat",
                close_retreat_pose,
                args.speed_m_s,
            ),
            MotionEvent(
                "move",
                "post-two-pin-close TCP -X shift",
                close_shift_pose,
                args.speed_m_s,
            ),
        ]
    )
    post_close_correction = motion.print_post_two_pin_close_image_alignment_plan(
        close_shift_pose,
        args,
    )
    post_close_alignment_will_run = bool(
        args.post_two_pin_close_image_align
        and motion.effective_post_two_pin_close_image_alignment_metadata(args) is not None
    )
    if post_close_correction is not None:
        events.append(
            MotionEvent(
                "move",
                "post-two-pin-close image alignment correction",
                post_close_correction,
                args.speed_m_s,
            )
        )
    if not post_close_alignment_will_run or not args.post_two_pin_aligned_insert_release:
        return initial_q_rad, initial_tcp_pose_ur, base_t_marker, events

    aligned_start = post_close_correction if post_close_correction is not None else close_shift_pose
    (
        aligned_insert_pose,
        release_retreat_pose,
        release_shift_z_pose,
        release_final_xy_pose,
        release_rotate_x_pose,
    ) = motion.post_two_pin_aligned_insert_release_target_poses(aligned_start, args)
    events.extend(
        [
            MotionEvent(
                "move",
                "post-two-pin aligned TCP -Y insertion",
                aligned_insert_pose,
                args.pin_insertion_speed_m_s,
            ),
            MotionEvent(
                "gripper",
                "post-two-pin aligned open",
                gripper_percent=0.0,
            ),
            MotionEvent(
                "move",
                "post-two-pin release TCP +Y retreat",
                release_retreat_pose,
                args.speed_m_s,
            ),
            MotionEvent(
                "move",
                "post-two-pin release TCP Z shift",
                release_shift_z_pose,
                args.speed_m_s,
            ),
            MotionEvent(
                "move",
                "post-two-pin release TCP final X/Y shift",
                release_final_xy_pose,
                args.speed_m_s,
            ),
            MotionEvent(
                "move",
                "post-two-pin release TCP final X rotation",
                release_rotate_x_pose,
                args.speed_m_s,
            ),
        ]
    )
    if args.stop_after_post_two_pin_release_rotation:
        return initial_q_rad, initial_tcp_pose_ur, base_t_marker, events

    after_rotation_targets = motion.post_two_pin_release_after_rotation_target_poses(
        release_rotate_x_pose,
        args,
    )
    for name, pose in after_rotation_targets:
        events.append(MotionEvent("move", name, pose, args.pin_insertion_speed_m_s))
    after_rotation_pose = after_rotation_targets[-1][1]
    events.append(
        MotionEvent(
            "gripper",
            "post-two-pin release final half-close",
            gripper_percent=args.post_two_pin_release_final_close_percent,
        )
    )
    post_close_yz_pose = motion.pose_translated_in_tcp_frame(
        after_rotation_pose,
        np.asarray(
            [
                0.0,
                args.post_two_pin_release_after_rotation_y_mm / 1000.0,
                motion.DEFAULT_POST_TWO_PIN_RELEASE_AFTER_ROTATION_Z_MM / 1000.0,
            ],
            dtype=float,
        ),
    )
    events.append(
        MotionEvent(
            "move",
            "post-two-pin after-close TCP Y/Z shift",
            post_close_yz_pose,
            args.pin_insertion_speed_m_s,
        )
    )
    extra_targets = motion.post_two_pin_after_close_target_poses(post_close_yz_pose, args)
    for name, pose in extra_targets:
        events.append(MotionEvent("move", name, pose, args.pin_insertion_speed_m_s))
    tail_targets = motion.post_two_pin_after_close_tail_target_poses(
        extra_targets[-1][1],
        args,
    )
    for name, pose in tail_targets:
        events.append(MotionEvent("move", name, pose, args.pin_insertion_speed_m_s))
    return initial_q_rad, initial_tcp_pose_ur, base_t_marker, events


def solve_animation_frames(
    args: argparse.Namespace,
    ik: PyrokiRTDEControlAdapter,
    initial_q_rad: np.ndarray,
    initial_tcp_pose_ur: np.ndarray,
    events: list[MotionEvent],
) -> list[AnimationFrame]:
    """Interpolate every real moveL event and solve its continuous joint path."""

    marker_grasp_event = next(event for event in events if event.name == "marker grasp moveL")
    assert marker_grasp_event.target_pose_ur is not None
    pickup_t_tray_center = tray_center_transform(
        marker_grasp_event.target_pose_ur,
        args.tray_center_tcp_z_mm,
    )
    current_q = np.asarray(initial_q_rad, dtype=float)
    current_pose = np.asarray(initial_tcp_pose_ur, dtype=float)
    gripper_percent = 0.0
    tray_attached = False
    base_t_tray_center = pickup_t_tray_center
    frames = [
        AnimationFrame(
            current_q.copy(),
            current_pose.copy(),
            "initial joint pose",
            gripper_percent,
            tray_attached,
            base_t_tray_center.copy(),
        )
    ]

    print(f"Solving {sum(event.kind == 'move' for event in events)} moveL stages with PyRoki...")
    for event_index, event in enumerate(events, start=1):
        if event.kind == "move":
            assert event.target_pose_ur is not None
            assert event.speed_m_s is not None
            samples = interpolate_move(
                current_pose,
                event.target_pose_ur,
                event.speed_m_s,
                args,
            )
            for sample_index, pose in enumerate(samples, start=1):
                if not ik.getInverseKinematicsHasSolution(
                    pose.tolist(),
                    current_q.tolist(),
                    motion.IK_POSITION_TOLERANCE_M,
                    motion.IK_ORIENTATION_TOLERANCE_RAD,
                ):
                    raise RuntimeError(
                        f"PyRoki found no strict IK solution for '{event.name}' "
                        f"sample {sample_index}/{len(samples)}."
                    )
                q_next = np.asarray(
                    ik.getInverseKinematics(
                        pose.tolist(),
                        current_q.tolist(),
                        motion.IK_POSITION_TOLERANCE_M,
                        motion.IK_ORIENTATION_TOLERANCE_RAD,
                    ),
                    dtype=float,
                )
                joint_step_deg = float(np.max(np.abs(np.rad2deg(q_next - current_q))))
                if joint_step_deg > motion.MAX_IK_JOINT_STEP_DEG:
                    raise RuntimeError(
                        f"IK branch jump of {joint_step_deg:.1f} deg during "
                        f"'{event.name}' sample {sample_index}/{len(samples)}."
                    )
                current_q = q_next
                current_pose = np.asarray(pose, dtype=float)
                if tray_attached:
                    base_t_tray_center = tray_center_transform(
                        current_pose,
                        args.tray_center_tcp_z_mm,
                    )
                frames.append(
                    AnimationFrame(
                        current_q.copy(),
                        current_pose.copy(),
                        event.name,
                        gripper_percent,
                        tray_attached,
                        base_t_tray_center.copy(),
                    )
                )
        elif event.kind == "gripper":
            assert event.gripper_percent is not None
            previous_percent = gripper_percent
            pause_frames = max(
                2,
                int(np.ceil(0.5 * args.playback_hz / args.time_scale)),
            )
            for fraction in np.linspace(0.0, 1.0, pause_frames + 1)[1:]:
                animated_percent = (1.0 - fraction) * previous_percent + (
                    fraction * event.gripper_percent
                )
                frames.append(
                    AnimationFrame(
                        current_q.copy(),
                        current_pose.copy(),
                        event.name,
                        float(animated_percent),
                        tray_attached,
                        base_t_tray_center.copy(),
                    )
                )
            gripper_percent = float(event.gripper_percent)
            if event.tray_action == "attach":
                tray_attached = True
                base_t_tray_center = tray_center_transform(
                    current_pose,
                    args.tray_center_tcp_z_mm,
                )
            elif event.tray_action == "detach":
                tray_attached = False
            frames.append(
                AnimationFrame(
                    current_q.copy(),
                    current_pose.copy(),
                    event.name,
                    gripper_percent,
                    tray_attached,
                    base_t_tray_center.copy(),
                )
            )
        elif event.kind == "dwell":
            pause_frames = max(
                1,
                int(np.ceil(event.dwell_s * args.playback_hz / args.time_scale)),
            )
            frames.extend(
                AnimationFrame(
                    current_q.copy(),
                    current_pose.copy(),
                    event.name,
                    gripper_percent,
                    tray_attached,
                    base_t_tray_center.copy(),
                )
                for _ in range(pause_frames)
            )
        else:
            raise AssertionError(f"Unhandled event kind: {event.kind}")
        print(
            f"  [{event_index:02d}/{len(events):02d}] {event.name}: {len(frames)} cumulative frames"
        )
    return frames


def interpolate_move(
    start_pose_ur: np.ndarray,
    target_pose_ur: np.ndarray,
    speed_m_s: float,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    """Sample a moveL target with geometric and approximate timing constraints."""

    start = np.asarray(start_pose_ur, dtype=float)
    target = np.asarray(target_pose_ur, dtype=float)
    translation_distance_m = float(np.linalg.norm(target[:3] - start[:3]))
    rotation_distance_deg = motion.rotation_delta_deg(start, target)
    geometric_segments = max(
        1,
        int(np.ceil(translation_distance_m / (args.trajectory_step_mm / 1000.0))),
        int(np.ceil(rotation_distance_deg / args.trajectory_rotation_step_deg)),
    )
    approximate_duration_s = max(
        translation_distance_m / speed_m_s,
        rotation_distance_deg / args.rotation_speed_deg_s,
    )
    timing_segments = max(
        1,
        int(np.ceil(approximate_duration_s * args.playback_hz / args.time_scale)),
    )
    segments = max(geometric_segments, timing_segments)
    fractions = np.linspace(0.0, 1.0, segments + 1)
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_rotvec(np.vstack([start[3:], target[3:]])),
    )(fractions)
    samples: list[np.ndarray] = []
    for fraction, rotation in zip(fractions[1:], rotations[1:], strict=True):
        pose = np.empty(6, dtype=float)
        pose[:3] = (1.0 - fraction) * start[:3] + fraction * target[:3]
        pose[3:] = rotation.as_rotvec()
        samples.append(pose)
    return samples


def tray_center_transform(tcp_pose_ur: np.ndarray, tcp_z_mm: float) -> np.ndarray:
    """Return the base transform of the animated tray adjustment frame."""

    return ur_pose_to_transform(np.asarray(tcp_pose_ur, dtype=float)) @ translation_transform(
        [0.0, 0.0, tcp_z_mm / 1000.0]
    )


def load_visual_tray_mesh(
    tray_path: Path,
) -> tuple[object, np.ndarray, Path]:
    """Load the short tray in metres and extract its four small-hole centers."""

    import trimesh

    resolved_path = motion.resolve_project_path(tray_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Visual tray mesh not found: {resolved_path}")
    mesh = trimesh.load(resolved_path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected one tray mesh in {resolved_path}.")

    # The supplied STL is authored in millimetres while the rest of the scene is SI.
    if float(np.max(mesh.extents)) > 1.0:
        mesh.apply_scale(0.001)
    if not 0.03 <= float(np.max(mesh.extents)) <= 0.30:
        raise ValueError(f"Unexpected tray size {mesh.extents.tolist()} m after unit conversion.")

    z_min, z_max = (float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2]))
    section_z = z_min + 0.15 * (z_max - z_min)
    section = mesh.section(
        plane_origin=np.asarray([0.0, 0.0, section_z], dtype=float),
        plane_normal=np.asarray([0.0, 0.0, 1.0], dtype=float),
    )
    if section is None:
        raise ValueError(f"Could not section tray mesh {resolved_path} to locate holes.")

    small_hole_xy: list[np.ndarray] = []
    for discrete in section.discrete:
        points = np.asarray(discrete, dtype=float)
        if len(points) < 8:
            continue
        xy = points[:, :2]
        if float(np.linalg.norm(xy[0] - xy[-1])) > 1e-5:
            continue
        center = 0.5 * (np.min(xy, axis=0) + np.max(xy, axis=0))
        radii = np.linalg.norm(xy - center, axis=1)
        radius = float(np.mean(radii))
        circularity_error = float(np.std(radii) / max(radius, 1e-12))
        if 0.0025 <= radius <= 0.0045 and circularity_error <= 0.08:
            small_hole_xy.append(center)

    if len(small_hole_xy) != 4:
        raise ValueError(
            f"Expected four 2.5-4.5 mm-radius tray holes in {resolved_path}, "
            f"found {len(small_hole_xy)}."
        )

    centers_xy = np.asarray(small_hole_xy, dtype=float)
    by_y = np.argsort(centers_xy[:, 1])
    lower = by_y[:2][np.argsort(centers_xy[by_y[:2], 0])]
    upper = by_y[2:][np.argsort(centers_xy[by_y[2:], 0])]
    # Match four_pin_feature_frame(): lower-left, lower-right, upper-right, upper-left.
    ordered_indices = np.asarray([lower[0], lower[1], upper[1], upper[0]], dtype=int)
    hole_centers = np.column_stack(
        (
            centers_xy[ordered_indices],
            np.full(4, 0.5 * (z_min + z_max), dtype=float),
        )
    )
    mesh.visual.face_colors = np.tile(
        np.asarray([205, 230, 145, 235], dtype=np.uint8),
        (len(mesh.faces), 1),
    )
    return mesh, hole_centers, resolved_path


def load_visual_cathode_plate_mesh(
    plate_path: Path,
) -> tuple[object, np.ndarray, Path]:
    """Load the handled cathode plate and its 40-by-60 mm four-hole pattern."""

    mesh, hole_centers, resolved_path = load_visual_tray_mesh(plate_path)
    bore_midpoints_m: list[float] = []
    for hole_xy in hole_centers[:, :2]:
        radial_distance_m = np.linalg.norm(mesh.vertices[:, :2] - hole_xy, axis=1)
        bore_vertices = mesh.vertices[(radial_distance_m >= 0.0031) & (radial_distance_m <= 0.0035)]
        if len(bore_vertices) < 20:
            raise ValueError(f"Could not extract the large-hole bore depth from {resolved_path}.")
        bore_midpoints_m.append(
            0.5 * (float(np.min(bore_vertices[:, 2])) + float(np.max(bore_vertices[:, 2])))
        )
    if float(np.ptp(bore_midpoints_m)) > 0.0002:
        raise ValueError(f"Cathode plate large-hole bore depths disagree in {resolved_path}.")
    hole_centers[:, 2] = float(np.mean(bore_midpoints_m))
    mesh.visual.face_colors = np.tile(
        np.asarray([190, 205, 225, 245], dtype=np.uint8),
        (len(mesh.faces), 1),
    )
    return mesh, hole_centers, resolved_path


def fit_rigid_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit target_T_source and return its point-wise RMS residual."""

    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape or source.shape != (4, 3):
        raise ValueError("Four corresponding 3D points are required for tray alignment.")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    fitted = transform_points(transform, source)
    rms_m = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
    return transform, rms_m


def visual_tray_transforms(
    args: argparse.Namespace,
    base_t_marker: np.ndarray,
    frames: list[AnimationFrame],
    tray_vertices: np.ndarray,
    tray_hole_centers: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, float, float, float]:
    """Fix the grasp attachment while keeping tray holes coaxial with holder pins."""

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
    _assembly_t_pin, selected_pin_centers, _adjacent = four_pin_feature_frame(
        motion.resolve_project_path(args.assembly_obj)
    )
    base_t_assembly = base_t_marker @ marker_t_assembly
    base_pin_centers = transform_points(base_t_assembly, selected_pin_centers)
    base_t_tray_pin_centered, fit_rms_m = fit_rigid_transform(
        tray_hole_centers,
        base_pin_centers,
    )
    if fit_rms_m > 0.001:
        raise ValueError(
            f"Short-tray holes do not match the holder pin pattern "
            f"(RMS residual {fit_rms_m * 1000.0:.3f} mm)."
        )

    insertion_frame = next(
        (frame for frame in reversed(frames) if frame.stage_name == "final pin insertion descend"),
        None,
    )
    grasp_frame = next(
        (frame for frame in reversed(frames) if frame.stage_name == "marker grasp moveL"),
        None,
    )
    if insertion_frame is None or grasp_frame is None:
        raise ValueError("Full preview must contain both tray grasp and final insertion stages.")

    base_t_tcp_inserted = ur_pose_to_transform(insertion_frame.tcp_pose_ur)
    base_t_tcp_grasp = ur_pose_to_transform(grasp_frame.tcp_pose_ur)

    # The four circular holes constrain lateral placement and orientation but allow
    # travel along their common pin axis. Use that remaining DOF to put the bottom
    # of the waiting tray exactly on the marker plane. The robot path, holder pose,
    # and rigid TCP-to-tray attachment all remain unchanged during playback.
    tcp_t_tray_pin_centered = np.linalg.inv(base_t_tcp_inserted) @ base_t_tray_pin_centered
    base_t_tray_waiting_pin_centered = base_t_tcp_grasp @ tcp_t_tray_pin_centered
    waiting_vertices = transform_points(
        base_t_tray_waiting_pin_centered,
        tray_vertices,
    )
    waiting_bottom_z_m = float(np.max(waiting_vertices[:, 2]))
    tray_axis_base_at_grasp = base_t_tray_waiting_pin_centered[:3, 2]
    axis_base_z = float(tray_axis_base_at_grasp[2])
    if abs(axis_base_z) < 0.1:
        raise ValueError(
            "Cannot place the initial tray on the marker plane by sliding along "
            "the pin axis because that axis is nearly horizontal at grasp."
        )
    axial_shift_m = (float(base_t_marker[2, 3]) - waiting_bottom_z_m) / axis_base_z
    base_t_tray_inserted = base_t_tray_pin_centered @ translation_transform(
        [0.0, 0.0, axial_shift_m]
    )
    tcp_t_tray = np.linalg.inv(base_t_tcp_inserted) @ base_t_tray_inserted
    base_t_tray_waiting = base_t_tcp_grasp @ tcp_t_tray
    adjusted_waiting_vertices = transform_points(base_t_tray_waiting, tray_vertices)
    marker_plane_error_m = float(np.max(adjusted_waiting_vertices[:, 2]) - base_t_marker[2, 3])
    detach_index = next(
        (
            index
            for index, frame in enumerate(frames)
            if frame.stage_name == "post-insert open" and not frame.tray_attached
        ),
        len(frames),
    )

    transforms: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if frame.tray_attached:
            transforms.append(ur_pose_to_transform(frame.tcp_pose_ur) @ tcp_t_tray)
        elif index < detach_index:
            transforms.append(base_t_tray_waiting.copy())
        else:
            transforms.append(base_t_tray_inserted.copy())
    return (
        transforms,
        base_pin_centers,
        fit_rms_m,
        axial_shift_m,
        marker_plane_error_m,
    )


def interpolate_transform(
    start: np.ndarray,
    target: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate translation and rotation between two rigid transforms."""

    alpha = float(np.clip(fraction, 0.0, 1.0))
    transform = np.eye(4)
    transform[:3, 3] = (1.0 - alpha) * np.asarray(start[:3, 3], dtype=float) + alpha * np.asarray(
        target[:3, 3], dtype=float
    )
    rotations = Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]], axis=0))
    transform[:3, :3] = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    return transform


def holder_pin_support_offset_m(
    holder_path: Path,
    pin_centers: np.ndarray,
    pin_axis: np.ndarray,
) -> float:
    """Return the orange-pin endpoint that meets the holder support plate."""

    import trimesh

    loaded = trimesh.load(holder_path, process=False)
    if not isinstance(loaded, trimesh.Scene):
        raise ValueError(f"Expected holder OBJ scene in {holder_path}.")
    pin_mesh = loaded.geometry.get(PIN_MATERIAL_NAME)
    if pin_mesh is None:
        raise ValueError(
            f"Could not find orange pin material {PIN_MATERIAL_NAME} in {holder_path}."
        )
    axis = np.asarray(pin_axis, dtype=float)
    axis /= np.linalg.norm(axis)
    support_offsets: list[float] = []
    components = pin_mesh.split(only_watertight=False)
    for pin_center in np.asarray(pin_centers, dtype=float):
        matching_vertices = [
            component.vertices
            for component in components
            if np.linalg.norm(0.5 * (component.bounds[0] + component.bounds[1]) - pin_center)
            <= 0.001
        ]
        if not matching_vertices:
            raise ValueError(f"Could not recover orange pin geometry near {pin_center.tolist()}.")
        vertices = np.vstack(matching_vertices)
        support_offsets.append(float(np.min((vertices - pin_center) @ axis)))
    if float(np.ptp(support_offsets)) > 0.0002:
        raise ValueError("Adjacent orange pin support endpoints are inconsistent.")
    return float(np.mean(support_offsets))


def visual_cathode_plate_animation(
    args: argparse.Namespace,
    base_t_marker: np.ndarray,
    frames: list[AnimationFrame],
    plate_vertices: np.ndarray,
    plate_hole_centers: np.ndarray,
) -> CathodePlateAnimation:
    """Animate the handled plate from adjacent two pins into the four-pin datum."""

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
    _assembly_t_pin, selected_pin_centers, adjacent_pin_centers = four_pin_feature_frame(
        motion.resolve_project_path(args.assembly_obj)
    )
    base_t_assembly = base_t_marker @ marker_t_assembly
    base_four_pin_centers = transform_points(base_t_assembly, selected_pin_centers)
    base_adjacent_pin_centers = transform_points(base_t_assembly, adjacent_pin_centers)

    base_t_plate_four_pin_centered, pattern_rms_m = fit_rigid_transform(
        plate_hole_centers,
        base_four_pin_centers,
    )
    if pattern_rms_m > 0.001:
        raise ValueError(
            "Cathode plate four-hole pattern does not match the holder four pins "
            f"(RMS residual {pattern_rms_m * 1000.0:.3f} mm)."
        )

    # The handle projects from the negative local-X side of the STL. Its two
    # neighboring large holes are the initial support pair on the adjacent pins.
    handle_side_indices = np.argsort(plate_hole_centers[:, 0])[:2]
    handle_side_indices = handle_side_indices[
        np.argsort(plate_hole_centers[handle_side_indices, 1])
    ]
    handle_side_holes = plate_hole_centers[handle_side_indices]
    base_t_plate_initial = base_t_plate_four_pin_centered.copy()
    plate_bottom_from_hole_plane_m = float(
        np.min(plate_vertices[:, 2]) - np.mean(plate_hole_centers[:, 2])
    )
    pin_support_from_center_m = holder_pin_support_offset_m(
        motion.resolve_project_path(args.assembly_obj),
        adjacent_pin_centers,
        base_t_assembly[:3, :3].T @ base_t_plate_initial[:3, 2],
    )
    initial_support_shift_m = pin_support_from_center_m - plate_bottom_from_hole_plane_m
    base_t_plate_initial[:3, 3] = (
        np.mean(base_adjacent_pin_centers, axis=0)
        - base_t_plate_initial[:3, :3] @ np.mean(handle_side_holes, axis=0)
        + base_t_plate_initial[:3, 2] * initial_support_shift_m
    )
    initial_handle_centers = transform_points(base_t_plate_initial, handle_side_holes)
    initial_two_pin_residuals = initial_handle_centers - base_adjacent_pin_centers
    initial_two_pin_lateral_residuals = initial_two_pin_residuals - np.outer(
        initial_two_pin_residuals @ base_t_plate_initial[:3, 2],
        base_t_plate_initial[:3, 2],
    )
    initial_two_pin_rms_m = float(
        np.sqrt(np.mean(np.sum(initial_two_pin_lateral_residuals**2, axis=1)))
    )
    if initial_two_pin_rms_m > 0.001:
        raise ValueError(
            "Cathode plate handle-side holes do not match the adjacent two pins "
            f"(RMS residual {initial_two_pin_rms_m * 1000.0:.3f} mm)."
        )

    attach_frame = max(
        index
        for index, frame in enumerate(frames)
        if frame.stage_name == "post-insert two-pin close"
    )
    detach_frame = max(
        index
        for index, frame in enumerate(frames)
        if frame.stage_name == "post-two-pin aligned open"
    )
    blend_start_frame = min(
        index
        for index, frame in enumerate(frames)
        if frame.stage_name == "post-two-pin-close TCP -X shift"
    )
    blend_end_frame = max(
        index
        for index, frame in enumerate(frames)
        if frame.stage_name == "post-two-pin-close image alignment correction"
    )

    base_t_tcp_attach = ur_pose_to_transform(frames[attach_frame].tcp_pose_ur)
    base_t_tcp_detach = ur_pose_to_transform(frames[detach_frame].tcp_pose_ur)
    tcp_t_plate_initial = np.linalg.inv(base_t_tcp_attach) @ base_t_plate_initial

    # Preserve the real insertion depth. The actual two-pin sequence deliberately
    # retreats 30 mm and reinserts 17 mm, leaving the plate 13 mm along the pin axis.
    rigid_release = base_t_tcp_detach @ tcp_t_plate_initial
    rigid_release_holes = transform_points(rigid_release, plate_hole_centers)
    pin_axis = base_t_plate_four_pin_centered[:3, 2]
    rigid_release_residuals = rigid_release_holes - base_four_pin_centers
    final_pin_axis_offset_m = float(np.mean(rigid_release_residuals @ pin_axis))
    base_t_plate_final = base_t_plate_four_pin_centered @ translation_transform(
        [0.0, 0.0, final_pin_axis_offset_m]
    )
    tcp_t_plate_final = np.linalg.inv(base_t_tcp_detach) @ base_t_plate_final

    attachment_correction_m = float(
        np.linalg.norm(tcp_t_plate_final[:3, 3] - tcp_t_plate_initial[:3, 3])
    )
    attachment_correction_deg = float(
        np.rad2deg(
            Rotation.from_matrix(
                tcp_t_plate_final[:3, :3] @ tcp_t_plate_initial[:3, :3].T
            ).magnitude()
        )
    )

    transforms: list[np.ndarray] = []
    for index, frame in enumerate(frames):
        if index <= attach_frame:
            transforms.append(base_t_plate_initial.copy())
            continue
        if index >= detach_frame:
            transforms.append(base_t_plate_final.copy())
            continue
        if index < blend_start_frame:
            tcp_t_plate = tcp_t_plate_initial
        elif index >= blend_end_frame:
            tcp_t_plate = tcp_t_plate_final
        else:
            blend_fraction = (index - blend_start_frame) / max(
                1,
                blend_end_frame - blend_start_frame,
            )
            tcp_t_plate = interpolate_transform(
                tcp_t_plate_initial,
                tcp_t_plate_final,
                blend_fraction,
            )
        transforms.append(ur_pose_to_transform(frame.tcp_pose_ur) @ tcp_t_plate)

    final_hole_centers = transform_points(base_t_plate_final, plate_hole_centers)
    final_residuals = final_hole_centers - base_four_pin_centers
    final_lateral_residuals = final_residuals - np.outer(
        final_residuals @ pin_axis,
        pin_axis,
    )
    final_four_pin_lateral_rms_m = float(
        np.sqrt(np.mean(np.sum(final_lateral_residuals**2, axis=1)))
    )
    return CathodePlateAnimation(
        transforms=transforms,
        attach_frame=attach_frame,
        detach_frame=detach_frame,
        initial_two_pin_rms_m=initial_two_pin_rms_m,
        initial_support_shift_m=initial_support_shift_m,
        final_four_pin_lateral_rms_m=final_four_pin_lateral_rms_m,
        final_pin_axis_offset_m=final_pin_axis_offset_m,
        attachment_correction_m=attachment_correction_m,
        attachment_correction_deg=attachment_correction_deg,
    )


def base_t_hande_mount(
    tcp_pose_ur: np.ndarray,
    tcp_offset_ur: list[float],
) -> np.ndarray:
    """Return the real Hand-E mesh frame rooted at tool0 and aimed toward TCP."""

    base_t_tcp = ur_pose_to_transform(np.asarray(tcp_pose_ur, dtype=float))
    tool0_t_tcp = ur_pose_to_transform(np.asarray(tcp_offset_ur, dtype=float))
    base_t_tool0 = base_t_tcp @ np.linalg.inv(tool0_t_tcp)
    direction = base_t_tcp[:3, 3] - base_t_tool0[:3, 3]
    distance = float(np.linalg.norm(direction))
    if distance < 1e-6:
        return base_t_tool0
    z_axis = direction / distance
    x_axis = base_t_tool0[:3, 0] - np.dot(base_t_tool0[:3, 0], z_axis) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = base_t_tcp[:3, 0] - np.dot(base_t_tcp[:3, 0], z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    transform = base_t_tool0.copy()
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    return transform


def load_preview_hande_mesh(
    mesh_path: Path,
    maximum_sample_vertices: int,
    color: tuple[int, int, int, int],
) -> object:
    """Create a light, deterministic envelope of a detailed Hand-E visual mesh."""

    import trimesh

    detailed = trimesh.load(mesh_path, force="mesh", process=True)
    if not isinstance(detailed, trimesh.Trimesh):
        raise ValueError(f"Expected one Hand-E mesh in {mesh_path}.")
    vertex_count = len(detailed.vertices)
    sampled_indices = np.linspace(
        0,
        vertex_count - 1,
        min(maximum_sample_vertices, vertex_count),
        dtype=int,
    )
    extrema_indices = np.concatenate(
        [
            np.argmin(detailed.vertices, axis=0),
            np.argmax(detailed.vertices, axis=0),
        ]
    )
    sampled_points = detailed.vertices[
        np.unique(np.concatenate([sampled_indices, extrema_indices]))
    ]
    envelope = trimesh.Trimesh(vertices=sampled_points, faces=[]).convex_hull
    envelope.visual.face_colors = np.tile(
        np.asarray(color, dtype=np.uint8),
        (len(envelope.faces), 1),
    )
    return envelope


def launch_visualization(
    args: argparse.Namespace,
    ik: PyrokiRTDEControlAdapter,
    base_t_marker: np.ndarray,
    frames: list[AnimationFrame],
) -> None:
    """Animate solved UR5e joints, TCP, gripper state, and the picked tray in viser."""

    import viser
    from viser.extras import ViserUrdf

    display_transform = _ceiling_mount_display_transform()
    server = viser.ViserServer(port=args.port)
    scene = server.scene
    add_base_plane(server, display_transform)
    add_frame(server, "/frames/robot_base", display_transform, axes_length=0.05)
    display_t_marker = display_transform @ base_t_marker
    add_frame(server, "/frames/marker_floor", display_t_marker, axes_length=0.025)
    add_marker_square(server, "/marker/floor_square", display_t_marker, 0.0254)

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
    assembly_points = add_ti_assembly_mesh(
        server,
        Path(args.assembly_obj),
        display_t_marker @ marker_t_assembly,
        frame_scale=0.30,
        show_bounds=False,
    )

    visualizer = URDFRobotVisualizer()
    root_transform = visualizer.root_transform_for_robot_base(display_transform)
    root_wxyz = Rotation.from_matrix(root_transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    scene.add_frame(
        "/animated_robot",
        position=root_transform[:3, 3],
        wxyz=root_wxyz,
        show_axes=False,
    )
    viser_urdf = ViserUrdf(
        server,
        ik.urdf,
        root_node_name="/animated_robot",
    )

    first_base_t_tcp = ur_pose_to_transform(frames[0].tcp_pose_ur)
    first_display_t_tcp = display_transform @ first_base_t_tcp
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

    first_display_t_hande = display_transform @ base_t_hande_mount(
        frames[0].tcp_pose_ur,
        args.tcp_offset_ur,
    )
    hande_root = scene.add_frame(
        "/animation/hande",
        position=first_display_t_hande[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_hande[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    hande_mesh_dir = motion.resolve_project_path(HANDE_MESH_DIR)
    coupler_mesh = load_preview_hande_mesh(
        hande_mesh_dir / "io_coupler.obj",
        600,
        (150, 155, 160, 255),
    )
    body_mesh = load_preview_hande_mesh(
        hande_mesh_dir / "hande.obj",
        1200,
        (45, 48, 52, 255),
    )
    finger_mesh = load_preview_hande_mesh(
        hande_mesh_dir / "finger.obj",
        500,
        (70, 72, 70, 255),
    )
    scene.add_mesh_trimesh(
        "/animation/hande/io_coupler",
        coupler_mesh,
    )
    scene.add_mesh_trimesh(
        "/animation/hande/body",
        body_mesh,
        position=(0.0, 0.0, HANDE_COUPLER_HEIGHT_M),
    )
    left_finger = scene.add_frame(
        "/animation/hande/left_finger",
        position=(
            HANDE_FINGER_OFFSET_M,
            0.0,
            HANDE_COUPLER_HEIGHT_M + HANDE_BODY_HEIGHT_M,
        ),
        show_axes=False,
    )
    scene.add_mesh_trimesh(
        "/animation/hande/left_finger/mesh",
        finger_mesh,
    )
    right_finger = scene.add_frame(
        "/animation/hande/right_finger",
        position=(
            -HANDE_FINGER_OFFSET_M,
            0.0,
            HANDE_COUPLER_HEIGHT_M + HANDE_BODY_HEIGHT_M,
        ),
        wxyz=(0.0, 0.0, 0.0, 1.0),
        show_axes=False,
    )
    scene.add_mesh_trimesh(
        "/animation/hande/right_finger/mesh",
        finger_mesh,
    )

    tray_mesh, tray_hole_centers, resolved_tray_path = load_visual_tray_mesh(
        Path(args.visual_tray_stl)
    )
    tray_handle_y_before_m = float(np.min(np.asarray(tray_mesh.vertices)[:, 1]))
    tray_handle_root_y_m = args.visual_tray_handle_root_y_mm / 1000.0
    shorten_lower_tray_handle(
        tray_mesh,
        tray_handle_root_y_m,
        args.visual_tray_handle_scale,
    )
    tray_handle_y_after_m = float(np.min(np.asarray(tray_mesh.vertices)[:, 1]))
    (
        tray_transforms,
        _base_pin_centers,
        fit_rms_m,
        tray_axial_shift_m,
        marker_plane_error_m,
    ) = visual_tray_transforms(
        args,
        base_t_marker,
        frames,
        np.asarray(tray_mesh.vertices, dtype=float),
        tray_hole_centers,
    )
    first_display_t_tray = display_transform @ tray_transforms[0]
    tray_root = scene.add_frame(
        "/animated_tray",
        position=first_display_t_tray[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_tray[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    scene.add_mesh_trimesh("/animated_tray/mesh", tray_mesh)
    for index, hole_center in enumerate(tray_hole_centers):
        scene.add_icosphere(
            f"/animated_tray/hole_centers/{index}",
            position=hole_center,
            radius=0.0015,
            color=(25, 170, 255),
        )

    (
        cathode_plate_mesh,
        cathode_plate_hole_centers,
        resolved_cathode_plate_path,
    ) = load_visual_cathode_plate_mesh(Path(args.visual_cathode_plate_stl))
    cathode_plate_animation = visual_cathode_plate_animation(
        args,
        base_t_marker,
        frames,
        np.asarray(cathode_plate_mesh.vertices, dtype=float),
        cathode_plate_hole_centers,
    )
    first_display_t_cathode_plate = display_transform @ cathode_plate_animation.transforms[0]
    cathode_plate_root = scene.add_frame(
        "/animated_cathode_plate",
        position=first_display_t_cathode_plate[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_cathode_plate[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    scene.add_mesh_trimesh(
        "/animated_cathode_plate/mesh",
        cathode_plate_mesh,
    )
    cathode_handle_side_indices = set(np.argsort(cathode_plate_hole_centers[:, 0])[:2].tolist())
    for index, hole_center in enumerate(cathode_plate_hole_centers):
        scene.add_icosphere(
            f"/animated_cathode_plate/hole_centers/{index}",
            position=hole_center,
            radius=0.0015,
            color=(255, 145, 35) if index in cathode_handle_side_indices else (180, 60, 235),
        )

    add_solved_paths(scene, display_transform, frames)
    tcp_path_points = transform_points(
        display_transform,
        np.asarray([frame.tcp_pose_ur[:3] for frame in frames], dtype=float),
    )
    tray_points = np.asarray(
        [(display_transform @ base_t_tray)[:3, 3] for base_t_tray in tray_transforms],
        dtype=float,
    )
    cathode_plate_points = np.asarray(
        [
            (display_transform @ base_t_plate)[:3, 3]
            for base_t_plate in cathode_plate_animation.transforms
        ],
        dtype=float,
    )
    set_camera(
        server,
        np.vstack(
            [
                assembly_points,
                tcp_path_points,
                tray_points,
                cathode_plate_points,
            ]
        ),
    )

    play = server.gui.add_checkbox("Play", initial_value=True)
    loop = server.gui.add_checkbox("Loop", initial_value=not args.no_loop)
    speed = server.gui.add_slider(
        "Playback multiplier",
        min=0.25,
        max=4.0,
        step=0.25,
        initial_value=1.0,
    )
    slider = server.gui.add_slider(
        "Frame",
        min=0,
        max=len(frames) - 1,
        step=1,
        initial_value=0,
    )
    status = server.gui.add_text("Motion stage", initial_value=frames[0].stage_name)
    gripper_status = server.gui.add_text("Gripper", initial_value="0.0% closed")
    tray_status = server.gui.add_text("Tray", initial_value="waiting at marker")
    cathode_plate_status = server.gui.add_text(
        "Cathode plate",
        initial_value="seated on adjacent two pins",
    )

    actual_port = server.get_port()
    print(f"Viser animation ready: http://localhost:{actual_port}")
    print(
        f"Solved {len(frames)} frames across "
        f"{len({frame.stage_name for frame in frames})} command stages."
    )
    print(f"Visual tray: {resolved_tray_path}")
    print(
        "Visual tray lower handle: "
        f"{(tray_handle_root_y_m - tray_handle_y_before_m) * 1000.0:.1f} mm -> "
        f"{(tray_handle_root_y_m - tray_handle_y_after_m) * 1000.0:.1f} mm "
        "(preview mesh only)."
    )
    print(f"Tray hole-to-holder pin lateral fit: {fit_rms_m * 1000.0:.6f} mm RMS across four axes.")
    print(
        "Tray pin-axis placement shift: "
        f"{tray_axial_shift_m * 1000.0:+.3f} mm; initial bottom-to-marker error: "
        f"{marker_plane_error_m * 1000.0:+.6f} mm."
    )
    print(f"Visual cathode plate: {resolved_cathode_plate_path}")
    print(
        "Cathode handle-hole to adjacent two-pin lateral fit: "
        f"{cathode_plate_animation.initial_two_pin_rms_m * 1000.0:.6f} mm RMS; "
        "plate seated on holder support "
        f"({cathode_plate_animation.initial_support_shift_m * 1000.0:+.3f} mm axis shift)."
    )
    print(
        "Cathode four-hole to four-pin lateral fit after insertion: "
        f"{cathode_plate_animation.final_four_pin_lateral_rms_m * 1000.0:.6f} mm RMS; "
        f"pin-axis depth {cathode_plate_animation.final_pin_axis_offset_m * 1000.0:+.3f} mm."
    )
    print(
        "Cathode visual grip correction during saved image-alignment stage: "
        f"{cathode_plate_animation.attachment_correction_m * 1000.0:.3f} mm, "
        f"{cathode_plate_animation.attachment_correction_deg:.3f} deg."
    )
    print("No RTDE or gripper connection was opened. Press Ctrl+C to stop.")

    frame_index = 0
    try:
        while True:
            if not play.value:
                frame_index = int(slider.value)
            frame = frames[frame_index]
            cfg = visualizer._configuration_for_urdf_object(
                ik.urdf,
                frame.q_rad,
                list(DEFAULT_JOINT_ORDER),
            )
            display_t_tcp = display_transform @ ur_pose_to_transform(frame.tcp_pose_ur)
            display_t_hande = display_transform @ base_t_hande_mount(
                frame.tcp_pose_ur,
                args.tcp_offset_ur,
            )
            display_t_tray = display_transform @ tray_transforms[frame_index]
            display_t_cathode_plate = (
                display_transform @ cathode_plate_animation.transforms[frame_index]
            )
            tcp_wxyz = Rotation.from_matrix(display_t_tcp[:3, :3]).as_quat()[[3, 0, 1, 2]]
            hande_wxyz = Rotation.from_matrix(display_t_hande[:3, :3]).as_quat()[[3, 0, 1, 2]]
            tray_wxyz = Rotation.from_matrix(display_t_tray[:3, :3]).as_quat()[[3, 0, 1, 2]]
            cathode_plate_wxyz = Rotation.from_matrix(display_t_cathode_plate[:3, :3]).as_quat()[
                [3, 0, 1, 2]
            ]
            finger_slide_m = HANDE_FINGER_RANGE_M * frame.gripper_percent / 100.0
            finger_x_m = HANDE_FINGER_OFFSET_M - finger_slide_m
            with server.atomic():
                viser_urdf.update_cfg(cfg)
                tcp_handle.position = display_t_tcp[:3, 3]
                tcp_handle.wxyz = tcp_wxyz
                tcp_marker.position = display_t_tcp[:3, 3]
                hande_root.position = display_t_hande[:3, 3]
                hande_root.wxyz = hande_wxyz
                tray_root.position = display_t_tray[:3, 3]
                tray_root.wxyz = tray_wxyz
                cathode_plate_root.position = display_t_cathode_plate[:3, 3]
                cathode_plate_root.wxyz = cathode_plate_wxyz
                left_finger.position = (
                    finger_x_m,
                    0.0,
                    HANDE_COUPLER_HEIGHT_M + HANDE_BODY_HEIGHT_M,
                )
                right_finger.position = (
                    -finger_x_m,
                    0.0,
                    HANDE_COUPLER_HEIGHT_M + HANDE_BODY_HEIGHT_M,
                )
                status.value = frame.stage_name
                gripper_status.value = f"{frame.gripper_percent:.1f}% closed"
                tray_status.value = "attached to TCP" if frame.tray_attached else "fixed in scene"
                if frame_index <= cathode_plate_animation.attach_frame:
                    cathode_plate_status.value = "seated on adjacent two pins"
                elif frame_index < cathode_plate_animation.detach_frame:
                    cathode_plate_status.value = "attached to TCP"
                else:
                    cathode_plate_status.value = "seated on four pins"
                if play.value:
                    slider.value = frame_index

            time.sleep(1.0 / (args.playback_hz * float(speed.value)))
            if play.value:
                if frame_index + 1 < len(frames):
                    frame_index += 1
                elif loop.value:
                    frame_index = 0
                else:
                    play.value = False
    except KeyboardInterrupt:
        print("Stopping offline assembly visualization.")


def add_solved_paths(
    scene: object,
    display_transform: np.ndarray,
    frames: list[AnimationFrame],
) -> None:
    """Draw one colored line strip for each command stage."""

    palette = (
        (40, 120, 250),
        (50, 200, 100),
        (245, 145, 35),
        (170, 70, 220),
        (225, 60, 120),
        (30, 180, 190),
        (90, 90, 90),
    )
    ordered_names: list[str] = []
    by_stage: dict[str, list[np.ndarray]] = {}
    for frame in frames:
        if frame.stage_name not in by_stage:
            ordered_names.append(frame.stage_name)
            by_stage[frame.stage_name] = []
        by_stage[frame.stage_name].append(frame.tcp_pose_ur[:3])
    for index, name in enumerate(ordered_names):
        points = np.asarray(by_stage[name], dtype=float)
        if len(points) < 2 or np.max(np.linalg.norm(points - points[0], axis=1)) < 1e-8:
            continue
        display_points = transform_points(display_transform, points)
        color = palette[index % len(palette)]
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
        scene.add_line_segments(
            f"/solved_motion/{index:02d}_{safe_name}",
            points=np.stack([display_points[:-1], display_points[1:]], axis=1),
            colors=np.tile(np.asarray(color, dtype=np.uint8), (len(points) - 1, 2, 1)),
            line_width=3.0,
        )


def print_summary(events: list[MotionEvent], frames: list[AnimationFrame]) -> None:
    """Print a compact audit of the real commands represented in the preview."""

    move_count = sum(event.kind == "move" for event in events)
    gripper_count = sum(event.kind == "gripper" for event in events)
    maximum_joint_step_deg = max(
        float(np.max(np.abs(np.rad2deg(next_frame.q_rad - frame.q_rad))))
        for frame, next_frame in zip(frames[:-1], frames[1:], strict=True)
    )
    print("Offline assembly preview summary:")
    print(f"  command events: {len(events)}")
    print(f"  moveL stages: {move_count}")
    print(f"  gripper events: {gripper_count}")
    print(f"  solved animation frames: {len(frames)}")
    print(f"  maximum solved joint step: {maximum_joint_step_deg:.3f} deg")
    print(f"  first stage: {events[0].name}")
    print(f"  final stage: {events[-1].name}")


def main() -> None:
    args = parse_args()
    initial_q_rad = np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
    print("Initializing offline PyRoki UR5e solver (no robot connection)...", flush=True)
    ik = PyrokiRTDEControlAdapter(
        initial_q_rad=initial_q_rad,
        tcp_offset_ur=np.asarray(args.tcp_offset_ur, dtype=float),
    )
    initial_q_rad, initial_tcp_pose_ur, base_t_marker, events = build_actual_motion_events(
        args,
        ik,
    )
    frames = solve_animation_frames(
        args,
        ik,
        initial_q_rad,
        initial_tcp_pose_ur,
        events,
    )
    print_summary(events, frames)
    if not args.plan_only:
        launch_visualization(args, ik, base_t_marker, frames)


if __name__ == "__main__":
    main()
