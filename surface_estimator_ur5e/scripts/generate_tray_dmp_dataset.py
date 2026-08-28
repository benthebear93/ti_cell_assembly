#!/usr/bin/env python3
"""Generate kinematically validated tray pick-and-insert DMP demonstrations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import quaternion
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation, Slerp

import marker_based_motion as motion
import preview_assembly_motion as preview
from surface_estimator_ur5e.pyroki_ik import (
    PyrokiRTDEControlAdapter,
    transform_to_ur_pose,
    ur_pose_to_transform,
)


DEFAULT_SIMPLE_DMP_ROOT = Path("~/simple_dmp")
DEFAULT_OUTPUT = Path("data/dmp/tray_pick_insert_dmp.npz")
PHASE_NAMES = (
    "approach_grasp",
    "grasp_close",
    "transfer_to_preinsert",
    "insert",
    "release",
)
DMP_PHASE_STAGE_NAMES = {
    "approach_grasp": ("marker grasp moveL",),
    "transfer_to_preinsert": (
        "post-grasp lift",
        "four-pin rotation",
        "post-rotation pin centering",
        "post-rotation pin-axis approach",
        "post-approach image alignment correction",
    ),
    "insert": ("final pin insertion descend",),
}


@dataclass(frozen=True)
class PhaseSeed:
    """One continuous Cartesian seed primitive."""

    name: str
    poses_ur: np.ndarray
    duration_s: float


@dataclass
class PhaseModel:
    """A simple_dmp model plus immutable trained weights."""

    seed: PhaseSeed
    model: Any
    normalized_time: np.ndarray
    position_weights: np.ndarray
    orientation_weights: np.ndarray
    reproduction_position_rms_m: float
    reproduction_orientation_rms_rad: float


@dataclass(frozen=True)
class Episode:
    """One accepted synthetic demonstration."""

    tcp_pose_ur: np.ndarray
    joint_position_rad: np.ndarray
    gripper_percent: np.ndarray
    phase_id: np.ndarray
    object_attached: np.ndarray
    time_s: np.ndarray
    time_scale: float
    initial_joint_delta_rad: np.ndarray
    initial_tcp_translation_delta_m: float
    initial_tcp_rotation_delta_rad: float
    max_free_space_deviation_m: float
    max_free_space_rotation_deviation_rad: float
    max_insertion_lateral_deviation_m: float
    max_joint_step_rad: float
    max_joint_speed_rad_s: float
    attempt_index: int


class CandidateRejected(RuntimeError):
    """Raised when a sampled DMP trajectory fails a safety preflight."""


def parse_args() -> argparse.Namespace:
    """Build a generator CLI on top of the real motion configuration."""

    parser = motion.build_arg_parser()
    parser.description = (
        "Generate tray pick-to-four-pin-insertion demonstrations with ~/simple_dmp. "
        "All trajectories are offline and preflighted with PyRoki; no robot connection "
        "is opened."
    )
    parser.set_defaults(
        align_to_four_pin_frame=True,
        insert_after_pin_approach=True,
        no_set_tcp=True,
        execute=False,
    )
    parser.add_argument("--simple-dmp-root", type=Path, default=DEFAULT_SIMPLE_DMP_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--dataset-hz", type=float, default=20.0)
    parser.add_argument("--dmp-internal-samples", type=int, default=501)
    parser.add_argument("--dmp-basis-functions", type=int, default=60)
    parser.add_argument("--dmp-alpha", type=float, default=48.0)
    parser.add_argument("--dmp-beta", type=float, default=12.0)
    parser.add_argument(
        "--weight-noise-std",
        type=float,
        default=0.00005,
        help=(
            "Relative trained-weight noise for grasp and free-space transfer DMPs. "
            "simple_dmp weights are large, so the safe default is 5e-5."
        ),
    )
    parser.add_argument(
        "--insertion-weight-noise-std",
        type=float,
        default=0.0,
        help="Relative DMP weight noise during constrained insertion (default: 0).",
    )
    parser.add_argument("--episode-time-scale-min", type=float, default=0.85)
    parser.add_argument("--episode-time-scale-max", type=float, default=1.15)
    parser.add_argument(
        "--goal-offset-marker-mm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("DX", "DY", "DZ"),
        help=(
            "Fixed marker-frame translation applied to the transfer/pre-insert and "
            "insertion DMP goals without retraining their nominal weights."
        ),
    )
    parser.add_argument(
        "--initial-joint-noise-deg",
        type=float,
        default=5.0,
        help=(
            "Independent uniform half-range around each nominal initial joint. "
            "The first episode remains nominal; 0 disables initial-state variation."
        ),
    )
    parser.add_argument(
        "--max-initial-tcp-translation-mm",
        type=float,
        default=75.0,
        help="Reject randomized starts farther than this from the nominal initial TCP.",
    )
    parser.add_argument(
        "--max-initial-tcp-rotation-deg",
        type=float,
        default=25.0,
        help="Reject randomized starts rotated farther than this from the nominal initial TCP.",
    )
    parser.add_argument("--gripper-transition-s", type=float, default=0.5)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--max-free-space-deviation-mm", type=float, default=15.0)
    parser.add_argument(
        "--max-free-space-rotation-deviation-deg",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--max-insertion-lateral-deviation-mm",
        type=float,
        default=0.25,
    )
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=180.0)
    parser.add_argument("--playback-hz", type=float, default=20.0)
    parser.add_argument("--time-scale", type=float, default=4.0)
    parser.add_argument("--trajectory-step-mm", type=float, default=5.0)
    parser.add_argument("--trajectory-rotation-step-deg", type=float, default=3.0)
    parser.add_argument("--rotation-speed-deg-s", type=float, default=30.0)
    args = parser.parse_args()

    if args.execute:
        parser.error("This generator is offline; --execute is intentionally forbidden.")
    if not args.start_from_initial_pose:
        parser.error("DMP generation requires --start-from-initial-pose.")
    if args.auto_pin_image_align:
        parser.error("Live image search is unavailable during offline DMP generation.")
    if any(
        (
            args.continue_pin_sequence_from_current,
            args.continue_post_two_pin_release_from_current,
            args.continue_post_two_pin_after_close_from_current,
            args.continue_post_two_pin_tail_from_current,
            args.continue_post_two_pin_tail_next_from_current,
            args.continue_post_two_pin_tail_extra_from_current,
        )
    ):
        parser.error("Current-pose continuation modes cannot generate a full tray episode.")
    positive_names = (
        "episodes",
        "dataset_hz",
        "dmp_internal_samples",
        "dmp_basis_functions",
        "dmp_alpha",
        "dmp_beta",
        "episode_time_scale_min",
        "episode_time_scale_max",
        "max_initial_tcp_translation_mm",
        "max_initial_tcp_rotation_deg",
        "gripper_transition_s",
        "max_free_space_deviation_mm",
        "max_free_space_rotation_deviation_deg",
        "max_insertion_lateral_deviation_mm",
        "max_joint_speed_deg_s",
        "playback_hz",
        "time_scale",
        "trajectory_step_mm",
        "trajectory_rotation_step_deg",
        "rotation_speed_deg_s",
    )
    for name in positive_names:
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.dmp_internal_samples < 50:
        parser.error("--dmp-internal-samples must be at least 50.")
    if args.dmp_basis_functions >= args.dmp_internal_samples:
        parser.error("--dmp-basis-functions must be smaller than internal samples.")
    if args.weight_noise_std < 0.0 or args.insertion_weight_noise_std < 0.0:
        parser.error("DMP weight noise standard deviations cannot be negative.")
    if args.initial_joint_noise_deg < 0.0:
        parser.error("--initial-joint-noise-deg cannot be negative.")
    if not np.all(np.isfinite(args.goal_offset_marker_mm)):
        parser.error("--goal-offset-marker-mm must contain three finite values.")
    if args.episode_time_scale_max < args.episode_time_scale_min:
        parser.error("Episode time-scale max must be >= min.")
    if args.max_attempts < 0:
        parser.error("--max-attempts cannot be negative.")
    return args


def load_cartesian_dmp_class(simple_dmp_root: Path) -> tuple[type, Path]:
    """Import CartesianDMP from the requested local simple_dmp checkout."""

    root = simple_dmp_root.expanduser().resolve()
    module_path = root / "dmp" / "dmp_cartesian.py"
    if not module_path.exists():
        raise FileNotFoundError(f"simple_dmp Cartesian implementation not found: {module_path}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("dmp.dmp_cartesian")
    imported_path = Path(inspect.getfile(module)).resolve()
    if imported_path != module_path:
        raise RuntimeError(f"Imported dmp_cartesian from {imported_path}, expected {module_path}.")
    return module.CartesianDMP, root


def prefix_through_tray_insertion(
    events: list[preview.MotionEvent],
) -> list[preview.MotionEvent]:
    """Keep the real command prefix through the first tray insertion."""

    for index, event in enumerate(events):
        if event.name == "final pin insertion descend":
            return events[: index + 1]
    raise ValueError("Actual motion plan does not contain final tray insertion.")


def phase_seed_from_frames(
    name: str,
    stage_names: tuple[str, ...],
    start_pose_ur: np.ndarray,
    frames: list[preview.AnimationFrame],
    events: list[preview.MotionEvent],
    args: argparse.Namespace,
) -> PhaseSeed:
    """Extract and de-duplicate one continuous phase from solved preview frames."""

    selected = [
        np.asarray(frame.tcp_pose_ur, dtype=float)
        for frame in frames
        if frame.stage_name in stage_names
    ]
    if not selected:
        raise ValueError(f"No solved frames found for DMP phase {name!r}.")
    poses = [np.asarray(start_pose_ur, dtype=float), *selected]
    deduplicated = [poses[0]]
    for pose in poses[1:]:
        translation_change = float(np.linalg.norm(pose[:3] - deduplicated[-1][:3]))
        rotation_change = motion.rotation_delta_deg(deduplicated[-1], pose)
        if translation_change > 1e-10 or rotation_change > 1e-7:
            deduplicated.append(pose)
    if len(deduplicated) < 2:
        raise ValueError(f"DMP phase {name!r} has no motion.")

    duration_s = 0.0
    current_pose = np.asarray(frames[0].tcp_pose_ur, dtype=float)
    stage_name_set = set(stage_names)
    for event in events:
        if event.kind != "move":
            continue
        assert event.target_pose_ur is not None
        assert event.speed_m_s is not None
        target = np.asarray(event.target_pose_ur, dtype=float)
        if event.name in stage_name_set:
            translation_duration = float(np.linalg.norm(target[:3] - current_pose[:3])) / float(
                event.speed_m_s
            )
            rotation_duration = motion.rotation_delta_deg(current_pose, target) / float(
                args.rotation_speed_deg_s
            )
            duration_s += max(translation_duration, rotation_duration)
        current_pose = target
    return PhaseSeed(
        name=name,
        poses_ur=np.asarray(deduplicated, dtype=float),
        duration_s=max(duration_s, 0.25),
    )


def resample_pose_sequence(poses_ur: np.ndarray, sample_count: int) -> np.ndarray:
    """Resample Cartesian positions and orientations over normalized phase time."""

    poses = np.asarray(poses_ur, dtype=float)
    if poses.ndim != 2 or poses.shape[1] != 6 or len(poses) < 2:
        raise ValueError(f"Expected at least two 6D poses, got {poses.shape}.")
    source_t = np.linspace(0.0, 1.0, len(poses))
    target_t = np.linspace(0.0, 1.0, sample_count)
    result = np.empty((sample_count, 6), dtype=float)
    for axis in range(3):
        result[:, axis] = np.interp(target_t, source_t, poses[:, axis])
    result[:, 3:] = Slerp(source_t, Rotation.from_rotvec(poses[:, 3:]))(target_t).as_rotvec()
    return result


def rotation_to_numpy_quaternion(rotations: Rotation) -> np.ndarray:
    """Convert scipy xyzw quaternions to numpy-quaternion wxyz storage."""

    xyzw = rotations.as_quat()
    return quaternion.as_quat_array(xyzw[:, [3, 0, 1, 2]])


def numpy_quaternion_to_rotation(quaternions: np.ndarray) -> Rotation:
    """Convert numpy-quaternion values to scipy Rotation."""

    wxyz = quaternion.as_float_array(quaternions)
    return Rotation.from_quat(wxyz[:, [1, 2, 3, 0]])


def enforce_pose_endpoints(
    poses_ur: np.ndarray,
    start_pose_ur: np.ndarray,
    goal_pose_ur: np.ndarray,
) -> np.ndarray:
    """Smoothly remove finite-horizon DMP residual while fixing both endpoints."""

    poses = np.asarray(poses_ur, dtype=float).copy()
    start = np.asarray(start_pose_ur, dtype=float)
    goal = np.asarray(goal_pose_ur, dtype=float)
    phase = np.linspace(0.0, 1.0, len(poses))
    blend = phase * phase * (3.0 - 2.0 * phase)
    start_error = start[:3] - poses[0, :3]
    goal_error = goal[:3] - poses[-1, :3]
    poses[:, :3] += (1.0 - blend[:, None]) * start_error + blend[:, None] * goal_error

    output_rotation = Rotation.from_rotvec(poses[:, 3:])
    endpoint_corrections = Rotation.from_matrix(
        np.stack(
            [
                (Rotation.from_rotvec(start[3:]) * output_rotation[0].inv()).as_matrix(),
                (Rotation.from_rotvec(goal[3:]) * output_rotation[-1].inv()).as_matrix(),
            ]
        )
    )
    correction = Slerp([0.0, 1.0], endpoint_corrections)(blend)
    poses[:, 3:] = (correction * output_rotation).as_rotvec()
    poses[0] = start
    poses[-1] = goal
    return poses


def train_phase_model(
    seed: PhaseSeed,
    cartesian_dmp_class: type,
    args: argparse.Namespace,
) -> PhaseModel:
    """Fit one position+quaternion DMP and report nominal reproduction error."""

    normalized_time = np.linspace(0.0, 1.0, args.dmp_internal_samples)
    training_pose = resample_pose_sequence(seed.poses_ur, args.dmp_internal_samples)
    training_quaternion = rotation_to_numpy_quaternion(Rotation.from_rotvec(training_pose[:, 3:]))
    model = cartesian_dmp_class(
        n_bfs=args.dmp_basis_functions,
        alpha=args.dmp_alpha,
        beta=args.dmp_beta,
        cs_alpha=-np.log(0.0001),
    )
    model.train(training_pose[:, :3], training_quaternion, normalized_time, 1.0)
    position_weights = np.asarray(model.position_dmp.w, dtype=float).copy()
    orientation_weights = np.asarray(model.quaternion_dmp.w, dtype=float).copy()
    position, _velocity, _acceleration, orientation, _omega, _d_omega = model.rollout(
        normalized_time,
        1.0,
    )
    reproduced = np.column_stack([position, numpy_quaternion_to_rotation(orientation).as_rotvec()])
    reproduced = enforce_pose_endpoints(reproduced, training_pose[0], training_pose[-1])
    position_rms_m = float(
        np.sqrt(np.mean(np.sum((reproduced[:, :3] - training_pose[:, :3]) ** 2, axis=1)))
    )
    orientation_error = (
        Rotation.from_rotvec(reproduced[:, 3:]).inv() * Rotation.from_rotvec(training_pose[:, 3:])
    ).magnitude()
    return PhaseModel(
        seed=seed,
        model=model,
        normalized_time=normalized_time,
        position_weights=position_weights,
        orientation_weights=orientation_weights,
        reproduction_position_rms_m=position_rms_m,
        reproduction_orientation_rms_rad=float(np.sqrt(np.mean(orientation_error**2))),
    )


def smooth_weight_noise(
    rng: np.random.Generator,
    weights: np.ndarray,
    relative_std: float,
) -> np.ndarray:
    """Create low-frequency coefficient noise scaled per Cartesian axis."""

    if relative_std == 0.0:
        return np.zeros_like(weights)
    noise = gaussian_filter1d(rng.normal(size=weights.shape), sigma=2.0, axis=1)
    noise_std = np.std(noise, axis=1, keepdims=True)
    noise /= np.where(noise_std > 1e-12, noise_std, 1.0)
    weight_rms = np.sqrt(np.mean(weights**2, axis=1, keepdims=True))
    return relative_std * weight_rms * noise


def rollout_phase(
    phase_model: PhaseModel,
    rng: np.random.Generator,
    relative_noise_std: float,
    sample_count: int,
    start_pose_ur: np.ndarray | None = None,
    goal_pose_ur: np.ndarray | None = None,
) -> np.ndarray:
    """Sample one DMP primitive and resample it at the dataset frequency."""

    model = phase_model.model
    start = np.asarray(
        phase_model.seed.poses_ur[0] if start_pose_ur is None else start_pose_ur,
        dtype=float,
    )
    goal = np.asarray(
        phase_model.seed.poses_ur[-1] if goal_pose_ur is None else goal_pose_ur,
        dtype=float,
    )
    model.position_dmp.p0 = start[:3].copy()
    model.position_dmp.gp = goal[:3].copy()
    endpoint_quaternions = rotation_to_numpy_quaternion(
        Rotation.from_rotvec(np.stack([start[3:], goal[3:]]))
    )
    model.quaternion_dmp.q0 = endpoint_quaternions[0]
    model.quaternion_dmp.go = endpoint_quaternions[1]
    model.position_dmp.w = phase_model.position_weights + smooth_weight_noise(
        rng,
        phase_model.position_weights,
        relative_noise_std,
    )
    model.quaternion_dmp.w = phase_model.orientation_weights + smooth_weight_noise(
        rng,
        phase_model.orientation_weights,
        relative_noise_std,
    )
    position, _velocity, _acceleration, orientation, _omega, _d_omega = model.rollout(
        phase_model.normalized_time,
        1.0,
    )
    internal_pose = np.column_stack(
        [position, numpy_quaternion_to_rotation(orientation).as_rotvec()]
    )
    internal_pose = enforce_pose_endpoints(
        internal_pose,
        start,
        goal,
    )
    return resample_pose_sequence(internal_pose, sample_count)


def phase_sample_count(seed: PhaseSeed, time_scale: float, dataset_hz: float) -> int:
    """Return a uniform-rate sample count including both phase endpoints."""

    return max(3, int(np.ceil(seed.duration_s * time_scale * dataset_hz)) + 1)


def insertion_lateral_deviation_m(poses_ur: np.ndarray) -> float:
    """Measure departure from the straight constrained insertion axis."""

    position = np.asarray(poses_ur, dtype=float)[:, :3]
    axis = position[-1] - position[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-9:
        return 0.0
    axis /= axis_norm
    offset = position - position[0]
    lateral = offset - np.outer(offset @ axis, axis)
    return float(np.max(np.linalg.norm(lateral, axis=1)))


def pose_deviation(
    candidate: np.ndarray,
    seed_poses: np.ndarray,
) -> tuple[float, float]:
    """Return maximum translational and rotational deviation at equal phase."""

    reference = resample_pose_sequence(seed_poses, len(candidate))
    translation = float(np.max(np.linalg.norm(candidate[:, :3] - reference[:, :3], axis=1)))
    rotation = (
        Rotation.from_rotvec(candidate[:, 3:]).inv() * Rotation.from_rotvec(reference[:, 3:])
    ).magnitude()
    return translation, float(np.max(rotation))


def sample_initial_configuration(
    ik: PyrokiRTDEControlAdapter,
    nominal_q_rad: np.ndarray,
    nominal_tcp_pose_ur: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
    nominal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Sample a joint-valid initial state and derive its physically consistent TCP."""

    half_range_rad = np.deg2rad(float(args.initial_joint_noise_deg))
    delta_rad = (
        np.zeros(6, dtype=float)
        if nominal or half_range_rad == 0.0
        else rng.uniform(-half_range_rad, half_range_rad, size=6)
    )
    q_rad = np.asarray(nominal_q_rad, dtype=float) + delta_rad
    lower = np.asarray(ik.robot.joints.lower_limits, dtype=float)
    upper = np.asarray(ik.robot.joints.upper_limits, dtype=float)
    if np.any(q_rad < lower) or np.any(q_rad > upper):
        raise CandidateRejected("randomized initial joints exceed URDF limits")

    tcp_pose_ur = np.asarray(
        ik.getForwardKinematics(q_rad.tolist(), ik.tcp_offset_ur.tolist()),
        dtype=float,
    )
    translation_delta_m = float(
        np.linalg.norm(tcp_pose_ur[:3] - np.asarray(nominal_tcp_pose_ur, dtype=float)[:3])
    )
    rotation_delta_rad = float(
        (
            Rotation.from_rotvec(np.asarray(nominal_tcp_pose_ur, dtype=float)[3:]).inv()
            * Rotation.from_rotvec(tcp_pose_ur[3:])
        ).magnitude()
    )
    if translation_delta_m > args.max_initial_tcp_translation_mm / 1000.0:
        raise CandidateRejected(f"initial TCP translation {translation_delta_m * 1000.0:.1f} mm")
    if rotation_delta_rad > np.deg2rad(args.max_initial_tcp_rotation_deg):
        raise CandidateRejected(f"initial TCP rotation {np.rad2deg(rotation_delta_rad):.1f} deg")
    return q_rad, tcp_pose_ur, delta_rad, translation_delta_m, rotation_delta_rad


def assemble_episode_poses(
    phase_rollouts: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Join continuous DMP primitives with discrete gripper transitions."""

    close_percent = float(args.gripper_close_percent)
    transition_steps = max(2, int(np.ceil(args.gripper_transition_s * args.dataset_hz)))
    poses: list[np.ndarray] = []
    gripper: list[np.ndarray] = []
    phase_ids: list[np.ndarray] = []
    attached: list[np.ndarray] = []

    def append_chunk(
        chunk_pose: np.ndarray,
        chunk_gripper: np.ndarray,
        phase_id: int,
        chunk_attached: np.ndarray,
    ) -> None:
        poses.append(np.asarray(chunk_pose, dtype=float))
        gripper.append(np.asarray(chunk_gripper, dtype=float))
        phase_ids.append(np.full(len(chunk_pose), phase_id, dtype=np.int16))
        attached.append(np.asarray(chunk_attached, dtype=bool))

    approach = phase_rollouts["approach_grasp"]
    append_chunk(approach, np.zeros(len(approach)), 0, np.zeros(len(approach), dtype=bool))

    close_pose = np.repeat(approach[-1][None, :], transition_steps, axis=0)
    close_gripper = np.linspace(0.0, close_percent, transition_steps + 1)[1:]
    close_attached = np.zeros(transition_steps, dtype=bool)
    close_attached[-1] = True
    append_chunk(close_pose, close_gripper, 1, close_attached)

    transfer = phase_rollouts["transfer_to_preinsert"][1:]
    append_chunk(
        transfer,
        np.full(len(transfer), close_percent),
        2,
        np.ones(len(transfer), dtype=bool),
    )
    insertion = phase_rollouts["insert"][1:]
    append_chunk(
        insertion,
        np.full(len(insertion), close_percent),
        3,
        np.ones(len(insertion), dtype=bool),
    )

    release_pose = np.repeat(insertion[-1][None, :], transition_steps, axis=0)
    release_gripper = np.linspace(close_percent, 0.0, transition_steps + 1)[1:]
    release_attached = np.ones(transition_steps, dtype=bool)
    release_attached[-1] = False
    append_chunk(release_pose, release_gripper, 4, release_attached)
    return (
        np.concatenate(poses),
        np.concatenate(gripper),
        np.concatenate(phase_ids),
        np.concatenate(attached),
    )


def solve_episode_ik(
    ik: PyrokiRTDEControlAdapter,
    tcp_pose_ur: np.ndarray,
    initial_q_rad: np.ndarray,
    dataset_hz: float,
    max_joint_speed_rad_s: float,
) -> tuple[np.ndarray, float, float]:
    """Solve each sampled TCP pose on one continuous UR5e joint branch."""

    q_current = np.asarray(initial_q_rad, dtype=float).copy()
    joint_positions = np.empty((len(tcp_pose_ur), 6), dtype=float)
    max_step_rad = 0.0
    max_speed_rad_s = 0.0
    for index, pose in enumerate(np.asarray(tcp_pose_ur, dtype=float)):
        if index == 0 and np.allclose(
            pose,
            np.asarray(ik.getForwardKinematics(q_current.tolist(), ik.tcp_offset_ur.tolist())),
            atol=1e-8,
        ):
            q_next = q_current.copy()
        else:
            if not ik.getInverseKinematicsHasSolution(
                pose.tolist(),
                q_current.tolist(),
                motion.IK_POSITION_TOLERANCE_M,
                motion.IK_ORIENTATION_TOLERANCE_RAD,
            ):
                raise CandidateRejected(f"strict IK failed at sample {index}")
            q_next = np.asarray(
                ik.getInverseKinematics(
                    pose.tolist(),
                    q_current.tolist(),
                    motion.IK_POSITION_TOLERANCE_M,
                    motion.IK_ORIENTATION_TOLERANCE_RAD,
                ),
                dtype=float,
            )
        step_rad = float(np.max(np.abs(q_next - q_current)))
        speed_rad_s = step_rad * dataset_hz
        max_step_rad = max(max_step_rad, step_rad)
        max_speed_rad_s = max(max_speed_rad_s, speed_rad_s)
        if np.rad2deg(step_rad) > motion.MAX_IK_JOINT_STEP_DEG:
            raise CandidateRejected(
                f"IK branch jump {np.rad2deg(step_rad):.1f} deg at sample {index}"
            )
        if speed_rad_s > max_joint_speed_rad_s:
            raise CandidateRejected(
                f"joint speed {np.rad2deg(speed_rad_s):.1f} deg/s at sample {index}"
            )
        joint_positions[index] = q_next
        q_current = q_next
    return joint_positions, max_step_rad, max_speed_rad_s


def sample_episode(
    phase_models: dict[str, PhaseModel],
    ik: PyrokiRTDEControlAdapter,
    nominal_initial_q_rad: np.ndarray,
    nominal_initial_tcp_pose_ur: np.ndarray,
    goal_offset_base_m: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
    episode_index: int,
    attempt_index: int,
) -> Episode:
    """Sample, validate, and solve one candidate demonstration."""

    nominal = episode_index == 0
    (
        episode_initial_q_rad,
        episode_initial_tcp_pose_ur,
        initial_joint_delta_rad,
        initial_tcp_translation_delta_m,
        initial_tcp_rotation_delta_rad,
    ) = sample_initial_configuration(
        ik,
        nominal_initial_q_rad,
        nominal_initial_tcp_pose_ur,
        rng,
        args,
        nominal,
    )
    time_scale = (
        1.0
        if nominal
        else float(rng.uniform(args.episode_time_scale_min, args.episode_time_scale_max))
    )
    phase_rollouts: dict[str, np.ndarray] = {}
    max_translation_deviation_m = 0.0
    max_rotation_deviation_rad = 0.0
    for name, phase_model in phase_models.items():
        relative_noise = (
            args.insertion_weight_noise_std if name == "insert" else args.weight_noise_std
        )
        if nominal:
            relative_noise = 0.0
        sample_count = phase_sample_count(phase_model.seed, time_scale, args.dataset_hz)
        start_pose_ur = episode_initial_tcp_pose_ur if name == "approach_grasp" else None
        goal_pose_ur = None
        if name == "transfer_to_preinsert":
            goal_pose_ur = phase_model.seed.poses_ur[-1].copy()
            goal_pose_ur[:3] += goal_offset_base_m
        elif name == "insert":
            start_pose_ur = phase_model.seed.poses_ur[0].copy()
            start_pose_ur[:3] += goal_offset_base_m
            goal_pose_ur = phase_model.seed.poses_ur[-1].copy()
            goal_pose_ur[:3] += goal_offset_base_m
        rollout = rollout_phase(
            phase_model,
            rng,
            relative_noise,
            sample_count,
            start_pose_ur=start_pose_ur,
            goal_pose_ur=goal_pose_ur,
        )
        phase_rollouts[name] = rollout
        if name != "insert":
            deviation_seed = phase_model.seed.poses_ur
            if start_pose_ur is not None or goal_pose_ur is not None:
                deviation_seed = rollout_phase(
                    phase_model,
                    rng,
                    0.0,
                    sample_count,
                    start_pose_ur=start_pose_ur,
                    goal_pose_ur=goal_pose_ur,
                )
            translation_deviation_m, rotation_deviation_rad = pose_deviation(
                rollout,
                deviation_seed,
            )
            max_translation_deviation_m = max(
                max_translation_deviation_m,
                translation_deviation_m,
            )
            max_rotation_deviation_rad = max(
                max_rotation_deviation_rad,
                rotation_deviation_rad,
            )

    insertion_deviation_m = insertion_lateral_deviation_m(phase_rollouts["insert"])
    if max_translation_deviation_m > args.max_free_space_deviation_mm / 1000.0:
        raise CandidateRejected(
            f"free-space deviation {max_translation_deviation_m * 1000.0:.2f} mm"
        )
    if max_rotation_deviation_rad > np.deg2rad(args.max_free_space_rotation_deviation_deg):
        raise CandidateRejected(
            f"free-space rotation deviation {np.rad2deg(max_rotation_deviation_rad):.2f} deg"
        )
    if insertion_deviation_m > args.max_insertion_lateral_deviation_mm / 1000.0:
        raise CandidateRejected(
            f"insertion lateral deviation {insertion_deviation_m * 1000.0:.3f} mm"
        )

    tcp_pose_ur, gripper_percent, phase_id, object_attached = assemble_episode_poses(
        phase_rollouts,
        args,
    )
    joint_position_rad, max_joint_step_rad, max_joint_speed_rad_s = solve_episode_ik(
        ik,
        tcp_pose_ur,
        episode_initial_q_rad,
        args.dataset_hz,
        np.deg2rad(args.max_joint_speed_deg_s),
    )
    return Episode(
        tcp_pose_ur=tcp_pose_ur,
        joint_position_rad=joint_position_rad,
        gripper_percent=gripper_percent,
        phase_id=phase_id,
        object_attached=object_attached,
        time_s=np.arange(len(tcp_pose_ur), dtype=float) / args.dataset_hz,
        time_scale=time_scale,
        initial_joint_delta_rad=initial_joint_delta_rad,
        initial_tcp_translation_delta_m=initial_tcp_translation_delta_m,
        initial_tcp_rotation_delta_rad=initial_tcp_rotation_delta_rad,
        max_free_space_deviation_m=max_translation_deviation_m,
        max_free_space_rotation_deviation_rad=max_rotation_deviation_rad,
        max_insertion_lateral_deviation_m=insertion_deviation_m,
        max_joint_step_rad=max_joint_step_rad,
        max_joint_speed_rad_s=max_joint_speed_rad_s,
        attempt_index=attempt_index,
    )


def local_tcp_deltas(tcp_pose_ur: np.ndarray) -> np.ndarray:
    """Return the next-pose action as a local SE(3) delta."""

    poses = np.asarray(tcp_pose_ur, dtype=float)
    delta = np.zeros_like(poses)
    for index in range(len(poses) - 1):
        current = ur_pose_to_transform(poses[index])
        following = ur_pose_to_transform(poses[index + 1])
        delta[index] = transform_to_ur_pose(np.linalg.inv(current) @ following)
    return delta


def save_dataset(
    output_path: Path,
    episodes: list[Episode],
    phase_models: dict[str, PhaseModel],
    rejected_reasons: list[str],
    simple_dmp_root: Path,
    tray_fit_rms_m: float,
    nominal_initial_q_rad: np.ndarray,
    nominal_initial_tcp_pose_ur: np.ndarray,
    goal_offset_base_m: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Save a compact NPZ dataset and human-readable validation metadata."""

    output = motion.resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_lengths = np.asarray([len(episode.tcp_pose_ur) for episode in episodes], dtype=np.int64)
    episode_ends = np.cumsum(episode_lengths)
    episode_starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    episode_id = np.concatenate(
        [np.full(length, index, dtype=np.int32) for index, length in enumerate(episode_lengths)]
    )
    tcp_pose_ur = np.concatenate([episode.tcp_pose_ur for episode in episodes])
    joint_position_rad = np.concatenate([episode.joint_position_rad for episode in episodes])
    gripper_percent = np.concatenate([episode.gripper_percent for episode in episodes])
    phase_id = np.concatenate([episode.phase_id for episode in episodes])
    object_attached = np.concatenate([episode.object_attached for episode in episodes])
    time_s = np.concatenate([episode.time_s for episode in episodes])
    action_tcp_pose_ur = np.concatenate(
        [np.vstack([episode.tcp_pose_ur[1:], episode.tcp_pose_ur[-1]]) for episode in episodes]
    )
    action_joint_position_rad = np.concatenate(
        [
            np.vstack([episode.joint_position_rad[1:], episode.joint_position_rad[-1]])
            for episode in episodes
        ]
    )
    action_gripper_percent = np.concatenate(
        [
            np.concatenate([episode.gripper_percent[1:], episode.gripper_percent[-1:]])
            for episode in episodes
        ]
    )
    action_tcp_delta_local = np.concatenate(
        [local_tcp_deltas(episode.tcp_pose_ur) for episode in episodes]
    )
    episode_initial_joint_position_rad = np.stack(
        [episode.joint_position_rad[0] for episode in episodes]
    )
    episode_initial_tcp_pose_ur = np.stack([episode.tcp_pose_ur[0] for episode in episodes])
    episode_initial_joint_delta_rad = np.stack(
        [episode.initial_joint_delta_rad for episode in episodes]
    )
    np.savez_compressed(
        output,
        episode_id=episode_id,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        time_s=time_s,
        tcp_pose_ur=tcp_pose_ur,
        joint_position_rad=joint_position_rad,
        gripper_percent=gripper_percent,
        phase_id=phase_id,
        object_attached=object_attached,
        action_tcp_pose_ur=action_tcp_pose_ur,
        action_tcp_delta_local=action_tcp_delta_local,
        action_joint_position_rad=action_joint_position_rad,
        action_gripper_percent=action_gripper_percent,
        episode_initial_joint_position_rad=episode_initial_joint_position_rad,
        episode_initial_tcp_pose_ur=episode_initial_tcp_pose_ur,
        episode_initial_joint_delta_rad=episode_initial_joint_delta_rad,
    )

    metadata_path = output.with_suffix(".json")
    metadata = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_tray_dmp_dataset.py",
        "source_motion": "scripts/marker_based_motion.py tray prefix through final pin insertion",
        "simple_dmp_root": str(simple_dmp_root),
        "output_npz": str(output),
        "episode_count": len(episodes),
        "total_samples": int(len(tcp_pose_ur)),
        "dataset_hz": float(args.dataset_hz),
        "random_seed": int(args.random_seed),
        "phase_names": list(PHASE_NAMES),
        "episode_ends_are_exclusive": True,
        "tray_pin_fit_rms_m": float(tray_fit_rms_m),
        "goal_retargeting": {
            "marker_offset_mm": np.asarray(args.goal_offset_marker_mm, dtype=float).tolist(),
            "base_offset_m": np.asarray(goal_offset_base_m, dtype=float).tolist(),
            "nominal_dmp_weights_reused": True,
            "retargeted_phases": ["transfer_to_preinsert", "insert"],
        },
        "exact_endpoint_policy": (
            "Each approach starts at FK(randomized initial joints). DMP finite-horizon "
            "residual is smoothly corrected so grasp and insertion endpoints remain "
            "identical to the real motion seed."
        ),
        "initial_state_randomization": {
            "strategy": "independent uniform joint offsets with FK-derived TCP start",
            "first_episode_nominal": True,
            "joint_half_range_deg": float(args.initial_joint_noise_deg),
            "max_tcp_translation_m": float(args.max_initial_tcp_translation_mm / 1000.0),
            "max_tcp_rotation_rad": float(np.deg2rad(args.max_initial_tcp_rotation_deg)),
            "nominal_joint_position_rad": np.asarray(nominal_initial_q_rad, dtype=float).tolist(),
            "nominal_tcp_pose_ur": np.asarray(nominal_initial_tcp_pose_ur, dtype=float).tolist(),
        },
        "validation": {
            "pyroki_strict_ik": True,
            "joint_limits": True,
            "joint_branch_jump": True,
            "joint_speed": True,
            "tray_pin_endpoint_fit": True,
            "collision_checked": False,
            "contact_physics_checked": False,
            "real_robot_executed": False,
            "camera_observations_included": False,
        },
        "dmp": {
            "internal_samples": int(args.dmp_internal_samples),
            "basis_functions": int(args.dmp_basis_functions),
            "alpha": float(args.dmp_alpha),
            "beta": float(args.dmp_beta),
            "weight_noise_std": float(args.weight_noise_std),
            "insertion_weight_noise_std": float(args.insertion_weight_noise_std),
            "phase_reproduction": {
                name: {
                    "position_rms_m": float(model.reproduction_position_rms_m),
                    "orientation_rms_rad": float(model.reproduction_orientation_rms_rad),
                    "seed_duration_s": float(model.seed.duration_s),
                    "seed_pose_count": int(len(model.seed.poses_ur)),
                }
                for name, model in phase_models.items()
            },
        },
        "episodes": [
            {
                "index": index,
                "samples": len(episode.tcp_pose_ur),
                "duration_s": float(episode.time_s[-1]),
                "time_scale": float(episode.time_scale),
                "initial_joint_delta_rad": episode.initial_joint_delta_rad.tolist(),
                "initial_tcp_translation_delta_m": float(episode.initial_tcp_translation_delta_m),
                "initial_tcp_rotation_delta_rad": float(episode.initial_tcp_rotation_delta_rad),
                "max_free_space_deviation_m": float(episode.max_free_space_deviation_m),
                "max_free_space_rotation_deviation_rad": float(
                    episode.max_free_space_rotation_deviation_rad
                ),
                "max_insertion_lateral_deviation_m": float(
                    episode.max_insertion_lateral_deviation_m
                ),
                "max_joint_step_rad": float(episode.max_joint_step_rad),
                "max_joint_speed_rad_s": float(episode.max_joint_speed_rad_s),
                "accepted_attempt": int(episode.attempt_index),
            }
            for index, episode in enumerate(episodes)
        ],
        "rejected_candidate_count": len(rejected_reasons),
        "rejected_candidate_reasons": rejected_reasons,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output, metadata_path


def tray_fit_rms_m(
    args: argparse.Namespace,
    base_t_marker: np.ndarray,
    seed_frames: list[preview.AnimationFrame],
) -> float:
    """Recover the same visual tray-to-four-pin endpoint diagnostic."""

    tray_mesh, tray_hole_centers, _path = preview.load_visual_tray_mesh(
        preview.DEFAULT_VISUAL_TRAY_STL
    )
    preview.shorten_lower_tray_handle(
        tray_mesh,
        preview.DEFAULT_VISUAL_TRAY_HANDLE_ROOT_Y_MM / 1000.0,
        preview.DEFAULT_VISUAL_TRAY_HANDLE_SCALE,
    )
    _transforms, _pin_centers, fit_rms_m, _shift, _marker_error = preview.visual_tray_transforms(
        args,
        base_t_marker,
        seed_frames,
        np.asarray(tray_mesh.vertices, dtype=float),
        tray_hole_centers,
    )
    return float(fit_rms_m)


def main() -> None:
    args = parse_args()
    motion.validate_args(args)
    cartesian_dmp_class, simple_dmp_root = load_cartesian_dmp_class(args.simple_dmp_root)
    initial_q_rad = np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
    print("Initializing offline PyRoki solver; no RTDE connection will be opened...", flush=True)
    ik = PyrokiRTDEControlAdapter(
        initial_q_rad=initial_q_rad,
        tcp_offset_ur=np.asarray(args.tcp_offset_ur, dtype=float),
    )
    initial_q_rad, initial_tcp_pose_ur, base_t_marker, all_events = (
        preview.build_actual_motion_events(args, ik)
    )
    tray_events = prefix_through_tray_insertion(all_events)
    seed_frames = preview.solve_animation_frames(
        args,
        ik,
        initial_q_rad,
        initial_tcp_pose_ur,
        tray_events,
    )

    phase_seeds: dict[str, PhaseSeed] = {}
    phase_start = np.asarray(initial_tcp_pose_ur, dtype=float)
    for name in ("approach_grasp", "transfer_to_preinsert", "insert"):
        seed = phase_seed_from_frames(
            name,
            DMP_PHASE_STAGE_NAMES[name],
            phase_start,
            seed_frames,
            tray_events,
            args,
        )
        phase_seeds[name] = seed
        phase_start = seed.poses_ur[-1]

    print(f"Training {len(phase_seeds)} Cartesian DMP primitives from {simple_dmp_root}...")
    phase_models = {
        name: train_phase_model(seed, cartesian_dmp_class, args)
        for name, seed in phase_seeds.items()
    }
    for name, model in phase_models.items():
        print(
            f"  {name}: reproduction RMS "
            f"{model.reproduction_position_rms_m * 1000.0:.3f} mm, "
            f"{np.rad2deg(model.reproduction_orientation_rms_rad):.3f} deg"
        )

    pin_fit_rms_m = tray_fit_rms_m(args, base_t_marker, seed_frames)
    goal_offset_base_m = base_t_marker[:3, :3] @ (
        np.asarray(args.goal_offset_marker_mm, dtype=float) / 1000.0
    )
    if np.linalg.norm(goal_offset_base_m) > 0.0:
        print(
            "Retargeting transfer/insertion DMP goals by marker-frame offset mm "
            f"{np.asarray(args.goal_offset_marker_mm, dtype=float).tolist()} "
            "without retraining nominal weights."
        )
    rng = np.random.default_rng(args.random_seed)
    maximum_attempts = args.max_attempts or max(args.episodes * 20, args.episodes)
    episodes: list[Episode] = []
    rejected_reasons: list[str] = []
    for attempt_index in range(maximum_attempts):
        if len(episodes) >= args.episodes:
            break
        try:
            episode = sample_episode(
                phase_models,
                ik,
                initial_q_rad,
                initial_tcp_pose_ur,
                goal_offset_base_m,
                rng,
                args,
                len(episodes),
                attempt_index,
            )
        except CandidateRejected as exc:
            rejected_reasons.append(str(exc))
            print(f"  reject attempt {attempt_index}: {exc}")
            continue
        episodes.append(episode)
        print(
            f"  accepted {len(episodes):03d}/{args.episodes:03d}: "
            f"{len(episode.tcp_pose_ur)} samples, "
            f"start Δq {np.max(np.abs(np.rad2deg(episode.initial_joint_delta_rad))):.1f} deg, "
            f"start Δtcp {episode.initial_tcp_translation_delta_m * 1000.0:.1f} mm, "
            f"free deviation {episode.max_free_space_deviation_m * 1000.0:.2f} mm, "
            f"max joint speed {np.rad2deg(episode.max_joint_speed_rad_s):.1f} deg/s"
        )
    if len(episodes) != args.episodes:
        raise RuntimeError(
            f"Accepted only {len(episodes)}/{args.episodes} episodes after "
            f"{maximum_attempts} attempts. Reduce DMP noise or inspect rejection logs."
        )

    output_path, metadata_path = save_dataset(
        args.output,
        episodes,
        phase_models,
        rejected_reasons,
        simple_dmp_root,
        pin_fit_rms_m,
        initial_q_rad,
        initial_tcp_pose_ur,
        goal_offset_base_m,
        args,
    )
    print(f"Saved DMP demonstrations: {output_path}")
    print(f"Saved validation metadata: {metadata_path}")
    print(f"Tray four-pin fit RMS: {pin_fit_rms_m * 1000.0:.6f} mm")
    print("Dataset is kinematic synthetic data; collision/contact and images are not included.")


if __name__ == "__main__":
    main()
