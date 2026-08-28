#!/usr/bin/env python3
"""Generate goal-conditioned DMP recovery demonstrations from a base dataset."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import generate_tray_dmp_dataset as dmpgen
import marker_based_motion as motion
import preview_tray_dmp_dataset as dataset_viewer
from surface_estimator_ur5e.pyroki_ik import (
    PyrokiRTDEControlAdapter,
    transform_to_ur_pose,
    ur_pose_to_transform,
)


DEFAULT_INPUT = Path("data/dmp/tray_pick_insert_dmp_initial6.npz")
DEFAULT_OUTPUT = Path("data/dmp/tray_pick_insert_dmp_recovery.npz")


@dataclass
class SourcePhase:
    """One episode's free-space phase and its trained nominal DMP."""

    episode_index: int
    phase_id: int
    tcp_pose_ur: np.ndarray
    joint_position_rad: np.ndarray
    gripper_percent: np.ndarray
    object_attached: np.ndarray
    model: dmpgen.PhaseModel


@dataclass(frozen=True)
class RecoveryExample:
    """One perturbed-state rollout paired with its varied nominal trajectory."""

    source_episode_index: int
    source_phase_id: int
    source_phase_start_pose_ur: np.ndarray
    source_phase_goal_pose_ur: np.ndarray
    nominal_tcp_pose_ur: np.ndarray
    nominal_joint_position_rad: np.ndarray
    nominal_start_delta_local: np.ndarray
    nominal_goal_delta_local: np.ndarray
    perturbation_fraction: float
    nominal_perturbation_pose_ur: np.ndarray
    perturbed_pose_ur: np.ndarray
    perturbation_delta_local: np.ndarray
    recovery_command_tcp_pose_ur: np.ndarray
    recovery_command_joint_position_rad: np.ndarray
    recovery_state_tcp_pose_ur: np.ndarray
    recovery_state_joint_position_rad: np.ndarray
    goal_pose_ur: np.ndarray
    gripper_percent: float
    object_attached: bool
    max_nominal_joint_speed_rad_s: float
    max_recovery_joint_speed_rad_s: float
    max_realized_position_error_m: float
    max_realized_orientation_error_rad: float
    accepted_attempt: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate offline recovery state-action demonstrations from a base tray DMP "
            "dataset. No MuJoCo, RTDE, gripper, or real robot connection is used."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simple-dmp-root", type=Path, default=dmpgen.DEFAULT_SIMPLE_DMP_ROOT)
    parser.add_argument("--source-phase", default="transfer_to_preinsert")
    parser.add_argument("--recoveries-per-episode", type=int, default=2)
    parser.add_argument("--random-seed", type=int, default=23)
    parser.add_argument("--max-attempts-per-recovery", type=int, default=25)
    parser.add_argument("--dmp-internal-samples", type=int, default=501)
    parser.add_argument("--dmp-basis-functions", type=int, default=60)
    parser.add_argument("--dmp-alpha", type=float, default=48.0)
    parser.add_argument("--dmp-beta", type=float, default=12.0)
    parser.add_argument("--start-position-variation-mm", type=float, default=5.0)
    parser.add_argument("--start-orientation-variation-deg", type=float, default=2.0)
    parser.add_argument("--goal-position-variation-mm", type=float, default=3.0)
    parser.add_argument("--goal-orientation-variation-deg", type=float, default=1.0)
    parser.add_argument("--perturb-position-mm", type=float, default=5.0)
    parser.add_argument("--perturb-orientation-deg", type=float, default=3.0)
    parser.add_argument("--perturb-fraction-min", type=float, default=0.25)
    parser.add_argument("--perturb-fraction-max", type=float, default=0.75)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=180.0)
    parser.add_argument("--max-realized-position-error-mm", type=float, default=0.05)
    parser.add_argument("--max-realized-orientation-error-deg", type=float, default=0.05)
    parser.add_argument(
        "--tcp-offset-ur",
        type=float,
        nargs=6,
        default=motion.DEFAULT_TCP_OFFSET_UR,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    args = parser.parse_args()

    positive = (
        "recoveries_per_episode",
        "max_attempts_per_recovery",
        "dmp_internal_samples",
        "dmp_basis_functions",
        "dmp_alpha",
        "dmp_beta",
        "max_joint_speed_deg_s",
        "max_realized_position_error_mm",
        "max_realized_orientation_error_deg",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    nonnegative = (
        "start_position_variation_mm",
        "start_orientation_variation_deg",
        "goal_position_variation_mm",
        "goal_orientation_variation_deg",
        "perturb_position_mm",
        "perturb_orientation_deg",
    )
    for name in nonnegative:
        if float(getattr(args, name)) < 0.0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative.")
    if args.dmp_internal_samples < 50:
        parser.error("--dmp-internal-samples must be at least 50.")
    if args.dmp_basis_functions >= args.dmp_internal_samples:
        parser.error("--dmp-basis-functions must be smaller than internal samples.")
    if not 0.0 < args.perturb_fraction_min < args.perturb_fraction_max < 1.0:
        parser.error("Perturbation fraction bounds must satisfy 0 < min < max < 1.")
    return args


def bounded_vector(rng: np.random.Generator, maximum_norm: float) -> np.ndarray:
    """Sample uniformly in a 3D ball with the requested maximum norm."""

    if maximum_norm == 0.0:
        return np.zeros(3, dtype=float)
    direction = rng.normal(size=3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.asarray([1.0, 0.0, 0.0])
    else:
        direction /= norm
    radius = maximum_norm * float(rng.random()) ** (1.0 / 3.0)
    return direction * radius


def varied_pose(
    pose_ur: np.ndarray,
    translation_delta_m: np.ndarray,
    rotation_delta_rad: np.ndarray,
) -> np.ndarray:
    """Apply base-frame translation and rotation perturbations to a UR pose."""

    transform = ur_pose_to_transform(np.asarray(pose_ur, dtype=float))
    transform[:3, 3] += np.asarray(translation_delta_m, dtype=float)
    transform[:3, :3] = (
        Rotation.from_rotvec(np.asarray(rotation_delta_rad, dtype=float)).as_matrix()
        @ transform[:3, :3]
    )
    return transform_to_ur_pose(transform)


def relative_pose_delta(from_pose_ur: np.ndarray, to_pose_ur: np.ndarray) -> np.ndarray:
    """Return the local SE(3) delta from one UR pose to another."""

    from_transform = ur_pose_to_transform(np.asarray(from_pose_ur, dtype=float))
    to_transform = ur_pose_to_transform(np.asarray(to_pose_ur, dtype=float))
    return transform_to_ur_pose(np.linalg.inv(from_transform) @ to_transform)


def strict_ik_solution(
    ik: PyrokiRTDEControlAdapter,
    pose_ur: np.ndarray,
    q_near_rad: np.ndarray,
    label: str,
) -> np.ndarray:
    """Solve one strict IK state and reject discontinuous branches."""

    if not ik.getInverseKinematicsHasSolution(
        np.asarray(pose_ur, dtype=float).tolist(),
        np.asarray(q_near_rad, dtype=float).tolist(),
        motion.IK_POSITION_TOLERANCE_M,
        motion.IK_ORIENTATION_TOLERANCE_RAD,
    ):
        raise dmpgen.CandidateRejected(f"strict IK failed for {label}")
    solution = np.asarray(
        ik.getInverseKinematics(
            np.asarray(pose_ur, dtype=float).tolist(),
            np.asarray(q_near_rad, dtype=float).tolist(),
            motion.IK_POSITION_TOLERANCE_M,
            motion.IK_ORIENTATION_TOLERANCE_RAD,
        ),
        dtype=float,
    )
    step_deg = float(np.rad2deg(np.max(np.abs(solution - q_near_rad))))
    if step_deg > motion.MAX_IK_JOINT_STEP_DEG:
        raise dmpgen.CandidateRejected(f"{label} IK branch jump {step_deg:.1f} deg")
    return solution


def realized_tcp_poses(
    ik: PyrokiRTDEControlAdapter,
    joint_position_rad: np.ndarray,
) -> np.ndarray:
    """Record FK-realized TCP states for every solved joint command."""

    return np.asarray(
        [
            ik.getForwardKinematics(q.tolist(), ik.tcp_offset_ur.tolist())
            for q in np.asarray(joint_position_rad, dtype=float)
        ],
        dtype=float,
    )


def maximum_pose_error(
    realized_pose_ur: np.ndarray,
    command_pose_ur: np.ndarray,
) -> tuple[float, float]:
    """Return maximum FK position and orientation command errors."""

    position_error = np.linalg.norm(realized_pose_ur[:, :3] - command_pose_ur[:, :3], axis=1)
    orientation_error = (
        Rotation.from_rotvec(realized_pose_ur[:, 3:]).inv()
        * Rotation.from_rotvec(command_pose_ur[:, 3:])
    ).magnitude()
    return float(np.max(position_error)), float(np.max(orientation_error))


def source_phases(
    dataset: dataset_viewer.DmpDataset,
    phase_id: int,
    cartesian_dmp_class: type,
    args: argparse.Namespace,
) -> list[SourcePhase]:
    """Extract and train the selected contiguous phase from every base episode."""

    sources: list[SourcePhase] = []
    for episode_index in range(dataset.episode_count):
        episode = dataset.episode_slice(episode_index)
        phase_indices = np.flatnonzero(dataset.phase_id[episode] == phase_id)
        if len(phase_indices) < 5 or np.any(np.diff(phase_indices) != 1):
            raise ValueError(
                f"Episode {episode_index} phase {phase_id} is missing or non-contiguous."
            )
        local_slice = slice(int(phase_indices[0]), int(phase_indices[-1]) + 1)
        tcp_pose_ur = np.asarray(dataset.tcp_pose_ur[episode][local_slice], dtype=float)
        joint_position_rad = np.asarray(
            dataset.joint_position_rad[episode][local_slice], dtype=float
        )
        gripper_percent = np.asarray(dataset.gripper_percent[episode][local_slice], dtype=float)
        object_attached = np.asarray(dataset.object_attached[episode][local_slice], dtype=bool)
        seed = dmpgen.PhaseSeed(
            name=f"episode_{episode_index}_{dataset.phase_names[phase_id]}",
            poses_ur=tcp_pose_ur,
            duration_s=(len(tcp_pose_ur) - 1) / dataset.dataset_hz,
        )
        sources.append(
            SourcePhase(
                episode_index=episode_index,
                phase_id=phase_id,
                tcp_pose_ur=tcp_pose_ur,
                joint_position_rad=joint_position_rad,
                gripper_percent=gripper_percent,
                object_attached=object_attached,
                model=dmpgen.train_phase_model(seed, cartesian_dmp_class, args),
            )
        )
    return sources


def sample_recovery(
    source: SourcePhase,
    ik: PyrokiRTDEControlAdapter,
    cartesian_dmp_class: type,
    dataset_hz: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
    attempt_index: int,
) -> RecoveryExample:
    """Generate a varied nominal path and a DMP recovery from a perturbed state."""

    start_translation = bounded_vector(rng, args.start_position_variation_mm / 1000.0)
    start_rotation = bounded_vector(rng, np.deg2rad(args.start_orientation_variation_deg))
    goal_translation = bounded_vector(rng, args.goal_position_variation_mm / 1000.0)
    goal_rotation = bounded_vector(rng, np.deg2rad(args.goal_orientation_variation_deg))
    nominal_start = varied_pose(source.tcp_pose_ur[0], start_translation, start_rotation)
    nominal_goal = varied_pose(source.tcp_pose_ur[-1], goal_translation, goal_rotation)
    nominal_tcp_pose_ur = dmpgen.rollout_phase(
        source.model,
        rng,
        relative_noise_std=0.0,
        sample_count=len(source.tcp_pose_ur),
        start_pose_ur=nominal_start,
        goal_pose_ur=nominal_goal,
    )
    nominal_q_start = strict_ik_solution(
        ik,
        nominal_start,
        source.joint_position_rad[0],
        "varied nominal start",
    )
    nominal_joint_position_rad, _nominal_step, nominal_speed = dmpgen.solve_episode_ik(
        ik,
        nominal_tcp_pose_ur,
        nominal_q_start,
        dataset_hz,
        np.deg2rad(args.max_joint_speed_deg_s),
    )

    perturbation_fraction = float(rng.uniform(args.perturb_fraction_min, args.perturb_fraction_max))
    perturbation_index = int(round(perturbation_fraction * (len(nominal_tcp_pose_ur) - 1)))
    perturbation_index = int(np.clip(perturbation_index, 1, len(nominal_tcp_pose_ur) - 3))
    nominal_perturbation_pose = nominal_tcp_pose_ur[perturbation_index]
    perturb_translation = bounded_vector(rng, args.perturb_position_mm / 1000.0)
    perturb_rotation = bounded_vector(rng, np.deg2rad(args.perturb_orientation_deg))
    perturbed_pose = varied_pose(
        nominal_perturbation_pose,
        perturb_translation,
        perturb_rotation,
    )

    nominal_suffix = nominal_tcp_pose_ur[perturbation_index:]
    suffix_seed = dmpgen.PhaseSeed(
        name="perturbed_recovery_suffix",
        poses_ur=nominal_suffix,
        duration_s=(len(nominal_suffix) - 1) / dataset_hz,
    )
    suffix_model = dmpgen.train_phase_model(suffix_seed, cartesian_dmp_class, args)
    recovery_command_tcp_pose_ur = dmpgen.rollout_phase(
        suffix_model,
        rng,
        relative_noise_std=0.0,
        sample_count=len(nominal_suffix),
        start_pose_ur=perturbed_pose,
        goal_pose_ur=nominal_goal,
    )
    recovery_q_start = strict_ik_solution(
        ik,
        perturbed_pose,
        nominal_joint_position_rad[perturbation_index],
        "perturbed recovery start",
    )
    recovery_command_joint_position_rad, _recovery_step, recovery_speed = dmpgen.solve_episode_ik(
        ik,
        recovery_command_tcp_pose_ur,
        recovery_q_start,
        dataset_hz,
        np.deg2rad(args.max_joint_speed_deg_s),
    )
    recovery_state_joint_position_rad = recovery_command_joint_position_rad.copy()
    recovery_state_tcp_pose_ur = realized_tcp_poses(ik, recovery_state_joint_position_rad)
    position_error_m, orientation_error_rad = maximum_pose_error(
        recovery_state_tcp_pose_ur,
        recovery_command_tcp_pose_ur,
    )
    if position_error_m > args.max_realized_position_error_mm / 1000.0:
        raise dmpgen.CandidateRejected(
            f"FK realization position error {position_error_m * 1000.0:.3f} mm"
        )
    if orientation_error_rad > np.deg2rad(args.max_realized_orientation_error_deg):
        raise dmpgen.CandidateRejected(
            f"FK realization orientation error {np.rad2deg(orientation_error_rad):.3f} deg"
        )
    if not np.allclose(recovery_command_tcp_pose_ur[-1], nominal_goal, atol=1e-10):
        raise dmpgen.CandidateRejected("recovery DMP did not preserve its varied goal")

    return RecoveryExample(
        source_episode_index=source.episode_index,
        source_phase_id=source.phase_id,
        source_phase_start_pose_ur=source.tcp_pose_ur[0].copy(),
        source_phase_goal_pose_ur=source.tcp_pose_ur[-1].copy(),
        nominal_tcp_pose_ur=nominal_tcp_pose_ur,
        nominal_joint_position_rad=nominal_joint_position_rad,
        nominal_start_delta_local=relative_pose_delta(source.tcp_pose_ur[0], nominal_start),
        nominal_goal_delta_local=relative_pose_delta(source.tcp_pose_ur[-1], nominal_goal),
        perturbation_fraction=perturbation_fraction,
        nominal_perturbation_pose_ur=nominal_perturbation_pose.copy(),
        perturbed_pose_ur=perturbed_pose,
        perturbation_delta_local=relative_pose_delta(
            nominal_perturbation_pose,
            perturbed_pose,
        ),
        recovery_command_tcp_pose_ur=recovery_command_tcp_pose_ur,
        recovery_command_joint_position_rad=recovery_command_joint_position_rad,
        recovery_state_tcp_pose_ur=recovery_state_tcp_pose_ur,
        recovery_state_joint_position_rad=recovery_state_joint_position_rad,
        goal_pose_ur=nominal_goal,
        gripper_percent=float(source.gripper_percent[0]),
        object_attached=bool(source.object_attached[0]),
        max_nominal_joint_speed_rad_s=nominal_speed,
        max_recovery_joint_speed_rad_s=recovery_speed,
        max_realized_position_error_m=position_error_m,
        max_realized_orientation_error_rad=orientation_error_rad,
        accepted_attempt=attempt_index,
    )


def local_target_deltas(
    state_tcp_pose_ur: np.ndarray,
    target_tcp_pose_ur: np.ndarray,
) -> np.ndarray:
    """Compute local SE(3) actions from realized states to arbitrary targets."""

    return np.asarray(
        [
            relative_pose_delta(state_pose, target_pose)
            for state_pose, target_pose in zip(
                state_tcp_pose_ur,
                target_tcp_pose_ur,
                strict=True,
            )
        ],
        dtype=float,
    )


def episode_boundaries(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.cumsum(np.asarray(lengths, dtype=np.int64))
    starts = np.concatenate([np.asarray([0], dtype=np.int64), ends[:-1]])
    return starts, ends


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_dataset(
    output_path: Path,
    input_path: Path,
    examples: list[RecoveryExample],
    rejected_reasons: list[str],
    dataset_hz: float,
    source_phase_name: str,
    simple_dmp_root: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Save nominal paths, recovery rollouts, and realized state-action labels."""

    output = motion.resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nominal_lengths = np.asarray([len(example.nominal_tcp_pose_ur) for example in examples])
    recovery_lengths = np.asarray(
        [len(example.recovery_command_tcp_pose_ur) for example in examples]
    )
    nominal_starts, nominal_ends = episode_boundaries(nominal_lengths)
    recovery_starts, recovery_ends = episode_boundaries(recovery_lengths)

    nominal_tcp_pose_ur = np.concatenate([example.nominal_tcp_pose_ur for example in examples])
    nominal_joint_position_rad = np.concatenate(
        [example.nominal_joint_position_rad for example in examples]
    )
    recovery_command_tcp_pose_ur = np.concatenate(
        [example.recovery_command_tcp_pose_ur for example in examples]
    )
    recovery_command_joint_position_rad = np.concatenate(
        [example.recovery_command_joint_position_rad for example in examples]
    )
    recovery_state_tcp_pose_ur = np.concatenate(
        [example.recovery_state_tcp_pose_ur for example in examples]
    )
    recovery_state_joint_position_rad = np.concatenate(
        [example.recovery_state_joint_position_rad for example in examples]
    )
    recovery_action_tcp_pose_ur = np.concatenate(
        [
            np.vstack(
                [
                    example.recovery_command_tcp_pose_ur[1:],
                    example.recovery_command_tcp_pose_ur[-1],
                ]
            )
            for example in examples
        ]
    )
    recovery_action_joint_position_rad = np.concatenate(
        [
            np.vstack(
                [
                    example.recovery_command_joint_position_rad[1:],
                    example.recovery_command_joint_position_rad[-1],
                ]
            )
            for example in examples
        ]
    )
    recovery_goal_tcp_pose_ur = np.concatenate(
        [
            np.repeat(example.goal_pose_ur[None, :], length, axis=0)
            for example, length in zip(examples, recovery_lengths, strict=True)
        ]
    )
    recovery_action_tcp_delta_local = local_target_deltas(
        recovery_state_tcp_pose_ur,
        recovery_action_tcp_pose_ur,
    )
    recovery_goal_delta_local = local_target_deltas(
        recovery_state_tcp_pose_ur,
        recovery_goal_tcp_pose_ur,
    )
    recovery_action_joint_delta_rad = (
        recovery_action_joint_position_rad - recovery_state_joint_position_rad
    )
    recovery_id = np.concatenate(
        [np.full(length, index, dtype=np.int32) for index, length in enumerate(recovery_lengths)]
    )
    recovery_time_s = np.concatenate(
        [np.arange(length, dtype=float) / dataset_hz for length in recovery_lengths]
    )
    recovery_gripper_percent = np.concatenate(
        [
            np.full(length, example.gripper_percent, dtype=float)
            for example, length in zip(examples, recovery_lengths, strict=True)
        ]
    )
    recovery_object_attached = np.concatenate(
        [
            np.full(length, example.object_attached, dtype=bool)
            for example, length in zip(examples, recovery_lengths, strict=True)
        ]
    )

    np.savez_compressed(
        output,
        nominal_starts=nominal_starts,
        nominal_ends=nominal_ends,
        nominal_tcp_pose_ur=nominal_tcp_pose_ur,
        nominal_joint_position_rad=nominal_joint_position_rad,
        recovery_id=recovery_id,
        recovery_starts=recovery_starts,
        recovery_ends=recovery_ends,
        recovery_time_s=recovery_time_s,
        recovery_state_tcp_pose_ur=recovery_state_tcp_pose_ur,
        recovery_state_joint_position_rad=recovery_state_joint_position_rad,
        recovery_command_tcp_pose_ur=recovery_command_tcp_pose_ur,
        recovery_command_joint_position_rad=recovery_command_joint_position_rad,
        recovery_action_tcp_pose_ur=recovery_action_tcp_pose_ur,
        recovery_action_tcp_delta_local=recovery_action_tcp_delta_local,
        recovery_action_joint_position_rad=recovery_action_joint_position_rad,
        recovery_action_joint_delta_rad=recovery_action_joint_delta_rad,
        recovery_goal_tcp_pose_ur=recovery_goal_tcp_pose_ur,
        recovery_goal_delta_local=recovery_goal_delta_local,
        recovery_gripper_percent=recovery_gripper_percent,
        recovery_object_attached=recovery_object_attached,
        source_episode_index=np.asarray(
            [example.source_episode_index for example in examples], dtype=np.int32
        ),
        source_phase_id=np.asarray(
            [example.source_phase_id for example in examples], dtype=np.int16
        ),
        source_phase_start_pose_ur=np.stack(
            [example.source_phase_start_pose_ur for example in examples]
        ),
        source_phase_goal_pose_ur=np.stack(
            [example.source_phase_goal_pose_ur for example in examples]
        ),
        nominal_start_delta_local=np.stack(
            [example.nominal_start_delta_local for example in examples]
        ),
        nominal_goal_delta_local=np.stack(
            [example.nominal_goal_delta_local for example in examples]
        ),
        perturbation_fraction=np.asarray(
            [example.perturbation_fraction for example in examples], dtype=float
        ),
        nominal_perturbation_pose_ur=np.stack(
            [example.nominal_perturbation_pose_ur for example in examples]
        ),
        perturbed_pose_ur=np.stack([example.perturbed_pose_ur for example in examples]),
        perturbation_delta_local=np.stack(
            [example.perturbation_delta_local for example in examples]
        ),
        goal_pose_ur=np.stack([example.goal_pose_ur for example in examples]),
    )

    metadata_path = output.with_suffix(".json")
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/generate_tray_dmp_recovery_dataset.py",
        "source_dataset": str(input_path),
        "source_dataset_sha256": file_sha256(input_path),
        "output_npz": str(output),
        "simple_dmp_root": str(simple_dmp_root),
        "source_phase": source_phase_name,
        "recovery_count": len(examples),
        "nominal_sample_count": int(len(nominal_tcp_pose_ur)),
        "recovery_sample_count": int(len(recovery_state_tcp_pose_ur)),
        "dataset_hz": float(dataset_hz),
        "random_seed": int(args.random_seed),
        "realization_backend": "pyroki_fk_kinematic",
        "variations": {
            "start_position_max_m": float(args.start_position_variation_mm / 1000.0),
            "start_orientation_max_rad": float(np.deg2rad(args.start_orientation_variation_deg)),
            "goal_position_max_m": float(args.goal_position_variation_mm / 1000.0),
            "goal_orientation_max_rad": float(np.deg2rad(args.goal_orientation_variation_deg)),
            "perturb_position_max_m": float(args.perturb_position_mm / 1000.0),
            "perturb_orientation_max_rad": float(np.deg2rad(args.perturb_orientation_deg)),
            "perturb_fraction": [
                float(args.perturb_fraction_min),
                float(args.perturb_fraction_max),
            ],
        },
        "action_semantics": {
            "recovery_action_tcp_pose_ur": "next DMP command pose; final sample holds goal",
            "recovery_action_tcp_delta_local": (
                "local SE(3) delta from FK-realized state to next DMP command"
            ),
            "recovery_goal_delta_local": "local SE(3) residual from state to varied goal",
            "recovery_action_joint_position_rad": (
                "next strict-IK joint target; final sample holds goal"
            ),
        },
        "validation": {
            "pyroki_strict_ik": True,
            "joint_limits": True,
            "joint_branch_jump": True,
            "joint_speed": True,
            "fk_realized_states_recorded": True,
            "mujoco_executed": False,
            "real_robot_executed": False,
            "collision_checked": False,
            "contact_physics_checked": False,
        },
        "task_scope": (
            "Goal variation applies only to the selected free-space/pre-insert phase. "
            "It does not replace the exact physical four-pin insertion goal in the base dataset."
        ),
        "examples": [
            {
                "index": index,
                "source_episode": int(example.source_episode_index),
                "nominal_samples": len(example.nominal_tcp_pose_ur),
                "recovery_samples": len(example.recovery_state_tcp_pose_ur),
                "perturbation_fraction": float(example.perturbation_fraction),
                "max_nominal_joint_speed_rad_s": float(example.max_nominal_joint_speed_rad_s),
                "max_recovery_joint_speed_rad_s": float(example.max_recovery_joint_speed_rad_s),
                "max_realized_position_error_m": float(example.max_realized_position_error_m),
                "max_realized_orientation_error_rad": float(
                    example.max_realized_orientation_error_rad
                ),
                "accepted_attempt": int(example.accepted_attempt),
            }
            for index, example in enumerate(examples)
        ],
        "rejected_candidate_count": len(rejected_reasons),
        "rejected_candidate_reasons": rejected_reasons,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output, metadata_path


def main() -> None:
    args = parse_args()
    dataset = dataset_viewer.load_dataset(args.input, 0.0)
    input_path = dataset.path
    if args.source_phase not in dataset.phase_names:
        raise ValueError(
            f"Unknown source phase {args.source_phase!r}; available: {dataset.phase_names}."
        )
    source_phase_id = dataset.phase_names.index(args.source_phase)
    cartesian_dmp_class, simple_dmp_root = dmpgen.load_cartesian_dmp_class(args.simple_dmp_root)
    print("Initializing offline PyRoki solver; no RTDE connection will be opened...", flush=True)
    ik = PyrokiRTDEControlAdapter(
        initial_q_rad=np.asarray(dataset.joint_position_rad[0], dtype=float),
        tcp_offset_ur=np.asarray(args.tcp_offset_ur, dtype=float),
    )
    sources = source_phases(dataset, source_phase_id, cartesian_dmp_class, args)
    print(
        f"Trained {len(sources)} source-phase DMPs for {args.source_phase!r}; "
        f"sampling {args.recoveries_per_episode} recoveries per episode."
    )

    rng = np.random.default_rng(args.random_seed)
    examples: list[RecoveryExample] = []
    rejected_reasons: list[str] = []
    for source in sources:
        for recovery_slot in range(args.recoveries_per_episode):
            accepted = None
            for local_attempt in range(args.max_attempts_per_recovery):
                try:
                    accepted = sample_recovery(
                        source,
                        ik,
                        cartesian_dmp_class,
                        dataset.dataset_hz,
                        rng,
                        args,
                        local_attempt,
                    )
                except dmpgen.CandidateRejected as exc:
                    reason = (
                        f"episode {source.episode_index} recovery {recovery_slot} "
                        f"attempt {local_attempt}: {exc}"
                    )
                    rejected_reasons.append(reason)
                    print(f"  reject {reason}")
                    continue
                break
            if accepted is None:
                raise RuntimeError(
                    f"Could not generate recovery {recovery_slot} for episode "
                    f"{source.episode_index} after {args.max_attempts_per_recovery} attempts."
                )
            examples.append(accepted)
            print(
                f"  accepted {len(examples):03d}: episode {source.episode_index}, "
                f"fraction {accepted.perturbation_fraction:.2f}, "
                f"{len(accepted.recovery_state_tcp_pose_ur)} recovery samples, "
                f"max speed {np.rad2deg(accepted.max_recovery_joint_speed_rad_s):.1f} deg/s"
            )

    output_path, metadata_path = save_dataset(
        args.output,
        input_path,
        examples,
        rejected_reasons,
        dataset.dataset_hz,
        args.source_phase,
        simple_dmp_root,
        args,
    )
    print(f"Saved DMP recovery dataset: {output_path}")
    print(f"Saved recovery metadata: {metadata_path}")
    print("Realized states use PyRoki FK only; MuJoCo and the real robot were not used.")


if __name__ == "__main__":
    main()
