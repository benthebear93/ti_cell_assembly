#!/usr/bin/env python3
"""Train and persist one reusable DMP primitive for every real moveL stage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import generate_tray_dmp_dataset as dmpgen
import marker_based_motion as motion
import preview_assembly_motion as preview
from surface_estimator_ur5e.pyroki_ik import PyrokiRTDEControlAdapter


DEFAULT_OUTPUT = Path("data/dmp/all_assembly_motion_dmps.npz")


def parse_args() -> argparse.Namespace:
    """Build an offline full-motion DMP training CLI."""

    parser = motion.build_arg_parser()
    parser.description = (
        "Train one reusable Cartesian DMP for every moveL stage in the complete real "
        "assembly sequence. Gripper and dwell events are stored as discrete events. "
        "No robot or gripper connection is opened."
    )
    parser.set_defaults(
        align_to_four_pin_frame=True,
        insert_after_pin_approach=True,
        no_set_tcp=True,
        execute=False,
    )
    parser.add_argument("--simple-dmp-root", type=Path, default=dmpgen.DEFAULT_SIMPLE_DMP_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dmp-internal-samples", type=int, default=501)
    parser.add_argument("--dmp-basis-functions", type=int, default=60)
    parser.add_argument("--dmp-alpha", type=float, default=48.0)
    parser.add_argument("--dmp-beta", type=float, default=12.0)
    parser.add_argument("--playback-hz", type=float, default=20.0)
    parser.add_argument("--time-scale", type=float, default=4.0)
    parser.add_argument("--trajectory-step-mm", type=float, default=5.0)
    parser.add_argument("--trajectory-rotation-step-deg", type=float, default=3.0)
    parser.add_argument("--rotation-speed-deg-s", type=float, default=30.0)
    args = parser.parse_args()

    if args.execute:
        parser.error("DMP training is offline; --execute is intentionally forbidden.")
    if not args.start_from_initial_pose:
        parser.error("Full-motion DMP training requires --start-from-initial-pose.")
    if args.auto_pin_image_align:
        parser.error("Live image search is unavailable during offline DMP training.")
    continuation_flags = (
        args.continue_pin_sequence_from_current,
        args.continue_post_two_pin_release_from_current,
        args.continue_post_two_pin_after_close_from_current,
        args.continue_post_two_pin_tail_from_current,
        args.continue_post_two_pin_tail_next_from_current,
        args.continue_post_two_pin_tail_extra_from_current,
    )
    if any(continuation_flags):
        parser.error("Current-pose continuation modes cannot train the full sequence.")
    positive = (
        "dmp_internal_samples",
        "dmp_basis_functions",
        "dmp_alpha",
        "dmp_beta",
        "playback_hz",
        "time_scale",
        "trajectory_step_mm",
        "trajectory_rotation_step_deg",
        "rotation_speed_deg_s",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.dmp_internal_samples < 50:
        parser.error("--dmp-internal-samples must be at least 50.")
    if args.dmp_basis_functions >= args.dmp_internal_samples:
        parser.error("--dmp-basis-functions must be smaller than internal samples.")
    return args


def train_move_primitives(
    events: list[preview.MotionEvent],
    initial_tcp_pose_ur: np.ndarray,
    cartesian_dmp_class: type,
    args: argparse.Namespace,
) -> tuple[list[dmpgen.PhaseModel], np.ndarray]:
    """Train each ordered move event and return the event-to-primitive map."""

    models: list[dmpgen.PhaseModel] = []
    event_primitive_index = np.full(len(events), -1, dtype=np.int32)
    current_pose = np.asarray(initial_tcp_pose_ur, dtype=float)
    for event_index, event in enumerate(events):
        if event.kind != "move":
            continue
        assert event.target_pose_ur is not None
        assert event.speed_m_s is not None
        target_pose = np.asarray(event.target_pose_ur, dtype=float)
        sampled_poses = preview.interpolate_move(
            current_pose,
            target_pose,
            float(event.speed_m_s),
            args,
        )
        translation_duration = float(np.linalg.norm(target_pose[:3] - current_pose[:3])) / float(
            event.speed_m_s
        )
        rotation_duration = motion.rotation_delta_deg(current_pose, target_pose) / float(
            args.rotation_speed_deg_s
        )
        seed = dmpgen.PhaseSeed(
            name=f"event_{event_index:02d}:{event.name}",
            poses_ur=np.asarray([current_pose, *sampled_poses], dtype=float),
            duration_s=max(translation_duration, rotation_duration, 0.25),
        )
        model = dmpgen.train_phase_model(seed, cartesian_dmp_class, args)
        event_primitive_index[event_index] = len(models)
        models.append(model)
        current_pose = target_pose
    return models, event_primitive_index


def event_arrays(
    events: list[preview.MotionEvent],
    event_primitive_index: np.ndarray,
) -> dict[str, np.ndarray]:
    """Encode the complete ordered move/discrete event program in NPZ arrays."""

    target_pose_ur = np.full((len(events), 6), np.nan, dtype=float)
    speed_m_s = np.full(len(events), np.nan, dtype=float)
    gripper_percent = np.full(len(events), np.nan, dtype=float)
    dwell_s = np.zeros(len(events), dtype=float)
    for index, event in enumerate(events):
        if event.target_pose_ur is not None:
            target_pose_ur[index] = np.asarray(event.target_pose_ur, dtype=float)
        if event.speed_m_s is not None:
            speed_m_s[index] = float(event.speed_m_s)
        if event.gripper_percent is not None:
            gripper_percent[index] = float(event.gripper_percent)
        dwell_s[index] = float(event.dwell_s)
    return {
        "event_kind": np.asarray([event.kind for event in events]),
        "event_name": np.asarray([event.name for event in events]),
        "event_target_pose_ur": target_pose_ur,
        "event_speed_m_s": speed_m_s,
        "event_gripper_percent": gripper_percent,
        "event_dwell_s": dwell_s,
        "event_tray_action": np.asarray([event.tray_action or "" for event in events]),
        "event_primitive_index": np.asarray(event_primitive_index, dtype=np.int32),
    }


def save_models(
    output_path: Path,
    models: list[dmpgen.PhaseModel],
    events: list[preview.MotionEvent],
    event_primitive_index: np.ndarray,
    initial_q_rad: np.ndarray,
    initial_tcp_pose_ur: np.ndarray,
    simple_dmp_root: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    """Persist reusable DMP weights, seeds, endpoints, and the event program."""

    output = motion.resolve_project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seed_lengths = np.asarray([len(model.seed.poses_ur) for model in models], dtype=np.int64)
    seed_ends = np.cumsum(seed_lengths)
    seed_starts = np.concatenate([np.asarray([0], dtype=np.int64), seed_ends[:-1]])
    arrays = event_arrays(events, event_primitive_index)
    np.savez_compressed(
        output,
        primitive_name=np.asarray([model.seed.name for model in models]),
        primitive_duration_s=np.asarray([model.seed.duration_s for model in models]),
        primitive_start_pose_ur=np.stack([model.seed.poses_ur[0] for model in models]),
        primitive_goal_pose_ur=np.stack([model.seed.poses_ur[-1] for model in models]),
        position_weights=np.stack([model.position_weights for model in models]),
        orientation_weights=np.stack([model.orientation_weights for model in models]),
        seed_starts=seed_starts,
        seed_ends=seed_ends,
        seed_poses_ur=np.concatenate([model.seed.poses_ur for model in models]),
        reproduction_position_rms_m=np.asarray(
            [model.reproduction_position_rms_m for model in models]
        ),
        reproduction_orientation_rms_rad=np.asarray(
            [model.reproduction_orientation_rms_rad for model in models]
        ),
        initial_joint_position_rad=np.asarray(initial_q_rad, dtype=float),
        initial_tcp_pose_ur=np.asarray(initial_tcp_pose_ur, dtype=float),
        tcp_offset_ur=np.asarray(args.tcp_offset_ur, dtype=float),
        dmp_internal_samples=np.asarray(args.dmp_internal_samples, dtype=np.int32),
        dmp_basis_functions=np.asarray(args.dmp_basis_functions, dtype=np.int32),
        dmp_alpha=np.asarray(args.dmp_alpha, dtype=float),
        dmp_beta=np.asarray(args.dmp_beta, dtype=float),
        dmp_cs_alpha=np.asarray(-np.log(0.0001), dtype=float),
        **arrays,
    )

    metadata_path = output.with_suffix(".json")
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "trainer": "scripts/train_all_motion_dmps.py",
        "source_motion": "scripts/marker_based_motion.py complete offline execution path",
        "simple_dmp_root": str(simple_dmp_root),
        "output_npz": str(output),
        "event_count": len(events),
        "move_primitive_count": len(models),
        "discrete_event_count": sum(event.kind != "move" for event in events),
        "weight_reuse": (
            "Load position_weights/orientation_weights once, then change primitive "
            "start/goal before rollout; no goal-specific retraining is required."
        ),
        "dmp": {
            "internal_samples": int(args.dmp_internal_samples),
            "basis_functions": int(args.dmp_basis_functions),
            "alpha": float(args.dmp_alpha),
            "beta": float(args.dmp_beta),
            "cs_alpha": float(-np.log(0.0001)),
        },
        "primitives": [
            {
                "index": index,
                "name": model.seed.name,
                "duration_s": float(model.seed.duration_s),
                "seed_pose_count": len(model.seed.poses_ur),
                "position_rms_m": float(model.reproduction_position_rms_m),
                "orientation_rms_rad": float(model.reproduction_orientation_rms_rad),
            }
            for index, model in enumerate(models)
        ],
        "events": [
            {
                "index": index,
                "kind": event.kind,
                "name": event.name,
                "primitive_index": int(event_primitive_index[index]),
                "gripper_percent": (
                    None if event.gripper_percent is None else float(event.gripper_percent)
                ),
                "dwell_s": float(event.dwell_s),
                "tray_action": event.tray_action,
            }
            for index, event in enumerate(events)
        ],
        "validation": {
            "source_path_solved_with_pyroki": True,
            "saved_weights_reloaded_without_training": False,
            "real_robot_executed": False,
            "collision_checked": False,
            "contact_physics_checked": False,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return output, metadata_path


def reload_reproduction_errors(
    artifact_path: Path,
    cartesian_dmp_class: type,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct every rollout from saved weights without calling train()."""

    with np.load(artifact_path) as data:
        basis_functions = int(data["dmp_basis_functions"])
        alpha = float(data["dmp_alpha"])
        beta = float(data["dmp_beta"])
        cs_alpha = float(data["dmp_cs_alpha"])
        internal_samples = int(data["dmp_internal_samples"])
        starts = np.asarray(data["seed_starts"], dtype=np.int64)
        ends = np.asarray(data["seed_ends"], dtype=np.int64)
        seeds = np.asarray(data["seed_poses_ur"], dtype=float)
        start_poses = np.asarray(data["primitive_start_pose_ur"], dtype=float)
        goal_poses = np.asarray(data["primitive_goal_pose_ur"], dtype=float)
        position_weights = np.asarray(data["position_weights"], dtype=float)
        orientation_weights = np.asarray(data["orientation_weights"], dtype=float)

    normalized_time = np.linspace(0.0, 1.0, internal_samples)
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    for index, (seed_start, seed_end) in enumerate(zip(starts, ends, strict=True)):
        model = cartesian_dmp_class(
            n_bfs=basis_functions,
            alpha=alpha,
            beta=beta,
            cs_alpha=cs_alpha,
        )
        start_pose = start_poses[index]
        goal_pose = goal_poses[index]
        model.position_dmp.w = position_weights[index].copy()
        model.position_dmp.p0 = start_pose[:3].copy()
        model.position_dmp.gp = goal_pose[:3].copy()
        endpoint_quaternions = dmpgen.rotation_to_numpy_quaternion(
            Rotation.from_rotvec(np.stack([start_pose[3:], goal_pose[3:]]))
        )
        model.quaternion_dmp.w = orientation_weights[index].copy()
        model.quaternion_dmp.q0 = endpoint_quaternions[0]
        model.quaternion_dmp.go = endpoint_quaternions[1]
        position, _velocity, _acceleration, orientation, _omega, _d_omega = model.rollout(
            normalized_time,
            1.0,
        )
        reproduced = np.column_stack(
            [position, dmpgen.numpy_quaternion_to_rotation(orientation).as_rotvec()]
        )
        reproduced = dmpgen.enforce_pose_endpoints(reproduced, start_pose, goal_pose)
        reference = dmpgen.resample_pose_sequence(
            seeds[int(seed_start) : int(seed_end)],
            internal_samples,
        )
        position_errors.append(
            float(np.sqrt(np.mean(np.sum((reproduced[:, :3] - reference[:, :3]) ** 2, axis=1))))
        )
        rotation_error = (
            Rotation.from_rotvec(reproduced[:, 3:]).inv() * Rotation.from_rotvec(reference[:, 3:])
        ).magnitude()
        orientation_errors.append(float(np.sqrt(np.mean(rotation_error**2))))
    return np.asarray(position_errors), np.asarray(orientation_errors)


def mark_reload_verified(
    metadata_path: Path,
    position_errors: np.ndarray,
    orientation_errors: np.ndarray,
) -> None:
    metadata = json.loads(metadata_path.read_text())
    metadata["validation"]["saved_weights_reloaded_without_training"] = True
    metadata["reload_verification"] = {
        "maximum_position_rms_m": float(np.max(position_errors)),
        "maximum_orientation_rms_rad": float(np.max(orientation_errors)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    motion.validate_args(args)
    cartesian_dmp_class, simple_dmp_root = dmpgen.load_cartesian_dmp_class(args.simple_dmp_root)
    initial_q_rad = np.deg2rad(np.asarray(args.initial_q_deg, dtype=float))
    print("Initializing offline PyRoki solver; no RTDE connection will be opened...", flush=True)
    ik = PyrokiRTDEControlAdapter(
        initial_q_rad=initial_q_rad,
        tcp_offset_ur=np.asarray(args.tcp_offset_ur, dtype=float),
    )
    initial_q_rad, initial_tcp_pose_ur, _base_t_marker, events = preview.build_actual_motion_events(
        args, ik
    )
    frames = preview.solve_animation_frames(
        args,
        ik,
        initial_q_rad,
        initial_tcp_pose_ur,
        events,
    )
    print(f"Training {sum(event.kind == 'move' for event in events)} moveL DMP primitives...")
    models, event_primitive_index = train_move_primitives(
        events,
        initial_tcp_pose_ur,
        cartesian_dmp_class,
        args,
    )
    for index, model in enumerate(models):
        print(
            f"  [{index:02d}] {model.seed.name}: "
            f"{model.reproduction_position_rms_m * 1000.0:.3f} mm, "
            f"{np.rad2deg(model.reproduction_orientation_rms_rad):.3f} deg RMS"
        )

    output_path, metadata_path = save_models(
        args.output,
        models,
        events,
        event_primitive_index,
        initial_q_rad,
        initial_tcp_pose_ur,
        simple_dmp_root,
        args,
    )
    reload_position_error, reload_orientation_error = reload_reproduction_errors(
        output_path,
        cartesian_dmp_class,
    )
    trained_position_error = np.asarray([model.reproduction_position_rms_m for model in models])
    trained_orientation_error = np.asarray(
        [model.reproduction_orientation_rms_rad for model in models]
    )
    if not np.allclose(reload_position_error, trained_position_error, atol=1e-12, rtol=1e-10):
        raise RuntimeError("Reloaded position DMP errors differ from trained models.")
    if not np.allclose(
        reload_orientation_error,
        trained_orientation_error,
        atol=1e-12,
        rtol=1e-10,
    ):
        raise RuntimeError("Reloaded orientation DMP errors differ from trained models.")
    mark_reload_verified(metadata_path, reload_position_error, reload_orientation_error)
    print(f"Saved all-motion DMP weights: {output_path}")
    print(f"Saved event/model metadata: {metadata_path}")
    print(
        "Reload-without-training verification passed; maximum RMS "
        f"{np.max(reload_position_error) * 1000.0:.3f} mm, "
        f"{np.rad2deg(np.max(reload_orientation_error)):.3f} deg."
    )


if __name__ == "__main__":
    main()
