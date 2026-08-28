#!/usr/bin/env python3
"""Animate one robot moving a tray from nominal insertion to an offset goal."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import marker_based_motion as motion
import preview_assembly_motion as preview
import preview_tray_dmp_dataset as dmp_viewer
from surface_estimator_ur5e.io import DEFAULT_JOINT_ORDER
from surface_estimator_ur5e.pyroki_ik import PyrokiRTDEControlAdapter, ur_pose_to_transform
from surface_estimator_ur5e.robot_model import URDFRobotVisualizer
from surface_estimator_ur5e.visualization import _ceiling_mount_display_transform
from visualize_marker_frame import (
    add_base_plane,
    add_frame,
    add_marker_square,
    marker_relative_transform,
    transform_points,
    translation_transform,
)


DEFAULT_NOMINAL_DATASET = Path("data/dmp/tray_pick_insert_dmp_initial6.npz")
DEFAULT_OFFSET_DATASET = Path("data/dmp/tray_pick_insert_dmp_xplus20_yminus50.npz")


@dataclass(frozen=True)
class CombinedSequence:
    """One continuous robot/tray animation assembled from two DMP rollouts."""

    frames: tuple[preview.AnimationFrame, ...]
    tray_transforms: tuple[np.ndarray, ...]
    nominal_pin_fit_rms_m: float
    offset_pin_fit_rms_m: float
    attachment_continuity_rms_m: float
    bridge_distance_m: float


def parse_args() -> argparse.Namespace:
    """Build the read-only nominal-to-offset sequence viewer CLI."""

    parser = motion.build_arg_parser()
    parser.description = (
        "Animate one normally colored robot that performs the nominal tray task, "
        "regrasps and lifts the tray at the nominal pins, transfers it to the offset "
        "goal, and inserts/releases it there. No hardware connection is opened."
    )
    parser.set_defaults(no_set_tcp=True, execute=False)
    parser.add_argument("--nominal-dataset", type=Path, default=DEFAULT_NOMINAL_DATASET)
    parser.add_argument("--offset-dataset", type=Path, default=DEFAULT_OFFSET_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument(
        "--playback-hz",
        type=float,
        default=0.0,
        help="Override both dataset sampling rates; 0 reads their JSON metadata.",
    )
    parser.add_argument("--bridge-speed-m-s", type=float, default=0.04)
    parser.add_argument("--bridge-step-mm", type=float, default=2.0)
    parser.add_argument("--regrasp-frames", type=int, default=10)
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
        "--visual-flange-adapter-stl",
        type=Path,
        default=preview.DEFAULT_VISUAL_FLANGE_ADAPTER_STL,
    )
    args = parser.parse_args()
    if args.execute:
        parser.error("The sequence viewer is offline; --execute is forbidden.")
    if args.episode < 0:
        parser.error("--episode cannot be negative.")
    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.playback_hz < 0.0:
        parser.error("--playback-hz cannot be negative.")
    if args.bridge_speed_m_s <= 0.0:
        parser.error("--bridge-speed-m-s must be positive.")
    if args.bridge_step_mm <= 0.0:
        parser.error("--bridge-step-mm must be positive.")
    if args.regrasp_frames < 2:
        parser.error("--regrasp-frames must be at least 2.")
    if not 0.0 < args.visual_tray_handle_scale <= 1.0:
        parser.error("--visual-tray-handle-scale must be > 0 and <= 1.")
    return args


def args_for_dataset(
    args: argparse.Namespace,
    dataset: dmp_viewer.DmpDataset,
) -> argparse.Namespace:
    """Return isolated holder geometry arguments for a dataset's goal offset."""

    result = copy.deepcopy(args)
    result.assembly_origin_x_mm += float(dataset.goal_offset_marker_mm[0])
    result.assembly_origin_y_mm += float(dataset.goal_offset_marker_mm[1])
    result.assembly_origin_z_mm += float(dataset.goal_offset_marker_mm[2])
    return result


def assembly_display_transform(
    args: argparse.Namespace,
    display_t_marker: np.ndarray,
) -> np.ndarray:
    """Compute the display transform of one unmodified holder mesh."""

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
    return display_t_marker @ marker_t_assembly


def box_corners(bounds: np.ndarray) -> np.ndarray:
    """Return the eight corners of an axis-aligned bounding box."""

    lower, upper = np.asarray(bounds, dtype=float)
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=float,
    )


def add_original_holder(
    scene: object,
    obj_path: Path,
    display_t_assembly: np.ndarray,
    root: str,
) -> np.ndarray:
    """Add one holder with its original materials under a unique scene path."""

    import trimesh

    resolved = motion.resolve_project_path(obj_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Holder OBJ not found: {resolved}")
    mesh = trimesh.load(resolved, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected one holder mesh in {resolved}.")
    mesh.apply_transform(display_t_assembly)
    scene.add_mesh_trimesh(f"{root}/mesh", mesh)
    return box_corners(mesh.bounds)


def phase_indices(
    dataset: dmp_viewer.DmpDataset,
    episode_index: int,
    phase_id: int,
) -> np.ndarray:
    """Return global sample indices for one contiguous episode phase."""

    episode = dataset.episode_slice(episode_index)
    local = np.flatnonzero(dataset.phase_id[episode] == phase_id)
    if len(local) == 0:
        raise ValueError(f"Episode {episode_index} has no phase {phase_id}.")
    if not np.array_equal(local, np.arange(local[0], local[-1] + 1)):
        raise ValueError(f"Episode {episode_index} phase {phase_id} is not contiguous.")
    return local + int(episode.start)


def animation_frame(
    q_rad: np.ndarray,
    tcp_pose_ur: np.ndarray,
    stage_name: str,
    gripper_percent: float,
    tray_attached: bool,
) -> preview.AnimationFrame:
    """Create one compact frame for the composed sequence."""

    return preview.AnimationFrame(
        q_rad=np.asarray(q_rad, dtype=float).copy(),
        tcp_pose_ur=np.asarray(tcp_pose_ur, dtype=float).copy(),
        stage_name=stage_name,
        gripper_percent=float(gripper_percent),
        tray_attached=bool(tray_attached),
        base_t_tray_center=np.eye(4),
    )


def interpolate_bridge(
    start_pose_ur: np.ndarray,
    goal_pose_ur: np.ndarray,
    dataset_hz: float,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    """Sample a smooth moveL bridge between nominal and offset pre-insert poses."""

    start = np.asarray(start_pose_ur, dtype=float)
    goal = np.asarray(goal_pose_ur, dtype=float)
    distance_m = float(np.linalg.norm(goal[:3] - start[:3]))
    rotation_deg = motion.rotation_delta_deg(start, goal)
    segments = max(
        1,
        int(np.ceil(distance_m / (args.bridge_step_mm / 1000.0))),
        int(np.ceil(distance_m / args.bridge_speed_m_s * dataset_hz)),
        int(np.ceil(rotation_deg / 2.0)),
    )
    fractions = np.linspace(0.0, 1.0, segments + 1)[1:]
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_rotvec(np.vstack([start[3:], goal[3:]])),
    )(fractions)
    poses: list[np.ndarray] = []
    for fraction, rotation in zip(fractions, rotations, strict=True):
        pose = np.empty(6, dtype=float)
        pose[:3] = (1.0 - fraction) * start[:3] + fraction * goal[:3]
        pose[3:] = rotation.as_rotvec()
        poses.append(pose)
    return poses


def solve_continuous_ik(
    ik: PyrokiRTDEControlAdapter,
    tcp_pose_ur: np.ndarray,
    seed_q_rad: np.ndarray,
    stage_name: str,
) -> np.ndarray:
    """Solve one strict IK sample and reject branch jumps."""

    pose = np.asarray(tcp_pose_ur, dtype=float)
    seed = np.asarray(seed_q_rad, dtype=float)
    if not ik.getInverseKinematicsHasSolution(
        pose.tolist(),
        seed.tolist(),
        motion.IK_POSITION_TOLERANCE_M,
        motion.IK_ORIENTATION_TOLERANCE_RAD,
    ):
        raise RuntimeError(f"PyRoki found no strict IK solution during {stage_name!r}.")
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
        raise RuntimeError(f"IK branch jump of {joint_step_deg:.1f} deg during {stage_name!r}.")
    return q_rad


def build_combined_sequence(
    args: argparse.Namespace,
    nominal: dmp_viewer.DmpDataset,
    offset: dmp_viewer.DmpDataset,
    base_t_marker: np.ndarray,
    tray_mesh: object,
    tray_hole_centers: np.ndarray,
    ik: PyrokiRTDEControlAdapter,
) -> CombinedSequence:
    """Compose nominal task, regrasp/lift, bridge, and offset insertion."""

    episode_index = args.episode
    if episode_index >= min(nominal.episode_count, offset.episode_count):
        raise ValueError(
            f"--episode {episode_index} is outside the common range 0.."
            f"{min(nominal.episode_count, offset.episode_count) - 1}."
        )
    nominal_frames = dmp_viewer.episode_animation_frames(nominal, episode_index)
    offset_frames = dmp_viewer.episode_animation_frames(offset, episode_index)
    nominal_args = args_for_dataset(args, nominal)
    offset_args = args_for_dataset(args, offset)
    nominal_trays, nominal_fit_m = dmp_viewer.tray_transforms_for_episode(
        nominal_args,
        base_t_marker,
        nominal_frames,
        tray_mesh,
        tray_hole_centers,
    )
    offset_trays, offset_fit_m = dmp_viewer.tray_transforms_for_episode(
        offset_args,
        base_t_marker,
        offset_frames,
        tray_mesh,
        tray_hole_centers,
    )

    frames: list[preview.AnimationFrame] = []
    tray_transforms: list[np.ndarray] = []
    nominal_slice = nominal.episode_slice(episode_index)
    nominal_phase_id = nominal.phase_id[nominal_slice]
    for frame, tray_transform, phase_id in zip(
        nominal_frames,
        nominal_trays,
        nominal_phase_id,
        strict=True,
    ):
        frames.append(
            animation_frame(
                frame.q_rad,
                frame.tcp_pose_ur,
                f"nominal / {nominal.phase_names[int(phase_id)]}",
                frame.gripper_percent,
                frame.tray_attached,
            )
        )
        tray_transforms.append(np.asarray(tray_transform, dtype=float).copy())

    final_nominal_frame = frames[-1]
    final_nominal_tray = tray_transforms[-1]
    for fraction in np.linspace(0.0, 1.0, args.regrasp_frames + 1)[1:]:
        frames.append(
            animation_frame(
                final_nominal_frame.q_rad,
                final_nominal_frame.tcp_pose_ur,
                "nominal → offset / regrasp tray",
                100.0 * fraction,
                fraction >= 1.0,
            )
        )
        tray_transforms.append(final_nominal_tray.copy())

    nominal_insert_global = phase_indices(nominal, episode_index, 3)
    nominal_episode_start = int(nominal_slice.start)
    reverse_local_indices = [
        int(index - nominal_episode_start) for index in nominal_insert_global[-2::-1]
    ]
    for local_index in reverse_local_indices:
        source_frame = nominal_frames[local_index]
        frames.append(
            animation_frame(
                source_frame.q_rad,
                source_frame.tcp_pose_ur,
                "nominal → offset / lift tray from nominal pins",
                100.0,
                True,
            )
        )
        tray_transforms.append(np.asarray(nominal_trays[local_index], dtype=float).copy())

    nominal_preinsert_pose = frames[-1].tcp_pose_ur
    nominal_preinsert_tray = tray_transforms[-1]
    tcp_t_tray = (
        np.linalg.inv(ur_pose_to_transform(nominal_preinsert_pose)) @ nominal_preinsert_tray
    )
    offset_insert_global = phase_indices(offset, episode_index, 3)
    offset_episode_start = int(offset.episode_starts[episode_index])
    offset_insert_local = offset_insert_global - offset_episode_start
    offset_preinsert_pose = offset_frames[int(offset_insert_local[0])].tcp_pose_ur
    bridge_poses = interpolate_bridge(
        nominal_preinsert_pose,
        offset_preinsert_pose,
        min(nominal.dataset_hz, offset.dataset_hz),
        args,
    )
    current_q = frames[-1].q_rad
    for pose in bridge_poses:
        current_q = solve_continuous_ik(
            ik,
            pose,
            current_q,
            "nominal-to-offset pre-insert bridge",
        )
        frames.append(
            animation_frame(
                current_q,
                pose,
                "nominal → offset / transfer lifted tray",
                100.0,
                True,
            )
        )
        tray_transforms.append(ur_pose_to_transform(pose) @ tcp_t_tray)

    for local_index in offset_insert_local[1:]:
        source_frame = offset_frames[int(local_index)]
        current_q = solve_continuous_ik(
            ik,
            source_frame.tcp_pose_ur,
            current_q,
            "offset insertion",
        )
        frames.append(
            animation_frame(
                current_q,
                source_frame.tcp_pose_ur,
                "offset / insert",
                100.0,
                True,
            )
        )
        tray_transforms.append(ur_pose_to_transform(source_frame.tcp_pose_ur) @ tcp_t_tray)

    combined_offset_inserted_tray = tray_transforms[-1].copy()
    expected_offset_inserted_tray = np.asarray(
        offset_trays[int(offset_insert_local[-1])],
        dtype=float,
    )
    combined_holes = transform_points(combined_offset_inserted_tray, tray_hole_centers)
    expected_holes = transform_points(expected_offset_inserted_tray, tray_hole_centers)
    attachment_continuity_rms_m = float(
        np.sqrt(np.mean(np.sum((combined_holes - expected_holes) ** 2, axis=1)))
    )
    if attachment_continuity_rms_m > 0.0001:
        raise RuntimeError(
            "The rigid nominal grasp does not match the offset insertion: "
            f"{attachment_continuity_rms_m * 1000.0:.3f} mm RMS."
        )

    offset_release_global = phase_indices(offset, episode_index, 4)
    for global_index in offset_release_global:
        local_index = int(global_index - offset_episode_start)
        source_frame = offset_frames[local_index]
        if source_frame.tray_attached:
            tray_transform = ur_pose_to_transform(source_frame.tcp_pose_ur) @ tcp_t_tray
        else:
            tray_transform = combined_offset_inserted_tray
        frames.append(
            animation_frame(
                current_q,
                source_frame.tcp_pose_ur,
                "offset / release",
                source_frame.gripper_percent,
                source_frame.tray_attached,
            )
        )
        tray_transforms.append(tray_transform.copy())

    if len(frames) != len(tray_transforms):
        raise AssertionError("Frame and tray-transform counts differ.")
    bridge_distance_m = float(
        np.linalg.norm(offset_preinsert_pose[:3] - nominal_preinsert_pose[:3])
    )
    return CombinedSequence(
        frames=tuple(frames),
        tray_transforms=tuple(tray_transforms),
        nominal_pin_fit_rms_m=nominal_fit_m,
        offset_pin_fit_rms_m=offset_fit_m,
        attachment_continuity_rms_m=attachment_continuity_rms_m,
        bridge_distance_m=bridge_distance_m,
    )


def add_motion_path(
    scene: object,
    display_transform: np.ndarray,
    frames: tuple[preview.AnimationFrame, ...],
) -> object:
    """Draw the single composed Cartesian path without recoloring robot geometry."""

    points = transform_points(
        display_transform,
        np.asarray([frame.tcp_pose_ur[:3] for frame in frames], dtype=float),
    )
    segments = np.stack([points[:-1], points[1:]], axis=1)
    colors = np.full((len(segments), 2, 3), 75, dtype=np.uint8)
    return scene.add_line_segments(
        "/combined_motion/path",
        points=segments,
        colors=colors,
        line_width=3.0,
    )


def launch_viewer(
    args: argparse.Namespace,
    nominal: dmp_viewer.DmpDataset,
    offset: dmp_viewer.DmpDataset,
) -> None:
    """Launch the one-robot nominal-to-offset sequence scene."""

    import viser
    from viser.extras import ViserUrdf

    _marker_id, base_t_marker_raw, _marker_data = motion.load_marker_transform(args.marker_pose)
    base_t_marker = (
        motion.floor_constrained_marker_transform(base_t_marker_raw)
        if args.marker_frame_mode == "floor"
        else base_t_marker_raw
    )
    display_transform = _ceiling_mount_display_transform()
    display_t_marker = display_transform @ base_t_marker
    server = viser.ViserServer(port=args.port)
    scene = server.scene
    add_base_plane(server, display_transform)
    add_frame(server, "/frames/robot_base", display_transform, axes_length=0.04)
    add_frame(server, "/frames/marker_floor", display_t_marker, axes_length=0.020)
    add_marker_square(server, "/marker/floor_square", display_t_marker, 0.0254)

    nominal_args = args_for_dataset(args, nominal)
    offset_args = args_for_dataset(args, offset)
    nominal_holder_points = add_original_holder(
        scene,
        Path(args.assembly_obj),
        assembly_display_transform(nominal_args, display_t_marker),
        "/holders/nominal",
    )
    offset_holder_points = add_original_holder(
        scene,
        Path(args.assembly_obj),
        assembly_display_transform(offset_args, display_t_marker),
        "/holders/offset",
    )

    initial_q_rad = np.asarray(
        nominal.joint_position_rad[nominal.episode_starts[args.episode]],
        dtype=float,
    )
    ik = PyrokiRTDEControlAdapter(
        initial_q_rad=initial_q_rad,
        tcp_offset_ur=np.asarray(args.tcp_offset_ur, dtype=float),
    )
    visualizer = URDFRobotVisualizer()
    tray_mesh, tray_hole_centers, resolved_tray_path = preview.load_visual_tray_mesh(
        Path(args.visual_tray_stl)
    )
    preview.shorten_lower_tray_handle(
        tray_mesh,
        args.visual_tray_handle_root_y_mm / 1000.0,
        args.visual_tray_handle_scale,
    )
    sequence = build_combined_sequence(
        args,
        nominal,
        offset,
        base_t_marker,
        tray_mesh,
        tray_hole_centers,
        ik,
    )

    robot_root_transform = visualizer.root_transform_for_robot_base(display_transform)
    scene.add_frame(
        "/animated_robot",
        position=robot_root_transform[:3, 3],
        wxyz=Rotation.from_matrix(robot_root_transform[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    viser_urdf = ViserUrdf(server, ik.urdf, root_node_name="/animated_robot")

    first_frame = sequence.frames[0]
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

    first_display_t_tray = display_transform @ sequence.tray_transforms[0]
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

    path_handle = add_motion_path(scene, display_transform, sequence.frames)
    display_path_points = transform_points(
        display_transform,
        np.asarray([frame.tcp_pose_ur[:3] for frame in sequence.frames]),
    )
    dmp_viewer.set_dmp_camera(
        server,
        np.vstack([nominal_holder_points, offset_holder_points, display_path_points]),
    )

    play = server.gui.add_checkbox("Play", initial_value=True)
    loop = server.gui.add_checkbox("Loop", initial_value=not args.no_loop)
    show_path = server.gui.add_checkbox("Motion path", initial_value=True)
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
        max=len(sequence.frames) - 1,
        step=1,
        initial_value=0,
    )
    stage_status = server.gui.add_text("Motion stage", initial_value=first_frame.stage_name)
    gripper_status = server.gui.add_text(
        "Gripper", initial_value=f"{first_frame.gripper_percent:.1f}% closed"
    )
    tray_status = server.gui.add_text("Tray", initial_value="waiting at marker")

    @show_path.on_update
    def _toggle_path(_event: object) -> None:
        path_handle.visible = bool(show_path.value)

    actual_port = server.get_port()
    print(f"Sequential nominal-to-offset viewer ready: http://localhost:{actual_port}")
    print(f"Nominal dataset: {nominal.path}")
    print(f"Offset dataset:  {offset.path}")
    print(f"Goal offset: {offset.goal_offset_marker_mm.tolist()} mm")
    print(f"Combined sequence: {len(sequence.frames)} frames, episode {args.episode}")
    print(f"Nominal-to-offset pre-insert bridge: {sequence.bridge_distance_m * 1000.0:.3f} mm")
    print(
        "Tray fit RMS: nominal "
        f"{sequence.nominal_pin_fit_rms_m * 1000.0:.6f} mm, offset "
        f"{sequence.offset_pin_fit_rms_m * 1000.0:.6f} mm"
    )
    print(
        "Rigid grasp continuity at offset: "
        f"{sequence.attachment_continuity_rms_m * 1000.0:.6f} mm RMS"
    )
    print(f"Visual tray: {resolved_tray_path}")
    print(f"Visual flange adapter: {resolved_adapter_path}")
    print("No RTDE or gripper connection was opened. Press Ctrl+C to stop.")

    frame_index = 0
    playback_hz = min(nominal.dataset_hz, offset.dataset_hz)
    try:
        while True:
            if not play.value:
                frame_index = min(int(frame_slider.value), len(sequence.frames) - 1)
            frame = sequence.frames[frame_index]
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
            display_t_tray = display_transform @ sequence.tray_transforms[frame_index]
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
                tray_root.position = display_t_tray[:3, 3]
                tray_root.wxyz = Rotation.from_matrix(display_t_tray[:3, :3]).as_quat()[
                    [3, 0, 1, 2]
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
                if frame.tray_attached:
                    tray_status.value = "attached to TCP"
                elif frame.stage_name == "offset / release":
                    tray_status.value = "seated at offset goal"
                elif frame.stage_name in {
                    "nominal / release",
                    "nominal → offset / regrasp tray",
                }:
                    tray_status.value = "seated at nominal pins"
                else:
                    tray_status.value = "waiting at marker"
                if play.value:
                    frame_slider.value = frame_index

            time.sleep(1.0 / (playback_hz * float(speed.value)))
            if play.value:
                if frame_index + 1 < len(sequence.frames):
                    frame_index += 1
                elif loop.value:
                    frame_index = 0
                else:
                    play.value = False
    except KeyboardInterrupt:
        print("Stopping sequential nominal-to-offset viewer.")


def main() -> None:
    args = parse_args()
    nominal = dmp_viewer.load_dataset(args.nominal_dataset, args.playback_hz)
    offset = dmp_viewer.load_dataset(args.offset_dataset, args.playback_hz)
    launch_viewer(args, nominal, offset)


if __name__ == "__main__":
    main()
