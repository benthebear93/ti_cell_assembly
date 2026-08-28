#!/usr/bin/env python3
"""Inspect and animate generated tray DMP demonstrations in Viser."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

import marker_based_motion as motion
import preview_assembly_motion as preview
from surface_estimator_ur5e.io import DEFAULT_JOINT_ORDER
from surface_estimator_ur5e.pyroki_ik import PyrokiRTDEControlAdapter, ur_pose_to_transform
from surface_estimator_ur5e.robot_model import URDFRobotVisualizer
from surface_estimator_ur5e.visualization import _ceiling_mount_display_transform
from visualize_marker_frame import (
    add_base_plane,
    add_frame,
    add_marker_square,
    add_ti_assembly_mesh,
    marker_relative_transform,
    transform_points,
    translation_transform,
)


DEFAULT_DATASET = Path("data/dmp/tray_pick_insert_dmp_smoke8.npz")
PHASE_COLORS = np.asarray(
    [
        [40, 120, 250],
        [245, 180, 40],
        [40, 190, 90],
        [235, 65, 55],
        [185, 75, 220],
    ],
    dtype=np.uint8,
)
FALLBACK_PHASE_NAMES = (
    "approach_grasp",
    "grasp_close",
    "transfer_to_preinsert",
    "insert",
    "release",
)
PHASE_STAGE_NAMES = (
    "marker grasp moveL",
    "marker target close",
    "post-grasp transfer",
    "final pin insertion descend",
    "post-insert open",
)


@dataclass(frozen=True)
class DmpDataset:
    """Validated arrays required by the interactive viewer."""

    path: Path
    tcp_pose_ur: np.ndarray
    joint_position_rad: np.ndarray
    gripper_percent: np.ndarray
    phase_id: np.ndarray
    object_attached: np.ndarray
    episode_starts: np.ndarray
    episode_ends: np.ndarray
    dataset_hz: float
    phase_names: tuple[str, ...]
    goal_offset_marker_mm: np.ndarray

    @property
    def episode_count(self) -> int:
        return len(self.episode_ends)

    @property
    def maximum_episode_length(self) -> int:
        return int(np.max(self.episode_ends - self.episode_starts))

    def episode_slice(self, index: int) -> slice:
        return slice(int(self.episode_starts[index]), int(self.episode_ends[index]))


def parse_args() -> argparse.Namespace:
    """Extend real workcell geometry options with DMP viewer controls."""

    parser = motion.build_arg_parser()
    parser.description = (
        "Animate a generated tray DMP dataset in Viser. This command is read-only and "
        "never connects to the robot or gripper."
    )
    parser.set_defaults(no_set_tcp=True, execute=False)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-loop", action="store_true")
    parser.add_argument(
        "--playback-hz",
        type=float,
        default=0.0,
        help="Override dataset sampling rate; 0 reads dataset JSON metadata.",
    )
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
    parser.add_argument(
        "--goal-offset-marker-mm",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help=(
            "Override the dataset metadata's fixed marker-frame goal/holder offset. "
            "Normally this is read automatically."
        ),
    )
    args = parser.parse_args()
    if args.execute:
        parser.error("The DMP viewer is offline; --execute is intentionally forbidden.")
    if not 0 < args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    if args.playback_hz < 0.0:
        parser.error("--playback-hz cannot be negative.")
    if not 0.0 < args.visual_tray_handle_scale <= 1.0:
        parser.error("--visual-tray-handle-scale must be > 0 and <= 1.")
    if args.goal_offset_marker_mm is not None and not np.all(
        np.isfinite(args.goal_offset_marker_mm)
    ):
        parser.error("--goal-offset-marker-mm must contain three finite values.")
    return args


def load_dataset(dataset_path: Path, playback_hz_override: float) -> DmpDataset:
    """Load the generator NPZ and reject malformed episode boundaries."""

    resolved_path = motion.resolve_project_path(dataset_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"DMP dataset not found: {resolved_path}")
    with np.load(resolved_path) as loaded:
        required = (
            "tcp_pose_ur",
            "joint_position_rad",
            "gripper_percent",
            "phase_id",
            "object_attached",
            "episode_starts",
            "episode_ends",
        )
        missing = [name for name in required if name not in loaded]
        if missing:
            raise ValueError(f"DMP dataset is missing arrays: {missing}")
        arrays = {name: np.asarray(loaded[name]).copy() for name in required}

    tcp_pose_ur = np.asarray(arrays["tcp_pose_ur"], dtype=float)
    joint_position_rad = np.asarray(arrays["joint_position_rad"], dtype=float)
    gripper_percent = np.asarray(arrays["gripper_percent"], dtype=float)
    phase_id = np.asarray(arrays["phase_id"], dtype=np.int16)
    object_attached = np.asarray(arrays["object_attached"], dtype=bool)
    episode_starts = np.asarray(arrays["episode_starts"], dtype=np.int64)
    episode_ends = np.asarray(arrays["episode_ends"], dtype=np.int64)
    sample_count = len(tcp_pose_ur)
    if tcp_pose_ur.shape != (sample_count, 6):
        raise ValueError(f"Expected tcp_pose_ur (N, 6), got {tcp_pose_ur.shape}.")
    if joint_position_rad.shape != (sample_count, 6):
        raise ValueError(f"Expected joint_position_rad (N, 6), got {joint_position_rad.shape}.")
    for name, array in (
        ("gripper_percent", gripper_percent),
        ("phase_id", phase_id),
        ("object_attached", object_attached),
    ):
        if array.shape != (sample_count,):
            raise ValueError(f"Expected {name} (N,), got {array.shape}.")
    if episode_starts.ndim != 1 or episode_ends.shape != episode_starts.shape:
        raise ValueError("Episode starts and ends must be same-length 1D arrays.")
    if len(episode_ends) == 0:
        raise ValueError("DMP dataset contains no episodes.")
    expected_starts = np.concatenate([np.asarray([0], dtype=np.int64), episode_ends[:-1]])
    if not np.array_equal(episode_starts, expected_starts):
        raise ValueError("Episode starts are not contiguous with prior exclusive ends.")
    if int(episode_ends[-1]) != sample_count or np.any(episode_ends <= episode_starts):
        raise ValueError("Invalid episode boundaries in DMP dataset.")
    if not all(
        np.all(np.isfinite(array)) for array in (tcp_pose_ur, joint_position_rad, gripper_percent)
    ):
        raise ValueError("DMP dataset contains non-finite state values.")

    metadata_path = resolved_path.with_suffix(".json")
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    metadata_hz = float(metadata.get("dataset_hz", 20.0))
    dataset_hz = playback_hz_override if playback_hz_override > 0.0 else metadata_hz
    if not np.isfinite(dataset_hz) or dataset_hz <= 0.0:
        raise ValueError(f"Dataset playback rate must be positive, got {dataset_hz}.")
    phase_names_value = metadata.get("phase_names", FALLBACK_PHASE_NAMES)
    phase_names = tuple(str(value) for value in phase_names_value)
    maximum_phase = int(np.max(phase_id))
    if (
        maximum_phase >= len(phase_names)
        or maximum_phase >= len(PHASE_STAGE_NAMES)
        or int(np.min(phase_id)) < 0
    ):
        raise ValueError(
            f"phase_id range [{int(np.min(phase_id))}, {maximum_phase}] does not "
            f"match {len(phase_names)} phase names."
        )
    goal_retargeting = metadata.get("goal_retargeting", {})
    goal_offset_marker_mm = np.zeros(3, dtype=float)
    if isinstance(goal_retargeting, dict):
        goal_offset_marker_mm = np.asarray(
            goal_retargeting.get("marker_offset_mm", goal_offset_marker_mm),
            dtype=float,
        )
    if goal_offset_marker_mm.shape != (3,) or not np.all(np.isfinite(goal_offset_marker_mm)):
        raise ValueError("Dataset goal_retargeting.marker_offset_mm must contain 3 values.")
    return DmpDataset(
        path=resolved_path,
        tcp_pose_ur=tcp_pose_ur,
        joint_position_rad=joint_position_rad,
        gripper_percent=gripper_percent,
        phase_id=phase_id,
        object_attached=object_attached,
        episode_starts=episode_starts,
        episode_ends=episode_ends,
        dataset_hz=dataset_hz,
        phase_names=phase_names,
        goal_offset_marker_mm=goal_offset_marker_mm,
    )


def episode_animation_frames(
    dataset: DmpDataset,
    episode_index: int,
) -> list[preview.AnimationFrame]:
    """Adapt one saved episode to the tray transform helper's frame contract."""

    episode = dataset.episode_slice(episode_index)
    tcp_pose_ur = dataset.tcp_pose_ur[episode]
    joint_position_rad = dataset.joint_position_rad[episode]
    gripper_percent = dataset.gripper_percent[episode]
    phase_id = dataset.phase_id[episode]
    attached = dataset.object_attached[episode]
    identity = np.eye(4)
    return [
        preview.AnimationFrame(
            q_rad=np.asarray(q, dtype=float),
            tcp_pose_ur=np.asarray(pose, dtype=float),
            stage_name=PHASE_STAGE_NAMES[int(phase)],
            gripper_percent=float(gripper),
            tray_attached=bool(is_attached),
            base_t_tray_center=identity.copy(),
        )
        for q, pose, gripper, phase, is_attached in zip(
            joint_position_rad,
            tcp_pose_ur,
            gripper_percent,
            phase_id,
            attached,
            strict=True,
        )
    ]


def add_episode_paths(
    scene: object,
    display_transform: np.ndarray,
    dataset: DmpDataset,
) -> tuple[list[object], list[object]]:
    """Draw thin all-episode paths and thick phase-colored selection paths."""

    all_handles: list[object] = []
    selected_handles: list[object] = []
    for episode_index in range(dataset.episode_count):
        episode = dataset.episode_slice(episode_index)
        points = transform_points(display_transform, dataset.tcp_pose_ur[episode, :3])
        phase_id = dataset.phase_id[episode]
        segments = np.stack([points[:-1], points[1:]], axis=1)
        gray = np.full((len(segments), 2, 3), 80, dtype=np.uint8)
        all_handle = scene.add_line_segments(
            f"/dmp_paths/all/episode_{episode_index:04d}",
            points=segments,
            colors=gray,
            line_width=1.75,
        )
        all_handles.append(all_handle)

        segment_colors = PHASE_COLORS[np.clip(phase_id[:-1], 0, len(PHASE_COLORS) - 1)]
        selected_handle = scene.add_line_segments(
            f"/dmp_paths/selected/episode_{episode_index:04d}",
            points=segments,
            colors=np.repeat(segment_colors[:, None, :], 2, axis=1),
            line_width=5.0,
            visible=episode_index == 0,
        )
        selected_handles.append(selected_handle)
    return all_handles, selected_handles


def episode_start_delta(dataset: DmpDataset, episode_index: int) -> tuple[float, float, float]:
    """Return max joint, TCP translation, and TCP rotation deltas from episode zero."""

    nominal_index = int(dataset.episode_starts[0])
    episode_start = int(dataset.episode_starts[episode_index])
    joint_delta_deg = float(
        np.max(
            np.abs(
                np.rad2deg(
                    dataset.joint_position_rad[episode_start]
                    - dataset.joint_position_rad[nominal_index]
                )
            )
        )
    )
    nominal_tcp = dataset.tcp_pose_ur[nominal_index]
    episode_tcp = dataset.tcp_pose_ur[episode_start]
    translation_delta_mm = float(np.linalg.norm(episode_tcp[:3] - nominal_tcp[:3]) * 1000.0)
    rotation_delta_deg = float(
        np.rad2deg(
            (
                Rotation.from_rotvec(nominal_tcp[3:]).inv() * Rotation.from_rotvec(episode_tcp[3:])
            ).magnitude()
        )
    )
    return joint_delta_deg, translation_delta_mm, rotation_delta_deg


def tray_transforms_for_episode(
    args: argparse.Namespace,
    base_t_marker: np.ndarray,
    frames: list[preview.AnimationFrame],
    tray_mesh: object,
    tray_hole_centers: np.ndarray,
) -> tuple[list[np.ndarray], float]:
    """Compute the rigid grasped tray transforms and return pin-fit RMS."""

    transforms, _pins, fit_rms_m, _axial_shift, _plane_error = preview.visual_tray_transforms(
        args,
        base_t_marker,
        frames,
        np.asarray(tray_mesh.vertices, dtype=float),
        tray_hole_centers,
    )
    return transforms, float(fit_rms_m)


def set_dmp_camera(server: object, points: np.ndarray) -> None:
    """Frame the compact assembly task more tightly than the full-workcell view."""

    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    direction = np.asarray([0.7, -1.0, 0.55], dtype=float)
    direction /= np.linalg.norm(direction)
    position = center + direction * max(0.45, 3.0 * radius)

    if hasattr(server, "initial_camera"):
        server.initial_camera.position = tuple(position)
        server.initial_camera.look_at = tuple(center)
        if hasattr(server.initial_camera, "up"):
            server.initial_camera.up = (0.0, 0.0, 1.0)

    if hasattr(server, "on_client_connect"):

        @server.on_client_connect
        def _set_client_camera(client: object) -> None:
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(center)
            if hasattr(client.camera, "up"):
                client.camera.up = (0.0, 0.0, 1.0)


def launch_viewer(
    args: argparse.Namespace,
    dataset: DmpDataset,
) -> None:
    """Launch the interactive multi-episode Viser scene."""

    import viser
    from viser.extras import ViserUrdf

    _marker_id, base_t_marker_raw, _marker_data = motion.load_marker_transform(args.marker_pose)
    base_t_marker = (
        motion.floor_constrained_marker_transform(base_t_marker_raw)
        if args.marker_frame_mode == "floor"
        else base_t_marker_raw
    )
    goal_offset_marker_mm = (
        dataset.goal_offset_marker_mm
        if args.goal_offset_marker_mm is None
        else np.asarray(args.goal_offset_marker_mm, dtype=float)
    )
    args.assembly_origin_x_mm += float(goal_offset_marker_mm[0])
    args.assembly_origin_y_mm += float(goal_offset_marker_mm[1])
    args.assembly_origin_z_mm += float(goal_offset_marker_mm[2])
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

    initial_q_rad = np.asarray(dataset.joint_position_rad[0], dtype=float)
    ik = PyrokiRTDEControlAdapter(
        initial_q_rad=initial_q_rad,
        tcp_offset_ur=np.asarray(args.tcp_offset_ur, dtype=float),
    )
    visualizer = URDFRobotVisualizer()
    root_transform = visualizer.root_transform_for_robot_base(display_transform)
    scene.add_frame(
        "/animated_robot",
        position=root_transform[:3, 3],
        wxyz=Rotation.from_matrix(root_transform[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    viser_urdf = ViserUrdf(server, ik.urdf, root_node_name="/animated_robot")

    first_tcp_pose = np.asarray(dataset.tcp_pose_ur[0], dtype=float)
    first_display_t_tcp = display_transform @ ur_pose_to_transform(first_tcp_pose)
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

    first_base_t_adapter = preview.base_t_hande_mount(first_tcp_pose, args.tcp_offset_ur)
    first_display_t_adapter = display_transform @ first_base_t_adapter
    flange_adapter_root = scene.add_frame(
        "/animation/flange_camera_adapter",
        position=first_display_t_adapter[:3, 3],
        wxyz=Rotation.from_matrix(first_display_t_adapter[:3, :3]).as_quat()[[3, 0, 1, 2]],
        show_axes=False,
    )
    flange_adapter_mesh, resolved_adapter_path = preview.load_preview_flange_adapter_mesh(
        Path(args.visual_flange_adapter_stl)
    )
    scene.add_mesh_trimesh(
        "/animation/flange_camera_adapter/mesh",
        flange_adapter_mesh,
    )

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

    tray_mesh, tray_hole_centers, resolved_tray_path = preview.load_visual_tray_mesh(
        Path(args.visual_tray_stl)
    )
    preview.shorten_lower_tray_handle(
        tray_mesh,
        args.visual_tray_handle_root_y_mm / 1000.0,
        args.visual_tray_handle_scale,
    )
    current_frames = episode_animation_frames(dataset, 0)
    current_tray_transforms, tray_fit_rms_m = tray_transforms_for_episode(
        args,
        base_t_marker,
        current_frames,
        tray_mesh,
        tray_hole_centers,
    )
    first_display_t_tray = display_transform @ current_tray_transforms[0]
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

    all_path_handles, selected_path_handles = add_episode_paths(
        scene,
        display_transform,
        dataset,
    )
    episode_start_points = transform_points(
        display_transform,
        dataset.tcp_pose_ur[dataset.episode_starts, :3],
    )
    start_colors = np.repeat(
        np.asarray([[20, 175, 220]], dtype=np.uint8),
        dataset.episode_count,
        axis=0,
    )
    start_points_handle = scene.add_point_cloud(
        "/dmp_paths/initial_tcp_points",
        points=episode_start_points,
        colors=start_colors,
        point_size=0.009,
        point_shape="circle",
    )
    display_path_points = transform_points(display_transform, dataset.tcp_pose_ur[:, :3])
    set_dmp_camera(server, np.vstack([assembly_points, display_path_points]))

    play = server.gui.add_checkbox("Play", initial_value=True)
    loop = server.gui.add_checkbox("Loop", initial_value=not args.no_loop)
    show_all_paths = server.gui.add_checkbox("All DMP paths", initial_value=True)
    show_selected_path = server.gui.add_checkbox("Selected path", initial_value=True)
    show_initial_tcps = server.gui.add_checkbox("Initial TCPs", initial_value=True)
    episode_selector = server.gui.add_dropdown(
        "Episode",
        options=[str(index) for index in range(dataset.episode_count)],
        initial_value="0",
    )
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
        max=dataset.maximum_episode_length - 1,
        step=1,
        initial_value=0,
    )
    episode_status = server.gui.add_text(
        "Episode info",
        initial_value=f"0 / {dataset.episode_count - 1} ({len(current_frames)} frames)",
    )
    initial_joint_delta_deg, initial_tcp_delta_mm, initial_tcp_delta_deg = episode_start_delta(
        dataset, 0
    )
    start_joint_status = server.gui.add_text(
        "Start joint",
        initial_value=f"max |dq| {initial_joint_delta_deg:.1f} deg",
    )
    start_tcp_status = server.gui.add_text(
        "Start TCP",
        initial_value=f"{initial_tcp_delta_mm:.1f} mm / {initial_tcp_delta_deg:.1f} deg",
    )
    phase_status = server.gui.add_text(
        "Phase",
        initial_value=dataset.phase_names[int(dataset.phase_id[0])],
    )
    gripper_status = server.gui.add_text(
        "Gripper",
        initial_value=f"{dataset.gripper_percent[0]:.1f}% closed",
    )
    tray_status = server.gui.add_text("Tray", initial_value="waiting at marker")

    @show_all_paths.on_update
    def _toggle_all_paths(_event: object) -> None:
        with server.atomic():
            for handle in all_path_handles:
                handle.visible = bool(show_all_paths.value)

    current_episode_index = 0

    @show_selected_path.on_update
    def _toggle_selected_path(_event: object) -> None:
        with server.atomic():
            for index, handle in enumerate(selected_path_handles):
                handle.visible = bool(show_selected_path.value) and index == current_episode_index

    @show_initial_tcps.on_update
    def _toggle_initial_tcps(_event: object) -> None:
        start_points_handle.visible = bool(show_initial_tcps.value)

    actual_port = server.get_port()
    print(f"DMP dataset viewer ready: http://localhost:{actual_port}")
    print(
        f"Dataset: {dataset.path} ({dataset.episode_count} episodes, "
        f"{len(dataset.tcp_pose_ur)} samples at {dataset.dataset_hz:.1f} Hz)"
    )
    print(f"Visual tray: {resolved_tray_path}")
    print(f"Visual flange adapter: {resolved_adapter_path}")
    print(f"Goal/holder marker offset: {goal_offset_marker_mm.tolist()} mm")
    print(f"Tray hole-to-pin fit: {tray_fit_rms_m * 1000.0:.6f} mm RMS")
    print("No RTDE or gripper connection was opened. Press Ctrl+C to stop.")

    frame_index = 0
    try:
        while True:
            requested_episode = int(episode_selector.value)
            if requested_episode != current_episode_index:
                current_episode_index = requested_episode
                current_frames = episode_animation_frames(dataset, current_episode_index)
                current_tray_transforms, _fit = tray_transforms_for_episode(
                    args,
                    base_t_marker,
                    current_frames,
                    tray_mesh,
                    tray_hole_centers,
                )
                frame_index = 0
                frame_slider.value = 0
                episode_status.value = (
                    f"{current_episode_index} / {dataset.episode_count - 1} "
                    f"({len(current_frames)} frames)"
                )
                joint_delta_deg, tcp_delta_mm, tcp_delta_deg = episode_start_delta(
                    dataset,
                    current_episode_index,
                )
                start_joint_status.value = f"max |dq| {joint_delta_deg:.1f} deg"
                start_tcp_status.value = f"{tcp_delta_mm:.1f} mm / {tcp_delta_deg:.1f} deg"
                with server.atomic():
                    for index, handle in enumerate(selected_path_handles):
                        handle.visible = (
                            bool(show_selected_path.value) and index == current_episode_index
                        )

            if not play.value:
                frame_index = min(int(frame_slider.value), len(current_frames) - 1)
            frame = current_frames[frame_index]
            cfg = visualizer._configuration_for_urdf_object(
                ik.urdf,
                frame.q_rad,
                list(DEFAULT_JOINT_ORDER),
            )
            display_t_tcp = display_transform @ ur_pose_to_transform(frame.tcp_pose_ur)
            base_t_adapter = preview.base_t_hande_mount(
                frame.tcp_pose_ur,
                args.tcp_offset_ur,
            )
            display_t_adapter = display_transform @ base_t_adapter
            base_t_hande = base_t_adapter @ translation_transform(
                [0.0, 0.0, preview.FLANGE_ADAPTER_STACK_HEIGHT_M]
            )
            display_t_hande = display_transform @ base_t_hande
            display_t_tray = display_transform @ current_tray_transforms[frame_index]
            tcp_wxyz = Rotation.from_matrix(display_t_tcp[:3, :3]).as_quat()[[3, 0, 1, 2]]
            adapter_wxyz = Rotation.from_matrix(display_t_adapter[:3, :3]).as_quat()[[3, 0, 1, 2]]
            hande_wxyz = Rotation.from_matrix(display_t_hande[:3, :3]).as_quat()[[3, 0, 1, 2]]
            tray_wxyz = Rotation.from_matrix(display_t_tray[:3, :3]).as_quat()[[3, 0, 1, 2]]
            finger_slide_m = preview.HANDE_FINGER_RANGE_M * frame.gripper_percent / 100.0
            finger_x_m = preview.HANDE_FINGER_OFFSET_M - finger_slide_m
            phase_index = int(
                dataset.phase_id[dataset.episode_slice(current_episode_index)][frame_index]
            )
            with server.atomic():
                viser_urdf.update_cfg(cfg)
                tcp_handle.position = display_t_tcp[:3, 3]
                tcp_handle.wxyz = tcp_wxyz
                tcp_marker.position = display_t_tcp[:3, 3]
                flange_adapter_root.position = display_t_adapter[:3, 3]
                flange_adapter_root.wxyz = adapter_wxyz
                hande_root.position = display_t_hande[:3, 3]
                hande_root.wxyz = hande_wxyz
                tray_root.position = display_t_tray[:3, 3]
                tray_root.wxyz = tray_wxyz
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
                phase_status.value = dataset.phase_names[phase_index]
                gripper_status.value = f"{frame.gripper_percent:.1f}% closed"
                if phase_index == 0:
                    tray_status.value = "waiting at marker"
                elif frame.tray_attached:
                    tray_status.value = "attached to TCP"
                else:
                    tray_status.value = "seated on four pins"
                if play.value:
                    frame_slider.value = frame_index

            time.sleep(1.0 / (dataset.dataset_hz * float(speed.value)))
            if play.value:
                if frame_index + 1 < len(current_frames):
                    frame_index += 1
                elif loop.value:
                    frame_index = 0
                else:
                    play.value = False
    except KeyboardInterrupt:
        print("Stopping DMP dataset viewer.")


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset, args.playback_hz)
    launch_viewer(args, dataset)


if __name__ == "__main__":
    main()
