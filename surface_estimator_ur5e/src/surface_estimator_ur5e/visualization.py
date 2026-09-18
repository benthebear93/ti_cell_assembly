"""Viser visualization for contact measurements and estimated surfaces."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from surface_estimator_ur5e.geometry import PlaneEstimate, make_surface_frame
from surface_estimator_ur5e.io import ContactData, ContactPose
from surface_estimator_ur5e.robot_model import SimpleUR5eVisualizer, URDFRobotVisualizer
from surface_estimator_ur5e.transforms import transform_points as _transform_points


def launch_visualization(
    contact_data: ContactData,
    plane_estimate: PlaneEstimate,
    selected_contact_index: int = 0,
) -> None:
    """Launch a viser server and keep it alive."""

    import viser

    server = viser.ViserServer()
    display_transform = _ceiling_mount_display_transform()
    points_base = contact_data.contact_points()
    points = _transform_points(display_transform, points_base)
    selected = contact_data.contacts[selected_contact_index]
    selected_contact_point = points[selected_contact_index]
    centroid = _transform_point(display_transform, plane_estimate.centroid)
    normal = display_transform[:3, :3] @ plane_estimate.normal
    surface_frame = display_transform @ make_surface_frame(
        plane_estimate.centroid,
        plane_estimate.normal,
    )
    _set_global_workcell_camera(server, points, selected_contact_point)

    add_workcell_context(server, points, selected_contact_point)
    add_frames(server, contact_data.contacts, surface_frame, display_transform=display_transform)
    add_contact_points(server, points, selected_index=selected_contact_index)
    add_plane(server, centroid, normal, size=_plane_size(points))
    add_normal_arrow(server, centroid, normal)

    urdf_visualizer = URDFRobotVisualizer()
    try:
        robot_root_transform = urdf_visualizer.root_transform_for_robot_base(display_transform)
        urdf_visualizer.add_to_scene(
            server,
            selected.q_rad,
            root_name="/ur5e_mesh",
            joint_order=contact_data.robot.joint_order,
            root_transform=robot_root_transform,
        )
        tool0_transform = display_transform @ urdf_visualizer.link_transform_in_robot_base(
            selected.q_rad,
            link_name="tool0",
            joint_order=contact_data.robot.joint_order,
        )
        add_tcp_extension(server, tool0_transform, display_transform @ selected.tcp_transform)
        print("Loaded UR5e mesh from robot_descriptions/yourdfpy.")
    except Exception as exc:
        print(f"warning: URDF mesh visualization unavailable ({exc}); using simple UR5e sketch.")
        SimpleUR5eVisualizer().add_to_scene(
            server,
            selected.q_rad,
            root_name="/ur5e_simple",
            joint_order=contact_data.robot.joint_order,
            root_transform=display_transform,
        )

    print("Viser server is running. Open the printed URL in your browser.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping visualization.")


def add_contact_points(
    server: object,
    points: np.ndarray,
    selected_index: int | None = None,
) -> None:
    """Add contact points as small spheres."""

    scene = _scene(server)
    for index, point in enumerate(points):
        name = f"/contacts/contact_{index + 1}"
        selected = index == selected_index
        radius = 0.04 if selected else 0.025
        color = (250, 190, 40) if selected else (230, 90, 60)
        if hasattr(scene, "add_icosphere"):
            scene.add_icosphere(name, radius=radius, position=point, color=color)
        elif hasattr(scene, "add_point_cloud"):
            scene.add_point_cloud(
                name,
                points=np.asarray([point]),
                colors=np.asarray([color], dtype=np.uint8),
                point_size=2.0 * radius,
            )


def add_workcell_context(
    server: object,
    points: np.ndarray,
    selected_contact_point: np.ndarray,
) -> None:
    """Add ceiling plane, mount marker, and selected pose contact guide."""

    scene = _scene(server)
    size = max(1.4, 2.5 * float(np.max(np.linalg.norm(points[:, :2], axis=1))))
    half = size / 2.0
    z = 0.0
    vertices = np.asarray(
        [
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ],
        dtype=float,
    )
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    if hasattr(scene, "add_mesh_simple"):
        try:
            scene.add_mesh_simple(
                "/workcell/ceiling",
                vertices=vertices,
                faces=faces,
                color=(205, 210, 215),
                opacity=0.32,
                side="double",
            )
        except TypeError:
            scene.add_mesh_simple(
                "/workcell/ceiling",
                vertices=vertices,
                faces=faces,
                color=(205, 210, 215),
            )

    if hasattr(scene, "add_icosphere"):
        scene.add_icosphere("/workcell/base_mount", radius=0.055, position=(0.0, 0.0, 0.0), color=(45, 55, 65))

    if hasattr(scene, "add_line_segments"):
        scene.add_line_segments(
            "/workcell/selected_pose_contact_guide",
            points=np.asarray([[[0.0, 0.0, 0.0], selected_contact_point]], dtype=float),
            colors=np.asarray([[[90, 90, 90], [250, 190, 40]]], dtype=np.uint8),
            line_width=2.0,
        )


def add_tcp_extension(
    server: object,
    tool0_transform: np.ndarray,
    tcp_transform: np.ndarray,
) -> None:
    """Draw the missing gripper/TCP extension from URDF tool0 to measured TCP."""

    scene = _scene(server)
    tool0_position = tool0_transform[:3, 3]
    tcp_position = tcp_transform[:3, 3]
    if hasattr(scene, "add_line_segments"):
        segments = np.asarray(
            [
                [tool0_position, tcp_position],
                [tcp_position + tcp_transform[:3, :3] @ np.array([-0.035, 0.0, -0.02]), tcp_position],
                [tcp_position + tcp_transform[:3, :3] @ np.array([0.035, 0.0, -0.02]), tcp_position],
            ],
            dtype=float,
        )
        scene.add_line_segments(
            "/tooling/tcp_extension",
            points=segments,
            colors=np.tile(np.asarray([30, 30, 30], dtype=np.uint8), (3, 2, 1)),
            line_width=5.0,
        )
    _add_frame(server, "/frames/urdf_tool0", tool0_transform, axes_length=0.07)


def add_plane(
    server: object,
    centroid: np.ndarray,
    normal: np.ndarray,
    size: float = 1.0,
) -> None:
    """Add a transparent rectangular plane patch."""

    scene = _scene(server)
    frame = make_surface_frame(centroid, normal)
    half = size / 2.0
    local_vertices = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ]
    )
    vertices = (frame[:3, :3] @ local_vertices.T).T + frame[:3, 3]
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    color = (80, 170, 150)

    if hasattr(scene, "add_mesh_simple"):
        try:
            scene.add_mesh_simple(
                "/surface/plane",
                vertices=vertices,
                faces=faces,
                color=color,
                opacity=0.35,
                side="double",
            )
            return
        except TypeError:
            scene.add_mesh_simple("/surface/plane", vertices=vertices, faces=faces, color=color)
            return
    if hasattr(scene, "add_mesh_trimesh"):
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        scene.add_mesh_trimesh("/surface/plane", mesh=mesh)


def add_normal_arrow(server: object, centroid: np.ndarray, normal: np.ndarray) -> None:
    """Add a normal vector arrow or line segment."""

    start = np.asarray(centroid, dtype=float)
    end = start + 0.25 * np.asarray(normal, dtype=float)
    scene = _scene(server)
    if hasattr(scene, "add_line_segments"):
        scene.add_line_segments(
            "/surface/normal",
            points=np.asarray([[[*start], [*end]]], dtype=float),
            colors=np.asarray([[[30, 40, 40], [30, 40, 40]]], dtype=np.uint8),
            line_width=4.0,
        )


def add_frames(
    server: object,
    contacts: Iterable[ContactPose],
    surface_frame: np.ndarray,
    display_transform: np.ndarray | None = None,
) -> None:
    """Add robot base, TCP, and surface coordinate frames."""

    display_transform = np.eye(4) if display_transform is None else display_transform
    _add_frame(server, "/frames/robot_base", display_transform, axes_length=0.2)
    for index, contact in enumerate(contacts):
        _add_frame(
            server,
            f"/frames/tcp_{index + 1}",
            display_transform @ contact.tcp_transform,
            axes_length=0.08,
        )
    _add_frame(server, "/frames/surface", surface_frame, axes_length=0.18)


def _add_frame(server: object, name: str, transform: np.ndarray, axes_length: float) -> None:
    scene = _scene(server)
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


def _plane_size(points: np.ndarray) -> float:
    centered = points - points.mean(axis=0)
    extent = float(np.max(np.linalg.norm(centered, axis=1)))
    return max(0.6, 2.5 * extent)


def _scene(server: object) -> object:
    return getattr(server, "scene", server)


def _set_global_workcell_camera(
    server: object,
    points: np.ndarray,
    selected_contact_point: np.ndarray,
) -> None:
    """Set a global initial camera showing ceiling base and contacted surface."""

    center = 0.5 * np.asarray(selected_contact_point, dtype=float)
    center[2] = 0.5 * selected_contact_point[2]
    scene_radius = float(np.max(np.linalg.norm(points - center, axis=1)))
    ceiling_to_surface = abs(float(selected_contact_point[2]))
    distance = max(1.8, 2.2 * scene_radius, 1.6 * ceiling_to_surface)
    position = center + np.array([0.9 * distance, -1.25 * distance, 0.65 * distance])

    if hasattr(server, "initial_camera"):
        server.initial_camera.position = tuple(position)
        server.initial_camera.look_at = tuple(center)
        if hasattr(server.initial_camera, "up"):
            server.initial_camera.up = (0.0, 0.0, 1.0)
        if hasattr(server.initial_camera, "fov"):
            server.initial_camera.fov = 0.9

    if hasattr(server, "on_client_connect"):
        @server.on_client_connect
        def _(client: object) -> None:
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(center)
            if hasattr(client.camera, "up"):
                client.camera.up = (0.0, 0.0, 1.0)


def _ceiling_mount_display_transform() -> np.ndarray:
    """Rotate robot-base coordinates so base +Z-down data appears below the ceiling."""

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("x", 180.0, degrees=True).as_matrix()
    return transform


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ np.asarray(point, dtype=float) + transform[:3, 3]
