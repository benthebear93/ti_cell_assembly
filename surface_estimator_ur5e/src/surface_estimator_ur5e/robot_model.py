"""Robot visualization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_ROBOT_DESCRIPTION = "ur5e_description"
DEFAULT_JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR5E_ACTUATOR_NAMES = tuple(name.removesuffix("_joint") for name in DEFAULT_JOINT_ORDER)


class RobotVisualizerBase(Protocol):
    """Protocol for robot visualizers."""

    def add_to_scene(
        self,
        server: object,
        q_rad: np.ndarray,
        root_name: str = "/robot",
        joint_order: list[str] | None = None,
        root_transform: np.ndarray | None = None,
    ) -> None:
        """Add the robot to a viser scene."""


@dataclass
class SimpleUR5eVisualizer:
    """Simple UR5e fallback visualization using approximate FK and line segments."""

    base_transform: np.ndarray | None = None

    # Standard UR5e DH parameters, adequate for a visual sketch.
    a: tuple[float, ...] = (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0)
    d: tuple[float, ...] = (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996)
    alpha: tuple[float, ...] = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)

    def add_to_scene(
        self,
        server: object,
        q_rad: np.ndarray,
        root_name: str = "/robot",
        joint_order: list[str] | None = None,
        root_transform: np.ndarray | None = None,
    ) -> None:
        """Add an approximate UR5e skeleton and joint frames."""

        q = np.asarray(q_rad, dtype=float)
        if q.shape != (6,):
            raise ValueError("UR5e joint vector must have shape (6,).")

        if root_transform is not None:
            self.base_transform = root_transform
        transforms = self.forward_kinematics(q)
        points = np.vstack([transform[:3, 3] for transform in transforms])

        _add_polyline(server, f"{root_name}/links", points, color=(60, 120, 210), line_width=6.0)
        for index, transform in enumerate(transforms):
            _add_frame(server, f"{root_name}/joint_{index}", transform, axes_length=0.08)
        _add_hande_sketch(server, f"{root_name}/hande", transforms[-1])

    def forward_kinematics(self, q_rad: np.ndarray) -> list[np.ndarray]:
        """Return base, joint, and tool transforms from approximate UR5e DH parameters."""

        transform = np.eye(4) if self.base_transform is None else np.array(self.base_transform)
        transforms = [transform.copy()]
        for theta, a_i, d_i, alpha_i in zip(q_rad, self.a, self.d, self.alpha, strict=True):
            transform = transform @ _dh(theta, a_i, d_i, alpha_i)
            transforms.append(transform.copy())
        return transforms


@dataclass
class URDFRobotVisualizer:
    """URDF mesh visualization using viser and robot_descriptions."""

    urdf_path: Path | None = None
    description_name: str = DEFAULT_ROBOT_DESCRIPTION

    def add_to_scene(
        self,
        server: object,
        q_rad: np.ndarray,
        root_name: str = "/robot",
        joint_order: list[str] | None = None,
        root_transform: np.ndarray | None = None,
    ) -> None:
        """Add UR5e visual meshes and update the actuated joint configuration."""

        from viser.extras import ViserUrdf

        scene = getattr(server, "scene", server)
        if hasattr(scene, "add_frame"):
            root_transform = np.eye(4) if root_transform is None else root_transform
            root_wxyz = Rotation.from_matrix(root_transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
            try:
                scene.add_frame(
                    root_name,
                    wxyz=root_wxyz,
                    position=root_transform[:3, 3],
                    show_axes=False,
                )
            except TypeError:
                scene.add_frame(root_name, wxyz=root_wxyz, position=root_transform[:3, 3])

        urdf_or_path = self.urdf_path if self.urdf_path is not None else self._load_description()
        urdf = ViserUrdf(server, urdf_or_path, root_node_name=root_name)
        cfg = self._configuration_for_urdf(urdf, q_rad, joint_order)
        urdf.update_cfg(cfg)

    def root_transform_for_robot_base(self, display_transform: np.ndarray) -> np.ndarray:
        """Return a root transform that places URDF's ``base`` frame at the robot base."""

        urdf = self._load_description()
        urdf_base_in_world = urdf.get_transform("base", "world")
        return display_transform @ np.linalg.inv(urdf_base_in_world)

    def link_transform_in_robot_base(
        self,
        q_rad: np.ndarray,
        link_name: str = "tool0",
        joint_order: list[str] | None = None,
    ) -> np.ndarray:
        """Compute a URDF link transform expressed in the UR controller base frame."""

        urdf = self._load_description()
        cfg = self._configuration_for_urdf_object(urdf, q_rad, joint_order)
        urdf.update_cfg(cfg)
        return urdf.get_transform(link_name, "base")

    def _load_description(self) -> object:
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        return load_robot_description(self.description_name)

    @staticmethod
    def _configuration_for_urdf(
        urdf: object,
        q_rad: np.ndarray,
        joint_order: list[str] | None,
    ) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (6,):
            raise ValueError("UR5e joint vector must have shape (6,).")

        urdf_joint_names = list(urdf.get_actuated_joint_names())
        if joint_order is None:
            return q[: len(urdf_joint_names)]

        by_name = {name: float(value) for name, value in zip(joint_order, q, strict=True)}
        cfg = np.zeros(len(urdf_joint_names), dtype=float)
        missing: list[str] = []
        for index, name in enumerate(urdf_joint_names):
            if name in by_name:
                cfg[index] = by_name[name]
            else:
                missing.append(name)

        if missing and len(urdf_joint_names) == len(q):
            return q
        if missing:
            print(f"warning: no measured joint values for URDF joints: {', '.join(missing)}")
        return cfg

    @staticmethod
    def _configuration_for_urdf_object(
        urdf: object,
        q_rad: np.ndarray,
        joint_order: list[str] | None,
    ) -> np.ndarray:
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (6,):
            raise ValueError("UR5e joint vector must have shape (6,).")

        urdf_joint_names = list(urdf.actuated_joint_names)
        if joint_order is None:
            return q[: len(urdf_joint_names)]

        by_name = {name: float(value) for name, value in zip(joint_order, q, strict=True)}
        cfg = np.zeros(len(urdf_joint_names), dtype=float)
        missing: list[str] = []
        for index, name in enumerate(urdf_joint_names):
            if name in by_name:
                cfg[index] = by_name[name]
            else:
                missing.append(name)
        if missing and len(urdf_joint_names) == len(q):
            return q
        if missing:
            print(f"warning: no measured joint values for URDF joints: {', '.join(missing)}")
        return cfg


def _dh(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _add_frame(server: object, name: str, transform: np.ndarray, axes_length: float = 0.1) -> None:
    wxyz = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    scene = getattr(server, "scene", server)
    if hasattr(scene, "add_frame"):
        scene.add_frame(name, wxyz=wxyz, position=transform[:3, 3], axes_length=axes_length)


def _add_polyline(
    server: object,
    name: str,
    points: np.ndarray,
    color: tuple[int, int, int],
    line_width: float,
) -> None:
    scene = getattr(server, "scene", server)
    if hasattr(scene, "add_spline_catmull_rom"):
        scene.add_spline_catmull_rom(
            name,
            positions=points,
            color=color,
            line_width=line_width,
        )
    elif hasattr(scene, "add_line_segments"):
        scene.add_line_segments(
            name,
            points=np.stack([points[:-1], points[1:]], axis=1),
            colors=np.tile(np.asarray(color, dtype=np.uint8), (len(points) - 1, 2, 1)),
            line_width=line_width,
        )


def _add_hande_sketch(server: object, name: str, tcp_transform: np.ndarray) -> None:
    """Draw a tiny Robotiq Hand-E style palm and two fingers at the tool frame."""

    local_segments = np.asarray(
        [
            [[-0.04, 0.0, 0.0], [0.04, 0.0, 0.0]],
            [[-0.035, 0.0, 0.0], [-0.035, 0.0, 0.12]],
            [[0.035, 0.0, 0.0], [0.035, 0.0, 0.12]],
        ],
        dtype=float,
    )
    rotation = tcp_transform[:3, :3]
    translation = tcp_transform[:3, 3]
    world_segments = (rotation @ local_segments.reshape(-1, 3).T).T.reshape(-1, 2, 3)
    world_segments = world_segments + translation

    scene = getattr(server, "scene", server)
    if hasattr(scene, "add_line_segments"):
        scene.add_line_segments(
            name,
            points=world_segments,
            colors=np.tile(np.asarray([30, 30, 30], dtype=np.uint8), (3, 2, 1)),
            line_width=4.0,
        )
