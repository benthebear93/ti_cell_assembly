"""YAML input parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from surface_estimator_ur5e.geometry import compute_contact_point, estimate_plane_from_points


@dataclass(frozen=True)
class RobotConfig:
    """Robot metadata from the input file."""

    name: str = "ur5e_hande"
    base_frame: str = "robot_base"
    tool_frame: str = "tcp"
    pose_convention: str = "xyz_quat_xyzw"
    joint_order: list[str] = field(default_factory=lambda: list(DEFAULT_JOINT_ORDER))


@dataclass(frozen=True)
class ToolConfig:
    """Tool/contact metadata from the input file."""

    contact_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass(frozen=True)
class ContactPose:
    """One measured contact pose."""

    name: str
    q_rad: np.ndarray
    tcp_position_m: np.ndarray
    tcp_orientation_xyzw: np.ndarray
    note: str = ""

    @property
    def tcp_transform(self) -> np.ndarray:
        """Return the 4x4 transform of this TCP pose."""

        from surface_estimator_ur5e.geometry import quaternion_xyzw_to_matrix

        transform = np.eye(4)
        transform[:3, :3] = quaternion_xyzw_to_matrix(self.tcp_orientation_xyzw)
        transform[:3, 3] = self.tcp_position_m
        return transform


@dataclass(frozen=True)
class ContactData:
    """Validated contact file contents."""

    robot: RobotConfig
    tool: ToolConfig
    contacts: list[ContactPose]
    normal_direction: str = "auto"
    normal_direction_vector: np.ndarray | None = None
    flip_normal: bool = False

    def contact_points(self) -> np.ndarray:
        """Return all physical contact points in the robot base frame."""

        return np.vstack(
            [
                compute_contact_point(
                    contact.tcp_position_m,
                    contact.tcp_orientation_xyzw,
                    self.tool.contact_offset_m,
                )
                for contact in self.contacts
            ]
        )


DEFAULT_JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def load_contact_file(path: str | Path) -> ContactData:
    """Load and validate a YAML contact file."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return validate_contact_data(raw)


def validate_contact_data(data: dict[str, Any]) -> ContactData:
    """Validate YAML data and convert it to dataclasses.

    Quaternions are normalized with a warning if needed.
    """

    if not isinstance(data, dict):
        raise ValueError("Top-level YAML value must be a mapping.")

    robot_raw = _mapping(data.get("robot", {}), "robot")
    tool_raw = _mapping(data.get("tool", {}), "tool")
    contacts_raw = data.get("contacts")
    if not isinstance(contacts_raw, list):
        raise ValueError("'contacts' must be a list.")
    if len(contacts_raw) < 3:
        raise ValueError("At least three contacts are required.")

    robot = RobotConfig(
        name=str(robot_raw.get("name", "ur5e_hande")),
        base_frame=str(robot_raw.get("base_frame", "robot_base")),
        tool_frame=str(robot_raw.get("tool_frame", "tcp")),
        pose_convention=str(robot_raw.get("pose_convention", "xyz_quat_xyzw")),
        joint_order=list(robot_raw.get("joint_order", DEFAULT_JOINT_ORDER)),
    )
    if len(robot.joint_order) != 6:
        raise ValueError("'robot.joint_order' must contain six UR5e joints.")
    if robot.pose_convention not in {"xyz_quat_xyzw", "ur_polyscope_xyz_rotvec"}:
        raise ValueError(
            "Supported pose conventions are 'xyz_quat_xyzw' and 'ur_polyscope_xyz_rotvec'."
        )

    tool = ToolConfig(contact_offset_m=_vector(tool_raw.get("contact_offset_m", [0, 0, 0]), 3))

    contacts: list[ContactPose] = []
    for index, raw_contact in enumerate(contacts_raw):
        contact_raw = _mapping(raw_contact, f"contacts[{index}]")
        q_rad = _read_joint_vector(contact_raw, index)
        position = _vector(
            contact_raw.get("tcp_position_m"),
            3,
            f"contacts[{index}].tcp_position_m",
        )
        quat = _read_orientation_xyzw(contact_raw, index, robot.pose_convention)
        quat_norm = np.linalg.norm(quat)
        if quat_norm <= 0.0:
            raise ValueError(f"contacts[{index}].tcp_orientation_xyzw has zero norm.")
        if not np.isclose(quat_norm, 1.0, atol=1e-4):
            warnings.warn(
                f"Normalizing quaternion for contact {index} with norm {quat_norm:.6f}.",
                stacklevel=2,
            )
            quat = quat / quat_norm

        contacts.append(
            ContactPose(
                name=str(contact_raw.get("name", f"contact_{index + 1}")),
                q_rad=q_rad,
                tcp_position_m=position,
                tcp_orientation_xyzw=quat,
                note=str(contact_raw.get("note", "")),
            )
        )

    surface_raw = _mapping(data.get("surface_estimation", {}), "surface_estimation")
    normal_direction = str(
        data.get("normal_direction", surface_raw.get("normal_direction", "auto"))
    )
    normal_direction_vector_raw = data.get(
        "normal_direction_vector",
        surface_raw.get("normal_direction_vector"),
    )
    normal_direction_vector = (
        None if normal_direction_vector_raw is None else _vector(normal_direction_vector_raw, 3)
    )
    if normal_direction_vector is not None:
        normal_hint: str | np.ndarray = normal_direction_vector
    else:
        normal_hint = normal_direction
    if normal_direction not in {"auto", "positive_z", "negative_z"}:
        raise ValueError("'normal_direction' must be 'auto', 'positive_z', or 'negative_z'.")

    contact_data = ContactData(
        robot=robot,
        tool=tool,
        contacts=contacts,
        normal_direction=normal_direction,
        normal_direction_vector=normal_direction_vector,
        flip_normal=bool(data.get("flip_normal", surface_raw.get("flip_normal", False))),
    )

    estimate_plane_from_points(
        contact_data.contact_points(),
        normal_hint=normal_hint,
        flip=contact_data.flip_normal,
    )
    return contact_data


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"'{name}' must be a mapping.")
    return value


def _vector(value: Any, length: int, name: str = "vector") -> np.ndarray:
    if value is None:
        raise ValueError(f"'{name}' is required.")
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,):
        raise ValueError(f"'{name}' must have length {length}.")
    return vector


def _read_joint_vector(contact_raw: dict[str, Any], index: int) -> np.ndarray:
    if "q_rad" in contact_raw:
        return _vector(contact_raw.get("q_rad"), 6, f"contacts[{index}].q_rad")
    if "q_deg" in contact_raw:
        return np.deg2rad(_vector(contact_raw.get("q_deg"), 6, f"contacts[{index}].q_deg"))
    raise ValueError(f"contacts[{index}] must contain 'q_rad' or 'q_deg'.")


def _read_orientation_xyzw(
    contact_raw: dict[str, Any],
    index: int,
    pose_convention: str,
) -> np.ndarray:
    if "tcp_orientation_xyzw" in contact_raw:
        return _vector(
            contact_raw.get("tcp_orientation_xyzw"),
            4,
            f"contacts[{index}].tcp_orientation_xyzw",
        )
    if pose_convention == "ur_polyscope_xyz_rotvec" and "tcp_rotation_vector_rad" in contact_raw:
        rotvec = _vector(
            contact_raw.get("tcp_rotation_vector_rad"),
            3,
            f"contacts[{index}].tcp_rotation_vector_rad",
        )
        return Rotation.from_rotvec(rotvec).as_quat()
    raise ValueError(
        f"contacts[{index}] must contain 'tcp_orientation_xyzw'"
        " or, for UR PolyScope input, 'tcp_rotation_vector_rad'."
    )
