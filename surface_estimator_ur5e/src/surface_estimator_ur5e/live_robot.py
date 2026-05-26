"""Read-only UR5e capture helpers using ur_rtde."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from surface_estimator_ur5e.io import DEFAULT_JOINT_ORDER


DEFAULT_ROBOT_IP = "192.168.0.24"


@dataclass(frozen=True)
class LiveRobotSample:
    """One read-only sample from the UR controller."""

    q_rad: np.ndarray
    tcp_pose_ur: np.ndarray

    @property
    def tcp_position_m(self) -> np.ndarray:
        """TCP position in robot base frame."""

        return self.tcp_pose_ur[:3]

    @property
    def tcp_rotation_vector_rad(self) -> np.ndarray:
        """UR/PolyScope TCP rotation vector."""

        return self.tcp_pose_ur[3:]

    @property
    def tcp_orientation_xyzw(self) -> np.ndarray:
        """TCP orientation quaternion in scipy/ROS xyzw order."""

        return Rotation.from_rotvec(self.tcp_rotation_vector_rad).as_quat()


def capture_current_pose(
    robot_ip: str = DEFAULT_ROBOT_IP,
    samples: int = 1,
    sample_period_s: float = 0.02,
) -> LiveRobotSample:
    """Connect to the UR5e with RTDE receive and capture the current state.

    This function is read-only: it does not instantiate RTDEControlInterface and it does
    not send motion, IO, payload, or TCP commands to the robot.
    """

    if samples < 1:
        raise ValueError("'samples' must be at least 1.")
    if sample_period_s < 0.0:
        raise ValueError("'sample_period_s' must be non-negative.")

    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError(
            "ur_rtde is not installed. Run 'uv sync' from the repository root."
        ) from exc

    rtde_receive = RTDEReceiveInterface(robot_ip)
    try:
        q_values: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        rotations: list[Rotation] = []
        for index in range(samples):
            q_values.append(np.asarray(rtde_receive.getActualQ(), dtype=float))
            tcp_pose = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
            if tcp_pose.shape != (6,):
                raise RuntimeError(f"Expected 6D TCP pose from RTDE, got {tcp_pose.shape}.")
            positions.append(tcp_pose[:3])
            rotations.append(Rotation.from_rotvec(tcp_pose[3:]))
            if index < samples - 1 and sample_period_s > 0.0:
                time.sleep(sample_period_s)

        q_rad = np.mean(np.vstack(q_values), axis=0)
        tcp_position = np.mean(np.vstack(positions), axis=0)
        tcp_rotation = Rotation.concatenate(rotations).mean()
        tcp_pose_ur = np.concatenate([tcp_position, tcp_rotation.as_rotvec()])
        return LiveRobotSample(q_rad=q_rad, tcp_pose_ur=tcp_pose_ur)
    finally:
        if hasattr(rtde_receive, "disconnect"):
            rtde_receive.disconnect()


def append_contact_capture(
    output_path: str | Path,
    sample: LiveRobotSample,
    name: str | None = None,
    note: str = "",
    robot_ip: str = DEFAULT_ROBOT_IP,
    contact_offset_m: list[float] | None = None,
) -> dict[str, Any]:
    """Append a captured contact to a YAML file, creating the file if needed."""

    path = Path(output_path)
    data = _load_or_create_capture_yaml(path, contact_offset_m=contact_offset_m)
    contacts = data.setdefault("contacts", [])
    if not isinstance(contacts, list):
        raise ValueError(f"'{path}' has a non-list 'contacts' field.")

    contact_name = name or f"contact_{len(contacts) + 1}"
    contact = {
        "name": contact_name,
        "q_rad": _round_list(sample.q_rad),
        "q_deg": _round_list(np.rad2deg(sample.q_rad), decimals=6),
        "tcp_position_m": _round_list(sample.tcp_position_m),
        "tcp_position_mm": _round_list(1000.0 * sample.tcp_position_m, decimals=3),
        "tcp_rotation_vector_rad": _round_list(sample.tcp_rotation_vector_rad),
        "tcp_orientation_xyzw": _round_list(sample.tcp_orientation_xyzw),
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "robot_ip": robot_ip,
        "note": note,
    }
    contacts.append(contact)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return contact


def _load_or_create_capture_yaml(
    path: Path,
    contact_offset_m: list[float] | None,
) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if not isinstance(loaded, dict):
            raise ValueError(f"'{path}' must contain a YAML mapping.")
        return loaded

    return {
        "robot": {
            "name": "ur5e_hande",
            "base_frame": "robot_base",
            "tool_frame": "tcp",
            "pose_convention": "ur_polyscope_xyz_rotvec",
            "joint_units_original": "rad",
            "joint_order": list(DEFAULT_JOINT_ORDER),
        },
        "tool": {
            "contact_offset_m": contact_offset_m or [0.0, 0.0, 0.0],
            "note": "Set nonzero if the displayed TCP pose is not the physical contact point.",
        },
        "surface_estimation": {
            "normal_direction": "auto",
            "flip_normal": False,
        },
        "contacts": [],
    }


def _round_list(values: np.ndarray, decimals: int = 10) -> list[float]:
    return [round(float(value), decimals) for value in np.asarray(values, dtype=float).ravel()]
