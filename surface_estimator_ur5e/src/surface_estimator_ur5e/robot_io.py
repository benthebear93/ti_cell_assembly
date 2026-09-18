"""Small RTDE connection helpers; importing this module never connects a robot."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_ROBOT_IP = "192.168.0.24"


def connect_rtde_control(robot_ip: str) -> Any:
    try:
        from rtde_control import RTDEControlInterface
    except ImportError as exc:
        raise RuntimeError(
            "ur_rtde control module is not installed. Run 'uv sync' from the repo root."
        ) from exc
    return RTDEControlInterface(robot_ip)


def connect_rtde_receive(robot_ip: str) -> Any:
    try:
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError("ur_rtde is not installed. Run 'uv sync' from the repo root.") from exc
    return RTDEReceiveInterface(robot_ip)


def read_current_tcp_pose(robot_ip: str) -> tuple[Any, np.ndarray]:
    """Open receive and read one pose. The caller owns the returned connection."""
    receive = connect_rtde_receive(robot_ip)
    try:
        pose = np.asarray(receive.getActualTCPPose(), dtype=float)
        if pose.shape != (6,):
            raise RuntimeError(f"Expected 6D TCP pose from RTDE, got {pose.shape}.")
        if not np.all(np.isfinite(pose)):
            raise RuntimeError("RTDE returned a TCP pose with non-finite values.")
        return receive, pose
    except Exception:
        receive.disconnect()
        raise


def set_tcp_offset(rtde_control: Any, tcp_offset_ur: Sequence[float]) -> None:
    tcp_offset = [float(value) for value in tcp_offset_ur]
    if not rtde_control.setTcp(tcp_offset):
        raise RuntimeError(f"Failed to set active TCP offset: {tcp_offset}")
