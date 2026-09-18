"""Slow joint-sequence execution for UR robots using ur_rtde."""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from surface_estimator_ur5e.live_robot import DEFAULT_ROBOT_IP
from surface_estimator_ur5e.transforms import (
    pose_translated_in_tcp_frame as _tool_offset_target_pose,
)
from surface_estimator_ur5e.transforms import (
    ur_pose_to_transform as _pose_vector_to_transform,
)

DEFAULT_TCP_OFFSET_UR = np.array([0.0, 0.0, 0.152, 0.0, 0.0, 0.0])
DEFAULT_GRIPPER_PORT = 63352
MAX_TOOL_MOVEL_OFFSET_M = 0.060
MAX_TOOL_MOVEL_OFFSET_MM = int(round(MAX_TOOL_MOVEL_OFFSET_M * 1000.0))
DEFAULT_COMPLIANCE_SELECTION = (1, 1, 0, 0, 0, 0)
DEFAULT_COMPLIANCE_WRENCH = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
DEFAULT_COMPLIANCE_LIMITS = (0.005, 0.005, 0.015, 0.05, 0.05, 0.05)


@dataclass(frozen=True)
class ComplianceConfig:
    """Force-mode compliance settings for one Cartesian tool move."""

    task_frame: str = "tcp"
    selection_vector: tuple[int, ...] = DEFAULT_COMPLIANCE_SELECTION
    wrench: tuple[float, ...] = DEFAULT_COMPLIANCE_WRENCH
    force_type: int = 2
    limits: tuple[float, ...] = DEFAULT_COMPLIANCE_LIMITS
    damping: float = 0.08
    gain_scaling: float = 0.5


@dataclass(frozen=True)
class JointWaypoint:
    """One named joint waypoint."""

    name: str
    q_rad: np.ndarray

    @property
    def q_deg(self) -> np.ndarray:
        """Joint waypoint in degrees."""

        return np.rad2deg(self.q_rad)


@dataclass(frozen=True)
class GripperCommand:
    """One Robotiq gripper command in the motion sequence."""

    name: str
    position_percent: float
    speed: int = 80
    force: int = 80
    wait: bool = True
    timeout_s: float = 5.0
    post_dwell_s: float = 1.0

    @property
    def position_raw(self) -> int:
        """Robotiq raw position command: 0=open, 255=closed."""

        return int(round(np.clip(self.position_percent, 0.0, 100.0) / 100.0 * 255.0))


@dataclass(frozen=True)
class ToolFrameMoveL:
    """One or more TCP-frame Cartesian moveL offsets."""

    name: str
    offsets_m: list[np.ndarray]
    speed_m_s: float = 0.005
    acceleration_m_s2: float = 0.01
    dwell_s: float = 0.1
    compliance: ComplianceConfig | None = None


SequenceStep = JointWaypoint | GripperCommand | ToolFrameMoveL


class RobotiqHandEGripper:
    """Socket client for the Robotiq URCap gripper server on the UR controller."""

    def __init__(self, host: str, port: int = DEFAULT_GRIPPER_PORT, timeout_s: float = 2.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._socket: socket.socket | None = None

    def connect(self) -> None:
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        self._socket.settimeout(self.timeout_s)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def get_position(self) -> int:
        """Return raw gripper position, 0=open and 255=closed."""

        return int(self._get_var("POS"))

    def is_open(self, threshold_raw: int = 5) -> bool:
        """Return whether the gripper is fully open within a raw-position threshold."""

        return self.get_position() <= threshold_raw

    def move_to_percent(self, command: GripperCommand) -> None:
        """Move to a percentage closed command, where 0=open and 100=closed."""

        self._set_vars(
            {
                "POS": command.position_raw,
                "SPE": command.speed,
                "FOR": command.force,
                "GTO": 1,
            }
        )
        if command.wait:
            self._wait_for_motion(command.position_raw, command.timeout_s)
        if command.post_dwell_s > 0.0:
            time.sleep(command.post_dwell_s)

    def _wait_for_motion(self, target_raw: int, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_pos = self.get_position()
        while time.monotonic() < deadline:
            last_pos = self.get_position()
            obj = self._get_var("OBJ", default=None)
            if abs(last_pos - target_raw) <= 3 or obj in {1, 2, 3}:
                return
            time.sleep(0.1)
        raise TimeoutError(
            f"Timed out waiting for gripper target {target_raw}; last position was {last_pos}."
        )

    def _get_var(self, name: str, default: int | None = None) -> int | None:
        response = self._send(f"GET {name}\n")
        parts = response.strip().split()
        if len(parts) != 2 or parts[0] != name:
            if default is not None:
                return default
            raise RuntimeError(f"Unexpected Robotiq response to GET {name!r}: {response!r}")
        try:
            return int(parts[1])
        except ValueError:
            if default is not None:
                return default
            raise RuntimeError(f"Robotiq response did not contain an integer: {response!r}")

    def _set_vars(self, values: dict[str, int]) -> None:
        command = "SET " + " ".join(f"{name} {int(value)}" for name, value in values.items()) + "\n"
        response = self._send(command)
        if response.strip().lower() != "ack":
            raise RuntimeError(f"Robotiq SET command was not acknowledged: {response!r}")

    def _send(self, command: str) -> str:
        if self._socket is None:
            raise RuntimeError("Robotiq gripper socket is not connected.")
        self._socket.sendall(command.encode("ascii"))
        return self._socket.recv(1024).decode("ascii", errors="replace")


def load_joint_sequence(path: str | Path) -> list[SequenceStep]:
    """Load an ordered YAML mapping of named joint waypoints."""

    sequence_path = Path(path)
    with sequence_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError("Motion sequence YAML must be a mapping of waypoint_name -> q_deg/q_rad.")

    waypoints: list[SequenceStep] = []
    for name, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"Waypoint '{name}' must be a mapping.")
        if "q_rad" in value or "q_deg" in value:
            waypoints.append(JointWaypoint(name=str(name), q_rad=_read_q_rad(value, str(name))))
        elif "gripper_percent" in value or "gripper" in value:
            waypoints.append(_read_gripper_command(value, str(name)))
        elif "tool_movel_offsets_m" in value:
            waypoints.append(_read_tool_movel(value, str(name)))
        else:
            raise ValueError(
                f"Step '{name}' must contain q_deg/q_rad, gripper_percent, or tool_movel_offsets_m."
            )
    if not waypoints:
        raise ValueError("Motion sequence contains no steps.")
    return waypoints


def run_joint_sequence(
    waypoints: list[SequenceStep],
    robot_ip: str = DEFAULT_ROBOT_IP,
    speed_rad_s: float = 0.05,
    acceleration_rad_s2: float = 0.05,
    linear_speed_m_s: float = 0.01,
    linear_acceleration_m_s2: float = 0.02,
    blend_radius_m: float = 0.0,
    dwell_s: float = 0.5,
    max_start_delta_deg: float = 15.0,
    final_tool_negative_z_m: float = 0.0,
    tcp_offset_ur: np.ndarray | list[float] | None = None,
    gripper_port: int = DEFAULT_GRIPPER_PORT,
    require_gripper_open: bool = True,
    gripper_open_threshold_raw: int = 5,
    start_at: str | None = None,
    stop_before: str | None = None,
    stop_after: str | None = None,
    allow_force_mode: bool = False,
    execute: bool = False,
) -> None:
    """Dry-run or execute a slow moveJ sequence with ur_rtde.

    ``execute=False`` prints the plan only. With ``execute=True`` this function sends
    joint-space motion commands to the robot.
    """

    tcp_offset = DEFAULT_TCP_OFFSET_UR if tcp_offset_ur is None else np.asarray(tcp_offset_ur, dtype=float)
    selected_waypoints = select_sequence_steps(
        waypoints,
        start_at=start_at,
        stop_before=stop_before,
        stop_after=stop_after,
    )
    is_partial_sequence = _step_names(selected_waypoints) != _step_names(waypoints)
    effective_final_tool_negative_z_m = 0.0 if is_partial_sequence else final_tool_negative_z_m
    _validate_motion_parameters(
        speed_rad_s,
        acceleration_rad_s2,
        linear_speed_m_s,
        linear_acceleration_m_s2,
        blend_radius_m,
        dwell_s,
        max_start_delta_deg,
        effective_final_tool_negative_z_m,
        tcp_offset,
        gripper_port,
        gripper_open_threshold_raw,
    )
    _print_plan(
        selected_waypoints,
        robot_ip,
        speed_rad_s,
        acceleration_rad_s2,
        linear_speed_m_s,
        linear_acceleration_m_s2,
        blend_radius_m,
        dwell_s,
        max_start_delta_deg,
        effective_final_tool_negative_z_m,
        tcp_offset,
        gripper_port,
        require_gripper_open,
        gripper_open_threshold_raw,
    )
    if is_partial_sequence and final_tool_negative_z_m > 0.0:
        print("partial sequence selected: final Cartesian moveL will be skipped.")

    if not execute:
        print("DRY RUN ONLY: add --execute to send robot and gripper commands.")
        return
    if _has_compliant_tool_movel_steps(selected_waypoints) and not allow_force_mode:
        raise RuntimeError(
            "This sequence contains a forceMode compliance step. Re-check the path, then rerun "
            "with --allow-force-mode if you deliberately want compliant force-mode execution."
        )

    try:
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface
    except ImportError as exc:
        raise RuntimeError("ur_rtde is not installed. Run 'uv sync' from the repository root.") from exc

    joint_waypoints = _joint_steps(selected_waypoints)
    if not joint_waypoints:
        raise ValueError("Motion sequence must contain at least one joint waypoint.")

    rtde_receive = None
    rtde_control = None
    gripper = None
    try:
        rtde_receive = RTDEReceiveInterface(robot_ip)
        rtde_control = RTDEControlInterface(robot_ip)
        _raise_if_robot_not_ready(rtde_receive)
        should_check_initial_gripper_open = (
            require_gripper_open
            and bool(selected_waypoints)
            and selected_waypoints[0].name == "initial_start_pose"
        )
        if _has_gripper_steps(selected_waypoints) or should_check_initial_gripper_open:
            gripper = RobotiqHandEGripper(robot_ip, port=gripper_port)
            gripper.connect()
            gripper_position = gripper.get_position()
            print(f"gripper raw position at start: {gripper_position} (0=open, 255=closed)")
            if should_check_initial_gripper_open and gripper_position > gripper_open_threshold_raw:
                raise RuntimeError(
                    "Gripper is not fully open at the initial pose. "
                    f"Position is {gripper_position}, threshold is {gripper_open_threshold_raw}. "
                    "Open the gripper first or run with --skip-gripper-open-check."
                )
        if not rtde_control.setTcp(tcp_offset.tolist()):
            raise RuntimeError(f"Failed to set active TCP offset to {_fmt(tcp_offset)}.")
        print(f"active TCP offset set to: {_fmt(tcp_offset)}")
        current_q = np.asarray(rtde_receive.getActualQ(), dtype=float)
        print(f"current q_deg: {_fmt(np.rad2deg(current_q))}")
        _raise_if_start_pose_too_far(current_q, joint_waypoints[0], max_start_delta_deg)

        for index, waypoint in enumerate(selected_waypoints, start=1):
            if isinstance(waypoint, JointWaypoint):
                print(f"[{index}/{len(selected_waypoints)}] moveJ -> {waypoint.name}")
                success = rtde_control.moveJ(
                    waypoint.q_rad.tolist(),
                    speed_rad_s,
                    acceleration_rad_s2,
                    False,
                )
                if not success:
                    raise RuntimeError(f"moveJ returned false for waypoint '{waypoint.name}'.")
                if dwell_s > 0.0:
                    time.sleep(dwell_s)
            else:
                if isinstance(waypoint, GripperCommand):
                    if gripper is None:
                        raise RuntimeError(
                            "Motion sequence contains a gripper command, but no gripper is connected."
                        )
                    print(
                        f"[{index}/{len(selected_waypoints)}] gripper -> {waypoint.name}: "
                        f"{waypoint.position_percent:.1f}% closed"
                    )
                    gripper.move_to_percent(waypoint)
                    continue

                print(
                    f"[{index}/{len(selected_waypoints)}] tool-frame moveL -> {waypoint.name}: "
                    f"{len(waypoint.offsets_m)} offsets"
                )
                _execute_tool_frame_movel_offsets(
                    rtde_receive,
                    rtde_control,
                    waypoint,
                    default_speed_m_s=linear_speed_m_s,
                    default_acceleration_m_s2=linear_acceleration_m_s2,
                )
                continue

        if effective_final_tool_negative_z_m > 0.0:
            tcp_pose = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
            target_tcp_pose = _tool_negative_z_target_pose(tcp_pose, effective_final_tool_negative_z_m)
            print(
                "final moveL along tool -Z: "
                f"{1000.0 * effective_final_tool_negative_z_m:.1f} mm"
            )
            print(f"current tcp_pose: {_fmt(tcp_pose)}")
            print(f"target tcp_pose: {_fmt(target_tcp_pose)}")
            success = rtde_control.moveL(
                target_tcp_pose.tolist(),
                linear_speed_m_s,
                linear_acceleration_m_s2,
                False,
            )
            if not success:
                raise RuntimeError("Final tool-frame moveL returned false.")
        print("Motion sequence complete.")
    finally:
        if rtde_control is not None and hasattr(rtde_control, "stopScript"):
            rtde_control.stopScript()
        if rtde_receive is not None and hasattr(rtde_receive, "disconnect"):
            rtde_receive.disconnect()
        if rtde_control is not None and hasattr(rtde_control, "disconnect"):
            rtde_control.disconnect()
        if gripper is not None:
            gripper.close()


def preview_joint_sequence(
    waypoints: list[SequenceStep],
    steps_per_segment: int = 40,
    playback_hz: float = 15.0,
    port: int = 8080,
    final_tool_negative_z_m: float = 0.0,
    tcp_offset_ur: np.ndarray | list[float] | None = None,
) -> None:
    """Preview a joint sequence offline in viser with a UR5e mesh."""

    if steps_per_segment < 2:
        raise ValueError("--steps-per-segment must be at least 2.")
    if playback_hz <= 0.0:
        raise ValueError("--playback-hz must be positive.")
    tcp_offset = DEFAULT_TCP_OFFSET_UR if tcp_offset_ur is None else np.asarray(tcp_offset_ur, dtype=float)
    if tcp_offset.shape != (6,):
        raise ValueError("--tcp-offset-ur must contain exactly six values: x y z rx ry rz.")
    if not 0.0 <= final_tool_negative_z_m <= MAX_TOOL_MOVEL_OFFSET_M:
        raise ValueError(
            f"--final-tool-negative-z-m must be between 0 and {MAX_TOOL_MOVEL_OFFSET_M:.3f} m."
        )

    import viser
    from viser.extras import ViserUrdf

    from surface_estimator_ur5e.io import DEFAULT_JOINT_ORDER
    from surface_estimator_ur5e.robot_model import URDFRobotVisualizer
    from surface_estimator_ur5e.visualization import _ceiling_mount_display_transform

    joint_waypoints = _joint_steps(waypoints)
    if not joint_waypoints:
        raise ValueError("Motion sequence must contain at least one joint waypoint to preview.")

    display_transform = _ceiling_mount_display_transform()
    visualizer = URDFRobotVisualizer()
    robot = visualizer._load_description()
    root_transform = visualizer.root_transform_for_robot_base(display_transform)

    server = viser.ViserServer(port=port)
    scene = getattr(server, "scene", server)
    _add_preview_root(scene, "/preview_robot", root_transform)
    viser_urdf = ViserUrdf(server, robot, root_node_name="/preview_robot")

    joint_order = list(DEFAULT_JOINT_ORDER)
    path = interpolate_waypoints(joint_waypoints, steps_per_segment=steps_per_segment)
    tool0_points = _tool0_path_points(robot, visualizer, path, joint_order, display_transform)
    tcp_poses = _tcp_path_poses(
        robot,
        visualizer,
        path,
        joint_order,
        display_transform,
        tcp_offset,
    )
    tcp_points = np.asarray([pose[:3, 3] for pose in tcp_poses])
    tool_movel_preview_points = _tool_movel_preview_points(
        robot,
        visualizer,
        waypoints,
        joint_order,
        display_transform,
        tcp_offset,
    )
    final_movel_points = _final_movel_points(tcp_poses[-1], final_tool_negative_z_m)
    _add_preview_context(scene, tool0_points)
    _add_path(scene, "/motion/tool0_path", tool0_points)
    _add_path(scene, "/motion/tcp_path", tcp_points, color=(40, 160, 250), line_width=1.5)
    if len(tool_movel_preview_points) > 0:
        _add_path(
            scene,
            "/motion/tool_frame_movel_path",
            tool_movel_preview_points,
            color=(230, 60, 220),
            line_width=2.0,
        )
    _add_path(scene, "/motion/final_movel_tcp_path", final_movel_points, color=(30, 210, 120), line_width=2.0)
    _add_waypoint_markers(scene, robot, visualizer, joint_waypoints, joint_order, display_transform)
    current_tool_marker = _add_current_tool_marker(scene, tcp_points[0])
    _set_preview_camera(server, tool0_points)
    total_frames = len(path) + len(final_movel_points)

    play = slider = status = None
    if hasattr(server, "gui"):
        play = server.gui.add_checkbox("Play", initial_value=True)
        slider = server.gui.add_slider(
            "Preview frame",
            min=0,
            max=total_frames - 1,
            step=1,
            initial_value=0,
        )
        status = server.gui.add_text("Current waypoint", initial_value="starting")

    print("Preview server is running. Open the printed URL in your browser.")
    print(f"Requested port: {port}")
    print("Scene tree should contain /preview_robot and /motion.")
    print("Blue path = TCP path with 152 mm TCP offset.")
    print("Green path = final moveL 10 mm along TCP/tool -Z.")
    print("During the final moveL preview, the marker moves but the mesh remains at the last joint pose.")
    print("Animating sequence. Press Ctrl+C in this terminal to stop.")
    frame = 0
    try:
        while True:
            if play is not None and not play.value and slider is not None:
                frame = int(slider.value)
            frame_index = frame % total_frames
            joint_frame_index = min(frame_index, len(path) - 1)
            waypoint = path[joint_frame_index]
            cfg = visualizer._configuration_for_urdf_object(robot, waypoint.q_rad, joint_order)
            viser_urdf.update_cfg(cfg)
            if frame_index < len(path):
                marker_position = tcp_points[frame_index]
                status_text = waypoint.name
            else:
                movel_index = frame_index - len(path)
                marker_position = final_movel_points[movel_index]
                status_text = "final_moveL_tool_negative_z"
            current_tool_marker.position = marker_position
            if slider is not None and play is not None and play.value:
                slider.value = frame_index
            if status is not None:
                status.value = status_text
            time.sleep(1.0 / playback_hz)
            if play is None or play.value:
                frame += 1
    except KeyboardInterrupt:
        print("Stopping preview.")


def interpolate_waypoints(
    waypoints: list[JointWaypoint],
    steps_per_segment: int,
) -> list[JointWaypoint]:
    """Linearly interpolate a joint path between waypoints for offline preview."""

    if len(waypoints) == 1:
        return waypoints

    path: list[JointWaypoint] = []
    for start, end in zip(waypoints[:-1], waypoints[1:], strict=True):
        for step in range(steps_per_segment):
            if path and step == 0:
                continue
            alpha = step / float(steps_per_segment)
            q_rad = (1.0 - alpha) * start.q_rad + alpha * end.q_rad
            path.append(JointWaypoint(name=f"{start.name}_to_{end.name}_{step:03d}", q_rad=q_rad))
    path.append(waypoints[-1])
    return path


def select_sequence_steps(
    steps: list[SequenceStep],
    start_at: str | None = None,
    stop_before: str | None = None,
    stop_after: str | None = None,
) -> list[SequenceStep]:
    """Select a contiguous subset of sequence steps by name."""

    if stop_before is not None and stop_after is not None:
        raise ValueError("Use only one of --stop-before or --stop-after.")

    names = [step.name for step in steps]
    start_index = 0 if start_at is None else _step_index(names, start_at)
    if stop_before is not None:
        end_index = _step_index(names, stop_before)
    elif stop_after is not None:
        end_index = _step_index(names, stop_after) + 1
    else:
        end_index = len(steps)

    selected = steps[start_index:end_index]
    if not selected:
        raise ValueError("Selected sequence is empty. Check start/stop step names.")
    return selected


def _step_index(names: list[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        available = ", ".join(names)
        raise ValueError(f"Unknown step '{name}'. Available steps: {available}") from exc


def _read_q_rad(value: dict[str, Any], name: str) -> np.ndarray:
    if "q_rad" in value:
        q_rad = np.asarray(value["q_rad"], dtype=float)
    elif "q_deg" in value:
        q_rad = np.deg2rad(np.asarray(value["q_deg"], dtype=float))
    else:
        raise ValueError(f"Waypoint '{name}' must contain q_deg or q_rad.")
    if q_rad.shape != (6,):
        raise ValueError(f"Waypoint '{name}' must contain exactly six joint values.")
    if not np.all(np.isfinite(q_rad)):
        raise ValueError(f"Waypoint '{name}' contains non-finite joint values.")
    if np.any(np.abs(q_rad) > 2.5 * np.pi):
        raise ValueError(f"Waypoint '{name}' has a joint outside +/-450 degrees.")
    return q_rad


def _read_gripper_command(value: dict[str, Any], name: str) -> GripperCommand:
    raw = value.get("gripper", value)
    if not isinstance(raw, dict):
        raise ValueError(f"Gripper step '{name}' must be a mapping.")
    percent = float(raw.get("position_percent", raw.get("gripper_percent", value.get("gripper_percent", 0.0))))
    if not 0.0 <= percent <= 100.0:
        raise ValueError(f"Gripper step '{name}' percent must be between 0 and 100.")
    speed = int(raw.get("speed", value.get("speed", 80)))
    force = int(raw.get("force", value.get("force", 80)))
    if not 0 <= speed <= 255:
        raise ValueError(f"Gripper step '{name}' speed must be between 0 and 255.")
    if not 0 <= force <= 255:
        raise ValueError(f"Gripper step '{name}' force must be between 0 and 255.")
    return GripperCommand(
        name=name,
        position_percent=percent,
        speed=speed,
        force=force,
        wait=bool(raw.get("wait", value.get("wait", True))),
        timeout_s=float(raw.get("timeout_s", value.get("timeout_s", 5.0))),
        post_dwell_s=float(raw.get("post_dwell_s", value.get("post_dwell_s", 1.0))),
    )


def _read_compliance_config(value: Any, name: str) -> ComplianceConfig | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return ComplianceConfig() if value else None
    if not isinstance(value, dict):
        raise ValueError(f"Tool moveL step '{name}' compliance must be a mapping or boolean.")

    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"Tool moveL step '{name}' compliance.enabled must be true or false.")
    if not enabled:
        return None

    task_frame = str(value.get("task_frame", "tcp")).lower()
    if task_frame not in {"tcp", "base"}:
        raise ValueError(f"Tool moveL step '{name}' compliance.task_frame must be 'tcp' or 'base'.")

    selection_vector = _read_selection_vector(
        value.get("selection_vector", DEFAULT_COMPLIANCE_SELECTION),
        name,
    )
    wrench = _read_six_float_tuple(
        value.get("wrench", DEFAULT_COMPLIANCE_WRENCH),
        f"Tool moveL step '{name}' compliance.wrench",
    )
    if np.any(np.abs(np.asarray(wrench[:3], dtype=float)) > 30.0):
        raise ValueError(f"Tool moveL step '{name}' compliance force must be <= 30 N.")
    if np.any(np.abs(np.asarray(wrench[3:], dtype=float)) > 5.0):
        raise ValueError(f"Tool moveL step '{name}' compliance torque must be <= 5 Nm.")

    force_type = int(value.get("type", value.get("force_type", 2)))
    if force_type not in {1, 2, 3}:
        raise ValueError(f"Tool moveL step '{name}' compliance type must be 1, 2, or 3.")

    limits = _read_six_float_tuple(
        value.get("limits", DEFAULT_COMPLIANCE_LIMITS),
        f"Tool moveL step '{name}' compliance.limits",
    )
    limits_array = np.asarray(limits, dtype=float)
    if np.any(limits_array <= 0.0):
        raise ValueError(f"Tool moveL step '{name}' compliance limits must be positive.")
    if np.any(limits_array[:3] > 0.05):
        raise ValueError(f"Tool moveL step '{name}' translational compliance limits must be <= 0.05.")
    if np.any(limits_array[3:] > 0.5):
        raise ValueError(f"Tool moveL step '{name}' rotational compliance limits must be <= 0.5.")

    damping = float(value.get("damping", 0.08))
    if not 0.0 <= damping <= 1.0:
        raise ValueError(f"Tool moveL step '{name}' compliance damping must be between 0 and 1.")

    gain_scaling = float(value.get("gain_scaling", 0.5))
    if not 0.0 < gain_scaling <= 1.0:
        raise ValueError(f"Tool moveL step '{name}' compliance gain_scaling must be > 0 and <= 1.")

    return ComplianceConfig(
        task_frame=task_frame,
        selection_vector=selection_vector,
        wrench=wrench,
        force_type=force_type,
        limits=limits,
        damping=damping,
        gain_scaling=gain_scaling,
    )


def _read_selection_vector(value: Any, name: str) -> tuple[int, ...]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"Tool moveL step '{name}' compliance.selection_vector must have length 6.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Tool moveL step '{name}' compliance.selection_vector contains non-finite values.")
    if not np.all((vector == 0) | (vector == 1)):
        raise ValueError(f"Tool moveL step '{name}' compliance.selection_vector must contain 0 or 1.")
    return tuple(int(item) for item in vector)


def _read_six_float_tuple(value: Any, label: str) -> tuple[float, ...]:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,):
        raise ValueError(f"{label} must have length 6.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values.")
    return tuple(float(item) for item in vector)


def _read_tool_movel(value: dict[str, Any], name: str) -> ToolFrameMoveL:
    offsets_raw = value.get("tool_movel_offsets_m")
    if not isinstance(offsets_raw, list) or not offsets_raw:
        raise ValueError(f"Tool moveL step '{name}' must contain a non-empty offsets list.")

    offsets: list[np.ndarray] = []
    for index, offset_raw in enumerate(offsets_raw):
        offset = np.asarray(offset_raw, dtype=float)
        if offset.shape != (3,):
            raise ValueError(f"Tool moveL step '{name}' offset {index} must have length 3.")
        if not np.all(np.isfinite(offset)):
            raise ValueError(f"Tool moveL step '{name}' offset {index} contains non-finite values.")
        if float(np.linalg.norm(offset)) > MAX_TOOL_MOVEL_OFFSET_M:
            raise ValueError(
                f"Tool moveL step '{name}' offset {index} is larger than "
                f"{MAX_TOOL_MOVEL_OFFSET_MM} mm."
            )
        offsets.append(offset)

    speed_m_s = float(value.get("speed_m_s", 0.005))
    acceleration_m_s2 = float(value.get("acceleration_m_s2", 0.01))
    dwell_s = float(value.get("dwell_s", 0.1))
    if not 0.0 < speed_m_s <= 0.05:
        raise ValueError(f"Tool moveL step '{name}' speed_m_s must be > 0 and <= 0.05.")
    if not 0.0 < acceleration_m_s2 <= 0.1:
        raise ValueError(
            f"Tool moveL step '{name}' acceleration_m_s2 must be > 0 and <= 0.1."
        )
    if dwell_s < 0.0:
        raise ValueError(f"Tool moveL step '{name}' dwell_s must be non-negative.")

    return ToolFrameMoveL(
        name=name,
        offsets_m=offsets,
        speed_m_s=speed_m_s,
        acceleration_m_s2=acceleration_m_s2,
        dwell_s=dwell_s,
        compliance=_read_compliance_config(value.get("compliance"), name),
    )


def _validate_motion_parameters(
    speed_rad_s: float,
    acceleration_rad_s2: float,
    linear_speed_m_s: float,
    linear_acceleration_m_s2: float,
    blend_radius_m: float,
    dwell_s: float,
    max_start_delta_deg: float,
    final_tool_negative_z_m: float,
    tcp_offset: np.ndarray,
    gripper_port: int,
    gripper_open_threshold_raw: int,
) -> None:
    if not 0.0 < speed_rad_s <= 0.25:
        raise ValueError("--speed-rad-s must be > 0 and <= 0.25 for this slow runner.")
    if not 0.0 < acceleration_rad_s2 <= 0.25:
        raise ValueError("--acceleration-rad-s2 must be > 0 and <= 0.25 for this slow runner.")
    if not 0.0 < linear_speed_m_s <= 0.05:
        raise ValueError("--linear-speed-m-s must be > 0 and <= 0.05 for this slow runner.")
    if not 0.0 < linear_acceleration_m_s2 <= 0.1:
        raise ValueError(
            "--linear-acceleration-m-s2 must be > 0 and <= 0.1 for this slow runner."
        )
    if blend_radius_m != 0.0:
        raise ValueError("Only blend_radius_m=0 is supported for this cautious sequence runner.")
    if dwell_s < 0.0:
        raise ValueError("--dwell-s must be non-negative.")
    if max_start_delta_deg <= 0.0:
        raise ValueError("--max-start-delta-deg must be positive.")
    if not 0.0 <= final_tool_negative_z_m <= MAX_TOOL_MOVEL_OFFSET_M:
        raise ValueError(
            f"--final-tool-negative-z-m must be between 0 and {MAX_TOOL_MOVEL_OFFSET_M:.3f} m."
        )
    if tcp_offset.shape != (6,):
        raise ValueError("--tcp-offset-ur must contain exactly six values: x y z rx ry rz.")
    if not np.all(np.isfinite(tcp_offset)):
        raise ValueError("--tcp-offset-ur contains non-finite values.")
    if not 1 <= gripper_port <= 65535:
        raise ValueError("--gripper-port must be a valid TCP port.")
    if not 0 <= gripper_open_threshold_raw <= 255:
        raise ValueError("--gripper-open-threshold-raw must be between 0 and 255.")


def _print_plan(
    waypoints: list[SequenceStep],
    robot_ip: str,
    speed_rad_s: float,
    acceleration_rad_s2: float,
    linear_speed_m_s: float,
    linear_acceleration_m_s2: float,
    blend_radius_m: float,
    dwell_s: float,
    max_start_delta_deg: float,
    final_tool_negative_z_m: float,
    tcp_offset: np.ndarray,
    gripper_port: int,
    require_gripper_open: bool,
    gripper_open_threshold_raw: int,
) -> None:
    print(f"robot_ip: {robot_ip}")
    print(f"speed_rad_s: {speed_rad_s:.4f}")
    print(f"acceleration_rad_s2: {acceleration_rad_s2:.4f}")
    print(f"linear_speed_m_s: {linear_speed_m_s:.4f}")
    print(f"linear_acceleration_m_s2: {linear_acceleration_m_s2:.4f}")
    print(f"blend_radius_m: {blend_radius_m:.4f}")
    print(f"dwell_s: {dwell_s:.4f}")
    print(f"max_start_delta_deg: {max_start_delta_deg:.3f}")
    print(f"tcp_offset_ur: {_fmt(tcp_offset)}")
    print(f"gripper_port: {gripper_port}")
    print(f"require_gripper_open: {require_gripper_open}")
    print(f"gripper_open_threshold_raw: {gripper_open_threshold_raw}")
    print("steps:")
    for waypoint in waypoints:
        if isinstance(waypoint, JointWaypoint):
            print(f"  {waypoint.name}: q_deg={_fmt(waypoint.q_deg)}")
        elif isinstance(waypoint, GripperCommand):
            print(
                f"  {waypoint.name}: gripper={waypoint.position_percent:.1f}% closed "
                f"(raw={waypoint.position_raw}, speed={waypoint.speed}, force={waypoint.force}, "
                f"wait={waypoint.wait}, timeout_s={waypoint.timeout_s:.1f}, "
                f"post_dwell_s={waypoint.post_dwell_s:.1f})"
            )
        else:
            offsets_mm = [offset * 1000.0 for offset in waypoint.offsets_m]
            print(
                f"  {waypoint.name}: tool moveL offsets mm="
                f"{[_fmt(offset) for offset in offsets_mm]} "
                f"(speed_m_s={waypoint.speed_m_s:.4f}, "
                f"acceleration_m_s2={waypoint.acceleration_m_s2:.4f}, "
                f"dwell_s={waypoint.dwell_s:.2f})"
            )
            if waypoint.compliance is not None:
                compliance = waypoint.compliance
                print(
                    f"    compliance: forceMode task_frame={compliance.task_frame}, "
                    f"selection={list(compliance.selection_vector)}, "
                    f"wrench={_fmt(np.asarray(compliance.wrench))}, "
                    f"limits={_fmt(np.asarray(compliance.limits))}, "
                    f"damping={compliance.damping:.3f}, "
                    f"gain_scaling={compliance.gain_scaling:.3f}"
                )
    if final_tool_negative_z_m > 0.0:
        print(
            "final Cartesian move: "
            f"moveL {1000.0 * final_tool_negative_z_m:.1f} mm along TCP/tool -Z"
        )


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in np.asarray(values).ravel()) + "]"


def _joint_steps(steps: list[SequenceStep]) -> list[JointWaypoint]:
    return [step for step in steps if isinstance(step, JointWaypoint)]


def _has_gripper_steps(steps: list[SequenceStep]) -> bool:
    return any(isinstance(step, GripperCommand) for step in steps)


def _has_compliant_tool_movel_steps(steps: list[SequenceStep]) -> bool:
    return any(isinstance(step, ToolFrameMoveL) and step.compliance is not None for step in steps)


def _step_names(steps: list[SequenceStep]) -> list[str]:
    return [step.name for step in steps]


def _raise_if_robot_not_ready(rtde_receive: object) -> None:
    if hasattr(rtde_receive, "isEmergencyStopped") and rtde_receive.isEmergencyStopped():
        raise RuntimeError("Robot is emergency-stopped.")
    if hasattr(rtde_receive, "isProtectiveStopped") and rtde_receive.isProtectiveStopped():
        raise RuntimeError("Robot is protective-stopped.")
    if hasattr(rtde_receive, "getSafetyMode"):
        print(f"safety mode: {rtde_receive.getSafetyMode()}")
    if hasattr(rtde_receive, "getRobotMode"):
        print(f"robot mode: {rtde_receive.getRobotMode()}")


def _raise_if_start_pose_too_far(
    current_q: np.ndarray,
    first_waypoint: JointWaypoint,
    max_start_delta_deg: float,
) -> None:
    delta_deg = np.abs(np.rad2deg(current_q - first_waypoint.q_rad))
    print(f"first waypoint q_deg: {_fmt(first_waypoint.q_deg)}")
    print(f"start delta deg: {_fmt(delta_deg)}")
    if float(np.max(delta_deg)) > max_start_delta_deg:
        raise RuntimeError(
            "Current robot pose is too far from the first waypoint "
            f"'{first_waypoint.name}'. Max joint delta is {np.max(delta_deg):.3f} deg, "
            f"limit is {max_start_delta_deg:.3f} deg. Move the robot close to the first "
            "waypoint first, or deliberately raise --max-start-delta-deg after checking "
            "the path from the current pose."
        )


def _tool_negative_z_target_pose(tcp_pose: np.ndarray, distance_m: float) -> np.ndarray:
    """Return a UR TCP pose translated along local/tool negative Z."""
    return _tool_offset_target_pose(tcp_pose, np.array([0.0, 0.0, -distance_m]))


def _execute_tool_frame_movel_offsets(
    rtde_receive: object,
    rtde_control: object,
    command: ToolFrameMoveL,
    default_speed_m_s: float,
    default_acceleration_m_s2: float,
) -> None:
    del default_speed_m_s, default_acceleration_m_s2
    for offset_index, offset_m in enumerate(command.offsets_m, start=1):
        tcp_pose = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
        target_tcp_pose = _tool_offset_target_pose(tcp_pose, offset_m)
        print(
            f"  offset {offset_index}/{len(command.offsets_m)} in TCP frame mm: "
            f"{_fmt(1000.0 * offset_m)}"
        )
        print(f"  target tcp_pose: {_fmt(target_tcp_pose)}")
        force_mode_active = False
        try:
            if command.compliance is not None:
                _start_force_mode(rtde_control, command.compliance, tcp_pose, command.name)
                force_mode_active = True
            success = rtde_control.moveL(
                target_tcp_pose.tolist(),
                command.speed_m_s,
                command.acceleration_m_s2,
                False,
            )
            if not success:
                raise RuntimeError(f"Tool-frame moveL returned false for step '{command.name}'.")
        finally:
            if force_mode_active:
                _stop_force_mode(rtde_control, command.name)
        if command.dwell_s > 0.0:
            time.sleep(command.dwell_s)


def _start_force_mode(
    rtde_control: object,
    compliance: ComplianceConfig,
    tcp_pose: np.ndarray,
    step_name: str,
) -> None:
    task_frame = _force_mode_task_frame(compliance, tcp_pose)
    print(
        f"  forceMode compliance on for '{step_name}': "
        f"task_frame={compliance.task_frame}, selection={list(compliance.selection_vector)}"
    )
    if hasattr(rtde_control, "forceModeSetDamping"):
        result = rtde_control.forceModeSetDamping(compliance.damping)
        if result is False:
            raise RuntimeError(f"Failed to set forceMode damping for step '{step_name}'.")
    if hasattr(rtde_control, "forceModeSetGainScaling"):
        result = rtde_control.forceModeSetGainScaling(compliance.gain_scaling)
        if result is False:
            raise RuntimeError(f"Failed to set forceMode gain scaling for step '{step_name}'.")

    success = rtde_control.forceMode(
        task_frame.tolist(),
        list(compliance.selection_vector),
        list(compliance.wrench),
        compliance.force_type,
        list(compliance.limits),
    )
    if not success:
        raise RuntimeError(f"forceMode returned false for step '{step_name}'.")


def _stop_force_mode(rtde_control: object, step_name: str) -> None:
    if not hasattr(rtde_control, "forceModeStop"):
        return
    try:
        result = rtde_control.forceModeStop()
    except Exception as exc:  # pragma: no cover - depends on the robot controller state.
        print(f"  WARNING: forceModeStop failed after '{step_name}': {exc}")
        return
    if result is False:
        print(f"  WARNING: forceModeStop returned false after '{step_name}'.")


def _force_mode_task_frame(compliance: ComplianceConfig, tcp_pose: np.ndarray) -> np.ndarray:
    if compliance.task_frame == "tcp":
        return np.asarray(tcp_pose, dtype=float).copy()
    if compliance.task_frame == "base":
        return np.zeros(6, dtype=float)
    raise ValueError(f"Unsupported compliance task frame: {compliance.task_frame}")


def _add_preview_root(scene: object, name: str, transform: np.ndarray) -> None:
    if not hasattr(scene, "add_frame"):
        return
    wxyz = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]
    try:
        scene.add_frame(name, wxyz=wxyz, position=transform[:3, 3], show_axes=False)
    except TypeError:
        scene.add_frame(name, wxyz=wxyz, position=transform[:3, 3])


def _tool0_path_points(
    robot: object,
    visualizer: object,
    path: list[JointWaypoint],
    joint_order: list[str],
    display_transform: np.ndarray,
) -> np.ndarray:
    points = []
    for waypoint in path:
        cfg = visualizer._configuration_for_urdf_object(robot, waypoint.q_rad, joint_order)
        robot.update_cfg(cfg)
        transform = display_transform @ robot.get_transform("tool0", "base")
        points.append(transform[:3, 3])
    return np.asarray(points, dtype=float)


def _tcp_path_poses(
    robot: object,
    visualizer: object,
    path: list[JointWaypoint],
    joint_order: list[str],
    display_transform: np.ndarray,
    tcp_offset_ur: np.ndarray,
) -> list[np.ndarray]:
    tcp_offset_transform = _pose_vector_to_transform(tcp_offset_ur)
    tcp_poses: list[np.ndarray] = []
    for waypoint in path:
        cfg = visualizer._configuration_for_urdf_object(robot, waypoint.q_rad, joint_order)
        robot.update_cfg(cfg)
        tool0_transform = display_transform @ robot.get_transform("tool0", "base")
        tcp_poses.append(tool0_transform @ tcp_offset_transform)
    return tcp_poses


def _final_movel_points(tcp_pose: np.ndarray, distance_m: float, steps: int = 30) -> np.ndarray:
    start = tcp_pose[:3, 3]
    if distance_m <= 0.0:
        return np.asarray([start], dtype=float)
    end = start + tcp_pose[:3, :3] @ np.array([0.0, 0.0, -distance_m])
    return np.linspace(start, end, steps)


def _tool_movel_preview_points(
    robot: object,
    visualizer: object,
    steps: list[SequenceStep],
    joint_order: list[str],
    display_transform: np.ndarray,
    tcp_offset_ur: np.ndarray,
) -> np.ndarray:
    tcp_offset_transform = _pose_vector_to_transform(tcp_offset_ur)
    current_tcp_pose: np.ndarray | None = None
    points: list[np.ndarray] = []
    for step in steps:
        if isinstance(step, JointWaypoint):
            cfg = visualizer._configuration_for_urdf_object(robot, step.q_rad, joint_order)
            robot.update_cfg(cfg)
            current_tcp_pose = display_transform @ robot.get_transform("tool0", "base") @ tcp_offset_transform
        elif isinstance(step, ToolFrameMoveL) and current_tcp_pose is not None:
            points.append(current_tcp_pose[:3, 3].copy())
            for offset in step.offsets_m:
                next_tcp_pose = current_tcp_pose.copy()
                next_tcp_pose[:3, 3] = current_tcp_pose[:3, 3] + current_tcp_pose[:3, :3] @ offset
                points.append(next_tcp_pose[:3, 3].copy())
                current_tcp_pose = next_tcp_pose
    return np.asarray(points, dtype=float)


def _add_path(
    scene: object,
    name: str,
    points: np.ndarray,
    color: tuple[int, int, int] = (240, 160, 30),
    line_width: float = 1.5,
) -> None:
    if hasattr(scene, "add_line_segments") and len(points) >= 2:
        scene.add_line_segments(
            name,
            points=np.stack([points[:-1], points[1:]], axis=1),
            colors=np.tile(np.asarray(color, dtype=np.uint8), (len(points) - 1, 2, 1)),
            line_width=line_width,
        )
    if hasattr(scene, "add_point_cloud"):
        scene.add_point_cloud(
            f"{name}_points",
            points=points,
            colors=np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1)),
            point_size=0.006,
        )


def _add_waypoint_markers(
    scene: object,
    robot: object,
    visualizer: object,
    waypoints: list[JointWaypoint],
    joint_order: list[str],
    display_transform: np.ndarray,
) -> None:
    for index, waypoint in enumerate(waypoints):
        cfg = visualizer._configuration_for_urdf_object(robot, waypoint.q_rad, joint_order)
        robot.update_cfg(cfg)
        transform = display_transform @ robot.get_transform("tool0", "base")
        if hasattr(scene, "add_icosphere"):
            scene.add_icosphere(
                f"/motion/waypoints/{index:02d}_{waypoint.name}",
                radius=0.015,
                position=transform[:3, 3],
                color=(220, 70, 55),
            )
        if hasattr(scene, "add_label"):
            try:
                scene.add_label(
                    f"/motion/waypoint_labels/{index:02d}_{waypoint.name}",
                    text=f"{index + 1}: {waypoint.name}",
                    position=transform[:3, 3] + np.array([0.0, 0.0, 0.08]),
                )
            except TypeError:
                pass


def _add_preview_context(scene: object, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)
    center = 0.5 * (min_xyz + max_xyz)
    size = max(1.2, 1.5 * float(np.max(max_xyz[:2] - min_xyz[:2])))
    half = size / 2.0

    if hasattr(scene, "add_grid"):
        try:
            scene.add_grid(
                "/motion/reference_grid",
                width=size,
                height=size,
                position=(center[0], center[1], min_xyz[2] - 0.12),
            )
        except TypeError:
            pass

    if hasattr(scene, "add_frame"):
        scene.add_frame("/motion/preview_origin", axes_length=0.25)
    if hasattr(scene, "add_label"):
        try:
            scene.add_label(
                "/motion/preview_label",
                text="Offline joint-space preview. Orange = tool0 path, red = waypoints.",
                position=(center[0] - half, center[1] - half, max_xyz[2] + 0.25),
            )
        except TypeError:
            pass


def _add_current_tool_marker(scene: object, position: np.ndarray) -> object:
    if hasattr(scene, "add_frame"):
        return scene.add_frame(
            "/motion/current_tool0",
            position=np.asarray(position, dtype=float),
            axes_length=0.08,
            axes_radius=0.003,
        )
    if hasattr(scene, "add_icosphere"):
        return scene.add_icosphere(
            "/motion/current_tool0",
            radius=0.012,
            position=np.asarray(position, dtype=float),
            color=(40, 180, 250),
        )
    raise RuntimeError("This viser scene does not support a current tool marker.")


def _set_preview_camera(server: object, points: np.ndarray) -> None:
    center = np.mean(points, axis=0) if len(points) else np.zeros(3)
    extent = float(np.max(np.linalg.norm(points - center, axis=1))) if len(points) else 1.0
    distance = max(1.6, 2.5 * extent)
    position = center + np.array([0.9 * distance, -1.2 * distance, 0.65 * distance])

    if hasattr(server, "initial_camera"):
        server.initial_camera.position = tuple(position)
        server.initial_camera.look_at = tuple(center)
        if hasattr(server.initial_camera, "up"):
            server.initial_camera.up = (0.0, 0.0, 1.0)

    if hasattr(server, "on_client_connect"):
        @server.on_client_connect
        def _(client: object) -> None:
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(center)
            if hasattr(client.camera, "up"):
                client.camera.up = (0.0, 0.0, 1.0)
