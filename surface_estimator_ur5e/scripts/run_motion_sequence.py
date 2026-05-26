"""Run the saved UR5e joint sequence slowly with ur_rtde."""

from __future__ import annotations

import argparse
from pathlib import Path

from surface_estimator_ur5e.live_robot import DEFAULT_ROBOT_IP
from surface_estimator_ur5e.motion_sequence import (
    DEFAULT_GRIPPER_PORT,
    DEFAULT_TCP_OFFSET_UR,
    load_joint_sequence,
    run_joint_sequence,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=repo_root / "data" / "ur5e_motion_sequence.yaml")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--speed-rad-s", type=float, default=0.05)
    parser.add_argument("--acceleration-rad-s2", type=float, default=0.05)
    parser.add_argument("--linear-speed-m-s", type=float, default=0.01)
    parser.add_argument("--linear-acceleration-m-s2", type=float, default=0.02)
    parser.add_argument("--dwell-s", type=float, default=0.5)
    parser.add_argument("--max-start-delta-deg", type=float, default=15.0)
    parser.add_argument("--final-tool-negative-z-m", type=float, default=0.0)
    parser.add_argument("--gripper-port", type=int, default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--gripper-open-threshold-raw", type=int, default=5)
    parser.add_argument("--skip-gripper-open-check", action="store_true")
    parser.add_argument("--allow-force-mode", action="store_true")
    parser.add_argument("--start-at", default=None)
    parser.add_argument("--stop-before", default=None)
    parser.add_argument("--stop-after", default=None)
    parser.add_argument(
        "--tcp-offset-ur",
        nargs=6,
        type=float,
        default=DEFAULT_TCP_OFFSET_UR.tolist(),
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    waypoints = load_joint_sequence(args.input)
    run_joint_sequence(
        waypoints,
        robot_ip=args.robot_ip,
        speed_rad_s=args.speed_rad_s,
        acceleration_rad_s2=args.acceleration_rad_s2,
        linear_speed_m_s=args.linear_speed_m_s,
        linear_acceleration_m_s2=args.linear_acceleration_m_s2,
        dwell_s=args.dwell_s,
        max_start_delta_deg=args.max_start_delta_deg,
        final_tool_negative_z_m=args.final_tool_negative_z_m,
        tcp_offset_ur=args.tcp_offset_ur,
        gripper_port=args.gripper_port,
        require_gripper_open=not args.skip_gripper_open_check,
        gripper_open_threshold_raw=args.gripper_open_threshold_raw,
        start_at=args.start_at,
        stop_before=args.stop_before,
        stop_after=args.stop_after,
        allow_force_mode=args.allow_force_mode,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
