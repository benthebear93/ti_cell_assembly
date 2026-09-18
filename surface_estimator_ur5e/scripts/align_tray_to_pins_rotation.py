#!/usr/bin/env python3
"""Rotate the current TCP to match the four-pin insertion frame."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from surface_estimator_ur5e.calibration import load_marker_transform
from surface_estimator_ur5e.live_robot import DEFAULT_ROBOT_IP
from surface_estimator_ur5e.motion_sequence import (
    DEFAULT_GRIPPER_PORT,
    GripperCommand,
    RobotiqHandEGripper,
)
from surface_estimator_ur5e.robot_io import (
    connect_rtde_control,
    read_current_tcp_pose,
    set_tcp_offset,
)
from surface_estimator_ur5e.transforms import (
    marker_relative_transform,
    rotation_delta_deg,
    ur_pose_to_transform,
)
from surface_estimator_ur5e.workcell_geometry import (
    DEFAULT_ASSEMBLY_LOCAL_YAW_DEG,
    DEFAULT_ASSEMBLY_ORIGIN_X_MM,
    DEFAULT_ASSEMBLY_ORIGIN_Y_MM,
    DEFAULT_ASSEMBLY_ORIGIN_Z_MM,
    DEFAULT_ASSEMBLY_RPY_DEG,
    DEFAULT_MARKER_POSE,
    DEFAULT_TCP_OFFSET_UR,
    DEFAULT_TI_ASSEMBLY_OBJ,
    DEFAULT_VIRTUAL_TRAY_HANDLE_ROOT_Y_MM,
    DEFAULT_VIRTUAL_TRAY_HANDLE_SCALE,
    DEFAULT_VIRTUAL_TRAY_LOCAL_RX_DEG,
    DEFAULT_VIRTUAL_TRAY_OBJ,
    DEFAULT_VIRTUAL_TRAY_TCP_Z_MM,
    floor_constrained_marker_transform,
    four_pin_feature_frame,
    load_tray_geometry,
    resolve_project_path,
    tray_floor_clearance_m,
    with_base_z_lift,
)
from surface_estimator_ur5e.workcell_geometry import (
    make_four_pin_rotation_target as make_rotation_only_target,
)

DEFAULT_PLAN_OUTPUT = Path("data/markers/four_pin_rotation_align_plan.yaml")
DEFAULT_SAFE_FLOOR_CLEARANCE_MM = 15.0
DEFAULT_MIN_PRE_ROTATION_LIFT_MM = 15.0
DEFAULT_MAX_SAFETY_LIFT_MM = 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the current TCP position and rotate the TCP frame to match the four-pin "
            "insertion frame computed from the saved marker pose and holder OBJ."
        )
    )
    parser.add_argument("--marker-pose", type=Path, default=DEFAULT_MARKER_POSE)
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--assembly-obj", "--holder-obj", dest="assembly_obj", type=Path, default=DEFAULT_TI_ASSEMBLY_OBJ)
    parser.add_argument(
        "--assembly-origin-x-mm",
        "--holder-origin-x-mm",
        dest="assembly_origin_x_mm",
        type=float,
        default=DEFAULT_ASSEMBLY_ORIGIN_X_MM,
    )
    parser.add_argument(
        "--assembly-origin-y-mm",
        "--holder-origin-y-mm",
        dest="assembly_origin_y_mm",
        type=float,
        default=DEFAULT_ASSEMBLY_ORIGIN_Y_MM,
    )
    parser.add_argument(
        "--assembly-origin-z-mm",
        "--holder-origin-z-mm",
        dest="assembly_origin_z_mm",
        type=float,
        default=DEFAULT_ASSEMBLY_ORIGIN_Z_MM,
    )
    parser.add_argument(
        "--assembly-rpy-deg",
        "--holder-rpy-deg",
        dest="assembly_rpy_deg",
        nargs=3,
        type=float,
        default=DEFAULT_ASSEMBLY_RPY_DEG,
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument(
        "--assembly-local-yaw-deg",
        "--holder-local-yaw-deg",
        dest="assembly_local_yaw_deg",
        type=float,
        default=DEFAULT_ASSEMBLY_LOCAL_YAW_DEG,
    )
    parser.add_argument(
        "--marker-frame-mode",
        choices=("floor", "raw"),
        default="floor",
        help="Use the same marker frame convention as the visualization.",
    )
    parser.add_argument(
        "--tcp-rotation-offset-rpy-deg",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Optional local offset from the four-pin frame to the desired TCP frame.",
    )
    parser.add_argument(
        "--tray-center-tcp-z-mm",
        type=float,
        default=DEFAULT_VIRTUAL_TRAY_TCP_Z_MM,
        help="Held tray center offset in current/target TCP frame, used for reporting only.",
    )
    parser.add_argument("--tray-obj", type=Path, default=DEFAULT_VIRTUAL_TRAY_OBJ)
    parser.add_argument("--tray-local-rx-deg", type=float, default=DEFAULT_VIRTUAL_TRAY_LOCAL_RX_DEG)
    parser.add_argument("--tray-handle-root-y-mm", type=float, default=DEFAULT_VIRTUAL_TRAY_HANDLE_ROOT_Y_MM)
    parser.add_argument("--tray-handle-scale", type=float, default=DEFAULT_VIRTUAL_TRAY_HANDLE_SCALE)
    parser.add_argument(
        "--safe-floor-clearance-mm",
        type=float,
        default=DEFAULT_SAFE_FLOOR_CLEARANCE_MM,
        help="Minimum tray bounding-box clearance above the marker floor during rotation.",
    )
    parser.add_argument(
        "--max-safety-lift-mm",
        type=float,
        default=DEFAULT_MAX_SAFETY_LIFT_MM,
        help="Abort execute if the required pre-rotation lift is larger than this.",
    )
    parser.add_argument(
        "--min-pre-rotation-lift-mm",
        type=float,
        default=DEFAULT_MIN_PRE_ROTATION_LIFT_MM,
        help="Always lift at least this far before rotating, because the tray starts on the floor.",
    )
    parser.add_argument(
        "--skip-safety-lift",
        action="store_true",
        help="Do not lift before rotating; execute aborts if floor clearance is unsafe.",
    )
    parser.add_argument("--speed-m-s", type=float, default=0.01)
    parser.add_argument("--acceleration-m-s2", type=float, default=0.02)
    parser.add_argument("--max-rotation-deg", type=float, default=120.0)
    parser.add_argument("--allow-large-rotation", action="store_true")
    parser.add_argument(
        "--tcp-offset-ur",
        nargs=6,
        type=float,
        default=DEFAULT_TCP_OFFSET_UR,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Active UR TCP offset to set before reading and moving.",
    )
    parser.add_argument("--no-set-tcp", action="store_true")
    parser.add_argument("--close-gripper-first", action="store_true")
    parser.add_argument("--gripper-close-percent", type=float, default=75.0)
    parser.add_argument("--gripper-speed", type=int, default=80)
    parser.add_argument("--gripper-force", type=int, default=100)
    parser.add_argument("--gripper-port", type=int, default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--gripper-post-dwell-s", type=float, default=0.5)
    parser.add_argument(
        "--save-plan",
        nargs="?",
        const=str(DEFAULT_PLAN_OUTPUT),
        help="Save the computed rotation-only target plan YAML.",
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    numeric = [
        args.assembly_origin_x_mm,
        args.assembly_origin_y_mm,
        args.assembly_origin_z_mm,
        args.assembly_local_yaw_deg,
        args.tray_center_tcp_z_mm,
        args.speed_m_s,
        args.acceleration_m_s2,
        args.max_rotation_deg,
        args.gripper_close_percent,
        args.gripper_post_dwell_s,
        args.tray_local_rx_deg,
        args.tray_handle_root_y_mm,
        args.tray_handle_scale,
        args.safe_floor_clearance_mm,
        args.min_pre_rotation_lift_mm,
        args.max_safety_lift_mm,
    ]
    numeric.extend(args.assembly_rpy_deg)
    numeric.extend(args.tcp_rotation_offset_rpy_deg)
    numeric.extend(args.tcp_offset_ur)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("All numeric arguments must be finite.")
    if args.speed_m_s <= 0.0:
        raise ValueError("--speed-m-s must be positive.")
    if args.acceleration_m_s2 <= 0.0:
        raise ValueError("--acceleration-m-s2 must be positive.")
    if args.max_rotation_deg < 0.0:
        raise ValueError("--max-rotation-deg must be non-negative.")
    if not 0.0 <= args.gripper_close_percent <= 100.0:
        raise ValueError("--gripper-close-percent must be between 0 and 100.")
    if not 0 <= args.gripper_speed <= 255:
        raise ValueError("--gripper-speed must be between 0 and 255.")
    if not 0 <= args.gripper_force <= 255:
        raise ValueError("--gripper-force must be between 0 and 255.")
    if not 1 <= args.gripper_port <= 65535:
        raise ValueError("--gripper-port must be a valid TCP port.")
    if args.gripper_post_dwell_s < 0.0:
        raise ValueError("--gripper-post-dwell-s must be non-negative.")
    if not 0.0 < args.tray_handle_scale <= 1.0:
        raise ValueError("--tray-handle-scale must be > 0 and <= 1.")
    if args.safe_floor_clearance_mm < 0.0:
        raise ValueError("--safe-floor-clearance-mm must be non-negative.")
    if args.min_pre_rotation_lift_mm < 0.0:
        raise ValueError("--min-pre-rotation-lift-mm must be non-negative.")
    if args.max_safety_lift_mm < 0.0:
        raise ValueError("--max-safety-lift-mm must be non-negative.")


def compute_base_t_pin_frame(args: argparse.Namespace) -> tuple[int | None, np.ndarray, float]:
    marker_id, _marker_length_m, base_t_marker_raw, _marker_data = load_marker_transform(
        resolve_project_path(args.marker_pose)
    )
    base_t_marker = (
        floor_constrained_marker_transform(base_t_marker_raw)
        if args.marker_frame_mode == "floor"
        else base_t_marker_raw
    )
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
    assembly_t_pin, _selected_pin_centers, _adjacent_pin_centers = four_pin_feature_frame(
        resolve_project_path(args.assembly_obj)
    )
    return marker_id, base_t_marker @ marker_t_assembly @ assembly_t_pin, float(base_t_marker[2, 3])


def tray_center_position(tcp_pose_ur: np.ndarray, tray_center_tcp_z_mm: float) -> np.ndarray:
    rotation = Rotation.from_rotvec(tcp_pose_ur[3:6]).as_matrix()
    return tcp_pose_ur[:3] + rotation @ np.array([0.0, 0.0, tray_center_tcp_z_mm / 1000.0])


def close_gripper(args: argparse.Namespace) -> None:
    command = GripperCommand(
        name="align_tray_to_pins_close",
        position_percent=args.gripper_close_percent,
        speed=args.gripper_speed,
        force=args.gripper_force,
        wait=True,
        timeout_s=8.0,
        post_dwell_s=args.gripper_post_dwell_s,
    )
    gripper = RobotiqHandEGripper(args.robot_ip, port=args.gripper_port)
    try:
        gripper.connect()
        start_position = gripper.get_position()
        print(f"gripper raw position before close: {start_position} (0=open, 255=closed)")
        print(
            "Closing gripper before rotation: "
            f"{args.gripper_close_percent:.1f}% closed, "
            f"speed={args.gripper_speed}, force={args.gripper_force}"
        )
        gripper.move_to_percent(command)
        end_position = gripper.get_position()
        print(f"gripper raw position after close: {end_position}")
    finally:
        gripper.close()


def save_plan(
    path: Path,
    args: argparse.Namespace,
    marker_id: int | None,
    base_t_pin: np.ndarray,
    floor_z_m: float,
    current_tcp_pose_ur: np.ndarray,
    target_tcp_pose_ur: np.ndarray,
    lifted_current_tcp_pose_ur: np.ndarray,
    lifted_target_tcp_pose_ur: np.ndarray,
    angle_delta_deg: float,
    current_clearance_m: float,
    target_clearance_m: float,
    safety_lift_m: float,
) -> None:
    payload = {
        "marker_id": marker_id,
        "marker_pose_file": str(args.marker_pose),
        "marker_frame_mode": args.marker_frame_mode,
        "base_T_four_pin_center": matrix_list(base_t_pin),
        "marker_floor_z_m": round(float(floor_z_m), 10),
        "tcp_rotation_offset_rpy_deg": round_list(np.asarray(args.tcp_rotation_offset_rpy_deg)),
        "current_tcp_pose_ur": round_list(current_tcp_pose_ur),
        "target_tcp_pose_ur": round_list(target_tcp_pose_ur),
        "lifted_current_tcp_pose_ur": round_list(lifted_current_tcp_pose_ur),
        "lifted_target_tcp_pose_ur": round_list(lifted_target_tcp_pose_ur),
        "rotation_delta_deg": round(float(angle_delta_deg), 6),
        "tray_center_tcp_z_m": round(float(args.tray_center_tcp_z_mm / 1000.0), 10),
        "tray_local_rx_deg": round(float(args.tray_local_rx_deg), 6),
        "safe_floor_clearance_m": round(float(args.safe_floor_clearance_mm / 1000.0), 10),
        "min_pre_rotation_lift_m": round(float(args.min_pre_rotation_lift_mm / 1000.0), 10),
        "current_tray_floor_clearance_m": round(float(current_clearance_m), 10),
        "target_tray_floor_clearance_without_lift_m": round(float(target_clearance_m), 10),
        "safety_lift_m": round(float(safety_lift_m), 10),
        "current_tray_center_m": round_list(
            tray_center_position(current_tcp_pose_ur, args.tray_center_tcp_z_mm)
        ),
        "target_rotation_tray_center_m": round_list(
            tray_center_position(target_tcp_pose_ur, args.tray_center_tcp_z_mm)
        ),
        "lifted_target_rotation_tray_center_m": round_list(
            tray_center_position(lifted_target_tcp_pose_ur, args.tray_center_tcp_z_mm)
        ),
        "four_pin_center_m": round_list(base_t_pin[:3, 3]),
        "robot_ip": args.robot_ip,
        "tcp_offset_ur": round_list(np.asarray(args.tcp_offset_ur)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def round_list(values: np.ndarray, decimals: int = 10) -> list[float]:
    return [round(float(value), decimals) for value in np.asarray(values, dtype=float).reshape(-1)]


def matrix_list(matrix: np.ndarray, decimals: int = 10) -> list[list[float]]:
    return [[round(float(value), decimals) for value in row] for row in np.asarray(matrix)]


def fmt(values: np.ndarray, decimals: int = 6) -> str:
    return "[" + ", ".join(f"{float(value): .{decimals}f}" for value in np.asarray(values).reshape(-1)) + "]"


def main() -> None:
    args = parse_args()
    validate_args(args)

    marker_id, base_t_pin, floor_z_m = compute_base_t_pin_frame(args)
    tray_vertices, tray_attachment_local = load_tray_geometry(
        resolve_project_path(args.tray_obj), args.tray_handle_root_y_mm, args.tray_handle_scale
    )

    rtde_receive = None
    rtde_control = None
    try:
        if args.execute or not args.no_set_tcp:
            rtde_control = connect_rtde_control(args.robot_ip)
        if not args.no_set_tcp:
            set_tcp_offset(rtde_control, args.tcp_offset_ur)

        rtde_receive, current_tcp_pose_ur = read_current_tcp_pose(args.robot_ip)
        target_tcp_pose_ur = make_rotation_only_target(
            current_tcp_pose_ur,
            base_t_pin,
            args.tcp_rotation_offset_rpy_deg,
        )
        angle_delta_deg = rotation_delta_deg(current_tcp_pose_ur, target_tcp_pose_ur)
        current_tray_center = tray_center_position(current_tcp_pose_ur, args.tray_center_tcp_z_mm)
        target_tray_center = tray_center_position(target_tcp_pose_ur, args.tray_center_tcp_z_mm)
        current_clearance_m = tray_floor_clearance_m(
            ur_pose_to_transform(current_tcp_pose_ur),
            tray_vertices,
            tray_attachment_local,
            floor_z_m,
            args.tray_center_tcp_z_mm,
            args.tray_local_rx_deg,
        )
        target_clearance_m = tray_floor_clearance_m(
            ur_pose_to_transform(target_tcp_pose_ur),
            tray_vertices,
            tray_attachment_local,
            floor_z_m,
            args.tray_center_tcp_z_mm,
            args.tray_local_rx_deg,
        )
        requested_clearance_m = args.safe_floor_clearance_mm / 1000.0
        safety_lift_m = 0.0
        if not args.skip_safety_lift:
            safety_lift_m = max(
                args.min_pre_rotation_lift_mm / 1000.0,
                requested_clearance_m - current_clearance_m,
                requested_clearance_m - target_clearance_m,
            )
        lifted_current_tcp_pose_ur = with_base_z_lift(current_tcp_pose_ur, safety_lift_m)
        lifted_target_tcp_pose_ur = with_base_z_lift(target_tcp_pose_ur, safety_lift_m)
        lifted_target_tray_center = tray_center_position(
            lifted_target_tcp_pose_ur,
            args.tray_center_tcp_z_mm,
        )

        print("Four-pin rotation alignment plan:")
        print(f"  marker pose: {args.marker_pose}")
        print(f"  marker_id: {marker_id}")
        print(f"  marker_frame_mode: {args.marker_frame_mode}")
        print(f"  set tcp before plan: {not args.no_set_tcp}")
        print(f"  tcp offset UR: {fmt(np.asarray(args.tcp_offset_ur))}")
        print(
            "  holder origin marker mm: "
            f"{fmt(np.asarray([args.assembly_origin_x_mm, args.assembly_origin_y_mm, args.assembly_origin_z_mm]), decimals=3)}"
        )
        print(f"  holder rpy marker deg: {fmt(np.asarray(args.assembly_rpy_deg), decimals=3)}")
        print(f"  holder local yaw deg: {args.assembly_local_yaw_deg:.3f}")
        print(f"  four_pin_center position base m: {fmt(base_t_pin[:3, 3])}")
        print(f"  marker floor z in base m: {floor_z_m:.6f} (base +Z is physical down)")
        print(f"  current_tcp_pose_ur: {fmt(current_tcp_pose_ur)}")
        print(f"  target_tcp_pose_ur:  {fmt(target_tcp_pose_ur)}")
        print(f"  rotation_delta_deg: {angle_delta_deg:.3f}")
        print(f"  tray adjustment frame offset in TCP z mm: {args.tray_center_tcp_z_mm:.3f}")
        print(f"  tray local rx deg: {args.tray_local_rx_deg:.3f}")
        print(f"  current tray center base m: {fmt(current_tray_center)}")
        print(f"  tray center after rotation-only base m: {fmt(target_tray_center)}")
        print(f"  current tray floor clearance mm: {1000.0 * current_clearance_m:.3f}")
        print(f"  target tray floor clearance without lift mm: {1000.0 * target_clearance_m:.3f}")
        print(f"  requested safe floor clearance mm: {args.safe_floor_clearance_mm:.3f}")
        print(f"  minimum pre-rotation lift mm: {args.min_pre_rotation_lift_mm:.3f}")
        print(f"  planned safety lift along base -Z mm: {1000.0 * safety_lift_m:.3f}")
        print(f"  lifted_target_tcp_pose_ur: {fmt(lifted_target_tcp_pose_ur)}")
        print(f"  lifted target tray center base m: {fmt(lifted_target_tray_center)}")
        print(
            "  target tray center to four-pin center mm: "
            f"{1000.0 * np.linalg.norm(target_tray_center - base_t_pin[:3, 3]):.3f}"
        )
        print(
            "  lifted target tray center to four-pin center mm: "
            f"{1000.0 * np.linalg.norm(lifted_target_tray_center - base_t_pin[:3, 3]):.3f}"
        )

        if args.save_plan is not None:
            save_path = resolve_project_path(Path(args.save_plan))
            save_plan(
                save_path,
                args,
                marker_id,
                base_t_pin,
                floor_z_m,
                current_tcp_pose_ur,
                target_tcp_pose_ur,
                lifted_current_tcp_pose_ur,
                lifted_target_tcp_pose_ur,
                angle_delta_deg,
                current_clearance_m,
                target_clearance_m,
                safety_lift_m,
            )
            print(f"Saved rotation plan to {save_path}")

        if not args.execute:
            print("DRY RUN ONLY: add --execute to rotate the robot TCP.")
            return

        if angle_delta_deg > args.max_rotation_deg and not args.allow_large_rotation:
            raise RuntimeError(
                f"Rotation delta is {angle_delta_deg:.1f} deg, above "
                f"--max-rotation-deg {args.max_rotation_deg:.1f}. Re-check the frame or add "
                "--allow-large-rotation deliberately."
            )
        if safety_lift_m > args.max_safety_lift_mm / 1000.0:
            raise RuntimeError(
                f"Required safety lift is {1000.0 * safety_lift_m:.1f} mm, above "
                f"--max-safety-lift-mm {args.max_safety_lift_mm:.1f}. Re-check the pose."
            )
        if args.skip_safety_lift:
            worst_clearance_m = min(current_clearance_m, target_clearance_m)
            if worst_clearance_m < requested_clearance_m:
                raise RuntimeError(
                    f"Skipping safety lift would leave only {1000.0 * worst_clearance_m:.1f} mm "
                    f"floor clearance, below requested {args.safe_floor_clearance_mm:.1f} mm."
                )

        if args.close_gripper_first:
            close_gripper(args)
            time.sleep(0.1)

        if safety_lift_m > 1e-6:
            print(
                f"Lifting away from floor by {1000.0 * safety_lift_m:.1f} mm along base -Z "
                "before rotation."
            )
            lift_success = rtde_control.moveL(
                lifted_current_tcp_pose_ur.tolist(),
                args.speed_m_s,
                args.acceleration_m_s2,
                False,
            )
            if not lift_success:
                raise RuntimeError("Safety lift moveL returned false.")

        print(
            f"Executing lifted rotation-only moveL at speed={args.speed_m_s:.4f} m/s, "
            f"acceleration={args.acceleration_m_s2:.4f} m/s^2"
        )
        success = rtde_control.moveL(
            lifted_target_tcp_pose_ur.tolist(),
            args.speed_m_s,
            args.acceleration_m_s2,
            False,
        )
        if not success:
            raise RuntimeError("moveL returned false.")
        final_tcp_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
        print(f"final_tcp_pose_ur: {fmt(final_tcp_pose_ur)}")
        print("Four-pin rotation alignment complete.")
    finally:
        if rtde_control is not None and hasattr(rtde_control, "stopScript"):
            rtde_control.stopScript()
        if rtde_receive is not None and hasattr(rtde_receive, "disconnect"):
            rtde_receive.disconnect()
        if rtde_control is not None and hasattr(rtde_control, "disconnect"):
            rtde_control.disconnect()


if __name__ == "__main__":
    main()
