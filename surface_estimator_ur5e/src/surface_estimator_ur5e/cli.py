"""Command-line interface for surface estimation and visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np

from surface_estimator_ur5e.geometry import estimate_plane_from_points
from surface_estimator_ur5e.io import ContactData, load_contact_file
from surface_estimator_ur5e.live_robot import (
    DEFAULT_ROBOT_IP,
    append_contact_capture,
    capture_current_pose,
)
from surface_estimator_ur5e.mujoco_control import (
    launch_mujoco_control_viewer,
    launch_mujoco_grasp_demo,
)
from surface_estimator_ur5e.mujoco_scene import (
    write_mujoco_control_scene,
    write_mujoco_grasp_scene,
    write_mujoco_scene,
)
from surface_estimator_ur5e.motion_sequence import (
    DEFAULT_GRIPPER_PORT,
    DEFAULT_TCP_OFFSET_UR,
    load_joint_sequence,
    preview_joint_sequence,
    run_joint_sequence,
)
from surface_estimator_ur5e.visualization import launch_visualization


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""

    parser = argparse.ArgumentParser(prog="surface-estimator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "estimate",
        "visualize",
        "validate",
        "export-mujoco",
        "export-mujoco-control",
        "export-mujoco-grasp",
    ):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--input", required=True, type=Path)
        command_parser.add_argument(
            "--normal-direction",
            choices=("auto", "positive_z", "negative_z"),
            default=None,
        )
        command_parser.add_argument(
            "--normal-direction-vector",
            nargs=3,
            type=float,
            default=None,
            metavar=("X", "Y", "Z"),
        )
        command_parser.add_argument("--flip-normal", action="store_true")

    subparsers.choices["visualize"].add_argument("--selected-contact-index", type=int, default=0)
    mujoco_parser = subparsers.choices["export-mujoco"]
    mujoco_parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ur5e_hande_table_scene.xml"),
    )
    mujoco_parser.add_argument("--selected-contact-index", type=int, default=0)
    mujoco_parser.add_argument("--table-margin-m", type=float, default=0.18)
    mujoco_parser.add_argument("--table-thickness-m", type=float, default=0.05)
    mujoco_parser.add_argument("--min-table-half-extent-m", type=float, default=0.35)
    mujoco_parser.add_argument("--compile-check", action="store_true")

    mujoco_control_parser = subparsers.choices["export-mujoco-control"]
    mujoco_control_parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ur5e_hande_table_control_scene.xml"),
    )
    mujoco_control_parser.add_argument("--selected-contact-index", type=int, default=0)
    mujoco_control_parser.add_argument("--table-margin-m", type=float, default=0.18)
    mujoco_control_parser.add_argument("--table-thickness-m", type=float, default=0.05)
    mujoco_control_parser.add_argument("--min-table-half-extent-m", type=float, default=0.35)
    mujoco_control_parser.add_argument("--compile-check", action="store_true")

    mujoco_grasp_parser = subparsers.choices["export-mujoco-grasp"]
    mujoco_grasp_parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/ur5e_hande_tray_grasp_scene.xml"),
    )
    mujoco_grasp_parser.add_argument("--selected-contact-index", type=int, default=0)
    mujoco_grasp_parser.add_argument("--table-margin-m", type=float, default=0.18)
    mujoco_grasp_parser.add_argument("--table-thickness-m", type=float, default=0.05)
    mujoco_grasp_parser.add_argument("--min-table-half-extent-m", type=float, default=0.35)
    mujoco_grasp_parser.add_argument(
        "--tray-collision-mode",
        choices=("mesh", "proxy", "both"),
        default="proxy",
    )
    mujoco_grasp_parser.add_argument("--compile-check", action="store_true")

    run_mujoco_control_parser = subparsers.add_parser("run-mujoco-control")
    run_mujoco_control_parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("models/ur5e_hande_table_control_scene.xml"),
    )
    run_mujoco_control_parser.add_argument("--keyframe", default="contact_1")
    run_mujoco_control_parser.add_argument(
        "--limit-mode",
        choices=("error", "clamp", "ignore"),
        default="error",
    )

    run_mujoco_grasp_parser = subparsers.add_parser("run-mujoco-grasp")
    run_mujoco_grasp_parser.add_argument(
        "--mjcf",
        type=Path,
        default=Path("models/ur5e_hande_tray_grasp_scene.xml"),
    )
    run_mujoco_grasp_parser.add_argument("--keyframe", default="grasp_open")
    run_mujoco_grasp_parser.add_argument("--close-m", type=float, default=0.030)
    run_mujoco_grasp_parser.add_argument("--lift-m", type=float, default=0.08)
    run_mujoco_grasp_parser.add_argument(
        "--attach-mode",
        choices=("kinematic", "physics"),
        default="kinematic",
    )
    run_mujoco_grasp_parser.add_argument(
        "--viewer-camera",
        default="free",
        help="Main MuJoCo viewer camera; use 'free' for the overview free camera.",
    )
    run_mujoco_grasp_parser.add_argument(
        "--realsense-camera",
        default="realsense_rgb",
        help="MJCF camera used for the separate RealSense RGB, depth, and point-cloud windows.",
    )
    run_mujoco_grasp_parser.add_argument(
        "--no-realsense-windows",
        action="store_true",
        help="Disable the separate RealSense RGB, depth, and point-cloud windows.",
    )
    run_mujoco_grasp_parser.add_argument("--realsense-width", type=int, default=640)
    run_mujoco_grasp_parser.add_argument("--realsense-height", type=int, default=480)
    run_mujoco_grasp_parser.add_argument("--realsense-hz", type=float, default=15.0)
    run_mujoco_grasp_parser.add_argument("--pointcloud-stride", type=int, default=4)
    run_mujoco_grasp_parser.add_argument("--pointcloud-max-depth-m", type=float, default=2.0)
    run_mujoco_grasp_parser.add_argument(
        "--limit-mode",
        choices=("error", "clamp", "ignore"),
        default="clamp",
    )

    read_parser = subparsers.add_parser("read-robot")
    read_parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    read_parser.add_argument("--samples", type=int, default=1)
    read_parser.add_argument("--sample-period-s", type=float, default=0.02)

    capture_parser = subparsers.add_parser("capture-contact")
    capture_parser.add_argument("--output", required=True, type=Path)
    capture_parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    capture_parser.add_argument("--name", default=None)
    capture_parser.add_argument("--note", default="")
    capture_parser.add_argument("--samples", type=int, default=10)
    capture_parser.add_argument("--sample-period-s", type=float, default=0.02)

    sequence_parser = subparsers.add_parser("run-sequence")
    sequence_parser.add_argument("--input", required=True, type=Path)
    sequence_parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    sequence_parser.add_argument("--speed-rad-s", type=float, default=0.05)
    sequence_parser.add_argument("--acceleration-rad-s2", type=float, default=0.05)
    sequence_parser.add_argument("--linear-speed-m-s", type=float, default=0.01)
    sequence_parser.add_argument("--linear-acceleration-m-s2", type=float, default=0.02)
    sequence_parser.add_argument("--dwell-s", type=float, default=0.5)
    sequence_parser.add_argument("--max-start-delta-deg", type=float, default=10.0)
    sequence_parser.add_argument("--final-tool-negative-z-m", type=float, default=0.0)
    sequence_parser.add_argument("--gripper-port", type=int, default=DEFAULT_GRIPPER_PORT)
    sequence_parser.add_argument("--gripper-open-threshold-raw", type=int, default=5)
    sequence_parser.add_argument("--skip-gripper-open-check", action="store_true")
    sequence_parser.add_argument("--start-at", default=None)
    sequence_parser.add_argument("--stop-before", default=None)
    sequence_parser.add_argument("--stop-after", default=None)
    sequence_parser.add_argument(
        "--tcp-offset-ur",
        nargs=6,
        type=float,
        default=DEFAULT_TCP_OFFSET_UR.tolist(),
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    sequence_parser.add_argument("--execute", action="store_true")

    preview_parser = subparsers.add_parser("preview-sequence")
    preview_parser.add_argument("--input", required=True, type=Path)
    preview_parser.add_argument("--steps-per-segment", type=int, default=40)
    preview_parser.add_argument("--playback-hz", type=float, default=15.0)
    preview_parser.add_argument("--port", type=int, default=8080)
    preview_parser.add_argument("--final-tool-negative-z-m", type=float, default=0.0)
    preview_parser.add_argument(
        "--tcp-offset-ur",
        nargs=6,
        type=float,
        default=DEFAULT_TCP_OFFSET_UR.tolist(),
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            contact_data = _load_with_warnings(args.input)
            _estimate(contact_data, args)
            print(f"OK: {args.input} is valid.")
            print(f"contacts: {len(contact_data.contacts)}")
            return 0

        if args.command == "read-robot":
            sample = capture_current_pose(
                robot_ip=args.robot_ip,
                samples=args.samples,
                sample_period_s=args.sample_period_s,
            )
            _print_robot_sample(sample)
            return 0

        if args.command == "capture-contact":
            sample = capture_current_pose(
                robot_ip=args.robot_ip,
                samples=args.samples,
                sample_period_s=args.sample_period_s,
            )
            contact = append_contact_capture(
                output_path=args.output,
                sample=sample,
                name=args.name,
                note=args.note,
                robot_ip=args.robot_ip,
            )
            print(f"captured: {contact['name']}")
            print(f"output: {args.output}")
            print(f"tcp_position_m: {_fmt(np.asarray(contact['tcp_position_m']))}")
            print(
                "tcp_rotation_vector_rad: "
                f"{_fmt(np.asarray(contact['tcp_rotation_vector_rad']))}"
            )
            return 0

        if args.command == "run-sequence":
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
                execute=args.execute,
            )
            return 0

        if args.command == "preview-sequence":
            waypoints = load_joint_sequence(args.input)
            preview_joint_sequence(
                waypoints,
                steps_per_segment=args.steps_per_segment,
                playback_hz=args.playback_hz,
                port=args.port,
                final_tool_negative_z_m=args.final_tool_negative_z_m,
                tcp_offset_ur=args.tcp_offset_ur,
            )
            return 0

        if args.command == "run-mujoco-control":
            launch_mujoco_control_viewer(
                args.mjcf,
                keyframe=args.keyframe,
                limit_mode=args.limit_mode,
            )
            return 0

        if args.command == "run-mujoco-grasp":
            launch_mujoco_grasp_demo(
                args.mjcf,
                keyframe=args.keyframe,
                close_m=args.close_m,
                lift_m=args.lift_m,
                limit_mode=args.limit_mode,
                attach_mode=args.attach_mode,
                viewer_camera=args.viewer_camera,
                show_realsense_windows=not args.no_realsense_windows,
                realsense_camera=args.realsense_camera,
                realsense_width=args.realsense_width,
                realsense_height=args.realsense_height,
                realsense_hz=args.realsense_hz,
                pointcloud_stride=args.pointcloud_stride,
                pointcloud_max_depth_m=args.pointcloud_max_depth_m,
            )
            return 0

        contact_data = _load_with_warnings(args.input)
        plane = _estimate(contact_data, args)

        if args.command == "estimate":
            _print_estimate(contact_data, plane)
            return 0

        if args.command == "visualize":
            if (
                args.selected_contact_index < 0
                or args.selected_contact_index >= len(contact_data.contacts)
            ):
                raise ValueError("--selected-contact-index is out of range.")
            launch_visualization(contact_data, plane, args.selected_contact_index)
            return 0

        if args.command == "export-mujoco":
            if (
                args.selected_contact_index < 0
                or args.selected_contact_index >= len(contact_data.contacts)
            ):
                raise ValueError("--selected-contact-index is out of range.")
            metadata = write_mujoco_scene(
                contact_data,
                plane,
                output_path=args.output,
                selected_contact_index=args.selected_contact_index,
                table_margin_m=args.table_margin_m,
                table_thickness_m=args.table_thickness_m,
                min_table_half_extent_m=args.min_table_half_extent_m,
                compile_check=args.compile_check,
            )
            print(f"wrote MuJoCo scene: {metadata.output_path}")
            print(f"fk source: {metadata.fk_source}")
            print(f"table center: {_fmt(metadata.table_center_m)}")
            print(f"table half extents: {_fmt(metadata.table_half_extents_m)}")
            print(f"plane normal in MuJoCo world: {_fmt(metadata.plane_normal_world)}")
            return 0

        if args.command == "export-mujoco-control":
            if (
                args.selected_contact_index < 0
                or args.selected_contact_index >= len(contact_data.contacts)
            ):
                raise ValueError("--selected-contact-index is out of range.")
            metadata = write_mujoco_control_scene(
                contact_data,
                plane,
                output_path=args.output,
                selected_contact_index=args.selected_contact_index,
                table_margin_m=args.table_margin_m,
                table_thickness_m=args.table_thickness_m,
                min_table_half_extent_m=args.min_table_half_extent_m,
                compile_check=args.compile_check,
            )
            print(f"wrote actuated MuJoCo scene: {metadata.output_path}")
            print(f"fk source: {metadata.fk_source}")
            print(f"table center: {_fmt(metadata.table_center_m)}")
            print(f"table half extents: {_fmt(metadata.table_half_extents_m)}")
            print(f"plane normal in MuJoCo world: {_fmt(metadata.plane_normal_world)}")
            return 0

        if args.command == "export-mujoco-grasp":
            if (
                args.selected_contact_index < 0
                or args.selected_contact_index >= len(contact_data.contacts)
            ):
                raise ValueError("--selected-contact-index is out of range.")
            metadata = write_mujoco_grasp_scene(
                contact_data,
                plane,
                output_path=args.output,
                selected_contact_index=args.selected_contact_index,
                table_margin_m=args.table_margin_m,
                table_thickness_m=args.table_thickness_m,
                min_table_half_extent_m=args.min_table_half_extent_m,
                tray_collision_mode=args.tray_collision_mode,
                compile_check=args.compile_check,
            )
            print(f"wrote MuJoCo grasp scene: {metadata.output_path}")
            print(f"fk source: {metadata.fk_source}")
            print(f"table center: {_fmt(metadata.table_center_m)}")
            print(f"table half extents: {_fmt(metadata.table_half_extents_m)}")
            print(f"plane normal in MuJoCo world: {_fmt(metadata.plane_normal_world)}")
            return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


def _load_with_warnings(path: Path) -> ContactData:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        contact_data = load_contact_file(path)
    for warning in caught:
        print(f"warning: {warning.message}", file=sys.stderr)
    return contact_data


def _estimate(contact_data: ContactData, args: argparse.Namespace):
    normal_hint: str | np.ndarray | None
    if args.normal_direction_vector is not None:
        normal_hint = np.asarray(args.normal_direction_vector, dtype=float)
    elif args.normal_direction is not None:
        normal_hint = args.normal_direction
    elif contact_data.normal_direction_vector is not None:
        normal_hint = contact_data.normal_direction_vector
    else:
        normal_hint = contact_data.normal_direction
    return estimate_plane_from_points(
        contact_data.contact_points(),
        normal_hint=normal_hint,
        flip=bool(contact_data.flip_normal or args.flip_normal),
    )


def _print_estimate(contact_data: ContactData, plane) -> None:
    print(f"number of contacts: {len(contact_data.contacts)}")
    print(f"plane centroid: {_fmt(plane.centroid)}")
    print(f"plane normal: {_fmt(plane.normal)}")
    print(f"plane equation: {_fmt(plane.normal)} . x + {plane.d:.9f} = 0")
    print(f"RMS error: {plane.rms_error:.9f} m")
    print("signed distances:")
    for contact, distance in zip(contact_data.contacts, plane.signed_distances, strict=True):
        print(f"  {contact.name}: {distance:.9f} m")


def _print_robot_sample(sample) -> None:
    print(f"q_rad: {_fmt(sample.q_rad)}")
    print(f"q_deg: {_fmt(np.rad2deg(sample.q_rad))}")
    print(f"tcp_position_m: {_fmt(sample.tcp_position_m)}")
    print(f"tcp_rotation_vector_rad: {_fmt(sample.tcp_rotation_vector_rad)}")
    print(f"tcp_orientation_xyzw: {_fmt(sample.tcp_orientation_xyzw)}")


def _fmt(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value):.9f}" for value in values) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
