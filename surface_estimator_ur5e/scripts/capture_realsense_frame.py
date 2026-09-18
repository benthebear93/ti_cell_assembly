#!/usr/bin/env python3
"""Capture one RealSense color/depth frame for assembly alignment debugging."""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from surface_estimator_ur5e.calibration import DEFAULT_CALIBRATION, load_tool_to_camera
from surface_estimator_ur5e.robot_io import DEFAULT_ROBOT_IP, read_current_tcp_pose
from surface_estimator_ur5e.transforms import ur_pose_to_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one RealSense color frame and aligned depth frame."
    )
    parser.add_argument(
        "--output-dir",
        default="images/realsense_alignment",
        help="Directory where color/depth/metadata files are saved.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Filename prefix. Defaults to realsense_<timestamp>.",
    )
    parser.add_argument("--serial", help="RealSense serial number.")
    parser.add_argument("--color-width", type=int, default=1920)
    parser.add_argument("--color-height", type=int, default=1080)
    parser.add_argument("--depth-width", type=int, default=1280)
    parser.add_argument("--depth-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save color, depth in meters, and intrinsics to a compressed NPZ.",
    )
    parser.add_argument(
        "--robot-ip",
        default=DEFAULT_ROBOT_IP,
        help="UR robot IP used only for optional metadata pose capture.",
    )
    parser.add_argument(
        "--calibration",
        default=DEFAULT_CALIBRATION,
        help="Hand-eye XML used to write base_to_camera metadata when RTDE is available.",
    )
    parser.add_argument(
        "--no-robot-pose",
        action="store_true",
        help="Do not connect to RTDE to read the current TCP pose.",
    )
    return parser.parse_args()


def stream_intrinsics_dict(intrinsics) -> dict[str, float | int | str | list[float]]:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def start_pipeline(args: argparse.Namespace) -> rs.pipeline:
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(
        rs.stream.color,
        args.color_width,
        args.color_height,
        rs.format.bgr8,
        args.fps,
    )
    config.enable_stream(
        rs.stream.depth,
        args.depth_width,
        args.depth_height,
        rs.format.z16,
        args.fps,
    )
    pipeline.start(config)
    return pipeline


def wait_for_aligned_frames(
    pipeline: rs.pipeline,
    align_to_color: rs.align,
    warmup_frames: int,
    timeout_ms: int,
) -> tuple[rs.video_frame, rs.depth_frame]:
    if warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative.")

    color_frame = None
    depth_frame = None
    for _ in range(warmup_frames + 1):
        frames = pipeline.wait_for_frames(timeout_ms)
        aligned = align_to_color.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

    if not color_frame or not depth_frame:
        raise RuntimeError("Timed out before receiving both color and aligned depth frames.")
    return color_frame, depth_frame


def depth_visualization(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        scaled = np.zeros(depth_mm.shape, dtype=np.uint8)
    else:
        near = float(np.percentile(valid, 2.0))
        far = float(np.percentile(valid, 98.0))
        if not math.isfinite(near) or not math.isfinite(far) or far <= near:
            near = float(valid.min())
            far = float(valid.max())
        if far <= near:
            scaled = np.zeros(depth_mm.shape, dtype=np.uint8)
        else:
            clipped = np.clip(depth_mm.astype(np.float32), near, far)
            scaled = ((clipped - near) * (255.0 / (far - near))).astype(np.uint8)
            scaled[depth_mm == 0] = 0
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def read_robot_metadata(robot_ip: str, calibration_path: Path) -> dict:
    receive, pose = read_current_tcp_pose(robot_ip)
    try:
        tcp_pose = pose.tolist()
    finally:
        receive.disconnect()
    base_to_tool = ur_pose_to_transform(tcp_pose)
    tool_to_camera = load_tool_to_camera(calibration_path)
    base_to_camera = base_to_tool @ tool_to_camera
    return {
        "robot_ip": robot_ip,
        "actual_tcp_pose_ur": tcp_pose,
        "calibration": str(calibration_path),
        "tool_to_camera": tool_to_camera.tolist(),
        "base_to_camera": base_to_camera.tolist(),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix
    if prefix is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        prefix = f"realsense_{timestamp}"

    pipeline = start_pipeline(args)
    align_to_color = rs.align(rs.stream.color)
    try:
        color_frame, depth_frame = wait_for_aligned_frames(
            pipeline,
            align_to_color,
            args.warmup_frames,
            args.timeout_ms,
        )

        color_bgr = np.asanyarray(color_frame.get_data())
        depth_mm = np.asanyarray(depth_frame.get_data()).astype(np.uint16)
        depth_scale = float(
            pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
        )
        depth_m = depth_mm.astype(np.float32) * depth_scale

        color_path = output_dir / f"{prefix}_color.png"
        depth_path = output_dir / f"{prefix}_depth_mm.png"
        depth_vis_path = output_dir / f"{prefix}_depth_vis.png"
        metadata_path = output_dir / f"{prefix}_metadata.yaml"
        npz_path = output_dir / f"{prefix}_frame.npz"

        cv2.imwrite(str(color_path), color_bgr)
        cv2.imwrite(str(depth_path), depth_mm)
        cv2.imwrite(str(depth_vis_path), depth_visualization(depth_mm))

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        valid_depth = depth_m[depth_mm > 0]
        metadata = {
            "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "serial": args.serial,
            "streams": {
                "color_requested": [args.color_width, args.color_height, args.fps],
                "depth_requested": [args.depth_width, args.depth_height, args.fps],
                "aligned_depth_to": "color",
            },
            "files": {
                "color": str(color_path),
                "depth_mm": str(depth_path),
                "depth_visualization": str(depth_vis_path),
            },
            "depth_scale_m_per_unit": depth_scale,
            "color_intrinsics": stream_intrinsics_dict(color_intrinsics),
            "aligned_depth_intrinsics": stream_intrinsics_dict(depth_intrinsics),
            "depth_stats_m": {
                "valid_pixels": int(valid_depth.size),
                "min": float(valid_depth.min()) if valid_depth.size else None,
                "median": float(np.median(valid_depth)) if valid_depth.size else None,
                "max": float(valid_depth.max()) if valid_depth.size else None,
            },
        }

        if args.save_npz:
            np.savez_compressed(
                npz_path,
                color_bgr=color_bgr,
                depth_m=depth_m,
                depth_mm=depth_mm,
                color_intrinsics=np.array(
                    [
                        color_intrinsics.fx,
                        color_intrinsics.fy,
                        color_intrinsics.ppx,
                        color_intrinsics.ppy,
                    ],
                    dtype=np.float64,
                ),
            )
            metadata["files"]["npz"] = str(npz_path)

        if not args.no_robot_pose:
            try:
                calibration_path = Path(args.calibration)
                metadata["robot"] = read_robot_metadata(args.robot_ip, calibration_path)
            except Exception as exc:  # noqa: BLE001 - metadata is useful but optional.
                metadata["robot_pose_error"] = repr(exc)

        with metadata_path.open("w") as file:
            yaml.safe_dump(metadata, file, sort_keys=False)

        print("Saved RealSense capture:")
        print(f"  color: {color_path}")
        print(f"  depth: {depth_path}")
        print(f"  depth visualization: {depth_vis_path}")
        print(f"  metadata: {metadata_path}")
        if args.save_npz:
            print(f"  npz: {npz_path}")
        if "robot_pose_error" in metadata:
            print(f"  robot pose metadata failed: {metadata['robot_pose_error']}")
        elif "robot" in metadata:
            tcp = metadata["robot"]["actual_tcp_pose_ur"]
            print(
                "  actual TCP pose: "
                + " ".join(f"{value:.6f}" for value in tcp)
            )
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
