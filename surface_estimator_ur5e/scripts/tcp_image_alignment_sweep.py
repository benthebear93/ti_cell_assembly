#!/usr/bin/env python3
"""Image-guided TCP local rotation/translation sweep for fine tray/pin alignment."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from surface_estimator_ur5e.live_robot import DEFAULT_ROBOT_IP
from surface_estimator_ur5e.robot_io import (
    connect_rtde_control as open_rtde_control,
)
from surface_estimator_ur5e.robot_io import (
    connect_rtde_receive as open_rtde_receive,
)
from surface_estimator_ur5e.robot_io import (
    set_tcp_offset,
)
from surface_estimator_ur5e.transforms import (
    pose_with_local_xz_adjustment as pose_with_local_adjustment,
)
from surface_estimator_ur5e.workcell_geometry import DEFAULT_TCP_OFFSET_UR

DEFAULT_REALSENSE_SERIAL = "261322073147"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adjust the current TCP with small local-axis rotation and local X/Z translation "
            "steps, then save a RealSense RGB image after every step. Use this for "
            "human-in-the-loop pin/hole fine alignment."
        )
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--serial", default=DEFAULT_REALSENSE_SERIAL)
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="y",
        help="TCP local axis to rotate about. Default is local Y.",
    )
    parser.add_argument("--step-deg", type=float, default=0.5)
    parser.add_argument(
        "--max-abs-angle-deg",
        type=float,
        default=5.0,
        help="Abort commands that would exceed this absolute angle from the start pose.",
    )
    parser.add_argument(
        "--translation-step-mm",
        type=float,
        default=0.5,
        help="TCP local X/Z translation increment for image-guided position correction.",
    )
    parser.add_argument(
        "--max-abs-translation-mm",
        type=float,
        default=5.0,
        help="Abort commands that exceed this absolute local X or Z offset from the start pose.",
    )
    parser.add_argument("--speed-m-s", type=float, default=0.02)
    parser.add_argument("--acceleration-m-s2", type=float, default=0.04)
    parser.add_argument("--settle-s", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to images/tcp_image_alignment_<timestamp>.",
    )
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup-frames", type=int, default=10)
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument(
        "--tcp-offset-ur",
        nargs=6,
        type=float,
        default=DEFAULT_TCP_OFFSET_UR,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="Active UR TCP offset to set before reading and moving.",
    )
    parser.add_argument(
        "--no-set-tcp",
        action="store_true",
        help="Do not call RTDEControl.setTcp before reading and moving.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show an OpenCV preview window for each captured frame.",
    )
    parser.add_argument(
        "--open-viewer",
        action="store_true",
        help="Open an auto-refreshing browser viewer for latest_color.png.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually send moveL commands. Without this, only the current image is captured.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    values = [
        args.step_deg,
        args.max_abs_angle_deg,
        args.translation_step_mm,
        args.max_abs_translation_mm,
        args.speed_m_s,
        args.acceleration_m_s2,
        args.settle_s,
        args.color_width,
        args.color_height,
        args.fps,
        args.warmup_frames,
        args.timeout_ms,
        *args.tcp_offset_ur,
    ]
    if not np.all(np.isfinite(values)):
        raise ValueError("All numeric arguments must be finite.")
    if args.step_deg <= 0.0:
        raise ValueError("--step-deg must be positive.")
    if args.max_abs_angle_deg <= 0.0:
        raise ValueError("--max-abs-angle-deg must be positive.")
    if args.translation_step_mm <= 0.0:
        raise ValueError("--translation-step-mm must be positive.")
    if args.max_abs_translation_mm <= 0.0:
        raise ValueError("--max-abs-translation-mm must be positive.")
    if args.speed_m_s <= 0.0:
        raise ValueError("--speed-m-s must be positive.")
    if args.acceleration_m_s2 <= 0.0:
        raise ValueError("--acceleration-m-s2 must be positive.")
    if args.settle_s < 0.0:
        raise ValueError("--settle-s must be non-negative.")
    if args.color_width <= 0 or args.color_height <= 0:
        raise ValueError("Color image dimensions must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.warmup_frames < 0:
        raise ValueError("--warmup-frames must be non-negative.")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive.")


def start_color_pipeline(args: argparse.Namespace) -> rs.pipeline:
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
    pipeline.start(config)
    return pipeline


def warmup_color_stream(
    pipeline: rs.pipeline,
    warmup_frames: int,
    timeout_ms: int,
) -> rs.video_frame:
    frame = None
    for _ in range(warmup_frames + 1):
        frames = pipeline.wait_for_frames(timeout_ms)
        frame = frames.get_color_frame()
        if frame:
            continue
    if not frame:
        raise RuntimeError("Timed out before receiving a RealSense color frame.")
    return frame


def capture_color(pipeline: rs.pipeline, timeout_ms: int) -> tuple[np.ndarray, dict]:
    frames = pipeline.wait_for_frames(timeout_ms)
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("Timed out before receiving a RealSense color frame.")
    image = np.asanyarray(color_frame.get_data())
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    intrinsics_data = {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }
    return image, intrinsics_data


def fmt_pose(pose: np.ndarray) -> str:
    return "[" + ", ".join(f"{value:.6f}" for value in pose) + "]"


def write_viewer(output_dir: Path) -> Path:
    viewer_path = output_dir / "viewer.html"
    viewer_path.write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TCP Image Alignment</title>
  <style>
    body {
      margin: 0;
      background: #111;
      color: #eee;
      font: 14px/1.4 system-ui, sans-serif;
    }
    header {
      padding: 10px 14px;
      background: #1d1d1d;
      border-bottom: 1px solid #333;
    }
    img {
      display: block;
      width: 100vw;
      height: calc(100vh - 43px);
      object-fit: contain;
      background: #000;
    }
  </style>
</head>
<body>
  <header>latest_color.png auto-refreshes while the sweep is running.</header>
  <img id="latest" alt="latest RealSense color frame">
  <script>
    const img = document.getElementById("latest");
    function refresh() {
      img.src = "latest_color.png?t=" + Date.now();
    }
    refresh();
    setInterval(refresh, 300);
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return viewer_path


def maybe_open_viewer(viewer_path: Path, open_viewer: bool) -> None:
    if not open_viewer:
        return
    try:
        subprocess.Popen(["xdg-open", str(viewer_path.resolve())])
    except OSError as exc:
        print(f"failed to open viewer automatically: {exc}")


def write_capture(
    output_dir: Path,
    step_index: int,
    angle_deg: float,
    x_offset_mm: float,
    z_offset_mm: float,
    image_bgr: np.ndarray,
    intrinsics: dict,
    start_pose_ur: np.ndarray,
    target_pose_ur: np.ndarray,
    actual_pose_ur: np.ndarray,
    args: argparse.Namespace,
    label: str = "step",
) -> tuple[Path, Path]:
    safe_angle = f"{angle_deg:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    safe_x = f"{x_offset_mm:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    safe_z = f"{z_offset_mm:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    prefix = f"{label}_{step_index:03d}_{args.axis}_{safe_angle}deg_x{safe_x}mm_z{safe_z}mm"
    color_path = output_dir / f"{prefix}_color.png"
    metadata_path = output_dir / f"{prefix}_metadata.yaml"
    latest_path = output_dir / "latest_color.png"
    cv2.imwrite(str(color_path), image_bgr)
    shutil.copyfile(color_path, latest_path)

    metadata = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": label,
        "step_index": int(step_index),
        "axis": args.axis,
        "angle_deg": float(angle_deg),
        "tcp_local_x_offset_mm": float(x_offset_mm),
        "tcp_local_z_offset_mm": float(z_offset_mm),
        "step_deg": float(args.step_deg),
        "translation_step_mm": float(args.translation_step_mm),
        "max_abs_angle_deg": float(args.max_abs_angle_deg),
        "max_abs_translation_mm": float(args.max_abs_translation_mm),
        "execute": bool(args.execute),
        "set_tcp_before_motion": not args.no_set_tcp,
        "tcp_offset_ur": [float(value) for value in args.tcp_offset_ur],
        "start_tcp_pose_ur": start_pose_ur.tolist(),
        "target_tcp_pose_ur": target_pose_ur.tolist(),
        "actual_tcp_pose_ur": actual_pose_ur.tolist(),
        "color_intrinsics": intrinsics,
        "files": {
            "color": str(color_path),
            "latest_color": str(latest_path),
        },
    }
    with metadata_path.open("w") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)
    return color_path, metadata_path


def show_preview(
    window_name: str,
    image_bgr: np.ndarray,
    angle_deg: float,
    x_offset_mm: float,
    z_offset_mm: float,
) -> None:
    display = image_bgr.copy()
    cv2.putText(
        display,
        f"angle {angle_deg:+.3f} deg | TCP X {x_offset_mm:+.3f} mm | TCP Z {z_offset_mm:+.3f} mm",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(window_name, display)
    cv2.waitKey(1)


def prompt_command(
    args: argparse.Namespace,
    angle_deg: float,
    x_offset_mm: float,
    z_offset_mm: float,
) -> str:
    commands = "n/p=rot, x+/x-=TCP X, z+/z-=TCP Z, s=save, r=return start, q=quit"
    if not args.execute:
        commands = "s=save current as aligned, q=quit"
    response = input(
        f"angle {angle_deg:+.3f} deg, x {x_offset_mm:+.3f} mm, "
        f"z {z_offset_mm:+.3f} mm [{commands}] > "
    ).strip().lower()
    return response or "n"


def move_to_pose(
    rtde_control: Any,
    target_pose_ur: np.ndarray,
    speed_m_s: float,
    acceleration_m_s2: float,
    motion_name: str,
) -> None:
    print(f"{motion_name}: moveL target {fmt_pose(target_pose_ur)}")
    success = rtde_control.moveL(
        target_pose_ur.tolist(),
        float(speed_m_s),
        float(acceleration_m_s2),
    )
    if not success:
        raise RuntimeError(f"{motion_name} moveL returned false.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"images/tcp_image_alignment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = write_viewer(output_dir)
    maybe_open_viewer(viewer_path, args.open_viewer)

    rtde_control = open_rtde_control(args.robot_ip) if args.execute or not args.no_set_tcp else None
    if rtde_control is not None and not args.no_set_tcp:
        set_tcp_offset(rtde_control, args.tcp_offset_ur)
    rtde_receive = open_rtde_receive(args.robot_ip)
    start_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
    if start_pose_ur.shape != (6,):
        raise RuntimeError(f"Expected 6D TCP pose from RTDE, got {start_pose_ur.shape}.")

    pipeline = start_color_pipeline(args)
    window_name = "TCP Image Alignment Sweep"
    try:
        warmup_color_stream(pipeline, args.warmup_frames, args.timeout_ms)

        print("TCP image alignment sweep")
        print(f"  output_dir: {output_dir}")
        print(f"  viewer: {viewer_path.resolve()}")
        print(f"  axis: TCP local {args.axis.upper()}")
        print(f"  rotation step: {args.step_deg:.3f} deg")
        print(f"  translation step: {args.translation_step_mm:.3f} mm")
        print(f"  max abs angle: {args.max_abs_angle_deg:.3f} deg")
        print(f"  max abs local X/Z translation: {args.max_abs_translation_mm:.3f} mm")
        print(f"  execute: {args.execute}")
        print(f"  start_tcp_pose_ur: {fmt_pose(start_pose_ur)}")

        angle_deg = 0.0
        x_offset_mm = 0.0
        z_offset_mm = 0.0
        step_index = 0
        while True:
            target_pose_ur = pose_with_local_adjustment(
                start_pose_ur,
                args.axis,
                angle_deg,
                x_offset_mm,
                z_offset_mm,
            )
            actual_pose_ur = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
            image_bgr, intrinsics = capture_color(pipeline, args.timeout_ms)
            color_path, metadata_path = write_capture(
                output_dir,
                step_index,
                angle_deg,
                x_offset_mm,
                z_offset_mm,
                image_bgr,
                intrinsics,
                start_pose_ur,
                target_pose_ur,
                actual_pose_ur,
                args,
            )
            print(f"saved: {color_path}")
            print(f"meta:  {metadata_path}")

            if args.preview:
                show_preview(window_name, image_bgr, angle_deg, x_offset_mm, z_offset_mm)

            command = prompt_command(args, angle_deg, x_offset_mm, z_offset_mm)
            if command in {"q", "quit", "exit"}:
                break
            if command in {"s", "save", "aligned"}:
                aligned_image, aligned_intrinsics = capture_color(pipeline, args.timeout_ms)
                aligned_actual_pose = np.asarray(rtde_receive.getActualTCPPose(), dtype=float)
                aligned_color_path, aligned_metadata_path = write_capture(
                    output_dir,
                    step_index,
                    angle_deg,
                    x_offset_mm,
                    z_offset_mm,
                    aligned_image,
                    aligned_intrinsics,
                    start_pose_ur,
                    target_pose_ur,
                    aligned_actual_pose,
                    args,
                    label="aligned",
                )
                print(f"aligned saved: {aligned_color_path}")
                print(f"aligned meta:  {aligned_metadata_path}")
                break
            next_angle_deg = angle_deg
            next_x_offset_mm = x_offset_mm
            next_z_offset_mm = z_offset_mm
            if command in {"r", "return", "home", "0"}:
                next_angle_deg = 0.0
                next_x_offset_mm = 0.0
                next_z_offset_mm = 0.0
            elif command in {"p", "-", "prev", "back"}:
                next_angle_deg = angle_deg - args.step_deg
            elif command in {"x+", "+x", "xp", "xplus"}:
                next_x_offset_mm = x_offset_mm + args.translation_step_mm
            elif command in {"x-", "-x", "xm", "xminus"}:
                next_x_offset_mm = x_offset_mm - args.translation_step_mm
            elif command in {"z+", "+z", "zp", "zplus"}:
                next_z_offset_mm = z_offset_mm + args.translation_step_mm
            elif command in {"z-", "-z", "zm", "zminus"}:
                next_z_offset_mm = z_offset_mm - args.translation_step_mm
            else:
                next_angle_deg = angle_deg + args.step_deg

            if abs(next_angle_deg) > args.max_abs_angle_deg + 1e-9:
                print(
                    f"refusing {next_angle_deg:+.3f} deg: exceeds "
                    f"--max-abs-angle-deg {args.max_abs_angle_deg:.3f}"
                )
                continue
            if (
                abs(next_x_offset_mm) > args.max_abs_translation_mm + 1e-9
                or abs(next_z_offset_mm) > args.max_abs_translation_mm + 1e-9
            ):
                print(
                    f"refusing x {next_x_offset_mm:+.3f} mm, "
                    f"z {next_z_offset_mm:+.3f} mm: exceeds "
                    f"--max-abs-translation-mm {args.max_abs_translation_mm:.3f}"
                )
                continue
            if not args.execute:
                print("DRY RUN ONLY: add --execute to move the robot.")
                continue

            next_pose_ur = pose_with_local_adjustment(
                start_pose_ur,
                args.axis,
                next_angle_deg,
                next_x_offset_mm,
                next_z_offset_mm,
            )
            move_to_pose(
                rtde_control,
                next_pose_ur,
                args.speed_m_s,
                args.acceleration_m_s2,
                (
                    f"tcp local adjustment rot {next_angle_deg:+.3f} deg, "
                    f"x {next_x_offset_mm:+.3f} mm, z {next_z_offset_mm:+.3f} mm"
                ),
            )
            time.sleep(args.settle_s)
            angle_deg = next_angle_deg
            x_offset_mm = next_x_offset_mm
            z_offset_mm = next_z_offset_mm
            step_index += 1
    finally:
        pipeline.stop()
        if args.preview:
            cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
