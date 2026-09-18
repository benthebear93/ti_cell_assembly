#!/usr/bin/env python3
"""Run assembly stages through one robot session, with shared planning and limits."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import assembly_config as settings
import assembly_plan as plan
import numpy as np
import yaml
from assembly_config import parse_args, validate_args

from surface_estimator_ur5e.calibration import load_marker_transform
from surface_estimator_ur5e.motion_sequence import GripperCommand, RobotiqHandEGripper
from surface_estimator_ur5e.robot_io import (
    connect_rtde_control,
    read_current_tcp_pose,
    set_tcp_offset,
)
from surface_estimator_ur5e.transforms import (
    pose_with_local_xz_adjustment as pose_with_tcp_local_xz_rotation_adjustment,
)
from surface_estimator_ur5e.transforms import rotation_delta_deg
from surface_estimator_ur5e.workcell_geometry import floor_constrained_marker_transform


def normalize_alignment_gray(gray: np.ndarray) -> np.ndarray:
    image = np.asarray(gray, dtype=np.float32)
    image = image - float(np.mean(image))
    norm = float(np.linalg.norm(image))
    if norm < 1e-9:
        return np.zeros_like(image, dtype=np.float32)
    return image / norm


def crop_alignment_roi(
    image: np.ndarray, roi: list[int] | tuple[int, int, int, int] | None
) -> np.ndarray:
    if roi is None:
        return image
    x, y, width, height = [int(value) for value in roi]
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError(
            f"pin_image_roi {roi} exceeds image size {image.shape[1]}x{image.shape[0]}."
        )
    return image[y : y + height, x : x + width]


def load_reference_alignment_image(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    if args.pin_image_reference is None:
        raise RuntimeError("pin_image_reference is required for automatic image alignment.")
    reference_bgr = cv2.imread(str(args.pin_image_reference), cv2.IMREAD_COLOR)
    if reference_bgr is None:
        raise RuntimeError(f"Failed to read reference image: {args.pin_image_reference}")
    if reference_bgr.shape[:2] != (args.pin_image_height, args.pin_image_width):
        raise RuntimeError(
            f"Reference image is {reference_bgr.shape[1]}x{reference_bgr.shape[0]}, but capture "
            f"is configured for {args.pin_image_width}x{args.pin_image_height}."
        )
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    reference_roi = crop_alignment_roi(reference_gray, args.pin_image_roi)
    reference_roi = cv2.equalizeHist(reference_roi)
    return reference_bgr, normalize_alignment_gray(reference_roi)


def score_alignment_image(
    image_bgr: np.ndarray,
    reference_gray_normalized: np.ndarray,
    args: argparse.Namespace,
) -> float:
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    roi = crop_alignment_roi(gray, args.pin_image_roi)
    roi = cv2.equalizeHist(roi)
    roi_normalized = normalize_alignment_gray(roi)
    if roi_normalized.shape != reference_gray_normalized.shape:
        raise RuntimeError(
            f"Alignment ROI shape mismatch: {roi_normalized.shape} vs "
            f"{reference_gray_normalized.shape}."
        )
    return float(np.sum(reference_gray_normalized * roi_normalized))


def start_pin_image_pipeline(args: argparse.Namespace) -> Any:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    if args.pin_image_serial:
        config.enable_device(args.pin_image_serial)
    config.enable_stream(
        rs.stream.color,
        args.pin_image_width,
        args.pin_image_height,
        rs.format.bgr8,
        args.pin_image_fps,
    )
    pipeline.start(config)
    return pipeline


def capture_pin_image(pipeline: Any, args: argparse.Namespace) -> np.ndarray:
    frames = pipeline.wait_for_frames(args.pin_image_timeout_ms)
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("Timed out before receiving a RealSense color frame.")
    return np.asanyarray(color_frame.get_data())


def auto_alignment_output_dir(args: argparse.Namespace) -> Path:
    if args.pin_image_output_dir is not None:
        output_dir = args.pin_image_output_dir
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"images/auto_pin_image_alignment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_auto_alignment_capture(
    output_dir: Path,
    image_bgr: np.ndarray,
    iteration: int,
    candidate_index: int,
    state: tuple[float, float, float],
    score: float,
    pose_ur: np.ndarray,
    args: argparse.Namespace,
    label: str = "candidate",
) -> tuple[Path, Path]:
    import cv2

    angle_deg, x_offset_mm, z_offset_mm = state
    safe_angle = f"{angle_deg:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    safe_x = f"{x_offset_mm:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    safe_z = f"{z_offset_mm:+07.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
    prefix = (
        f"{label}_i{iteration:02d}_c{candidate_index:02d}_{safe_angle}deg_x{safe_x}mm_z{safe_z}mm"
    )
    color_path = output_dir / f"{prefix}_color.png"
    metadata_path = output_dir / f"{prefix}_metadata.yaml"
    latest_path = output_dir / "latest_color.png"
    cv2.imwrite(str(color_path), image_bgr)
    cv2.imwrite(str(latest_path), image_bgr)
    metadata = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label": label,
        "iteration": int(iteration),
        "candidate_index": int(candidate_index),
        "score": float(score),
        "axis": args.axis if hasattr(args, "axis") else "y",
        "angle_deg": float(angle_deg),
        "tcp_local_x_offset_mm": float(x_offset_mm),
        "tcp_local_z_offset_mm": float(z_offset_mm),
        "target_tcp_pose_ur": plan.round_list(pose_ur),
        "reference_image": str(args.pin_image_reference),
        "roi": list(args.pin_image_roi) if args.pin_image_roi is not None else None,
        "files": {
            "color": str(color_path),
            "latest_color": str(latest_path),
        },
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(metadata, file, sort_keys=False)
    return color_path, metadata_path


def unique_alignment_candidates(
    center: tuple[float, float, float],
    step_deg: float,
    step_mm: float,
    max_deg: float,
    max_mm: float,
) -> list[tuple[float, float, float]]:
    candidates = [center]
    for axis, step in enumerate((step_deg, step_mm, step_mm)):
        if step > 0:
            for sign in (1, -1):
                candidate = list(center)
                candidate[axis] += sign * step
                candidates.append(tuple(candidate))
    unique, seen = [], set()
    for candidate in candidates:
        key = tuple(round(float(v), 6) for v in candidate)
        within_bounds = all(
            abs(v) <= limit + 1e-9
            for v, limit in zip(candidate, (max_deg, max_mm, max_mm), strict=True)
        )
        if key not in seen and within_bounds:
            seen.add(key)
            unique.append(candidate)
    return unique


@dataclass
class Robot:
    """The connected interfaces and settings for one assembly run."""

    receive: Any
    control: Any
    args: argparse.Namespace

    @property
    def pose(self) -> np.ndarray:
        return np.asarray(self.receive.getActualTCPPose(), dtype=float)

    @property
    def joints(self) -> np.ndarray:
        return np.asarray(self.receive.getActualQ(), dtype=float)

    def preflight(self, targets: plan.Targets, q_near: np.ndarray | None = None) -> None:
        q_near = self.joints if q_near is None else q_near
        for name, pose in targets:
            q_near = plan.require_ik_solution(self.control, q_near, pose, name)

    def move(
        self, target: np.ndarray, name: str, *, slow: bool = False, segmented: bool = False
    ) -> None:
        current = self.pose
        distance_mm = float(np.linalg.norm(target[:3] - current[:3]) * 1000)
        rotation_deg = rotation_delta_deg(current, target)
        if not segmented and distance_mm < 0.01 and rotation_deg < 0.01:
            return
        speed = self.args.pin_insertion_speed_m_s if slow else self.args.speed_m_s
        acceleration = (
            self.args.pin_insertion_acceleration_m_s2 if slow else self.args.acceleration_m_s2
        )
        print(f"{name}: {distance_mm:.2f} mm, {rotation_deg:.2f} deg, {speed:.4f} m/s")
        if not self.control.moveL(target.tolist(), speed, acceleration, False):
            raise RuntimeError(f"{name} moveL returned false.")
        plan.check_joint_margins(self.joints, name)

    def moves(self, targets: plan.Targets, *, slow: bool = False) -> None:
        self.preflight(targets)
        for name, pose in targets:
            self.move(pose, name, slow=slow)

    def grip(self, name: str, percent: float, dwell_s: float = 0.0) -> None:
        command = GripperCommand(
            name=name,
            position_percent=percent,
            speed=self.args.gripper_speed,
            force=self.args.gripper_force,
            wait=True,
            timeout_s=5.0,
            post_dwell_s=dwell_s,
        )
        gripper = RobotiqHandEGripper(self.args.robot_ip, port=self.args.gripper_port)
        try:
            gripper.connect()
            print(f"{name}: {percent:.1f}% closed, raw position={gripper.get_position()}")
            gripper.move_to_percent(command)
        finally:
            gripper.close()

    def image_correction(self, *, post_close: bool = False) -> None:
        title = (
            "Post-two-pin-close image alignment" if post_close else "Post-approach image alignment"
        )
        if self.args.auto_pin_image_align:
            execute_auto_pin_image_alignment(self.receive, self.control, self.args, title)
            return
        metadata = (
            settings.effective_post_two_pin_close_image_alignment_metadata(self.args)
            if post_close
            else settings.effective_pin_image_alignment_metadata(self.args)
        )
        if metadata is not None:
            target = plan.print_pin_image_alignment_correction_plan(
                self.pose, self.args, title, metadata
            )
            self.moves([(title, target)])

    def align(self, base_t_pin: np.ndarray, floor_z: float) -> None:
        result, lift, rotation, _ = plan.select_safe_alignment_plan(
            self.control,
            self.pose,
            self.joints,
            base_t_pin,
            floor_z,
            self.args,
        )
        plan.print_four_pin_alignment_plan(result, base_t_pin, self.args)
        if (
            result["angle_delta_deg"] > self.args.max_rotation_deg
            and not self.args.allow_large_rotation
        ):
            raise RuntimeError("Rotation exceeds max_rotation_deg; re-check the frame and limit.")
        for name, target, waypoints in (
            ("post-grasp lift", result["lifted_current_tcp_pose_ur"], lift),
            ("four-pin rotation", result["lifted_target_tcp_pose_ur"], rotation),
        ):
            for pose in waypoints if self.args.segmented_alignment else [target]:
                self.move(pose, name, segmented=self.args.segmented_alignment)

    def pin_sequence(self, base_t_pin: np.ndarray) -> None:
        targets = plan.plan_pin_sequence(self.pose, base_t_pin, self.args)
        self.preflight(targets)
        if not self.args.execute:
            return
        self.moves(targets[:2])
        self.image_correction()
        if self.args.insert_after_pin_approach:
            insertion = plan.print_pin_insertion_plan(self.pose, base_t_pin, self.args)
            self.moves([("final pin insertion descend", insertion)], slow=True)
            self.post_insert(base_t_pin)

    def post_insert(self, base_t_pin: np.ndarray) -> None:
        if not self.args.post_insert_release_retreat:
            return
        path = plan.plan_post_insert(self.pose, base_t_pin, self.args)
        self.preflight(path.targets())
        self.grip("post_insert_open", 0, 0.5)
        self.moves(path.retreat)
        if not self.args.post_insert_move_to_two_pin:
            return
        self.grip("post_insert_two_pin_close", self.args.gripper_close_percent)
        time.sleep(settings.POST_TWO_PIN_CLOSE_DWELL_S)
        self.moves(path.transfer)
        if path.align_image:
            self.image_correction(post_close=True)
            if self.args.post_two_pin_aligned_insert_release:
                self.aligned_release()

    def aligned_release(self) -> None:
        targets = plan.print_post_two_pin_aligned_insert_release_plan(self.pose, self.args)
        if len(targets) != 5:
            raise RuntimeError("Expected insertion, retreat, Z shift, X/Y shift and X rotation.")
        self.preflight(targets)
        self.moves(targets[:1], slow=True)
        self.grip("post_two_pin_aligned_open", 0, 2.0)
        self.moves(targets[1:])
        if self.args.stop_after_post_two_pin_release_rotation:
            return
        self.release_close()
        for planner in (
            plan.print_post_two_pin_after_close_plan,
            plan.post_two_pin_after_close_tail_target_poses,
        ):
            self.moves(planner(self.pose, self.args), slow=True)

    def release_close(self, *, resume: bool = False) -> None:
        self.moves(
            plan.print_post_two_pin_release_after_rotation_plan(self.pose, self.args), slow=True
        )
        name = ("resume_" if resume else "") + "post_two_pin_release_final_half_close"
        self.grip(name, self.args.post_two_pin_release_final_close_percent)
        target = plan.tcp_shift(
            self.pose,
            y=self.args.post_two_pin_release_after_rotation_y_mm,
            z=settings.DEFAULT_POST_TWO_PIN_RELEASE_AFTER_ROTATION_Z_MM,
        )
        self.moves([("post-two-pin after-close TCP Y/Z shift", target)], slow=True)

    def resume(self) -> None:
        planners = {
            "release": plan.print_post_two_pin_release_resume_plan,
            "after-close": plan.print_post_two_pin_after_close_plan,
            "tail": plan.post_two_pin_after_close_tail_target_poses,
            "tail-next": plan.post_two_pin_after_close_tail_next_target_poses,
        }
        targets = planners[self.args.resume](self.pose, self.args)
        self.preflight(targets)
        if self.args.execute:
            self.moves(targets, slow=True)
            if (
                self.args.resume == "release"
                and not self.args.stop_after_post_two_pin_release_rotation
            ):
                self.release_close(resume=True)

    def marker_task(
        self, marker_id: int | None, marker: np.ndarray, pin: np.ndarray | None
    ) -> None:
        args = self.args
        if args.execute:
            self.grip("pre_motion_open", 0, 0.5)
        current_pose = self.pose
        if args.start_from_initial_pose:
            q = np.deg2rad(args.initial_q_deg).tolist()
            if args.execute:
                if not self.control.moveJ(
                    q, args.initial_speed_rad_s, args.initial_acceleration_rad_s2, False
                ):
                    raise RuntimeError("Initial pose moveJ returned false.")
                if args.initial_dwell_s > 0:
                    time.sleep(args.initial_dwell_s)
                current_pose = self.pose
            else:
                current_pose = np.asarray(
                    self.control.getForwardKinematics(q, list(args.tcp_offset_ur)), dtype=float
                )
        offset = plan.marker_target_offset(args)
        target = plan.make_target_pose_ur(marker, offset, current_pose, args.orientation)
        distance_mm = float(np.linalg.norm(target[:3] - current_pose[:3]) * 1000)
        print(f"Marker {marker_id}: task={args.task}, travel={distance_mm:.1f} mm")
        plan.report_targets([("marker grasp", target)])
        if pin is not None:
            self.marker_preflight(target, marker, pin)
        if args.save_target is not None:
            plan.save_target_yaml(
                args.save_target,
                marker_id,
                args.marker_pose,
                marker,
                offset,
                current_pose,
                target,
                args,
            )
        if not args.execute:
            return
        if distance_mm > args.max_distance_mm and not args.allow_large_move:
            raise RuntimeError(f"Target exceeds max_distance_mm {args.max_distance_mm:.1f}.")
        if not self.control.moveL(target.tolist(), args.speed_m_s, args.acceleration_m_s2, False):
            raise RuntimeError("Marker target moveL returned false.")
        if not args.skip_gripper_close:
            if args.gripper_delay_s > 0:
                time.sleep(args.gripper_delay_s)
            self.grip("marker_target_close", args.gripper_close_percent)
            time.sleep(settings.POST_GRASP_DWELL_S)
        if pin is not None:
            self.align(pin, float(marker[2, 3]))
            if args.approach_pin_after_rotation:
                self.pin_sequence(pin)

    def marker_preflight(self, target: np.ndarray, marker: np.ndarray, pin: np.ndarray) -> None:
        if self.args.execute:
            result = plan.four_pin_alignment_plan(target, pin, float(marker[2, 3]), self.args)
        else:
            q = (
                np.deg2rad(self.args.initial_q_deg)
                if self.args.start_from_initial_pose
                else self.joints
            )
            grasp_q = plan.require_ik_solution(self.control, q, target, "marker grasp")
            result, _, _, report = plan.select_safe_alignment_plan(
                self.control,
                target,
                grasp_q,
                pin,
                float(marker[2, 3]),
                self.args,
            )
            if self.args.approach_pin_after_rotation:
                targets = plan.plan_pin_sequence(
                    result["lifted_target_tcp_pose_ur"], pin, self.args
                )
                self.preflight(targets, report["q_path"][-1] if report["q_path"] else grasp_q)
        plan.print_four_pin_alignment_plan(result, pin, self.args)


@dataclass
class AlignmentSample:
    state: tuple[float, float, float]
    pose: np.ndarray
    score: float = -float("inf")
    image: np.ndarray | None = None


def execute_auto_pin_image_alignment(
    receive: Any, control: Any, args: argparse.Namespace, title: str = "Automatic image alignment"
) -> np.ndarray:
    robot = Robot(receive, control, args)
    _, reference = load_reference_alignment_image(args)
    output = auto_alignment_output_dir(args)
    start_pose, start_q = robot.pose, robot.joints
    best = AlignmentSample((0, 0, 0), start_pose.copy())
    step_mm, step_deg = args.auto_pin_image_search_step_mm, args.auto_pin_image_search_step_deg
    print(f"{title}: reference={args.pin_image_reference}, output={output}")
    pipeline = start_pin_image_pipeline(args)
    try:
        for _ in range(args.pin_image_warmup_frames):
            pipeline.wait_for_frames(args.pin_image_timeout_ms)
        for iteration in range(1, args.auto_pin_image_search_iterations + 1):
            candidates = unique_alignment_candidates(
                best.state,
                step_deg,
                step_mm,
                args.auto_pin_image_search_max_deg,
                args.auto_pin_image_search_max_mm,
            )
            for index, state in enumerate(candidates, start=1):
                target = pose_with_tcp_local_xz_rotation_adjustment(start_pose, "y", *state)
                plan.require_ik_solution(
                    control, start_q, target, "automatic image alignment candidate"
                )
                robot.move(target, f"image candidate i{iteration} c{index}")
                if args.auto_pin_image_settle_s > 0:
                    time.sleep(args.auto_pin_image_settle_s)
                image = capture_pin_image(pipeline, args)
                score = score_alignment_image(image, reference, args)
                save_auto_alignment_capture(
                    output, image, iteration, index, state, score, target, args
                )
                if score > best.score:
                    best = AlignmentSample(state, target, score, image)
            print(f"  iteration {iteration}: best={best.state}, score={best.score:.6f}")
            step_mm *= 0.5
            step_deg *= 0.5
        robot.move(best.pose, "automatic image alignment best pose")
        if best.image is None:
            best.image = capture_pin_image(pipeline, args)
        save_auto_alignment_capture(
            output,
            best.image,
            args.auto_pin_image_search_iterations,
            0,
            best.state,
            best.score,
            best.pose,
            args,
            label="best",
        )
        return robot.pose
    finally:
        pipeline.stop()


def main() -> None:
    args = parse_args()
    validate_args(args)
    print(
        f"Motion speeds: joint={args.initial_speed_rad_s} rad/s, normal={args.speed_m_s} m/s, insertion={args.pin_insertion_speed_m_s} m/s"
    )
    marker_id = marker = pin = None
    if args.resume in (None, "pins"):
        marker_id, _marker_length, raw, _ = load_marker_transform(args.marker_pose)
        marker = (
            floor_constrained_marker_transform(raw) if args.marker_frame_mode == "floor" else raw
        )
        if args.align_to_four_pin_frame or args.resume == "pins":
            pin = plan.compute_base_t_four_pin_frame(args, marker)
    receive = control = None
    try:
        if (
            args.execute
            or not args.no_set_tcp
            or args.start_from_initial_pose
            or args.align_to_four_pin_frame
            or args.resume
        ):
            control = connect_rtde_control(args.robot_ip)
        if not args.no_set_tcp:
            set_tcp_offset(control, args.tcp_offset_ur)
        receive, _ = read_current_tcp_pose(args.robot_ip)
        robot = Robot(receive, control, args)
        if args.resume == "pins":
            robot.pin_sequence(pin)
        elif args.resume:
            robot.resume()
        else:
            robot.marker_task(marker_id, marker, pin)
        if not args.execute:
            print("DRY RUN ONLY: add --execute to send motion commands.")
    finally:
        try:
            if control is not None:
                control.stopScript()
        finally:
            try:
                if receive is not None:
                    receive.disconnect()
            finally:
                if control is not None:
                    control.disconnect()


if __name__ == "__main__":
    main()
