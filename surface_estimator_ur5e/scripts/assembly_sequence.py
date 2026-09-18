"""Command-order plans shared by the static and animated assembly previews.

These plans use saved camera corrections. Execution still replans from measured
TCP poses at the gripper and live-camera boundaries in marker_based_motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Literal

import assembly_config as config
import assembly_plan as motion
import numpy as np


@dataclass(frozen=True)
class MotionEvent:
    kind: Literal["move", "gripper", "dwell"]
    name: str
    target_pose_ur: np.ndarray | None = None
    speed_m_s: float | None = None
    gripper_percent: float | None = None
    tray_action: Literal["attach", "detach"] | None = None
    dwell_s: float = 0.0


@dataclass(frozen=True)
class AssemblySequence:
    events: list[MotionEvent]
    base_t_pin: np.ndarray
    alignment: dict[str, Any]
    ik_report: dict[str, Any]
    grasp_q_rad: np.ndarray


def move_events(targets: motion.Targets, speed: float) -> list[MotionEvent]:
    return [MotionEvent("move", name, pose, speed) for name, pose in targets]


def pin_motion_events(
    start_pose: np.ndarray, base_t_pin: np.ndarray, args: argparse.Namespace
) -> list[MotionEvent]:
    """Plan from the completed four-pin rotation through the optional final tail."""
    if args.auto_pin_image_align:
        raise ValueError(
            "Previews require saved image corrections; live camera search is unavailable."
        )
    if not args.approach_pin_after_rotation:
        return []
    centering, approach = motion.print_pin_approach_plan(start_pose, base_t_pin, args)
    events = move_events(
        [
            ("post-rotation pin centering", centering),
            ("post-rotation pin-axis approach", approach),
        ],
        args.speed_m_s,
    )
    correction = motion.print_pin_image_alignment_correction_plan(approach, args)
    if correction is not None:
        events.append(
            MotionEvent(
                "move", "post-approach image alignment correction", correction, args.speed_m_s
            )
        )
    if not args.insert_after_pin_approach:
        return events
    insertion = motion.print_pin_insertion_plan(events[-1].target_pose_ur, base_t_pin, args)
    events.append(
        MotionEvent(
            "move",
            "final pin insertion descend",
            insertion,
            args.pin_insertion_speed_m_s,
        )
    )
    post_insert = motion.plan_post_insert(insertion, base_t_pin, args)
    if not post_insert.retreat:
        return events
    events.append(
        MotionEvent(
            "gripper",
            "post-insert open",
            gripper_percent=0.0,
            tray_action="detach",
        )
    )
    events.extend(move_events(post_insert.retreat, args.speed_m_s))
    if not args.post_insert_move_to_two_pin:
        return events
    events.extend(
        [
            MotionEvent(
                "gripper", "post-insert two-pin close", gripper_percent=args.gripper_close_percent
            ),
            MotionEvent(
                "dwell", "post-two-pin close dwell", dwell_s=config.POST_TWO_PIN_CLOSE_DWELL_S
            ),
        ]
    )
    events.extend(move_events(post_insert.transfer, args.speed_m_s))
    if post_insert.correction is not None:
        events.append(
            MotionEvent(
                "move",
                "post-two-pin-close image alignment correction",
                post_insert.correction,
                args.speed_m_s,
            )
        )
    if not post_insert.aligned_release:
        return events
    events.extend(move_events(post_insert.aligned_release[:1], args.pin_insertion_speed_m_s))
    events.append(MotionEvent("gripper", "post-two-pin aligned open", gripper_percent=0.0))
    events.extend(move_events(post_insert.aligned_release[1:], args.speed_m_s))
    if not post_insert.after_rotation:
        return events
    events.extend(move_events(post_insert.after_rotation, args.pin_insertion_speed_m_s))
    events.append(
        MotionEvent(
            "gripper",
            "post-two-pin release final half-close",
            gripper_percent=args.post_two_pin_release_final_close_percent,
        )
    )
    after_close = motion.tcp_shift(
        post_insert.after_rotation[-1][1],
        y=args.post_two_pin_release_after_rotation_y_mm,
        z=config.DEFAULT_POST_TWO_PIN_RELEASE_AFTER_ROTATION_Z_MM,
    )
    final_targets = [("post-two-pin after-close TCP Y/Z shift", after_close)]
    final_targets.extend(motion.post_two_pin_after_close_target_poses(after_close, args))
    final_targets.extend(
        motion.post_two_pin_after_close_tail_target_poses(final_targets[-1][1], args)
    )
    events.extend(move_events(final_targets, args.pin_insertion_speed_m_s))
    return events


def build_assembly_sequence(
    args: argparse.Namespace,
    ik: Any,
    base_t_marker: np.ndarray,
    start_pose: np.ndarray,
    start_q_rad: np.ndarray,
) -> AssemblySequence:
    """Plan a grasp, a checked lift/rotation, and the subsequent pin operations."""
    base_t_pin = motion.compute_base_t_four_pin_frame(args, base_t_marker)
    grasp = motion.make_target_pose_ur(
        base_t_marker,
        motion.marker_target_offset(args),
        start_pose,
        args.orientation,
    )
    grasp_q = motion.require_ik_solution(ik, start_q_rad, grasp, "marker grasp")
    grasp_q = motion.unwrap_joints_near(grasp_q, start_q_rad)
    alignment, _lift, _rotation, ik_report = motion.select_safe_alignment_plan(
        ik,
        grasp,
        grasp_q,
        base_t_pin,
        float(base_t_marker[2, 3]),
        args,
    )
    events = [
        MotionEvent("gripper", "pre-motion open", gripper_percent=0.0),
        MotionEvent("move", "marker grasp moveL", grasp, args.speed_m_s),
        MotionEvent(
            "gripper",
            "marker target close",
            gripper_percent=args.gripper_close_percent,
            tray_action="attach",
        ),
        MotionEvent("dwell", "post-grasp dwell", dwell_s=config.POST_GRASP_DWELL_S),
        MotionEvent(
            "move", "post-grasp lift", alignment["lifted_current_tcp_pose_ur"], args.speed_m_s
        ),
        MotionEvent(
            "move", "four-pin rotation", alignment["lifted_target_tcp_pose_ur"], args.speed_m_s
        ),
    ]
    events.extend(pin_motion_events(alignment["lifted_target_tcp_pose_ur"], base_t_pin, args))
    return AssemblySequence(events, base_t_pin, alignment, ik_report, grasp_q)
