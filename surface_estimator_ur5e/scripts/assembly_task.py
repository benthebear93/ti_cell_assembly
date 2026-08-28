#!/usr/bin/env python3
"""Top-level assembly task entrypoint.

This file is the task-facing runner. Stage-specific behavior still lives in the
existing scripts while this entrypoint gives the overall assembly workflow one
stable command surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys


@dataclass(frozen=True)
class TaskStage:
    """One runnable assembly task stage."""

    name: str
    module_name: str
    summary: str
    aliases: tuple[str, ...] = ()


STAGES: tuple[TaskStage, ...] = (
    TaskStage(
        name="marker-target",
        aliases=("marker-move", "pick-from-marker"),
        module_name="marker_based_motion",
        summary=(
            "Compute or execute a marker-relative TCP target. This includes the existing "
            "optional gripper close and holder four-pin alignment path."
        ),
    ),
    TaskStage(
        name="holder-align",
        aliases=("align-holder", "four-pin-align"),
        module_name="align_tray_to_pins_rotation",
        summary=(
            "Keep the current TCP position and rotate a held tray to the holder four-pin frame."
        ),
    ),
    TaskStage(
        name="visualize",
        aliases=("view", "marker-view"),
        module_name="visualize_marker_frame",
        summary="Visualize the marker frame, holder, tray, target, and UR5e state in viser.",
    ),
    TaskStage(
        name="preview-motion",
        aliases=("animate", "offline-motion"),
        module_name="preview_assembly_motion",
        summary=("Animate the real marker-based assembly command path offline with PyRoki IK."),
    ),
    TaskStage(
        name="train-all-motion-dmps",
        aliases=("train-motion-dmps", "all-motion-dmps"),
        module_name="train_all_motion_dmps",
        summary=("Train and persist DMP weights for every moveL in the complete assembly path."),
    ),
    TaskStage(
        name="generate-tray-dmp",
        aliases=("tray-dmp", "dmp-dataset"),
        module_name="generate_tray_dmp_dataset",
        summary=("Generate PyRoki-validated tray pick-and-insert DMP demonstrations offline."),
    ),
    TaskStage(
        name="preview-tray-dmp",
        aliases=("view-tray-dmp", "dmp-viewer"),
        module_name="preview_tray_dmp_dataset",
        summary="Animate and compare generated tray DMP demonstrations in Viser.",
    ),
    TaskStage(
        name="compare-tray-dmp",
        aliases=("compare-dmp", "dual-dmp-viewer"),
        module_name="preview_tray_dmp_comparison",
        summary=("Animate one robot doing nominal insert, regrasp/lift, then offset insert."),
    ),
    TaskStage(
        name="compare-all-motion-dmp",
        aliases=("all-motion-dmp-viewer", "full-dmp-comparison"),
        module_name="preview_all_motion_dmp_comparison",
        summary=("Run all 27 DMP moves for nominal, return/reset, then the full offset task."),
    ),
    TaskStage(
        name="generate-tray-dmp-recovery",
        aliases=("tray-dmp-recovery", "recovery-dataset"),
        module_name="generate_tray_dmp_recovery_dataset",
        summary=("Generate PyRoki-realized recovery demonstrations from perturbed DMP states."),
    ),
    TaskStage(
        name="read-marker",
        aliases=("capture-marker",),
        module_name="read_aruco_marker_to_base",
        summary="Capture an ArUco marker pose in the robot base frame.",
    ),
    TaskStage(
        name="capture-realsense",
        aliases=("capture-camera", "camera-frame"),
        module_name="capture_realsense_frame",
        summary="Capture one RealSense color frame, aligned depth frame, and metadata.",
    ),
    TaskStage(
        name="image-align-sweep",
        aliases=("tcp-image-sweep", "camera-align"),
        module_name="tcp_image_alignment_sweep",
        summary="Rotate the current TCP about a local axis in small image-checked steps.",
    ),
    TaskStage(
        name="run-sequence",
        aliases=("sequence",),
        module_name="run_motion_sequence",
        summary="Run an existing YAML motion sequence.",
    ),
)


def main(argv: list[str] | None = None) -> None:
    """Run an assembly task stage."""

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "list", "stages"}:
        print_help()
        return

    stage_name = args[0]
    stage = resolve_stage(stage_name)
    if stage is None:
        known = ", ".join(stage.name for stage in STAGES)
        raise SystemExit(f"Unknown assembly task stage '{stage_name}'. Known stages: {known}")

    run_stage(stage, args[1:])


def resolve_stage(name: str) -> TaskStage | None:
    """Resolve a canonical stage name or alias."""

    for stage in STAGES:
        if name == stage.name or name in stage.aliases:
            return stage
    return None


def run_stage(stage: TaskStage, stage_args: list[str]) -> None:
    """Forward execution to the script that owns the selected stage."""

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    module = importlib.import_module(stage.module_name)
    old_argv = sys.argv
    sys.argv = [f"{stage.module_name}.py", *stage_args]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def print_help() -> None:
    """Print task-level help without duplicating each stage's full CLI."""

    print("usage: assembly_task.py <stage> [stage options]")
    print()
    print("Top-level entrypoint for assembly task stages.")
    print()
    print("Stages:")
    for stage in STAGES:
        alias_text = f" (aliases: {', '.join(stage.aliases)})" if stage.aliases else ""
        print(f"  {stage.name}{alias_text}")
        print(f"    {stage.summary}")
    print()
    print("Examples:")
    print("  uv run python scripts/assembly_task.py visualize --show-holder --show-tray")
    print("  uv run python scripts/assembly_task.py preview-motion")
    print("  uv run python scripts/assembly_task.py train-all-motion-dmps")
    print("  uv run python scripts/assembly_task.py generate-tray-dmp --episodes 20")
    print("  uv run python scripts/assembly_task.py generate-tray-dmp-recovery")
    print("  uv run python scripts/assembly_task.py preview-tray-dmp --dataset data/dmp/demo.npz")
    print("  uv run python scripts/assembly_task.py compare-tray-dmp")
    print("  uv run python scripts/assembly_task.py compare-all-motion-dmp")
    print("  uv run python scripts/assembly_task.py marker-target --save-target")
    print("  uv run python scripts/assembly_task.py holder-align --save-plan")
    print()
    print("Use '<stage> --help' to see the original stage-specific options.")


if __name__ == "__main__":
    main()
