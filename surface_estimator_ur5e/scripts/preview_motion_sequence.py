"""Preview the saved UR5e joint sequence in viser."""

from __future__ import annotations

import argparse
from pathlib import Path

from surface_estimator_ur5e.motion_sequence import (
    DEFAULT_TCP_OFFSET_UR,
    load_joint_sequence,
    preview_joint_sequence,
)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=repo_root / "data" / "ur5e_motion_sequence.yaml")
    parser.add_argument("--steps-per-segment", type=int, default=40)
    parser.add_argument("--playback-hz", type=float, default=15.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--final-tool-negative-z-m", type=float, default=0.0)
    parser.add_argument(
        "--tcp-offset-ur",
        nargs=6,
        type=float,
        default=DEFAULT_TCP_OFFSET_UR.tolist(),
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
    )
    args = parser.parse_args()

    preview_joint_sequence(
        load_joint_sequence(args.input),
        steps_per_segment=args.steps_per_segment,
        playback_hz=args.playback_hz,
        port=args.port,
        final_tool_negative_z_m=args.final_tool_negative_z_m,
        tcp_offset_ur=args.tcp_offset_ur,
    )


if __name__ == "__main__":
    main()
