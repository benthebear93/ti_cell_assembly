"""Standalone alignment commands captured before extracting shared helpers."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import align_tray_to_pins_rotation as alignment
import numpy as np
import pytest
from motion_fakes import FakeRobot, fake_environment

PROJECT = Path(__file__).resolve().parents[1]
TRACES = json.loads((Path(__file__).parent / "fixtures/alignment_motion_before.json").read_text())


@pytest.mark.parametrize("case", TRACES.values(), ids=TRACES.keys())
def test_alignment_commands_match_before_extraction(case):
    with patch.object(sys, "argv", ["align_tray_to_pins_rotation"]):
        args = alignment.parse_args()
    vars(args).update(case["settings"])
    args.marker_pose = PROJECT / args.marker_pose
    args.auto_pin_image_align = False  # Shared fake environment's camera switch.
    robot = FakeRobot()
    with fake_environment(alignment, args, robot):
        if case["error"]:
            with pytest.raises(RuntimeError) as error:
                alignment.main()
            assert str(error.value) == case["error"]
        else:
            alignment.main()
    assert len(robot.events) == len(case["events"])
    for actual, expected in zip(robot.events, case["events"], strict=True):
        assert actual[0] == expected[0]
        if actual[0] in ("moveL", "setTcp"):
            np.testing.assert_allclose(actual[1], expected[1], atol=1e-12, rtol=0)
            assert actual[2:] == expected[2:]
        else:
            assert actual == expected
