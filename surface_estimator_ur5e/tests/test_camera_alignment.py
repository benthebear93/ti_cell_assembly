"""Exercise the search loop while replacing camera capture and image scoring."""

from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

import assembly_config as config
import marker_based_motion as motion
from motion_fakes import FakeRobot


def run_camera_search(module, scores, *, fail_move=False):
    args = config.parse_args([])
    args.auto_pin_image_align = True
    robot = FakeRobot(fail_move=fail_move)
    pipeline = Mock()
    captures = []
    frame_index = 0

    def capture(*_):
        nonlocal frame_index
        frame_index += 1
        return np.full((2, 2, 3), frame_index, dtype=np.uint8)

    def save(directory, image, iteration, index, state, score, pose, args, label="candidate"):
        captures.append(
            [iteration, index, list(state), score, pose.tolist(), label, int(image[0, 0, 0])]
        )
        return Path("fake_color.png"), Path("fake_metadata.yaml")

    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(
            patch.object(module, "load_reference_alignment_image", return_value=(None, None))
        )
        stack.enter_context(
            patch.object(module, "auto_alignment_output_dir", return_value=Path("unused"))
        )
        stack.enter_context(patch.object(module, "start_pin_image_pipeline", return_value=pipeline))
        stack.enter_context(patch.object(module, "capture_pin_image", side_effect=capture))
        stack.enter_context(patch.object(module, "score_alignment_image", side_effect=scores))
        stack.enter_context(patch.object(module, "save_auto_alignment_capture", side_effect=save))
        stack.enter_context(
            patch.object(
                module.time,
                "sleep",
                side_effect=lambda seconds: robot.events.append(["sleep", seconds]),
            )
        )
        try:
            module.execute_auto_pin_image_alignment(robot, robot, args)
        except RuntimeError:
            pipeline.stop.assert_called_once()
            raise
    pipeline.stop.assert_called_once()
    return {"events": robot.events, "captures": captures}


@pytest.mark.parametrize(
    "name,scores",
    [
        ("improving", list(range(14))),
        ("keep_earlier_best", [0, 1, 4, 1, 6, 1, 1] + [5] * 7),
    ],
)
def test_search_matches_original_candidate_order_and_best_image(name, scores):
    fixture = Path(__file__).parent / "fixtures/camera_search_before.json"
    expected = json.loads(fixture.read_text())[name]
    actual = run_camera_search(motion, scores)
    assert len(actual["events"]) == len(expected["events"])
    for got, want in zip(actual["events"], expected["events"], strict=True):
        assert got[0] == want[0]
        if got[0] == "moveL":
            np.testing.assert_allclose(got[1], want[1], atol=1e-12, rtol=0)
            assert got[2:] == want[2:]
        else:
            assert got == want
    assert len(actual["captures"]) == len(expected["captures"])
    for got, want in zip(actual["captures"], expected["captures"], strict=True):
        assert got[:4] == want[:4]
        np.testing.assert_allclose(got[4], want[4], atol=1e-12, rtol=0)
        assert got[5:] == want[5:]


def test_camera_stops_when_robot_move_fails():
    with pytest.raises(RuntimeError, match="returned false"):
        run_camera_search(motion, list(range(14)), fail_move=True)
