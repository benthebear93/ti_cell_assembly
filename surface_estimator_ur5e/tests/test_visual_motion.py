"""Preview geometry and event order, checked without robot or graphics connections."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import assembly_config as config
import preview_assembly_motion as preview
import visualize_marker_frame as viewer
from motion_fakes import FakeRobot, INITIAL_POSE, INITIAL_Q
from test_marker_motion import make_args
from surface_estimator_ur5e.calibration import load_marker_transform
from surface_estimator_ur5e.transforms import transform_points, ur_pose_to_transform
from surface_estimator_ur5e.visualization import _ceiling_mount_display_transform
from surface_estimator_ur5e.workcell_geometry import floor_constrained_marker_transform

BASELINE = json.loads((Path(__file__).parent / "fixtures/visual_motion_before.json").read_text())


def viewer_args(settings=None):
    with patch.object(sys, "argv", ["visualize_marker_frame"]):
        args = viewer.parse_args()
    for name, value in (settings or {}).items():
        setattr(args, name, Path(value) if "metadata" in name else value)
    return args


def marker_transform(args):
    marker = load_marker_transform(args.marker_pose)[2]
    return (
        floor_constrained_marker_transform(marker) if args.marker_frame_mode == "floor" else marker
    )


@pytest.mark.parametrize("case", BASELINE["viewer"].values(), ids=BASELINE["viewer"])
def test_trajectory_matches_before_cleanup(case):
    args = viewer_args(case["settings"])
    robot = FakeRobot()
    plan = viewer.build_motion_trajectory(
        marker_transform(args), INITIAL_POSE, INITIAL_Q, robot, args
    )
    assert len(plan.stages) == len(case["stages"])
    for actual, expected in zip(plan.stages, case["stages"], strict=True):
        assert len(actual.samples) == expected["sample_count"]
        np.testing.assert_allclose(actual.samples[0], expected["start"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual.samples[-1], expected["target"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(plan.tcp_points, case["tcp_points"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        plan.tray_reference_points, case["tray_reference_points"], rtol=0, atol=1e-12
    )
    assert robot.events == []


@pytest.mark.parametrize("case", BASELINE["viewer"].values(), ids=BASELINE["viewer"])
def test_static_frames_match_before_cleanup(case):
    args = viewer_args(case["settings"])
    frames = []
    with (
        patch.object(
            viewer,
            "add_frame",
            side_effect=lambda server, name, transform, *a, **kw: frames.append(transform),
        ),
        patch.object(viewer, "add_point"),
        patch.object(viewer, "add_line_segment"),
    ):
        viewer.add_motion_plan_frames(
            None, np.eye(4), marker_transform(args), ur_pose_to_transform(INITIAL_POSE), args
        )
    np.testing.assert_allclose(frames, case["frames"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", BASELINE["events"].values(), ids=BASELINE["events"])
def test_animation_commands_match_before_cleanup(case):
    args = make_args(case["settings"], execute=False)
    robot = FakeRobot()
    _, _, _, events = preview.build_actual_motion_events(args, robot)
    assert len(events) == len(case["events"])
    for actual, expected in zip(events, case["events"], strict=True):
        actual_fields = vars(actual).copy()
        target = actual_fields.pop("target_pose_ur")
        assert actual_fields == {k: v for k, v in expected.items() if k != "target_pose_ur"}
        if target is None:
            assert expected["target_pose_ur"] is None
        else:
            np.testing.assert_allclose(target, expected["target_pose_ur"], rtol=0, atol=1e-12)
    assert robot.events == []


def test_full_trajectory_includes_every_recorded_tail_move():
    args = viewer_args({"insert_after_pin_approach": True})
    # Match execution settings here; the static viewer intentionally has legacy tail defaults.
    for name, value in vars(config.MotionConfig()).items():
        if hasattr(args, name):
            setattr(args, name, value)
    plan = viewer.build_motion_trajectory(
        marker_transform(args), INITIAL_POSE, INITIAL_Q, FakeRobot(), args
    )
    expected = [e for e in BASELINE["events"]["assembly"]["events"] if e["kind"] == "move"]
    assert [stage.name for stage in plan.stages] == [event["name"] for event in expected]
    for stage, event in zip(plan.stages, expected, strict=True):
        np.testing.assert_allclose(
            stage.samples[-1], ur_pose_to_transform(event["target_pose_ur"]), rtol=0, atol=1e-12
        )
    # Opening the gripper leaves the tray at insertion while the TCP continues through the tail.
    insertion = viewer.build_motion_trajectory(
        marker_transform(args),
        INITIAL_POSE,
        INITIAL_Q,
        FakeRobot(),
        viewer_args({"insert_after_pin_approach": True, "post_insert_release_retreat": False}),
    )
    np.testing.assert_allclose(
        plan.tray_reference_points, insertion.tray_reference_points, atol=1e-12
    )


class RecordingScene:
    """Exercise real drawing helpers and reject invalid coordinates or overwritten paths."""

    def __init__(self):
        self.names = set()
        self.positions = []

    def record(self, name, **kwargs):
        assert name not in self.names
        self.names.add(name)
        for key in ("position", "points", "wxyz"):
            if key in kwargs:
                values = np.asarray(kwargs[key])
                assert np.all(np.isfinite(values))
        if "position" in kwargs:
            self.positions.append(np.asarray(kwargs["position"]))

    add_frame = record
    add_icosphere = record
    add_line_segments = record
    add_point_cloud = record


@pytest.mark.parametrize("trajectory", [False, True])
def test_full_preview_draws_with_unique_names_and_camera_in_display_frame(trajectory):
    args = viewer_args({"insert_after_pin_approach": True})
    marker = marker_transform(args)
    display = _ceiling_mount_display_transform()
    scene = RecordingScene()
    if trajectory:
        points = viewer.add_motion_trajectory(
            scene, display, marker, INITIAL_POSE, INITIAL_Q, FakeRobot(), args
        )
    else:
        points = viewer.add_motion_plan_frames(
            scene, display, marker, ur_pose_to_transform(INITIAL_POSE), args
        )
    assert len(scene.names) > 50
    assert points.shape[1] == 3 and np.isfinite(points).all()
    # Every rendered frame/point must contribute to camera fitting in the same coordinate frame.
    for position in scene.positions:
        assert np.min(np.linalg.norm(points - position, axis=1)) < 1e-12
    displayed_start = transform_points(display, INITIAL_POSE[None, :3])[0]
    assert np.min(np.linalg.norm(points - displayed_start, axis=1)) < 1e-12


@pytest.mark.parametrize(
    "limit", ["max_pin_insertion_mm", "max_pin_image_correction_mm", "max_pin_image_correction_deg"]
)
def test_preview_preserves_correction_and_insertion_limits(limit):
    args = viewer_args({"insert_after_pin_approach": True, limit: 0.0})
    with pytest.raises(RuntimeError, match="exceeds"):
        viewer.build_motion_trajectory(
            marker_transform(args), INITIAL_POSE, INITIAL_Q, FakeRobot(), args
        )


@pytest.mark.parametrize("fail_at", ["grasp", "pin"])
def test_unreachable_preview_target_is_rejected_without_moving(fail_at):
    args = viewer_args()
    robot = FakeRobot()
    original = viewer.motion.require_ik_solution

    def checked_ik(ik, q, pose, name):
        if (fail_at == "grasp" and name == "marker grasp") or (
            fail_at == "pin" and name == "post-rotation pin centering"
        ):
            ik.no_ik = True
        return original(ik, q, pose, name)

    with patch.object(viewer.motion, "require_ik_solution", side_effect=checked_ik):
        with pytest.raises(RuntimeError, match="No IK solution"):
            viewer.build_motion_trajectory(
                marker_transform(args), INITIAL_POSE, INITIAL_Q, robot, args
            )
    assert robot.events == []
