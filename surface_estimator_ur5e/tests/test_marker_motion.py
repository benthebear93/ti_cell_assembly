"""Compare commands with traces captured before the motion refactor."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import assembly_config as config
import assembly_plan as plan
import marker_based_motion as motion
from motion_fakes import FakeRobot, fake_environment

PROJECT = Path(__file__).resolve().parents[1]
TRACES = json.loads((Path(__file__).parent / "fixtures/marker_motion_before.json").read_text())


def make_args(settings=None, *, execute=True):
    settings = dict(settings or {})
    cli = ["--task", settings.pop("task", "pick")]
    if "resume" in settings:
        cli += ["--resume", settings.pop("resume")]
    args = config.parse_args(cli)
    vars(args).update(settings)
    args.execute = execute
    args.marker_pose = PROJECT / "data/markers/aruco_364_in_base.yaml"
    if args.auto_pin_image_align:
        # The camera is replaced in fake_environment; only file existence is validated.
        args.pin_image_reference = Path(__file__)
    return args


def prepare_robot(args, **kwargs):
    robot = FakeRobot(**kwargs)
    if args.resume == "pins":
        aligned = [e[1] for e in TRACES["assembly"]["events"] if e[0] == "moveL"][2]
        robot.pose = np.asarray(aligned).copy()
    return robot


@pytest.mark.parametrize("case", TRACES.values(), ids=TRACES.keys())
def test_robot_commands_match_before_refactor(case):
    args = make_args(case["settings"])
    robot = prepare_robot(args, measured_error=case["measured_error"])
    with fake_environment(motion, args, robot):
        motion.main()
    assert len(robot.events) == len(case["events"])
    for actual, expected in zip(robot.events, case["events"], strict=True):
        assert actual[0] == expected[0]
        if actual[0] in ("moveJ", "moveL", "setTcp"):
            np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=1e-12)
            assert actual[2:] == expected[2:]
        else:
            assert actual == expected


@pytest.mark.parametrize(
    "settings",
    [{"task": task} for task in ("pick", "align", "insert", "assembly")]
    + [
        {"task": "assembly", "resume": resume}
        for resume in ("pins", "release", "after-close", "tail", "tail-next")
    ],
)
def test_dry_run_never_moves_or_operates_gripper(settings):
    args = make_args(settings, execute=False)
    robot = prepare_robot(args)
    with fake_environment(motion, args, robot):
        motion.main()
    assert not {"moveJ", "moveL", "gripper"} & {event[0] for event in robot.events}
    assert robot.events[-3:] == [["stopScript"], ["disconnect"], ["disconnect"]]


@pytest.mark.parametrize(
    "field,value",
    [
        ("speed_m_s", 0),
        ("pin_insertion_speed_m_s", -1),
        ("max_safety_lift_mm", -1),
        ("offset_z_mm", float("nan")),
        ("gripper_close_percent", float("nan")),
        ("gripper_force", 256),
        ("gripper_port", 0),
        ("pin_image_roi", (0, 0, 3)),
        ("tcp_offset_ur", (0, 0, 0)),
        ("initial_q_deg", (0, 0)),
    ],
)
def test_invalid_configuration_fails_before_connecting(field, value):
    args = make_args({field: value})
    robot = FakeRobot()
    with fake_environment(motion, args, robot):
        with patch.object(motion, "connect_rtde_control") as connect:
            with pytest.raises(ValueError):
                motion.main()
            connect.assert_not_called()
    assert robot.events == []


def test_unreachable_resume_sends_no_motion():
    args = make_args({"resume": "tail"})
    robot = FakeRobot(no_ik=True)
    with fake_environment(motion, args, robot), pytest.raises(RuntimeError, match="No IK"):
        motion.main()
    assert not any(e[0] in ("moveL", "gripper") for e in robot.events)
    assert robot.events[-3:] == [["stopScript"], ["disconnect"], ["disconnect"]]


def test_failed_move_stops_the_sequence_and_disconnects():
    args = make_args({"resume": "tail"})
    robot = FakeRobot(fail_move=True)
    with fake_environment(motion, args, robot), pytest.raises(RuntimeError, match="returned false"):
        motion.main()
    assert sum(e[0] == "moveL" for e in robot.events) == 1
    assert robot.events[-3:] == [["stopScript"], ["disconnect"], ["disconnect"]]


@pytest.mark.parametrize("joint", (2, 4))
def test_actual_singularity_margin_stops_later_moves(joint):
    args = make_args({"resume": "tail"})
    robot = FakeRobot()
    robot.q[joint] = 0
    with (
        fake_environment(motion, args, robot),
        pytest.raises(RuntimeError, match="singularity margin"),
    ):
        motion.main()
    assert sum(e[0] == "moveL" for e in robot.events) == 1


def test_pin_approach_distance_limit_is_preserved():
    args = make_args({"task": "assembly", "resume": "pins", "max_pin_approach_mm": 0})
    robot = prepare_robot(args)
    with (
        fake_environment(motion, args, robot),
        pytest.raises(RuntimeError, match="max_pin_approach_mm"),
    ):
        motion.main()
    assert not any(e[0] in ("moveL", "gripper") for e in robot.events)


def test_cleanup_disconnects_even_when_stop_script_fails():
    args = make_args(execute=False)
    robot = FakeRobot()
    with fake_environment(motion, args, robot):
        with patch.object(robot, "stopScript", side_effect=RuntimeError("stop failed")):
            with pytest.raises(RuntimeError, match="stop failed"):
                motion.main()
    assert robot.events[-2:] == [["disconnect"], ["disconnect"]]


def test_preflight_rejects_ik_branch_jump():
    robot = FakeRobot()
    with patch.object(robot, "getInverseKinematics", return_value=(robot.q + 1).tolist()):
        with pytest.raises(RuntimeError, match="branch jump"):
            plan.preflight_alignment_ik(robot, robot.q, [robot.pose], [])


@pytest.mark.parametrize("q", [[0.0] * 5, [float("nan")] * 6])
def test_invalid_ik_result_is_rejected(q):
    robot = FakeRobot()
    with patch.object(robot, "getInverseKinematics", return_value=q):
        with pytest.raises(RuntimeError, match="Invalid IK result"):
            plan.require_ik_solution(robot, robot.q, robot.pose, "test target")


def test_image_translation_limit_checks_combined_x_z_distance(tmp_path):
    metadata = tmp_path / "correction.yaml"
    metadata.write_text(
        "axis: y\nangle_deg: 0\ntcp_local_x_offset_mm: 8\ntcp_local_z_offset_mm: 8\n"
    )
    args = make_args({"max_pin_image_correction_mm": 10})
    with pytest.raises(RuntimeError, match="max_pin_image_correction_mm"):
        plan.print_pin_image_alignment_correction_plan(
            FakeRobot().pose,
            args,
            metadata_path=metadata,
        )


@pytest.mark.parametrize("task", ("align", "insert", "assembly"))
def test_preview_and_robot_use_the_same_targets(task):
    import preview_assembly_motion as preview

    args = make_args({"task": task}, execute=False)
    _, _, _, events = preview.build_actual_motion_events(args, FakeRobot())
    expected_moves = [e for e in TRACES[task]["events"] if e[0] == "moveL"]
    preview_moves = [e for e in events if e.kind == "move"]
    assert len(preview_moves) == len(expected_moves)
    for event, command in zip(preview_moves, expected_moves, strict=True):
        np.testing.assert_allclose(event.target_pose_ur, command[1], rtol=0, atol=1e-12)
        assert event.speed_m_s == command[2]
