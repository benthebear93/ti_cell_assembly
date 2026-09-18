from pathlib import Path

import pytest

import assembly_config as config


def test_config_values_and_cli_override(tmp_path):
    path = tmp_path / "calibration.yaml"
    path.write_text("speed_m_s: 0.02\npin_insertion_speed_m_s: 0.003\ntray_obj: assets/tray.obj\n")
    args = config.parse_args(["--config", str(path), "--speed-m-s", "0.01", "--task", "assembly"])
    assert args.speed_m_s == 0.01
    assert args.pin_insertion_speed_m_s == 0.003
    assert args.tray_obj == Path("assets/tray.obj")
    assert args.align_to_four_pin_frame and args.insert_after_pin_approach
    assert args.post_insert_release_retreat
    assert not args.execute


@pytest.mark.parametrize(
    "data",
    [
        "execute: true",
        "resume: tail",
        "speed_m_ss: 0.05",
        "speed_m_s: fast",
        "speed_m_s: true",
        "skip_gripper_close: 'false'",
        "pin_image_fps: 30.5",
        "initial_q_deg: nope",
        "tray_obj: 42",
        "[]",
        "null",
    ],
)
def test_bad_yaml_cannot_silently_change_robot_behavior(tmp_path, data):
    path = tmp_path / "bad.yaml"
    path.write_text(data)
    with pytest.raises(ValueError):
        config.load_config(path)


def test_old_resume_flags_are_rejected():
    with pytest.raises(SystemExit):
        config.parse_args(["--continue-post-two-pin-tail-from-current"])


def test_removed_noop_resume_is_rejected():
    with pytest.raises(SystemExit):
        config.parse_args(["--resume", "tail-extra"])


@pytest.mark.parametrize(
    "task,align,insert,retreat",
    [
        ("pick", False, False, False),
        ("align", True, False, False),
        ("insert", True, True, False),
        ("assembly", True, True, True),
    ],
)
def test_tasks_have_explicit_stopping_points(task, align, insert, retreat):
    args = config.parse_args(["--task", task])
    assert (
        args.align_to_four_pin_frame,
        args.insert_after_pin_approach,
        args.post_insert_release_retreat,
    ) == (align, insert, retreat)
    assert args.start_from_initial_pose
    assert not args.execute


def test_insertion_task_cannot_silently_skip_pin_approach():
    args = config.parse_args(["--task", "insert"])
    args.approach_pin_after_rotation = False
    with pytest.raises(ValueError, match="Insertion tasks"):
        config.validate_args(args)


def test_example_config_preserves_defaults():
    from dataclasses import asdict

    project = Path(__file__).resolve().parents[1]
    assert asdict(config.load_config(project / "data/assembly_motion.yaml")) == asdict(
        config.MotionConfig()
    )
