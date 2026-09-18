"""Coordinate conventions, calibration formats, and RTDE resource ownership."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from surface_estimator_ur5e import calibration, robot_io, transforms
from surface_estimator_ur5e import workcell_geometry as workcell

PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("angle", [0, 1e-9, 90, 179.999, 180, -180, 270])
def test_ur_pose_round_trip_preserves_the_transform(angle):
    pose = np.array([0.1, -0.2, 0.3, 0.0, np.deg2rad(angle), 0.0])
    matrix = transforms.ur_pose_to_transform(pose)
    round_trip = transforms.ur_pose_to_transform(transforms.transform_to_ur_pose(matrix))
    np.testing.assert_allclose(round_trip, matrix, atol=1e-14)
    np.testing.assert_allclose(matrix[:3, 3], pose[:3])


def test_local_translation_uses_the_starting_frame_before_rotation():
    # Starting TCP +X points along base +Y. Rotating it afterward must not
    # redirect the requested 10 mm translation toward the new TCP +X.
    start = np.array([1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2])
    original = start.copy()
    result = transforms.pose_with_local_xz_adjustment(start, "z", 90.0, 10.0, -20.0)
    np.testing.assert_allclose(result[:3], [1.0, 2.01, 2.98], atol=1e-14)
    np.testing.assert_allclose(
        transforms.ur_pose_to_transform(result)[:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=1e-14
    )
    np.testing.assert_array_equal(start, original)


@pytest.mark.parametrize(
    "orientation", ["transform_matrix", "rotation_matrix", "rotation_quaternion_xyzw"]
)
def test_marker_yaml_orientation_formats_are_equivalent(tmp_path, orientation):
    expected = transforms.ur_pose_to_transform([0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2])
    pose = {"translation_m": expected[:3, 3].tolist()}
    pose[orientation] = {
        "transform_matrix": expected.tolist(),
        "rotation_matrix": expected[:3, :3].tolist(),
        "rotation_quaternion_xyzw": Rotation.from_matrix(expected[:3, :3]).as_quat().tolist(),
    }[orientation]
    data = {"marker_id": 364, "marker_length_m": 0.0254, "pose_in_base": pose}
    path = tmp_path / "marker.yaml"
    path.write_text(yaml.safe_dump(data))
    marker_id, side_m, matrix, metadata = calibration.load_marker_transform(path)
    assert (marker_id, side_m, metadata) == (364, 0.0254, data)
    np.testing.assert_allclose(matrix, expected, atol=1e-14)


@pytest.mark.parametrize("matrix", [np.zeros((3, 4)), np.full((4, 4), np.nan)])
def test_invalid_saved_transform_is_rejected(tmp_path, matrix):
    path = tmp_path / "marker.yaml"
    path.write_text(yaml.safe_dump({"pose_in_base": {"transform_matrix": matrix.tolist()}}))
    with pytest.raises(ValueError, match="4x4|non-finite"):
        calibration.load_marker_transform(path)


def test_hand_eye_loader_selects_camera_and_preserves_translation_units(tmp_path):
    path = tmp_path / "calibration.xml"
    path.write_text("""<WorkCellCalibration>
      <CalibrationResult><MovingTransform frame="Marker"/></CalibrationResult>
      <CalibrationResult><MovingTransform frame="Camera">
        <Vector3D>0.01 -0.02 0.03</Vector3D>
        <Rotation3D><Rotation3D>0 -1 0 1 0 0 0 0 1</Rotation3D></Rotation3D>
      </MovingTransform></CalibrationResult>
    </WorkCellCalibration>""")
    np.testing.assert_allclose(
        calibration.load_tool_to_camera(path),
        [
            [0, -1, 0, 0.01],
            [1, 0, 0, -0.02],
            [0, 0, 1, 0.03],
            [0, 0, 0, 1],
        ],
    )


def test_tray_attachment_maps_to_tcp_and_floor_clearance_has_correct_sign():
    tcp = transforms.ur_pose_to_transform([0.1, -0.2, 0.3, 0.0, 0.0, np.pi / 2])
    attachment = np.array([0.0, -0.05, 0.02])
    vertices = np.array([attachment, attachment + [0, 0.01, 0]])
    placed = workcell.tray_vertices_in_base(tcp, vertices, attachment, 20.0, 90.0)
    np.testing.assert_allclose(placed, [[0.1, -0.2, 0.32], [0.1, -0.2, 0.33]])
    assert workcell.tray_floor_clearance_m(
        tcp, vertices, attachment, 0.35, 20, 90
    ) == pytest.approx(0.02)
    assert workcell.tray_floor_clearance_m(
        tcp, vertices, attachment, 0.32, 20, 90
    ) == pytest.approx(-0.01)


@pytest.mark.parametrize("rpy", [(25, -15, 60), (0, 90, 0)])
def test_floor_frame_keeps_origin_and_projects_axes(rpy):
    raw = transforms.make_transform(
        Rotation.from_euler("xyz", rpy, degrees=True).as_matrix(), [1, 2, 3]
    )
    floor = workcell.floor_constrained_marker_transform(raw)
    np.testing.assert_allclose(floor[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(floor[:3, 2], [0, 0, 1])
    np.testing.assert_allclose(floor[:3, :3].T @ floor[:3, :3], np.eye(3), atol=1e-14)
    assert np.linalg.det(floor[:3, :3]) == pytest.approx(1)


def test_degenerate_floor_frame_fails_instead_of_returning_nan():
    with pytest.raises(ValueError, match="project marker x-axis"):
        workcell.floor_constrained_marker_transform(np.zeros((4, 4)))


def test_project_paths_do_not_depend_on_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert workcell.resolve_project_path(workcell.DEFAULT_TI_ASSEMBLY_OBJ).is_file()
    assert workcell.resolve_project_path(tmp_path) == tmp_path


@pytest.mark.parametrize("pose", [[0] * 5, [0, 0, np.nan, 0, 0, 0]])
def test_invalid_tcp_sample_closes_receive_connection(monkeypatch, pose):
    receive = Mock()
    receive.getActualTCPPose.return_value = pose
    monkeypatch.setattr(robot_io, "connect_rtde_receive", lambda ip: receive)
    with pytest.raises(RuntimeError, match="6D TCP|non-finite"):
        robot_io.read_current_tcp_pose("test-robot")
    receive.disconnect.assert_called_once_with()


def test_valid_tcp_sample_leaves_connection_owned_by_caller(monkeypatch):
    receive = Mock()
    receive.getActualTCPPose.return_value = [0, 0, 0.2, 0, 0, 0]
    monkeypatch.setattr(robot_io, "connect_rtde_receive", lambda ip: receive)
    connection, pose = robot_io.read_current_tcp_pose("test-robot")
    assert connection is receive
    np.testing.assert_array_equal(pose, receive.getActualTCPPose.return_value)
    receive.disconnect.assert_not_called()


def test_failed_tcp_setting_is_not_ignored():
    control = Mock()
    control.setTcp.return_value = False
    with pytest.raises(RuntimeError, match="Failed to set active TCP"):
        robot_io.set_tcp_offset(control, (0, 0, 0.158, -1.5707, 0, 0))


def test_camera_metadata_releases_robot_before_loading_calibration(monkeypatch):
    # The helper needs neither a physical camera nor its driver to run this test.
    monkeypatch.setitem(sys.modules, "pyrealsense2", SimpleNamespace())
    capture = importlib.import_module("capture_realsense_frame")
    receive = Mock()
    monkeypatch.setattr(capture, "read_current_tcp_pose", lambda ip: (receive, np.zeros(6)))
    monkeypatch.setattr(
        capture, "load_tool_to_camera", Mock(side_effect=ValueError("bad calibration"))
    )
    with pytest.raises(ValueError, match="bad calibration"):
        capture.read_robot_metadata("test-robot", Path("bad.xml"))
    receive.disconnect.assert_called_once_with()


def test_motion_imports_do_not_load_viewer_or_hardware_drivers():
    code = """
import sys
class BlockHardwareAndViewer:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rtde_control', 'rtde_receive', 'pyrealsense2', 'viser', 'visualize_marker_frame'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockHardwareAndViewer())
import marker_based_motion
import align_tray_to_pins_rotation
"""
    env = dict(
        os.environ, PYTHONPATH=os.pathsep.join([str(PROJECT / "src"), str(PROJECT / "scripts")])
    )
    subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )
