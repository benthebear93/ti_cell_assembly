"""Load saved ArUco poses and hand-eye calibration without hardware imports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from surface_estimator_ur5e.transforms import make_transform

DEFAULT_CALIBRATION = "data/calibration/realsense_ur5e/Calibration.xml"
DEFAULT_INTRINSICS = "data/calibration/realsense_ur5e/camera_intrinsics.yaml"


def load_marker_transform(path: Path) -> tuple[int | None, float, np.ndarray, dict[str, Any]]:
    """Return marker ID, side length in metres, base transform, and source data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Marker pose file not found: {path}. Create it first with: "
            "uv run python scripts/read_aruco_marker_to_base.py "
            "--marker-length-mm 25.4 --marker-id 364 --save --save-samples 50"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    pose = data.get("pose_in_base")
    if not isinstance(pose, dict):
        raise ValueError(f"{path} must contain pose_in_base.")

    if "transform_matrix" in pose:
        base_t_marker = np.asarray(pose["transform_matrix"], dtype=float)
    else:
        translation = np.asarray(pose.get("translation_m"), dtype=float).reshape(3)
        if "rotation_quaternion_xyzw" in pose:
            rotation = Rotation.from_quat(pose["rotation_quaternion_xyzw"]).as_matrix()
        elif "rotation_matrix" in pose:
            rotation = np.asarray(pose["rotation_matrix"], dtype=float).reshape(3, 3)
        else:
            raise ValueError(f"{path} pose_in_base must include marker orientation.")
        base_t_marker = make_transform(rotation, translation)

    if base_t_marker.shape != (4, 4):
        raise ValueError(f"{path} transform_matrix must be 4x4.")
    if not np.all(np.isfinite(base_t_marker)):
        raise ValueError(f"{path} transform_matrix contains non-finite values.")

    marker_id = data.get("marker_id")
    marker_length_m = float(data.get("marker_length_m", 0.0254))
    return int(marker_id) if marker_id is not None else None, marker_length_m, base_t_marker, data


def parse_float_list(text: str | None) -> list[float]:
    if text is None:
        raise ValueError("Missing numeric text in calibration XML.")
    return [float(value) for value in text.split()]


def load_tool_to_camera(path: Path) -> np.ndarray:
    """Read the calibrated tool-to-camera transform in metres from XML."""
    tree = ET.parse(path)
    root = tree.getroot()

    for result in root.findall(".//CalibrationResult"):
        moving = result.find("MovingTransform")
        if moving is None or moving.attrib.get("frame") != "Camera":
            continue

        translation = np.array(parse_float_list(moving.findtext("Vector3D")), dtype=float)
        rotation_elem = moving.find("Rotation3D/Rotation3D")
        if rotation_elem is None:
            raise RuntimeError(f"Missing MovingTransform rotation in {path}")
        rotation = np.array(parse_float_list(rotation_elem.text), dtype=float).reshape(3, 3)
        return make_transform(rotation, translation)

    raise RuntimeError(f"Could not find MovingTransform frame='Camera' in {path}")
