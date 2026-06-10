"""MJCF export for a fixed ceiling-mounted UR5e, Hand-E, and fitted table."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from surface_estimator_ur5e.geometry import PlaneEstimate
from surface_estimator_ur5e.io import ContactData
from surface_estimator_ur5e.robot_model import SimpleUR5eVisualizer, URDFRobotVisualizer
from surface_estimator_ur5e.visualization import (
    _ceiling_mount_display_transform,
    _transform_point,
    _transform_points,
)


@dataclass(frozen=True)
class MujocoSceneMetadata:
    """Useful values from the generated MuJoCo scene."""

    output_path: Path
    table_center_m: np.ndarray
    table_quat_wxyz: np.ndarray
    table_half_extents_m: np.ndarray
    plane_centroid_world_m: np.ndarray
    plane_normal_world: np.ndarray
    fk_source: str


_URDF_LINK_PATH = (
    "base",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
    "tool0",
)

_LINK_RADII_M = (0.065, 0.06, 0.05, 0.044, 0.038, 0.034, 0.03)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UR5E_MJCF_PATH = _PROJECT_ROOT / "assets" / "universal_robots_ur5e" / "ur5e.xml"
_HANDE_MESH_DIR = _PROJECT_ROOT / "assets" / "robotiq_hande_description" / "meshes"
_HANDE_MESH_FILES = {
    "hande_io_coupler": _HANDE_MESH_DIR / "io_coupler.obj",
    "hande_body": _HANDE_MESH_DIR / "hande.obj",
    "hande_finger": _HANDE_MESH_DIR / "finger.obj",
}
_HANDE_COUPLER_HEIGHT_M = 0.011
_HANDE_BODY_HEIGHT_M = 0.099
_HANDE_FINGER_OPENING_M = 0.020
_HANDE_GRASP_FINGER_OFFSET_M = 0.038
_HANDE_GRASP_FINGER_RANGE_M = 0.030
_HANDE_GRASP_CLOSE_M = 0.030
_TI_TRAY_MESH_PATH = _PROJECT_ROOT / "assets" / "ti_tray" / "ti_tray.obj"
_TI_TRAY_PARTS_DIR = _PROJECT_ROOT / "assets" / "ti_tray" / "mujoco_parts"
_TABLE_ALUMINUM_RGBA = (0.72, 0.74, 0.72, 1.0)
_TRAY_GREEN_RGBA = (0.0, 0.58, 0.22, 1.0)
_TI_TRAY_TABLE_CLEARANCE_M = 0.001
_TI_TRAY_TOTAL_MASS_KG = 0.005
_TI_TRAY_PROTRUSION_CENTER = (0.0, -0.05150, 0.02750)
_TI_TRAY_PROTRUSION_SIZE = (0.01100, 0.01200, 0.01650)
_UR5E_ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
_UR5E_MJCF_JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_UR5E_MJCF_JOINT_LIMITS_RAD = {
    "shoulder_pan_joint": (-6.2831, 6.2831),
    "shoulder_lift_joint": (-6.2831, 6.2831),
    "elbow_joint": (-3.1415, 3.1415),
    "wrist_1_joint": (-6.2831, 6.2831),
    "wrist_2_joint": (-6.2831, 6.2831),
    "wrist_3_joint": (-6.2831, 6.2831),
}
_JOINT_LIMIT_TOL_RAD = 1e-6


@dataclass(frozen=True)
class _MaterialAsset:
    name: str
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class _MeshAsset:
    name: str
    path: Path
    inertia: str | None = None


@dataclass(frozen=True)
class _StaticMeshGeom:
    name: str
    mesh_name: str
    material_name: str | None
    pos: np.ndarray
    quat_wxyz: np.ndarray


@dataclass(frozen=True)
class _RobotGeometry:
    source: str
    tool0_transform_world: np.ndarray
    mesh_geoms: tuple[_StaticMeshGeom, ...] = ()
    mesh_assets: tuple[_MeshAsset, ...] = ()
    material_assets: tuple[_MaterialAsset, ...] = ()
    capsule_points: np.ndarray | None = None


def write_mujoco_scene(
    contact_data: ContactData,
    plane_estimate: PlaneEstimate,
    output_path: str | Path,
    selected_contact_index: int = 0,
    table_margin_m: float = 0.18,
    table_thickness_m: float = 0.05,
    min_table_half_extent_m: float = 0.35,
    compile_check: bool = False,
) -> MujocoSceneMetadata:
    """Write a fixed MJCF scene for MuJoCo.

    The MuJoCo world matches the browser visualization convention: the robot base
    is fixed at the ceiling origin and the contacted table is below it.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xml, metadata = build_mujoco_scene_xml(
        contact_data=contact_data,
        plane_estimate=plane_estimate,
        output_path=output,
        selected_contact_index=selected_contact_index,
        table_margin_m=table_margin_m,
        table_thickness_m=table_thickness_m,
        min_table_half_extent_m=min_table_half_extent_m,
    )
    output.write_text(xml, encoding="utf-8")
    if compile_check:
        compile_mujoco_xml_path(output)
    return metadata


def write_mujoco_control_scene(
    contact_data: ContactData,
    plane_estimate: PlaneEstimate,
    output_path: str | Path,
    selected_contact_index: int = 0,
    table_margin_m: float = 0.18,
    table_thickness_m: float = 0.05,
    min_table_half_extent_m: float = 0.35,
    compile_check: bool = False,
) -> MujocoSceneMetadata:
    """Write an actuated MJCF scene with the UR5e position controller intact."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xml, metadata = build_mujoco_control_scene_xml(
        contact_data=contact_data,
        plane_estimate=plane_estimate,
        output_path=output,
        selected_contact_index=selected_contact_index,
        table_margin_m=table_margin_m,
        table_thickness_m=table_thickness_m,
        min_table_half_extent_m=min_table_half_extent_m,
    )
    output.write_text(xml, encoding="utf-8")
    if compile_check:
        compile_mujoco_xml_path(output)
    return metadata


def write_mujoco_grasp_scene(
    contact_data: ContactData,
    plane_estimate: PlaneEstimate,
    output_path: str | Path,
    selected_contact_index: int = 0,
    table_margin_m: float = 0.18,
    table_thickness_m: float = 0.05,
    min_table_half_extent_m: float = 0.35,
    tray_collision_mode: str = "proxy",
    compile_check: bool = False,
) -> MujocoSceneMetadata:
    """Write an actuated MJCF scene for grasping the tray protrusion."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    xml, metadata = build_mujoco_control_scene_xml(
        contact_data=contact_data,
        plane_estimate=plane_estimate,
        output_path=output,
        selected_contact_index=selected_contact_index,
        table_margin_m=table_margin_m,
        table_thickness_m=table_thickness_m,
        min_table_half_extent_m=min_table_half_extent_m,
        dynamic_tray=True,
        actuated_hande=True,
        tray_collision_mode=tray_collision_mode,
    )
    output.write_text(xml, encoding="utf-8")
    if compile_check:
        compile_mujoco_xml_path(output)
    return metadata


def build_mujoco_scene_xml(
    contact_data: ContactData,
    plane_estimate: PlaneEstimate,
    output_path: Path,
    selected_contact_index: int = 0,
    table_margin_m: float = 0.18,
    table_thickness_m: float = 0.05,
    min_table_half_extent_m: float = 0.35,
) -> tuple[str, MujocoSceneMetadata]:
    """Build an MJCF XML string and return export metadata."""

    if selected_contact_index < 0 or selected_contact_index >= len(contact_data.contacts):
        raise ValueError("selected_contact_index is out of range.")
    if table_thickness_m <= 0.0:
        raise ValueError("table_thickness_m must be positive.")
    if table_margin_m < 0.0:
        raise ValueError("table_margin_m must be non-negative.")
    if min_table_half_extent_m <= 0.0:
        raise ValueError("min_table_half_extent_m must be positive.")

    display_transform = _ceiling_mount_display_transform()
    contact_points_world = _transform_points(display_transform, contact_data.contact_points())
    centroid_world = _transform_point(display_transform, plane_estimate.centroid)
    fitted_normal_world = display_transform[:3, :3] @ plane_estimate.normal
    fitted_normal_world = _unit(fitted_normal_world)
    if fitted_normal_world[2] < 0.0:
        fitted_normal_world = -fitted_normal_world

    normal_world = np.array([0.0, 0.0, 1.0])
    table_frame = _horizontal_table_frame(centroid_world)
    table_half_extents = _table_half_extents(
        contact_points_world,
        table_frame,
        table_margin_m=table_margin_m,
        table_thickness_m=table_thickness_m,
        min_half_extent_m=min_table_half_extent_m,
    )
    table_center = centroid_world - normal_world * table_half_extents[2]
    table_quat = _quat_wxyz(table_frame[:3, :3])

    selected = contact_data.contacts[selected_contact_index]
    robot_geometry = _robot_geometry_world(
        selected.q_rad,
        contact_data.robot.joint_order,
        display_transform,
    )
    tcp_transform_world = display_transform @ selected.tcp_transform

    root = ET.Element("mujoco", {"model": "ceiling_ur5e_hande_estimated_table"})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "coordinate": "local",
            "autolimits": "true",
        },
    )
    ET.SubElement(root, "option", {"timestep": "0.002", "gravity": "0 0 -9.81"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.55 0.55 0.55",
            "diffuse": "0.45 0.45 0.45",
            "specular": "0.12 0.12 0.12",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "1 1 1 1"})
    ET.SubElement(
        root,
        "statistic",
        {
            "center": _fmt_vec(_scene_center(contact_points_world)),
            "extent": "1.8",
        },
    )

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "white_skybox",
            "type": "skybox",
            "builtin": "flat",
            "rgb1": "1 1 1",
            "rgb2": "1 1 1",
            "width": "32",
            "height": "32",
        },
    )
    _add_material(asset, "ceiling_mat", (0.72, 0.75, 0.78, 1.0))
    _add_material(asset, "table_mat", _TABLE_ALUMINUM_RGBA)
    _add_material(asset, "robot_mat", (0.22, 0.35, 0.62, 1.0))
    _add_material(asset, "joint_mat", (0.08, 0.10, 0.12, 1.0))
    _add_material(asset, "hande_mat", (0.08, 0.08, 0.08, 1.0))
    _add_material(asset, "hande_body_mat", (0.09, 0.10, 0.11, 1.0))
    _add_material(asset, "hande_finger_mat", (0.18, 0.18, 0.17, 1.0))
    _add_material(asset, "hande_metal_mat", (0.58, 0.60, 0.62, 1.0))
    _add_material(asset, "contact_mat", (1.0, 0.66, 0.12, 1.0))
    _add_material(asset, "tray_mat", _TRAY_GREEN_RGBA)
    tray_meshes = _ti_tray_mesh_assets()
    for mesh in tray_meshes:
        _add_mesh(asset, mesh, output_path.parent)
    for mesh in _hande_mesh_assets():
        _add_mesh(asset, mesh, output_path.parent)
    for material in robot_geometry.material_assets:
        _add_material(asset, material.name, material.rgba)
    for mesh in robot_geometry.mesh_assets:
        _add_mesh(asset, mesh, output_path.parent)

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"name": "key", "pos": "0 -1 1.2", "dir": "0 1 -1"})
    ET.SubElement(
        world,
        "camera",
        {
            "name": "overview",
            "pos": "1.2 -1.5 0.6",
            "xyaxes": "0.78 0.62 0 -0.24 0.30 0.92",
        },
    )

    world.append(
        ET.Comment(
            "Plane centroid and normals are expressed in MuJoCo world coordinates. "
            "The table top uses the fitted centroid but is flattened to the robot-base plane."
        )
    )
    world.append(
        ET.Comment(
            f"centroid={_fmt_vec(centroid_world)} fitted_normal={_fmt_vec(fitted_normal_world)} "
            f"table_normal={_fmt_vec(normal_world)} "
            f"rms_error_m={plane_estimate.rms_error:.9f}"
        )
    )

    _add_box_geom(
        world,
        name="ceiling_panel",
        pos=(0.0, 0.0, 0.025),
        quat=(1.0, 0.0, 0.0, 0.0),
        size=(0.75, 0.75, 0.025),
        material="ceiling_mat",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "ceiling_mount",
            "type": "cylinder",
            "pos": "0 0 -0.035",
            "size": "0.085 0.035",
            "material": "joint_mat",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _add_box_geom(
        world,
        name="estimated_table",
        pos=table_center,
        quat=table_quat,
        size=table_half_extents,
        material="table_mat",
        rgba=_TABLE_ALUMINUM_RGBA,
    )
    if tray_meshes:
        _add_ti_tray_geom(
            world,
            table_frame,
            target_center_world=contact_points_world[selected_contact_index],
            mesh_assets=tray_meshes,
        )

    if robot_geometry.mesh_geoms:
        _add_static_mesh_geoms(world, robot_geometry.mesh_geoms)
    elif robot_geometry.capsule_points is not None:
        _add_fixed_robot_geoms(world, robot_geometry.capsule_points)
    _add_hande_geoms(world, robot_geometry.tool0_transform_world, tcp_transform_world)
    _add_contact_markers(world, contact_points_world, selected_contact_index)

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    metadata = MujocoSceneMetadata(
        output_path=output_path,
        table_center_m=table_center,
        table_quat_wxyz=table_quat,
        table_half_extents_m=table_half_extents,
        plane_centroid_world_m=centroid_world,
        plane_normal_world=normal_world,
        fk_source=robot_geometry.source,
    )
    return xml + "\n", metadata


def build_mujoco_control_scene_xml(
    contact_data: ContactData,
    plane_estimate: PlaneEstimate,
    output_path: Path,
    selected_contact_index: int = 0,
    table_margin_m: float = 0.18,
    table_thickness_m: float = 0.05,
    min_table_half_extent_m: float = 0.35,
    dynamic_tray: bool = False,
    actuated_hande: bool = False,
    tray_collision_mode: str = "proxy",
) -> tuple[str, MujocoSceneMetadata]:
    """Build an actuated UR5e MJCF XML string and return export metadata."""

    if not _UR5E_MJCF_PATH.exists():
        raise FileNotFoundError(f"UR5e MJCF asset not found: {_UR5E_MJCF_PATH}")
    if selected_contact_index < 0 or selected_contact_index >= len(contact_data.contacts):
        raise ValueError("selected_contact_index is out of range.")
    if table_thickness_m <= 0.0:
        raise ValueError("table_thickness_m must be positive.")
    if table_margin_m < 0.0:
        raise ValueError("table_margin_m must be non-negative.")
    if min_table_half_extent_m <= 0.0:
        raise ValueError("min_table_half_extent_m must be positive.")
    _validate_tray_collision_mode(tray_collision_mode)

    display_transform = _ceiling_mount_display_transform()
    contact_points_world = _transform_points(display_transform, contact_data.contact_points())
    centroid_world = _transform_point(display_transform, plane_estimate.centroid)
    fitted_normal_world = display_transform[:3, :3] @ plane_estimate.normal
    fitted_normal_world = _unit(fitted_normal_world)
    if fitted_normal_world[2] < 0.0:
        fitted_normal_world = -fitted_normal_world

    normal_world = np.array([0.0, 0.0, 1.0])
    table_frame = _horizontal_table_frame(centroid_world)
    table_half_extents = _table_half_extents(
        contact_points_world,
        table_frame,
        table_margin_m=table_margin_m,
        table_thickness_m=table_thickness_m,
        min_half_extent_m=min_table_half_extent_m,
    )
    table_center = centroid_world - normal_world * table_half_extents[2]
    table_quat = _quat_wxyz(table_frame[:3, :3])

    selected = contact_data.contacts[selected_contact_index]
    qpos = _ur5e_control_qpos_for_contact(selected.q_rad, contact_data.robot.joint_order)

    root = ET.parse(_UR5E_MJCF_PATH).getroot()
    root.set("model", "ceiling_ur5e_hande_table_control")
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    _rewrite_ur5e_mesh_paths(root, output_path.parent)
    compiler.set("angle", "radian")
    compiler.set("coordinate", "local")
    compiler.set("autolimits", "true")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")
    option.set("impratio", "20")

    _replace_root_child(root, _visual_element(), after_tag="option")
    _replace_root_child(
        root,
        ET.Element(
            "statistic",
            {
                "center": _fmt_vec(_scene_center(contact_points_world)),
                "extent": "1.8",
            },
        ),
        after_tag="visual",
    )

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        _insert_after(root, asset, "default")
    _add_control_scene_assets(asset, output_path.parent)

    world = root.find("worldbody")
    if world is None:
        raise ValueError("UR5e MJCF asset does not contain a worldbody.")
    _wrap_ur5e_body_for_display(world, display_transform)
    _attach_hande_to_wrist(root, actuated=actuated_hande)
    if actuated_hande or dynamic_tray:
        _stiffen_ur5e_position_actuators(root)
    if actuated_hande:
        _add_hande_gripper_actuators(root)
    tray_position, tray_quat = _ti_tray_body_pose(
        table_frame,
        target_center_world=contact_points_world[selected_contact_index],
    )
    if actuated_hande or dynamic_tray:
        _set_grasp_keyframes(root, qpos, selected_contact_index, tray_position, tray_quat)
    else:
        _set_ur5e_contact_keyframe(root, qpos, selected_contact_index)
    _add_control_scene_world_geoms(
        world,
        contact_points_world=contact_points_world,
        selected_contact_index=selected_contact_index,
        centroid_world=centroid_world,
        fitted_normal_world=fitted_normal_world,
        normal_world=normal_world,
        table_frame=table_frame,
        table_center=table_center,
        table_quat=table_quat,
        table_half_extents=table_half_extents,
        plane_estimate=plane_estimate,
        dynamic_tray=dynamic_tray,
        tray_collision_mode=tray_collision_mode,
    )

    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode")
    metadata = MujocoSceneMetadata(
        output_path=output_path,
        table_center_m=table_center,
        table_quat_wxyz=table_quat,
        table_half_extents_m=table_half_extents,
        plane_centroid_world_m=centroid_world,
        plane_normal_world=normal_world,
        fk_source="mjcf_actuated_grasp" if actuated_hande else "mjcf_actuated_position",
    )
    return xml + "\n", metadata


def compile_mujoco_xml(xml: str) -> None:
    """Compile MJCF XML if the optional mujoco package is installed."""

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional 'mujoco' package is not installed; rerun without "
            "--compile-check or install mujoco."
        ) from exc
    mujoco.MjModel.from_xml_string(xml)


def compile_mujoco_xml_path(path: str | Path) -> None:
    """Compile an MJCF file if the optional mujoco package is installed."""

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The optional 'mujoco' package is not installed; rerun without "
            "--compile-check or install mujoco."
        ) from exc
    mujoco.MjModel.from_xml_path(str(path))


def _visual_element() -> ET.Element:
    visual = ET.Element("visual")
    ET.SubElement(
        visual,
        "headlight",
        {
            "ambient": "0.55 0.55 0.55",
            "diffuse": "0.45 0.45 0.45",
            "specular": "0.12 0.12 0.12",
        },
    )
    ET.SubElement(visual, "rgba", {"haze": "1 1 1 1"})
    return visual


def _replace_root_child(root: ET.Element, child: ET.Element, after_tag: str) -> None:
    for existing in list(root):
        if existing.tag == child.tag:
            root.remove(existing)
    _insert_after(root, child, after_tag)


def _insert_after(root: ET.Element, child: ET.Element, after_tag: str) -> None:
    insert_index = None
    for index, existing in enumerate(list(root)):
        if existing.tag == after_tag:
            insert_index = index + 1
    if insert_index is None:
        root.append(child)
        return
    root.insert(insert_index, child)


def _rewrite_ur5e_mesh_paths(root: ET.Element, output_dir: Path) -> None:
    compiler = root.find("compiler")
    mesh_dir_raw = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_dir = _UR5E_MJCF_PATH.parent / mesh_dir_raw
    asset = root.find("asset")
    if asset is None:
        return
    for mesh in asset.findall("mesh"):
        file_raw = mesh.get("file")
        if not file_raw:
            continue
        file_path = Path(file_raw)
        source_path = file_path if file_path.is_absolute() else mesh_dir / file_path
        mesh.set("file", os.path.relpath(source_path, output_dir).replace(os.sep, "/"))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)


def _add_control_scene_assets(asset: ET.Element, output_dir: Path) -> None:
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "white_skybox",
            "type": "skybox",
            "builtin": "flat",
            "rgb1": "1 1 1",
            "rgb2": "1 1 1",
            "width": "32",
            "height": "32",
        },
    )
    _add_material(asset, "ceiling_mat", (0.72, 0.75, 0.78, 1.0))
    _add_material(asset, "table_mat", _TABLE_ALUMINUM_RGBA)
    _add_material(asset, "joint_mat", (0.08, 0.10, 0.12, 1.0))
    _add_material(asset, "hande_body_mat", (0.09, 0.10, 0.11, 1.0))
    _add_material(asset, "hande_finger_mat", (0.18, 0.18, 0.17, 1.0))
    _add_material(asset, "hande_metal_mat", (0.58, 0.60, 0.62, 1.0))
    _add_material(asset, "contact_mat", (1.0, 0.66, 0.12, 1.0))
    _add_material(asset, "tray_mat", _TRAY_GREEN_RGBA)
    for mesh in _ti_tray_mesh_assets():
        _add_mesh(asset, mesh, output_dir)
    for mesh in _hande_mesh_assets():
        _add_mesh(asset, mesh, output_dir)


def _add_control_scene_world_geoms(
    world: ET.Element,
    contact_points_world: np.ndarray,
    selected_contact_index: int,
    centroid_world: np.ndarray,
    fitted_normal_world: np.ndarray,
    normal_world: np.ndarray,
    table_frame: np.ndarray,
    table_center: np.ndarray,
    table_quat: np.ndarray,
    table_half_extents: np.ndarray,
    plane_estimate: PlaneEstimate,
    dynamic_tray: bool = False,
    tray_collision_mode: str = "proxy",
) -> None:
    ET.SubElement(world, "light", {"name": "key", "pos": "0 -1 1.2", "dir": "0 1 -1"})
    ET.SubElement(
        world,
        "camera",
        {
            "name": "overview",
            "pos": "1.2 -1.5 0.6",
            "xyaxes": "0.78 0.62 0 -0.24 0.30 0.92",
        },
    )
    world.append(
        ET.Comment(
            "Plane centroid and normals are expressed in MuJoCo world coordinates. "
            "The table top uses the fitted centroid but is flattened to the robot-base plane."
        )
    )
    world.append(
        ET.Comment(
            f"centroid={_fmt_vec(centroid_world)} fitted_normal={_fmt_vec(fitted_normal_world)} "
            f"table_normal={_fmt_vec(normal_world)} "
            f"rms_error_m={plane_estimate.rms_error:.9f}"
        )
    )
    _add_box_geom(
        world,
        name="ceiling_panel",
        pos=(0.0, 0.0, 0.025),
        quat=(1.0, 0.0, 0.0, 0.0),
        size=(0.75, 0.75, 0.025),
        material="ceiling_mat",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        world,
        "geom",
        {
            "name": "ceiling_mount",
            "type": "cylinder",
            "pos": "0 0 -0.035",
            "size": "0.085 0.035",
            "material": "joint_mat",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _add_box_geom(
        world,
        name="estimated_table",
        pos=table_center,
        quat=table_quat,
        size=table_half_extents,
        material="table_mat",
        rgba=_TABLE_ALUMINUM_RGBA,
    )
    tray_meshes = _ti_tray_mesh_assets()
    if tray_meshes:
        if dynamic_tray:
            _add_dynamic_ti_tray_body(
                world,
                table_frame,
                target_center_world=contact_points_world[selected_contact_index],
                mesh_assets=tray_meshes,
                collision_mode=tray_collision_mode,
            )
        else:
            _add_ti_tray_geom(
                world,
                table_frame,
                target_center_world=contact_points_world[selected_contact_index],
                mesh_assets=tray_meshes,
            )
    _add_contact_markers(world, contact_points_world, selected_contact_index)


def _wrap_ur5e_body_for_display(world: ET.Element, display_transform: np.ndarray) -> None:
    base = None
    for child in list(world):
        if child.tag == "body" and child.get("name") == "base":
            base = child
            break
    if base is None:
        raise ValueError("UR5e MJCF asset does not contain a top-level 'base' body.")

    base_rotation = _matrix_from_quat_wxyz(_parse_vec(base.get("quat", "1 0 0 0"), 4))
    base_pos = _parse_vec(base.get("pos", "0 0 0"), 3)
    root_rotation = display_transform[:3, :3] @ base_rotation.T
    root_translation = -root_rotation @ base_pos

    world.remove(base)
    wrapper = ET.Element(
        "body",
        {
            "name": "ur5e_root",
            "pos": _fmt_vec(root_translation),
            "quat": _fmt_vec(_quat_wxyz(root_rotation)),
        },
    )
    wrapper.append(base)
    insert_index = len(world)
    for index, child in enumerate(list(world)):
        if child.tag not in {"light", "camera"}:
            insert_index = index
            break
    world.insert(insert_index, wrapper)


def _attach_hande_to_wrist(root: ET.Element, actuated: bool = False) -> None:
    if not _hande_mesh_assets():
        return
    wrist = _find_body(root, "wrist_3_link")
    if wrist is None:
        raise ValueError("UR5e MJCF asset does not contain a 'wrist_3_link' body.")
    if _find_body(wrist, "hande_mount") is not None:
        return

    mount = ET.SubElement(
        wrist,
        "body",
        {
            "name": "hande_mount",
            "pos": "0 0.1 0",
            "quat": "-1 1 0 0",
        },
    )
    ET.SubElement(
        mount,
        "site",
        {
            "name": "grasp_center_site",
            "pos": "0 0 0.154",
            "size": "0.006",
            "rgba": "0.1 0.7 1 0.8",
        },
    )

    def add_mesh(
        parent: ET.Element,
        name: str,
        mesh_name: str,
        local_pos: tuple[float, float, float],
        local_rotation: np.ndarray,
        material: str,
    ) -> None:
        ET.SubElement(
            parent,
            "geom",
            {
                "name": name,
                "type": "mesh",
                "mesh": mesh_name,
                "pos": _fmt_vec(local_pos),
                "quat": _fmt_vec(_quat_wxyz(local_rotation)),
                "material": material,
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            },
        )

    identity = np.eye(3)
    flip_about_z = Rotation.from_euler("z", np.pi).as_matrix()
    finger_z = _HANDE_COUPLER_HEIGHT_M + _HANDE_BODY_HEIGHT_M
    add_mesh(
        mount,
        "hande_io_coupler",
        "hande_io_coupler",
        (0.0, 0.0, 0.0),
        identity,
        "hande_metal_mat",
    )
    add_mesh(
        mount,
        "hande_body",
        "hande_body",
        (0.0, 0.0, _HANDE_COUPLER_HEIGHT_M),
        identity,
        "hande_body_mat",
    )
    if actuated:
        _add_actuated_hande_fingers(mount, finger_z)
        return
    add_mesh(
        mount,
        "hande_left_finger",
        "hande_finger",
        (_HANDE_FINGER_OPENING_M, 0.0, finger_z),
        identity,
        "hande_finger_mat",
    )
    add_mesh(
        mount,
        "hande_right_finger",
        "hande_finger",
        (-_HANDE_FINGER_OPENING_M, 0.0, finger_z),
        flip_about_z,
        "hande_finger_mat",
    )


def _add_actuated_hande_fingers(mount: ET.Element, finger_z: float) -> None:
    left = ET.SubElement(
        mount,
        "body",
        {
            "name": "hande_left_finger_body",
            "pos": _fmt_vec((_HANDE_GRASP_FINGER_OFFSET_M, 0.0, finger_z)),
        },
    )
    ET.SubElement(
        left,
        "joint",
        {
            "name": "hande_left_finger_joint",
            "type": "slide",
            "axis": "-1 0 0",
            "range": _fmt_vec((0.0, _HANDE_GRASP_FINGER_RANGE_M)),
            "damping": "8",
            "armature": "0.002",
        },
    )
    ET.SubElement(
        left,
        "geom",
        {
            "name": "hande_left_finger_visual",
            "type": "mesh",
            "mesh": "hande_finger",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "material": "hande_finger_mat",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )
    _add_hande_finger_pad(left, "hande_left_pad_collision", (0.003, 0.0, 0.044))

    right = ET.SubElement(
        mount,
        "body",
        {
            "name": "hande_right_finger_body",
            "pos": _fmt_vec((-_HANDE_GRASP_FINGER_OFFSET_M, 0.0, finger_z)),
        },
    )
    ET.SubElement(
        right,
        "joint",
        {
            "name": "hande_right_finger_joint",
            "type": "slide",
            "axis": "1 0 0",
            "range": _fmt_vec((0.0, _HANDE_GRASP_FINGER_RANGE_M)),
            "damping": "8",
            "armature": "0.002",
        },
    )
    ET.SubElement(
        right,
        "geom",
        {
            "name": "hande_right_finger_visual",
            "type": "mesh",
            "mesh": "hande_finger",
            "pos": "0 0 0",
            "quat": "0 0 0 1",
            "material": "hande_finger_mat",
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        },
    )
    _add_hande_finger_pad(right, "hande_right_pad_collision", (-0.003, 0.0, 0.044))


def _add_hande_finger_pad(
    parent: ET.Element,
    name: str,
    local_pos: tuple[float, float, float],
) -> None:
    ET.SubElement(
        parent,
        "geom",
        {
            "name": name,
            "type": "box",
            "pos": _fmt_vec(local_pos),
            "size": "0.0035 0.009 0.0035",
            "material": "hande_finger_mat",
            "mass": "0.015",
            "friction": "5.0 0.10 0.01",
            "condim": "4",
            "priority": "2",
            "contype": "2",
            "conaffinity": "2",
            "solref": "0.004 1",
            "solimp": "0.95 0.99 0.001",
        },
    )


def _add_hande_gripper_actuators(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    for name, joint in (
        ("hande_left_finger", "hande_left_finger_joint"),
        ("hande_right_finger", "hande_right_finger_joint"),
    ):
        if root.find(f".//general[@name='{name}']") is not None:
            continue
        ET.SubElement(
            actuator,
            "general",
            {
                "name": name,
                "joint": joint,
                "gaintype": "fixed",
                "biastype": "affine",
                "ctrlrange": _fmt_vec((0.0, _HANDE_GRASP_FINGER_RANGE_M)),
                "gainprm": "3000",
                "biasprm": "0 -3000 -80",
                "forcerange": "-300 300",
            },
        )


def _stiffen_ur5e_position_actuators(root: ET.Element) -> None:
    for name in _UR5E_ACTUATOR_NAMES:
        actuator = root.find(f"./actuator/general[@name='{name}']")
        if actuator is None:
            continue
        actuator.set("gainprm", "10000")
        actuator.set("biasprm", "0 -10000 -200")
        actuator.set("forcerange", "-600 600")


def _set_ur5e_contact_keyframe(
    root: ET.Element,
    qpos: np.ndarray,
    selected_contact_index: int,
) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    key_name = f"contact_{selected_contact_index + 1}"
    for key in list(keyframe):
        if key.tag == "key" and key.get("name") == key_name:
            keyframe.remove(key)
    keyframe.insert(
        0,
        ET.Element(
            "key",
            {
                "name": key_name,
                "qpos": _fmt_vec(qpos),
                "ctrl": _fmt_vec(qpos),
            },
        ),
    )


def _set_grasp_keyframes(
    root: ET.Element,
    qpos: np.ndarray,
    selected_contact_index: int,
    tray_position: np.ndarray,
    tray_quat: np.ndarray,
) -> None:
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.SubElement(root, "keyframe")
    for key in list(keyframe):
        keyframe.remove(key)

    open_qpos = np.concatenate(
        (
            np.asarray(qpos, dtype=float),
            np.array([0.0, 0.0], dtype=float),
            np.asarray(tray_position, dtype=float),
            np.asarray(tray_quat, dtype=float),
        )
    )
    open_ctrl = np.concatenate((np.asarray(qpos, dtype=float), np.array([0.0, 0.0])))
    closed_ctrl = np.concatenate(
        (
            np.asarray(qpos, dtype=float),
            np.array([_HANDE_GRASP_CLOSE_M, _HANDE_GRASP_CLOSE_M], dtype=float),
        )
    )
    for name, qpos_values, ctrl_values in (
        (f"contact_{selected_contact_index + 1}", open_qpos, open_ctrl),
        ("grasp_open", open_qpos, open_ctrl),
        ("grasp_closed", open_qpos, closed_ctrl),
    ):
        ET.SubElement(
            keyframe,
            "key",
            {
                "name": name,
                "qpos": _fmt_vec(qpos_values),
                "ctrl": _fmt_vec(ctrl_values),
            },
        )


def _ur5e_control_qpos_for_contact(q_rad: np.ndarray, joint_order: list[str]) -> np.ndarray:
    q = np.asarray(q_rad, dtype=float)
    by_name = {name: float(value) for name, value in zip(joint_order, q, strict=True)}
    qpos = np.zeros(len(_UR5E_MJCF_JOINT_ORDER), dtype=float)
    for index, joint_name in enumerate(_UR5E_MJCF_JOINT_ORDER):
        if joint_name not in by_name:
            raise ValueError(f"No contact joint value for MJCF joint '{joint_name}'.")
        value = by_name[joint_name]
        if joint_name == "shoulder_pan_joint":
            value -= np.pi
        value = _fit_revolute_value_to_joint_limit(joint_name, value)
        qpos[index] = value
    return qpos


def _fit_revolute_value_to_joint_limit(joint_name: str, value: float) -> float:
    low, high = _UR5E_MJCF_JOINT_LIMITS_RAD[joint_name]
    fitted = float(value)
    period = 2.0 * np.pi
    while fitted < low - _JOINT_LIMIT_TOL_RAD:
        fitted += period
    while fitted > high + _JOINT_LIMIT_TOL_RAD:
        fitted -= period
    if fitted < low - _JOINT_LIMIT_TOL_RAD or fitted > high + _JOINT_LIMIT_TOL_RAD:
        raise ValueError(
            f"Contact target for {joint_name} is outside the Menagerie UR5e joint range: "
            f"{value:.6f} rad not in [{low:.6f}, {high:.6f}] rad."
        )
    return float(np.clip(fitted, low, high))


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.findall(".//body"):
        if body.get("name") == name:
            return body
    return None


def _robot_geometry_world(
    q_rad: np.ndarray,
    joint_order: list[str],
    display_transform: np.ndarray,
) -> _RobotGeometry:
    try:
        return _ur5e_mjcf_mesh_geometry_world(q_rad, joint_order, display_transform)
    except Exception:
        points, tool0_transform, source = _robot_points_world(
            q_rad,
            joint_order,
            display_transform,
        )
        return _RobotGeometry(
            source=source,
            tool0_transform_world=tool0_transform,
            capsule_points=points,
        )


def _ur5e_mjcf_mesh_geometry_world(
    q_rad: np.ndarray,
    joint_order: list[str],
    display_transform: np.ndarray,
) -> _RobotGeometry:
    if not _UR5E_MJCF_PATH.exists():
        raise FileNotFoundError(f"UR5e MJCF asset not found: {_UR5E_MJCF_PATH}")

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(_UR5E_MJCF_PATH))
    data = mujoco.MjData(model)
    data.qpos[:] = _mjcf_qpos_for_contact(model, q_rad, joint_order)
    mujoco.mj_forward(model, data)

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id < 0:
        raise ValueError("UR5e MJCF asset does not contain a 'base' body.")
    base_rotation = Rotation.from_quat(data.xquat[base_id][[1, 2, 3, 0]]).as_matrix()
    root_rotation = display_transform[:3, :3] @ base_rotation.T
    root_translation = -root_rotation @ data.xpos[base_id]

    mesh_geoms: list[_StaticMeshGeom] = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        if mesh_name is None:
            continue
        material_name = None
        material_id = int(model.geom_matid[geom_id])
        if material_id >= 0:
            material_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_MATERIAL,
                material_id,
            )
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_name is None:
            geom_name = f"ur5e_mesh_{geom_id:02d}_{mesh_name}"
        body_id = int(model.geom_bodyid[geom_id])
        body_rotation = data.xmat[body_id].reshape(3, 3)
        mesh_geoms.append(
            _StaticMeshGeom(
                name=geom_name,
                mesh_name=mesh_name,
                material_name=material_name,
                pos=root_rotation @ data.xpos[body_id] + root_translation,
                quat_wxyz=_quat_wxyz(root_rotation @ body_rotation),
            )
        )

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise ValueError("UR5e MJCF asset does not contain an 'attachment_site'.")
    tool0_transform = np.eye(4)
    tool0_transform[:3, :3] = root_rotation @ data.site_xmat[site_id].reshape(3, 3)
    tool0_transform[:3, 3] = root_rotation @ data.site_xpos[site_id] + root_translation

    return _RobotGeometry(
        source="mjcf_mesh",
        tool0_transform_world=tool0_transform,
        mesh_geoms=tuple(mesh_geoms),
        mesh_assets=tuple(_ur5e_mesh_assets()),
        material_assets=tuple(_ur5e_material_assets()),
    )


def _mjcf_qpos_for_contact(
    model: object,
    q_rad: np.ndarray,
    joint_order: list[str],
) -> np.ndarray:
    import mujoco

    q = np.asarray(q_rad, dtype=float)
    by_name = {name: float(value) for name, value in zip(joint_order, q, strict=True)}
    qpos = np.zeros(model.nq, dtype=float)
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name is None:
            continue
        if joint_name not in by_name:
            raise ValueError(f"No contact joint value for MJCF joint '{joint_name}'.")
        value = by_name[joint_name]
        if joint_name == "shoulder_pan_joint":
            # The bundled MuJoCo UR5e base convention is rotated by pi around the
            # shoulder-pan axis relative to the captured UR controller poses.
            value -= np.pi
        qpos[int(model.jnt_qposadr[joint_id])] = value
    return qpos


def _ur5e_mesh_assets() -> list[_MeshAsset]:
    root = ET.parse(_UR5E_MJCF_PATH).getroot()
    compiler = root.find("compiler")
    mesh_dir = compiler.get("meshdir", "") if compiler is not None else ""
    assets: list[_MeshAsset] = []
    for mesh in root.findall("./asset/mesh"):
        file_name = mesh.get("file")
        if not file_name:
            continue
        name = mesh.get("name", Path(file_name).stem)
        assets.append(
            _MeshAsset(
                name=name,
                path=_UR5E_MJCF_PATH.parent / mesh_dir / file_name,
            )
        )
    return assets


def _ur5e_material_assets() -> list[_MaterialAsset]:
    root = ET.parse(_UR5E_MJCF_PATH).getroot()
    materials: list[_MaterialAsset] = []
    for material in root.findall("./asset/material"):
        name = material.get("name")
        rgba_raw = material.get("rgba")
        if name is None or rgba_raw is None:
            continue
        rgba = tuple(float(value) for value in rgba_raw.split())
        if len(rgba) == 3:
            rgba = (*rgba, 1.0)
        if len(rgba) != 4:
            continue
        materials.append(_MaterialAsset(name=name, rgba=rgba))
    return materials


def _hande_mesh_assets() -> list[_MeshAsset]:
    if not all(path.exists() for path in _HANDE_MESH_FILES.values()):
        return []
    return [
        _MeshAsset(name=name, path=path)
        for name, path in _HANDE_MESH_FILES.items()
    ]


def _ti_tray_mesh_assets() -> list[_MeshAsset]:
    part_paths = sorted(_TI_TRAY_PARTS_DIR.glob("ti_tray_part_*.obj"))
    if part_paths:
        return [
            _MeshAsset(name=f"ti_tray_part_{index:02d}", path=path, inertia="shell")
            for index, path in enumerate(part_paths)
        ]
    if not _TI_TRAY_MESH_PATH.exists():
        return []
    return [_MeshAsset(name="ti_tray", path=_TI_TRAY_MESH_PATH, inertia="shell")]


def _robot_points_world(
    q_rad: np.ndarray,
    joint_order: list[str],
    display_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        visualizer = URDFRobotVisualizer()
        urdf = visualizer._load_description()
        cfg = visualizer._configuration_for_urdf_object(urdf, q_rad, joint_order)
        urdf.update_cfg(cfg)
        points = []
        for link_name in _URDF_LINK_PATH:
            transform = display_transform @ urdf.get_transform(link_name, "base")
            points.append(transform[:3, 3])
        tool0_transform = display_transform @ urdf.get_transform("tool0", "base")
        return (
            _deduplicate_adjacent_points(np.asarray(points, dtype=float)),
            tool0_transform,
            "urdf",
        )
    except Exception:
        fallback = SimpleUR5eVisualizer(base_transform=display_transform)
        transforms = fallback.forward_kinematics(q_rad)
        points = np.vstack([transform[:3, 3] for transform in transforms])
        return _deduplicate_adjacent_points(points), transforms[-1], "simple_dh"


def _deduplicate_adjacent_points(points: np.ndarray, min_distance_m: float = 1e-4) -> np.ndarray:
    kept = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point - kept[-1])) >= min_distance_m:
            kept.append(point)
    return np.asarray(kept, dtype=float)


def _add_static_mesh_geoms(world: ET.Element, mesh_geoms: tuple[_StaticMeshGeom, ...]) -> None:
    for geom in mesh_geoms:
        attrs = {
            "name": geom.name,
            "type": "mesh",
            "mesh": geom.mesh_name,
            "pos": _fmt_vec(geom.pos),
            "quat": _fmt_vec(geom.quat_wxyz),
            "contype": "0",
            "conaffinity": "0",
            "group": "2",
        }
        if geom.material_name is not None:
            attrs["material"] = geom.material_name
        ET.SubElement(world, "geom", attrs)


def _add_fixed_robot_geoms(world: ET.Element, points: np.ndarray) -> None:
    for index, (start, end) in enumerate(zip(points[:-1], points[1:], strict=True)):
        length = float(np.linalg.norm(end - start))
        if length < 1e-4:
            continue
        radius = _LINK_RADII_M[min(index, len(_LINK_RADII_M) - 1)]
        ET.SubElement(
            world,
            "geom",
            {
                "name": f"ur5e_link_{index}",
                "type": "capsule",
                "fromto": _fmt_vec([*start, *end]),
                "size": f"{radius:.6f}",
                "material": "robot_mat",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    for index, point in enumerate(points):
        ET.SubElement(
            world,
            "geom",
            {
                "name": f"ur5e_joint_{index}",
                "type": "sphere",
                "pos": _fmt_vec(point),
                "size": "0.045",
                "material": "joint_mat",
                "contype": "0",
                "conaffinity": "0",
            },
        )


def _add_hande_geoms(
    world: ET.Element,
    tool0_transform: np.ndarray,
    tcp_transform: np.ndarray,
) -> None:
    if _hande_mesh_assets():
        _add_hande_mesh_geoms(world, tool0_transform, tcp_transform)
        return
    _add_hande_sketch_geoms(world, tool0_transform, tcp_transform)


def _add_hande_mesh_geoms(
    world: ET.Element,
    tool0_transform: np.ndarray,
    tcp_transform: np.ndarray,
) -> None:
    hande_transform = _hande_mount_transform(tool0_transform, tcp_transform)
    rotation = hande_transform[:3, :3]
    origin = hande_transform[:3, 3]

    def add_mesh(
        name: str,
        mesh_name: str,
        local_pos: tuple[float, float, float],
        local_rotation: np.ndarray,
        material: str,
    ) -> None:
        ET.SubElement(
            world,
            "geom",
            {
                "name": name,
                "type": "mesh",
                "mesh": mesh_name,
                "pos": _fmt_vec(origin + rotation @ np.asarray(local_pos, dtype=float)),
                "quat": _fmt_vec(_quat_wxyz(rotation @ local_rotation)),
                "material": material,
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            },
        )

    identity = np.eye(3)
    flip_about_z = Rotation.from_euler("z", np.pi).as_matrix()
    add_mesh(
        "hande_io_coupler",
        "hande_io_coupler",
        (0.0, 0.0, 0.0),
        identity,
        "hande_metal_mat",
    )
    add_mesh(
        "hande_body",
        "hande_body",
        (0.0, 0.0, _HANDE_COUPLER_HEIGHT_M),
        identity,
        "hande_body_mat",
    )
    finger_z = _HANDE_COUPLER_HEIGHT_M + _HANDE_BODY_HEIGHT_M
    add_mesh(
        "hande_left_finger",
        "hande_finger",
        (_HANDE_FINGER_OPENING_M, 0.0, finger_z),
        identity,
        "hande_finger_mat",
    )
    add_mesh(
        "hande_right_finger",
        "hande_finger",
        (-_HANDE_FINGER_OPENING_M, 0.0, finger_z),
        flip_about_z,
        "hande_finger_mat",
    )


def _hande_mount_transform(tool0_transform: np.ndarray, tcp_transform: np.ndarray) -> np.ndarray:
    transform = np.array(tool0_transform, dtype=float, copy=True)
    direction = tcp_transform[:3, 3] - tool0_transform[:3, 3]
    distance = float(np.linalg.norm(direction))
    if distance < 1e-4:
        return transform

    z_axis = direction / distance
    x_axis = tool0_transform[:3, 0]
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    if float(np.linalg.norm(x_axis)) < 1e-4:
        x_axis = tcp_transform[:3, 0] - np.dot(tcp_transform[:3, 0], z_axis) * z_axis
    x_axis = _unit(x_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))
    x_axis = _unit(np.cross(y_axis, z_axis))
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    return transform


def _add_hande_sketch_geoms(
    world: ET.Element,
    tool0_transform: np.ndarray,
    tcp_transform: np.ndarray,
) -> None:
    tool0 = tool0_transform[:3, 3]
    tcp = tcp_transform[:3, 3]
    extension = tcp - tool0
    if float(np.linalg.norm(extension)) > 1e-4:
        ET.SubElement(
            world,
            "geom",
            {
                "name": "hande_tool0_to_tcp",
                "type": "capsule",
                "fromto": _fmt_vec([*tool0, *tcp]),
                "size": "0.025",
                "material": "hande_mat",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    rotation = tcp_transform[:3, :3]
    palm_center = tcp - rotation @ np.array([0.0, 0.0, 0.08])
    _add_box_geom(
        world,
        name="hande_palm",
        pos=palm_center,
        quat=_quat_wxyz(rotation),
        size=(0.06, 0.035, 0.025),
        material="hande_mat",
        contype="0",
        conaffinity="0",
    )

    for side, x_offset in (("left", -0.032), ("right", 0.032)):
        finger_center = tcp + rotation @ np.array([x_offset, 0.0, -0.035])
        _add_box_geom(
            world,
            name=f"hande_{side}_finger",
            pos=finger_center,
            quat=_quat_wxyz(rotation),
            size=(0.012, 0.014, 0.055),
            material="hande_mat",
            contype="0",
            conaffinity="0",
        )


def _add_contact_markers(
    world: ET.Element,
    points: np.ndarray,
    selected_contact_index: int,
) -> None:
    for index, point in enumerate(points):
        ET.SubElement(
            world,
            "geom",
            {
                "name": f"contact_marker_{index + 1}",
                "type": "sphere",
                "pos": _fmt_vec(point),
                "size": "0.018" if index == selected_contact_index else "0.012",
                "material": "contact_mat",
                "contype": "0",
                "conaffinity": "0",
            },
        )


def _add_ti_tray_geom(
    world: ET.Element,
    table_frame: np.ndarray,
    target_center_world: np.ndarray,
    mesh_assets: list[_MeshAsset],
) -> None:
    position, quat = _ti_tray_body_pose(table_frame, target_center_world)
    for index, mesh in enumerate(mesh_assets):
        ET.SubElement(
            world,
            "geom",
            {
                "name": f"ti_tray_{index:02d}",
                "type": "mesh",
                "mesh": mesh.name,
                "pos": _fmt_vec(position),
                "quat": _fmt_vec(quat),
                "material": "tray_mat",
                "rgba": _fmt_vec(_TRAY_GREEN_RGBA),
                "contype": "0",
                "conaffinity": "0",
            },
        )


def _add_dynamic_ti_tray_body(
    world: ET.Element,
    table_frame: np.ndarray,
    target_center_world: np.ndarray,
    mesh_assets: list[_MeshAsset],
    collision_mode: str = "proxy",
) -> None:
    _validate_tray_collision_mode(collision_mode)
    position, quat = _ti_tray_body_pose(table_frame, target_center_world)
    body = ET.SubElement(
        world,
        "body",
        {
            "name": "ti_tray_body",
            "pos": _fmt_vec(position),
            "quat": _fmt_vec(quat),
        },
    )
    ET.SubElement(body, "freejoint", {"name": "ti_tray_freejoint"})
    ET.SubElement(
        body,
        "site",
        {
            "name": "ti_tray_protrusion_site",
            "pos": _fmt_vec(_TI_TRAY_PROTRUSION_CENTER),
            "size": "0.006",
            "rgba": "1 0.3 0.05 0.8",
        },
    )
    mesh_collision = collision_mode in {"mesh", "both"}
    collision_meshes = [
        mesh for mesh in mesh_assets if _mesh_can_form_collision_hull(mesh.path)
    ]
    mesh_mass = _TI_TRAY_TOTAL_MASS_KG / max(len(collision_meshes), 1)
    for index, mesh in enumerate(mesh_assets):
        use_collision = mesh_collision and _mesh_can_form_collision_hull(mesh.path)
        attrs = {
            "name": f"ti_tray_mesh_{index:02d}",
            "type": "mesh",
            "mesh": mesh.name,
            "material": "tray_mat",
            "rgba": _fmt_vec(_TRAY_GREEN_RGBA),
            "group": "2",
        }
        if use_collision:
            attrs.update(
                {
                    "mass": f"{mesh_mass:.9f}",
                    "friction": "5.0 0.10 0.01",
                    "condim": "4",
                    "priority": "2",
                    "contype": "3",
                    "conaffinity": "3",
                    "solref": "0.004 1",
                    "solimp": "0.95 0.99 0.001",
                }
            )
        else:
            attrs.update({"contype": "0", "conaffinity": "0", "density": "0"})
        ET.SubElement(body, "geom", attrs)
    if collision_mode in {"proxy", "both"}:
        _add_ti_tray_proxy_collision(body, massless=collision_mode == "both")


def _add_ti_tray_proxy_collision(body: ET.Element, massless: bool = False) -> None:
    protrusion_attrs = {
        "name": "ti_tray_protrusion_grasp_collision",
        "type": "box",
        "pos": _fmt_vec(_TI_TRAY_PROTRUSION_CENTER),
        "size": _fmt_vec(_TI_TRAY_PROTRUSION_SIZE),
        "rgba": "0 0.58 0.22 0.35",
        "friction": "5.0 0.10 0.01",
        "condim": "4",
        "priority": "2",
        "contype": "2",
        "conaffinity": "2",
        "solref": "0.004 1",
        "solimp": "0.95 0.99 0.001",
    }
    if massless:
        protrusion_attrs["density"] = "0"
    else:
        protrusion_attrs["mass"] = f"{_TI_TRAY_TOTAL_MASS_KG:.9f}"
    ET.SubElement(body, "geom", protrusion_attrs)


def _validate_tray_collision_mode(mode: str) -> None:
    if mode not in {"proxy", "mesh", "both"}:
        raise ValueError("tray_collision_mode must be one of: proxy, mesh, both.")


def _mesh_can_form_collision_hull(path: Path, min_extent_m: float = 1e-5) -> bool:
    bounds = _obj_vertex_bounds(path)
    extents = bounds[1] - bounds[0]
    return int(np.count_nonzero(extents > min_extent_m)) == 3


def _ti_tray_body_pose(
    table_frame: np.ndarray,
    target_center_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = _obj_vertex_bounds(_TI_TRAY_MESH_PATH)
    raw_center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    raw_bottom_z = bounds[0, 2]
    local_anchor = np.array(
        [raw_center_xy[0], raw_center_xy[1], raw_bottom_z - _TI_TRAY_TABLE_CLEARANCE_M],
        dtype=float,
    )
    rotation = table_frame[:3, :3]
    target_local = rotation.T @ (target_center_world - table_frame[:3, 3])
    target_local[2] = 0.0
    position = table_frame[:3, 3] + rotation @ (target_local - local_anchor)
    return position, _quat_wxyz(rotation)


def _table_half_extents(
    contact_points_world: np.ndarray,
    table_frame: np.ndarray,
    table_margin_m: float,
    table_thickness_m: float,
    min_half_extent_m: float,
) -> np.ndarray:
    local = (table_frame[:3, :3].T @ (contact_points_world - table_frame[:3, 3]).T).T
    half_xy = np.max(np.abs(local[:, :2]), axis=0) + table_margin_m
    half_xy = np.maximum(half_xy, min_half_extent_m)
    return np.array([half_xy[0], half_xy[1], table_thickness_m / 2.0], dtype=float)


def _horizontal_table_frame(origin: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = origin
    return transform


def _add_material(asset: ET.Element, name: str, rgba: tuple[float, float, float, float]) -> None:
    ET.SubElement(asset, "material", {"name": name, "rgba": _fmt_vec(rgba)})


def _add_mesh(asset: ET.Element, mesh: _MeshAsset, output_dir: Path) -> None:
    relative_path = os.path.relpath(mesh.path, output_dir).replace(os.sep, "/")
    attrs = {
        "name": mesh.name,
        "file": relative_path,
    }
    if mesh.inertia is not None:
        attrs["inertia"] = mesh.inertia
    ET.SubElement(
        asset,
        "mesh",
        attrs,
    )


def _add_box_geom(
    parent: ET.Element,
    name: str,
    pos: np.ndarray | tuple[float, float, float],
    quat: np.ndarray | tuple[float, float, float, float],
    size: np.ndarray | tuple[float, float, float],
    material: str,
    rgba: np.ndarray | tuple[float, float, float, float] | None = None,
    contype: str | None = None,
    conaffinity: str | None = None,
) -> None:
    attrs = {
        "name": name,
        "type": "box",
        "pos": _fmt_vec(pos),
        "quat": _fmt_vec(quat),
        "size": _fmt_vec(size),
        "material": material,
    }
    if rgba is not None:
        attrs["rgba"] = _fmt_vec(rgba)
    if contype is not None:
        attrs["contype"] = contype
    if conaffinity is not None:
        attrs["conaffinity"] = conaffinity
    ET.SubElement(parent, "geom", attrs)


def _scene_center(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    center[2] *= 0.5
    return center


def _quat_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def _matrix_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=float)
    return Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()


def _fmt_vec(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _parse_vec(raw: str, expected_size: int) -> np.ndarray:
    values = np.asarray([float(value) for value in raw.split()], dtype=float)
    if values.shape != (expected_size,):
        raise ValueError(f"Expected {expected_size} values, got: {raw}")
    return values


def _obj_vertex_bounds(path: Path) -> np.ndarray:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError(f"OBJ mesh has no vertices: {path}")
    array = np.asarray(vertices, dtype=float)
    return np.vstack((array.min(axis=0), array.max(axis=0)))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return vector / norm
