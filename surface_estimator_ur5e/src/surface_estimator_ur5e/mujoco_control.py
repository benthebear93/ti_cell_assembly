"""Small MuJoCo viewer runner for the actuated UR5e scene."""

from __future__ import annotations

from pathlib import Path
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

_UR5E_ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)
_UR5E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_HANDE_ACTUATOR_NAMES = ("hande_left_finger", "hande_right_finger")
_HANDE_JOINT_NAMES = ("hande_left_finger_joint", "hande_right_finger_joint")


def launch_mujoco_control_viewer(
    xml_path: str | Path,
    keyframe: str | None = "contact_1",
    limit_mode: str = "error",
) -> None:
    """Open MuJoCo viewer with qpos and ctrl initialized from a keyframe."""

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise RuntimeError("The optional 'mujoco' package is not installed.") from exc

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _validate_limit_mode(limit_mode)
    if keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"MuJoCo keyframe not found: {keyframe}")
        data.qpos[:] = _apply_joint_limits(model, model.key_qpos[key_id], limit_mode)
        if model.nu:
            data.ctrl[:] = _apply_ctrl_limits(model, model.key_ctrl[key_id], limit_mode)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            if model.nu:
                data.ctrl[:] = _apply_ctrl_limits(model, data.ctrl, "clamp")
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - step_start
            time.sleep(max(0.0, float(model.opt.timestep) - elapsed))


def launch_mujoco_grasp_demo(
    xml_path: str | Path,
    keyframe: str | None = "grasp_open",
    close_m: float = 0.030,
    lift_m: float = 0.08,
    limit_mode: str = "clamp",
    attach_mode: str = "physics",
) -> None:
    """Open a scripted MuJoCo demo that pinches the tray protrusion."""

    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise RuntimeError("The optional 'mujoco' package is not installed.") from exc

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _validate_limit_mode(limit_mode)
    _validate_attach_mode(attach_mode)
    if keyframe:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"MuJoCo keyframe not found: {keyframe}")
        data.qpos[:] = _apply_joint_limits(model, model.key_qpos[key_id], limit_mode)
        data.ctrl[:] = _apply_ctrl_limits(model, model.key_ctrl[key_id], limit_mode)
    mujoco.mj_forward(model, data)

    ur_qpos_adrs = _joint_qpos_addresses(model, _UR5E_JOINT_NAMES)
    hande_qpos_adrs = _joint_qpos_addresses(model, _HANDE_JOINT_NAMES)
    ur_ctrl_ids = _actuator_ids(model, _UR5E_ACTUATOR_NAMES)
    hande_ctrl_ids = _actuator_ids(model, _HANDE_ACTUATOR_NAMES)
    protrusion_site = _site_id(model, "ti_tray_protrusion_site")
    grasp_site = _site_id(model, "grasp_center_site")
    tray_body = _body_id(model, "ti_tray_body")
    tray_freejoint = _joint_id(model, "ti_tray_freejoint")
    grasp_target = data.site_xpos[protrusion_site].copy()
    grasp_rotation = _desired_grasp_rotation(data, tray_body)
    q_grasp = _solve_site_pose_ik(
        model,
        data,
        site_id=grasp_site,
        target_pos=grasp_target,
        target_rotation=grasp_rotation,
        joint_names=_UR5E_JOINT_NAMES,
    )
    q_lift = _solve_site_pose_ik(
        model,
        data,
        site_id=grasp_site,
        target_pos=grasp_target + np.array([0.0, 0.0, lift_m], dtype=float),
        target_rotation=grasp_rotation,
        joint_names=_UR5E_JOINT_NAMES,
        initial_qpos=q_grasp,
    )
    q_start = q_grasp.copy()

    if keyframe:
        data.qpos[:] = _apply_joint_limits(model, model.key_qpos[key_id], limit_mode)
        data.ctrl[:] = _apply_ctrl_limits(model, model.key_ctrl[key_id], limit_mode)
    data.qpos[ur_qpos_adrs] = q_grasp
    data.ctrl[ur_ctrl_ids] = q_grasp
    data.qpos[hande_qpos_adrs] = 0.0
    data.ctrl[hande_ctrl_ids] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    initial_tray_transform = _body_transform(data, tray_body)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _set_grasp_viewer_camera(viewer, data, grasp_site, protrusion_site)
        start_time = time.time()
        attached_relative_transform = None
        while viewer.is_running():
            step_start = time.time()
            elapsed_total = step_start - start_time
            ur_target, gripper_target = _grasp_demo_targets(
                elapsed_total,
                q_start=q_start,
                q_grasp=q_grasp,
                q_lift=q_lift,
                close_m=close_m,
            )
            data.ctrl[ur_ctrl_ids] = ur_target
            data.ctrl[hande_ctrl_ids] = gripper_target
            data.ctrl[:] = _apply_ctrl_limits(model, data.ctrl, "clamp")
            if elapsed_total < 3.5:
                mujoco.mj_forward(model, data)
                _set_freejoint_transform(model, data, tray_freejoint, initial_tray_transform)
                mujoco.mj_forward(model, data)
            elif attach_mode == "kinematic":
                mujoco.mj_forward(model, data)
                if attached_relative_transform is None:
                    attached_relative_transform = (
                        _invert_transform(_site_transform(data, grasp_site))
                        @ _body_transform(data, tray_body)
                    )
                desired_tray_transform = (
                    _site_transform(data, grasp_site) @ attached_relative_transform
                )
                _set_freejoint_transform(model, data, tray_freejoint, desired_tray_transform)
                mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - step_start
            time.sleep(max(0.0, float(model.opt.timestep) - elapsed))


def _grasp_demo_targets(
    elapsed_s: float,
    q_start: np.ndarray,
    q_grasp: np.ndarray,
    q_lift: np.ndarray,
    close_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if elapsed_s < 0.5:
        return q_start, np.array([0.0, 0.0], dtype=float)
    if elapsed_s < 1.0:
        alpha = (elapsed_s - 0.5) / 0.5
        return _lerp(q_start, q_grasp, alpha), np.array([0.0, 0.0], dtype=float)
    if elapsed_s < 2.5:
        alpha = (elapsed_s - 1.0) / 1.5
        close = float(_lerp(np.array([0.0]), np.array([close_m]), alpha)[0])
        return q_grasp, np.array([close, close], dtype=float)
    if elapsed_s < 3.5:
        return q_grasp, np.array([close_m, close_m], dtype=float)
    if elapsed_s < 6.5:
        alpha = (elapsed_s - 3.5) / 3.0
        return _lerp(q_grasp, q_lift, alpha), np.array([close_m, close_m], dtype=float)
    return q_lift, np.array([close_m, close_m], dtype=float)


def _solve_site_position_ik(
    model: object,
    data: object,
    site_id: int,
    target_pos: np.ndarray,
    joint_names: tuple[str, ...],
    initial_qpos: np.ndarray | None = None,
    max_iterations: int = 120,
    tolerance_m: float = 2e-4,
) -> np.ndarray:
    import mujoco

    qpos_adrs = _joint_qpos_addresses(model, joint_names)
    dof_adrs = _joint_dof_addresses(model, joint_names)
    if initial_qpos is not None:
        data.qpos[qpos_adrs] = initial_qpos
    target = np.asarray(target_pos, dtype=float)
    for _ in range(max_iterations):
        mujoco.mj_forward(model, data)
        error = target - data.site_xpos[site_id]
        if float(np.linalg.norm(error)) < tolerance_m:
            break
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = jacp[:, dof_adrs]
        damping = 2e-4
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), error)
        delta = np.clip(delta, -0.05, 0.05)
        candidate = data.qpos.copy()
        candidate[qpos_adrs] += delta
        data.qpos[:] = _apply_joint_limits(model, candidate, "clamp")
    mujoco.mj_forward(model, data)
    return data.qpos[qpos_adrs].copy()


def _solve_site_pose_ik(
    model: object,
    data: object,
    site_id: int,
    target_pos: np.ndarray,
    target_rotation: np.ndarray,
    joint_names: tuple[str, ...],
    initial_qpos: np.ndarray | None = None,
    max_iterations: int = 180,
    position_tolerance_m: float = 2e-4,
    rotation_tolerance_rad: float = 2e-3,
) -> np.ndarray:
    import mujoco

    qpos_adrs = _joint_qpos_addresses(model, joint_names)
    dof_adrs = _joint_dof_addresses(model, joint_names)
    if initial_qpos is not None:
        data.qpos[qpos_adrs] = initial_qpos
    target = np.asarray(target_pos, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)
    rotation_weight = 0.18
    for _ in range(max_iterations):
        mujoco.mj_forward(model, data)
        current_rotation = data.site_xmat[site_id].reshape(3, 3)
        pos_error = target - data.site_xpos[site_id]
        rot_error = Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
        if (
            float(np.linalg.norm(pos_error)) < position_tolerance_m
            and float(np.linalg.norm(rot_error)) < rotation_tolerance_rad
        ):
            break
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack((jacp, rotation_weight * jacr))[:, dof_adrs]
        error = np.concatenate((pos_error, rotation_weight * rot_error))
        damping = 8e-4
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), error)
        delta = np.clip(delta, -0.04, 0.04)
        candidate = data.qpos.copy()
        candidate[qpos_adrs] += delta
        data.qpos[:] = _apply_joint_limits(model, candidate, "clamp")
    mujoco.mj_forward(model, data)
    return data.qpos[qpos_adrs].copy()


def _desired_grasp_rotation(data: object, tray_body: int) -> np.ndarray:
    tray_rotation = data.xmat[tray_body].reshape(3, 3)
    closing_axis = _unit(tray_rotation[:, 0])
    finger_axis = _unit(-tray_rotation[:, 2])
    lateral_axis = _unit(np.cross(finger_axis, closing_axis))
    finger_axis = _unit(np.cross(closing_axis, lateral_axis))
    return np.column_stack((closing_axis, lateral_axis, finger_axis))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return np.asarray(vector, dtype=float) / norm


def _joint_qpos_addresses(model: object, joint_names: tuple[str, ...]) -> np.ndarray:
    addresses: list[int] = []
    for name in joint_names:
        joint_id = _joint_id(model, name)
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=int)


def _joint_dof_addresses(model: object, joint_names: tuple[str, ...]) -> np.ndarray:
    addresses: list[int] = []
    for name in joint_names:
        joint_id = _joint_id(model, name)
        addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(addresses, dtype=int)


def _joint_id(model: object, name: str) -> int:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise ValueError(f"MuJoCo joint not found: {name}")
    return int(joint_id)


def _body_id(model: object, name: str) -> int:
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {name}")
    return int(body_id)


def _actuator_ids(model: object, names: tuple[str, ...]) -> np.ndarray:
    import mujoco

    ids: list[int] = []
    for name in names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"MuJoCo actuator not found: {name}")
        ids.append(actuator_id)
    return np.asarray(ids, dtype=int)


def _site_id(model: object, name: str) -> int:
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {name}")
    return int(site_id)


def _lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    clipped = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - clipped) * start + clipped * end


def _set_grasp_viewer_camera(
    viewer: object,
    data: object,
    grasp_site: int,
    protrusion_site: int,
) -> None:
    lookat = 0.5 * (data.site_xpos[grasp_site] + data.site_xpos[protrusion_site])
    lookat = np.asarray(lookat, dtype=float)
    lookat[2] += 0.025
    viewer.cam.type = 0
    viewer.cam.lookat[:] = lookat
    viewer.cam.distance = 0.34
    viewer.cam.azimuth = 145.0
    viewer.cam.elevation = -18.0


def _body_transform(data: object, body_id: int) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[body_id].reshape(3, 3)
    transform[:3, 3] = data.xpos[body_id]
    return transform


def _site_transform(data: object, site_id: int) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    transform[:3, 3] = data.site_xpos[site_id]
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def _set_freejoint_transform(
    model: object,
    data: object,
    joint_id: int,
    transform: np.ndarray,
) -> None:
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_adr : qpos_adr + 3] = transform[:3, 3]
    quat_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat_xyzw[[3, 0, 1, 2]]
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def _validate_limit_mode(limit_mode: str) -> None:
    if limit_mode not in {"error", "clamp", "ignore"}:
        raise ValueError("limit_mode must be one of: error, clamp, ignore.")


def _validate_attach_mode(attach_mode: str) -> None:
    if attach_mode not in {"kinematic", "physics"}:
        raise ValueError("attach_mode must be one of: kinematic, physics.")


def _apply_joint_limits(model: object, qpos: np.ndarray, limit_mode: str) -> np.ndarray:
    _validate_limit_mode(limit_mode)
    checked = np.asarray(qpos, dtype=float).copy()
    if limit_mode == "ignore":
        return checked

    violations: list[str] = []
    for joint_id in range(model.njnt):
        if not int(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (2, 3):
            continue
        adr = int(model.jnt_qposadr[joint_id])
        low, high = model.jnt_range[joint_id]
        value = float(checked[adr])
        if low - 1e-9 <= value <= high + 1e-9:
            checked[adr] = float(np.clip(value, low, high))
            continue
        name = _name(model, "joint", joint_id)
        violations.append(f"{name}: {value:.6f} not in [{low:.6f}, {high:.6f}]")
        if limit_mode == "clamp":
            checked[adr] = float(np.clip(value, low, high))

    _handle_limit_violations("joint target", violations, limit_mode)
    return checked


def _apply_ctrl_limits(model: object, ctrl: np.ndarray, limit_mode: str) -> np.ndarray:
    _validate_limit_mode(limit_mode)
    checked = np.asarray(ctrl, dtype=float).copy()
    if limit_mode == "ignore":
        return checked

    violations: list[str] = []
    for actuator_id in range(model.nu):
        if not int(model.actuator_ctrllimited[actuator_id]):
            continue
        low, high = model.actuator_ctrlrange[actuator_id]
        value = float(checked[actuator_id])
        if low - 1e-9 <= value <= high + 1e-9:
            checked[actuator_id] = float(np.clip(value, low, high))
            continue
        name = _name(model, "actuator", actuator_id)
        violations.append(f"{name}: {value:.6f} not in [{low:.6f}, {high:.6f}]")
        if limit_mode == "clamp":
            checked[actuator_id] = float(np.clip(value, low, high))

    _handle_limit_violations("actuator control", violations, limit_mode)
    return checked


def _handle_limit_violations(kind: str, violations: list[str], limit_mode: str) -> None:
    if not violations:
        return
    message = f"{kind} exceeds MuJoCo limits: " + "; ".join(violations)
    if limit_mode == "error":
        raise ValueError(message)
    if limit_mode == "clamp":
        warnings.warn(message + "; clamped.", RuntimeWarning, stacklevel=2)


def _name(model: object, obj_type: str, obj_id: int) -> str:
    import mujoco

    if obj_type == "joint":
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, obj_id)
    elif obj_type == "actuator":
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, obj_id)
    else:
        name = None
    return name or f"{obj_type}_{obj_id}"
