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
_TRAY_HOLE_SITE_NAMES = (
    "ti_tray_hole_1_site",
    "ti_tray_hole_2_site",
    "ti_tray_hole_3_site",
    "ti_tray_hole_4_site",
)
_ASSEMBLY_POST_SITE_NAMES = (
    "ti_assembly_post_1_site",
    "ti_assembly_post_2_site",
    "ti_assembly_post_3_site",
    "ti_assembly_post_4_site",
)
# Keep the released pose lightly seated on the proxy post-stop collar. Below
# this the collar is over-preloaded; above it the tray drops and impacts.
_TRAY_RELEASE_NORMAL_OFFSET_M = 0.0066
_TRAY_INSERT_NORMAL_OFFSETS_M = (0.085, 0.055, 0.035, 0.020, 0.012, 0.0066)
_TRAY_ALIGN_DONE_S = 10.0
_TRAY_INSERT_START_S = 10.8
_TRAY_OPEN_START_S = 13.0
_TRAY_OPEN_DONE_S = 14.0
_TRAY_RELEASE_POSE_DONE_S = 14.7
_TRAY_RELEASE_S = 15.0
_TRAY_RELEASE_DAMPING_DONE_S = 15.35
_TRAY_RELEASE_DAMPING_SCALE = 0.2
_PRESSER_HANDLE_PUSH_START_S = 15.45
_PRESSER_HANDLE_PUSH_DONE_S = 20.80
_PRESSER_HANDLE_PUSH_ANGLES_RAD = (0.0, 0.32, 0.68, 0.98)
_PRESSER_HANDLE_PUSH_PRE_OFFSET_M = 0.060
_PRESSER_HANDLE_PUSH_CONTACT_OFFSET_M = 0.012
_PRESSER_HANDLE_PUSH_AXIS_OFFSET_M = -0.002
_PRESSER_HANDLE_PUSH_GRIPPER_M = 0.010
_PRESSER_HANDLE_TRAY_RETREAT_OFFSET_M = (0.0, -0.095, 0.090)
_PRESSER_HANDLE_CLEARANCE_OFFSET_M = (0.100, -0.095, 0.230)
_PRESSER_HANDLE_HIGH_APPROACH_OFFSET_M = (0.0, 0.0, 0.090)
_REALSENSE_CAMERA_NAME = "realsense_rgb"


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
    attach_mode: str = "kinematic",
    viewer_camera: str | None = "free",
    show_realsense_windows: bool = True,
    realsense_camera: str = _REALSENSE_CAMERA_NAME,
    realsense_width: int = 640,
    realsense_height: int = 480,
    realsense_hz: float = 15.0,
    pointcloud_stride: int = 4,
    pointcloud_max_depth_m: float = 2.0,
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
    ur_dof_adrs = _joint_dof_addresses(model, _UR5E_JOINT_NAMES)
    hande_dof_adrs = _joint_dof_addresses(model, _HANDE_JOINT_NAMES)
    ur_ctrl_ids = _actuator_ids(model, _UR5E_ACTUATOR_NAMES)
    hande_ctrl_ids = _actuator_ids(model, _HANDE_ACTUATOR_NAMES)
    protrusion_site = _site_id(model, "ti_tray_protrusion_site")
    grasp_site = _site_id(model, "grasp_center_site")
    assembly_post_sites = tuple(
        site_id
        for site_id in (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in _ASSEMBLY_POST_SITE_NAMES
        )
        if site_id >= 0
    )
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
    q_insert_path, q_release = _solve_tray_place_ik(
        model,
        data,
        grasp_site=grasp_site,
        tray_body=tray_body,
        joint_names=_UR5E_JOINT_NAMES,
        q_grasp=q_grasp,
        q_lift=q_lift,
    )
    q_handle_push_path = _solve_presser_handle_push_ik(
        model,
        data,
        pusher_site=grasp_site,
        joint_names=_UR5E_JOINT_NAMES,
        q_release=q_release,
    )
    q_handle_push_segment_durations = _post_insert_segment_durations(
        model,
        data,
        site_id=grasp_site,
        joint_names=_UR5E_JOINT_NAMES,
        q_insert_path=q_insert_path,
        q_release=q_release,
        q_handle_push_path=q_handle_push_path,
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
    initial_attached_relative_transform = (
        _invert_transform(_site_transform(data, grasp_site)) @ initial_tray_transform
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        realsense_windows = None
        if show_realsense_windows:
            realsense_windows = _RealSenseRenderWindows(
                model,
                camera_name=realsense_camera,
                width=realsense_width,
                height=realsense_height,
                fps=realsense_hz,
                pointcloud_stride=pointcloud_stride,
                pointcloud_max_depth_m=pointcloud_max_depth_m,
            )
        try:
            if not _set_fixed_viewer_camera(model, viewer, viewer_camera):
                _set_grasp_viewer_camera(
                    viewer,
                    data,
                    grasp_site,
                    protrusion_site,
                    assembly_post_sites=assembly_post_sites,
                )
            start_time = time.time()
            attached_relative_transform = initial_attached_relative_transform
            released_tray = False
            while viewer.is_running():
                step_start = time.time()
                elapsed_total = step_start - start_time
                ur_target, gripper_target = _grasp_demo_targets(
                    elapsed_total,
                    q_start=q_start,
                    q_grasp=q_grasp,
                    q_lift=q_lift,
                    q_insert_path=q_insert_path,
                    q_release=q_release,
                    q_handle_push_path=q_handle_push_path,
                    q_handle_push_segment_durations=q_handle_push_segment_durations,
                    close_m=close_m,
                )
                data.ctrl[ur_ctrl_ids] = ur_target
                data.ctrl[hande_ctrl_ids] = gripper_target
                data.ctrl[:] = _apply_ctrl_limits(model, data.ctrl, "clamp")
                if attach_mode == "kinematic" and 3.5 <= elapsed_total < _TRAY_RELEASE_S:
                    data.qpos[ur_qpos_adrs] = ur_target
                    data.qpos[hande_qpos_adrs] = gripper_target
                    data.qvel[ur_dof_adrs] = 0.0
                    data.qvel[hande_dof_adrs] = 0.0
                if elapsed_total < 3.5:
                    mujoco.mj_forward(model, data)
                    _set_freejoint_transform(model, data, tray_freejoint, initial_tray_transform)
                    mujoco.mj_forward(model, data)
                elif attach_mode == "kinematic" and elapsed_total < _TRAY_RELEASE_S:
                    mujoco.mj_forward(model, data)
                    if attached_relative_transform is None:
                        attached_relative_transform = (
                            _invert_transform(_site_transform(data, grasp_site))
                            @ _body_transform(data, tray_body)
                        )
                    attached_tray_transform = (
                        _site_transform(data, grasp_site) @ attached_relative_transform
                    )
                    _set_freejoint_transform(model, data, tray_freejoint, attached_tray_transform)
                    mujoco.mj_forward(model, data)
                elif attach_mode == "kinematic" and not released_tray:
                    data.qpos[ur_qpos_adrs] = q_release
                    data.qpos[hande_qpos_adrs] = 0.0
                    data.qvel[ur_dof_adrs] = 0.0
                    data.qvel[hande_dof_adrs] = 0.0
                    data.ctrl[ur_ctrl_ids] = q_release
                    data.ctrl[hande_ctrl_ids] = 0.0
                    _zero_freejoint_velocity(model, data, tray_freejoint)
                    mujoco.mj_forward(model, data)
                    released_tray = True
                mujoco.mj_step(model, data)
                if (
                    attach_mode == "kinematic"
                    and released_tray
                    and elapsed_total < _TRAY_RELEASE_DAMPING_DONE_S
                ):
                    _scale_freejoint_velocity(
                        model,
                        data,
                        tray_freejoint,
                        _TRAY_RELEASE_DAMPING_SCALE,
                    )
                if realsense_windows is not None:
                    realsense_windows.update(data, step_start)
                viewer.sync()
                elapsed = time.time() - step_start
                time.sleep(max(0.0, float(model.opt.timestep) - elapsed))
        finally:
            if realsense_windows is not None:
                realsense_windows.close()


class _RealSenseRenderWindows:
    def __init__(
        self,
        model: object,
        camera_name: str,
        width: int,
        height: int,
        fps: float,
        pointcloud_stride: int,
        pointcloud_max_depth_m: float,
    ) -> None:
        import mujoco
        import cv2
        import open3d as o3d

        if width <= 0 or height <= 0:
            raise ValueError("RealSense render width and height must be positive.")
        if fps <= 0.0:
            raise ValueError("RealSense render FPS must be positive.")
        if pointcloud_stride <= 0:
            raise ValueError("pointcloud_stride must be positive.")
        if pointcloud_max_depth_m <= 0.0:
            raise ValueError("pointcloud_max_depth_m must be positive.")

        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"MuJoCo camera not found: {camera_name}")

        self.cv2 = cv2
        self.o3d = o3d
        self.model = model
        self.camera_name = camera_name
        self.camera_id = int(camera_id)
        self.width = int(width)
        self.height = int(height)
        self.interval_s = 1.0 / float(fps)
        self.pointcloud_stride = int(pointcloud_stride)
        self.pointcloud_max_depth_m = float(pointcloud_max_depth_m)
        self.last_render_time = -np.inf
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)
        self.rgb_window_name = "RealSense RGB"
        self.depth_window_name = "RealSense Depth"
        self.rgb_window_open = True
        self.depth_window_open = True

        cv2.namedWindow(self.rgb_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.rgb_window_name, self.width, self.height)
        cv2.namedWindow(self.depth_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.depth_window_name, self.width, self.height)

        self.pointcloud = o3d.geometry.PointCloud()
        self.pointcloud_visualizer = o3d.visualization.Visualizer()
        created = self.pointcloud_visualizer.create_window(
            window_name="RealSense Point Cloud",
            width=self.width,
            height=self.height,
            visible=True,
        )
        if not created:
            self.renderer.close()
            cv2.destroyWindow(self.rgb_window_name)
            cv2.destroyWindow(self.depth_window_name)
            raise RuntimeError("Failed to create the Open3D RealSense point-cloud window.")
        self.pointcloud_visualizer.add_geometry(self.pointcloud)
        render_option = self.pointcloud_visualizer.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.asarray((0.02, 0.025, 0.03), dtype=float)
        view_control = self.pointcloud_visualizer.get_view_control()
        view_control.set_lookat((0.0, 0.0, 0.7))
        view_control.set_front((0.0, 0.0, -1.0))
        view_control.set_up((0.0, -1.0, 0.0))
        view_control.set_zoom(0.65)
        self.pointcloud_window_open = True

    def update(self, data: object, now_s: float) -> None:
        if (
            not self.rgb_window_open
            and not self.depth_window_open
            and not self.pointcloud_window_open
        ):
            return
        if now_s - self.last_render_time < self.interval_s:
            return
        self.last_render_time = float(now_s)

        rgb, depth = self._render_rgb_depth(data)
        if self.rgb_window_open:
            self.rgb_window_open = self._opencv_window_is_open(self.rgb_window_name)
            if self.rgb_window_open:
                self._draw_rgb(rgb)
        if self.depth_window_open:
            self.depth_window_open = self._opencv_window_is_open(self.depth_window_name)
            if self.depth_window_open:
                self._draw_depth(depth)
        if self.rgb_window_open or self.depth_window_open:
            key = int(self.cv2.waitKey(1)) & 0xFF
            if key in (ord("q"), 27):
                self.rgb_window_open = False
                self.depth_window_open = False
                self.cv2.destroyWindow(self.rgb_window_name)
                self.cv2.destroyWindow(self.depth_window_name)

        if self.pointcloud_window_open:
            points, colors = _depth_to_pointcloud(
                depth,
                rgb,
                fovy_deg=float(self.model.cam_fovy[self.camera_id]),
                stride=self.pointcloud_stride,
                max_depth_m=self.pointcloud_max_depth_m,
            )
            self._draw_pointcloud(points, colors)

    def close(self) -> None:
        if getattr(self, "cv2", None) is not None:
            for window_name in (
                getattr(self, "rgb_window_name", None),
                getattr(self, "depth_window_name", None),
            ):
                if window_name:
                    try:
                        self.cv2.destroyWindow(window_name)
                    except Exception:
                        pass
        if getattr(self, "pointcloud_visualizer", None) is not None:
            self.pointcloud_visualizer.destroy_window()
            self.pointcloud_visualizer = None
        if getattr(self, "renderer", None) is not None:
            self.renderer.close()
            self.renderer = None

    def _render_rgb_depth(self, data: object) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8)
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        depth = np.asarray(self.renderer.render(), dtype=float)
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def _draw_rgb(self, rgb: np.ndarray) -> None:
        bgr = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2BGR)
        self.cv2.imshow(self.rgb_window_name, bgr)

    def _draw_pointcloud(self, points: np.ndarray, colors: np.ndarray) -> None:
        o3d = self.o3d
        self.pointcloud.points = o3d.utility.Vector3dVector(points)
        self.pointcloud.colors = o3d.utility.Vector3dVector(
            np.asarray(colors, dtype=float) / 255.0
        )
        self.pointcloud_visualizer.update_geometry(self.pointcloud)
        self.pointcloud_window_open = bool(self.pointcloud_visualizer.poll_events())
        if self.pointcloud_window_open:
            self.pointcloud_visualizer.update_renderer()

    def _draw_depth(self, depth: np.ndarray) -> None:
        valid = np.isfinite(depth) & (depth > 0.0)
        clipped = np.clip(depth, 0.0, self.pointcloud_max_depth_m)
        normalized = 1.0 - clipped / self.pointcloud_max_depth_m
        depth_u8 = np.asarray(np.clip(normalized * 255.0, 0, 255), dtype=np.uint8)
        depth_color = self.cv2.applyColorMap(depth_u8, self.cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
        self.cv2.imshow(self.depth_window_name, depth_color)

    def _opencv_window_is_open(self, window_name: str) -> bool:
        try:
            return self.cv2.getWindowProperty(window_name, self.cv2.WND_PROP_VISIBLE) >= 1.0
        except Exception:
            return False


def _depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    fovy_deg: float,
    stride: int,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    rows = np.arange(0, height, stride, dtype=float)
    cols = np.arange(0, width, stride, dtype=float)
    uu, vv = np.meshgrid(cols, rows)
    sampled_depth = depth[::stride, ::stride]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth > 0.0)
        & (sampled_depth < max_depth_m)
    )
    if not np.any(valid):
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=np.uint8)

    fovy_rad = np.deg2rad(float(fovy_deg))
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    z = sampled_depth[valid]
    x = (uu[valid] - cx) * z / fx
    y = -(vv[valid] - cy) * z / fy
    points = np.column_stack((x, y, z))
    colors = rgb[::stride, ::stride][valid]
    return points, np.asarray(colors, dtype=np.uint8)


def _grasp_demo_targets(
    elapsed_s: float,
    q_start: np.ndarray,
    q_grasp: np.ndarray,
    q_lift: np.ndarray,
    q_insert_path: tuple[np.ndarray, ...],
    q_release: np.ndarray,
    q_handle_push_path: tuple[np.ndarray, ...],
    close_m: float,
    q_handle_push_segment_durations: tuple[float, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    q_align = q_insert_path[0]
    q_place = q_insert_path[-1]
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
    if elapsed_s < _TRAY_ALIGN_DONE_S:
        alpha = (elapsed_s - 6.5) / (_TRAY_ALIGN_DONE_S - 6.5)
        return _lerp(q_lift, q_align, alpha), np.array([close_m, close_m], dtype=float)
    if elapsed_s < _TRAY_INSERT_START_S:
        return q_align, np.array([close_m, close_m], dtype=float)
    if elapsed_s < _TRAY_OPEN_START_S:
        alpha = (elapsed_s - _TRAY_INSERT_START_S) / (
            _TRAY_OPEN_START_S - _TRAY_INSERT_START_S
        )
        return _piecewise_lerp(q_insert_path, alpha), np.array(
            [close_m, close_m],
            dtype=float,
        )
    if elapsed_s < _TRAY_OPEN_DONE_S:
        alpha = (elapsed_s - _TRAY_OPEN_START_S) / (
            _TRAY_OPEN_DONE_S - _TRAY_OPEN_START_S
        )
        opening = float(_lerp(np.array([close_m]), np.array([0.0]), alpha)[0])
        return q_place, np.array([opening, opening], dtype=float)
    if elapsed_s < _TRAY_RELEASE_POSE_DONE_S:
        alpha = (elapsed_s - _TRAY_OPEN_DONE_S) / (
            _TRAY_RELEASE_POSE_DONE_S - _TRAY_OPEN_DONE_S
        )
        return _lerp(q_place, q_release, alpha), np.array([0.0, 0.0], dtype=float)
    if elapsed_s < _PRESSER_HANDLE_PUSH_START_S:
        alpha = (elapsed_s - _TRAY_RELEASE_S) / (
            _PRESSER_HANDLE_PUSH_START_S - _TRAY_RELEASE_S
        )
        gripper = float(np.clip(alpha, 0.0, 1.0) * _PRESSER_HANDLE_PUSH_GRIPPER_M)
        return q_release, np.array([gripper, gripper], dtype=float)
    if not q_handle_push_path or elapsed_s < _PRESSER_HANDLE_PUSH_START_S:
        return q_release, np.array([_PRESSER_HANDLE_PUSH_GRIPPER_M] * 2, dtype=float)
    push_elapsed_s = elapsed_s - _PRESSER_HANDLE_PUSH_START_S
    if q_handle_push_segment_durations:
        path_duration_s = float(np.sum(q_handle_push_segment_durations))
        if push_elapsed_s < path_duration_s:
            return _piecewise_lerp_by_durations(
                (q_release, *q_handle_push_path),
                push_elapsed_s,
                q_handle_push_segment_durations,
            ), np.array(
                [_PRESSER_HANDLE_PUSH_GRIPPER_M, _PRESSER_HANDLE_PUSH_GRIPPER_M],
                dtype=float,
            )
        return q_handle_push_path[-1], np.array(
            [_PRESSER_HANDLE_PUSH_GRIPPER_M, _PRESSER_HANDLE_PUSH_GRIPPER_M],
            dtype=float,
        )
    if elapsed_s < _PRESSER_HANDLE_PUSH_DONE_S:
        alpha = push_elapsed_s / (_PRESSER_HANDLE_PUSH_DONE_S - _PRESSER_HANDLE_PUSH_START_S)
        return _piecewise_lerp((q_release, *q_handle_push_path), alpha), np.array(
            [_PRESSER_HANDLE_PUSH_GRIPPER_M, _PRESSER_HANDLE_PUSH_GRIPPER_M],
            dtype=float,
        )
    return q_handle_push_path[-1], np.array(
        [_PRESSER_HANDLE_PUSH_GRIPPER_M, _PRESSER_HANDLE_PUSH_GRIPPER_M],
        dtype=float,
    )


def _solve_tray_place_ik(
    model: object,
    data: object,
    grasp_site: int,
    tray_body: int,
    joint_names: tuple[str, ...],
    q_grasp: np.ndarray,
    q_lift: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    import mujoco

    tray_hole_sites = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in _TRAY_HOLE_SITE_NAMES
    ]
    assembly_post_sites = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in _ASSEMBLY_POST_SITE_NAMES
    ]
    if min(tray_hole_sites + assembly_post_sites) < 0:
        fallback = q_lift.copy()
        return (fallback,), fallback.copy()

    qpos_adrs = _joint_qpos_addresses(model, joint_names)
    data.qpos[qpos_adrs] = q_grasp
    mujoco.mj_forward(model, data)
    tray_to_grasp = _invert_transform(_body_transform(data, tray_body)) @ _site_transform(
        data, grasp_site
    )

    tray_holes_local = np.asarray(
        [model.site_pos[site_id] for site_id in tray_hole_sites],
        dtype=float,
    )
    assembly_posts_world = np.asarray(
        [data.site_xpos[site_id] for site_id in assembly_post_sites],
        dtype=float,
    )
    base_tray_transform = _fit_transform(tray_holes_local, assembly_posts_world)
    approach_normal = _unit(base_tray_transform[:3, 2])

    q_insert_path = []
    initial_qpos = q_lift
    for offset_m in _TRAY_INSERT_NORMAL_OFFSETS_M:
        insert_transform = base_tray_transform.copy()
        insert_transform[:3, 3] += float(offset_m) * approach_normal
        insert_grasp = insert_transform @ tray_to_grasp
        q_insert = _solve_site_pose_ik(
            model,
            data,
            site_id=grasp_site,
            target_pos=insert_grasp[:3, 3],
            target_rotation=insert_grasp[:3, :3],
            joint_names=joint_names,
            initial_qpos=initial_qpos,
            max_iterations=260,
            position_tolerance_m=8e-4,
            rotation_tolerance_rad=8e-3,
        )
        q_insert_path.append(q_insert)
        initial_qpos = q_insert

    release_transform = base_tray_transform.copy()
    release_transform[:3, 3] += _TRAY_RELEASE_NORMAL_OFFSET_M * approach_normal
    release_grasp = release_transform @ tray_to_grasp
    q_release = _solve_site_pose_ik(
        model,
        data,
        site_id=grasp_site,
        target_pos=release_grasp[:3, 3],
        target_rotation=release_grasp[:3, :3],
        joint_names=joint_names,
        initial_qpos=q_insert_path[-1],
        max_iterations=260,
        position_tolerance_m=8e-4,
        rotation_tolerance_rad=8e-3,
    )
    return tuple(q_insert_path), q_release


def _solve_presser_handle_push_ik(
    model: object,
    data: object,
    pusher_site: int,
    joint_names: tuple[str, ...],
    q_release: np.ndarray,
) -> tuple[np.ndarray, ...]:
    import mujoco

    pivot_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "ti_presser_handle_pivot_site",
    )
    grip_site = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "ti_presser_handle_grip_site",
    )
    handle_joint = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        "ti_presser_handle_hinge_joint",
    )
    if min(pivot_site, grip_site, handle_joint) < 0:
        return ()

    qpos_adrs = _joint_qpos_addresses(model, joint_names)
    handle_qadr = int(model.jnt_qposadr[handle_joint])
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    try:
        data.qpos[qpos_adrs] = q_release
        data.qpos[handle_qadr] = 0.0
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        targets = _presser_handle_push_targets(
            model,
            data,
            pivot_site=pivot_site,
            grip_site=grip_site,
            handle_joint=handle_joint,
        )
        release_pos = data.site_xpos[pusher_site].copy()
        release_rotation = data.site_xmat[pusher_site].reshape(3, 3).copy()
        retreat_pos = release_pos + np.asarray(
            _PRESSER_HANDLE_TRAY_RETREAT_OFFSET_M,
            dtype=float,
        )
        clearance_pos = release_pos + np.asarray(
            _PRESSER_HANDLE_CLEARANCE_OFFSET_M,
            dtype=float,
        )
        q_retreat = _solve_site_pose_ik(
            model,
            data,
            site_id=pusher_site,
            target_pos=retreat_pos,
            target_rotation=release_rotation,
            joint_names=joint_names,
            initial_qpos=q_release,
            max_iterations=360,
            position_tolerance_m=8.0e-4,
            rotation_tolerance_rad=2.5e-2,
        )
        q_clearance = _solve_site_pose_ik(
            model,
            data,
            site_id=pusher_site,
            target_pos=clearance_pos,
            target_rotation=release_rotation,
            joint_names=joint_names,
            initial_qpos=q_retreat,
            max_iterations=360,
            position_tolerance_m=1.0e-3,
            rotation_tolerance_rad=2.5e-2,
        )
        path: list[np.ndarray] = [q_retreat, q_clearance]
        initial_qpos = q_clearance
        if targets:
            high_approach_pos = targets[0][0] + np.asarray(
                _PRESSER_HANDLE_HIGH_APPROACH_OFFSET_M,
                dtype=float,
            )
            q_high_approach = _solve_site_position_xaxis_ik(
                model,
                data,
                site_id=pusher_site,
                target_pos=high_approach_pos,
                target_x_axis=targets[0][1][:, 0],
                joint_names=joint_names,
                initial_qpos=initial_qpos,
                max_iterations=420,
                position_tolerance_m=1.0e-3,
                axis_tolerance_rad=5e-2,
            )
            path.append(q_high_approach)
            initial_qpos = q_high_approach
        for target_pos, target_rotation in targets:
            data.qpos[handle_qadr] = 0.0
            q_push = _solve_site_position_xaxis_ik(
                model,
                data,
                site_id=pusher_site,
                target_pos=target_pos,
                target_x_axis=target_rotation[:, 0],
                joint_names=joint_names,
                initial_qpos=initial_qpos,
                max_iterations=420,
                position_tolerance_m=5e-4,
                axis_tolerance_rad=4e-2,
            )
            path.append(q_push)
            initial_qpos = q_push
        return tuple(path)
    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        mujoco.mj_forward(model, data)


def _presser_handle_push_targets(
    model: object,
    data: object,
    pivot_site: int,
    grip_site: int,
    handle_joint: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    import mujoco

    handle_qadr = int(model.jnt_qposadr[handle_joint])
    handle_body = int(model.jnt_bodyid[handle_joint])
    saved_qpos = data.qpos.copy()
    targets: list[tuple[np.ndarray, np.ndarray]] = []
    first_grip = None
    first_tangent = None
    first_axis = None
    first_rotation = None
    try:
        for angle in _PRESSER_HANDLE_PUSH_ANGLES_RAD:
            data.qpos[handle_qadr] = float(angle)
            mujoco.mj_forward(model, data)
            pivot = data.site_xpos[pivot_site].copy()
            grip = data.site_xpos[grip_site].copy()
            axis = data.xmat[handle_body].reshape(3, 3) @ model.jnt_axis[handle_joint]
            tangent = _unit(np.cross(axis, grip - pivot))
            rotation = _presser_handle_push_rotation(axis, tangent)
            if first_grip is None:
                first_grip = grip
                first_tangent = tangent
                first_axis = _unit(axis)
                first_rotation = rotation
            targets.append(
                (
                    grip
                    + _unit(axis) * _PRESSER_HANDLE_PUSH_AXIS_OFFSET_M
                    - tangent * _PRESSER_HANDLE_PUSH_CONTACT_OFFSET_M,
                    rotation,
                )
            )
    finally:
        data.qpos[:] = saved_qpos
        mujoco.mj_forward(model, data)

    if (
        first_grip is None
        or first_tangent is None
        or first_axis is None
        or first_rotation is None
    ):
        return ()
    approach = (
        first_grip
        + first_axis * _PRESSER_HANDLE_PUSH_AXIS_OFFSET_M
        - first_tangent * _PRESSER_HANDLE_PUSH_PRE_OFFSET_M
    )
    return ((approach, first_rotation), *targets)


def _presser_handle_push_rotation(axis: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    x_axis = _unit(axis)
    z_axis = _unit(tangent - np.dot(tangent, x_axis) * x_axis)
    y_axis = _unit(np.cross(z_axis, x_axis))
    z_axis = _unit(np.cross(x_axis, y_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


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


def _solve_site_position_xaxis_ik(
    model: object,
    data: object,
    site_id: int,
    target_pos: np.ndarray,
    target_x_axis: np.ndarray,
    joint_names: tuple[str, ...],
    initial_qpos: np.ndarray | None = None,
    max_iterations: int = 180,
    position_tolerance_m: float = 2e-4,
    axis_tolerance_rad: float = 2e-2,
) -> np.ndarray:
    import mujoco

    qpos_adrs = _joint_qpos_addresses(model, joint_names)
    dof_adrs = _joint_dof_addresses(model, joint_names)
    if initial_qpos is not None:
        data.qpos[qpos_adrs] = initial_qpos
    target = np.asarray(target_pos, dtype=float)
    target_x = _unit(np.asarray(target_x_axis, dtype=float))
    axis_weight = 0.08
    for _ in range(max_iterations):
        mujoco.mj_forward(model, data)
        current_rotation = data.site_xmat[site_id].reshape(3, 3)
        current_x = _unit(current_rotation[:, 0])
        pos_error = target - data.site_xpos[site_id]
        axis_error = np.cross(current_x, target_x)
        axis_error_norm = float(np.linalg.norm(axis_error))
        if (
            float(np.linalg.norm(pos_error)) < position_tolerance_m
            and axis_error_norm < axis_tolerance_rad
        ):
            break
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack((jacp, axis_weight * jacr))[:, dof_adrs]
        error = np.concatenate((pos_error, axis_weight * axis_error))
        damping = 5e-4
        delta = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), error)
        delta = np.clip(delta, -0.04, 0.04)
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


def _post_insert_segment_durations(
    model: object,
    data: object,
    site_id: int,
    joint_names: tuple[str, ...],
    q_insert_path: tuple[np.ndarray, ...],
    q_release: np.ndarray,
    q_handle_push_path: tuple[np.ndarray, ...],
) -> tuple[float, ...]:
    if not q_handle_push_path or len(q_insert_path) < 2:
        return ()
    insert_positions = _site_positions_for_joint_path(
        model,
        data,
        site_id,
        joint_names,
        q_insert_path,
    )
    insert_length_m = _cartesian_path_length(insert_positions)
    insert_duration_s = _TRAY_OPEN_START_S - _TRAY_INSERT_START_S
    if insert_length_m <= 1e-6 or insert_duration_s <= 0.0:
        return ()
    speed_m_s = insert_length_m / insert_duration_s
    post_positions = _site_positions_for_joint_path(
        model,
        data,
        site_id,
        joint_names,
        (q_release, *q_handle_push_path),
    )
    segment_lengths = np.linalg.norm(np.diff(post_positions, axis=0), axis=1)
    return tuple(float(length / speed_m_s) for length in segment_lengths)


def _site_positions_for_joint_path(
    model: object,
    data: object,
    site_id: int,
    joint_names: tuple[str, ...],
    q_path: tuple[np.ndarray, ...],
) -> np.ndarray:
    import mujoco

    qpos_adrs = _joint_qpos_addresses(model, joint_names)
    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    positions: list[np.ndarray] = []
    try:
        for qpos in q_path:
            data.qpos[qpos_adrs] = qpos
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            positions.append(data.site_xpos[site_id].copy())
    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        mujoco.mj_forward(model, data)
    return np.asarray(positions, dtype=float)


def _cartesian_path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero-length vector.")
    return np.asarray(vector, dtype=float) / norm


def _fit_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source = np.asarray(source_points, dtype=float)
    target = np.asarray(target_points, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_points and target_points must be matching Nx3 arrays.")

    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u_mat, _, vh_mat = np.linalg.svd(covariance)
    rotation = vh_mat.T @ u_mat.T
    if float(np.linalg.det(rotation)) < 0.0:
        vh_mat[-1, :] *= -1.0
        rotation = vh_mat.T @ u_mat.T

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


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


def _piecewise_lerp(points: tuple[np.ndarray, ...], alpha: float) -> np.ndarray:
    if not points:
        raise ValueError("points must contain at least one waypoint.")
    if len(points) == 1:
        return points[0].copy()
    clipped = float(np.clip(alpha, 0.0, 1.0))
    scaled = clipped * (len(points) - 1)
    index = min(int(scaled), len(points) - 2)
    local_alpha = scaled - index
    return _lerp(points[index], points[index + 1], local_alpha)


def _piecewise_lerp_by_durations(
    points: tuple[np.ndarray, ...],
    elapsed_s: float,
    segment_durations_s: tuple[float, ...],
) -> np.ndarray:
    if len(points) != len(segment_durations_s) + 1:
        raise ValueError("segment_durations_s must have one fewer element than points.")
    if len(points) == 1:
        return points[0].copy()
    elapsed = max(float(elapsed_s), 0.0)
    for index, duration_s in enumerate(segment_durations_s):
        duration = max(float(duration_s), 1e-6)
        if elapsed <= duration:
            return _lerp(points[index], points[index + 1], elapsed / duration)
        elapsed -= duration
    return points[-1].copy()


def _set_fixed_viewer_camera(
    model: object,
    viewer: object,
    camera_name: str | None,
) -> bool:
    if camera_name is None or camera_name.lower() in {"", "free"}:
        return False

    import mujoco

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        warnings.warn(
            f"MuJoCo camera not found: {camera_name}; using the free overview camera.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = int(camera_id)
    return True


def _set_grasp_viewer_camera(
    viewer: object,
    data: object,
    grasp_site: int,
    protrusion_site: int,
    assembly_post_sites: tuple[int, ...] = (),
) -> None:
    points = [data.site_xpos[grasp_site], data.site_xpos[protrusion_site]]
    points.extend(data.site_xpos[site_id] for site_id in assembly_post_sites)
    point_array = np.asarray(points, dtype=float)
    lookat = np.mean(point_array, axis=0)
    lookat = np.asarray(lookat, dtype=float)
    lookat[2] += 0.025
    viewer.cam.type = 0
    viewer.cam.lookat[:] = lookat
    span = float(np.linalg.norm(np.max(point_array, axis=0) - np.min(point_array, axis=0)))
    viewer.cam.distance = max(0.45, 2.2 * span)
    viewer.cam.azimuth = 138.0
    viewer.cam.elevation = -16.0


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


def _zero_freejoint_velocity(model: object, data: object, joint_id: int) -> None:
    qvel_adr = int(model.jnt_dofadr[joint_id])
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def _scale_freejoint_velocity(
    model: object,
    data: object,
    joint_id: int,
    scale: float,
) -> None:
    qvel_adr = int(model.jnt_dofadr[joint_id])
    data.qvel[qvel_adr : qvel_adr + 6] *= float(scale)


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
