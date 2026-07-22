"""Offline UR5e inverse kinematics backed by PyRoki.

The real assembly program uses the small FK/IK subset of ``RTDEControlInterface``
while planning and preflighting Cartesian ``moveL`` targets.  This module exposes
the same subset without opening a robot connection, so the real planning helpers
can also drive an offline visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
from scipy.spatial.transform import Rotation


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: Any,
    target_wxyz: Any,
    target_position: Any,
    rest_pose: Any,
) -> Any:
    """Solve one pose target while biasing the result to the prior joint branch."""

    joint_var = robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz),
                target_position,
            ),
            target_link_index,
            pos_weight=100.0,
            ori_weight=50.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=rest_pose,
            weight=0.05,
        ),
        pk.costs.limit_constraint(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]
    solution = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
        .analyze()
        .solve(
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return solution[joint_var]


@dataclass
class PyrokiRTDEControlAdapter:
    """Provide the FK/IK methods used by the real motion planner, fully offline."""

    initial_q_rad: np.ndarray
    tcp_offset_ur: np.ndarray
    description_name: str = "ur5e_description"
    target_link_name: str = "tool0"
    _urdf: Any = field(init=False, repr=False)
    _robot: pk.Robot = field(init=False, repr=False)
    _target_link_index: Any = field(init=False, repr=False)
    _world_t_base: np.ndarray = field(init=False, repr=False)
    _base_t_world: np.ndarray = field(init=False, repr=False)
    _tool0_t_tcp: np.ndarray = field(init=False, repr=False)
    _last_key: tuple[Any, ...] | None = field(init=False, default=None, repr=False)
    _last_result: np.ndarray | None = field(init=False, default=None, repr=False)
    _last_errors: tuple[float, float] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        initial_q = np.asarray(self.initial_q_rad, dtype=float)
        tcp_offset = np.asarray(self.tcp_offset_ur, dtype=float)
        if initial_q.shape != (6,) or not np.all(np.isfinite(initial_q)):
            raise ValueError("initial_q_rad must contain six finite joint angles.")
        if tcp_offset.shape != (6,) or not np.all(np.isfinite(tcp_offset)):
            raise ValueError("tcp_offset_ur must contain six finite values.")

        self.initial_q_rad = initial_q
        self.tcp_offset_ur = tcp_offset
        self._urdf = load_robot_description(self.description_name)
        actuated_names = tuple(self._urdf.actuated_joint_names)
        expected_names = (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
        if actuated_names != expected_names:
            raise RuntimeError(
                "Unexpected UR5e actuated joint order: "
                f"{actuated_names}; expected {expected_names}."
            )

        self._robot = pk.Robot.from_urdf(
            self._urdf,
            default_joint_cfg=self.initial_q_rad,
        )
        self._target_link_index = jnp.asarray(
            self._robot.links.names.index(self.target_link_name),
            dtype=jnp.int32,
        )
        self._world_t_base = np.asarray(
            self._urdf.get_transform("base", "world"),
            dtype=float,
        )
        self._base_t_world = np.linalg.inv(self._world_t_base)
        self._tool0_t_tcp = ur_pose_to_transform(self.tcp_offset_ur)

    @property
    def urdf(self) -> Any:
        """Return the loaded yourdfpy model for use by the visualizer."""

        return self._urdf

    @property
    def robot(self) -> pk.Robot:
        """Return the PyRoki robot model."""

        return self._robot

    def getForwardKinematics(
        self,
        q_rad: list[float],
        tcp_offset_ur: list[float] | None = None,
    ) -> list[float]:
        """Match ``RTDEControlInterface.getForwardKinematics`` for the planner."""

        q = np.asarray(q_rad, dtype=float)
        if q.shape != (6,):
            raise ValueError(f"Expected six joint angles, got {q.shape}.")
        tool0_t_tcp = (
            self._tool0_t_tcp
            if tcp_offset_ur is None
            else ur_pose_to_transform(np.asarray(tcp_offset_ur, dtype=float))
        )
        world_link_poses = np.asarray(self._robot.forward_kinematics(jnp.asarray(q)))
        world_t_tool0 = wxyz_xyz_to_transform(
            world_link_poses[self._robot.links.names.index(self.target_link_name)]
        )
        base_t_tcp = self._base_t_world @ world_t_tool0 @ tool0_t_tcp
        return transform_to_ur_pose(base_t_tcp).tolist()

    def getInverseKinematicsHasSolution(
        self,
        pose_ur: list[float],
        q_near_rad: list[float],
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> bool:
        """Return whether PyRoki reaches the requested RTDE pose tolerances."""

        try:
            self._solve(
                pose_ur,
                q_near_rad,
                position_tolerance_m,
                orientation_tolerance_rad,
            )
        except (RuntimeError, ValueError, FloatingPointError):
            return False
        assert self._last_errors is not None
        position_error_m, orientation_error_rad = self._last_errors
        return bool(
            position_error_m <= float(position_tolerance_m)
            and orientation_error_rad <= float(orientation_tolerance_rad)
        )

    def getInverseKinematics(
        self,
        pose_ur: list[float],
        q_near_rad: list[float],
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> list[float]:
        """Return a continuous six-joint IK solution near ``q_near_rad``."""

        solution = self._solve(
            pose_ur,
            q_near_rad,
            position_tolerance_m,
            orientation_tolerance_rad,
        )
        assert self._last_errors is not None
        position_error_m, orientation_error_rad = self._last_errors
        if position_error_m > float(position_tolerance_m):
            raise RuntimeError(
                f"PyRoki IK position error {position_error_m:.3e} m exceeds "
                f"{float(position_tolerance_m):.3e} m."
            )
        if orientation_error_rad > float(orientation_tolerance_rad):
            raise RuntimeError(
                f"PyRoki IK orientation error {orientation_error_rad:.3e} rad exceeds "
                f"{float(orientation_tolerance_rad):.3e} rad."
            )
        return solution.tolist()

    def _solve(
        self,
        pose_ur: list[float],
        q_near_rad: list[float],
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> np.ndarray:
        target_pose = np.asarray(pose_ur, dtype=float)
        q_near = np.asarray(q_near_rad, dtype=float)
        if target_pose.shape != (6,) or not np.all(np.isfinite(target_pose)):
            raise ValueError("IK target must be a finite six-dimensional UR pose.")
        if q_near.shape != (6,) or not np.all(np.isfinite(q_near)):
            raise ValueError("IK seed must contain six finite joint angles.")

        key = (
            *target_pose.tolist(),
            *q_near.tolist(),
            float(position_tolerance_m),
            float(orientation_tolerance_rad),
        )
        if key == self._last_key and self._last_result is not None:
            return self._last_result.copy()

        base_t_tcp_target = ur_pose_to_transform(target_pose)
        base_t_tool0_target = base_t_tcp_target @ np.linalg.inv(self._tool0_t_tcp)
        world_t_tool0_target = self._world_t_base @ base_t_tool0_target
        target_wxyz = Rotation.from_matrix(world_t_tool0_target[:3, :3]).as_quat()[[3, 0, 1, 2]]
        result = np.asarray(
            _solve_ik_jax(
                self._robot,
                self._target_link_index,
                jnp.asarray(target_wxyz),
                jnp.asarray(world_t_tool0_target[:3, 3]),
                jnp.asarray(q_near),
            ),
            dtype=float,
        )
        if result.shape != (6,) or not np.all(np.isfinite(result)):
            raise RuntimeError(f"PyRoki returned an invalid IK solution: {result}.")

        unwrapped = result + 2.0 * np.pi * np.round((q_near - result) / (2.0 * np.pi))
        lower = np.asarray(self._robot.joints.lower_limits, dtype=float)
        upper = np.asarray(self._robot.joints.upper_limits, dtype=float)
        if np.all(unwrapped >= lower - 1e-6) and np.all(unwrapped <= upper + 1e-6):
            result = unwrapped

        actual_pose = np.asarray(
            self.getForwardKinematics(result.tolist(), self.tcp_offset_ur.tolist()),
            dtype=float,
        )
        position_error_m = float(np.linalg.norm(actual_pose[:3] - target_pose[:3]))
        orientation_error_rad = float(
            (
                Rotation.from_rotvec(actual_pose[3:]).inv() * Rotation.from_rotvec(target_pose[3:])
            ).magnitude()
        )
        self._last_key = key
        self._last_result = result.copy()
        self._last_errors = (position_error_m, orientation_error_rad)
        return result


def ur_pose_to_transform(pose_ur: np.ndarray) -> np.ndarray:
    """Convert UR ``[x, y, z, rx, ry, rz]`` pose to a homogeneous transform."""

    pose = np.asarray(pose_ur, dtype=float)
    if pose.shape != (6,):
        raise ValueError(f"Expected a six-dimensional UR pose, got {pose.shape}.")
    transform = np.eye(4)
    transform[:3, 3] = pose[:3]
    transform[:3, :3] = Rotation.from_rotvec(pose[3:]).as_matrix()
    return transform


def transform_to_ur_pose(transform: np.ndarray) -> np.ndarray:
    """Convert a homogeneous transform to UR position plus rotation vector."""

    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 transform, got {matrix.shape}.")
    pose = np.empty(6, dtype=float)
    pose[:3] = matrix[:3, 3]
    pose[3:] = Rotation.from_matrix(matrix[:3, :3]).as_rotvec()
    return pose


def wxyz_xyz_to_transform(pose: np.ndarray) -> np.ndarray:
    """Convert PyRoki's ``[qw, qx, qy, qz, x, y, z]`` pose to a transform."""

    values = np.asarray(pose, dtype=float)
    if values.shape != (7,):
        raise ValueError(f"Expected a seven-dimensional PyRoki pose, got {values.shape}.")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(values[:4][[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = values[4:]
    return transform
