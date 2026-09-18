"""Deterministic robot/gripper doubles. These never import hardware drivers."""

from contextlib import ExitStack, redirect_stdout
import io
from unittest.mock import patch

import numpy as np


INITIAL_POSE = np.array(
    [-0.2368442756, -0.4167985216, 0.7063085774, -1.2052082715, -1.208916083, 1.2219476042]
)
INITIAL_Q = np.deg2rad([44.85, -15.96, -81.60, 7.72, 89.63, 135.01])


class FakeRobot:
    def __init__(self, *, fail_move=False, no_ik=False, measured_error=False):
        self.pose = INITIAL_POSE.copy()
        self.q = INITIAL_Q.copy()
        self.events = []
        self.fail_move = fail_move
        self.no_ik = no_ik
        self.measured_error = measured_error
        self.ik_calls = 0

    def getActualTCPPose(self):
        return self.pose.tolist()

    def getActualQ(self):
        return self.q.tolist()

    def setTcp(self, offset):
        self.events.append(["setTcp", list(offset)])
        return True

    def getForwardKinematics(self, q, offset):
        return INITIAL_POSE.tolist()

    def getInverseKinematicsHasSolution(self, *args):
        self.ik_calls += 1
        return not self.no_ik

    def getInverseKinematics(self, *args):
        return self.q.tolist()

    def moveJ(self, target, speed, acceleration, asynchronous):
        self.events.append(["moveJ", target, speed, acceleration, asynchronous])
        self.q = np.asarray(target)
        self.pose = INITIAL_POSE.copy()
        return not self.fail_move

    def moveL(self, target, speed, acceleration, asynchronous):
        self.events.append(["moveL", target, speed, acceleration, asynchronous])
        self.pose = np.asarray(target).copy()
        if self.measured_error:
            self.pose[:3] += np.array([0.0001, -0.0002, 0.00015])
        return not self.fail_move

    def stopScript(self):
        self.events.append(["stopScript"])

    def disconnect(self):
        self.events.append(["disconnect"])


def fake_environment(module, args, robot):
    stack = ExitStack()
    stack.enter_context(redirect_stdout(io.StringIO()))
    stack.enter_context(patch.object(module, "parse_args", return_value=args))
    stack.enter_context(patch.object(module, "connect_rtde_control", return_value=robot))
    stack.enter_context(
        patch.object(module, "read_current_tcp_pose", return_value=(robot, robot.pose.copy()))
    )
    stack.enter_context(
        patch.object(
            module.time,
            "sleep",
            side_effect=lambda seconds: robot.events.append(["sleep", seconds]),
        )
    )

    class Gripper:
        def __init__(self, robot_ip, port):
            pass

        def connect(self):
            pass

        def get_position(self):
            return 0

        def move_to_percent(self, command):
            robot.events.append(
                [
                    "gripper",
                    command.name,
                    command.position_percent,
                    command.speed,
                    command.force,
                    command.wait,
                    command.timeout_s,
                    command.post_dwell_s,
                ]
            )

        def close(self):
            pass

    stack.enter_context(patch.object(module, "RobotiqHandEGripper", Gripper))
    if args.auto_pin_image_align:

        def align_image(receive, control, settings, title=""):
            receive.pose[:3] += np.array([0.0004, 0.0008, 0.0002])
            robot.events.append(["image_alignment"])
            return receive.pose.copy()

        stack.enter_context(patch.object(module, "execute_auto_pin_image_alignment", align_image))
    return stack
