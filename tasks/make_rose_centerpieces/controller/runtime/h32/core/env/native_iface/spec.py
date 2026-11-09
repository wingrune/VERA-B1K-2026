"""Native BEHAVIOR-1K observation and action layout seen by RGB-D planners and policies.

Source of truth: OmniGibson bd049de (``robots/robot.py`` proprio dict, ``eval/evaluator.py`` camera order,
``eval/eval_utils.py`` intrinsics and 23D action) and the recorded initial-observation fixtures of the native run.
Table numbers (2.1 to 2.4, 3) refer to the interface contract tables of the porting guide.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

ROBOT = 'robot_r1'
PROPRIO_KEY = f'{ROBOT}::proprio'
CAM_POSES_KEY = f'{ROBOT}::cam_rel_poses'
TASK_ID_KEY = 'task_id'
PROPRIO_DIM = 61
ACTION_DIM = 23

# Contract 2.1: task_id is carried by the native obs dict but never read by the policies. The port module of a task
# supplies its own id (TASK_ID in tasks/<id>_<name>/port/scene.py), there is no table of tasks here.

# Native robot frame: robot.get_position_orientation() is the base_footprint_link pose, which on R1Pro
# coincides with base_link (the port's MANIFEST base_link_name).
BASE_LINK = 'base_link'
EEF_LINKS: Dict[str, str] = {'left': 'left_eef_link', 'right': 'right_eef_link'}


@dataclass(frozen=True)
class CameraRole:
    """One native robot camera (contract 2.1 and 2.3)."""

    name: str
    link: str
    size: int
    fx: float  # pixels at the native size, fx == fy, principal point size / 2
    focal_length: float  # USD camera prim, mm
    horizontal_aperture: float  # USD camera prim, mm; evaluator.py:53 overrides the head to 40

    @property
    def prefix(self) -> str:
        return f'{ROBOT}::{ROBOT}:{self.link}:Camera:0'

    @property
    def rgb_key(self) -> str:
        return self.prefix + '::rgb'

    @property
    def depth_key(self) -> str:
        return self.prefix + '::depth_linear'


# Order is the cam_rel_poses order (evaluator camera_sensor_names): left wrist, right wrist, head.
CAMERAS: Tuple[CameraRole, ...] = (
    CameraRole('left_wrist', 'left_realsense_link', 480, 388.6639, 17.0, 20.995),
    CameraRole('right_wrist', 'right_realsense_link', 480, 388.6639, 17.0, 20.995),
    CameraRole('head', 'zed_link', 720, 306.0, 17.0, 40.0),
)
# Native VisionSensor clipping (vision_sensor.py:142). Newton ignores the prim clipping when Isaac Lab does
# not spawn the camera and takes the far plane from NewtonWarpRendererCfg.max_distance (default 1000),
# writing 0 beyond it, so sim_check passes the far value there.
CLIPPING = (0.001, 1.0e7)


def _arm(side: str) -> Tuple[str, ...]:
    return tuple(f'{side}_arm_joint{i}' for i in range(1, 8))


def _fingers(side: str) -> Tuple[str, ...]:
    return (f'{side}_gripper_finger_joint1', f'{side}_gripper_finger_joint2')


TRUNK = tuple(f'torso_joint{i}' for i in range(1, 5))
BASE_JOINTS = ('base_footprint_x_joint', 'base_footprint_y_joint', 'base_footprint_rz_joint')
BASE_YAW_JOINT = 'base_footprint_rz_joint'

# Contract 2.2: proprio 61D, (field, start, end). Joint fields name their joints in order.
PROPRIO_SLICES: Dict[str, Tuple[int, int]] = {
    'base_qvel': (0, 3),
    'arm_left_qpos': (3, 10), 'arm_left_qvel': (10, 17),
    'eef_left_pos': (17, 20), 'eef_left_quat': (20, 24),
    'gripper_left_qpos': (24, 26), 'gripper_left_qvel': (26, 28),
    'arm_right_qpos': (28, 35), 'arm_right_qvel': (35, 42),
    'eef_right_pos': (42, 45), 'eef_right_quat': (45, 49),
    'gripper_right_qpos': (49, 51), 'gripper_right_qvel': (51, 53),
    'trunk_qpos': (53, 57), 'trunk_qvel': (57, 61),
}
JOINT_FIELDS: Dict[str, Tuple[str, ...]] = {
    'arm_left': _arm('left'), 'arm_right': _arm('right'),
    'gripper_left': _fingers('left'), 'gripper_right': _fingers('right'),
    'trunk': TRUNK,
}

# Contract 3: native 23D action, (name, start, end, joints). Base is (vx, vy, wz) in the robot frame.
ACTION_SLICES: Tuple[Tuple[str, int, int, Tuple[str, ...]], ...] = (
    ('base', 0, 3, BASE_JOINTS),
    ('trunk', 3, 7, TRUNK),
    ('arm_left', 7, 14, _arm('left')),
    ('gripper_left', 14, 15, _fingers('left')),
    ('arm_right', 15, 22, _arm('right')),
    ('gripper_right', 22, 23, _fingers('right')),
)
BASE_OUTPUT_LIMITS = ((-0.75, -0.75, -1.0), (0.75, 0.75, 1.0))
GRIPPER_OUTPUT_LIMITS = (0.0, 0.05)
