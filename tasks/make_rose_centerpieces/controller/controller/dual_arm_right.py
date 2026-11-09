"""Right pad servo for serial dual grasps; all geometry is robot-base relative.

Caller holds shared trunk and left hand, and supplies a full-robot validator that
checks scene, self, BOTH held payloads and their mutual separation. This module
never changes base, trunk, left arm or left gripper action slots.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from controller import ArmServo

class RightArmServo(ArmServo):
    def __init__(self, assets):
        super().__init__(assets, arm='right')
        self.point = np.array([0., 0., -.072])
        self.retain_target = True
        self.locked_trunk = None
        self.last_status = 'idle'

    def solve(self, q, goal, rotation, validator=None, locked=None):
        # Keep first hand's FK unchanged: the four trunk DOFs are shared.
        locks = dict(locked or {})
        locks.update({i: float(q[self.indices[i]]) for i in range(4)})
        return super().solve(q, goal, rotation, validator=validator, locked=locks)

    def set_goal(self, q, goal, rotation, validator):
        """Admit an absolute target only after 101 swept full-q checks.

        Validator is mandatory and must include both payloads where held. No
        missing validator or blocked plan may leave an earlier target active.
        """
        self.target = None
        self.locked_trunk = None
        if not callable(validator):
            self.last_status = 'missing_validator'
            return False
        target = self.solve(q, goal, rotation, validator=validator)
        if target is None:
            self.last_status = 'ik_or_endpoint_blocked'
            return False
        # Narrow IK bounds are numerical, not permission to move the trunk.
        target[:4] = q[self.indices[:4]]
        for fraction in np.linspace(0., 1., 101):
            trial = q.copy()
            trial[self.indices] += fraction * (target - q[self.indices])
            if not validator(trial):
                self.last_status = 'swept_path_blocked'
                return False
        self.target = target
        self.locked_trunk = q[self.indices[:4]].copy()
        self.last_status = 'ready'
        return True

    def set_joint_goal(self, q, right_joints, validator):
        """Admit a measured public right parking posture at the current trunk."""
        self.target=None;self.locked_trunk=None
        joints=np.asarray(right_joints,dtype=float)
        if (not callable(validator) or joints.shape!=(7,) or not np.isfinite(joints).all()
            or np.any(joints<self.limits[0,4:]) or np.any(joints>self.limits[1,4:])):
            self.last_status='invalid_joint_goal';return False
        target=q[self.indices].copy();target[4:]=joints
        for fraction in np.linspace(0.,1.,101):
            trial=q.copy();trial[self.indices]+=fraction*(target-q[self.indices])
            if not validator(trial):
                self.last_status='swept_path_blocked';return False
        self.target=target;self.locked_trunk=q[self.indices[:4]].copy()
        self.last_status='ready';return True

    def hold(self, action, q):
        action[15:22] = q[self.indices[4:]]
        return action

    def step(self, action, q, goal, rotation, validator=None, grip=None):
        """Bounded retained absolute command; fail closed unless freshly checked.

        Uses previous ACTUAL issued right command, never an inferred IK action.
        Goal admission is explicit via set_goal; no unchecked DLS fallback.
        """
        pose = self.pose(q)
        error = (float(np.linalg.norm(np.asarray(goal)-pose[:3,3])),
                 float(np.linalg.norm(R.from_matrix(rotation@pose[:3,:3].T).as_rotvec())))
        if self.target is None or not callable(validator):
            self.last_status = 'no_checked_target'
            self.hold(action, q)
            return error
        if (np.max(np.abs(q[self.indices[:4]]-self.locked_trunk)) > .025 or
            np.max(np.abs(action[3:7]-q[self.indices[:4]])) > .025):
            self.last_status = 'shared_trunk_not_held'
            self.hold(action, q)
            return error
        measured = q[self.indices[4:]]
        previous = np.clip(action[15:22], measured-.12, measured+.12)
        advance = self.target[4:]-previous
        command = previous + advance*min(1., .024/max(float(np.max(np.abs(advance))),1e-9))
        command = np.clip(command, measured-.12, measured+.12)
        command = np.clip(command, self.limits[0,4:]+1e-5, self.limits[1,4:]-1e-5)
        # Include the exact trunk/left commands that will be issued this frame.
        trial = q.copy()
        trial[self.indices[:4]] = action[3:7]
        left = [self.names.index('left_arm_joint'+str(i)) for i in range(1,8)]
        trial[left] = action[7:14]
        trial[self.indices[4:]] = command
        # Check the short command sweep as well as the admitted full path.
        if not all(validator(q+(trial-q)*f) for f in np.linspace(0.,1.,11)):
            self.last_status = 'live_command_blocked'
            self.hold(action, q)
            return error
        action[15:22] = command
        if grip is not None:
            action[22] = float(grip)
        self.last_status = 'moving'
        return error

    def payload_pose(self, q, held_local):
        return self.pose(q) @ held_local
