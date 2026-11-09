"""Joint-space motion primitives for the trunk + left arm, and the 23-D action layout.

Proprioception (61): base velocity 0:3, left arm 3:10, left fingers 24:26, right arm 28:35,
right fingers 49:51, trunk 53:57. Action (23): base velocity 0:3, trunk 3:7, left arm 7:14,
left gripper 14 (+1 open, -1 close), right arm 15:22, right gripper 22.
"""
import numpy as np


def chain_from_proprio(s):
    """Trunk(4) + left arm(7) joint positions in the order used by WholeBodyIK.columns."""
    return np.r_[s[53:57], s[3:10]]


def compose_action(chain_command, right_arm, base=(0., 0., 0.), left_grip=1., right_grip=1.):
    return np.r_[np.asarray(base, float), chain_command[:4], chain_command[4:], left_grip, right_arm, right_grip]


class JointMove:
    """Follows joint waypoints with a per-step rate limit and reports stalls (contact).

    The command never leads the measured position by more than @lead, so an obstructed joint
    does not wind the position controller up against the obstacle.
    """

    def __init__(self, waypoints, rate=.03, lead=.12, tolerance=.02, stall_steps=20):
        self.waypoints = [np.asarray(w, float) for w in waypoints]; self.rate = rate; self.lead = lead
        self.tolerance = tolerance; self.stall_steps = stall_steps
        self.index = 0; self.command = None; self.quiet = 0; self.previous = None; self.steps = 0

    @property
    def done(self):
        return self.index >= len(self.waypoints)

    def step(self, measured):
        """Returns (command, status) with status in 'moving', 'done', 'stalled'."""
        measured = np.asarray(measured, float); self.steps += 1
        if self.command is None:
            self.command = measured.copy()
        if self.done:
            return self.waypoints[-1].copy(), 'done'
        target = self.waypoints[self.index]
        delta = target - self.command
        self.command = self.command + delta * min(1., self.rate / max(np.max(np.abs(delta)), 1e-9))
        self.command = np.clip(self.command, measured - self.lead, measured + self.lead)
        moved = np.inf if self.previous is None else float(np.max(np.abs(measured - self.previous)))
        self.previous = measured.copy()
        remaining = float(np.max(np.abs(target - measured)))
        self.quiet = self.quiet + 1 if (moved < .0015 and remaining > self.tolerance) else 0
        if remaining < self.tolerance:
            self.index += 1; self.quiet = 0
            if self.done:
                return target.copy(), 'done'
        if self.quiet >= self.stall_steps:
            return self.command.copy(), 'stalled'
        return self.command.copy(), 'moving'


def top_down_rotation(closing_xy):
    """Gripper orientation with the fingers pointing straight down and closing along @closing_xy."""
    closing = np.array([closing_xy[0], closing_xy[1], 0.]); closing /= np.linalg.norm(closing)
    up = np.array([0., 0., 1.])
    return np.column_stack([np.cross(closing, up), closing, up])


def interpolate_cartesian(ik, q_full, start, goal, rotation, validator=None, step=.03):
    """IK solutions along a straight tool-point segment, each seeded by the previous one.

    Returns a list of 11-vectors ending at @goal, or None if any intermediate pose has no
    solution close to its predecessor (which would mean a jump between IK branches).
    """
    count = max(1, int(np.ceil(np.linalg.norm(np.asarray(goal) - start) / step)))
    full = np.asarray(q_full, float).copy(); out = []
    for alpha in np.linspace(0, 1, count + 1)[1:]:
        point = start + alpha * (np.asarray(goal) - start)
        # Keep the local redundancy solution along a Cartesian segment. Pulling
        # every waypoint toward a global standing rest pose can jump branches
        # or violate the load limit even for a 5 mm translation.
        solution = ik.solve(full, point, rotation, count=4, validator=validator,
                            rest_pose=full[ik.columns])
        if solution is None or np.max(np.abs(solution - full[ik.columns])) > .6:
            return None
        full[ik.columns] = solution; out.append(solution)
    return out
