"""Generic step-wise motion skills, written as generators that yield one 23-D action per step.

Taken over from the fax controller (m05): the context with the collision validator, base driving,
joint-path planning and following. Task-specific skills are in battery_skills.py.

Frames: M = map frame of the office prior, B = robot base_link.
The estimator supplies base() = (x, y, yaw) of B in M.
"""
import numpy as np
from pathlib import Path
from wbik import WholeBodyIK
from robot_collision import RobotCollision
from kinematics import from_proprio
from skills import JointMove

TIP = (0., 0., -.0788)                       # fingertip centre in the gripper frame
BASE_LINEAR_GAIN = .75                       # m/s per unit of base command
NAV_TRUNK = np.array([1.02, -1.45, -.47, 0.])  # folded trunk for travelling
TALL_TRUNK = np.array([.3, -.6, -.3, 0.])    # upright trunk: its column stays within 0.1 m of the base centre line
GUARD_MARGIN = .015        # every part of the robot stays this far from the boxes in Context.guard (standing batteries)
SCENE_MARGIN = .03                           # clearance kept between the body / idle arm and the scene
HAND_MARGIN = -.012                          # the working hand may come this close to (into) a surface it works on
BASE_Z = .012                                # height of the base_link origin above the floor
# Arms folded in front of the chest for the upright trunk: nothing wider than the shoulders (0.32 m from the
# centre line), nothing below 0.96 m, hands 0.24 m ahead. Found offline with the IK; the right arm mirrors the left.
TUCK = {'left': np.array([.459, -.003, -.131, -1.951, .048, .689, .156]),
        'right': np.array([.459, .003, .131, -1.951, -.048, .689, -.156])}
MAX_LOAD = .85                               # largest static gravity torque accepted, as a fraction of a joint's effort limit
ARMS = {'left': dict(joints=slice(3, 10), fingers=slice(24, 26), side=1.),
        'right': dict(joints=slice(28, 35), fingers=slice(49, 51), side=-1.)}


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose2(x, y, yaw):
    """base_link pose in M for a base standing on the floor (its origin rides BASE_Z above it)."""
    t = np.eye(4); c, s = np.cos(yaw), np.sin(yaw); t[:2, :2] = [[c, -s], [s, c]]; t[:3, 3] = [x, y, BASE_Z]; return t


class Context:
    def __init__(self, folder, scene, estimator, log=print):
        folder = Path(folder)
        self.iks = {arm: WholeBodyIK(folder, arm=arm, tool=TIP) for arm in ARMS}
        self.body = self.iks['left'].body
        self.collision = RobotCollision(folder / 'robot_collision_spheres.json')
        names = self.collision.names
        self.base_part = np.array([n.startswith(('wheel', 'steer', 'base_link')) for n in names])
        self.hands = {arm: np.array([n.startswith(f'{arm}_gripper') or n in [f'{arm}_arm_link7', f'{arm}_realsense_link', f'{arm}_arm_link6'] for n in names]) for arm in ARMS}
        self.scene = scene; self.est = estimator; self.log = log
        self.s = None; self.arm = 'left'; self.steps = 0
        self.trunk_command = None; self.arm_command = {}; self.grip = {'left': 1., 'right': 1.}
        self.events = []
        self.guard = np.zeros((0, 6))     # boxes (M) around small things the arms must not sweep through: standing batteries

    # ---- state -------------------------------------------------------------------------
    def update(self, s):
        self.s = np.asarray(s, float); self.steps += 1
        if self.trunk_command is None:
            self.trunk_command = self.s[53:57].copy()
            self.arm_command = {arm: self.s[spec['joints']].copy() for arm, spec in ARMS.items()}

    @property
    def ik(self):
        return self.iks[self.arm]

    @property
    def hand(self):
        return self.hands[self.arm]

    @property
    def q(self):
        return from_proprio(self.s, self.body.names)

    @property
    def chain(self):
        """Trunk(4) + active arm(7), in the order of the active IK's columns."""
        return np.r_[self.s[53:57], self.s[ARMS[self.arm]['joints']]]

    def fingers(self, arm=None):
        return self.s[ARMS[arm or self.arm]['fingers']]

    def base_matrix(self):
        return pose2(*self.est.base())

    def tip(self, arm=None):
        return self.iks[arm or self.arm].tool_pose(self.q)

    def event(self, name, **values):
        row = dict(event=name, step=self.steps, **values); self.events.append(row); self.log('EVENT', row)

    def set_grip(self, value, arm=None):
        self.grip[arm or self.arm] = value

    def action(self, chain=None, base=(0., 0., 0.)):
        if chain is not None:
            chain = np.asarray(chain, float); self.trunk_command = chain[:4].copy(); self.arm_command[self.arm] = chain[4:].copy()
        return np.r_[np.asarray(base, float), self.trunk_command, self.arm_command['left'], self.grip['left'],
                     self.arm_command['right'], self.grip['right']]

    # ---- collision and load -------------------------------------------------------------
    def load(self, full, arm=None):
        """Largest static gravity torque over trunk and arm joints as a fraction of the effort limit, and its joint."""
        columns = self.iks[arm or self.arm].columns; poses, frames = self.body.fk(full)
        ratio = np.abs(self.body.gravity_torques(poses, frames, columns)) / np.array([self.body.effort[self.body.names[c]] for c in columns])
        k = int(np.argmax(ratio)); return float(ratio[k]), self.body.names[columns[k]]

    def validator(self, base=None, relax_from=None, static_only=False, max_load=None, hand_margin=HAND_MARGIN):
        """Self-collision + scene (height field and boxes) for the whole robot. @base overrides the base pose in M.

        The working hand may touch the surface it works on (hand_margin); the chassis is tested against
        furniture only, not against the floor it stands on. relax_from: a full configuration (normally the
        current one); clearances already tighter than the margins there are tolerated as long as they do not
        get worse. max_load: reject configurations that overload a joint statically.
        """
        world = self.base_matrix() if base is None else base
        radii = self.collision.radii; hand = self.hand; chassis = self.base_part
        margin = np.where(hand, hand_margin, np.where(chassis, 0., SCENE_MARGIN)); ignore = np.where(chassis, .06, 0.)
        self_baseline = None; arm = self.arm

        guard = self.guard; guard_margin = GUARD_MARGIN

        def scene_gap(poses):
            centers = self.collision.centers(poses)
            return self.scene.sphere_clearance(centers @ world[:3, :3].T + world[:3, 3], radii, ignore_below=ignore, static_only=static_only)

        def guard_gap(poses):
            centers = self.collision.centers(poses) @ world[:3, :3].T + world[:3, 3]
            nearest = np.clip(centers[None], guard[:, None, :3], guard[:, None, 3:])
            return (np.linalg.norm(centers[None] - nearest, axis=2) - radii[None]).min(axis=0)

        if relax_from is not None:
            poses0 = self.body.fk(relax_from)[0]
            margin = np.minimum(margin, scene_gap(poses0) - .004); self_baseline = self.collision.margins(poses0)
            if len(guard):
                guard_margin = np.minimum(guard_margin, guard_gap(poses0) - .004)

        def check(full, poses):
            if not self.collision.valid(poses, self_baseline):
                return False
            if np.any(scene_gap(poses) < margin):
                return False
            if len(guard) and np.any(guard_gap(poses) < guard_margin):
                return False
            return max_load is None or self.load(full, arm)[0] <= max_load
        return check


# ---- primitive skills ----------------------------------------------------------------------
def hold(ctx, steps):
    for _ in range(steps):
        yield ctx.action()


def follow(ctx, waypoints, rate=.03, max_steps=400):
    move = JointMove(waypoints, rate=rate); status = 'moving'; count = 0
    while status == 'moving' and count < max_steps:
        command, status = move.step(ctx.chain); count += 1
        yield ctx.action(chain=command)
    return status


def drive(ctx, goal_xy, goal_yaw, speed=.5, direct=False, static_only=False, tolerance=.035, yaw_tolerance=.04, max_steps=1500, face_travel=True, keepout=(), min_speed=0.):
    """Holonomic drive to a base pose in M along a planned route (or straight if @direct).

    min_speed: floor on the closing speed near the goal (m/s), for goals with a tolerance of a
    centimetre or two: the proportional command alone fades before such a tolerance is met.
    """
    goal_xy = np.asarray(goal_xy, float)
    x, y, yaw = ctx.est.base()
    path = [np.array([x, y]), goal_xy] if direct else ctx.scene.path((x, y), goal_xy, static_only=static_only, keepout=keepout)
    if path is None:
        ctx.event('drive', ok=False, reason='no route', goal=goal_xy.round(2).tolist()); return 'no_route'
    path = [np.asarray(p, float) for p in path]
    lengths = np.r_[0., np.cumsum([np.linalg.norm(b - a) for a, b in zip(path, path[1:])])]
    command = np.zeros(3); history = []
    for count in range(max_steps):
        x, y, yaw = ctx.est.base(); position = np.array([x, y])
        remaining = float(np.linalg.norm(goal_xy - position)); yaw_error = wrap(goal_yaw - yaw)
        if remaining < tolerance and abs(yaw_error) < yaw_tolerance:
            for _ in range(4):
                yield ctx.action(base=(0., 0., 0.))
            return 'arrived'
        # Closest point on the route, then a carrot ahead of it.
        along = 0.; nearest = np.inf
        for a, b, s0 in zip(path, path[1:], lengths):
            ab = b - a; t = np.clip(np.dot(position - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1); d = np.linalg.norm(position - (a + t * ab))
            if d < nearest - 1e-6:
                nearest = d; along = s0 + t * np.linalg.norm(ab)
        target_s = min(along + .35, lengths[-1]); k = int(np.searchsorted(lengths, target_s, side='right') - 1); k = min(k, len(path) - 2)
        carrot = path[k] + (path[k + 1] - path[k]) * ((target_s - lengths[k]) / max(lengths[k + 1] - lengths[k], 1e-9))
        to_carrot = carrot - position; distance_left = lengths[-1] - along
        velocity = to_carrot / max(np.linalg.norm(to_carrot), 1e-9) * min(speed, max(.04, 1.2 * (distance_left + nearest)))
        if remaining < .12:
            velocity = (goal_xy - position) * 1.5
            if min_speed and remaining > tolerance:
                velocity = velocity * max(1., min_speed / max(float(np.linalg.norm(velocity)), 1e-9))
        heading = goal_yaw
        if face_travel and distance_left > 1.0:
            heading = np.arctan2(to_carrot[1], to_carrot[0])
        yaw_error = wrap(heading - yaw)
        if abs(yaw_error) > .6 and distance_left > .5:
            velocity = velocity * .25          # turn mostly in place before travelling
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array([c * velocity[0] + s * velocity[1], -s * velocity[0] + c * velocity[1]]) / BASE_LINEAR_GAIN
        wanted = np.r_[np.clip(local, -.9, .9), np.clip(1.6 * yaw_error, -.8, .8)]
        command = command + np.clip(wanted - command, -.06, .06)
        yield ctx.action(base=command)
        # Stalled = the base itself has not moved or turned for three seconds while being commanded.
        # (Distance to the goal is no measure: a route around the cubicle block first leads away from it.)
        history.append((x, y, yaw))
        if len(history) > 90:
            x0, y0, yaw0 = history[-90]
            if np.hypot(x - x0, y - y0) < .02 and abs(wrap(yaw - yaw0)) < .03:
                for _ in range(4):
                    yield ctx.action(base=(0., 0., 0.))
                ctx.event('drive', ok=False, reason='stalled', remaining=round(remaining, 3)); return 'stalled'
    return 'timeout'


def plan_joint_path(ctx, goal, check, seconds=8., seed=0):
    """Collision-checked joint path of the active chain from the current configuration to @goal.

    Tries the straight segment first, then a bidirectional RRT. Returns a list of waypoints ending
    at @goal, or None.
    """
    import time
    columns = ctx.ik.columns; base_q = ctx.q; start = ctx.chain.copy(); goal = np.asarray(goal, float)
    lower, upper = ctx.ik.lower, ctx.ik.upper; rng = np.random.default_rng(seed); deadline = time.monotonic() + seconds

    def valid(x):
        full = base_q.copy(); full[columns] = x
        return check(full, ctx.body.fk(full)[0])

    def edge(a, b):
        count = max(1, int(np.ceil(np.max(np.abs(b - a)) / .08)))
        return all(valid(a + (b - a) * k / count) for k in range(1, count + 1))

    if not valid(goal):
        return None
    if edge(start, goal):
        return [goal]
    trees = [([start], [-1]), ([goal], [-1])]

    def extend(tree, sample):
        nodes, parents = tree
        j = int(np.argmin([np.max(np.abs(x - sample)) for x in nodes])); delta = sample - nodes[j]; size = np.max(np.abs(delta))
        new = nodes[j] + delta * min(1., .30 / max(size, 1e-9))
        if not edge(nodes[j], new):
            return None, False
        nodes.append(new); parents.append(j); return len(nodes) - 1, size <= .30

    def chain(tree, j):
        out = []
        while j >= 0:
            out.append(tree[0][j]); j = tree[1][j]
        return out[::-1]

    iteration = 0
    while time.monotonic() < deadline:
        side = iteration % 2; iteration += 1
        blend = rng.random()
        sample = np.clip(start * blend + goal * (1 - blend) + rng.normal(0, .45, len(start)), lower + 1e-4, upper - 1e-4)
        a, _ = extend(trees[side], sample)
        if a is None:
            continue
        for _ in range(20):
            b, joined = extend(trees[1 - side], trees[side][0][a])
            if b is None:
                break
            if joined:
                left, right = chain(trees[side], a), chain(trees[1 - side], b)
                path = left + right[::-1] if side == 0 else right + left[::-1]
                short = [path[0]]; i = 0
                while i < len(path) - 1:
                    j = len(path) - 1
                    while j > i + 1 and not edge(path[i], path[j]):
                        j -= 1
                    short.append(path[j]); i = j
                return short[1:]
    return None


def move_to(ctx, goal, rate=.03, max_steps=500, label='move', static_only=False, max_load=None):
    """Planned, collision-checked joint motion of the active chain to @goal. Returns the follow status."""
    check = ctx.validator(relax_from=ctx.q, static_only=static_only, max_load=max_load)
    path = plan_joint_path(ctx, goal, check)
    if path is None:
        ctx.event(label, ok=False, reason='no collision-free joint path'); return 'no_path'
    status = yield from follow(ctx, path, rate=rate, max_steps=max_steps)
    if status != 'done':
        ctx.event(label, ok=False, reason=status, waypoints=len(path))
    return status
