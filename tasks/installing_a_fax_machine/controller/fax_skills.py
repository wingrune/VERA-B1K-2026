"""Step-wise skills for the fax task, written as generators that yield one 23-D action per step.

A skill reads the latest proprioception and state estimates through the shared Context and is
resumed after every simulator step, so a sequential script stays readable while still fitting
the policy's forward(obs) -> action interface.

Frames: M = map frame of the office prior, B = robot base_link, O = fax model frame.
The estimator supplies base() = (x, y, yaw) of B in M and fax() = 4x4 pose of O in M (or None).

Either arm can do the work. In the narrow aisles the fax often leaves room for the base on one
side only, and which arm reaches it from there depends on that side.
"""
import numpy as np
from pathlib import Path
from wbik import WholeBodyIK
from robot_collision import RobotCollision
from kinematics import from_proprio
from skills import JointMove, top_down_rotation, interpolate_cartesian

TIP = (0., 0., -.0788)                       # fingertip centre in the gripper frame
BUTTON = np.array([.0308, .0872, .0179])     # toggle marker in the fax frame
GRASP = np.array([-.008, 0.0, .010])         # fingertip centre at closure, fax frame
FAX_CENTER = np.array([.016, -.006, .002]); FAX_HALF = np.array([.070, .130, .030])
PAYLOAD = np.array([[0., -.085, 0.], [0., 0., 0.], [0., .085, 0.]]) + FAX_CENTER; PAYLOAD_RADIUS = .085
# Where the fax should be in the base frame for a floor grasp with the LEFT arm (mirrored for the right).
FLOOR_REACH = [(.45, .20), (.50, .25), (.55, .30), (.40, .30), (.60, .20), (.45, .10), (.55, .10), (.65, .30)]
BASE_LINEAR_GAIN = .75                       # m/s per unit of base command
NAV_TRUNK = np.array([1.02, -1.45, -.47, 0.])
PRESS_OFFSET = .006                          # the fingertips aim this far down-slope of the marker centre
FAX_KEEPOUT = .60                            # base centre stays this far from the fax while travelling
SCENE_MARGIN = .04                           # clearance kept between the body / idle arm and the office geometry
BASE_Z = .012                                # height of the base_link origin above the floor
ROBUST_SOLUTIONS = 3                         # a floor stance is taken at once if this many arm configurations reach the grasp
MORE_SEEDS = 72                              # IK seeds of the second search when the robot already stands at the stance
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
        # Links that are meant to work next to the floor / the object, per arm.
        self.hands = {arm: np.array([n.startswith(f'{arm}_gripper') or n in [f'{arm}_arm_link7', f'{arm}_realsense_link', f'{arm}_arm_link6'] for n in names]) for arm in ARMS}
        self.arm_parts = {arm: np.array([n.startswith((f'{arm}_arm', f'{arm}_gripper', f'{arm}_realsense')) for n in names]) for arm in ARMS}
        self.scene = scene; self.est = estimator; self.log = log
        self.s = None; self.arm = 'left'; self.steps = 0
        self.trunk_command = None; self.arm_command = {}; self.grip = {'left': 1., 'right': 1.}
        self.held = None                      # gripper <- fax transform while carrying
        self.events = []
        self.on_refresh = None                # optional callable: re-estimate the fax pose from the cameras now
        self.released = None                  # fax pose in the base frame at the moment it was let go

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

    @property
    def fingers(self):
        return self.s[ARMS[self.arm]['fingers']]

    def base_matrix(self):
        return pose2(*self.est.base())

    def fax_in_base(self, refresh=False):
        if refresh and self.on_refresh is not None:
            self.on_refresh()
        fax = self.est.fax()
        return None if fax is None else np.linalg.inv(self.base_matrix()) @ fax

    def tip(self):
        return self.ik.tool_pose(self.q)

    def event(self, name, **values):
        row = dict(event=name, step=self.steps, **values); self.events.append(row); self.log('EVENT', row)

    def set_grip(self, value):
        self.grip[self.arm] = value

    def action(self, chain=None, base=(0., 0., 0.)):
        if chain is not None:
            chain = np.asarray(chain, float); self.trunk_command = chain[:4].copy(); self.arm_command[self.arm] = chain[4:].copy()
        return np.r_[np.asarray(base, float), self.trunk_command, self.arm_command['left'], self.grip['left'],
                     self.arm_command['right'], self.grip['right']]

    # ---- collision ---------------------------------------------------------------------
    def validator(self, base=None, payload=False, compact=None, relax_from=None, static_only=False, over=None):
        """Self-collision + office height field (+ carried fax). @base overrides the base pose in M.

        relax_from: a full configuration (normally the current one). Clearances that are already
        tighter than the margins there are tolerated as long as they do not get worse, so a motion
        can start from a tight spot and leave it.
        over: a height above the floor (m). In front of the base the working arm and the carried
        fax must stay above it. Used in the cubicle nook, where the chair has been pushed somewhere
        ahead of the base and is no longer where the map has it.
        """
        world = self.base_matrix() if base is None else base
        radii = self.collision.radii; hand = self.hand; arm_part = self.arm_parts[self.arm]; link = self.ik.link
        ignore = np.where(hand, .12, 0.); moving = ~self.base_part
        margin = np.where(hand[moving], -.02, SCENE_MARGIN); self_baseline = None

        def scene_gap(poses):
            centers = self.collision.centers(poses)
            in_map = centers[moving] @ world[:3, :3].T + world[:3, 3]
            return centers, self.scene.sphere_clearance(in_map, radii[moving], ignore_below=ignore[moving], static_only=static_only)

        if relax_from is not None:
            poses0 = self.body.fk(relax_from)[0]
            margin = np.minimum(margin, scene_gap(poses0)[1] - .004); self_baseline = self.collision.margins(poses0)

        def check(full, poses):
            if not self.collision.valid(poses, self_baseline):
                return False
            centers, gap = scene_gap(poses)
            if compact is not None and np.any((np.linalg.norm(centers[:, :2], axis=1) + radii)[arm_part] > compact):
                return False
            if np.any(gap < margin):
                return False
            if over is not None:
                ahead = arm_part & (centers[:, 0] > .38)
                if np.any((centers[:, 2] + BASE_Z - radii)[ahead] < over):
                    return False
            if payload and self.held is not None:
                grip = poses[link] @ self.held
                points = PAYLOAD @ grip[:3, :3].T + grip[:3, 3]
                if over is not None and grip[0, 3] > .38:
                    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * FAX_HALF + FAX_CENTER
                    if float((corners @ grip[:3, :3].T + grip[:3, 3])[:, 2].min()) + BASE_Z < over:
                        return False
                body = centers[~hand]
                if np.any(np.linalg.norm(points[:, None] - body[None], axis=2) < PAYLOAD_RADIUS + radii[~hand][None] - .01):
                    return False
                in_map = points @ world[:3, :3].T + world[:3, 3]
                if np.any(self.scene.sphere_clearance(in_map, np.full(len(points), PAYLOAD_RADIUS), static_only=static_only) < .01):
                    return False
            return True
        return check

    def solve(self, point, rotation, count=20, base=None, payload=False, q=None, free_axis=None, compact=None):
        return self.ik.solve(self.q if q is None else q, np.asarray(point, float), rotation, count=count,
                             validator=self.validator(base, payload, compact), free_axis=free_axis)


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


def move_to(ctx, goal, payload=False, rate=.03, max_steps=500, label='move', static_only=False, over=None):
    """Planned, collision-checked joint motion of the active chain to @goal. Returns the follow status."""
    check = ctx.validator(payload=payload, relax_from=ctx.q, static_only=static_only, over=over)
    path = plan_joint_path(ctx, goal, check)
    if path is None:
        ctx.event(label, ok=False, reason='no collision-free joint path'); return 'no_path'
    status = yield from follow(ctx, path, rate=rate, max_steps=max_steps)
    if status != 'done':
        ctx.event(label, ok=False, reason=status, waypoints=len(path))
    return status


# ---- floor stance, press, grasp ------------------------------------------------------------
def grasp_rotations(fax_in_base):
    closing = fax_in_base[:3, 0]
    return [top_down_rotation(closing[:2]), top_down_rotation(-closing[:2])]


def grasp_point(fax_in_base):
    point = fax_in_base[:3, 3] + fax_in_base[:3, :3] @ np.array([GRASP[0], GRASP[1], 0.]); point[2] = fax_in_base[2, 3] + GRASP[2]
    return point


def floor_prestance(ctx, base_xy, fax_xy):
    """Point a little behind the stance, away from the fax: the route ends there, outside the keep-out
    disc, and the last stretch to the stance is a straight move with the final heading."""
    away = np.asarray(base_xy, float) - fax_xy; away /= max(np.linalg.norm(away), 1e-9)
    for back in (.30, .22, .15, .08, 0.):
        point = np.asarray(base_xy, float) + back * away
        if ctx.scene.base_free(point, margin=.02) and np.linalg.norm(point - fax_xy) > FAX_KEEPOUT + .02:
            return point
    return np.asarray(base_xy, float)


def plan_floor_stance(ctx, fax_world, limit=12, exclude=()):
    """(base_xy, yaw, arm): a base pose in M from which press and grasp are reachable collision-free.

    exclude: stances (as returned here) that have been tried and must not be offered again.
    A stance is taken at once when several distinct arm configurations reach the grasp
    (ROBUST_SOLUTIONS). One that is reached by fewer is kept in reserve and used only if nothing
    better is found: on start 312 the single configuration that fitted beside a partition end was
    lost when the base stopped three centimetres short.
    """
    x, y, yaw = ctx.est.base(); fax_xy = fax_world[:2, 3]; candidates = []
    outline = ctx.collision.names == 'base_link'
    centres = ctx.collision.local[outline][:, :2]; reach = ctx.collision.radii[outline] + np.linalg.norm(FAX_HALF[:2]) + .03
    for arm, spec in ARMS.items():
        offsets = [(fx, spec['side'] * fy) for fx, fy in FLOOR_REACH]
        offsets = [(fx, fy) for fx, fy in offsets if np.all(np.hypot(centres[:, 0] - fx, centres[:, 1] - fy) > reach)]
        for theta in np.linspace(-np.pi, np.pi, 24, endpoint=False):
            c, s = np.cos(theta), np.sin(theta)
            for fx, fy in offsets:
                base_xy = fax_xy - np.array([c * fx - s * fy, s * fx + c * fy])
                if not ctx.scene.base_free(base_xy, margin=.03):
                    continue
                # Reaching the floor makes the upper body lean about 0.6-0.9 m ahead of the base:
                # that volume must be empty in the prior map (the cameras look down at that moment).
                ahead = np.array([[base_xy[0] + d * c, base_xy[1] + d * s, .95] for d in (.30, .50, .70, .90)])
                room = 0.
                for radius in (.22, .30, .38, .46):
                    if np.any(ctx.scene.sphere_clearance(ahead, np.full(len(ahead), radius)) < 0.):
                        break
                    room = radius
                if room == 0.:
                    continue
                # Prefer leaning where there is elbow room: in the one-metre aisles a lean towards a
                # partition leaves the shoulders and the idle arm a few centimetres, and they catch.
                cost = np.linalg.norm(base_xy - [x, y]) + .25 * abs(wrap(theta - yaw)) + 4. * (.46 - room)
                candidates.append((cost, base_xy, theta, arm))
    candidates.sort(key=lambda item: item[0]); tried = 0; unreachable = 0; reserve = None
    nominal = ctx.q.copy(); nominal[ctx.iks['left'].columns[:4]] = NAV_TRUNK
    previous = ctx.arm
    try:
        for cost, base_xy, theta, arm in candidates:
            if tried >= limit:
                break
            if any(arm == a and np.linalg.norm(base_xy - np.asarray(b)) < .08 and abs(wrap(theta - t)) < .14 for b, t, a in exclude):
                continue
            if ctx.scene.path((x, y), floor_prestance(ctx, base_xy, fax_xy), keepout=[(fax_xy, FAX_KEEPOUT)]) is None:
                unreachable += 1; continue
            tried += 1; ctx.arm = arm
            world = pose2(base_xy[0], base_xy[1], theta); fax = np.linalg.inv(world) @ fax_world
            button = fax[:3, 3] + fax[:3, :3] @ BUTTON
            solutions = 0
            for rotation in grasp_rotations(fax):
                info = {}
                if plan_descent(ctx, grasp_point(fax), rotation, .12, base=world, q=nominal, count=14, info=info) is None:
                    continue
                if all(plan_descent(ctx, button + [PRESS_OFFSET, 0., -.006], option, .096, base=world, q=nominal, count=10) is None for option in grasp_rotations(fax)):
                    continue
                solutions = max(solutions, info['solutions'])
            if solutions == 0:
                continue
            found = dict(base=[round(float(v), 3) for v in (*base_xy, theta)], arm=arm, fax_in_base=fax[:2, 3].round(3).tolist(), solutions=solutions)
            if solutions >= ROBUST_SOLUTIONS:
                ctx.event('stance', **found, tried=tried, unreachable=unreachable)
                return base_xy, theta, arm
            if reserve is None:
                reserve = (found, (base_xy, theta, arm))
    finally:
        ctx.arm = previous
    if reserve is not None:
        ctx.event('stance', **reserve[0], tried=tried, unreachable=unreachable, marginal=True)
        return reserve[1]
    ctx.event('stance', ok=False, candidates=len(candidates), tried=tried, unreachable=unreachable, excluded=len(exclude))
    return None


def plan_descent(ctx, point, rotation, height, base=None, q=None, static_only=False, free_yaw=False, count=24, info=None, more=0):
    """A straight vertical approach to @point (base frame) from @height above it.

    The contact pose is solved first and the way up is followed from there, so the pose above and
    the pose at contact belong to the same arm configuration. Solving the upper pose first, as
    before, sometimes picked a configuration that could not come straight down.
    Returns (pose above, joint waypoints down to contact, tool rotation used) or None.

    info: optional dict; receives 'solutions', the number of collision-free contact configurations
    found with @count seeds (a measure of how much room the pose has).
    more: when nothing is found with @count seeds, search once more with this many. For use where
    the robot already stands at the stance and the alternative is to give the stance up.
    """
    full = ctx.q if q is None else np.asarray(q, float)
    check = ctx.validator(base=base, static_only=static_only)
    for seeds, keep in ((count, 6), (more, 12)):
        if not seeds:
            break
        lows = ctx.ik.solve(full, np.asarray(point, float), rotation, count=seeds, validator=check,
                            free_axis=np.array([0., 0., 1.]) if free_yaw else None, all_solutions=True)
        if info is not None and seeds == count:
            info['solutions'] = len(lows)
        for low in lows[:keep]:
            start = full.copy(); start[ctx.ik.columns] = low
            used = ctx.ik.tool_pose(start)[:3, :3] if free_yaw else rotation
            up = interpolate_cartesian(ctx.ik, start, np.asarray(point, float), np.asarray(point, float) + [0., 0., height], used, validator=check, step=.02)
            if up is None:
                continue
            return up[-1], up[::-1][1:] + [low], used
    return None


def choose_rotation(ctx, fax):
    best = None
    for rotation in grasp_rotations(fax):
        plan = plan_descent(ctx, grasp_point(fax), rotation, .12, more=MORE_SEEDS)
        if plan is not None:
            cost = float(np.linalg.norm(plan[0] - ctx.chain))
            if best is None or cost < best[0]:
                best = (cost, rotation)
    return None if best is None else best[1]


def press_button(ctx, rotation, offset=PRESS_OFFSET, static_only=False, free_yaw=False, depth=.006):
    """Touch the toggle marker with the closed fingertips. Returns True if the motion completed.

    free_yaw: the closed gripper may come down at any yaw (used on the desk, where reach is easy).
    """
    def target():
        fax = ctx.fax_in_base(refresh=True)
        return fax[:3, 3] + fax[:3, :3] @ (BUTTON + [offset, 0., -depth])
    plan = plan_descent(ctx, target(), rotation, .096, static_only=static_only, free_yaw=free_yaw, more=MORE_SEEDS)
    if plan is None:
        ctx.event('press', ok=False, reason='no straight approach to the button'); return False
    above, down, rotation = plan
    ctx.set_grip(-1.)
    status = yield from move_to(ctx, above, rate=.035, label='press', static_only=static_only)
    if status != 'done':
        ctx.set_grip(1.); return False
    yield from hold(ctx, 3)
    # A closer look from above: re-aim the same straight approach at the refreshed button position.
    button = target()
    down = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], button, rotation, step=.015) or down
    status = yield from follow(ctx, down, rate=.012, max_steps=250)
    yield from hold(ctx, 12)
    ctx.event('press', ok=True, status=status, tip_minus_button=(ctx.tip()[:3, 3] - (button + [0., 0., depth])).round(4).tolist())
    yield from follow(ctx, [above], rate=.03)
    ctx.set_grip(1.)
    yield from hold(ctx, 18)
    return True


def grasp_from_floor(ctx, rotation):
    """Top-down pinch across the fax's raised back step; relies on the simulator's assisted grasp."""
    plan = plan_descent(ctx, grasp_point(ctx.fax_in_base()), rotation, .12, more=MORE_SEEDS)
    if plan is None:
        ctx.event('grasp', ok=False, reason='no straight approach to the grasp'); return False
    pre, down, _ = plan
    ctx.set_grip(1.)
    status = yield from move_to(ctx, pre, rate=.035, label='grasp')
    if status != 'done':
        return False
    yield from hold(ctx, 3)
    point = grasp_point(ctx.fax_in_base(refresh=True))
    down = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], point, rotation, step=.02) or down
    status = yield from follow(ctx, down, rate=.012, max_steps=300)
    ctx.set_grip(-1.)
    yield from hold(ctx, 30)
    fingers = ctx.fingers.copy()
    # Record where the fax sits in the hand, for carrying and placing.
    fax = ctx.fax_in_base(); ctx.held = np.linalg.inv(ctx.body.fk(ctx.q)[0][ctx.ik.link]) @ fax
    lift = ctx.solve(ctx.tip()[:3, 3] + [0, 0, .20], rotation, count=8)
    if lift is not None:
        yield from follow(ctx, [lift], rate=.02)
    yield from hold(ctx, 10)
    ctx.event('grasp', ok=True, descend=status, fingers=fingers.round(4).tolist(), lifted=lift is not None)
    return True


# ---- carrying and placing ------------------------------------------------------------------
def to_carry(ctx, rotation):
    """Bring the held fax up in front of the chest, inside the base outline as far as possible."""
    point = np.array([.38, .10 * ARMS[ctx.arm]['side'], .98]); free = np.array([0., 0., 1.])
    for attempt in range(3):
        goal = None
        for compact in (.56, .62, .70):
            goal = ctx.ik.solve(ctx.q, point, rotation, count=24, validator=ctx.validator(payload=True, compact=compact), free_axis=free, seed=attempt)
            if goal is not None:
                break
        if goal is None:
            ctx.event('carry', ok=False, reason='no carry pose'); return False
        status = yield from move_to(ctx, goal, payload=True, label='carry')
        if status == 'done':
            ctx.event('carry', ok=True, attempt=attempt, tip=ctx.tip()[:3, 3].round(3).tolist()); return True
    return False


def carrel_sites(scene, arm='left'):
    """Place sites of the west-column cubicles: entry point, final stance, target on the desk (all in M).

    Each of these cubicles has an L-shaped desk around a nook that is open towards the aisle and to
    the west. The base drives into the nook along its west side and reaches over the desk edge.
    """
    sites = {}; side = ARMS[arm]['side']
    for key, carrel in scene.meta['carrels'].items():
        lo, hi = np.asarray(carrel['aabb_lo']), np.asarray(carrel['aabb_hi'])
        if lo[0] > 9.5:
            continue                          # east column: the nook is closed on both sides
        x = lo[0] + .24
        if 'hzalfx' in carrel['name']:        # nook opens to +y, the robot faces -y
            yaw = -np.pi / 2; entry = (x, hi[1] + .45); stance = (x, hi[1] - .28)
        else:                                 # mirrored model: nook opens to -y, the robot faces +y
            yaw = np.pi / 2; entry = (x, lo[1] - .45); stance = (x, lo[1] + .28)
        c, s = np.cos(yaw), np.sin(yaw); forward, lateral = .62, .05 * side
        target = (stance[0] + c * forward - s * lateral, stance[1] + s * forward + c * lateral)
        sites[key] = dict(entry=entry, stance=stance, yaw=yaw, target=target)
    return sites


def choose_site(ctx):
    x, y, _ = ctx.est.base(); best = None
    for key, site in carrel_sites(ctx.scene, ctx.arm).items():
        route = ctx.scene.path((x, y), site['entry'])
        if route is None:
            continue
        length = sum(np.linalg.norm(np.asarray(b) - a) for a, b in zip(route, route[1:]))
        if best is None or length < best[0]:
            best = (length, key, site)
    return None if best is None else (best[1], best[2])


def place_on_desk(ctx, target_xy):
    """Lower the held fax onto the desk at @target_xy (M) and let go."""
    world = ctx.base_matrix(); ci, cj = ctx.scene.cell(np.asarray(target_xy, float)); desk_z = float(ctx.scene.top[ci, cj])
    target = np.linalg.inv(world) @ np.r_[target_xy, desk_z, 1.]
    # Height of the fingertips above the fax's underside, as grasped.
    grip = ctx.body.fk(ctx.q)[0][ctx.ik.link] @ ctx.held
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * FAX_HALF + FAX_CENTER
    bottom = float((corners @ grip[:3, :3].T + grip[:3, 3])[:, 2].min()); tip_over_bottom = float(ctx.tip()[2, 3] - bottom)
    free = np.array([0., 0., 1.]); rotation = ctx.tip()[:3, :3]; reached = False
    # The chair that stood in the nook has been pushed aside on the way in: plan against the fixed
    # furniture only, and keep the arm and the fax above chair-back height on the way over. If a
    # spot is out of reach, or the arm is stopped on the way to it, nearby spots on the same desk
    # are tried in turn.
    chair_top = .86
    check = ctx.validator(payload=True, static_only=True, over=chair_top)
    for forward, lateral in ((0., 0.), (-.06, 0.), (.05, 0.), (0., .08), (0., -.08), (-.10, .08), (-.10, -.08)):
        goal_point = np.array([target[0] + forward, target[1] + lateral, target[2] + tip_over_bottom + .012])
        spot = world @ np.r_[goal_point[:2], 0., 1.]; si, sj = ctx.scene.cell(spot[:2])
        if not ctx.scene.desk[si, sj]:
            continue
        above = ctx.ik.solve(ctx.q, goal_point + [0, 0, .10], rotation, count=30, validator=check, free_axis=free)
        if above is None:
            continue
        status = yield from move_to(ctx, above, payload=True, label='place', static_only=True, over=chair_top)
        if status == 'done':
            reached = True; break
    if not reached:
        ctx.event('place', ok=False, reason='no reachable spot above the desk', target_in_base=target[:3].round(3).tolist())
        return False                          # keep holding the fax rather than dropping it somewhere
    down = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], goal_point, ctx.tip()[:3, :3], step=.02)
    status = 'skipped'
    if down is not None:
        status = yield from follow(ctx, down, rate=.012, max_steps=250)
    yield from hold(ctx, 8)
    ctx.released = ctx.body.fk(ctx.q)[0][ctx.ik.link] @ ctx.held      # where the fax is left, in the base frame
    ctx.released[2, 3] = target[2] + (grip[2, 3] - bottom)            # ... resting on the desk, not hovering above it
    ctx.set_grip(1.)
    yield from hold(ctx, 25)
    ctx.held = None
    up = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], ctx.tip()[:3, 3] + [0, 0, .12], ctx.tip()[:3, :3], step=.03)
    if up is not None:
        yield from follow(ctx, up, rate=.02)
    ctx.event('place', ok=True, descend=status, desk_z=desk_z, tip=ctx.tip()[:3, 3].round(3).tolist())
    return True
