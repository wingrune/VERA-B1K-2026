"""Cleaning the trumpet: what the robot does, from its own estimates (trumpet_observed.TrumpetEstimator).

The scrub brush takes the dust's visual particles off whatever lies within its own box (axis-aligned, round its visual
mesh) grown by 2 cm on every side (OmniGibson ParticleRemover, adjacency): it need not touch them. Wiped along the
trumpet flat, the brush stays over the highest part under it and the dust in the hollows beside it stays (offline,
over the 30 dev starts, at most a third of the starts left clean however close the lanes). Held by one end and tilted
TILT from upright, the brush's box reaches from 2 cm under its lower end to over 10 cm above it, and that end alone
goes down into a low place beside a speck: 10-13 such dabs took every particle of every dev start, offline, and with
the simulator's true poses (dev/jobs/dab_trial.py) 18-19 of 19-20 went in some 1300 steps.

1. Look at the desk from the room (the trumpet's pose, the brush, the specks).
2. Take the brush by its end nearer the base, from above, from a stance north of the desk (the stool in front of the
   desk's middle keeps the base 0.8 m off there).
3. Turn it over the desk's near edge, high, away from the trumpet.
4. Rounds: the hand up and aside, looks; dabs for the specks known, the lowest that keep the whole brush CLEAR over
   the trumpet's collision surface (priors/trumpet_fields.npz, posed as last fitted); moves between dabs keep the
   brush SAFE over the trumpet's highest point.
"""
import numpy as np
from manip import hold, follow, move_to, drive, BASE_Z, SCENE_MARGIN, pose2
from car_skills import tuck
from car_pipeline import straight, settle
from skills import interpolate_cartesian
import decor

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
SOUTH = -np.pi / 2
# (where the desk shows (map), facing it; from east of it the brush shows beside the bell that hid it from the north, 1)
# (and close by the desk's ends, looking down steeply: behind the trumpet, 0.17 m off its far side, the brush was hidden
# from all the others, 317)
VANTAGES = ((-5.25, 1.30), (-5.75, 1.10), (-4.75, 1.10), (-5.25, 1.80), (-5.85, 0.90), (-4.65, 0.90), (-4.0, 0.45), (-4.0, 1.0), (-5.85, 0.72), (-4.65, 0.72))
DESK_TOP = .7734
GRIP = .025                         # the fingers take the brush this far from its middle (the other end goes down)
TILT = .6                           # the brush this far from upright, its lower end towards the base
CLEAR = .006                        # the brush this high over the trumpet's collision surface
DIL = .006                          # ... grown by this on every side (the pose's error)
# (tilt from upright, turn of the fingers' heading from south), the first that takes a speck: offline over the 30 dev
# starts with true poses one attitude left 0.33 particles a start out of reach, these nine 0.13 (87 % of the starts
# clean); out of reach were dust by the valves' block, whose collision hulls stand high
# (and turned further, for what lies low in the bell's mouth: with the bell east or west a turn of 1.2-1.8 took the
# particle no other attitude did, ab1/ac1 306, 310)
ATTITUDES = ((.6, 0.), (.6, .6), (.6, -.6), (.4, 0.), (.8, 0.), (.4, .6), (.4, -.6), (.8, .6), (.8, -.6), (.6, -1.2), (.6, 1.2), (.6, -1.8), (.6, 1.8))
SAFE = .035                         # moves between dabs: the brush's lowest point this high over the trumpet's top
L, W, H, M = .127, .036, .021, .02  # the brush's length, width, height; the remover's margin
# (where the brush sits in the hand beyond what the plan says, tool frame: the simulator's pose of the brush against the
# tool's at the bottom of 149 dabs of 5 dev runs, the base's true pose: 1.1-1.65 cm further towards its far end in all,
# within 1 cm across and 0.7 cm deeper in the hand; across, the wrist camera tells)
IN_HAND = .013                      # (along the brush, towards its far end)
LOOK_HAND = (.25, .35, 1.25)        # the hand up and aside while the head camera looks (base frame: ahead, aside, height)
LIMIT = 7850                        # no new dab after this step (the horizon is 7960)
GLIDE = .35                         # the next dab this near, the brush glides there low over the trumpet instead of up, across and down
GLIDE_UP = .006                     # ... that much higher than a dab's bottom (3 mm before: the glides pushed the trumpet, ac1)
PUSHED = .03                        # the trumpet moved this far (m; three times that in rad) since the round was planned: planned again
# (1 cm was within what the look with the hand by the trumpet scatters: 9-27 new rounds a run for nothing, ae1)
BLIND_FROM = 4000                   # the blind dabs (with the specks seen) from this step on at the latest
PER_ROUND = 16                      # blind dabs a round (a round, the hand up and a look, took 130 steps; 25 rounds a run)
FAST = .045                         # joint rate up from a dab and over to the next (0.03 before; the way down stays 0.015)
INNER = np.array([.006, .006, .003])  # a target this far inside the box (offline over the 30 dev starts with true poses: 1 cm across left 7 particles a start out of reach, 6 mm 0.6)


def rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def ry(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1., 0], [-s, 0, c]])


def in_base(base, p):
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(p, float)[:2] - [x, y]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def go(ctx, est, goal, yaw, **options):
    goal = np.asarray(goal, float)[:2]; here = np.asarray(est.base()[:2], float)
    if ctx.scene.path(here, goal) is None:
        ctx.event('route', ok=False, goal=np.round(goal, 2).tolist()); return 'no_route'
    status = yield from drive(ctx, goal, yaw, lookahead=.25, **options)
    return status


def turn_to(ctx, est, yaw):
    here = est.base()
    status = yield from drive(ctx, here[:2], yaw, direct=True, face_travel=False, tolerance=.08, yaw_tolerance=.03, max_steps=500)
    return status


def looks(ctx, est, n=3, dust=True):
    yield from settle(ctx, est, 8)
    out = None
    for _ in range(n):
        yield from hold(ctx, 3)
        out = est.look(dust=dust)
    return out


class Fields:
    """The trumpet's collision surface (its own frame, priors/trumpet_fields.npz) over map points, for a pose."""

    def __init__(self, path):
        f = np.load(path); self.lo = f['lo']; self.res = float(f['res']); self.top = f['collision_top']

    def surface(self, T, pts, dil=DIL):
        """Highest collision surface of the trumpet posed @T over the map points @pts (N, 2), grown by @dil; the desk
        where there is none."""
        Ti = np.linalg.inv(T); out = np.full(len(pts), DESK_TOP)
        for dx in (-dil, 0., dil):
            for dy in (-dil, 0., dil):
                q = np.c_[pts + [dx, dy], np.full(len(pts), T[2, 3]), np.ones(len(pts))] @ Ti.T
                i = np.floor((q[:, 0] - self.lo[0]) / self.res).astype(int); j = np.floor((q[:, 1] - self.lo[1]) / self.res).astype(int)
                ok = (i >= 0) & (i < self.top.shape[0]) & (j >= 0) & (j < self.top.shape[1])
                h = np.full(len(pts), np.nan); h[ok] = self.top[i[ok], j[ok]]; m = np.isfinite(h)
                if m.any():
                    w = np.c_[q[m, 0], q[m, 1], h[m], np.ones(m.sum())] @ T.T
                    out[m] = np.maximum(out[m], w[:, 2])
        return out


def body_clear(ctx, pose, chassis=.03):
    """The base at @pose (map) clear of what the map has, the arms as they are now; the chassis by @chassis (the base
    stands a few centimetres off where it was sent, and by the stool the arm planner then found the chassis in it and
    no way for the arm, three times, 301)."""
    poses = ctx.body.fk(ctx.q)[0]; w = pose2(*pose); radii = ctx.collision.radii; ch = ctx.base_part
    c = ctx.collision.centers(poses) @ w[:3, :3].T + w[:3, 3]
    g = ctx.scene.sphere_clearance(c, radii, ignore_below=np.where(ch, .06, 0.))
    return bool(np.all(g[ch] >= chassis)) and bool(np.all(g[~ch] >= SCENE_MARGIN))


def stances(ctx, x, spread=.45, target=None):
    """Base poses facing south over the desk, the nearest to @x first; for a @target (map xy, the brush) also turned
    towards it, the nearest to it first (by the stool, 0.8 m off the desk's middle, the brush on the desk's far side
    was 0.97 m off the stance straight north of it and out of reach, while beside the stool, turned, 0.9 m, 314, 316)."""
    out = []
    for y in np.arange(.68, .96, .03):
        for dx in np.arange(-spread, spread + 1e-6, .05):
            yaws = [SOUTH]
            if target is not None:
                to = float(np.arctan2(target[1] - y, target[0] - (x + dx)))
                yaws += [to] if abs(wrap(to - SOUTH)) > .05 and abs(wrap(to - SOUTH)) < .6 else []
            for yaw in yaws:
                pose = (float(x + dx), float(y), float(yaw))
                # (and the chassis clear 2 cm off it every way: the map's room round the stool falls from 6 cm to -9 cm
                # within 5 mm, and the base stands up to 1.5 cm off where it was sent, af1 301)
                if body_clear(ctx, pose) and all(chassis_clear(ctx, (pose[0] + a, pose[1] + b, pose[2]), 0.) for a, b in ((.02, 0.), (-.02, 0.), (0., -.02))):
                    cost = abs(dx) + 2. * (y - .68) if target is None else float(np.hypot(target[0] - pose[0], target[1] - pose[1])) + .3 * abs(wrap(yaw - SOUTH))
                    out.append((cost, pose))
    return [p for _, p in sorted(out)]


def stand(ctx, est, pose):
    """Get the base to @pose by the desk: a route to a place north of it the route planner reaches, then straight."""
    here = np.asarray(est.base()[:2], float)
    if float(np.linalg.norm(here - pose[:2])) < .03 and abs(wrap(est.base()[2] - pose[2])) < .05:
        return 'arrived'
    if float(np.linalg.norm(here - pose[:2])) > .6 or here[1] > pose[1] + .5:
        staging = np.array([pose[0], pose[1] + .40])
        for _ in range(4):
            if ctx.scene.base_free(staging, margin=-.05):
                break
            staging = staging + [0., .1]
        status = yield from go(ctx, est, staging, pose[2], tolerance=.06)
        if status not in ('arrived',) and float(np.linalg.norm(np.asarray(est.base()[:2]) - staging)) > .15:
            return status
        yield from turn_to(ctx, est, pose[2])
    status = yield from straight(ctx, pose[:2], pose[2], speed=.15, tolerance=.015)
    yield from settle(ctx, est, 6)
    for _ in range(2):
        here = est.base()
        if chassis_clear(ctx, here):
            break
        # (stood, as the localizer reads it, a centimetre or two into the stool as the map has it: every arm plan then
        # fails, three attempts at the brush, ac1-af1 301; a few centimetres back north, away from the desk)
        ctx.event('stand', nudged=True, at=np.round(here, 3).tolist())
        status = yield from straight(ctx, np.array([here[0], here[1] + .03]), pose[2], speed=.1, tolerance=.01)
        yield from settle(ctx, est, 6)
    return status


def chassis_clear(ctx, pose, margin=.01):
    """The chassis at @pose (map) this far from what the map has (the arm planner wants it clear of it)."""
    poses = ctx.body.fk(ctx.q)[0]; w = pose2(*pose); ch = ctx.base_part
    c = ctx.collision.centers(poses)[ch] @ w[:3, :3].T + w[:3, 3]
    g = ctx.scene.sphere_clearance(c, ctx.collision.radii[ch], ignore_below=np.full(int(ch.sum()), .06))
    return bool(np.all(g >= margin))


def find(ctx, est):
    """The trumpet and the brush on the desk: looks from the vantages, nearest first, till both are known."""
    left = [v for v in VANTAGES]
    while left and (est.trumpet is None or est.brush is None):
        here = np.asarray(est.base()[:2]); left.sort(key=lambda v: float(np.hypot(v[0] - here[0], v[1] - here[1]))); spot = left.pop(0)
        yaw = float(np.arctan2(-.0 - spot[1], -5.25 - spot[0]))
        if spot[1] < .8:
            # (by the desk the route planner's disc does not stand: there by a route to north of it, then straight)
            if est.trumpet is not None:
                yaw = float(np.arctan2(est.trumpet['y'] - spot[1], est.trumpet['x'] - spot[0]))
            pose = (spot[0], spot[1], yaw)
            if not body_clear(ctx, pose):
                continue
            yield from stand(ctx, est, pose)
        else:
            status = yield from go(ctx, est, spot, yaw, tolerance=.10)
        yield from turn_to(ctx, est, yaw)
        yield from looks(ctx, est, 3)
        ctx.event('find', at=list(spot), trumpet=est.trumpet is not None, brush=est.brush is not None, specks=len(est.specks))
    if est.trumpet is not None and est.brush is None and TRUMPET_PTS is not None:
        yield from search_behind(ctx, est)
    return est.trumpet is not None and est.brush is not None


def search_behind(ctx, est):
    """The brush hidden from all the vantages by the trumpet (0.17 m off its far side, 317, in all rounds): the wrist
    camera, the hand held high over the trumpet's near side and turned to look down past it."""
    T = est.T(); P = (np.c_[TRUMPET_PTS, np.ones(len(TRUMPET_PTS))] @ T.T)[:, :3]
    north = float(P[:, 1].max()) + .04; tx = float(T[0, 3])
    options = stances(ctx, tx, spread=.3)
    if not options:
        ctx.event('search', ok=False, reason='no stance'); return
    yield from stand(ctx, est, options[0]); base = est.base(); used = None
    trumpet_obstacle(ctx, T, False)
    for dx in (0., -.22, .22):
        spot = np.r_[tx + dx, north]; arm = 'left' if dx > 0 else 'right'
        if used is not None and used != arm:
            yield from tuck(ctx, arms=(used,))
        ctx.arm = arm; plan = None
        for z, a in ((DESK_TOP + .40, .7), (DESK_TOP + .32, .85), (DESK_TOP + .45, .55)):
            plan = decor.plan_pinch(ctx, np.r_[in_base(base, spot), z - BASE_Z], ry(-a) @ decor.top_rotation(np.pi / 2), way_in=.01, hand_margin=-.03, count=24)
            if plan is not None:
                break
        if plan is None:
            ctx.event('search', at=np.round(spot, 3).tolist(), arm=arm, ok=False, reason='no plan'); continue
        status = yield from move_to(ctx, plan['grip'], rate=.05, label='search'); used = arm
        yield from hold(ctx, 6)
        found = est.wrist_search(arm)
        ctx.event('search', at=np.round(spot, 3).tolist(), arm=arm, status=status, seen=found is not None, debug=getattr(est, 'wrist_debug', None),
                  brush=None if found is None else dict(xy=np.round(found['xy'], 3).tolist(), length=round(found['length'], 3)))
        if found is not None:
            est.brush = dict(found, step=est.ob.steps); break
    if used is not None:
        yield from tuck(ctx, arms=(used,))
    ctx.obstacles.pop('trumpet', None)


class Held:
    """The brush in the hand: its box corners and underside in the tool frame, from where it was taken."""

    def __init__(self, centre, axis, rotation, point):
        """@centre (map xyz) and @axis (unit, map, towards the end the fingers took) of the brush lying on the desk;
        @rotation, @point: the tool's attitude and point (map) when the fingers closed."""
        u = np.r_[axis[:2], 0.]; u /= np.linalg.norm(u); n = np.array([-u[1], u[0], 0.]); z = np.array([0., 0., 1.])
        R = rotation; mid = R.T @ (np.asarray(centre, float) - point)
        self.far = R.T @ -u; wid = R.T @ n; hei = R.T @ z
        self.corners = np.array([mid + a * (L / 2) * self.far + b * (W / 2) * wid + c * (H / 2) * hei for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        self.under = np.array([mid + a * (L / 2) * self.far + b * (W / 2) * wid + c * (H / 2) * hei for a in np.linspace(-1, 1, 12) for b in (-1, 0, 1) for c in (-1, 1)])
        self.mid = mid; self.wid = wid; self.hei = hei; self._shape()

    def _shape(self):
        mid, wid, hei = self.mid, self.wid, self.hei
        self.corners = np.array([mid + a * (L / 2) * self.far + b * (W / 2) * wid + c * (H / 2) * hei for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        self.under = np.array([mid + a * (L / 2) * self.far + b * (W / 2) * wid + c * (H / 2) * hei for a in np.linspace(-1, 1, 12) for b in (-1, 0, 1) for c in (-1, 1)])
        # (and the hand round the brush: the fingers and the palm, tool frame (z back towards the palm, y along the
        # closing), kept 1 cm further off the trumpet: the gripper and the wrist caught the trumpet and dragged it off
        # the desk or 12-16 cm along it in 5 of 10 runs, y1, z1)
        hand = np.array([[x, y, z] for x in (-.03, .03) for y in (-.045, 0., .045) for z in (0., .04, .08, .12)])
        self.clear_pts = np.vstack([self.under, hand]); self.clear_extra = np.r_[np.zeros(len(self.under)), np.full(len(hand), .010)]

    def need(self, R):
        """Points to keep clear (relative to the fingers, map axes, for the tool's attitude @R) and the room each."""
        return (R @ self.clear_pts.T).T, self.clear_extra

    def correct(self, shift):
        """The brush is @shift (tool frame) off where it was believed."""
        self.mid = self.mid + np.asarray(shift, float); self._shape()

    def tilted(self, heading=SOUTH, tilt=None):
        """Tool rotation (map): the fingers along @heading and down by @tilt (TILT), the brush's far end down and
        towards the base by as much from upright."""
        tilt = TILT if tilt is None else tilt
        a = np.array([np.cos(heading) * np.cos(tilt), np.sin(heading) * np.cos(tilt), -np.sin(tilt)])
        d = np.array([0., 0., -1.]) - (-np.sin(tilt)) * a; d /= np.linalg.norm(d)
        A = np.c_[np.array([0., 0., -1.]), self.far, np.cross([0., 0., -1.], self.far)]
        Bm = np.c_[a, d, np.cross(a, d)]
        U, _, Vt = np.linalg.svd(Bm @ A.T); R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        return R


def tool_world(ctx, est):
    base = est.base(); m = ctx.tip(); R = rz(base[2])
    return R @ m[:3, 3] + [base[0], base[1], BASE_Z], R @ m[:3, :3]


def why_no_plan(ctx, point, rotation):
    """For the event log: is there an IK solution at all, and what of the robot the validator finds in the way."""
    sol = ctx.ik.solve(ctx.q, point, rotation, count=32)
    if sol is None:
        return 'no ik'
    q = ctx.q.copy(); q[ctx.ik.columns] = sol; poses = ctx.body.fk(q)[0]
    out = dict(valid=bool(ctx.validator(hand_margin=-.03)(q, poses)), own=bool(ctx.collision.valid(poses, None)))
    world = ctx.base_matrix(); centres = ctx.collision.centers(poses) @ world[:3, :3].T + world[:3, 3]
    gaps = ctx.scene.sphere_clearance(centres, ctx.collision.radii, ignore_below=np.where(ctx.base_part, .06, 0.))
    out['nearest'] = [(ctx.collision.names[k], round(float(gaps[k]), 3), np.round(centres[k], 2).tolist()) for k in np.argsort(gaps)[:3]]
    out['obstacles'] = list(ctx.obstacles.keys()); out['payload'] = list(ctx.payload.keys()); out['guard'] = len(ctx.guard)
    return out


def trumpet_obstacle(ctx, T, touch):
    """The trumpet posed @T as a thing lying about for the arm planner; touch: the working hand may touch it."""
    ctx.obstacles['trumpet'] = ((np.c_[TRUMPET_PTS, np.ones(len(TRUMPET_PTS))] @ T.T)[:, :3], np.full(len(TRUMPET_PTS), .012), touch)


def pre_clear(ctx, plan, base=None):
    """The start of a pinch @plan (the hand 10 cm over the grip) clear of the trumpet, the hand too: the way there is
    planned so (reaching for a brush behind the trumpet, with the hand let touch it, the hand swept it 20-30 cm off and
    the brush with it, 311, 316)."""
    if 'trumpet' not in ctx.obstacles:
        return True
    pts, size, touch = ctx.obstacles['trumpet']; ctx.obstacles['trumpet'] = (pts, size, False)
    try:
        full = ctx.q.copy(); full[ctx.ik.columns] = plan['pre']
        return bool(ctx.validator(base=base, hand_margin=-.03)(full, ctx.body.fk(full)[0]))
    finally:
        ctx.obstacles['trumpet'] = (pts, size, touch)


def take_brush(ctx, est):
    """The brush from the desk, by its end nearer the base, from above. Returns (arm, Held) or None."""
    failed = []                         # (stances from which there was no way to the brush, as the base stood there)
    for attempt in range(3):
        if est.T() is not None and TRUMPET_PTS is not None:
            # (the trumpet in the arm planner's way: reaching for a brush behind it the arm pushed it 10 cm, 316)
            # (the hand itself may come near it: the brush can lie a few centimetres off the trumpet, 301)
            trumpet_obstacle(ctx, est.T(), True)
        b = est.brush
        if b is None:
            return None
        centre = np.r_[b['xy'], DESK_TOP + .012]; axis = np.asarray(b['direction'], float)
        plan = None; chosen = None; first = None
        for pose in [p_ for p_ in stances(ctx, float(b['xy'][0]), target=b['xy']) if all(np.hypot(p_[0] - f[0], p_[1] - f[1]) > .06 for f in failed)][:12]:
            u = axis * (1. if axis @ (np.asarray(pose[:2]) - b['xy']) > 0 else -1.)       # (towards the base)
            grip = centre + GRIP * np.r_[u, 0.]
            closing = float(np.arctan2(u[0], -u[1])) - pose[2]                         # (across the brush)
            for arm in ('right', 'left'):
                ctx.arm = arm
                plan = decor.plan_pinch(ctx, np.r_[in_base(pose, grip), grip[2] - .0006 - BASE_Z], decor.top_rotation(closing), base=pose2(*pose), way_in=.10, hand_margin=-.03, count=32)
                if plan is not None:
                    first = first or (pose, arm, u, closing)
                    # (after a stance that gave no way as the base stood, the first with a way, its start clear or not:
                    # one of 0.9 m reach chosen for its clear start had none 1 cm off it, three times, 308)
                    if failed or pre_clear(ctx, plan, pose2(*pose)):
                        chosen = (pose, arm, u, closing); break
                    plan = None
            if plan is not None:
                break
        if chosen is None:
            # (none whose start is clear of the trumpet: the first, the hand let touch it on the way)
            chosen = first
        if chosen is None:
            ctx.event('stance', ok=False, thing='brush'); return None
        pose, arm, u, closing = chosen; chosen_brush = dict(b)
        ctx.event('stance', thing='brush', arm=arm, pose=np.round(pose, 3).tolist())
        status = yield from stand(ctx, est, pose)
        # (from here, near: the brush again)
        yield from looks(ctx, est, 2, dust=False)
        b = est.brush; centre = np.r_[b['xy'], DESK_TOP + .012]; axis = np.asarray(b['direction'], float)
        # (the same end as the stance was chosen for: a brush lying across the base's heading has both ends as near,
        # and the new look's axis, the other way round, put the grip at the other end, where the arm had no way, 8)
        base = est.base(); u_old = u; u = axis * (1. if axis @ u_old > 0 else -1.)
        ctx.arm = arm; plan = None; first = None
        for u_ in (u, -u):
            grip = centre + GRIP * np.r_[u_, 0.]; closing = float(np.arctan2(u_[0], -u_[1])) - base[2]
            plan = decor.plan_pinch(ctx, np.r_[in_base(base, grip), grip[2] - .0006 - BASE_Z], decor.top_rotation(closing), way_in=.10, hand_margin=-.03, count=48)
            if plan is not None:
                first = first or (plan, u_, grip, closing)
                if pre_clear(ctx, plan):
                    u = u_; break
                plan = None
        if plan is None and first is not None:
            plan, u, grip, closing = first
        if plan is None:
            # (the brush as seen from the room, where the stance was chosen for it: the closer look's brush gave no way
            # three times over, 301)
            b0 = chosen_brush; centre = np.r_[b0['xy'], DESK_TOP + .012]; u = u_old
            grip = centre + GRIP * np.r_[u, 0.]; closing = float(np.arctan2(u[0], -u[1])) - base[2]
            plan = decor.plan_pinch(ctx, np.r_[in_base(base, grip), grip[2] - .0006 - BASE_Z], decor.top_rotation(closing), way_in=.10, hand_margin=-.03, count=48)
            ctx.event('brush', ok=plan is not None, reason='no plan for the near look', near=dict(xy=np.round(b['xy'], 3).tolist(), direction=np.round(b['direction'], 3).tolist()),
                      far=dict(xy=np.round(b0['xy'], 3).tolist(), direction=np.round(b0['direction'], 3).tolist()))
        if plan is None:
            ctx.event('brush', ok=False, reason='no plan from the stance', why=why_no_plan(ctx, np.r_[in_base(base, grip), grip[2] - .0006 - BASE_Z], decor.top_rotation(closing)))
            failed.append(pose); continue
        # (a last look from over it with the wrist camera: the head camera's middle of the brush was 1-2 cm off along
        # it, and the fingers closed past its end, 301, 302)
        # (over to the plan's start round the trumpet, the hand too; the way in, down, may touch it)
        pts_, size_, _ = ctx.obstacles.get('trumpet', (None, None, None))
        if pts_ is not None:
            ctx.obstacles['trumpet'] = (pts_, size_, False)
        status = yield from move_to(ctx, plan['pre'], rate=.04, label='brush')
        if pts_ is not None:
            ctx.obstacles['trumpet'] = (pts_, size_, True)
        yield from hold(ctx, 4)
        seen = est.wrist_brush(arm, ctx.tip()[:3, 3]) if status == 'done' else None
        if seen is not None:
            mid_b, ax_b, length = seen
            to_base = -in_base(base, b['xy']); ub = ax_b * (1. if ax_b @ to_base > 0 else -1.)    # (base frame, towards the base)
            g_b = mid_b + GRIP * ub; closing_b = float(np.arctan2(ub[0], -ub[1]))
            better = decor.plan_pinch(ctx, np.r_[g_b, grip[2] - .0006 - BASE_Z], decor.top_rotation(closing_b), way_in=.10, hand_margin=-.03, count=48)
            ctx.event('brush_look', shift=np.round(g_b - in_base(base, grip), 3).tolist(), length=round(length, 3), replanned=better is not None)
            if better is not None:
                plan = better; closing = closing_b + base[2] - base[2]
                c, s_ = np.cos(base[2]), np.sin(base[2])
                grip = np.r_[base[0] + c * g_b[0] - s_ * g_b[1], base[1] + s_ * g_b[0] + c * g_b[1], grip[2]]
                mid_w = np.r_[base[0] + c * mid_b[0] - s_ * mid_b[1], base[1] + s_ * mid_b[0] + c * mid_b[1]]
                centre = np.r_[mid_w, centre[2]]; u = np.array([c * ub[0] - s_ * ub[1], s_ * ub[0] + c * ub[1]])
        else:
            ctx.event('brush_look', seen=False, debug=getattr(est, 'wrist_debug', None))
        # (the fingers that stopped on the brush are 3.6 cm apart in all; 9 cm apart after the lift the brush had gone,
        # and 14 dabs took nothing, 0)
        # (and 3.7-4.0 cm in all 11 takes of the brush; 4.6 cm on a tube of the trumpet beside it, taken instead at the
        # arm's reach, 311)
        ok = yield from decor.pinch(ctx, plan, label='brush', gap=.033, lift=.10, most=.043)
        if ok:
            R = rz(base[2]) @ decor.top_rotation(closing)
            held = Held(centre, u, R, grip)
            # (the two fingers do not close alike: one went to 0 and the other stopped at 0.038 on the brush, and the
            # brush sat 1.9 cm off the hand's middle line along the closing axis, half their difference; the plan's
            # place for it then missed the specks it was let down for, 8, 124)
            f = ctx.fingers(); held.correct(np.array([0., (float(f[0]) - float(f[1])) / 2, .007]) + IN_HAND * held.far)
            ctx.event('taken', thing='brush', arm=arm, mid=np.round(held.mid, 4).tolist(), fingers=np.round(f, 4).tolist())
            return arm, held
        yield from tuck(ctx, arms=(arm,))
        yield from looks(ctx, est, 2, dust=False)
    return None


def plan_attitudes(fields, T, targets, held, most=40, attitudes=None):
    """plan_dabs over ATTITUDES: the first attitude for all targets, the next ones for what it cannot take:
    [(fingers, [indices], R_t)], [indices no attitude takes]. (Targets left over for the number of dabs are neither.)"""
    targets = np.asarray(targets, float).reshape(-1, 3); left = list(range(len(targets))); out = []
    for tilt, turn in (ATTITUDES if attitudes is None else attitudes):
        if not left:
            break
        R_t = held.tilted(SOUTH + turn, tilt)
        dabs, capped, unreach = plan_dabs(fields, T, targets[left], held, R_t, most)
        out += [(fp, [left[i] for i in took], R_t) for fp, took in dabs]
        left = [left[i] for i in unreach]
    return out, left


def plan_dabs(fields, T, targets, held, R_t, most=40):
    """Greedy: fingers' points (map) whose boxes take @targets (xyz, map), each the lowest that keeps the whole brush
    CLEAR over the trumpet and takes the most: [(fingers, [indices])], [indices left for the number of dabs],
    [indices of the targets no placement takes]."""
    cw = (R_t @ held.corners.T).T; uw, extra = held.need(R_t)
    box_lo = cw.min(0) - M; box_hi = cw.max(0) + M; span = box_hi[:2] - box_lo[:2]
    targets = np.asarray(targets, float).reshape(-1, 3)
    left = list(range(len(targets))); out = []; unreach = []
    while left and len(out) < most:
        best = None; takes = set()
        for k in left[:10]:
            p = targets[k]
            for fx in np.linspace(.1, .9, 5):
                for fy in np.linspace(.1, .9, 5):
                    f_xy = p[:2] - (box_lo[:2] + np.array([fx, fy]) * span)
                    z = float((fields.surface(T, uw[:, :2] + f_xy) + CLEAR + extra - uw[:, 2]).max())
                    fp = np.r_[f_xy, z]; rel = targets - fp
                    take = np.all((rel > box_lo + INNER) & (rel < box_hi - INNER), axis=1)
                    if take[k]:
                        takes.add(k)
                    else:
                        continue
                    tk = [i for i in left if take[i]]
                    if best is None or len(tk) > len(best[1]) or (len(tk) == len(best[1]) and z < best[0][2]):
                        best = (fp, tk)
        # (the ones among these no placement takes: out of this attitude's reach; the rest go on being planned)
        unreach += [k for k in left[:10] if k not in takes]
        left = [i for i in left if i not in unreach]
        if best is not None:
            out.append(best); left = [i for i in left if i not in best[1]]
    return out, left, unreach


def clean(ctx, est, arm, held):
    fields = Fields(FIELDS)
    # (the trumpet's upper surface, its own frame, every 1 cm: what the dabs' boxes have passed over is kept; when the
    # head camera shows no more specks, the rest of it is dabbed blind while there is time: the specks it does not
    # show, under the far side's tubes or too faint, were 4-10 of the 20 left in four of six runs, o1)
    surf = est.surface_dab; covered = np.zeros(len(surf), bool); blind = False; unreach = np.zeros(len(surf), bool); passes = 0
    R_t = held.tilted(SOUTH)              # (the attitude the hand has now)
    lowest = lambda R: float((R @ held.corners.T).T[:, 2].min())
    est.holding_brush = True; ctx.arm = arm; done = 0; empty = 0; sides = []

    def hand_up():
        """The hand up and aside, out of the head camera's view of the desk (straight up first)."""
        base = est.base(); side = 1. if arm == 'left' else -1.
        t = ctx.tip()[:3, 3]; R_b = rz(-base[2]) @ R_t
        up = interpolate_cartesian(ctx.ik, ctx.q, t, np.r_[t[:2], max(t[2], LOOK_HAND[2] - BASE_Z)], R_b, step=.03)
        if up is not None:
            yield from follow(ctx, up, rate=.04, max_steps=200)
        p = decor.plan_pinch(ctx, np.r_[LOOK_HAND[0], side * LOOK_HAND[1], LOOK_HAND[2] - BASE_Z], R_b, way_in=.01, hand_margin=-.03, count=24)
        if p is not None:
            yield from move_to(ctx, p['grip'], rate=.05, label='hand_up')
        tw, _ = tool_world(ctx, est); est.hand = tw

    def glide_path(a, b, R_d, offset):
        """Fingers' points (map) from @a to @b, every 1 cm, each the lowest that keeps the whole brush (@R_d) CLEAR
        over the trumpet there (+@offset: the hand's attitude as it is)."""
        uw, extra = held.need(R_d); n = max(2, int(np.ceil(np.linalg.norm(b[:2] - a[:2]) / .01)) + 1); out = []
        for t in np.linspace(0., 1., n)[1:]:
            xy = a[:2] + t * (b[:2] - a[:2])
            out.append(np.r_[xy, float((fields.surface(T, uw[:, :2] + xy) + CLEAR + GLIDE_UP + extra - uw[:, 2]).max()) + offset])
        return out

    idle = 0; few = 0
    measured = False
    bottom = None                       # (fingers, attitude, safe height, offset): the brush down by the trumpet after a dab

    def measure(held):
        """Where the held brush's lowest part is against where the model has it (tool frame shift), from the head
        camera over the trumpet: two looks that agree within 6 mm, or None. (The brush sits in the fingers up to 3.5
        cm off where the plan and the fingers say, along their closing, 8, 59; the head camera sees the lower end, which
        is what goes down by the speck.)"""
        yield from hold(ctx, 4); seen = []
        for _ in range(3):
            yield from hold(ctx, 3)
            tw, Rw = tool_world(ctx, est); end = est.brush_end(tw)
            if end is None:
                continue
            cw = (Rw @ held.corners.T).T + tw; lowest = cw[np.argsort(cw[:, 2])[:2]].mean(axis=0)
            seen.append((Rw.T @ (end[0] - lowest), end[1]))
        if len(seen) < 2:
            ctx.event('in_hand', seen=len(seen)); return None
        a = np.array([v for v, _ in seen]); spread = float(np.max(np.linalg.norm(a - a.mean(axis=0), axis=1)))
        shift = a.mean(axis=0)
        # (logged only: against the simulator's pose of the brush this look moved the model 1-2 cm the wrong way, 124;
        # the wrist camera of the holding hand, after the turn, is what corrects it)
        ctx.event('in_hand', shift=np.round(shift, 4).tolist(), spread=round(spread, 4), points=[n for _, n in seen], used=False)
        return None

    for round_ in range(60):
        if est.ob.steps > LIMIT or idle >= 3:
            break
        measured = True                  # (the head camera's look at the brush in the hand is no longer taken: the wrist camera's is)
        # (the hand up and a look each round while specks are seen; blind, every other round: the look is for the
        # trumpet's pose and for specks the dabs have not yet passed over)
        if not blind or round_ % 3 == 0:
            yield from hand_up(); bottom = None
            yield from looks(ctx, est, 2)
        else:
            # (the trumpet's pose at least, with the hand where it is: the dabs push it)
            tw, _ = tool_world(ctx, est); est.hand = tw
            yield from settle(ctx, est, 4); est.look(dust=False)
        targets = est.speck_points()
        T = est.T()
        if T is not None:
            # (the trumpet as a thing lying about for the arm planner, as it lies now: the joint moves of the arm, up to
            # the look and over to a dab, swept the wrist through it, y1 302)
            trumpet_obstacle(ctx, T, False)
        ctx.event('round', n=round_, specks=len(targets), trumpet=None if est.trumpet is None else [round(est.trumpet['x'], 3), round(est.trumpet['y'], 3), round(est.trumpet['yaw'], 3)])
        if not blind and round_ >= 2 and 0 < len(targets) <= 2:
            # (one or two specks a round: the blind dabs, those specks first; a round to dab two specks and look again
            # took 150-200 steps, and 20 such rounds, the specks seen anew each time a few centimetres off, left the
            # surface 3000 steps for the blind dabs too few, 304)
            few += 1
            if few >= 2:
                blind = True; ctx.event('blind', covered=round(float(covered.mean()), 3), reason='few specks')
        if not blind and est.ob.steps > BLIND_FROM:
            # (specks seen anew round after round, a few centimetres off the ones dabbed: 4-9 a round for 14 rounds while
            # 3 particles were left, the blind dabs from step 6500, af1 306)
            blind = True; ctx.event('blind', covered=round(float(covered.mean()), 3), reason='time')
        if not targets and not blind:
            # (nothing left that this stance shows: a look from the other end of the desk, then the blind dabs)
            empty += 1
            here = est.base()
            if T is None:
                break
            x = float(T[0, 3]) + (.35 if here[0] < T[0, 3] else -.35)
            options = [p for p in stances(ctx, x, spread=.2) if abs(p[0] - here[0]) > .2]
            if empty <= 1 and options:
                yield from stand(ctx, est, options[0])
                continue
            blind = True; ctx.event('blind', covered=round(float(covered.mean()), 3), seen_clean=round(float((est.seen_clean >= 2).mean()), 3), left=int((~covered & (est.seen_clean < 2)).sum()))
        if blind:
            # (what the dabs have not passed over, what the camera has shown least first: shown clean twice was still
            # no proof, 7 of 20 particles lay there in 8 close views of start 8, missed by the speck detector)
            # (after the specks still seen, in the same round)
            left = np.flatnonzero(~covered)
            if not len(left) and passes < 3:
                # (all of it passed over and time left: again, from where dust is likeliest; a pass took particles
                # the trumpet's pose, a centimetre or two off, had the box miss: one left at the end of 5000 blind
                # steps, 302, 308)
                passes += 1; covered = unreach.copy(); left = np.flatnonzero(~covered)
                ctx.event('blind', again=passes, left=int(len(left)))
            # (where dust is likely and the camera has shown least, first)
            left = left[np.argsort(-est.dust_density[left] / (1. + est.seen_clean[left]), kind='stable')]
            if not len(left) and not targets:
                break
            W = (np.c_[surf[left], np.ones(len(left))] @ T.T)[:, :3]
            targets = targets + [(-1 - int(k), w) for k, w in zip(left, W)]
        empty = 0
        ids = [i for i, _ in targets]; pts = np.array([p for _, p in targets])
        dabs, missed = plan_attitudes(fields, T, pts, held, most=PER_ROUND) if blind else plan_attitudes(fields, T, pts, held)
        # (one attitude after another, along the trumpet in each: the next dab glides there, 20 steps against 50)
        order = {}
        for _, _, R_ in dabs:
            order.setdefault(id(R_), len(order))
        dabs.sort(key=lambda d: (order[id(d[2])], d[0][0]))
        ring = np.array([[T[0, 3] + a, T[1, 3] + b] for a in np.arange(-.5, .5, .02) for b in np.arange(-.15, .16, .03)])
        top_all = float(fields.surface(T, ring).max())
        done_before = done; idle += 1
        if not dabs and blind and any(i >= 0 for i in ids):
            # (blind, specks seen again that no dab takes: let be, and on with the surface in the next round, not idle;
            # three such rounds ended a run with 3000 steps left, 303)
            for s_ in est.specks:
                s_['dabs'] = max(s_['dabs'], 2)
            idle = max(0, idle - 1)
            continue
        if not dabs and not blind:
            # (specks left that no dab takes: the blind dabs, not the end; three such rounds ended a run at 2200 of
            # 7960 steps with 8 particles left, 302)
            blind = True; idle = 0; ctx.event('blind', covered=round(float(covered.mean()), 3), reason='specks out of reach')
            for s_ in est.specks:
                s_['dabs'] = max(s_['dabs'], 2)        # (those specks are let be: the blind dabs pass near them)
        if missed:
            ctx.event('unreachable', specks=[ids[i] for i in missed][:20], n=len(missed))
            # (blind, the surface no attitude takes is let be: it stopped the round, and the run with 3000 steps left, 19)
            for i in missed:
                if ids[i] < 0:
                    covered[-1 - ids[i]] = True; unreach[-1 - ids[i]] = True
            for i in missed:
                for s_ in est.specks:
                    if s_['id'] == ids[i]:
                        s_['dabs'] += 1         # (none takes it from where it is believed: tried no more than twice)
        for fp, took, R_d in dabs[:(PER_ROUND if blind else 8)]:
            if est.ob.steps > LIMIT:
                break
            low = lowest(R_d); z_safe = top_all + SAFE - low
            entry = dict(fingers=np.round(fp, 3).tolist(), specks=[ids[i] for i in took])
            for i in took:
                for s in est.specks:
                    if s['id'] == ids[i]:
                        s['dabs'] += 1
            glided = False
            if bottom is not None:
                # (the next dab near, at the same attitude: the brush glides there low, its box passing over all between
                # them, as a hand scrubs; up, across and down took 50 steps a dab and 90-110 dabs ran out of time, u1)
                bfp, bR, bsafe, boff = bottom; base = est.base(); R_b = rz(-base[2]) @ bR
                if bR is R_d and float(np.linalg.norm(fp[:2] - bfp[:2])) < GLIDE:
                    ok = True; passed = []
                    for wpt in glide_path(bfp, fp, R_d, boff):
                        seg = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], np.r_[in_base(base, wpt), wpt[2] - BASE_Z], R_b, step=.01)
                        if seg is None:
                            ok = False; break
                        yield from follow(ctx, seg, rate=.015, max_steps=60); passed.append(wpt)
                    for wpt in passed[::3]:
                        cw = (R_d @ held.corners.T).T + wpt; blo = cw.min(0) - M + .003; bhi = cw.max(0) + M - .003
                        W = (np.c_[surf, np.ones(len(surf))] @ T.T)[:, :3]; covered |= np.all((W > blo) & (W < bhi), axis=1)
                    glided = ok; entry['glided'] = ok
                if not glided:
                    back = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], np.r_[ctx.tip()[:2, 3], bsafe - BASE_Z], R_b, step=.02)
                    if back is not None:
                        yield from follow(ctx, back, rate=FAST, max_steps=300)
                bottom = None
            base = est.base(); plan = None
            if glided:
                plan = True; pose = tuple(base)
            for k_pose, pose in enumerate([] if glided else [tuple(base)] + stances(ctx, float(fp[0]))[:6]):
                if k_pose and not body_clear(ctx, pose):
                    continue
                plan = decor.plan_pinch(ctx, np.r_[in_base(pose, fp), z_safe - BASE_Z], rz(-pose[2]) @ R_d, base=pose2(*pose), way_in=.01, hand_margin=-.03, count=24)
                if plan is not None:
                    break
            if plan is None:
                entry['error'] = 'no plan'; ctx.event('dab', **entry); continue
            if float(np.linalg.norm(np.asarray(pose[:2]) - np.asarray(base[:2]))) > .02:
                # (another stance: the hand up, the base there, then a new round from there: the trumpet as seen from
                # there, with the base's pose as it is there)
                t = ctx.tip()[:3, 3]
                up = interpolate_cartesian(ctx.ik, ctx.q, t, np.r_[t[:2], max(t[2], z_safe - BASE_Z)], ctx.tip()[:3, :3], step=.03)
                if up is not None:
                    yield from follow(ctx, up, rate=.04, max_steps=200)
                for i in took:
                    for s_ in est.specks:
                        if s_['id'] == ids[i]:
                            s_['dabs'] -= 1
                yield from stand(ctx, est, pose); ctx.event('stance', thing='dabs', pose=np.round(pose, 2).tolist())
                idle = 0; bottom = None
                break
            turned = float(np.degrees(np.arccos(np.clip((np.trace(R_t.T @ R_d) - 1) / 2, -1, 1)))) > 3.
            if glided:
                pass
            elif turned:
                # (another attitude: up clear of the trumpet with the one it has, then turned on the way over)
                t = ctx.tip()[:3, 3]; hi = max(z_safe, top_all + SAFE - lowest(R_t)) + .06
                up = interpolate_cartesian(ctx.ik, ctx.q, t, np.r_[t[:2], max(t[2], hi - BASE_Z)], ctx.tip()[:3, :3], step=.03)
                if up is not None:
                    yield from follow(ctx, up, rate=.04, max_steps=200)
                p = decor.plan_pinch(ctx, np.r_[in_base(base, fp), z_safe - BASE_Z], rz(-base[2]) @ R_d, way_in=.01, hand_margin=-.03, count=24)
                if p is None:
                    entry['error'] = 'no way to turn'; ctx.event('dab', **entry); continue
                yield from move_to(ctx, p['grip'], rate=.05, label='turn_brush')
                R_t = R_d
            else:
                R_t = R_d
                R_b = rz(-base[2]) @ R_t; t = ctx.tip()[:3, 3]; bad = False
                # (the arm, not only the brush, clear of the trumpet on the way over: across 0.6 m of it the arm swept
                # it 0.3 m off and turned it 0.34 rad, af1 306; where the straight way is not clear, a planned one)
                check = ctx.validator(relax_from=ctx.q, hand_margin=-.03)
                for goal in (np.r_[t[:2], max(t[2], z_safe - BASE_Z)], np.r_[in_base(base, fp), z_safe - BASE_Z]):
                    seg = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], goal, R_b, validator=check, step=.02)
                    if seg is None:
                        bad = True; break
                    yield from follow(ctx, seg, rate=FAST, max_steps=300)
                if bad:
                    p = decor.plan_pinch(ctx, np.r_[in_base(base, fp), z_safe - BASE_Z], R_b, way_in=.01, hand_margin=-.03, count=24)
                    if p is None:
                        entry['error'] = 'no way over'; ctx.event('dab', **entry); continue
                    yield from move_to(ctx, p['grip'], rate=.05, label='above')
            base = est.base(); R_b = rz(-base[2]) @ R_t
            if not measured:
                # (where the brush is in the hand, as the head camera shows it over the trumpet: the fingers close where
                # they meet it, not where the plan had them; 1-2 cm off, the box misses the speck it is let down for)
                measured = True; shift = yield from measure(held)
                if shift is not None and float(np.linalg.norm(shift)) > .008:
                    held.correct(shift); low = lowest(R_t)
                    for i in took:
                        for s_ in est.specks:
                            if s_['id'] == ids[i]:
                                s_['dabs'] -= 1
                    idle = 0; break
            # (the hand's attitude as the arm holds it: the brush's lowest point from that, not from the one asked for)
            _, Rw = tool_world(ctx, est); low_now = float((Rw @ held.corners.T).T[:, 2].min())
            fz = fp[2] + low - low_now
            if not glided:
                down = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], np.r_[in_base(base, fp), fz - BASE_Z], R_b, step=.01)
                if down is None:
                    entry['error'] = 'no way down'; ctx.event('dab', **entry); continue
                yield from follow(ctx, down, rate=.015, max_steps=300)
            yield from hold(ctx, 3)
            tw, Rw = tool_world(ctx, est); cw = (Rw @ held.corners.T).T + tw
            ctx.event('dab_bottom', box=[np.round(cw.min(0), 3).tolist(), np.round(cw.max(0), 3).tolist()], tool=np.round(tw, 4).tolist(), rotation=np.round(Rw, 4).tolist(), mid=np.round(held.mid, 4).tolist(), far=np.round(held.far, 4).tolist(),
                      tool_base=np.round(ctx.tip(), 4).tolist(), base=np.round(est.base(), 4).tolist(), targets=[np.round(pts[i], 3).tolist() for i in took])
            bottom = (fp, R_d, z_safe, low - low_now)        # (up again only if the next dab is not glided to)
            done += 1; idle = 0; ctx.event('dab', **entry)
            # (the surface this dab's box passed over, as the trumpet lies now)
            cw = (Rw @ held.corners.T).T + tw; blo = cw.min(0) - M + .003; bhi = cw.max(0) + M - .003
            W = (np.c_[surf, np.ones(len(surf))] @ T.T)[:, :3]
            covered |= np.all((W > blo) & (W < bhi), axis=1)
            if done % 4 == 0:
                # (has the trumpet moved: the dabs push it, and the next ones, glided over where it had lain, pushed it
                # on, 0.3-0.6 m and off the desk in 3 of 20 runs, ac1 316, 318, 320: a look with the hand where it is,
                # and the round planned again from where it lies now)
                tw_, _ = tool_world(ctx, est); est.hand = tw_
                yield from hold(ctx, 2); est.look(dust=False); T_now = est.T()
                if T_now is not None:
                    moved = float(np.linalg.norm(T_now[:2, 3] - T[:2, 3])); turned_by = abs(wrap(np.arctan2(T_now[1, 0], T_now[0, 0]) - np.arctan2(T[1, 0], T[0, 0])))
                    if moved > PUSHED or turned_by > 3 * PUSHED:
                        ctx.event('pushed', moved=round(moved, 3), turned=round(turned_by, 3)); bottom = None; break
    return done


FIELDS = None
TRUMPET_PTS = None


def full_task(ctx, est):
    global FIELDS
    from pathlib import Path
    FIELDS = Path(__file__).parent / 'priors' / 'trumpet_fields.npz'
    global TRUMPET_PTS
    c = np.load(Path(__file__).parent / 'priors' / 'trumpet_model.npz')['collision'].astype(float)
    TRUMPET_PTS = c[np.random.default_rng(0).choice(len(c), min(len(c), 400), replace=False)]
    ctx.holding = {}
    yield from tuck(ctx)
    found = yield from find(ctx, est)
    ctx.event('stage', done='find', trumpet=est.trumpet is not None, brush=est.brush is not None, specks=len(est.specks))
    if not found:
        return 'not found'
    taken = yield from take_brush(ctx, est)
    ctx.event('stage', done='take', ok=taken is not None)
    if taken is None:
        return 'brush not taken'
    arm, held = taken; ctx.holding = {arm: 'brush'}
    # (turned over the desk's near edge, high: the brush swings its length round the fingers)
    base = est.base(); ctx.arm = arm; R_t = held.tilted(SOUTH)
    ready = np.r_[base[0], .36, DESK_TOP + .32]
    p = decor.plan_pinch(ctx, np.r_[in_base(base, ready), ready[2] - BASE_Z], rz(-base[2]) @ R_t, way_in=.01, hand_margin=-.03, count=48)
    if p is None:
        return 'brush not turned'
    yield from move_to(ctx, p['grip'], rate=.03, label='turn')
    # (where the brush sits in the fingers, as the wrist camera of that hand shows it: up to 3.5 cm off along the
    # fingers' closing from where the plan and the fingers' joints put it, 8, 59; the head camera, from over 0.5 m
    # and at the edge of its view, was 1-10 cm off)
    yield from hold(ctx, 8)
    seen = []
    for _ in range(4):
        yield from hold(ctx, 2)
        h = est.held_brush(arm, ctx.tip())
        if h is not None:
            seen.append(h)
    if seen:
        centres = np.array([c for c, _, _, _ in seen]); centre_m = np.median(centres, axis=0)
        # (the camera sees the brush's back, and its near part runs out of the picture: only the shift along the
        # fingers' closing is taken; along its length and its depth in the hand stay as the model has them)
        shift = centre_m - held.mid; shift[0] = 0.; shift[2] = 0.
        ctx.event('held', centre=np.round(centre_m, 4).tolist(), model=np.round(held.mid, 4).tolist(), shift=np.round(shift, 4).tolist(),
                  length=round(float(np.median([l for _, _, l, _ in seen])), 3), points=[n for _, _, _, n in seen])
        if float(np.linalg.norm(shift)) < .05:
            held.correct(shift)
    else:
        ctx.event('held', seen=False, debug=getattr(est, 'held_debug', None))
    done = yield from clean(ctx, est, arm, held)
    ctx.event('stage', done='clean', dabs=done, specks=len(est.specks))
    return f'{done} dabs'
