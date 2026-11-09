"""The fax task driven by onboard observations: localise, find the fax, then run the skills.

Everything the skills need about the world comes from Observer (map-based localisation and
RGB-D fax detection). No simulator state is read here.
"""
import numpy as np
from observer import Observer, pose2, wrap, BASE_Z
from fax_skills import (plan_floor_stance, floor_prestance, choose_rotation, grasp_rotations, press_button, grasp_from_floor, to_carry,
                        choose_site, place_on_desk, drive, hold, FAX_KEEPOUT, PAYLOAD, PAYLOAD_RADIUS, ARMS, BUTTON)
from localizer import ANGULAR_GAIN

# Where to look from when the fax is not in sight: the east end of each aisle, facing west.
AISLE_VIEWS = [(12.15, 1.30), (12.15, -2.70), (12.15, -6.62)]


class ObservedEstimator:
    """State estimates for fax_skills.Context, fed by the Observer.

    Two frames are kept apart on purpose. The base pose in the map comes from the localiser and
    may be corrected by a few centimetres at any time. The fax, once seen, is remembered relative
    to dead-reckoned odometry, which is smooth: over the last half metre to the fax and while the
    arm works, its error is millimetres. Manipulation targets therefore never see a map correction.
    """

    def __init__(self, observer):
        self.ob = observer; self.odom = np.zeros(3); self.fax_odom = None; self.fits = []; self.marker = None

    def predict(self, s, last_action, dt=1 / 30):
        """Dead reckoning: measured base translation (accurate) and commanded turn rate (the measured one is biased)."""
        vx, vy = float(s[0]), float(s[1]); wz = float(last_action[2]) * ANGULAR_GAIN
        yaw = self.odom[2] + .5 * wz * dt; c, sn = np.cos(yaw), np.sin(yaw)
        self.odom = np.array([self.odom[0] + (c * vx - sn * vy) * dt, self.odom[1] + (sn * vx + c * vy) * dt, self.odom[2] + wz * dt])

    def base(self):
        return tuple(self.ob.loc.pose)

    def fax_in_base(self):
        return None if self.fax_odom is None else np.linalg.inv(pose2(*self.odom, z=BASE_Z)) @ self.fax_odom

    def fax(self):
        relative = self.fax_in_base()
        return None if relative is None else pose2(*self.base(), z=BASE_Z) @ relative

    def set_fax_from_base(self, pose_in_base):
        """Store a fresh fit; the estimate is the per-component median of the recent ones (the fax does not move)."""
        world = pose2(*self.odom, z=BASE_Z) @ pose_in_base
        self.fits = (self.fits + [(world[0, 3], world[1, 3], np.arctan2(world[1, 0], world[0, 0]))])[-5:]
        fits = np.array(self.fits); yaw = fits[-1, 2] + np.median(wrap(fits[:, 2] - fits[-1, 2]))
        self.fax_odom = pose2(np.median(fits[:, 0]), np.median(fits[:, 1]), yaw, z=world[2, 3])

    def set_fax_guess(self, xy_map):
        """Position-only guess (map frame) to aim the first close look at."""
        guess = np.linalg.inv(pose2(*self.base(), z=BASE_Z)) @ pose2(xy_map[0], xy_map[1], 0., z=.0362)
        self.fax_odom = pose2(*self.odom, z=BASE_Z) @ guess; self.fits = []

    def shift_fax_in_base(self, shift_xy):
        """Slides the estimate by @shift_xy (base frame) keeping its orientation."""
        c, s = np.cos(self.odom[2]), np.sin(self.odom[2]); delta = np.array([c * shift_xy[0] - s * shift_xy[1], s * shift_xy[0] + c * shift_xy[1]])
        self.fax_odom = self.fax_odom.copy(); self.fax_odom[:2, 3] += delta
        self.fits = [(x + delta[0], y + delta[1], yaw) for x, y, yaw in self.fits]


class Runtime:
    """Per-step glue: feeds observations to the Observer, keeps the localiser running, steps the script."""

    def __init__(self, ctx, observer, estimator, script):
        self.ctx = ctx; self.ob = observer; self.est = estimator; self.script = script(self)
        self.last_action = np.zeros(23); self.moving = 0; self.finished = None
        self.rejected = []; self.views_done = set()
        ctx.on_refresh = self.refresh_fax

    def step(self, data):
        s = data['robot_r1::proprio'].astype(float)
        self.ctx.update(s)
        payload = None
        if self.ctx.held is not None:
            payload = (self.ctx.ik.link, PAYLOAD @ self.ctx.held[:3, :3].T + self.ctx.held[:3, 3], PAYLOAD_RADIUS)
        self.ob.frame(data, payload)
        self.est.predict(s, self.last_action)
        loc = self.ob.loc
        if loc.pose is not None:
            loc.predict(self.last_action[:3])
            self.moving = 18 if np.any(np.abs(self.last_action[:3]) > .01) else max(0, self.moving - 1)
            # Correct while the base moves (and shortly after). A standing base keeps its pose: scans
            # taken while the arm works near the camera must not nudge it.
            if self.moving and self.ob.steps % 3 == 0:
                loc.correct(self.ob.scan())
        if self.finished is None:
            try:
                action = next(self.script)
            except StopIteration as stop:
                self.finished = stop.value or 'done'; action = self.ctx.action()
        else:
            action = self.ctx.action()
        self.last_action = np.asarray(action, float)
        return self.last_action

    # ---- observation-based helpers used by the script -----------------------------------
    def refresh_fax(self):
        """Re-estimate the fax from the current images. True if the estimate was updated.

        Close to contact the wrist camera sees the toggle marker from straight above; aligning the
        model's button with it fixes the position better than the head view, which the arm blocks.
        """
        relative = self.est.fax_in_base()
        if relative is None:
            return False
        seen = self.ob.wrist_marker(self.ctx.arm)
        if seen is not None:
            position, colour, pixels = seen
            expected = relative[:3, 3] + relative[:3, :3] @ BUTTON
            shift = position[:2] - expected[:2]
            if np.linalg.norm(shift) < .12 and abs(position[2] - expected[2]) < .05:
                self.est.shift_fax_in_base(shift); self.est.marker = colour
                self.ctx.event('marker_align', shift=np.round(shift, 4).tolist(), colour=colour, pixels=pixels)
                return True
        out = self.ob.fit_fax(relative[:2, 3])
        if out is None:
            return False
        self.est.set_fax_from_base(out[0]); self.est.marker = out[1]['marker']; return True


def turn_in_place(rt, angle, rate=.8):
    """Rotates the base by about @angle (commanded); yields actions. Returns the commanded angle."""
    ctx = rt.ctx; steps = int(round(abs(angle) / (rate * ANGULAR_GAIN) * 30)); done = 0.; command = 0.
    for k in range(steps + 14):
        wanted = np.sign(angle) * rate if k < steps else 0.
        command += np.clip(wanted - command, -.07, .07)
        done += command * ANGULAR_GAIN / 30
        yield ctx.action(base=(0., 0., command))
    return done


def look_for_fax(rt, sightings):
    """Adds the current frame's fax candidates (in M) to @sightings, if the pose is known."""
    if rt.ob.loc.pose is None:
        return
    for found in rt.ob.detect_far(rt.ob.loc.pose):
        if any(np.linalg.norm(found['xy'] - other) < .40 for other in rt.rejected):
            continue                          # looked at closely before and found not to be the fax
        sightings.append((found['xy'], found['points'], found['red']))


def best_sighting(sightings, minimum=2):
    """Position seen repeatedly, or None. Candidates within 15 cm of each other count as one."""
    if not sightings:
        return None
    points = np.array([s[0] for s in sightings]); weight = np.array([s[1] for s in sightings], float); red = np.array([s[2] for s in sightings])
    best = None
    for k in range(len(points)):
        near = np.linalg.norm(points - points[k], axis=1) < .15
        score = near.sum() + 3. * (red[near] > 0).sum()
        if near.sum() >= minimum and (best is None or score > best[0]):
            best = (score, np.average(points[near], axis=0, weights=weight[near]))
    return None if best is None else best[1]


def localize(rt):
    """Global localisation by looking around on the spot. Returns True when a pose was accepted."""
    ctx, ob = rt.ctx, rt.ob
    yield from hold(ctx, 3)                   # the first frames after reset can still show the previous pose
    scans = [(ob.scan(), 0.)]; turned = 0.
    for attempt in range(5):
        if attempt:
            turned += yield from turn_in_place(rt, 2 * np.pi / 3)
            yield from hold(ctx, 2)
            scans.append((ob.scan(), turned))
        if len(scans) < 2:
            continue
        pose, score, margin = ob.localize_global(scans)
        ctx.event('localize', scans=len(scans), score=round(score, 3), margin=round(margin, 3), pose=None if pose is None else np.round(pose, 3).tolist(),
                  rivals=[np.round(p, 2).tolist() for _, p in ob.loc.shortlist] if pose is not None else [])
        # A clear winner is taken at once. Look-alike aisles may leave rivals even after a full turn:
        # then the best one is used and the others are followed along until the views separate them.
        if pose is not None and score > .5 and (margin > .06 or len(scans) >= 4):
            shift = lambda p: np.array([p[0], p[1], wrap(p[2] + turned)])
            ob.loc.start_tracking(shift(pose), [shift(p) for _, p in ob.loc.shortlist] if margin <= .12 else [])
            ob.loc.correct(ob.scan())
            return True
    return False


def find_fax(rt, look_around=True):
    """Looks around, then visits the aisle ends not yet visited, until something fax-like stands on the
    floor. Returns xy in M or None. Can be called again after a candidate was rejected."""
    ctx, ob = rt.ctx, rt.ob; sightings = []
    if look_around:
        # A full turn where we stand.
        for _ in range(6):
            look_for_fax(rt, sightings)
            yield from turn_in_place(rt, np.pi / 3)
            yield from hold(ctx, 2)
        look_for_fax(rt, sightings)
        target = best_sighting(sightings)
        if target is not None:
            return target
    while True:
        x, y, _ = rt.est.base()
        remaining = [v for v in AISLE_VIEWS if v not in rt.views_done and ctx.scene.path((x, y), v) is not None]
        if not remaining:
            return None
        view = min(remaining, key=lambda v: abs(v[1] - y) + abs(v[0] - x)); rt.views_done.add(view)
        sightings = []
        status = yield from drive_and_look(rt, view, np.pi, sightings)
        for sweep in (-.6, 1.2, -.6):
            look_for_fax(rt, sightings)
            yield from turn_in_place(rt, sweep)
            yield from hold(ctx, 2)
        look_for_fax(rt, sightings)
        target = best_sighting(sightings)
        ctx.event('view', at=list(view), status=status, sightings=len(sightings), found=target is not None)
        if target is not None:
            return target


def drive_and_look(rt, goal_xy, goal_yaw, sightings, **kwargs):
    """drive() that also collects fax sightings on the way."""
    mover = drive(rt.ctx, goal_xy, goal_yaw, **kwargs)
    try:
        while True:
            action = next(mover)
            if rt.ob.steps % 6 == 0:
                look_for_fax(rt, sightings)
            yield action
    except StopIteration as stop:
        return stop.value


def inspect_fax(rt, fax_xy):
    """Drives to about a metre from the fax, faces it and fits its pose. Returns True when a fit was accepted."""
    ctx = rt.ctx; x, y, _ = rt.est.base(); options = []
    # Any spot the base fits in, 0.85-1.25 m from the fax. In the narrow aisles that is a thin strip.
    for dx in np.arange(-1.3, 1.31, .08):
        for dy in np.arange(-1.3, 1.31, .08):
            distance = np.hypot(dx, dy)
            if not .72 <= distance <= 1.25:
                continue
            point = fax_xy + np.array([dx, dy])
            if ctx.scene.base_free(point, margin=.01):
                options.append((np.linalg.norm(point - [x, y]) + .5 * abs(distance - 1.0), point, np.arctan2(-dy, -dx)))
    options.sort(key=lambda item: item[0]); tried = []
    for _, point, heading in options:
        if len(tried) >= 3:
            break
        if any(np.linalg.norm(point - other) < .22 for other in tried):
            continue                          # a different viewpoint each time
        if ctx.scene.path(rt.est.base()[:2], point, keepout=[(fax_xy, FAX_KEEPOUT)]) is None:
            continue
        tried.append(point)
        status = yield from drive(ctx, point, heading, keepout=[(fax_xy, FAX_KEEPOUT)])
        yield from hold(ctx, 3)
        rt.est.set_fax_guess(fax_xy)          # position only, until a fit succeeds
        accepted = 0
        for _ in range(3):
            accepted += int(rt.refresh_fax())
            yield from hold(ctx, 2)
        ctx.event('inspect', status=status, accepted=accepted, stats={k: (round(v, 4) if isinstance(v, float) else v) for k, v in rt.ob.fax_stats.items()})
        if accepted:
            return True
    ctx.event('inspect', ok=False, options=len(options), tried=len(tried))
    return False


def observed_pipeline(rt):
    ctx, est = rt.ctx, rt.est
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    # Search until a candidate survives a close look. A candidate that does not fit the fax model is
    # remembered and the search goes on; the last pass revisits every aisle end once more.
    fitted = False
    for attempt in range(8):
        fax_xy = yield from find_fax(rt, look_around=attempt == 0)
        if fax_xy is None:
            if attempt and len(rt.views_done) == len(AISLE_VIEWS) and not getattr(rt, 'second_pass', False):
                rt.second_pass = True; rt.views_done.clear(); continue
            return 'fax_not_found'
        ctx.event('fax_seen', xy=np.round(fax_xy, 3).tolist())
        fitted = yield from inspect_fax(rt, fax_xy)
        if fitted:
            break
        rt.rejected.append(np.asarray(fax_xy, float)); rt.est.fax_odom = None; rt.est.fits = []
        ctx.event('candidate_rejected', xy=np.round(fax_xy, 3).tolist())
    if not fitted:
        return 'fax_not_fitted'
    # Up to five tries at the floor. A stance that could not be reached, or from which the grasp
    # turned out to be out of reach, is not offered again; an empty grasp is retried from a fresh
    # plan. The button is not pressed again once the marker has been seen green.
    turned_on = False; refused = []; grasped = False
    for attempt in range(5):
        fax = est.fax()
        stance = plan_floor_stance(ctx, fax, exclude=refused)
        if stance is None:
            if attempt == 0:
                return 'no_stance'
            break
        ctx.arm = stance[2]
        pre = floor_prestance(ctx, stance[0], fax[:2, 3])
        status = yield from drive(ctx, pre, stance[1], keepout=[(fax[:2, 3], FAX_KEEPOUT)])
        if status != 'arrived':
            refused.append(stance); ctx.event('pipeline', ok=False, reason='stance not reached', status=status); continue
        yield from hold(ctx, 3)
        rt.refresh_fax()                      # a closer look before the last straight stretch
        # The last stretch ends within 1.5 cm of the planned pose. With the general 3.5 cm the base
        # stopped at the near edge of the tolerance every time, i.e. always that much further from the fax.
        status = yield from drive(ctx, stance[0], stance[1], speed=.25, direct=True, tolerance=.015, min_speed=.04)
        short = float(np.linalg.norm(np.asarray(stance[0], float) - est.base()[:2]))
        if status != 'arrived' and short > .05:
            refused.append(stance); ctx.event('pipeline', ok=False, reason='stance not reached', status=status, short=round(short, 3)); continue
        rotation = choose_rotation(ctx, ctx.fax_in_base())
        if rotation is None:
            refused.append(stance); ctx.event('pipeline', ok=False, reason='grasp not reachable from stance', short=round(short, 3)); continue
        # Press until the marker is seen green (it is red while the fax is off); at most three times.
        # The closed gripper can come down at either of the two grasp yaws, so try the other if one is out of reach.
        for press in range(0 if turned_on else 3):
            est.marker = None; pressed = False
            for option in sorted(grasp_rotations(ctx.fax_in_base()), key=lambda r: -float(np.trace(r.T @ rotation))):
                pressed = yield from press_button(ctx, option)
                if pressed:
                    break
            yield from hold(ctx, 3)
            rt.refresh_fax()
            ctx.event('press_check', attempt=press, pressed=pressed, marker=est.marker)
            turned_on = est.marker == 'green'
            if turned_on or not pressed:
                break
        grasped = yield from grasp_from_floor(ctx, rotation)
        # Fingers that closed completely hold nothing.
        if grasped and float(np.sum(ctx.fingers)) > .02:
            break
        grasped = False
        ctx.event('grasp_check', ok=False, fingers=ctx.fingers.round(4).tolist())
        ctx.set_grip(1.); ctx.held = None
        yield from hold(ctx, 20)
    if not grasped:
        return 'grasp_failed'
    carried = yield from to_carry(ctx, rotation)
    if not carried:
        return 'carry_failed'
    chosen = choose_site(ctx)
    if chosen is None:
        return 'no_site'
    key, site = chosen
    ctx.event('site', carrel=key)
    status = yield from drive(ctx, site['entry'], site['yaw'], speed=.45)
    if status != 'arrived':
        return 'entry_' + status
    status = yield from drive(ctx, site['stance'], site['yaw'], speed=.2, direct=True, tolerance=.05, yaw_tolerance=.08)
    placed = yield from place_on_desk(ctx, site['target'])
    if not placed:
        return 'place_failed'
    yield from hold(ctx, 10)
    if not turned_on:
        # The button could not be pressed on the floor (or was not seen to turn green): do it here.
        # The fax lies where the hand let go of it, and the wrist camera looks straight down at it.
        for press in range(3):
            est.fits = []; est.set_fax_from_base(ctx.released); est.marker = None
            # The fax settled onto the desk after release, so its height is known less well than on the
            # floor: aim a little deeper and let contact stop the fingers.
            pressed = yield from press_button(ctx, ctx.tip()[:3, :3], static_only=True, free_yaw=True, depth=.016)
            yield from hold(ctx, 3)
            rt.refresh_fax()
            ctx.event('press_on_desk', attempt=press, pressed=pressed, marker=est.marker)
            if est.marker == 'green' or not pressed:
                break
    yield from hold(ctx, 20)
    return 'done'
