"""The car task driven by onboard observations: localise, find the toy box, the camera and the racket, run the skills.

Everything the skills need about the world comes from Observer (map-based localisation; the toy box as a thing of
its size on floor the map has empty; the camera and the racket as what stands and lies on the coffee table; the
trunk lid's edge). No simulator state is read here.
"""
import numpy as np
from observer import pose2, wrap
from manip import hold, TALL_TRUNK
from car_pipeline import full_task
from localizer import turn_rate, true_velocity, command_velocity

ZONE = .35                  # the box is looked for on the living-room floor (the task puts it there) and this far beyond its edge (m)
GOOD_RANGE = {'box': 1.6, 'camera': 1.15, 'racket': 1.15}      # a sighting from nearer than this places the thing well enough for the hand
STILL_GOOD = (1.5, 2.0)     # ... while the base has since driven less than this (m) and turned less than this (rad)
SAME = .08                  # a sighting further than this from what was known is a thing that has been moved (m)
LIVING = (0.8, 2.4)         # the detectors run only where the base is east of / north of this (map x, y): beyond the corridor


class ObservedEstimator:
    """State estimates for the skills, fed by the Observer.

    The base pose in the map comes from the localiser (and, behind the car, from the trunk lid's edge). The box,
    the camera and the racket are remembered in the map frame; for the moment of contact the last sighting from
    close by is carried along by dead reckoning, so that an error of the map pose does not enter the pinch.
    """

    def __init__(self, observer, scene):
        self.ob = observer; self.scene = scene
        floor = scene.meta['scope']['floor.n.01_2']; self.floor = (np.array(floor['aabb_lo'][:2]), np.array(floor['aabb_hi'][:2]))
        self.odom = np.zeros(3); self.commanded = 0.; self.velocity = np.zeros(2); self.held_up = 0; self.settling = False
        self.driven = 0.; self.turned = 0.          # how far the base has driven and turned in all (m, rad)
        self.things = {}            # 'box', 'camera', 'racket' -> what is known of each
        self.news = []              # what was found or changed since the script last asked: for the event log
        self.source = None

    def predict(self, s, last_action, dt=1 / 30):
        """Dead reckoning from the velocity command, which the base follows closely in free motion (see
        localizer.command_velocity), and from the base's own report of its translation where the command does
        not tell: when the base is held up (the report then reads next to nothing) and while it coasts to a stop.
        The reported yaw rate is biased and not used."""
        command = np.asarray(last_action, float)[:2]; size = float(np.linalg.norm(command))
        self.commanded = max(size, .9 * self.commanded)
        report = true_velocity(s[:2]); model = command_velocity(command); speed = float(np.linalg.norm(report))
        if size >= .02:
            # Held up = the report stays under half of what was commanded for a fifth of a second.
            self.held_up = self.held_up + 1 if speed < .5 * float(np.linalg.norm(model)) else 0
            self.velocity = report if self.held_up >= 6 else model.copy()
        else:
            # Coasting: the report, limited to what the last commands could have produced. Turning on the spot
            # the report shows a drift that is not there: with no translation commanded for a second, none is counted.
            self.held_up = 0; limit = 1.3 * self.commanded + .05
            self.velocity = report * min(1., limit / max(speed, 1e-9)) if self.commanded > .01 else np.zeros(2)
        vx, vy = float(self.velocity[0]), float(self.velocity[1]); wz = turn_rate(last_action[2], driving=np.hypot(last_action[0], last_action[1]) > .06)
        yaw = self.odom[2] + .5 * wz * dt; c, sn = np.cos(yaw), np.sin(yaw)
        self.odom = np.array([self.odom[0] + (c * vx - sn * vy) * dt, self.odom[1] + (sn * vx + c * vy) * dt, self.odom[2] + wz * dt])
        self.driven += float(np.hypot(vx, vy)) * dt; self.turned += abs(wz) * dt

    def base(self):
        return tuple(self.ob.loc.pose)

    # ---- the map pose ------------------------------------------------------------------------
    def hold_axis(self, axis):
        """Keep map coordinates by dead reckoning (@axis: 'x', 'y', 'xy' or None): for the passage through the
        doorway, where a few centimetres matter and what the cameras see (the door leaf swinging) is not what the
        map has. The heading is still fitted."""
        self.ob.loc.lock = None if axis is None else np.array([float('x' in axis), float('y' in axis)])

    def settle(self, flag):
        """Let the localiser correct the pose while the base stands still (it normally does so only in motion)."""
        self.settling = bool(flag)

    def fix_at_lid(self):
        """Take the base pose from the trunk lid's rear edge, if the closed lid shows. Returns the shift applied
        (m, m, rad), or None."""
        found = self.ob.lid_pose()
        if found is None:
            return None
        pose, rms, samples = found; old = np.asarray(self.ob.loc.pose, float); shift = np.array([pose[0] - old[0], pose[1] - old[1], wrap(pose[2] - old[2])])
        if abs(shift[0]) > .30 or abs(shift[1]) > .30 or abs(shift[2]) > .20:
            return None                     # not the lid: the map pose is not that far off
        self.ob.loc.pose = np.asarray(pose, float); self.ob.loc.travel = 0.; self.ob.loc.turn = 0.
        return shift

    def fix_at_bumper(self):
        """Take the base position from the car's bumper, seen from behind the car (lid open or shut). Returns the
        shift applied (m, m), or None."""
        found = self.ob.bumper_shift(self.ob.loc.pose)
        if found is None or abs(found[0]) > .25 or abs(found[1]) > .19:
            return None
        pose = np.asarray(self.ob.loc.pose, float); self.ob.loc.pose = np.array([pose[0] + found[0], pose[1] + found[1], pose[2]]); self.ob.loc.travel = 0.; self.ob.loc.turn = 0.
        return found[:2]

    def lid_shows_closed(self):
        """True if the trunk lid's edge shows where it lies when the lid is shut (and not something else at that
        height: a pose taken from it must agree with the map pose)."""
        found = self.ob.lid_pose()
        if found is None:
            return False
        pose = found[0]; old = np.asarray(self.ob.loc.pose, float)
        return abs(pose[0] - old[0]) < .20 and abs(pose[1] - old[1]) < .20 and abs(wrap(pose[2] - old[2])) < .15

    # ---- looking ---------------------------------------------------------------------------
    def _take(self, name, xy, direction, point, axis, distance, extra=None):
        """A sighting of @name: where it is in the map (@xy, @direction) and in the base frame (@point, @axis)."""
        odom = pose2(*self.odom); known = self.things.get(name)
        if known is not None and float(np.linalg.norm(known['xy'] - xy)) > SAME + .03 * max(known['range'], distance):
            self.news.append(dict(thing=name, moved_from=known['xy'].round(3).tolist(), to=np.round(xy, 3).tolist())); known = None
        if known is None:
            known = self.things[name] = dict(xy=np.asarray(xy, float).copy(), direction=float(direction), range=distance, count=0)
            self.news.append(dict(thing=name, found=np.round(xy, 3).tolist(), direction=round(float(direction), 3), range=round(distance, 2)))
        known['count'] += 1
        fresh = self._still_good(known)
        if distance < 1.15 * known['range'] or not fresh:
            known.update(xy=np.asarray(xy, float).copy(), direction=float(direction), range=distance)
        if not fresh or distance < 1.15 * known.get('odom_range', 1e9):
            known.update(odom=odom[:2, :2] @ np.asarray(point, float)[:2] + odom[:2, 3], odom_direction=float(axis) + self.odom[2], odom_range=distance, odom_at=(self.driven, self.turned))
            if extra:
                known.update(extra)

    def _still_good(self, known):
        return 'odom' in known and self.driven - known['odom_at'][0] < STILL_GOOD[0] and self.turned - known['odom_at'][1] < STILL_GOOD[1]

    def see(self, what=('box', 'table')):
        """Take what the current head image shows of the box and of the things on the table into the memory."""
        base = self.base(); seen = {}
        if 'box' in what and 'box' not in getattr(self, 'held', ()):
            box = self.ob.box_on_floor(base)
            if box is not None and not box['cut'] and np.all(box['xy'] > self.floor[0] - ZONE) and np.all(box['xy'] < self.floor[1] + ZONE):
                self._take('box', box['xy'], box['yaw'], box['point'], box['axis'], float(np.linalg.norm(box['point']))); seen['box'] = box['points']
        if 'table' in what:
            found = self.ob.table_things(base)
            camera, racket = found['camera'], found['racket']
            if camera is not None and not camera['cut'] and 'camera' not in getattr(self, 'held', ()):
                self._take('camera', camera['xy'], wrap(camera['closing'] + base[2]), camera['point'][:2], camera['closing'], float(np.linalg.norm(camera['point'][:2]))); seen['camera'] = camera['points']
            if racket is not None and not racket['cut'] and 'racket' not in getattr(self, 'held', ()):
                self._take('racket', racket['xy'], wrap(racket['axis'] + base[2]), racket['point'], racket['axis'], float(np.linalg.norm(racket['point']))); seen['racket'] = racket['points']
        return seen

    def where(self, name):
        """Map xy of 'box', 'camera' or 'racket' (for the camera: the hump on its top; for the racket: where the
        hand takes its handle), or None if not known."""
        known = self.things.get(name)
        return None if known is None else known['xy'].copy()

    def direction(self, name):
        """In the map: the box's long side (modulo half a turn); the way the fingers close on the camera's hump
        (modulo half a turn); from the racket's handle to its head."""
        known = self.things.get(name)
        return None if known is None else known['direction']

    def point(self, name):
        """The same place in the base frame, for the hand: the last sighting from close by, carried along by dead
        reckoning (over the last metre its error is a few millimetres, whatever the map pose does meanwhile);
        without a fresh one, the place in the map seen from the map pose. Returns (xy, direction)."""
        known = self.things.get(name)
        if known is None:
            return None
        if self._still_good(known):
            self.source = 'odom'
            return (np.linalg.inv(pose2(*self.odom)) @ np.r_[known['odom'], 0., 1.])[:2], float(wrap(known['odom_direction'] - self.odom[2]))
        self.source = 'map'; base = self.base()
        return (np.linalg.inv(pose2(*base)) @ np.r_[known['xy'], 0., 1.])[:2], float(wrap(known['direction'] - base[2]))

    def fresh(self, name):
        """True if @name has been seen from within GOOD_RANGE and the base has not moved much since."""
        known = self.things.get(name)
        return known is not None and self._still_good(known) and known['odom_range'] < GOOD_RANGE[name]

    def hand_view(self, arm, kind, tip, closing):
        """Where the thing under @arm's hand is as its wrist camera shows it (see Observer.hand_view)."""
        return self.ob.hand_view(arm, kind, tip, closing)

    def taken(self, name):
        """The hand has taken @name: it is no longer where it was seen."""
        self.things.pop(name, None)

    def forget(self, name):
        known = self.things.pop(name, None)
        if known is not None:
            self.news.append(dict(thing=name, forgotten=known['xy'].round(3).tolist()))


class Runtime:
    """Per-step glue: feeds observations to the Observer, keeps the localiser running, steps the script."""

    def __init__(self, ctx, observer, estimator, script):
        self.ctx = ctx; self.ob = observer; self.est = estimator; self.script = script(self)
        self.last_action = np.zeros(23); self.moving = 0; self.finished = None

    def step(self, data):
        s = data['robot_r1::proprio'].astype(float)
        self.ctx.update(s)
        self.ob.frame(data)
        # What the hands hold is not part of the scene: it is cut out of the scans and of what the detectors see.
        if self.ctx.payload:
            poses = self.ctx.body.fk(self.ctx.q)[0]; parts = [self.ctx.carried(poses, arm) for arm in self.ctx.payload]
            self.ob.held = (np.vstack([p for p, _ in parts]), np.concatenate([r for _, r in parts]))
        else:
            self.ob.held = None
        self.est.held = set(getattr(self.ctx, 'holding', {}).values())
        self.est.predict(s, self.last_action)
        loc = self.ob.loc
        if loc.pose is not None:
            loc.predict(self.last_action[:3], measured=self.est.velocity)
            self.moving = 18 if np.any(np.abs(self.last_action[:3]) > .01) else max(0, self.moving - 1)
            upright = float(np.max(np.abs(s[53:57] - TALL_TRUNK))) < .06
            # Correct while the base moves (and shortly after). A standing base keeps its pose: scans taken
            # while the arms work in front of the camera must not nudge it.
            if (self.moving or self.est.settling) and upright and self.ob.steps % 3 == 0:
                loc.correct(self.ob.scan())
            # Keep looking for the things on the way through the living room, with the trunk upright and not in a
            # fast turn (the heading then lags, and with it everything seen far off).
            if self.moving and upright and abs(self.last_action[2]) < .3 and self.ob.steps % 6 == 0 and loc.pose[0] > LIVING[0] and loc.pose[1] > LIVING[1]:
                self.est.see()
        while self.est.news:
            self.ctx.event('track', **self.est.news.pop(0))
        if self.finished is None:
            try:
                action = next(self.script)
            except StopIteration as stop:
                self.finished = stop.value or 'done'; action = self.ctx.action()
        else:
            action = self.ctx.action()
        self.last_action = np.asarray(action, float)
        return self.last_action


def turn_in_place(rt, angle, rate=.8):
    """Rotates the base by about @angle (commanded); yields actions. Returns the commanded angle."""
    ctx = rt.ctx; steps = int(round(abs(angle) / turn_rate(rate) * 30)); done = 0.; command = 0.
    for k in range(steps + 14):
        wanted = np.sign(angle) * rate if k < steps else 0.
        command += np.clip(wanted - command, -.07, .07)
        done += turn_rate(command) / 30
        yield ctx.action(base=(0., 0., command))
    return done


def accept_start(pose, score, margin, looks):
    """Whether a start pose found from @looks looks is taken: @score is its agreement with the map (0..1), @margin
    its lead over the best clearly different pose."""
    return pose is not None and looks >= 3 and score > .5 and (margin > .06 or looks >= 4)


def localize(rt):
    """Global localisation in the garage by looking around on the spot. Returns True when a pose was accepted."""
    ctx, ob = rt.ctx, rt.ob
    yield from hold(ctx, 3)                   # the first frames after reset can still show the previous pose
    scans = [(ob.scan(), 0.)]; turned = 0.
    # Three looks a third of a turn apart, then three more in between them.
    for attempt, turn in enumerate((0., 2 * np.pi / 3, 2 * np.pi / 3, np.pi / 3, 2 * np.pi / 3, 2 * np.pi / 3)):
        if attempt:
            turned += yield from turn_in_place(rt, turn)
            yield from hold(ctx, 2)
            scans.append((ob.scan(), turned))
        if len(scans) < 2:
            continue
        pose, score, margin = ob.localize_global(scans)
        ctx.event('localize', scans=len(scans), score=round(score, 3), margin=round(margin, 3), pose=None if pose is None else np.round(pose, 3).tolist(),
                  rivals=[np.round(p, 2).tolist() for _, p in ob.loc.shortlist] if pose is not None else [])
        if accept_start(pose, score, margin, len(scans)):
            shift = lambda p: np.array([p[0], p[1], wrap(p[2] + turned)])
            ob.loc.start_tracking(shift(pose), [shift(p) for _, p in ob.loc.shortlist] if margin <= .12 else [])
            ob.loc.correct(ob.scan())
            return True
    return False


def observed_pipeline(rt):
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    result = yield from full_task(rt.ctx, rt.est)
    return result
