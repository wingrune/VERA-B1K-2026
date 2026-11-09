"""The trash task driven by onboard observations: localise, look for the cans and the trash can, run the skills.

Everything the skills need about the world comes from Observer (map-based localisation; soda cans and the trash
can as things of their size on floor the map has empty). No simulator state is read here.
"""
import numpy as np
from observer import pose2, wrap
from manip import hold, TALL_TRUNK
from trash_pipeline import full_task, route, others
from localizer import turn_rate, true_velocity, command_velocity
import drawer

MERGE = (.06, .02)          # (batteries on the coffee table stand 0.14 m apart) sightings of one can lie within this distance of each other (m), plus this per metre of the
                            # farther sighting's range (a degree of heading error is 2 cm per metre)
GLASS_NORTH = 99.          # (no zones without glass in this scene: glasses are only looked for over the sink)
GLASS_WEST = (-99., 99.)
ZONE = .35                  # cans are looked for on the living-room floor and the trash can on the kitchen floor (the task
                            # puts them there), and this far beyond the floor's edge (m)
FRESH = 2.0                 # a sighting from nearer than this replaces what was known of the thing (m)
CONFIRMED = 2               # a can counts once it has been seen this many times
MISSES = 6                  # ... and is forgotten after this many looks at its place that did not show it
EXPECTED = ((.9, 1.6), np.radians(30))      # (a battery is 3.4 cm across: from further off too few points show) where a known can must show in the head image: range ahead (m) and bearing
CLOSE = 1.6                 # a sighting from nearer than this, carried by dead reckoning, gives the place for the hand (m)
BIN_ZONE = .30              # (the coffee table is in the map: nothing is seen there)
                            # the trash can, or a can above it in the hand (no start has a can closer than 0.37 m to it)
BIN_ZONE_LYING = .80        # ... but a can that has fallen next to the trash can lies as close as this
SAME_CAN = .16              # a can seen lying this close to where one stood (or the other way round) is that can
STILL_GOOD = (1.5, 2.0)     # a sighting from close by is fresh while the base has since driven less than this (m) and
                            # turned less than this (rad): what it says of the can's place then still holds for the hand
                            # (dead reckoning over a long way with turns drifts by centimetres)
NEAREST = .62               # a can nearer than this is cut by the lower edge of the head camera's image: its sighting is not used
GOOD_RANGE = 1.05           # a sighting from nearer than this places a can to half a centimetre (0.65-0.9 m: median 5 mm, at most
                            # 9-22 mm; from 1.1-1.6 m: median 10 mm, up to 35 mm, on recorded runs with true poses)
# Where the robot stands to look around when cans are missing: east of the coffee table, in the south and in the
# north of the living room (the table hides the floor right behind it from one of them, not from both), and at
# the south end of the gap between the sofa and the table (a can in that gap shows from nowhere else: from the
# east the table, 0.42 m high, hides it).
DRAWER_SPOT = tuple(drawer.open_part(drawer.PULL))    # the middle of the top drawer's open part: where the batteries are let go
# Lookouts: east of the coffee table, and north-east of it with the bar in view (none in the kitchen: see KITCHEN_Y).
LOOKOUTS = {'cans': ((.30, -.90), (.20, .55), (-.60, .55)), 'bin': ()}


class ObservedEstimator:
    """State estimates for the skills, fed by the Observer.

    The base pose in the map comes from the localiser. Cans and the trash can are remembered in the map frame
    (they stand still until the robot takes or pushes them); for the moment of contact the last sighting from
    close by is carried along by dead reckoning, so that an error of the map pose does not enter the pinch.
    """

    def __init__(self, observer, scene):
        self.ob = observer; self.scene = scene
        floors = scene.meta['scope']; self.floor = {'soda': floors['floor.n.01_1'], 'bin': floors['floor.n.01_1']}
        self.odom = np.zeros(3); self.commanded = 0.; self.velocity = np.zeros(2); self.held_up = 0; self.settling = False
        self.driven = 0.; self.turned = 0.          # how far the base has driven and turned in all (m, rad)
        self.tracks = {}; self.next_id = 1; self.source = None
        # The drawer is fixed: the batteries are let go over its open part.
        self.bin_track = dict(xy=np.array(DRAWER_SPOT, float), range=0., count=99, misses=0)
        self.drawer = 0.
        self.news = []              # cans that were found or given up since the script last asked: for the event log

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

    # ---- looking ---------------------------------------------------------------------------
    def _on_floor(self, kind, xy):
        lo, hi = np.array(self.floor[kind]['aabb_lo'][:2]), np.array(self.floor[kind]['aabb_hi'][:2])
        if kind == 'soda' and (float(xy[1]) > GLASS_NORTH or (float(xy[0]) < GLASS_WEST[0] and float(xy[1]) > GLASS_WEST[1])):
            return False
        return bool(np.all(xy > lo - ZONE) and np.all(xy < hi + ZONE))

    def _update(self, track, thing, distance, odom):
        """Take a sighting into a track: from close by it replaces what was known, from far off only a
        sighting from farther still."""
        track['count'] += 1; track['misses'] = 0
        if distance < FRESH or distance < 1.2 * track['range']:
            track['xy'] = thing['xy'].copy(); track['range'] = distance
            if 'grip' in thing:
                track['grip'] = thing['grip']; track['surface'] = thing['surface']
            if 'yaw' in thing:
                track['yaw'] = thing['yaw']
        # For the hand: the nearest sighting while the base has not moved much since; a farther one only replaces
        # one that has grown old.
        if distance < CLOSE and (not self._still_good(track) or distance < 1.15 * track['odom_range']):
            track['odom'] = odom[:2, :2] @ thing['point'] + odom[:2, 3]; track['odom_range'] = distance; track['odom_at'] = (self.driven, self.turned)
            if 'axis' in thing:
                track['odom_axis'] = thing['axis'] + self.odom[2]

    def _still_good(self, track):
        return 'odom' in track and self.driven - track['odom_at'][0] < STILL_GOOD[0] and self.turned - track['odom_at'][1] < STILL_GOOD[1]

    def see(self):
        """Take what the current head image shows of cans and of the trash can into the map-frame memory."""
        base = self.base(); odom = pose2(*self.odom); seen = {}; matched = set()
        for thing in self.ob.floor_things(base):
            lying = thing['kind'] == 'lying'
            if thing['kind'] is None or thing['cut'] or not self._on_floor('bin' if thing['kind'] == 'bin' else 'soda', thing['xy']):
                continue
            distance = float(np.linalg.norm(thing['point']))
            if thing['kind'] != 'bin' and distance < NEAREST:
                continue
            if thing['kind'] == 'bin':
                continue
                if self.bin_track is None:
                    self.bin_track = dict(xy=thing['xy'].copy(), range=distance, count=0, misses=0)
                self._update(self.bin_track, thing, distance, odom); seen['bin'] = thing['points']
                continue
            if self.bin_track is not None and float(np.linalg.norm(thing['xy'] - self.bin_track['xy'])) < (BIN_ZONE_LYING if lying else BIN_ZONE):
                continue
            # The nearest known can, if this sighting can be one of it: the allowance grows with the range of
            # the farther of the two sightings.
            gaps = {k: float(np.linalg.norm(track['xy'] - thing['xy'])) - MERGE[1] * max(track['range'], distance) for k, track in self.tracks.items()
                    if k not in matched and track['lying'] == lying}
            k = min(gaps, key=gaps.get) if gaps else None
            if k is None or gaps[k] > MERGE[0]:
                k = self.next_id; self.next_id += 1
                self.tracks[k] = dict(xy=thing['xy'].copy(), range=distance, count=0, misses=0, lying=lying)
                self.news.append(dict(can=k, lying=lying, found=thing['xy'].round(3).tolist(), base=np.round(base, 3).tolist(), point=thing['point'].round(3).tolist(),
                                      top=round(thing['top'], 3), width=round(thing['width'], 3), points=thing['points']))
            self._update(self.tracks[k], thing, distance, odom); matched.add(k); seen['soda'] = seen.get('soda', 0) + 1
            # A can stands or lies, not both: one that was seen standing here and now lies (it was knocked over), or
            # the other way round, is known as what it is now.
            for j in [j for j, other in self.tracks.items() if other['lying'] != lying and j not in matched and float(np.linalg.norm(other['xy'] - thing['xy'])) < SAME_CAN]:
                self.news.append(dict(can=j, now=k, lying=lying, was=self.tracks[j]['xy'].round(3).tolist())); del self.tracks[j]
            if distance < FRESH:
                # Seen from close by: what was taken for another can next to it from far off was this one.
                for j in [j for j, other in self.tracks.items() if j != k and j not in matched and other['range'] > FRESH and other['lying'] == lying
                          and float(np.linalg.norm(other['xy'] - thing['xy'])) < MERGE[0] + MERGE[1] * other['range']]:
                    self.news.append(dict(can=j, same_as=k, was=self.tracks[j]['xy'].round(3).tolist())); del self.tracks[j]
        # A can that should show in this image and does not is no longer there (taken for a can by mistake,
        # or knocked away).
        c, s = np.cos(base[2]), np.sin(base[2])
        for k in [k for k in self.tracks if k not in matched]:
            d = self.tracks[k]['xy'] - base[:2]; local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
            if EXPECTED[0][0] < local[0] and float(np.linalg.norm(local)) < EXPECTED[0][1] and abs(np.arctan2(local[1], local[0])) < EXPECTED[1]:
                self.tracks[k]['misses'] += 1
                if self.tracks[k]['misses'] >= MISSES:
                    self.news.append(dict(can=k, gone=self.tracks[k]['xy'].round(3).tolist(), seen=self.tracks[k]['count'])); del self.tracks[k]
        return seen

    def known(self):
        """Numbers of the cans that are known to stand somewhere."""
        return sorted(k for k, track in self.tracks.items() if track['count'] >= CONFIRMED)

    def soda(self, k):
        track = self.tracks.get(k)
        return None if track is None or track['count'] < CONFIRMED else track['xy'].copy()

    def bin(self):
        return None if self.bin_track is None or self.bin_track['count'] < CONFIRMED else self.bin_track['xy'].copy()

    def soda_grip(self, k):
        """Height (m above the floor) at which the fingertips' centre closes on battery @k: 2 cm under its top."""
        track = self.tracks.get(k)
        return None if track is None else track.get('grip')

    def soda_surface(self, k):
        track = self.tracks.get(k)
        return None if track is None else track.get('surface')

    def mark_taken(self, k):
        self.tracks.pop(k, None)

    def forget(self, k):
        """The hand closed on nothing where can @k was believed to be: it is not known any more until it is seen again."""
        track = self.tracks.pop(k, None)
        if track is not None:
            self.news.append(dict(can=k, forgotten=track['xy'].round(3).tolist()))

    def fresh(self, k):
        """True if can @k has been seen from within GOOD_RANGE and the base has not moved much since (STILL_GOOD)."""
        track = self.tracks.get(k)
        return track is not None and self._still_good(track) and track['odom_range'] < GOOD_RANGE

    def lying(self, k):
        """For a can that has fallen over: the direction of its axis in the map (rad, modulo half a turn); else None."""
        track = self.tracks.get(k)
        return track.get('yaw') if track is not None and track['lying'] else None

    def soda_axis(self, k):
        """The same in the base frame, for the hand: from the last sighting close by, carried by dead reckoning."""
        track = self.tracks.get(k)
        if track is None or not track['lying']:
            return None
        return float(wrap(track['odom_axis'] - self.odom[2])) if 'odom_axis' in track else float(wrap(track['yaw'] - self.base()[2]))

    def _point(self, track):
        """A remembered thing in the base frame: its last sighting from close by, carried along by dead
        reckoning (over the last metre its error is a few millimetres, whatever the map pose does meanwhile);
        without one, its place in the map seen from the map pose."""
        if track is None:
            return None
        if 'odom' in track:
            self.source = 'odom'
            return (np.linalg.inv(pose2(*self.odom)) @ np.r_[track['odom'], 0., 1.])[:2]
        self.source = 'map'
        return (np.linalg.inv(pose2(*self.base())) @ np.r_[track['xy'], 0., 1.])[:2]

    def soda_point(self, k):
        return self._point(self.tracks.get(k))

    def gap(self, ignore=()):
        """The two sides of a gap between furniture straight ahead, as the head camera shows them: a line for each
        (see Observer.gap_sides)."""
        return self.ob.gap_sides(ignore=ignore)

    def bin_point(self):
        return self._point(self.bin_track)


class Runtime:
    """Per-step glue: feeds observations to the Observer, keeps the localiser running, steps the script."""

    def __init__(self, ctx, observer, estimator, script):
        self.ctx = ctx; self.ob = observer; self.est = estimator; self.script = script(self)
        self.last_action = np.zeros(23); self.moving = 0; self.finished = None
        self.looked = {}            # lookouts that showed nothing, per kind of survey

    def step(self, data):
        s = data['robot_r1::proprio'].astype(float)
        self.ctx.update(s)
        self.ob.frame(data)
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
            # Keep looking for cans on the way, with the trunk upright (bent over a can the camera looks at the
            # hands) and not in a fast turn (the heading then lags, and with it everything seen far off).
            if self.moving and upright and abs(self.last_action[2]) < .3 and self.ob.steps % 5 == 0:
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


def localize(rt):
    """Global localisation in the kitchen by looking around on the spot. Returns True when a pose was accepted."""
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
        if pose is not None and len(scans) >= 3 and score > .5 and (margin > .06 or len(scans) >= 4):
            shift = lambda p: np.array([p[0], p[1], wrap(p[2] + turned)])
            ob.loc.start_tracking(shift(pose), [shift(p) for _, p in ob.loc.shortlist] if margin <= .12 else [])
            ob.loc.correct(ob.scan())
            return True
    return False


def look_around(rt):
    """A full turn on the spot in quarters, taking in each view. Returns what each view showed."""
    ctx, est = rt.ctx, rt.est; seen = []
    for quarter in range(5):
        yield from hold(ctx, 3)
        seen.append(est.see())
        if quarter < 4:
            yield from turn_in_place(rt, np.pi / 2)
    return seen


def survey(rt, where):
    """Look around for what the next stage needs and tell the estimator.

    'start': a full turn where the robot stands. 'cans' / 'bin': from the lookouts of the living room / of the
    kitchen, until something is known ('cans': a can; 'bin': the trash can).
    """
    ctx, est = rt.ctx, rt.est; views = []
    if where == 'relocalize':
        # (x02) a fresh global fix where the map pose may have drifted (the passage into the kitchen)
        before = tuple(np.round(est.base(), 3))
        ok = yield from localize(rt)
        ctx.event('relocalize', ok=bool(ok), before=list(before), after=list(np.round(est.base(), 3)))
        return
    if where == 'start':
        views += yield from look_around(rt)
    else:
        # Nearest first; a lookout that showed nothing is not gone to again before the others have been tried.
        here = np.asarray(est.base()[:2], float); done = rt.looked.setdefault(where, set()); before = set(est.known())
        if len(done) >= len(LOOKOUTS[where]):
            done.clear()
        for spot in sorted([xy for xy in LOOKOUTS[where] if xy not in done], key=lambda xy: float(np.linalg.norm(np.asarray(xy) - here))):
            yaw = float(np.arctan2(spot[1] - est.base()[1], spot[0] - est.base()[0]))
            status = yield from route(ctx, np.asarray(spot, float), yaw, keepout=lambda: others(est), tolerance=.10, est=est)
            if status == 'no_route':
                # A piece of glass seen in a passage shuts it with its keep-out disc (start 305): the base may
                # pass over a piece 4.5 cm high.
                status = yield from route(ctx, np.asarray(spot, float), yaw, keepout=lambda: [], tolerance=.10, est=est)
            if status != 'arrived':
                ctx.event('lookout', ok=False, where=where, spot=list(spot), status=status); continue
            views += yield from look_around(rt)
            done.add(spot)
            # Pieces left alone after failed pinches stay known: only a piece not known before ends the round.
            if (where == 'cans' and set(est.known()) - before) or (where == 'bin' and est.bin() is not None):
                break
    ctx.event('survey', where=where, cans={k: est.soda(k).round(3).tolist() for k in est.known()}, bin=None if est.bin() is None else est.bin().round(3).tolist(),
              views=[{k: int(v) for k, v in row.items()} for row in views], rejects=rt.ob.rejects if where != 'cans' and est.bin() is None else [])


def observed_pipeline(rt):
    import dish_pipeline
    rt.est.see = lambda *a, **k: {}         # (the battery tracker is not used: dish_pipeline keeps its own list of plates)
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    result = yield from dish_pipeline.full_task(rt)
    return result
