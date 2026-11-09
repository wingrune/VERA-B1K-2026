"""The battery task driven by onboard observations: localise, look for the can and the batteries, run the skills.

Everything the skills need about the world comes from Observer (map-based localisation, blue battery caps,
the can as an unmapped object on the copy-room floor). No simulator state is read here.
"""
import numpy as np
from observer import Observer, pose2, wrap, BASE_Z
from manip import hold
from battery_skills import GRASP_HEIGHT, CAN_TOP
from battery_pipeline import full_task, surfaces_of, straight, TURN_SPOT, NORTH
from localizer import turn_rate, true_velocity, command_velocity

LOOSE = .25                 # a known battery is recognised this far from where the map pose puts it (m)
MERGE = .07                 # sightings of one battery lie within this distance of each other (m)
WRIST_REACH = .16            # the wrist camera's cap is taken for the battery up to this far from where it was expected (m)
CAP_SIDE = .013             # a cap seen side-on by the wrist camera: the battery's axis lies this far behind its visible face
CAP_BACK = .011             # a cap seen from the side: its centre lies this far behind the visible surface


def surface_holds(surface, xy, slack=.04):
    lo, hi = np.asarray(surface['lo'], float), np.asarray(surface['hi'], float)
    return bool(np.all(xy > lo - slack) and np.all(xy < hi + slack))


WALL_END_AGREES = .025     # the wall end is read to about a centimetre: a pose that agrees with it this well is left alone


class ObservedEstimator:
    """State estimates for the skills, fed by the Observer.

    The base pose in the map comes from the localiser. Batteries and the can are remembered in the map
    frame (they stand still until the robot takes or pushes them); at the moment of contact their position
    is measured again in the base frame, so that an error of the map pose does not enter the grasp.
    """

    def __init__(self, observer, scene):
        self.ob = observer; self.surfaces = surfaces_of(scene); self.floor = scene.meta['start_floor']
        self.odom = np.zeros(3); self.tracks = {'cabinet': [], 'desk': []}; self.ids = {}; self.can_sightings = []; self.can_yaws = []; self.taken = set(); self.settling = False
        self.edge = None; self.edge_reads = 0; self.edge_last = None; self.edge_slides = 0; self.edge_slide = None; self.commanded = 0.; self.velocity = np.zeros(2); self.held_up = 0; self.slowed = 0

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
            # Slowed down = pushing a chair along. Forwards and backwards the report is good to a few per cent
            # and reads a little high in free motion: where it stays under nine tenths of the command, it is
            # what the base does (the odometry that carries a battery's last sighting to the stance was 10 cm
            # long after a chair had been pushed on the way). Sideways, and for the forward part of a mostly
            # sideways move, the report is no use for this.
            if abs(model[0]) > .03 and abs(model[0]) > 2. * abs(model[1]) and abs(report[0]) < .9 * abs(model[0]) and np.sign(report[0]) * np.sign(model[0]) >= 0:
                self.slowed += 1
            else:
                self.slowed = 0
            if self.slowed >= 6 and self.held_up < 6:
                self.velocity[0] = report[0]
        else:
            # Coasting: the report, limited to what the last commands could have produced (wedged against
            # furniture and tilted, the base reported speeds it did not have, and the pose went 25 m away in a
            # minute). Turning on the spot the report shows a drift of 6 mm/s that is not there: with no
            # translation commanded for a second, none is counted.
            self.held_up = 0; limit = 1.3 * self.commanded + .05
            self.velocity = report * min(1., limit / max(speed, 1e-9)) if self.commanded > .01 else np.zeros(2)
        vx, vy = float(self.velocity[0]), float(self.velocity[1]); wz = turn_rate(last_action[2], driving=np.hypot(last_action[0], last_action[1]) > .06)
        yaw = self.odom[2] + .5 * wz * dt; c, sn = np.cos(yaw), np.sin(yaw)
        self.odom = np.array([self.odom[0] + (c * vx - sn * vy) * dt, self.odom[1] + (sn * vx + c * vy) * dt, self.odom[2] + wz * dt])

    def base(self):
        return tuple(self.ob.loc.pose)

    def hold_axis(self, axis):
        """Keep map coordinates by dead reckoning (@axis: 'x', 'y', 'xy' or None): for the passage through a doorway
        or past the desk, where a few centimetres matter and what the cameras see there (a door leaf swinging, a
        chair being pushed) is not what the map has. The heading is still fitted."""
        self.ob.loc.lock = None if axis is None else np.array([float('x' in axis), float('y' in axis)])

    def align_to_wall_end(self, end_y):
        """Set the map y of the pose from the end of the wall seen ahead on the left, whose map y is @end_y.
        Returns the shift applied (m), or None if the end is not in view."""
        seen = self.ob.wall_end()
        if seen is None:
            return None
        pose = self.ob.loc.pose; c, s = np.cos(pose[2]), np.sin(pose[2])
        y = end_y - s * seen[0] - c * seen[1]; shift = float(y - pose[1])
        if abs(shift) > .25:
            return None
        if abs(shift) > WALL_END_AGREES:
            self.ob.loc.pose = np.array([pose[0], y, pose[2]])
        return shift

    def settle(self, flag):
        """Let the localiser correct the pose while the base stands still (it normally does so only in motion)."""
        self.settling = bool(flag)

    def follow_edge(self, edge):
        """Take the pose from an edge of the desk top instead of from scan matching (@edge: dict(point, normal,
        height) in M, the normal pointing off the table to where the robot is; None ends it).

        North of the desk and beside its end the scan match is off by up to 15 cm and 3 deg (little of the map is
        in view there, and chairs that have been pushed about are), while the passage past the desk's end leaves
        2 cm to either side. The edge of the desk top is in plain view there: it gives the coordinate across it
        and the heading to a centimetre and half a degree. The coordinate along it is dead reckoned meanwhile.
        """
        self.edge = edge; self.ob.loc.frozen = edge is not None; self.edge_reads = 0; self.edge_last = None; self.edge_slides = 0; self.edge_slide = None

    def read_edge(self, gain=.5):
        """One reading of the followed edge into the pose. Returns (shift across the edge, turn) or None if it is not in view."""
        edge = self.edge; pose = self.ob.loc.pose; n = np.asarray(edge['normal'], float)
        toward = wrap(np.arctan2(-n[1], -n[0]) - pose[2])
        if abs(toward) > np.radians(100):
            return None
        line = self.ob.table_edge(edge['height'], toward=toward)
        if line is None:
            return None
        a, b, strips, ends = line; turn = float(np.arctan(b)); shift = float(n @ np.asarray(edge['point'], float) + a * np.cos(turn) - n @ pose[:2])
        if abs(turn) > np.radians(12) or abs(shift) > .30:
            return None
        if strips < 16:          # a short piece of edge: good for the position across it, not for the heading
            turn = 0.
        pose = np.array([pose[0] + gain * shift * n[0], pose[1] + gain * shift * n[1], wrap(pose[2] + gain * turn)])
        # An end of the edge in view gives the coordinate along it as well (dead reckoned otherwise: pushing the
        # chair sideways along the desk, that came out 20 cm long).
        w = np.array([n[1], -n[0]]); slide = None
        if abs(turn) < np.radians(1.5) and abs(shift) < .02:
            for end, corner in zip(ends, edge.get('ends', (None, None))):
                if end is not None and corner is not None:
                    slide = float(w @ np.asarray(corner, float) - end - w @ pose[:2])
            if slide is not None and abs(slide) < .45:
                pose[:2] += gain * slide * w; self.edge_slides += 1; self.edge_slide = slide
        self.ob.loc.pose = pose
        self.edge_reads += 1; self.edge_last = (shift, turn)
        return shift, turn

    # ---- looking ---------------------------------------------------------------------------
    def see(self):
        """Take what the current head image shows of batteries and of the can into the map-frame memory."""
        base = self.base(); seen = {}; odom = pose2(*self.odom)
        for name, surface in self.surfaces.items():
            tracks = self.tracks[name]; full = len(tracks) >= (1 if name == 'cabinet' else 2); used = []
            for b in self.ob.batteries(base, surface, slack=LOOSE if full else .04):
                gaps = [float(np.linalg.norm(track['xy'] - b['xy'])) if not any(track is u for u in used) else np.inf for track in tracks]
                nearest = int(np.argmin(gaps)) if gaps else None
                if nearest is not None and gaps[nearest] < MERGE:
                    track = tracks[nearest]; track['sum'] += b['xy']; track['count'] += 1; track['xy'] = track['sum'] / track['count']
                elif full and nearest is not None and gaps[nearest] < LOOSE:
                    # A battery that is known, seen where the map pose is off (behind the desk it can be by a
                    # decimetre): its place in the map is left alone, the sighting itself is kept (below).
                    track = tracks[nearest]
                elif surface_holds(surface, b['xy']):
                    track = dict(xy=b['xy'].copy(), sum=b['xy'].copy(), count=1); tracks.append(track)
                else:
                    continue
                used.append(track)
                # The same sighting relative to dead-reckoned odometry, which is smooth: over the last metre to
                # the battery its error is about a centimetre, whatever the map pose does meanwhile.
                track['odom'] = odom[:2, :2] @ self._cap_centre(b['point']) + odom[:2, 3]; track['odom_step'] = self.ob.steps
                seen[name] = seen.get(name, 0) + 1
        can = self.ob.can(base, self.floor)
        if can is not None and can['rim'] >= 12:
            self.can_sightings.append(can['xy']); seen['can'] = can['points']
            if can['yaw'] is not None:
                self.can_yaws.append(can['yaw'])
        return seen

    def _cap_centre(self, point):
        """Battery axis (base frame xy) from a cap seen from the side: a little behind the visible surface."""
        away = point[:2] - self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[2][:2]
        return point[:2] + CAP_BACK * away / max(np.linalg.norm(away), 1e-9)

    def _track(self, k):
        """The remembered battery @k: 3 = the one on the cabinet, 1 and 2 = those on the desk."""
        if k in self.taken:
            return None, None
        name = 'cabinet' if k == 3 else 'desk'
        if k not in self.ids:
            # Numbers are given once, to the tracks seen most often.
            free = sorted([t for t in self.tracks[name] if not any(t is u for u in self.ids.values())], key=lambda t: -t['count'])
            for number in ([3] if name == 'cabinet' else [1, 2]):
                if number not in self.ids and free:
                    self.ids[number] = free.pop(0)
        return self.ids.get(k), self.surfaces[name]

    def battery(self, k):
        track, surface = self._track(k)
        return None if track is None else np.r_[track['xy'], surface['height'] + GRASP_HEIGHT]

    def can(self):
        if not self.can_sightings:
            return None
        return np.r_[np.median(np.array(self.can_sightings[-7:]), axis=0), CAN_TOP / 2]

    def mark_taken(self, k):
        self.taken.add(k)

    def forget_can(self):
        """The can has been moved, or is about to be looked at from close by: what was seen before no longer counts."""
        self.can_sightings = []; self.can_yaws = []

    def can_yaw(self):
        """Heading of the can's long side in the map (it is a rectangle, so modulo half a turn), or None."""
        if not self.can_yaws:
            return None
        doubled = 2. * np.array(self.can_yaws[-7:])          # average directions that are equal modulo pi
        return float(.5 * np.arctan2(np.mean(np.sin(doubled)), np.mean(np.cos(doubled))))

    # ---- at the moment of contact ------------------------------------------------------------
    def battery_point(self, k, surface):
        """Grasp point of battery @k in the base frame: measured now if the head camera sees it, else from memory."""
        track, _ = self._track(k)
        if track is None:
            return None
        base = self.base(); remembered = (np.linalg.inv(pose2(*base)) @ np.r_[track['xy'], 0., 1.])[:2]; best = None
        recent = 'odom' in track and self.ob.steps - track['odom_step'] < 900
        carried = (np.linalg.inv(pose2(*self.odom)) @ np.r_[track['odom'], 0., 1.])[:2] if recent else None
        # Where to expect it: from the last look carried by odometry (good to a centimetre or two), else from the
        # map, which is only as good as the map pose is here.
        expected, reach = (carried, .08) if recent else (remembered, LOOSE)
        for b in self.ob.batteries(base, surface, slack=LOOSE):
            gap = float(np.linalg.norm(self._cap_centre(b['point']) - expected))
            if gap < reach and (best is None or gap < best[0]):
                best = (gap, b['point'])
        if best is not None:
            xy = self._cap_centre(best[1]); source = 'seen now'
            world = pose2(*base); seen = world[:2, :2] @ xy + world[:2, 3]
            if np.linalg.norm(seen - track['xy']) < MERGE:
                track['sum'] += seen; track['count'] += 1; track['xy'] = track['sum'] / track['count']
            track['odom'] = pose2(*self.odom)[:2, :2] @ xy + pose2(*self.odom)[:2, 3]; track['odom_step'] = self.ob.steps
        elif recent:
            xy = carried; source = 'seen %d steps ago' % (self.ob.steps - track['odom_step'])
        else:
            xy = remembered; source = 'map memory'
        self.source = source
        return np.array([xy[0], xy[1], surface['height'] + GRASP_HEIGHT - BASE_Z])

    def wrist_point(self, arm, point):
        """The battery cap as the wrist camera of @arm sees it, near the expected grasp @point (base frame), or None."""
        best = None; camera = self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[0 if arm == 'left' else 1][:3]
        for blob in self.ob.blue_blobs(arm, minimum=30):
            p = blob['point']; gap = float(np.linalg.norm(p[:2] - np.asarray(point, float)[:2]))
            if gap < WRIST_REACH and abs(p[2] - point[2]) < .08 and (best is None or gap < best[0]):
                best = (gap, p)
        if best is None:
            return None
        # The blob is the part of the cap that faces the camera: seen from above its middle is the battery's axis,
        # seen from the side the axis lies behind it by most of the cap's radius.
        p = best[1]; view = (p - camera) / max(float(np.linalg.norm(p - camera)), 1e-9)
        return np.array([p[0] + CAP_SIDE * view[0], p[1] + CAP_SIDE * view[1], point[2]])

    def can_point(self):
        """Centre of the can's opening in the base frame: measured now if the head camera sees its rim, else from memory."""
        remembered = self.can()
        if remembered is None:
            return None
        base = self.base(); can = self.ob.can(base, self.floor); xy = remembered[:2]
        if can is not None and can['rim'] >= 20 and np.linalg.norm(can['xy'] - remembered[:2]) < .20:
            xy = can['xy']; self.can_sightings.append(can['xy'])
        p = (np.linalg.inv(pose2(*base)) @ np.r_[xy, 0., 1.])[:2]
        return np.array([p[0], p[1], CAN_TOP - BASE_Z])


class Runtime:
    """Per-step glue: feeds observations to the Observer, keeps the localiser running, steps the script."""

    def __init__(self, ctx, observer, estimator, script):
        self.ctx = ctx; self.ob = observer; self.est = estimator; self.script = script(self)
        self.last_action = np.zeros(23); self.moving = 0; self.finished = None

    def step(self, data):
        s = data['robot_r1::proprio'].astype(float)
        self.ctx.update(s)
        self.ob.frame(data)
        self.est.predict(s, self.last_action)
        loc = self.ob.loc
        if loc.pose is not None:
            loc.predict(self.last_action[:3], measured=self.est.velocity)
            self.moving = 18 if np.any(np.abs(self.last_action[:3]) > .01) else max(0, self.moving - 1)
            # Correct while the base moves (and shortly after). A standing base keeps its pose: scans
            # taken while the arm works near the camera must not nudge it. The script may ask for it.
            if self.moving or self.est.settling:
                if self.est.edge is not None:
                    if self.ob.steps % 2 == 0:
                        self.est.read_edge()
                elif self.ob.steps % 3 == 0:
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
    """Global localisation in the copy room by looking around on the spot. Returns True when a pose was accepted."""
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
        # The copy room looks the same all along its cabinet: two views can agree on a pose that is shifted
        # along it, so at least three (a full turn) are taken before a pose is accepted.
        if pose is not None and len(scans) >= 3 and score > .5 and (margin > .06 or len(scans) >= 4):
            shift = lambda p: np.array([p[0], p[1], wrap(p[2] + turned)])
            ob.loc.start_tracking(shift(pose), [shift(p) for _, p in ob.loc.shortlist] if margin <= .12 else [])
            ob.loc.correct(ob.scan())
            return True
    return False


def survey(rt, where):
    """Look around for what the next stage needs and tell the estimator."""
    ctx, est = rt.ctx, rt.est; seen = []
    if where in ('copy_room', 'can'):
        # A full turn on the spot: the room is 3.5 m by 1.7 m, everything in it comes into view.
        for quarter in range(5):
            yield from hold(ctx, 3)
            seen.append(est.see())
            if quarter < 4:
                yield from turn_in_place(rt, np.pi / 2)
        can = est.can(); battery = est.battery(3)
        ctx.event('survey', where=where, can=None if can is None else can[:2].round(3).tolist(), battery=None if battery is None else battery[:2].round(3).tolist(),
                  views=[{k: int(v) for k, v in row.items()} for row in seen])
    else:
        # From the turning spot inside the office the desk spans more than one view: look north, then north-west.
        for yaw in (NORTH, NORTH + .75, NORTH):
            yield from straight(ctx, TURN_SPOT, yaw)
            yield from hold(ctx, 3)
            seen.append(est.see())
        ctx.event('survey', where=where, batteries=[None if est.battery(k) is None else est.battery(k)[:2].round(3).tolist() for k in (1, 2)],
                  views=[{k: int(v) for k, v in row.items()} for row in seen])


def observed_pipeline(rt):
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    result = yield from full_task(rt.ctx, rt.est, survey=lambda where: survey(rt, where))
    return result
