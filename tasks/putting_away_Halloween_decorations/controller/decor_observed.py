"""Putting away the Halloween decorations from what the robot's own cameras show: the estimator and the entry point.

The base pose and the per-step runtime are those of the controllers for the car tasks (car_observed.py), with the map
of this task's scene. What is new here is what is known of the six things on the living-room floor (decor_view.py):
two pumpkins, three candles and the caldron, each remembered where the nearest look had it.
"""
import numpy as np
from manip import TALL_TRUNK, pose2
from car_observed import ObservedEstimator, localize
from observer import HEAD, WRIST, SLOT, FOCAL
from localizer import camera_points, BASE_Z
from decor_view import DecorView

KINDS = ('pumpkin', 'candle', 'caldron')
COUNT = dict(pumpkin=2, candle=3, caldron=1)
NEAR = 1.6                          # a look from nearer than this replaces what was known from further away
SAME = dict(pumpkin=.08, candle=.07, caldron=.15)       # a sighting this near to a known thing of its kind is that thing
GONE_AFTER = 2                      # a thing that looks from near do not show where it was known this many times is forgotten
STILL_GOOD = (1.5, 2.0)             # a sighting is carried by dead reckoning while the base has since driven less than this (m) and turned less (rad)


class DecorEstimator(ObservedEstimator):
    def __init__(self, observer, scene, folder):
        super().__init__(observer, scene)
        self.view = DecorView(scene); self.items = {}; self.next_id = 0; self.looks = 0
        self.taken_ids = set()      # in a hand or in a drawer: not looked for any more
        self.ignore = []            # boxes (x0, y0, x1, y1) of the floor not looked at: open drawers
        self.switched = None

    # (The things are looked for here, not by the car tasks' detectors.)
    def see(self, what=()):
        return {}

    def predict(self, s, last_action, dt=1 / 30):
        super().predict(s, last_action, dt)
        switched = self.ob.loc.debug.get('switched_to') if getattr(self.ob.loc, 'debug', None) else None
        if switched is not None and switched != self.switched:
            self.news.append(dict(start_pose_switched_to=switched))
        self.switched = switched
        # On the way, with the trunk upright and not in a fast turn, every sixth frame is looked at for the things.
        if self.ob.loc.pose is not None and self.ob.steps % 6 == 0 and abs(float(last_action[2])) < .3 and float(np.max(np.abs(np.asarray(s, float)[53:57] - TALL_TRUNK))) < .06:
            self.look()

    def known(self, kind=None):
        """The things known and not yet taken: [item dict], of @kind if given."""
        return [item for item in self.items.values() if kind is None or item['kind'] == kind]

    def look(self):
        """Take what the head camera shows into what is known. Returns what this look showed."""
        base = self.base(); self.looks += 1
        seen = self.view.see(self.ob, base, ignore=self.ignore, image=self.ob.data[HEAD + 'rgb'][..., :3])
        matched = set()
        near = lambda new: min([float(np.hypot(*(i['xy'] - new['xy']))) for i in self.items.values() if i['kind'] == new['kind']] or [9.])
        for new in sorted(seen, key=near):          # (sightings of known things first)
            kind = new['kind']; best = None
            for item in self.items.values():
                if item['kind'] != kind or item['id'] in matched:
                    continue
                d = float(np.hypot(*(item['xy'] - new['xy'])))
                if d < SAME[kind] + .02 * min(item['range'], new['range']) and (best is None or d < best[1]):
                    best = (item, d)
            if best is None:
                there = [i for i in self.known(kind) if i['id'] not in matched]
                if kind == 'caldron' and there and not new['cut']:
                    # (there is one caldron, and nothing else looks like it: seen elsewhere, it was pushed there - by the
                    # chassis, 0.24 m, in 1 run of 20)
                    item = there[0]; matched.add(item['id']); item.update(xy=np.asarray(new['xy'], float), range=new['range'], step=self.ob.steps, n=new['n'], misses=0)
                    for key in ('odom_xy', 'odom_range', 'odom_at'):
                        item.pop(key, None)
                    self.news.append(dict(moved=kind, xy=np.round(item['xy'], 3).tolist(), range=round(new['range'], 2)))
                    continue
                if len(self.known(kind)) + sum(1 for i in self.taken_ids if i[0] == kind) >= COUNT[kind]:
                    continue                    # (as many of that kind are known as there are: a second sighting of one)
                item = dict(id=(kind, self.next_id), kind=kind, xy=np.asarray(new['xy'], float), range=new['range'], step=self.ob.steps, n=new['n'], misses=0)
                self.next_id += 1; self.items[item['id']] = item; matched.add(item['id'])
                self.news.append(dict(seen=kind, xy=np.round(item['xy'], 3).tolist(), range=round(new['range'], 2)))
                continue
            item = best[0]; matched.add(item['id']); item['misses'] = 0
            if new['range'] < max(item['range'], NEAR) and not new['cut']:
                item.update(xy=np.asarray(new['xy'], float), range=new['range'], step=self.ob.steps, n=new['n'])
            # ... and where it is from the dead-reckoned pose, for the moment the hand goes to it: the map pose is put
            # right by the localiser on the way in, by up to 3 cm, and the pinch of a 5 mm wick has a few millimetres
            # to spare (the hand closed on nothing at the first whole runs).
            if not new['cut'] and new['range'] <= item.get('odom_range', 9.) + .3 or not self.fresh(item):
                c, s_ = np.cos(base[2]), np.sin(base[2]); d = np.asarray(new['xy'], float) - np.asarray(base[:2], float)
                local = np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]])
                item.update(odom_xy=(pose2(*self.odom) @ np.r_[local, 0., 1.])[:2], odom_range=new['range'], odom_at=(self.driven, self.turned))
        # What a look from near does not show where it was known has been moved (pushed, knocked over).
        for item in list(self.items.values()):
            if item['id'] not in matched and self.in_view(item['xy'], base):
                item['misses'] += 1
                if item['misses'] >= GONE_AFTER:
                    self.news.append(dict(lost=item['kind'], at=np.round(item['xy'], 3).tolist())); self.items.pop(item['id'])
        return seen

    def fresh(self, item):
        return 'odom_xy' in item and self.driven - item['odom_at'][0] < STILL_GOOD[0] and self.turned - item['odom_at'][1] < STILL_GOOD[1]

    def relative(self, item):
        """Where @item is in the base frame now: from its latest near sighting by dead reckoning if that is fresh,
        else from the map."""
        if self.fresh(item):
            return (np.linalg.inv(pose2(*self.odom)) @ np.r_[item['odom_xy'], 0., 1.])[:2]
        base = self.base(); c, s = np.cos(base[2]), np.sin(base[2]); d = np.asarray(item['xy'], float) - np.asarray(base[:2], float)
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    @staticmethod
    def in_view(xy, base, nearest=1.0, farthest=2.6):
        """Does the head camera show the floor at map @xy from base pose @base well?"""
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float) - [x, y]; ahead = c * d[0] + s * d[1]; aside = -s * d[0] + c * d[1]
        return bool(nearest < ahead < farthest and abs(np.arctan2(aside, ahead - .24)) < np.radians(36.))

    # What the wrist camera's look leaves between the fingers' line and the wick (the stem) once the hand has been put
    # right by it: along the line of sight from the base and to its left (m), measured against the true poses on 51
    # candle pinches and 43 pumpkin pinches of whole runs (sd 1.0/0.6 mm for the wick, 11/7 mm for the stem). With
    # the wick 2.8 mm to the side of the fingers' middle the simulator's assisted grasp took it on 1 pinch in 3
    # (against 26 of 28 with it in the middle), and the candle fell when it was lifted.
    WRIST_BIAS = dict(candle=(-.0008, .0028), pumpkin=(-.0028, .0023))

    def wrist_look(self, arm, kind, tip):
        found = self._wrist_look(arm, kind, tip)
        if found is None:
            return None
        sight = float(np.arctan2(tip[1], tip[0])); along = np.array([np.cos(sight), np.sin(sight)]); across = np.array([-along[1], along[0]])
        a, b = self.WRIST_BIAS[kind]
        return np.asarray(found, float) + a * along + b * across

    def _wrist_look(self, arm, kind, tip):
        """Where the stem (the wick) is as the wrist camera of @arm shows it from above: base-frame xy, or None.
        tip: the fingers' ends now (base frame), some 10 cm above it. Seen from straight above the candle's flat top is
        a whole disc (its middle is the wick); the pumpkin's stem is what stands above its body."""
        points = camera_points(self.ob.data[WRIST[arm] + 'depth_linear'], self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]], FOCAL[arm], stride=2, max_range=1.)
        self.wrist_debug = dict(points=len(points))
        if len(points) < 50:
            return None
        tip = np.asarray(tip, float); z = points[:, 2] + BASE_Z; near = np.linalg.norm(points[:, :2] - tip[:2], axis=1)
        if kind == 'candle':
            top = points[(near < .09) & (z > .108) & (z < .128)]
            self.wrist_debug.update(top=len(top), z_near=np.round(np.percentile(z[near < .09], [5, 50, 95]), 3).tolist() if (near < .09).any() else None)
            if len(top) < 40:
                return None
            lo, hi = np.percentile(top[:, :2], 2, axis=0), np.percentile(top[:, :2], 98, axis=0)
            self.wrist_debug.update(extent=np.round(hi - lo, 3).tolist())
            if np.any(hi - lo > .125) or np.any(hi - lo < .07):
                return None                     # (not one whole top)
            return .5 * (lo + hi)
        # (the band the fingers close in: the body ends at 0.098-0.102 where the pumpkin leans, and the stem's top
        # stands a centimetre further over than its middle)
        # The stem's top: the highest points near the hand (from above the stem shows as a small blob 2-2.5 cm above the
        # body round it; the band of fixed heights took in the body where the pumpkin leans, and the hand was put 1-2 cm off).
        close = (near < .07) & (z < .16)
        if close.sum() < 20:
            self.wrist_debug.update(stem=0); return None
        peak = float(np.percentile(z[close], 99.5))
        stem = points[close & (z > peak - .008)]
        extent = np.ptp(stem[:, :2], axis=0) if len(stem) else np.zeros(2)
        self.wrist_debug.update(stem=len(stem), peak=round(peak, 3), extent=np.round(extent, 3).tolist())
        # (a stem's top up to 0.137 m was seen: 6 pinches of one pumpkin in 1 run of 20 went in without this look)
        if len(stem) < 6 or not .110 < peak < .150 or np.any(extent > .045):
            return None
        return stem[:, :2].mean(axis=0)

    def wrist_floor(self, arm, tip):
        """Height of what the wrist camera of @arm shows under the hand (within 6 cm of @tip, base frame), or None:
        a drawer's floor (0.083) where the drawer is out, the room's floor where it is not."""
        points = camera_points(self.ob.data[WRIST[arm] + 'depth_linear'], self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]], FOCAL[arm], stride=2, max_range=1.2)
        if len(points) < 50:
            return None
        under = points[np.linalg.norm(points[:, :2] - np.asarray(tip, float)[:2], axis=1) < .06]
        return float(np.median(under[:, 2] + BASE_Z)) if len(under) >= 30 else None

    def wrist_face(self, arm, tip, heading=0.):
        """Where the drawer's lip is as the wrist camera of @arm shows the drawer's front from the hand that is about to
        take the lip (level, its fingers along @heading, base frame, 8 cm short of it): base-frame xy of the lip's
        middle in its depth for the fingers' ends, the hand's sideways place kept; or None. The front below the lip is
        a flat face square to the fingers; the lip stands out of it by 2 cm."""
        points = camera_points(self.ob.data[WRIST[arm] + 'depth_linear'], self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]], FOCAL[arm], stride=2, max_range=.8)
        if len(points) < 50:
            return None
        tip = np.asarray(tip, float); along = np.array([np.cos(heading), np.sin(heading)]); across = np.array([-along[1], along[0]])
        d = points[:, :2] - tip[:2]; a = d @ along; b = d @ across; z = points[:, 2]
        face = points[(np.abs(b) < .12) & (a > -.02) & (a < .30) & (z < tip[2] - .02) & (z > tip[2] - .12)]
        if len(face) < 40:
            return None
        depth = float(np.median((face[:, :2] - tip[:2]) @ along))
        if np.percentile((face[:, :2] - tip[:2]) @ along, 90) - np.percentile((face[:, :2] - tip[:2]) @ along, 10) > .03:
            return None                         # (not one flat face)
        return tip[:2] + (depth - .014) * along

    def taken(self, item_id):
        """The thing is in a hand or in a drawer: it is not looked for any more."""
        self.items.pop(item_id, None); self.taken_ids.add(item_id)

    def put_back(self, item_id, xy):
        """A thing a hand had is on the floor again (it fell): looked for again near @xy."""
        self.taken_ids.discard(item_id)
        kind = item_id[0]; self.items[item_id] = dict(id=item_id, kind=kind, xy=np.asarray(xy, float), range=9., step=self.ob.steps, n=0, misses=0)


def observed_pipeline(rt):
    from decor_pipeline import full_task
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    return (yield from full_task(rt.ctx, rt.est))
