"""Cleaning up the plates and the food from what the robot's own cameras show: the estimator and the entry point.

The base pose and the per-step runtime are those of the earlier controllers (car_observed.py), with the map of this
task's scene. What is new here is what is known of the four things on the breakfast table (table_view.py): two plates,
each with its pizza, and two bowls, each remembered where the nearest look had it.
"""
import numpy as np
from manip import TALL_TRUNK, pose2, BASE_Z
from car_observed import ObservedEstimator, localize
from observer import HEAD, WRIST, SLOT, FOCAL
from localizer import camera_points
from table_view import TableView

KINDS = ('plate', 'bowl')
COUNT = dict(plate=2, bowl=2)
NEAR = 2.2                          # a look from nearer than this replaces what was known from further away
SAME = dict(plate=.10, bowl=.10)    # a sighting this near to a known thing of its kind is that thing
GONE_AFTER = 3                      # a thing that looks from near do not show where it was known this many times is forgotten
STILL_GOOD = (1.5, 2.0)             # a sighting is carried by dead reckoning while the base has since driven less than this (m) and turned less (rad)


BINS_FACE = -.058                   # the inner edge of the fridge door's bins across the door from its hinge's line (m; +
                                    # outward; the collision meshes: frame -0.009 to 0.062, panel 0.036 to 0.062, bins -0.058 to 0.035)


class PlatesEstimator(ObservedEstimator):
    def __init__(self, observer, scene, folder):
        super().__init__(observer, scene)
        self.view = TableView(scene); self.items = {}; self.next_id = 0; self.looks = 0
        self.taken_ids = set()      # in a hand, in the fridge or in the sink: not looked for any more
        self.switched = None

    @property
    def correct_bent(self):
        return bool(getattr(self, 'in_hand', None))

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
        seen = [s for s in self.view.see(self.ob, base, image=self.ob.data[HEAD + 'rgb'][..., :3]) if s['kind'] in KINDS]
        matched = set()
        near = lambda new: min([float(np.hypot(*(i['xy'] - new['xy']))) for i in self.items.values() if i['kind'] == new['kind']] or [9.])
        for new in sorted(seen, key=near):          # (sightings of known things first)
            kind = new['kind']; best = None
            for item in self.items.values():
                if item['kind'] != kind or item['id'] in matched:
                    continue
                d = float(np.hypot(*(item['xy'] - new['xy'])))
                if d < SAME[kind] and (best is None or d < best[1]):
                    best = (item, d)
            if best is None:
                if len(self.known(kind)) + sum(1 for i in self.taken_ids if i[0] == kind) >= COUNT[kind]:
                    continue
                # (one bowl seen as two groups in one look made two bowls at one place; the second was taken for
                # another bowl beside the first one's rim, and no hand went for it)
                if any(i['kind'] == kind and float(np.hypot(*(i['xy'] - new['xy']))) < SAME[kind] for i in self.items.values()):
                    continue
                item = dict(id=(kind, self.next_id), kind=kind, xy=np.asarray(new['xy'], float), range=new['range'], step=self.ob.steps, n=new['n'], misses=0, pizza=new.get('pizza'))
                self.next_id += 1; self.items[item['id']] = item; matched.add(item['id'])
                self.news.append(dict(seen=kind, xy=np.round(item['xy'], 3).tolist(), range=round(new['range'], 2)))
                continue
            item = best[0]; matched.add(item['id']); item['misses'] = 0
            if new['range'] < max(item['range'], NEAR) and not new['cut']:
                item.update(xy=np.asarray(new['xy'], float), range=new['range'], step=self.ob.steps, n=new['n'], pizza=new.get('pizza', item.get('pizza')))
            if not new['cut'] and new['range'] <= item.get('odom_range', 9.) + .3 or not self.fresh(item):
                c, s_ = np.cos(base[2]), np.sin(base[2]); d = np.asarray(new['xy'], float) - np.asarray(base[:2], float)
                local = np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]])
                item.update(odom_xy=(pose2(*self.odom) @ np.r_[local, 0., 1.])[:2], odom_range=new['range'], odom_at=(self.driven, self.turned))
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
    def in_view(xy, base, nearest=.7, farthest=3.0):
        """Does the head camera show the table's top at map @xy from base pose @base well?"""
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float) - [x, y]; ahead = c * d[0] + s * d[1]; aside = -s * d[0] + c * d[1]
        return bool(nearest < ahead < farthest and abs(np.arctan2(aside, ahead - .24)) < np.radians(36.))

    def wrist_front(self, arm, tip, approach, closing, depth, width=(.015, .09)):
        """Where the fingers' ends should go on what stands in front of @arm's hand (the fridge door's handle bar, or
        the door's free edge), from the wrist camera's look: @depth behind its front face, in its middle across the
        fingers' closing direction. @tip: the fingers' ends now; @approach, @closing: unit vectors (all base frame).
        Returns base-frame xy, or None. (The map pose is a few centimetres off in front of the fridge, and the bar is
        3 cm wide: the hand closed in front of it once.)"""
        points = camera_points(self.ob.data[WRIST[arm] + 'depth_linear'], self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]], FOCAL[arm], stride=2, max_range=1.)
        approach = np.asarray(approach, float); closing = np.asarray(closing, float); d = points - tip
        a = d @ approach; b = d @ closing; c = d[:, 2]
        near = points[(np.abs(c) < .07) & (np.abs(b) < .08) & (a > .0) & (a < .25)]
        self.wrist_debug = dict(points=len(points), near=len(near))
        if len(near) < 30:
            return None
        a = (near - tip) @ approach; b = (near - tip) @ closing
        front = float(np.percentile(a, 5)); bar = a < front + .02
        # (the east door's handle stands 5 cm from the west door's, its front as far out: the two are told apart by
        # the gap between them across the fingers, and the one nearer the planned grip is taken)
        across = np.sort(b[bar]); breaks = np.flatnonzero(np.diff(across) > .012)
        groups_ = np.split(across, breaks + 1) if len(across) else []
        groups_ = [g for g in groups_ if len(g) >= 15]
        self.wrist_debug.update(front=round(front, 3), bar=int(bar.sum()), groups=[[round(float(g[0]), 3), round(float(g[-1]), 3), len(g)] for g in groups_][:4])
        if not groups_:
            return None
        g = min(groups_, key=lambda g_: abs(.5 * (np.percentile(g_, 5) + np.percentile(g_, 95))))
        lo, hi = np.percentile(g, 5), np.percentile(g, 95)
        self.wrist_debug.update(across=round(float(hi - lo), 3))
        if not width[0] < hi - lo < width[1]:
            return None
        return (tip + (front + depth) * approach + .5 * (lo + hi) * closing)[:2]

    def plate_look(self, near_xy, height):
        """The middle of a plate held up at @height near map @near_xy (within 0.15 m), as the head camera shows it:
        map xy, or None. The plate's white rim round its pizza is a ring 0.15-0.20 m from its middle."""
        from table_view import TableView
        base = self.base(); local = self.ob.points('head', 1)
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
        z = local[:, 2] + BASE_Z; keep = np.abs(z - height) < .05
        local = local[keep]; z = z[keep]
        if len(local) < 50:
            self.plate_debug = dict(points=int(len(local))); return None
        xy = np.c_[x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]]
        keep = np.linalg.norm(xy - np.asarray(near_xy, float), axis=1) < .36
        xy, local = xy[keep], local[keep]
        image = self.ob.data[HEAD + 'rgb'][..., :3]
        rgb = TableView.colours(self.ob, local, image) if len(local) else np.zeros((0, 3))
        if len(rgb) < 40:
            self.plate_debug = dict(points=int(len(rgb))); return None
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        white = (r > 150) & (g > 150) & (b > 140) & (np.abs(r - b) < 45)
        rim = xy[white]
        self.plate_debug = dict(points=int(len(xy)), white=int(white.sum()))
        if len(rim) < 30:
            return None
        best = None; near_xy = np.asarray(near_xy, float)
        for dx in np.arange(-.10, .101, .01):
            for dy in np.arange(-.10, .101, .01):
                cxy = near_xy + [dx, dy]; d = np.linalg.norm(rim - cxy, axis=1)
                score = float(np.sum((d > .14) & (d < .208)) - 2. * np.sum(d > .22))
                if best is None or score > best[0]:
                    best = (score, cxy)
        self.plate_debug.update(score=round(best[0], 1), shift=np.round(best[1] - near_xy, 3).tolist())
        return best[1] if best[0] > 15 else None

    def door_look(self, hinge, length=.58):
        """The fridge door's angle (rad) as the head camera shows it, or None: the points that stand in front of the
        fridge within the door's reach from its hinge, 0.3-1.8 m up, are the leaf; the line through them runs along
        it. (Led round by the hand the door stood 3-10 degrees off where the hand's arc put it, and the hand closed
        beside its free edge.)"""
        base = self.base(); local = self.ob.points('head', 2); local = self.ob.clear_of_robot(local)
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
        xy = np.c_[x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]]; z = local[:, 2] + BASE_Z
        d = xy - np.asarray(hinge, float); r = np.linalg.norm(d, axis=1); phi = np.arctan2(d[:, 1], d[:, 0])
        sel = (r > .15) & (r < length - .02) & (z > .3) & (z < 1.8) & (d[:, 1] > .03) & (phi > np.radians(15)) & (phi < np.radians(150))
        self.door_debug = dict(points=int(sel.sum()))
        if sel.sum() < 60:
            return None
        pts = xy[sel]; centre = pts.mean(axis=0); axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]
        angle = float(np.arctan2(axis[1], axis[0]))
        if angle < 0:
            angle += np.pi
        spread = float(np.percentile(np.abs((pts - centre) @ np.array([-axis[1], axis[0]])), 90))
        self.door_debug.update(angle=round(float(np.degrees(angle)), 1), spread=round(spread, 3))
        if spread > .10:
            return None
        # Where its free end shows along it, against where the map pose puts it (the pose stood 4-6 cm off by the
        # fridge and the hand closed beside the free edge, 314, 319): the door as seen is that far along its line.
        u = np.array([np.cos(angle), np.sin(angle)]); v = np.array([-u[1], u[0]])
        wide = (r > .15) & (r < length + .10) & (z > .3) & (z < 1.8) & (phi > np.radians(15)) & (phi < np.radians(150))
        a = d[wide] @ u; b = d[wide] @ v; line = np.abs(b - float(np.median(d[sel] @ v))) < .10
        if line.sum() >= 40:
            far = float(np.percentile(a[line], 99))
            # (and across it: the bins' inner edges, what stands furthest in on the side the camera sees, are 5.8 cm in
            # from the hinge's line; the panel between them is recessed. At 96 degrees the hand closed beside the
            # edge, 305.)
            face = float(np.percentile(d[sel] @ v, 5))
            self.door_seen = dict(step=self.ob.steps, angle=angle, along=float(np.clip(far - length, -.08, .08)), across=float(np.clip(face - BINS_FACE, -.06, .06)))
            self.door_debug.update(far=round(far, 3), face=round(face, 3))
        return angle

    def taken(self, item_id):
        """The thing is in a hand, in the fridge or in the sink: it is not looked for any more."""
        self.items.pop(item_id, None); self.taken_ids.add(item_id)

    def put_back(self, item_id, xy):
        """A thing a hand had is somewhere again (it fell): looked for again near @xy."""
        self.taken_ids.discard(item_id)
        kind = item_id[0]; self.items[item_id] = dict(id=item_id, kind=kind, xy=np.asarray(xy, float), range=9., step=self.ob.steps, n=0, misses=0)


def observed_pipeline(rt):
    from plates_pipeline import full_task
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    return (yield from full_task(rt.ctx, rt.est))
