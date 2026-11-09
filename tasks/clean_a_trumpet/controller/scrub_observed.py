"""Scrubbing the bathroom floor from what the robot's own cameras show: the estimator and the entry point.

The base pose and the per-step runtime are those of the earlier controllers (car_observed.py), with the map of the
bathroom. What is new is what is known of the scrub brush (where it lies, which way) and of the dirt on the floor
(scrub_view.py): each mark seen is kept where it was seen; a mark is dropped when the floor where it lay is in view
from near by and shows nothing there twice running (scrubbed away, or never dirt).
"""
from pathlib import Path
import numpy as np
from car_observed import ObservedEstimator, localize
from observer import HEAD, FOCAL, WRIST, SLOT
from localizer import BASE_Z
from dirt_view import pixel_points
from scrub_view import red
from scrub_view import ScrubView

NEAR = 2.5                          # a brush seen from nearer than this replaces what was known from further away
SAME = .08                          # a mark this near to a known one is that one
GONE = 3                            # looks in a row that show the floor clean where a mark was (from within IN_VIEW)
COVER = dict(step=.05, reach=2.5)   # the floor's points kept as seen or not (a mark shows from 2.7 m: 150 pixels)


def camera_rotation(cam):
    qx, qy, qz, qw = cam[3:]
    return np.array([[1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)], [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                     [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]]) @ np.diag([1., -1., -1.])


def project(xy, base, cam, shape):
    """Where floor points @xy (N, 2, map) fall in the head camera's image for the base at @base: (u, v, depth along
    the camera's axis, in the image, distance from the base)."""
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float) - [x, y]
    local = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], np.zeros(len(d))]
    p = (local - cam[:3]) @ camera_rotation(cam)
    h, w = shape; f = FOCAL['head'] * w; z = np.maximum(p[:, 2], 1e-3)
    u = p[:, 0] / z * f + w / 2; v = p[:, 1] / z * f + h / 2
    ok = (p[:, 2] > .2) & (u >= 10) & (u < w - 10) & (v >= 10) & (v < h - 10)
    return u, v, p[:, 2], ok, np.hypot(local[:, 0], local[:, 1])


class ScrubEstimator(ObservedEstimator):
    def __init__(self, observer, scene, folder):
        super().__init__(observer, scene)
        # (in the bathroom a true fit scores 0.97-0.99; facing the window over the bath, its glass not seen and the
        # garden beyond out of range, fits of 0.65-0.75 were taken and walked the pose 0.19 m and 7 degrees off along
        # the strip by the bath, 317: those are left out, dead reckoning carries the pose there)
        observer.loc.accept = .80
        # (the mirrors over the basins and over the toilet show the room again behind the north wall: depth points up
        # to 2 m past it, and facing north by the counter the fit pulled the pose 0.37 m off, 317; the window's glass
        # shows the garden south of the room) What is seen past either is left out of the fit.
        observer.loc.unlike = list(observer.loc.unlike) + [(18.5, 12.02, 26.5, 15.0), (18.5, 6.0, 26.5, 9.60)]
        # (where the base can stand, for the fit: by what is at the chassis' height, the chassis' half width off it;
        # the disc of the route planner, 0.30 m off the counter's top too, put the strip by the bath and the shower out
        # of bounds for every fit there, and the pose went by dead reckoning 0.26 m off, 306)
        observer.loc.clear = scene.chassis_distance; observer.loc.stand = .22
        floor = scene.meta['start_floor']
        self.view = ScrubView(floor['lo'], floor['hi'], folder)
        self.brush = None; self.dirt = []; self.looks = 0; self.switched = None; self.holding_brush = False
        self.next_id = 0
        # Where dirt is put at the start: along a few narrow bands of the floor (the nook by the toilet, the strip
        # between the bath and the counter, by the shower, ...), the same in every start (the 1117 particles of the 280
        # starts not held out, priors/dirt_band.npz, 10 cm round them). A dark mark elsewhere is no dirt (seams behind
        # the bath, shadows), and only the floor in the bands has to be looked at.
        prior = Path(folder) / 'priors' / 'dirt_band.npz'
        self.band = dict(np.load(prior)) if prior.exists() else None
        # the floor's points (where nothing stands on it) and which of them a look has shown from near enough
        lo, hi = np.asarray(floor['lo'], float), np.asarray(floor['hi'], float)
        gx, gy = np.meshgrid(np.arange(lo[0] + .03, hi[0] - .02, COVER['step']), np.arange(lo[1] + .03, hi[1] - .02, COVER['step']), indexing='ij')
        pts = np.c_[gx.ravel(), gy.ravel()]
        self.cover_xy = pts[[self.on_floor(q) and self.in_band(q) for q in pts]]; self.covered = np.zeros(len(self.cover_xy), bool)
        self.cam = None; self.shape = None

    def see(self, what=()):
        return {}

    def predict(self, s, last_action, dt=1 / 30):
        super().predict(s, last_action, dt)
        switched = self.ob.loc.debug.get('switched_to') if getattr(self.ob.loc, 'debug', None) else None
        if switched is not None and switched != self.switched:
            self.news.append(dict(start_pose_switched_to=switched))
        self.switched = switched
        if self.ob.steps % 30 == 0 and getattr(self.ob.loc, 'debug', None):
            self.news.append(dict(loc=dict(self.ob.loc.debug, pose=np.round(self.ob.loc.pose, 3).tolist(), velocity=np.round(self.velocity, 3).tolist(), action=np.round(last_action[:3], 3).tolist())))
        if self.ob.loc.pose is not None and self.ob.steps % 6 == 0 and abs(float(last_action[2])) < .3:
            self.look()

    def look(self):
        """Take what the head camera shows into what is known. Returns (marks, brush) this look showed."""
        base = self.base(); self.looks += 1
        depth = self.ob.data[HEAD + 'depth_linear']; rgb = self.ob.data[HEAD + 'rgb']
        cam = self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[2]
        marks, brush = self.view.look(depth, rgb, cam, FOCAL['head'], base)
        self.cam = np.asarray(cam, float).copy(); self.shape = depth.shape
        if self.cam[2] > 1.25:
            self.cam_tall = self.cam        # (the camera with the trunk upright: what a survey's look will have)
        u, v, z, ok, rng = project(self.cover_xy, base, cam, depth.shape)
        k = np.flatnonzero(ok & (rng < COVER['reach']) & ~self.covered)
        if len(k):
            shown = np.abs(depth[v[k].astype(int), u[k].astype(int)] - z[k]) < .06
            self.covered[k[shown]] = True
        # (a mark where the map has something standing on the floor is not on the floor: dark spots on the bath, 319)
        marks = [m for m in marks if self.on_floor(m['xy']) and self.in_band(m['xy'])]
        # (nor one seen from nearer than 0.5 m: with the trunk bent over the floor the robot's own shadow is dark,
        # 3 of the 3 marks in the bands that were not dirt over 76 recorded views, and the 3 scrubbed for nothing in 319)
        marks = [m for m in marks if m['range'] > .5]
        if brush is not None and not self.on_floor(brush['xy']):
            brush = None                # (a red speck on the cabinet's foot by the nook was taken for the brush, 306)
        if brush is not None and not self.holding_brush:
            old = self.brush
            # (a red thing far from where the brush was is taken for it only when seen from near: a red speck 3 m off
            # by the cabinet took the brush's place while the hand was over it, 306)
            jump = old is not None and float(np.linalg.norm(brush['xy'] - old['xy'])) > .5
            if old is None or (jump and brush['range'] < 1.5) or (not jump and (brush['range'] < max(old['range'], NEAR) or float(np.linalg.norm(brush['xy'] - old['xy'])) > .15)):
                if old is None:
                    self.news.append(dict(seen='brush', xy=np.round(brush['xy'], 3).tolist(), range=round(brush['range'], 2)))
                self.brush = dict(brush, step=self.ob.steps)
        seen = set(); hand = getattr(self, 'hand_xy', None) if self.holding_brush else None
        if hand is not None:
            # (the brush in the hand, its dark bristles on the floor, is not dirt; nor does it show the floor there)
            marks = [m for m in marks if float(np.linalg.norm(m['xy'] - hand)) > .15]
        for m in marks:
            near = [d for d in self.dirt if float(np.linalg.norm(d['xy'] - m['xy'])) < SAME]
            if near:
                d = min(near, key=lambda d: float(np.linalg.norm(d['xy'] - m['xy'])))
                if m['range'] < d['range'] + .3:
                    d.update(xy=np.asarray(m['xy'], float), range=m['range'])
                d['seen'] += 1; d['clean'] = 0; seen.add(d['id'])
                if 'scrubbed_at' in d:
                    d['seen_since'] = d.get('seen_since', 0) + 1
            else:
                d = dict(id=self.next_id, xy=np.asarray(m['xy'], float), range=m['range'], seen=1, clean=0, step=self.ob.steps)
                self.next_id += 1; self.dirt.append(d); seen.add(d['id'])
                self.news.append(dict(seen='dirt', xy=np.round(d['xy'], 3).tolist(), range=round(m['range'], 2)))
        # a mark whose place is in view near by and shows clean floor
        for d in list(self.dirt):
            if d['id'] in seen or not self.in_view(d['xy'], base, depth, cam) or (hand is not None and float(np.linalg.norm(d['xy'] - hand)) < .15):
                continue
            d['clean'] += 1
            if d['clean'] >= GONE:
                self.dirt.remove(d); self.news.append(dict(gone='dirt', xy=np.round(d['xy'], 3).tolist()))
        return marks, brush

    def in_view(self, xy, base, depth, cam, reach=1.5):
        """Is the floor at @xy in the head camera's view and near enough to show a mark there?"""
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float) - [x, y]
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], 0.])
        if float(np.linalg.norm(local[:2])) > reach:
            return False
        p = camera_rotation(cam).T @ (local - cam[:3])
        if p[2] < .2:
            return False
        h, w = depth.shape; f = FOCAL['head'] * w
        u = p[0] / p[2] * f + w / 2; v = p[1] / p[2] * f + h / 2
        if not (10 <= u < w - 10 and 10 <= v < h - 10):
            return False
        # (nothing in front of it: the depth there is about as far as the floor point)
        return abs(float(depth[int(v), int(u)]) - p[2]) < .06

    def wrist_brush(self, arm, tip):
        """The brush as the wrist camera of @arm shows it from some 10 cm over it: (middle xy, unit long axis xy, length),
        base frame, or None. Its red body within 20 cm of @tip, under 6 cm over the floor."""
        depth = self.ob.data[WRIST[arm] + 'depth_linear']; rgb = self.ob.data[WRIST[arm] + 'rgb']
        cam = self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]]
        pts, v, u = pixel_points(depth, cam, FOCAL[arm], stride=2, max_range=.8)
        z = pts[:, 2] + BASE_Z; colour = rgb[v, u, :3].astype(np.float32)
        sel = red(colour) & (z < .06) & (np.linalg.norm(pts[:, :2] - np.asarray(tip, float)[:2], axis=1) < .20)
        self.wrist_debug = dict(red=int(sel.sum()))
        if sel.sum() < 30:
            return None
        q = pts[sel, :2]; centre = q.mean(axis=0); axis = np.linalg.svd(q - centre, full_matrices=False)[2][0]
        a = (q - centre) @ axis; length = float(np.percentile(a, 98) - np.percentile(a, 2))
        self.wrist_debug.update(length=round(length, 3))
        if not .05 < length < .22:
            return None                 # (part of it out of view, or not the brush)
        return centre, axis, length

    def in_band(self, xy):
        if self.band is None:
            return True
        i, j = np.floor((np.asarray(xy, float)[:2] - self.band['lo']) / float(self.band['res'])).astype(int); grid = self.band['band']
        return bool(0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and grid[i, j])

    def on_floor(self, xy):
        i, j = self.scene.cell(np.asarray(xy, float)[None]); i, j = int(i[0]), int(j[0])
        return bool(self.scene.inside(np.array([i]), np.array([j]))[0]) and float(self.scene.chassis_distance[i, j]) > .0

    def known_dirt(self):
        return [d for d in self.dirt]


def observed_pipeline(rt):
    from scrub_pipeline import full_task
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    return (yield from full_task(rt.ctx, rt.est))
