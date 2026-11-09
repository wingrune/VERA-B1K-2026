"""Cleaning the trumpet on the desk from what the robot's own cameras show: the estimator and the entry point.

The base pose and the per-step runtime are those of the earlier controllers (car_observed.py), with a map of the
bedroom. What is new (trumpet_view.py): the trumpet's pose on the desk, fitted to its yellow points with its model; the
red scrub brush lying on the desk; the dark specks of dust on the trumpet. A speck is kept in the trumpet's own frame
(it moves with the trumpet when the brush nudges it) and dropped when looks show its place on the trumpet and no speck
there GONE times in a row.
"""
from pathlib import Path
import numpy as np
from car_observed import ObservedEstimator, localize
from observer import HEAD, FOCAL, WRIST, SLOT
from dirt_view import pixel_points
from localizer import BASE_Z
from trumpet_view import TrumpetView, DESK, specks, pose_matrix, camera_rotation, yellow, red

SAME = .02                          # a speck this near (trumpet frame) to a known one is that one
GONE = 2                            # looks in a row that show a speck's place on the trumpet clean
HAND_CUT = .16                      # what the camera sees this near the hand that holds the brush is left out


class TrumpetEstimator(ObservedEstimator):
    def __init__(self, observer, scene, folder):
        super().__init__(observer, scene)
        # (the bedroom's west wall by the desk is a window from the floor up: what is seen through it is left out of the
        # fit)
        observer.loc.unlike = list(observer.loc.unlike) + [(-12., -1.5, -6.42, 2.5)]
        self.view = TrumpetView(str(Path(folder) / 'priors' / 'trumpet_model.npz'))
        self.fields = dict(np.load(Path(folder) / 'priors' / 'trumpet_fields.npz'))
        # (the trumpet's upper surface every 1 cm, its own frame, and how many looks in a row showed each point clean)
        vt = self.fields['visual_top']; lo_f = self.fields['lo']; res_f = float(self.fields['res'])
        ii, jj = np.nonzero(np.isfinite(vt)); keep = (ii % 2 == 0) & (jj % 2 == 0)
        self.surface = np.c_[lo_f[0] + (ii[keep] + .5) * res_f, lo_f[1] + (jj[keep] + .5) * res_f, vt[ii[keep], jj[keep]] - .004]
        # (the same points where the dust lies on them: on the collision surface, 3 mm under its top over 595 particles
        # of the dev starts' median, 1-1.5 cm over the visual one; the dabs' boxes aimed 1 cm too low)
        ct = self.fields['collision_top'][ii[keep], jj[keep]]
        self.surface_dab = self.surface.copy(); self.surface_dab[np.isfinite(ct), 2] = ct[np.isfinite(ct)] - .003
        self.seen_clean = np.zeros(len(self.surface), int)
        # (where on it the dust lies, over the 50 dev starts (30 training ones not held out and the 20 public): the
        # densest 20 % of the surface holds 71 % of the particles, 40 % holds 93 %; priors/dust_prior.npz, same points)
        prior = Path(folder) / 'priors' / 'dust_prior.npz'
        self.dust_density = np.load(prior)['density'].astype(float) if prior.exists() else np.ones(len(self.surface))
        # (the trumpet's frame stands this high: its model's lowest point on the desk's top)
        self.z0 = DESK['top'] - float(np.nanmin(self.view.model['bottom']))
        self.trumpet = None; self.brush = None; self.specks = []; self.next_id = 0; self.looks = 0
        self.holding_brush = False; self.hand = None      # map xyz of the hand that holds the brush (cut out of the views)

    def see(self, what=()):
        return {}

    def T(self):
        return None if self.trumpet is None else pose_matrix(self.trumpet)

    def frames(self):
        base = self.base(); depth = self.ob.data[HEAD + 'depth_linear']; rgb = self.ob.data[HEAD + 'rgb']
        cam = np.asarray(self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[2], float)
        return base, depth, rgb, cam

    def look(self, dust=True):
        """Take what the head camera shows into what is known: the trumpet's pose, the brush on the desk, the specks."""
        base, depth, rgb, cam = self.frames(); self.looks += 1
        P, C, rng = self.view.points(depth, rgb, cam, FOCAL['head'], base)
        if self.hand is not None and len(P):
            keep = np.linalg.norm(P - self.hand, axis=1) > HAND_CUT
            P, C, rng = P[keep], C[keep], rng[keep]
        pose = None
        if len(P) and self.trumpet is not None:
            # (from the last pose; kept if it stays near it: the dabs push the trumpet by centimetres)
            pose = self.view.trumpet(P, C, prior=self.trumpet)
            if pose is not None and (pose['res'] > .017 or float(np.hypot(pose['x'] - self.trumpet['x'], pose['y'] - self.trumpet['y'])) > .12
                                     or abs((pose['yaw'] - self.trumpet['yaw'] + np.pi) % (2 * np.pi) - np.pi) > .3):
                pose = None
        if pose is None and len(P):
            pose = self.view.trumpet(P, C)
            if pose is not None and self.trumpet is not None and pose['res'] > .012:
                pose = None             # (a fresh search, against the pose kept: only a sure fit, 1.2 cm, replaces it)
        if pose is not None and not (pose['res'] < .017 and pose['n'] > 300):
            self.news.append(dict(trumpet_fit='rejected', res=round(pose['res'], 4), n=pose['n']))
        # (close up, from the stances by the desk, a right fit leaves 1.0-1.6 cm, over 112 views of 14 dev starts: under
        # 1.2 cm only the first fit from the room was taken, and the trumpet, pushed 5-9 cm by the dabs, was believed
        # where it had been for 7000 steps, 19)
        if pose is not None and pose['res'] < .017 and pose['n'] > 300 and not (self.trumpet is None and pose['res'] > .012):
            ylw = yellow(C); near = float(np.median(rng[ylw])) if ylw.any() else 9.
            old = self.trumpet
            self.trumpet = dict(pose, z=self.z0, step=self.ob.steps, range=near)
            if old is None or float(np.hypot(old['x'] - pose['x'], old['y'] - pose['y'])) > .005 or abs(old['yaw'] - pose['yaw']) > .01:
                self.news.append(dict(seen='trumpet', pose=[round(pose['x'], 3), round(pose['y'], 3), round(pose['yaw'], 3)], res=round(pose['res'], 4), n=pose['n'], range=round(near, 2)))
        if not self.holding_brush and len(P):
            brush = self.view.brush(P, C, rng)
            if brush is not None and (self.brush is None or brush['range'] < self.brush['range'] + .3 or float(np.linalg.norm(brush['xy'] - self.brush['xy'])) > .1):
                if self.brush is None:
                    self.news.append(dict(seen='brush', xy=np.round(brush['xy'], 3).tolist(), length=round(brush['length'], 3), range=round(brush['range'], 2)))
                self.brush = dict(brush, step=self.ob.steps)
        found = []
        if dust and self.trumpet is not None:
            T = self.T()
            found = specks(self.view, depth, rgb, cam, FOCAL['head'], base, T)
            # (the dust lies on the trumpet's upper surface: 95 % of the particles of the 30 dev starts within 2 cm of the
            # top of its visual mesh; dark gaps between its tubes, deeper down, are no dust: 27 'specks' for 9
            # particles, 124; over 112 close views of 14 dev starts this kept 215 of the 221 particles found and left
            # out 3 in 10 of the false ones)
            found = [(p, n) for p, n in found if self.on_top(p, .025)]
            if self.hand is not None:
                found = [(p, n) for p, n in found if float(np.linalg.norm((T @ np.r_[p, 1.])[:3] - self.hand)) > HAND_CUT]
            self.fuse(found, base, depth, cam, T)
            self.mark_clean(found, base, depth, cam, T)
        return pose, found

    def mark_clean(self, found, base, depth, cam, T):
        """The surface points this look showed (in the picture, nothing in front) with no speck within 2.5 cm."""
        W = np.c_[self.surface, np.ones(len(self.surface))] @ T.T
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = W[:, :2] - [x, y]
        local = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], W[:, 2] - .012]
        q = (local - cam[:3]) @ camera_rotation(cam)
        h, w = depth.shape; f = FOCAL['head'] * w; z = np.maximum(q[:, 2], 1e-3)
        u = (q[:, 0] / z * f + w / 2).astype(int); v = (q[:, 1] / z * f + h / 2).astype(int)
        ok = (q[:, 2] > .2) & (u >= 5) & (u < w - 5) & (v >= 5) & (v < h - 5)
        vis = np.zeros(len(W), bool); vis[ok] = np.abs(depth[v[ok], u[ok]] - q[ok, 2]) < .015
        if self.hand is not None:
            vis &= np.linalg.norm(W[:, :3] - self.hand, axis=1) > HAND_CUT
        near = np.zeros(len(W), bool)
        if found:
            from scipy.spatial import cKDTree
            near = cKDTree(np.array([p for p, _ in found])).query(self.surface)[0] < .025
        self.seen_clean[vis & ~near] += 1; self.seen_clean[near] = 0

    def fuse(self, found, base, depth, cam, T):
        seen = set()
        for p, n in found:
            near = [s for s in self.specks if float(np.linalg.norm(s['p'] - p)) < SAME]
            if near:
                s = min(near, key=lambda s: float(np.linalg.norm(s['p'] - p)))
                s['p'] = (s['p'] * min(s['seen'], 4) + p) / (min(s['seen'], 4) + 1); s['seen'] += 1; s['missed'] = 0
            else:
                s = dict(id=self.next_id, p=np.asarray(p, float), seen=1, missed=0, dabs=0, step=self.ob.steps)
                self.next_id += 1; self.specks.append(s)
            seen.add(s['id'])
        for s in list(self.specks):
            if s['id'] in seen or not self.shows(s['p'], base, depth, cam, T):
                continue
            s['missed'] += 1
            if s['missed'] >= GONE:
                self.specks.remove(s); self.news.append(dict(gone='speck', id=s['id'], dabs=s['dabs']))

    def on_top(self, p, tolerance=.015):
        """Is the trumpet's point @p (its own frame) within @tolerance of the top of its visual mesh near by?"""
        f = self.fields; i = int((p[0] - f['lo'][0]) / float(f['res'])); j = int((p[1] - f['lo'][1]) / float(f['res']))
        patch = f['visual_top'][max(0, i - 1):i + 2, max(0, j - 1):j + 2]
        patch = patch[np.isfinite(patch)]
        return bool(len(patch)) and bool(np.any(np.abs(patch - float(p[2])) < tolerance))

    def shows(self, p, base, depth, cam, T):
        """Is the trumpet's point @p (its own frame) in the head camera's view, nothing in front of it?"""
        w_ = T @ np.r_[p, 1.]
        if self.hand is not None and float(np.linalg.norm(w_[:3] - self.hand)) < HAND_CUT:
            return False
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = w_[:2] - [x, y]
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], w_[2] - .012])
        q = camera_rotation(cam).T @ (local - cam[:3])
        if q[2] < .2:
            return False
        h, w = depth.shape; f = FOCAL['head'] * w
        u = q[0] / q[2] * f + w / 2; v = q[1] / q[2] * f + h / 2
        if not (5 <= u < w - 5 and 5 <= v < h - 5):
            return False
        return abs(float(depth[int(v), int(u)]) - q[2]) < .02

    def brush_in_hand(self, tool):
        """The brush the hand holds tilted, as the head camera shows it: (its lower end's point, unit axis down towards
        that end, length seen, points), map frame, or None. Its red points within 20 cm of the hand's @tool point."""
        base, depth, rgb, cam = self.frames()
        pts, v, u = pixel_points(depth, cam, FOCAL['head'], stride=1, max_range=2.)
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
        W = np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1], pts[:, 2] + BASE_Z]
        sel = red(rgb[v, u, :3].astype(np.float32)) & (np.linalg.norm(W - tool, axis=1) < .20)
        if sel.sum() < 40:
            return None
        Q = W[sel]; centre = Q.mean(axis=0); axis = np.linalg.svd(Q - centre, full_matrices=False)[2][0]
        if axis[2] > 0:
            axis = -axis                # (held tilted, the far end is the lower one)
        a = (Q - centre) @ axis; length = float(np.percentile(a, 99) - np.percentile(a, 1))
        # (the far end: the points within 1 cm of the far extreme, their middle across the brush)
        far = Q[a > np.percentile(a, 99) - .01]
        return far.mean(axis=0), axis, length, int(sel.sum())

    def brush_end(self, tool):
        """The held brush's lowest part as the head camera shows it: the middle of its red points within 1.5 cm of
        the lowest red point within 20 cm of the hand's @tool point (map), and how many; or None."""
        base, depth, rgb, cam = self.frames()
        pts, v, u = pixel_points(depth, cam, FOCAL['head'], stride=1, max_range=2.)
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
        W = np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1], pts[:, 2] + BASE_Z]
        sel = red(rgb[v, u, :3].astype(np.float32)) & (np.linalg.norm(W - tool, axis=1) < .20)
        if sel.sum() < 30:
            return None
        Q = W[sel]; low = Q[Q[:, 2] < Q[:, 2].min() + .015]
        if len(low) < 8:
            return None
        return low.mean(axis=0), int(len(low))

    def held_brush(self, arm, tool):
        """The brush in @arm's hand as that hand's wrist camera shows it, in the TOOL frame (@tool: the tool's 4x4 in
        the base frame, as the arm's joints have it): (middle of its red points, unit long axis, extent along it,
        points), or None. The camera rides on the hand: what it shows of the brush needs no pose of the base."""
        depth = self.ob.data[WRIST[arm] + 'depth_linear']; rgb = self.ob.data[WRIST[arm] + 'rgb']
        cam = self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]]
        # (the brush is 5-15 cm from this camera: nearer than pixel_points keeps)
        x, y, z, w = cam[3:]
        rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])
        h, wd = depth.shape; f = FOCAL[arm] * wd
        v, u = np.mgrid[1:h:2, 1:wd:2]; d = depth[v, u].astype(np.float32)
        ok = np.isfinite(d) & (d > .02) & (d < .5)
        pc = np.stack([(u + .5 - wd / 2) / f * d, (v + .5 - h / 2) / f * d, d], -1)[ok]
        pts = pc @ rot.T + cam[:3]; v, u = v[ok], u[ok]
        if not len(pts):
            return None
        local = (pts - tool[:3, 3]) @ tool[:3, :3]
        sel = red(rgb[v, u, :3].astype(np.float32)) & (np.linalg.norm(local, axis=1) < .2)
        self.held_debug = dict(red=int(sel.sum()), points=len(pts))
        if sel.sum() < 40:
            return None
        Q = local[sel]; centre = Q.mean(axis=0); axis = np.linalg.svd(Q - centre, full_matrices=False)[2][0]
        a = (Q - centre) @ axis
        return centre, axis, float(np.percentile(a, 99) - np.percentile(a, 1)), int(sel.sum())

    def wrist_brush(self, arm, tip):
        """The brush on the desk as the wrist camera of @arm shows it from some 10 cm over it: (middle xy, unit long
        axis xy, length), base frame, or None. Its red body within 20 cm of @tip, on the desk's top."""
        depth = self.ob.data[WRIST[arm] + 'depth_linear']; rgb = self.ob.data[WRIST[arm] + 'rgb']
        cam = self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]]
        pts, v, u = pixel_points(depth, cam, FOCAL[arm], stride=2, max_range=.8)
        z = pts[:, 2] + BASE_Z; colour = rgb[v, u, :3].astype(np.float32)
        sel = red(colour) & (z > DESK['top'] - .01) & (z < DESK['top'] + .05) & (np.linalg.norm(pts[:, :2] - np.asarray(tip, float)[:2], axis=1) < .20)
        self.wrist_debug = dict(red=int(sel.sum()))
        if sel.sum() < 30:
            return None
        q = pts[sel, :2]; centre = q.mean(axis=0); axis = np.linalg.svd(q - centre, full_matrices=False)[2][0]
        a = (q - centre) @ axis; length = float(np.percentile(a, 98) - np.percentile(a, 2))
        centre = centre + axis * float(np.percentile(a, 98) + np.percentile(a, 2)) / 2      # (the middle of its extent)
        self.wrist_debug.update(length=round(length, 3))
        if not .08 < length < .16:
            return None                 # (part of it out of view, or not the brush)
        return centre, axis, length

    def wrist_search(self, arm):
        """The brush on the desk's top anywhere in the wrist camera of @arm (the hand held up over the desk, looking
        down beyond the trumpet): dict as the head camera's look gives (map frame), or None."""
        depth = self.ob.data[WRIST[arm] + 'depth_linear']; rgb = self.ob.data[WRIST[arm] + 'rgb']
        cam = self.ob.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]]
        pts, v, u = pixel_points(depth, cam, FOCAL[arm], stride=2, max_range=1.5)
        z = pts[:, 2] + BASE_Z; colour = rgb[v, u, :3].astype(np.float32)
        sel = red(colour) & (z > DESK['top'] - .01) & (z < DESK['top'] + .05)
        self.wrist_debug = dict(red=int(sel.sum()), points=int(len(pts)))
        if sel.sum() < 30:
            return None
        q = pts[sel, :2]; centre = q.mean(axis=0); axis = np.linalg.svd(q - centre, full_matrices=False)[2][0]
        a = (q - centre) @ axis; length = float(np.percentile(a, 98) - np.percentile(a, 2))
        centre = centre + axis * float(np.percentile(a, 98) + np.percentile(a, 2)) / 2
        self.wrist_debug.update(length=round(length, 3))
        if not .06 < length < .16:
            return None
        x, y, yaw = self.base(); c, s_ = np.cos(yaw), np.sin(yaw)
        xy = np.array([x + c * centre[0] - s_ * centre[1], y + s_ * centre[0] + c * centre[1]])
        direction = np.array([c * axis[0] - s_ * axis[1], s_ * axis[0] + c * axis[1]])
        return dict(xy=xy, direction=direction, length=length, pixels=int(sel.sum()), range=float(np.linalg.norm(centre)))

    def speck_points(self, most_dabs=2):
        """The specks still to be taken, in the map frame: [(id, xyz)]."""
        T = self.T()
        return [] if T is None else [(s['id'], (T @ np.r_[s['p'], 1.])[:3]) for s in self.specks if s['dabs'] < most_dabs]


def observed_pipeline(rt):
    from trumpet_pipeline import full_task
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    return (yield from full_task(rt.ctx, rt.est))
