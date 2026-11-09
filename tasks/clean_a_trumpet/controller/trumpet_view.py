"""The trumpet, the scrub brush and the dust on the trumpet, from the head camera's colour and depth images.

All three lie on the desk (desk_aduafr_0, its top 0.773 m up; map x -5.99..-4.48, y -0.29..0.34):

- The trumpet is yellow brass, 0.92 m long, 0.25 wide, 0.17 high (its silver mouthpiece is not yellow). Its pose on
  the desk (x, y, heading) is fitted to the yellow points with the model of priors/trumpet_model.npz (its visual mesh
  in its own frame): the heading first from the points' long axis, both ways round, then a 2-D closest-point fit of the
  model's points that face up, from each start; the better fit wins (the bell stands 0.16 m high at one end, the
  mouthpiece's end is low).
- The brush is red, 12.7 x 3.6 x 2.1 cm, lying on the desk: red points within 4 cm over the desk's top.
- Dust: dark specks on the yellow: pixels much darker than the yellow round them, in small groups.

Everything returned is in the map frame M.
"""
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial import cKDTree
from dirt_view import pixel_points

DESK = dict(lo=np.array([-5.99, -.29]), hi=np.array([-4.48, .34]), top=.7734)


def yellow(colour):
    r, g, b = colour[:, 0], colour[:, 1], colour[:, 2]
    return (r > 120) & (g > 90) & (b < .6 * g) & (r > b + 60)


def red(colour):
    # (the brush's red has r/g 4-6; the desk's wood, which passed r > 1.6 g, 1.6-2.2: in the wrist camera's view of the
    # brush on the desk the wood made it 0.2 m long, and the look was thrown away, 19, 302)
    r, g, b = colour[:, 0], colour[:, 1], colour[:, 2]
    return (r > 70) & (r > 2.8 * g) & (r > 2.8 * b)


class TrumpetView:
    def __init__(self, model_path):
        m = np.load(model_path)
        v = m['visual'].astype(float)
        # the model's points that a camera from above sees: the highest in each 1 cm column
        key = np.floor(v[:, :2] / .01).astype(int); order = np.lexsort((-v[:, 2], key[:, 1], key[:, 0]))
        k = key[order]; first = np.r_[True, np.any(k[1:] != k[:-1], axis=1)]
        self.top_points = v[order][first]
        self.model = m
        # (the model's points for the fit: its visual mesh's vertices, thinned to one per 1 cm cell)
        cell = np.floor(v / .01).astype(int); _, first = np.unique(cell, axis=0, return_index=True)
        self.fit_points = v[np.sort(first)]
        self.debug = {}

    def points(self, depth, rgb, cam_rel, focal_ratio, base, stride=2):
        pts, v, u = pixel_points(depth, cam_rel, focal_ratio, stride)
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
        xy = np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]]
        z = pts[:, 2] + .012
        colour = rgb[v, u, :3].astype(np.float32)
        # (over the desk, and up to 0.2 m beyond its edges, lower there: the trumpet's bell can hang over the edge,
        # tilted and below the desk's top, 307)
        over = np.all(xy > DESK['lo'], axis=1) & np.all(xy < DESK['hi'], axis=1)
        near = np.all(xy > DESK['lo'] - .2, axis=1) & np.all(xy < DESK['hi'] + .2, axis=1)
        on = ((over & (z > DESK['top'] + .003)) | (near & ~over & (z > DESK['top'] - .12))) & (z < DESK['top'] + .25)
        return np.c_[xy, z][on], colour[on], np.linalg.norm(pts[on, :2], axis=1)

    def brush(self, P, colour, rng):
        sel = red(colour) & (P[:, 2] < DESK['top'] + .04)
        if sel.sum() < 15:
            return None
        q = P[sel, :2]; centre = q.mean(axis=0); axis = np.linalg.svd(q - centre, full_matrices=False)[2][0]
        a = (q - centre) @ axis; length = float(np.percentile(a, 98) - np.percentile(a, 2))
        if not .05 < length < .2:
            return None
        return dict(xy=centre, direction=axis, length=length, pixels=int(sel.sum()), range=float(np.median(rng[sel])))

    def trumpet(self, P, colour, prior=None):
        """dict(x, y, yaw, res, n) or None: the trumpet's pose on the desk from its yellow points, by a 3-D closest-
        point fit of its model (the height of its frame fixed by the desk) from several starts: the points' long axis
        both ways round, shifted along it by up to 0.15 m (seen from one side the fit slid along the trumpet by
        0.1-0.4 m and turned it end for end when started from the middle only)."""
        sel = yellow(colour)
        if sel.sum() < 200:
            self.debug = dict(yellow=int(sel.sum())); return None
        Y = P[sel]
        if len(Y) > 1500:
            Y = Y[np.random.default_rng(0).choice(len(Y), 1500, replace=False)]
        centre = Y[:, :2].mean(axis=0); axis = np.linalg.svd(Y[:, :2] - centre, full_matrices=False)[2][0]
        model = self.fit_points; z0 = DESK['top'] - float(np.nanmin(self.model['bottom']))
        best = None
        # (from the last pose where there is one: the dabs push the trumpet by centimetres, and from the points' axis
        # the fit came out end for end now and then with 1.4-1.5 cm left, 302)
        starts = [(float(prior['yaw']), np.array([prior['x'], prior['y']], float))] if prior is not None else \
            [(float(np.arctan2(axis[1], axis[0])) + flip, centre + shift * axis) for flip in (0., np.pi) for shift in (-.15, 0., .15)]
        for yaw0, t0 in starts:
            if True:
                yaw = yaw0; t = np.array(t0, float)
                for it in range(30):
                    c, s = np.cos(yaw), np.sin(yaw)
                    M = np.c_[model[:, :2] @ np.array([[c, -s], [s, c]]).T + t, model[:, 2] + z0]
                    d, j = cKDTree(M).query(Y); keep = d < max(.03, float(np.percentile(d, 80)))
                    A = Y[keep, :2]; B = M[j[keep], :2]
                    ca, cb = A.mean(axis=0), B.mean(axis=0); H = (B - cb).T @ (A - ca)
                    U, _, Vt = np.linalg.svd(H); Rd = Vt.T @ U.T
                    if np.linalg.det(Rd) < 0:
                        Vt[1] *= -1; Rd = Vt.T @ U.T
                    dyaw = float(np.arctan2(Rd[1, 0], Rd[0, 0])); yaw += dyaw; t = Rd @ (t - cb) + ca
                    if abs(dyaw) < 2e-4 and float(np.linalg.norm(ca - cb)) < 1e-4:
                        break
                c, s = np.cos(yaw), np.sin(yaw)
                M = np.c_[model[:, :2] @ np.array([[c, -s], [s, c]]).T + t, model[:, 2] + z0]
                d, _ = cKDTree(M).query(Y); res = float(np.median(d))
                if best is None or res < best['res']:
                    best = dict(x=float(t[0]), y=float(t[1]), yaw=float((yaw + np.pi) % (2 * np.pi) - np.pi), res=res, n=int(sel.sum()), fit=float(np.mean(d < .015)))
        best['T'] = self.refine(Y, best, model, z0)
        self.debug = dict(yellow=int(sel.sum()), res=round(best['res'], 4), fit=round(best['fit'], 3), tilt=best.get('tilt'))
        return best

    def refine(self, Y, pose, model, z0, iterations=25):
        """The planar fit made a full 3-D one (the trumpet lies tilted where it rests on the brush or hangs over the
        desk's edge: 2-6 degrees in 8 of the 50 dev starts, its ends 2-4 cm off the height the planar fit gives):
        point-to-point ICP of the model's points to the yellow ones from the planar pose. Kept only if the tilt stays
        under 10 degrees and the height within 4 cm, and the fit is better; else the planar pose (4x4)."""
        c, s = np.cos(pose['yaw']), np.sin(pose['yaw'])
        T0 = np.eye(4); T0[:2, :2] = [[c, -s], [s, c]]; T0[:3, 3] = [pose['x'], pose['y'], z0]
        T = T0.copy()
        for it in range(iterations):
            M = model @ T[:3, :3].T + T[:3, 3]
            d, j = cKDTree(M).query(Y); keep = d < max(.02, float(np.percentile(d, 80)))
            A = Y[keep]; B = M[j[keep]]; ca, cb = A.mean(axis=0), B.mean(axis=0)
            U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca)); R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2] *= -1; R = Vt.T @ U.T
            step = np.eye(4); step[:3, :3] = R; step[:3, 3] = ca - R @ cb
            T = step @ T
            if np.linalg.norm(step[:3, 3]) < 1e-4 and np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) < 2e-4:
                break
        tilt = float(np.degrees(np.arccos(np.clip(T[2, 2], -1, 1))))
        res = float(np.median(cKDTree(model @ T[:3, :3].T + T[:3, 3]).query(Y)[0]))
        pose['tilt'] = round(tilt, 1); pose['res3'] = round(res, 4)
        # (only where the planar fit is poor: on flat trumpets the 3-D fit to the side the camera sees made up a tilt of
        # 1-2 degrees and its error larger, over 112 close views of 14 dev starts)
        if 2.5 < tilt < 10. and abs(T[2, 3] - z0) < .04 and pose['res'] > .011 and res < pose['res'] - .002:
            pose['res'] = res
            return T
        return None

    def _height_error(self, Y, t, yaw):
        m = self.model; lo, res, top = m['lo'], float(m['res']), m['top']
        c, s = np.cos(yaw), np.sin(yaw); d = Y[:, :2] - t
        lx = c * d[:, 0] + s * d[:, 1]; ly = -s * d[:, 0] + c * d[:, 1]
        i = np.floor((lx - lo[0]) / res).astype(int); j = np.floor((ly - lo[1]) / res).astype(int)
        ok = (i >= 0) & (i < top.shape[0]) & (j >= 0) & (j < top.shape[1])
        h = np.full(len(Y), np.nan); h[ok] = top[i[ok], j[ok]]
        ok &= np.isfinite(h)
        if ok.sum() < 30:
            return 1.
        # (the trumpet's frame origin stands about 0.066 m over the desk: its lowest point is -0.065 in it)
        z0 = DESK['top'] - float(np.nanmin(m['bottom'])) - .0
        return float(np.median(np.abs((Y[ok, 2] - z0) - h[ok])))

    def dust(self, P, colour, pose, near=.02):
        """Dark specks on the trumpet: [xy z] in M of groups of pixels much darker than the yellow round them."""
        if pose is None:
            return []
        bright = colour @ np.array([.299, .587, .114])
        ylw = yellow(colour)
        if ylw.sum() < 50:
            return []
        ref = float(np.median(bright[ylw]))
        c, s = np.cos(pose['yaw']), np.sin(pose['yaw']); d = P[:, :2] - [pose['x'], pose['y']]
        lx = c * d[:, 0] + s * d[:, 1]; ly = -s * d[:, 0] + c * d[:, 1]
        on = (np.abs(ly) < .14) & (lx > -.5) & (lx < .45)
        dark = on & (bright < .35 * ref) & ~red(colour)
        if not dark.any():
            return []
        # (only those next to yellow: shadows on the desk are not on the trumpet)
        tree = cKDTree(P[ylw]); dd, _ = tree.query(P[dark], k=1)
        D = P[dark][dd < near]
        if not len(D):
            return []
        keys = np.floor(D / .02).astype(int); groups = {}
        for k, p in zip(map(tuple, keys), D):
            groups.setdefault(k, []).append(p)
        return [np.mean(g, axis=0) for g in groups.values() if len(g) >= 2]


def camera_rotation(cam):
    qx, qy, qz, qw = cam[3:]
    return np.array([[1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)], [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                     [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]]) @ np.diag([1., -1., -1.])


def pose_matrix(pose):
    """4x4 of the trumpet lying on the desk at @pose: its 3-D fit if there is one, else (x, y, yaw) with its frame's
    height fixed by the desk."""
    if pose.get('T') is not None:
        return np.asarray(pose['T'], float)
    c, s = np.cos(pose['yaw']), np.sin(pose['yaw'])
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]; T[:3, 3] = [pose['x'], pose['y'], pose['z']]; return T


SPECK = dict(value=70, saturation=140, near=.03, least=2, grow=4, yellow=True)


def specks(view, depth, rgb, cam, focal_ratio, base, T):
    """Dark specks on the trumpet posed @T (4x4, map), as the head camera shows them: [(xyz in the trumpet's own frame,
    pixels)]. The trumpet's outline in the image is its model's points projected (the silver mouthpiece and valve caps
    too, which are not yellow); a speck is a group of dark, greyish pixels in it (the dust is near black and grey; the
    trumpet's own shadows stay yellow-brown, saturated) whose depth puts it within SPECK['near'] of the model's surface.
    @view: TrumpetView (its model's points)."""
    h, w = depth.shape; f = focal_ratio * w; R = camera_rotation(cam)
    x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
    Bm = np.eye(4); Bm[:2, :2] = [[c, -s], [s, c]]; Bm[:3, 3] = [x, y, .012]          # (the base_link frame; cam is relative to it)
    model = view.fit_points
    # model points: trumpet frame -> map -> base -> camera
    P = np.c_[model, np.ones(len(model))] @ (np.linalg.inv(Bm) @ T).T
    q = (P[:, :3] - cam[:3]) @ R
    ok = q[:, 2] > .2
    u = (q[ok, 0] / q[ok, 2] * f + w / 2).astype(int); v = (q[ok, 1] / q[ok, 2] * f + h / 2).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    mask = np.zeros((h, w), bool); mask[v[inside], u[inside]] = True
    if mask.sum() < 50:
        return []
    mask = ndimage.binary_dilation(ndimage.binary_closing(mask, iterations=2), iterations=SPECK['grow'])
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV)
    # (and near the yellow the image shows: the fitted pose is a centimetre or two off, its outline with it)
    rr, gg, bb = [rgb[..., k].astype(np.int32) for k in range(3)]
    ylw = (rr > 120) & (gg > 90) & (bb < .6 * gg) & (rr > bb + 60)
    mask |= ndimage.binary_dilation(ylw & mask, iterations=4) if SPECK['yellow'] else False
    dark = mask & (hsv[..., 2] < SPECK['value']) & (hsv[..., 1] < SPECK['saturation']) & np.isfinite(depth) & (depth > .2)
    lab, n = ndimage.label(dark)
    if not n:
        return []
    vv, uu = np.nonzero(lab); ids = lab[vv, uu]; d = depth[vv, uu].astype(float)
    pc = np.stack([(uu + .5 - w / 2) / f * d, (vv + .5 - h / 2) / f * d, d], -1) @ R.T + cam[:3]
    world = np.c_[pc, np.ones(len(pc))] @ Bm.T
    local = (world @ np.linalg.inv(T).T)[:, :3]
    tree = cKDTree(view.model['visual'].astype(float))
    out = []
    for k in range(1, n + 1):
        m = ids == k
        if m.sum() < SPECK['least'] or m.sum() > 400:
            continue
        p = np.median(local[m], axis=0)
        if tree.query(p)[0] < SPECK['near']:
            out.append((p, int(m.sum())))
    return out
