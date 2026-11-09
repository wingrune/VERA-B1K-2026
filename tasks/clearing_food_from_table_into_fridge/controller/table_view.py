"""The two plates with their pizzas and the two bowls on the breakfast table, from the head camera's depth and colour
images.

The table's top is 0.679 m up and never moves (map x 3.826-4.756, y 0.693-2.183). On it: plates 0.408 m across and
2.4 cm high (white), on each a pizza 0.30 m across and 1.7-4 cm high (orange and red), not always in the plate's
middle; bowls 0.15 m across and 5.9 cm high (white), open at the top. What shows above the top is sorted by colour and
height: orange points are pizza; white points higher than a plate's rim are bowl; the other white points are plate.
Each kind is grouped, and each group's middle found: a pizza's from its points, a plate's by a circle of its radius
through its rim's points (the pizza hides its middle), a bowl's by a circle through its rim.

Everything returned is in the map frame M.
"""
import numpy as np
from scipy.ndimage import label
from manip import BASE_Z

TABLE = dict(x=(3.826, 4.756), y=(.693, 2.183), top=.679)
CELL = .02
PLATE = dict(radius=.204, rim=.703)
PIZZA = dict(radius=.15)
BOWL = dict(radius=.075, top=.738)
ABOVE = .013                        # points this far above the table's top belong to something on it (the camera puts the top itself at 0.686)
BOWL_FROM = .728                    # white points higher than this are a bowl's rim (0.73-0.745; a pizza's top shows up to 0.725)
PLATE_TOP = .716                    # white points lower than this are a plate's (its rim shows at 0.70-0.71)


def circle(points, radius=None):
    """Least-squares circle through @points (N, 2); with @radius given, only its centre is fitted (Gauss-Newton from the
    points' mean pushed away from where they are). Returns centre, radius."""
    x, y = points[:, 0], points[:, 1]; a = np.c_[x, y, np.ones(len(x))]; b = x * x + y * y
    sol = np.linalg.lstsq(a, b, rcond=None)[0]; centre = .5 * sol[:2]; r = float(np.sqrt(max(sol[2] + centre @ centre, 0.)))
    if radius is None:
        return centre, r
    if not np.all(np.isfinite(centre)) or abs(r - radius) > .05:
        centre = points.mean(axis=0)
    for _ in range(20):
        d = points - centre; dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
        residual = dist - radius; jac = -d / dist[:, None]
        step = np.linalg.lstsq(jac, -residual, rcond=None)[0]; centre = centre + step
        if float(np.linalg.norm(step)) < 1e-5:
            break
    return centre, radius


def groups(xy, min_points):
    """Connected groups of the 2 cm cells @xy (N, 2) take: list of index arrays."""
    if len(xy) < min_points:
        return []
    keys = np.floor(xy / CELL).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool); grid[tuple((keys - origin + 1).T)] = True
    tags, count = label(grid, structure=np.ones((3, 3))); tag = tags[tuple((keys - origin + 1).T)]
    return [np.flatnonzero(tag == k) for k in range(1, count + 1) if (tag == k).sum() >= min_points]


class TableView:
    def __init__(self, scene=None):
        self.scene = scene

    def points(self, ob, base_pose, stride=1):
        """Head-camera points above the table's top, in M: (xyz N x 3, rgb N x 3 or None)."""
        local = ob.points('head', stride); height = local[:, 2] + BASE_Z
        local = local[(height > TABLE['top'] + ABOVE) & (height < TABLE['top'] + .15)]
        if len(local) < 5:
            return np.zeros((0, 3)), None
        local = ob.clear_of_robot(local)
        x, y, yaw = base_pose; c, s = np.cos(yaw), np.sin(yaw)
        xy = np.c_[x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]]; z = local[:, 2] + BASE_Z
        (x0, x1), (y0, y1) = TABLE['x'], TABLE['y']
        ok = (xy[:, 0] > x0 - .02) & (xy[:, 0] < x1 + .02) & (xy[:, 1] > y0 - .02) & (xy[:, 1] < y1 + .02)
        return np.c_[xy[ok], z[ok]], local[ok]

    @staticmethod
    def colours(ob, local, image):
        u, v, size = ob.pixels(local, 'head'); scale = image.shape[0] / size
        return image[np.clip((v * scale).astype(int), 0, image.shape[0] - 1), np.clip((u * scale).astype(int), 0, image.shape[1] - 1)].astype(int)

    def see(self, ob, base_pose, image, stride=1):
        """What this look shows on the table: list of dict(kind ('plate', 'pizza', 'bowl'), xy, n, cut (the group
        touches the image's border), range, fit (mean distance of its points from the fitted circle, for plates and
        bowls)). @image: the head camera's colour image."""
        pts, local = self.points(ob, base_pose, stride)
        if len(pts) < 20:
            return []
        rgb = self.colours(ob, local, image); r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        orange = (r - b > 60) & (r > 110)
        white = (r > 150) & (g > 150) & (b > 140) & (np.abs(r - b) < 45)
        here = np.asarray(base_pose[:2], float); out = []

        def add(kind, idx, xy, fit=None):
            out.append(dict(kind=kind, xy=np.asarray(xy, float), n=int(len(idx)), cut=bool(ob.at_border(local[idx])), range=float(np.linalg.norm(np.asarray(xy) - here)), fit=fit))

        # pizzas: orange groups 0.15-0.35 m across
        for idx in groups(pts[orange, :2], 15):
            idx = np.flatnonzero(orange)[idx]; extent = float(max(np.ptp(pts[idx, :2], axis=0)))
            if extent < .10:
                continue
            if extent > .40:
                # (two pizzas touching: split along the long way)
                d = pts[idx, :2] - pts[idx, :2].mean(0); axis = np.linalg.svd(d, full_matrices=False)[2][0]; side = d @ axis > 0
                for part in (idx[side], idx[~side]):
                    add('pizza', part, pts[part, :2].mean(0))
                continue
            add('pizza', idx, pts[idx, :2].mean(0))
        pizzas = [o['xy'] for o in out]
        # bowls: what stands higher than a pizza's top and is not orange: a bowl's rim and walls
        high = ~orange & (pts[:, 2] > BOWL_FROM) & (pts[:, 2] < BOWL['top'] + .02)
        parts = []
        for idx in groups(pts[high, :2], 10):
            idx = np.flatnonzero(high)[idx]; extent = float(max(np.ptp(pts[idx, :2], axis=0)))
            if .20 < extent < .40:
                # (two bowls side by side: split along the long way)
                d = pts[idx, :2] - pts[idx, :2].mean(0); axis = np.linalg.svd(d, full_matrices=False)[2][0]; side = d @ axis > 0
                parts += [idx[side], idx[~side]]
            elif .04 <= extent <= .20:
                parts.append(idx)
        for idx in parts:
            if len(idx) < 8:
                continue
            rim = pts[idx]; rim = rim[rim[:, 2] > np.percentile(rim[:, 2], 50)]
            centre, _ = circle(rim[:, :2], BOWL['radius'] - .004) if len(rim) >= 8 else (pts[idx, :2].mean(0), None)
            if float(np.linalg.norm(centre - pts[idx, :2].mean(0))) > .08:
                centre = pts[idx, :2].mean(0)
            fit = float(np.mean(np.abs(np.linalg.norm(rim[:, :2] - centre, axis=1) - BOWL['radius'] + .004))) if len(rim) else None
            if fit is not None and fit > .012:
                continue                    # (not a bowl's round rim: 1-6 mm on 76 looks at bowls; a plate's rim raised on another plate showed here)
            add('bowl', idx, centre, fit=fit)
        bowls = [o['xy'] for o in out if o['kind'] == 'bowl']
        # plates: the white of a plate's rim is a ring 0.15-0.204 m from its middle (the pizza hides the rest); for each
        # pizza, the middle within 12 cm of it whose ring takes most of the low white points and leaves least of them
        # just outside it
        low = white & (pts[:, 2] < PLATE_TOP)
        for bxy in bowls:
            low &= np.linalg.norm(pts[:, :2] - bxy, axis=1) > BOWL['radius'] + .02
        rim = pts[low, :2]
        for k, pz in enumerate(pizzas):
            if len(rim) < 15:
                break
            near = rim[np.linalg.norm(rim - pz, axis=1) < .36]
            if len(near) < 15:
                continue
            best = None
            for dx in np.arange(-.12, .121, .01):
                for dy in np.arange(-.12, .121, .01):
                    if dx * dx + dy * dy > .0145:
                        continue
                    c = pz + [dx, dy]; d = np.linalg.norm(near - c, axis=1)
                    score = float(np.sum((d > .14) & (d < PLATE['radius'] + .004)) - 2. * np.sum((d > PLATE['radius'] + .012) & (d < PLATE['radius'] + .06)))
                    if best is None or score > best[0]:
                        best = (score, c)
            if best is not None and best[0] > 10:
                c = best[1]; ring = near[(np.linalg.norm(near - c, axis=1) > .14) & (np.linalg.norm(near - c, axis=1) < PLATE['radius'] + .004)]
                centre, _ = circle(ring, PLATE['radius'] - .006) if len(ring) >= 12 else (c, None)
                if float(np.linalg.norm(centre - c)) > .03:
                    centre = c
                idx = np.flatnonzero(low)[np.linalg.norm(rim - centre, axis=1) < PLATE['radius'] + .004]
                add('plate', idx, centre, fit=float(best[0])); out[-1]['pizza'] = np.asarray(pz, float)
        return out


def _bar_things(self, ob, base_pose, stride=2):
    """Tupperwares on the bar, from the head camera's depth only: groups of points 4-16 cm above the bar's top, 10-32 cm
    across. dict(kind='tupperware', xy (the middle of the group's extent), n, cut, range)."""
    from kitchen import BAR, TUPPERWARE
    local = ob.points('head', stride); height = local[:, 2] + BASE_Z
    local = local[(height > BAR['top'] + .04) & (height < BAR['top'] + TUPPERWARE['height'] + .03)]
    if len(local) < 10:
        return []
    local = ob.clear_of_robot(local)
    x, y, yaw = base_pose; c, s = np.cos(yaw), np.sin(yaw)
    xy = np.c_[x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]]
    (x0, x1), (y0, y1) = BAR['x'], BAR['y']
    ok = (xy[:, 0] > x0 - .02) & (xy[:, 0] < x1 + .02) & (xy[:, 1] > y0 - .02) & (xy[:, 1] < y1 + .02)
    if self.scene is not None and len(xy):
        # (what the map has standing on the bar is not a tupperware)
        i, j = self.scene.cell(xy); inside = self.scene.inside(i, j)
        ic, jc = np.clip(i, 0, self.scene.shape[0] - 1), np.clip(j, 0, self.scene.shape[1] - 1)
        ok &= inside & (self.scene.top[ic, jc] < BAR['top'] + .03)
    xy, local = xy[ok], local[ok]
    if len(xy) < 10:
        return []
    here = np.asarray(base_pose[:2], float); out = []
    for idx in groups(xy, 10):
        g = xy[idx]; extent = g.max(0) - g.min(0)
        if not (.10 < float(extent.max()) < .32):
            continue
        middle = (g.max(0) + g.min(0)) / 2
        if not (x0 + .06 < middle[0] < x1 - .08 and y0 + .07 < middle[1] < y1 - .07):
            continue        # (z02: a tupperware's middle stands 0.11 m or more in from the bar's edges; a patch at the bar's south-east corner, x 8.40 y -0.88, passed for one and kept the second tupperware from being looked for, 302)
        out.append(dict(kind='tupperware', xy=middle, n=int(len(idx)), cut=bool(ob.at_border(local[idx])), range=float(np.linalg.norm(middle - here))))
    return out


TableView.bar_things = _bar_things

