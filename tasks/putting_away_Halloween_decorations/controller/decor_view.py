"""The two pumpkins, the three candles and the caldron on the living-room floor, from the head camera's depth image.

Points that stand above the floor where the map has free floor are grouped; a group is one of the three kinds by its
height and the room it takes on the floor: the caldron 0.25 m high and 0.29 m across, a candle 0.117 m high (a 5 mm
wick on top) and 0.10 m across with a flat top, a pumpkin 0.10 m high (a stem up to 0.125 m) and 0.15 m across.
Colour tells a pumpkin (orange) from a candle (cream) where the colour image is given. Where a hand takes each: the middle of the candle's top (its wick), the pumpkin's stem (its highest points), the
caldron's middle (the hand then takes its rim on the near side).

Everything returned is in the map frame M.
"""
import numpy as np
from scipy.ndimage import label
from manip import BASE_Z

CELL = .02
ABOVE_FLOOR = .015                  # points this far above the floor belong to something
BAND_TOP = .40
CLEAR = .06                         # ... and this far from anything the map has (walls, furniture)
REACH = 4.5                         # points further from the base than this are not used
CALDRON = dict(top=(.20, .30), across=(.12, .36), rim=.20, radius=.147)
CANDLE = dict(top=(.105, .155), across=.13, cap=.105, bias=(.004, .0037))
PUMPKIN = dict(top=(.080, .135), across=(.09, .19), stem=.104)
CAP_SHARE = .25                     # a candle shows its flat top: at least this share of its points lie above CANDLE['cap']
ORANGE = .35                        # a group whose points are this much orange (red well above blue) is a pumpkin


def circle(points):
    """Least-squares circle through @points (N, 2): centre, radius."""
    x, y = points[:, 0], points[:, 1]; a = np.c_[x, y, np.ones(len(x))]; b = x * x + y * y
    sol = np.linalg.lstsq(a, b, rcond=None)[0]; centre = .5 * sol[:2]
    return centre, float(np.sqrt(max(sol[2] + centre @ centre, 0.)))


class DecorView:
    def __init__(self, scene):
        self.scene = scene

    def groups(self, ob, base_pose, stride=1, ignore=()):
        """Groups of points above the free floor: list of dict(points (N, 3 in M), top, centre (xy), n, cut (the group
        touches the image's border)). ignore: boxes (x0, y0, x1, y1) in M whose points are not used (an open drawer)."""
        local = ob.points('head', stride); height = local[:, 2] + BASE_Z
        local = local[(height > ABOVE_FLOOR) & (height < BAND_TOP) & (np.hypot(local[:, 0], local[:, 1]) < REACH)]
        if len(local) < 5:
            return []
        local = ob.clear_of_robot(local)
        x, y, yaw = base_pose; c, s = np.cos(yaw), np.sin(yaw)
        xy = np.c_[x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]]; z = local[:, 2] + BASE_Z
        i, j = self.scene.cell(xy); ok = self.scene.inside(i, j); ic, jc = np.clip(i, 0, self.scene.shape[0] - 1), np.clip(j, 0, self.scene.shape[1] - 1)
        ok &= (self.scene.top[ic, jc] < .03) & (self.scene.distance[ic, jc] > CLEAR)
        for x0, y0, x1, y1 in ignore:
            ok &= ~((xy[:, 0] > x0) & (xy[:, 0] < x1) & (xy[:, 1] > y0) & (xy[:, 1] < y1))
        xy, z, local = xy[ok], z[ok], local[ok]
        if len(xy) < 5:
            return []
        out = []
        keys = np.floor(xy / CELL).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool); grid[tuple((keys - origin + 1).T)] = True
        tags, count = label(grid, structure=np.ones((3, 3))); tag = tags[tuple((keys - origin + 1).T)]
        for index in range(1, count + 1):
            member = tag == index
            if member.sum() < 5:
                continue
            out.append(dict(points=np.c_[xy[member], z[member]], top=float(np.percentile(z[member], 98)), centre=xy[member].mean(axis=0), n=int(member.sum()), cut=bool(ob.at_border(local[member])), local=local[member]))
        return out

    @staticmethod
    def orange(ob, local, image):
        """Share of the points @local (base frame) that are orange in the head camera's colour image @image."""
        u, v, size = ob.pixels(local, 'head'); scale = image.shape[0] / size
        rgb = image[np.clip((v * scale).astype(int), 0, image.shape[0] - 1), np.clip((u * scale).astype(int), 0, image.shape[1] - 1)].astype(int)
        return float(np.mean((rgb[:, 0] - rgb[:, 2] > 70) & (rgb[:, 0] > 90)))

    def see(self, ob, base_pose, stride=1, ignore=(), image=None):
        """What this look shows: list of dict(kind ('pumpkin', 'candle', 'caldron'), xy (where a hand takes it; for the
        caldron its middle), top, across, n, cut, range, orange). image: the head camera's colour image (H x W x 3),
        or None (the kind is then told by shape alone)."""
        out = []
        for group in self.groups(ob, base_pose, stride, ignore):
            points = group['points']; top = group['top']; extent = float(max(np.ptp(points[:, :2], axis=0)))
            orange = None if image is None else self.orange(ob, group['local'], image)
            rng = float(np.hypot(*(group['centre'] - np.asarray(base_pose[:2], float))))
            thing = None
            if CALDRON['top'][0] < top < CALDRON['top'][1] and CALDRON['across'][0] < extent < CALDRON['across'][1] and group['n'] >= 12:
                rim = points[points[:, 2] > CALDRON['rim']]
                centre = group['centre']
                if len(rim) >= 12:
                    fitted, radius = circle(rim[:, :2])
                    if abs(radius - CALDRON['radius']) < .05:
                        centre = fitted
                thing = dict(kind='caldron', xy=np.asarray(centre, float), top=top, across=extent)
            elif CANDLE['top'][0] < top < CANDLE['top'][1] and extent < CANDLE['across'] and group['n'] >= 8 and (orange < ORANGE if orange is not None else float(np.mean(points[:, 2] > CANDLE['cap'])) >= CAP_SHARE):
                # (the top's points lean to the side the camera sees it from: by 0.4 cm + 0.37 % of the range, measured on
                # 245 recorded looks; put right along the line of sight)
                cap = points[points[:, 2] > CANDLE['cap']]; xy = cap[:, :2].mean(axis=0); away = xy - np.asarray(base_pose[:2], float); away /= max(float(np.linalg.norm(away)), 1e-6)
                thing = dict(kind='candle', xy=xy + (CANDLE['bias'][0] + CANDLE['bias'][1] * rng) * away, top=top, across=extent)
            elif PUMPKIN['top'][0] < top < PUMPKIN['top'][1] and PUMPKIN['across'][0] < extent < PUMPKIN['across'][1] and group['n'] >= 8 and (orange is None or orange >= ORANGE):
                stem = points[points[:, 2] > PUMPKIN['stem']]
                xy = stem[:, :2].mean(axis=0) if len(stem) >= 3 else circle(points[:, :2])[0] if group['n'] >= 20 else group['centre']
                thing = dict(kind='pumpkin', xy=np.asarray(xy, float), top=top, across=extent, stem=int(len(stem)))
            if thing is not None:
                thing.update(n=group['n'], cut=group['cut'], range=rng, orange=orange)
                out.append(thing)
        return out
