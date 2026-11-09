"""State estimation for the battery task from onboard observations only.

Inputs: head and wrist RGB-D, the camera poses relative to the base, proprioception, plus the static map
of the rooms. No simulator state.

  * base pose in the map: Localizer (scan matching against the map)
  * batteries: their caps are saturated blue; a blue blob whose 3-D position lies just above a known work
    surface (the cabinet, the desk) is a battery
  * trash can: something knee-high standing on copy-room floor that the map says is empty
"""
import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import label
from localizer import Localizer, camera_points, BASE_Z
from robot_collision import RobotCollision
from wbik import Body
from kinematics import from_proprio

CAMERA = 'robot_r1::robot_r1:%s:Camera:0::'
HEAD = CAMERA % 'zed_link'
WRIST = {'left': CAMERA % 'left_realsense_link', 'right': CAMERA % 'right_realsense_link'}
SLOT = {'left': 0, 'right': 1, 'head': 2}          # order of the cameras in cam_rel_poses
FOCAL = {'head': 306. / 720., 'left': 388.66 / 480., 'right': 388.66 / 480.}
CAP_HUE = (98, 116); CAP_SATURATION = 110; CAP_VALUE = 45      # OpenCV HSV of the battery cap (hue 105, saturation 235)
BATTERY_TOP = .061             # height of a standing battery
CAN_HEIGHT = .445


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose2(x, y, yaw, z=0.):
    t = np.eye(4); c, s = np.cos(yaw), np.sin(yaw); t[:2, :2] = [[c, -s], [s, c]]; t[:3, 3] = [x, y, z]; return t


def camera_rotation(rel):
    x, y, z, w = rel[3:]
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])


CAN_OUTSIDE = .35           # the can is looked for this far outside the floor it started on (m): in the doorway


class Observer:
    def __init__(self, folder, scene):
        folder = Path(folder); self.scene = scene; self.loc = Localizer(scene)
        self.body = Body(folder); self.collision = RobotCollision(folder / 'robot_collision_spheres.json')
        self.steps = 0; self._cache = {}

    # ---- per-step input ------------------------------------------------------------------
    def frame(self, data):
        """data: dict of numpy arrays for this step (proprio, cam_rel_poses, RGB + depth of the three cameras)."""
        self.steps += 1; self.data = data; self._cache = {}
        self.q = from_proprio(data['robot_r1::proprio'].astype(float), self.body.names)

    def spheres(self):
        if 'spheres' not in self._cache:
            self._cache['spheres'] = (self.collision.centers(self.body.fk(self.q)[0]), self.collision.radii)
        return self._cache['spheres']

    def head_points(self, stride=6):
        key = ('head', stride)
        if key not in self._cache:
            self._cache[key] = camera_points(self.data[HEAD + 'depth_linear'], self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[2], FOCAL['head'], stride=stride)
        return self._cache[key]

    # ---- localisation --------------------------------------------------------------------
    def scan(self):
        return self.loc.scan(self.head_points(stride=3), exclude=self.spheres())

    def localize_global(self, scans):
        return self.loc.localize(scans)

    def wall_end(self, heights=(1.0, 1.9), ahead=(.5, 1.6), points=None):
        """Left end of a wall the robot faces, in the base frame: (x, y) of the last wall point before the view
        opens up, or None. Used in front of the office door, where the wall the door is in ends 0.6 m to the left.

        The wall is the mass of points at about one distance ahead; its end is where, going left, they stop while
        the view continues past the wall's plane. There the camera sees either something far behind the wall
        or, where the space beyond is deeper than the depth range (in front of the office door it is), only
        floor and ceiling running on behind the plane.
        """
        points = self.head_points(stride=2) if points is None else points; z = points[:, 2] + BASE_Z
        level = (z > heights[0]) & (z < heights[1]) & (points[:, 0] > ahead[0])
        band = points[level]; near = band[band[:, 0] < ahead[1]]
        if len(near) < 200:
            return None
        distance = float(np.median(near[:, 0])); wall = near[np.abs(near[:, 0] - distance) < .12]
        if len(wall) < 150:
            return None
        edge = float(np.percentile(wall[:, 1], 99.5))
        left = (points[:, 1] > edge + .03) & (points[:, 1] < edge + .30)
        beyond = points[left & level]; behind = int(np.sum(points[left][:, 0] > distance + .15))
        # If nothing at all shows behind the plane there, the "end" is the edge of the image and not of the wall.
        if len(beyond) >= 20:
            if float(np.median(beyond[:, 0])) < distance + .5:
                return None
        elif behind < 20:
            return None
        return np.array([float(np.median(wall[wall[:, 1] > edge - .04][:, 0])), edge])

    def table_edge(self, height, toward=0., points=None):
        """An edge of a table top @height above the floor, as a line in the base frame, or None if too little of
        it is in view. @toward is the direction (rad, base frame) from the robot across the edge onto the table:
        0 for a table the robot faces, +-pi/2 for one it drives along. Returns (a, b, strips, ends): in coordinates
        u (along @toward) and v (to the left of it) the edge is u = a + b * v; ends = (v where the edge ends on
        the right, on the left), each None unless the floor shows beyond it where the top would go on.

        The top is the horizontal patch of points at that height; its boundary on the robot's side, taken per
        strip of 3 cm along the edge, is the edge. Things standing in front of it at that height (the back of a
        chair pushed under the desk) are upright: they also show above the top, and strips where the nearest
        point is such a thing are left out. What hides part of the edge only moves the boundary away from the
        robot, so the line is fitted to the near side of the strips. Close to a table the camera does not see the
        near part of its top at all, and the boundary of the patch is then the lower border of the image: a strip
        counts only if something lower than the top (the floor in front of the table) shows on the robot's side
        of its boundary.
        """
        points = self.head_points(stride=2) if points is None else points; z = points[:, 2] + BASE_Z
        c, s = np.cos(toward), np.sin(toward)
        u = c * points[:, 0] + s * points[:, 1]; v = -s * points[:, 0] + c * points[:, 1]
        view = (points[:, 0] > .22) & (points[:, 0] < 1.9) & (np.abs(points[:, 1]) < 1.0) & (u > -.15) & (u < 1.7)
        flat = view & (np.abs(z - height) < .022); tall = view & (z > height + .06) & (z < height + .45); low = view & (z < height - .08)
        if flat.sum() < 200 or low.sum() < 50:
            return None
        cell = .03; top = np.c_[u[flat], v[flat]]
        upright = set(map(tuple, np.floor(np.c_[u[tall], v[tall]] / cell).astype(int))) if tall.any() else set()
        # Per strip: the nearest thing seen below the top.
        strips_low = np.floor(v[low] / cell).astype(int); order = np.argsort(strips_low); strips_low = strips_low[order]; u_low = u[low][order]
        first = np.flatnonzero(np.r_[True, np.diff(strips_low) != 0])
        nearest_low = dict(zip(strips_low[first].tolist(), np.minimum.reduceat(u_low, first).tolist()))
        keys = np.floor(top / cell).astype(int); order = np.lexsort((top[:, 0], keys[:, 1])); top, keys = top[order], keys[order]
        bounds = np.flatnonzero(np.diff(keys[:, 1])) + 1; rows = []
        for mine, mine_keys in zip(np.split(top, bounds), np.split(keys, bounds)):
            if len(mine) < 6 or mine[-1, 0] - mine[0, 0] < .10:        # a top continues away from its edge
                continue
            i0, j = int(mine_keys[0, 0]), int(mine_keys[0, 1])
            if any((i0 + di, j + dj) in upright for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                continue
            if nearest_low.get(j, np.inf) > mine[0, 0] - .03:           # nothing seen in front of it: the border of the image
                continue
            rows.append((float(np.mean(mine[:, 1])), float(mine[0, 0]), j))
        if len(rows) < 8:
            return None
        rows = np.array(rows); keep = np.ones(len(rows), bool); a = b = 0.
        for _ in range(6):
            if keep.sum() < 7 or np.ptp(rows[keep, 0]) < .20:
                return None
            b, a = np.polyfit(rows[keep, 0], rows[keep, 1], 1); residual = rows[:, 1] - (a + b * rows[:, 0])
            fresh = (residual < .012) & (residual > -.03)
            if np.array_equal(fresh, keep):
                break
            keep = fresh
        if abs(b) > .35:
            return None
        # Where the edge ends. The strips on the line form a run; past a real end of the top the next strips
        # show no top behind the line, but the floor there (past the border of the view they show nothing, and
        # behind something standing on the desk they show that thing).
        strips = np.sort(rows[keep, 2].astype(int)); depth = u - (a + b * v); strip_of = np.floor(v / cell).astype(int); ends = [None, None]
        behind = depth > .04; on_top = flat & behind; ground = low & (z < .06) & behind & (depth < .45)
        for index, (last, sign) in enumerate(((strips[0], -1), (strips[-1], 1))):
            beyond = (sign * (strip_of - last) >= 2) & (sign * (strip_of - last) <= 5)
            if np.sum(on_top & beyond) < 10 and np.sum(ground & beyond) >= 40:
                rim = flat & (np.abs(strip_of - last) <= 1) & (depth > -.02) & (depth < .10)
                if rim.sum() >= 5:
                    ends[index] = float(v[rim].max() if sign > 0 else v[rim].min())
        return float(a), float(b), int(keep.sum()), tuple(ends)

    # ---- batteries -----------------------------------------------------------------------
    def blue_blobs(self, camera='head', minimum=4):
        """Blue blobs of one camera as base-frame points: list of dict(point, pixels, spread)."""
        key = ('blobs', camera)
        if key in self._cache:
            return self._cache[key]
        prefix = HEAD if camera == 'head' else WRIST[camera]
        rgb = np.ascontiguousarray(self.data[prefix + 'rgb'][..., :3]); depth = self.data[prefix + 'depth_linear']
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = ((hsv[..., 0] >= CAP_HUE[0]) & (hsv[..., 0] <= CAP_HUE[1]) & (hsv[..., 1] >= CAP_SATURATION) & (hsv[..., 2] >= CAP_VALUE)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        rel = self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[camera]]; rot = camera_rotation(rel)
        h, w = depth.shape; f = FOCAL[camera] * w; out = []
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] < minimum:
                continue
            v, u = np.nonzero(labels == index); d = depth[v, u].astype(float); ok = np.isfinite(d) & (d > .08) & (d < 6.)
            if ok.sum() < minimum:
                continue
            u, v, d = u[ok], v[ok], d[ok]
            near = d < np.median(d) + .04          # pixels on the blob's rim can carry the depth of what is behind it
            pc = np.stack([(u + .5 - w / 2) / f * d, (v + .5 - h / 2) / f * d, d], -1)[near]
            points = pc @ rot.T + rel[:3]
            out.append(dict(point=np.median(points, axis=0), pixels=int(near.sum()), spread=float(np.max(np.ptp(points, axis=0)))))
        self._cache[key] = out
        return out

    def batteries(self, base_pose, surface, camera='head', slack=.04):
        """Batteries standing on @surface: list of dict(xy = map position, point = base-frame cap position, pixels).

        slack: how far outside the surface's footprint (as placed by @base_pose) a cap may appear. A cap is
        recognised by its colour and its height above the surface; the footprint only tells surfaces apart, so
        where the pose may be off by a decimetre (behind the desk) the caller allows for that.
        """
        world = pose2(*base_pose); found = []
        lo, hi = np.asarray(surface['lo'], float), np.asarray(surface['hi'], float)
        for blob in self.blue_blobs(camera):
            p = blob['point']; height = p[2] + BASE_Z - surface['height']
            xy = world[:2, :2] @ p[:2] + world[:2, 3]
            if not (-.02 < height < BATTERY_TOP + .04) or blob['spread'] > .09:
                continue
            if np.all(xy > lo - slack) and np.all(xy < hi + slack):
                found.append(dict(xy=xy, point=p.copy(), pixels=blob['pixels']))
        return found

    # ---- trash can -----------------------------------------------------------------------
    def can(self, base_pose, floor):
        """The trash can on @floor = dict(lo, hi) (map frame): dict(xy, rim points seen, points) or None."""
        points = self.head_points(stride=2); height = points[:, 2] + BASE_Z
        points = points[(height > .10) & (height < CAN_HEIGHT + .03)]
        if len(points) < 30:
            return None
        centres, radii = self.spheres()
        points = points[~(np.linalg.norm(points[:, None, :] - centres[None], axis=2) < radii[None] + .06).any(axis=1)]
        world = pose2(*base_pose); xy = points[:, :2] @ world[:2, :2].T + world[:2, 3]; z = points[:, 2] + BASE_Z
        i, j = self.scene.cell(xy); ok = self.scene.inside(i, j)
        ic, jc = np.clip(i, 0, self.scene.shape[0] - 1), np.clip(j, 0, self.scene.shape[1] - 1)
        lo, hi = np.asarray(floor['lo'], float), np.asarray(floor['hi'], float)
        # On floor the map has empty, in the room or just outside it: the can may stand in the doorway, half over
        # the room's edge (the walls themselves are in the map and are cut out by it).
        ok &= (self.scene.top[ic, jc] < .05) & (self.scene.distance[ic, jc] > .08) & np.all(xy > lo - CAN_OUTSIDE, axis=1) & np.all(xy < hi + CAN_OUTSIDE, axis=1)
        xy, z = xy[ok], z[ok]
        if len(xy) < 30:
            return None
        cell = .03; keys = np.floor(xy / cell).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool)
        grid[tuple((keys - origin + 1).T)] = True
        labels, count = label(grid, structure=np.ones((3, 3))); tag = labels[tuple((keys - origin + 1).T)]; best = None
        for index in range(1, count + 1):
            member = tag == index; n = int(member.sum())
            if n < 30:
                continue
            p, zz = xy[member], z[member]; extent = p.max(0) - p.min(0)
            if zz.max() < CAN_HEIGHT - .08 or extent.max() > .62 or extent.max() < .15:
                continue
            rim = p[zz > CAN_HEIGHT - .05]; yaw = None
            centre = (rim.min(0) + rim.max(0)) / 2 if len(rim) >= 12 else (p.min(0) + p.max(0)) / 2
            if len(rim) >= 150:         # from close by, when most of the rim is in view; a partial rim gives a box that is off
                # The rim is a rectangle of 0.42 m by 0.33 m: the smallest box around its points gives the can's
                # centre and heading (its long side) when both sides come out about right. (The top 5 cm of the
                # walls, seen from above, lie up to 5 cm inside that outline: the walls lean, so fitting the
                # outline itself to these points does not work.)
                (cx, cy), (w, h), angle = cv2.minAreaRect(rim.astype(np.float32)); long_side, short_side = max(w, h), min(w, h)
                if abs(long_side - .416) < .05 and abs(short_side - .325) < .05:
                    centre = np.array([cx, cy]); yaw = float(np.radians(angle if w >= h else angle + 90.))
            if best is None or n > best['points']:
                best = dict(xy=centre, rim=int(len(rim)), points=n, extent=extent, yaw=yaw)
        return best
