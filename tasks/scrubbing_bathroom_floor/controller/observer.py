"""State estimation for the car task from onboard observations only.

Inputs: the head camera's depth image, the camera poses relative to the base, proprioception, plus the static
map of the house's lower floor (which has the car and the coffee table). No simulator state.

  * base pose in the map: Localizer (scan matching against the map)
  * the toy box: a thing of its size (0.305 x 0.455 m, 0.155 m high) on floor that the map has empty; its
    place and direction from the rectangle of its rim
  * the camera and the racket: what stands and what lies on the coffee table (0.42 m high, in the map)
  * the trunk lid's rear edge, from the stance behind the car
"""
import numpy as np
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
CELL = .02
BOX_SIZE = (.455, .305, .155)   # the toy box: long side, short side, height of its rim above the floor
BOX_BAND = (.035, .21)          # heights above the floor in which it is looked for (with the camera in it, it reaches 0.175)
BOX_CLEAR = .04                 # ... on floor at least this far from anything the map has
TABLE_TOP = .423                # height of the coffee table's top
CAMERA_TOP = .157               # the camera stands this high on the table as the depth image shows it (the hump on its top; its body ends 0.125-0.136 up)
CAMERA_BODY = (.198, .113)      # its body from above: long and short side
HUMP_BAND = .009                # points this far below the camera's highest belong to the top of the hump (6.5 x 5.5 cm; the body's top starts 1.5 cm lower)
RACKET_BAND = (.010, .060)      # heights above the table in which the racket lies (its handle is 0.037 m thick, its head 0.027)
RACKET_LENGTH = .829; HANDLE_LENGTH = .27; HANDLE_WIDTH = .047
HANDLE_HOLD = .13               # the hand takes the handle this far from its end
LID_APEX = (-6.6686, 6.878)     # the point of the V that the closed trunk lid's rear edge makes, in the map (the car is fixed)
LID_SLOPE = .120; LID_ROUND = .060      # that V: metres back per metre to the side, and how round its point is
# The rear face of the car's bumper 0.50-0.68 m above the floor, as the depth camera shows it from behind the car:
# (y, x) in the map, every 4 cm across (the same to a fifth of a millimetre in 24 frames with true poses).
BUMPER = ((6.26, -6.7258), (6.30, -6.7500), (6.34, -6.7658), (6.38, -6.7752), (6.42, -6.7844), (6.46, -6.7938), (6.50, -6.8030), (6.54, -6.8115), (6.58, -6.8150), (6.62, -6.8178), (6.66, -6.8204),
          (6.70, -6.8229), (6.74, -6.8256), (6.78, -6.8282), (6.82, -6.8306), (6.86, -6.8334), (6.90, -6.8337), (6.94, -6.8312), (6.98, -6.8285), (7.02, -6.8258), (7.06, -6.8235), (7.10, -6.8207),
          (7.14, -6.8179), (7.18, -6.8153), (7.22, -6.8125), (7.26, -6.8047), (7.30, -6.7952), (7.34, -6.7860), (7.38, -6.7766), (7.42, -6.7674), (7.46, -6.7538), (7.50, -6.7301))


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose2(x, y, yaw, z=0.):
    t = np.eye(4); c, s = np.cos(yaw), np.sin(yaw); t[:2, :2] = [[c, -s], [s, c]]; t[:3, 3] = [x, y, z]; return t


def rectangle(points, step=np.radians(1.)):
    """The rectangle of least area round 2-D @points: (centre, direction of its long side (rad, modulo half a turn), long, short)."""
    best = None
    for angle in np.arange(0., np.pi / 2, step):
        c, s = np.cos(angle), np.sin(angle); u = points @ np.array([c, s]); v = points @ np.array([-s, c])
        lo = np.array([np.percentile(u, .5), np.percentile(v, .5)]); hi = np.array([np.percentile(u, 99.5), np.percentile(v, 99.5)]); size = hi - lo
        if best is None or size[0] * size[1] < best[0]:
            best = (size[0] * size[1], angle, lo, hi)
    _, angle, lo, hi = best; c, s = np.cos(angle), np.sin(angle); mid = (lo + hi) / 2; size = hi - lo
    centre = np.array([c * mid[0] - s * mid[1], s * mid[0] + c * mid[1]])
    return (centre, float(angle), float(size[0]), float(size[1])) if size[0] >= size[1] else (centre, float(wrap(angle + np.pi / 2)), float(size[1]), float(size[0]))


class Observer:
    def __init__(self, folder, scene):
        folder = Path(folder); self.scene = scene; self.loc = Localizer(scene)
        self.body = Body(folder); self.collision = RobotCollision(folder / 'robot_collision_spheres.json')
        self.steps = 0; self._cache = {}
        self.held = None            # (centres, radii) in the base frame of what the hands hold: cut out of the scans too
        table = next(o for o in scene.meta['objects'] if o['category'] == 'coffee_table')
        self.table = (np.array(table['aabb_lo'][:2]), np.array(table['aabb_hi'][:2]))

    # ---- per-step input ------------------------------------------------------------------
    def frame(self, data):
        """data: dict of numpy arrays for this step (proprio, cam_rel_poses, depth of the three cameras)."""
        self.steps += 1; self.data = data; self._cache = {}
        self.q = from_proprio(data['robot_r1::proprio'].astype(float), self.body.names)

    def spheres(self):
        """The robot's own spheres (base frame) and, with them, what its hands hold: not part of the scene."""
        if 'spheres' not in self._cache:
            centres, radii = self.collision.centers(self.body.fk(self.q)[0]), self.collision.radii
            if self.held is not None and len(self.held[0]):
                centres = np.vstack([centres, self.held[0]]); radii = np.r_[radii, self.held[1]]
            self._cache['spheres'] = (centres, radii)
        return self._cache['spheres']

    def points(self, camera='head', stride=2):
        key = (camera, stride)
        if key not in self._cache:
            prefix = HEAD if camera == 'head' else WRIST[camera]
            self._cache[key] = camera_points(self.data[prefix + 'depth_linear'], self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[camera]], FOCAL[camera], stride=stride)
        return self._cache[key]

    def head_points(self, stride=6):
        return self.points('head', stride)

    def clear_of_robot(self, points, margin=.05):
        centres, radii = self.spheres()
        return points[~(np.linalg.norm(points[:, None, :] - centres[None], axis=2) < radii[None] + margin).any(axis=1)] if len(points) else points

    def pixels(self, points, camera='head'):
        """Image coordinates (u, v) of base-frame @points in @camera's image, and the image size."""
        pose = self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[camera]]; x, y, z, w = pose[3:]
        rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])
        size = self.data[(HEAD if camera == 'head' else WRIST[camera]) + 'depth_linear'].shape[0]; f = FOCAL[camera] * size
        pc = (np.asarray(points, float) - pose[:3]) @ rot; depth = np.maximum(pc[:, 2], 1e-6)
        return pc[:, 0] / depth * f + size / 2, pc[:, 1] / depth * f + size / 2, size

    def at_border(self, points, camera='head', margin=8):
        u, v, size = self.pixels(points, camera)
        return bool(((u < margin) | (u > size - margin) | (v < margin) | (v > size - margin)).any())

    # ---- localisation --------------------------------------------------------------------
    def scan(self):
        return self.loc.scan(self.head_points(stride=3), exclude=self.spheres())

    def eye(self):
        """The head camera in the base frame, its height above the floor."""
        pose = self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT['head']]
        return float(pose[0]), float(pose[1]), float(pose[2]) + BASE_Z

    def localize_global(self, scans):
        self.loc.eye = self.eye()
        return self.loc.localize(scans)

    # ---- the toy box -------------------------------------------------------------------------
    def box_on_floor(self, base_pose, reach=3.5):
        """The toy box as the head camera shows it: dict(xy (map), yaw (map: direction of its long side, modulo half
        a turn), point, axis (the same in the base frame), long, short (m, as measured), points, cut), or None.

        Points between BOX_BAND heights, away from the robot's own body, on cells the map has empty, are grouped;
        a group as high as the box whose rim (the points near its top) fills a rectangle of the box's size is it.
        cut: it touches the border of the image, so the rectangle is not to be trusted.
        """
        points = self.points('head', stride=2); height = points[:, 2] + BASE_Z
        points = points[(height > BOX_BAND[0]) & (height < BOX_BAND[1]) & (np.hypot(points[:, 0], points[:, 1]) < reach)]
        points = self.clear_of_robot(points)
        if len(points) < 40:
            return None
        world = pose2(*base_pose); xy = points[:, :2] @ world[:2, :2].T + world[:2, 3]; z = points[:, 2] + BASE_Z
        i, j = self.scene.cell(xy); ok = self.scene.inside(i, j)
        ic, jc = np.clip(i, 0, self.scene.shape[0] - 1), np.clip(j, 0, self.scene.shape[1] - 1)
        ok &= (self.scene.top[ic, jc] < .03) & (self.scene.distance[ic, jc] > BOX_CLEAR)
        xy, z, local = xy[ok], z[ok], points[ok]
        if len(xy) < 40:
            return None
        keys = np.floor(xy / CELL).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool)
        grid[tuple((keys - origin + 1).T)] = True
        labels, count = label(grid, structure=np.ones((3, 3))); tag = labels[tuple((keys - origin + 1).T)]; best = None
        for index in range(1, count + 1):
            member = tag == index
            if member.sum() < 40:
                continue
            zz = z[member]; top = float(np.percentile(zz, 97)); extent = float(np.ptp(xy[member], axis=0).max())
            if not (BOX_SIZE[2] - .03 < top < BOX_BAND[1]) or not (.20 < extent < .62):
                continue
            rim = local[member][np.abs(zz - BOX_SIZE[2]) < .012]
            if len(rim) < 30:
                continue
            centre, angle, long, short = rectangle(rim[:, :2]); cut = self.at_border(local[member])
            fits = abs(long - BOX_SIZE[0]) < .04 and abs(short - BOX_SIZE[1]) < .04
            if not fits and not cut:
                continue
            thing = dict(point=centre, axis=angle, long=long, short=short, points=int(member.sum()), cut=bool(cut or not fits), top=top)
            if best is None or thing['points'] > best['points']:
                best = thing
        if best is not None:
            best['xy'] = world[:2, :2] @ best['point'] + world[:2, 3]; best['yaw'] = float(wrap(best['axis'] + base_pose[2]))
        return best

    # ---- what is on the coffee table ---------------------------------------------------------
    def table_things(self, base_pose, margin=.35):
        """What stands and lies on the coffee table as the head camera shows it: dict with
          camera: dict(point (base frame xyz: the middle of the hump on its top), closing (rad, base frame: the way the
                  fingers close on that hump = along the body's long side), xy (map), points, cut) or None
          racket: dict(point (base frame xy: where the hand takes the handle), axis (rad, base frame: from the
                  handle's end to the head), end (base frame xy of the handle's end), xy (map), points, cut) or None
        The table is in the map; things on it are the points above its top, over it or up to @margin beside it
        (a racket's handle can stick out over the edge) where the floor is empty.
        """
        points = self.points('head', stride=2); height = points[:, 2] + BASE_Z - TABLE_TOP
        points = points[(height > RACKET_BAND[0]) & (height < CAMERA_TOP + .03)]
        points = self.clear_of_robot(points); out = dict(camera=None, racket=None)
        if len(points) < 20:
            return out
        world = pose2(*base_pose); xy = points[:, :2] @ world[:2, :2].T + world[:2, 3]; lo, hi = self.table
        over = np.all(xy > lo, axis=1) & np.all(xy < hi, axis=1)
        i, j = self.scene.cell(xy); ic, jc = np.clip(i, 0, self.scene.shape[0] - 1), np.clip(j, 0, self.scene.shape[1] - 1)
        beside = np.all(xy > lo - margin, axis=1) & np.all(xy < hi + margin, axis=1) & ~over & (self.scene.top[ic, jc] < .03) & self.scene.inside(i, j)
        points = points[over | beside]; h = points[:, 2] + BASE_Z - TABLE_TOP
        # The camera: what stands higher than a racket lies.
        tall = points[h > .075]
        if len(tall) >= 20:
            keys = np.floor(tall[:, :2] / CELL).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool)
            grid[tuple((keys - origin + 1).T)] = True
            labels, count = label(grid, structure=np.ones((3, 3))); tag = labels[tuple((keys - origin + 1).T)]
            sizes = [(int((tag == index).sum()), index) for index in range(1, count + 1)]
            n, index = max(sizes); body = tall[tag == index]; top = float(np.percentile(body[:, 2], 99)) + BASE_Z - TABLE_TOP
            if n >= 20 and abs(top - CAMERA_TOP) < .02:
                # Its body from above is a rectangle 0.198 x 0.113: the long side gives the way the fingers close.
                near = points[(np.linalg.norm(points[:, :2] - body[:, :2].mean(0), axis=1) < .16) & (h > .03)]
                centre, angle, long, short = rectangle(near[:, :2])
                hump = body[body[:, 2] + BASE_Z - TABLE_TOP > top - HUMP_BAND]
                along = np.array([np.cos(angle), np.sin(angle)]); across = np.array([-along[1], along[0]])
                a = hump[:, :2] @ along; b = hump[:, :2] @ across
                middle = along * (np.percentile(a, 2) + np.percentile(a, 98)) / 2 + across * (np.percentile(b, 2) + np.percentile(b, 98)) / 2
                # The hump's top is 6.5 x 5.5 cm: where the highest points cover more, the body's top is among them
                # (from 1.2 m and more, in one view of ten) and their middle is not the hump's.
                spread = (float(np.ptp(a)), float(np.ptp(b)))
                out['camera'] = dict(point=np.r_[middle, TABLE_TOP + top - BASE_Z], closing=angle, body=centre, long=long, short=short, hump=spread,
                                     points=n, cut=bool(self.at_border(near) or spread[0] > .085 or spread[1] > .08), xy=world[:2, :2] @ middle + world[:2, 3])
        # The racket: what lies flat, apart from the camera's foot.
        flat = points[(h > RACKET_BAND[0]) & (h < RACKET_BAND[1])]
        if out['camera'] is not None:
            flat = flat[np.linalg.norm(flat[:, :2] - out['camera']['body'], axis=1) > .5 * np.hypot(*CAMERA_BODY) + .03]
        if len(flat) >= 30:
            keys = np.floor(flat[:, :2] / CELL).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool)
            grid[tuple((keys - origin + 1).T)] = True
            labels, count = label(grid, structure=np.ones((5, 5))) if False else label(grid, structure=np.ones((3, 3))); tag = labels[tuple((keys - origin + 1).T)]
            n, index = max((int((tag == index).sum()), index) for index in range(1, count + 1)); racket = flat[tag == index][:, :2]
            if n >= 30:
                centred = racket - racket.mean(0); direction = np.linalg.svd(centred, full_matrices=False)[2][0]
                along = centred @ direction; across = centred @ np.array([-direction[1], direction[0]]); length = float(np.ptp(along))
                if .45 < length < RACKET_LENGTH + .06:
                    whole = length > RACKET_LENGTH - .07
                    # The handle is the narrow end: compare the width of the two ends.
                    ends = []
                    for sign in (1., -1.):
                        tip = (sign * along).max(); part = across[(sign * along) > tip - .18]
                        ends.append((float(np.ptp(part)) if len(part) >= 4 else 9., sign, tip))
                    width, sign, tip = min(ends); toward_head = -sign * direction; part = (sign * along) > tip - .22
                    middle_across = float(np.median(across[part])); end = racket.mean(0) + direction * sign * tip + np.array([-direction[1], direction[0]]) * middle_across
                    # Only the whole racket tells where its handle is: seen in part (cut by the image, or its handle hidden
                    # behind the camera) the narrow end found is not the handle's.
                    out['racket'] = dict(point=end + toward_head * HANDLE_HOLD, axis=float(np.arctan2(toward_head[1], toward_head[0])), end=end, length=length, handle_width=width,
                                         points=n, cut=bool(self.at_border(flat[tag == index]) or not whole or width > HANDLE_WIDTH + .02), xy=world[:2, :2] @ (end + toward_head * HANDLE_HOLD) + world[:2, 3])
        return out

    # ---- the trunk lid ---------------------------------------------------------------------------
    def lid_edge(self, band=(1.10, 1.21), ahead=(.25, .95), half_width=.42, slice_width=.04):
        """The rear edge of the car's trunk lid as the head camera shows it from the stance behind the car: for
        slices across (base frame y), the nearest x of the lid's top surface: array of (y, x), or None.
        The top of the closed lid lies 1.12-1.17 m above the floor; nothing else behind the car does."""
        points = self.points('head', stride=2); z = points[:, 2] + BASE_Z
        points = points[(z > band[0]) & (z < band[1]) & (points[:, 0] > ahead[0]) & (points[:, 0] < ahead[1]) & (np.abs(points[:, 1]) < half_width)]
        points = self.clear_of_robot(points); rows = []
        for y in np.arange(-half_width, half_width, slice_width):
            part = points[(points[:, 1] >= y) & (points[:, 1] < y + slice_width)]
            if len(part) >= 12:
                rows.append((y + slice_width / 2, float(np.percentile(part[:, 0], 2))))
        return np.array(rows) if len(rows) >= 6 else None

    def lid_pose(self, limit=.004):
        """The base pose (x, y, yaw in M) from the trunk lid's rear edge, or None if the closed lid does not show.

        Seen from above the edge is a V with a rounded point, 0.12 m back per metre to either side (LID_SLOPE,
        LID_ROUND), its point at LID_APEX in the map; the car does not move. Where the V's point lies in the base
        frame, and how it is turned, give the pose: on frames with true poses to 0.2 mm along the car, 7 mm
        across and 0.1 degree, from the stance and from 0.6 m further back alike (dev/test_detectors.py).
        Returns (pose, rms of the fit in m, number of samples)."""
        edge = self.lid_edge(ahead=(.25, 1.5))
        if edge is None or len(edge) < 12:
            return None
        best = None
        for y0 in np.arange(-.25, .2501, .0025):
            d = edge[:, 0] - y0; shape = LID_SLOPE * (np.sqrt(d * d + LID_ROUND ** 2) - LID_ROUND); A = np.c_[np.ones(len(d)), d]
            a, m = np.linalg.lstsq(A, edge[:, 1] - shape, rcond=None)[0]; rms = float(np.sqrt(np.mean((A @ [a, m] - (edge[:, 1] - shape)) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, a, y0, m)
        rms, a, y0, m = best
        if rms > limit or abs(y0) > .24:
            return None
        c, s = np.cos(m), np.sin(m)
        return np.array([LID_APEX[0] - (c * a - s * y0), LID_APEX[1] - (s * a + c * y0), m]), rms, len(edge)

    def bumper_shift(self, pose, band=(.50, .68), ahead=(.45, 1.8), half_width=.66, slice_width=.02, limit=.004):
        """How far the map pose is off, by the car's bumper as the head camera shows it from behind the car (lid
        open or shut): (dx, dy) to add to @pose = (x, y, yaw), the rms of the fit (m) and the number of samples; or
        None if the bumper does not show. The heading is taken as it is.

        The bumper's rear face is sampled in slices across and compared with its profile in the map (BUMPER): a
        rounded front with a bend at each end, which fixes the place along the car to a millimetre and across it
        to a centimetre. It shows from half a metre away and more; nearer, it is below the image."""
        points = self.points('head', stride=1); z = points[:, 2] + BASE_Z
        points = points[(z > band[0]) & (z < band[1]) & (points[:, 0] > ahead[0]) & (points[:, 0] < ahead[1]) & (np.abs(points[:, 1]) < half_width)]
        points = self.clear_of_robot(points); rows = []
        for y in np.arange(-half_width, half_width, slice_width):
            part = points[(points[:, 1] >= y) & (points[:, 1] < y + slice_width)]
            if len(part) >= 8:
                rows.append((y + slice_width / 2, float(np.percentile(part[:, 0], 2))))
        if len(rows) < 20:
            return None
        rows = np.array(rows); c, s = np.cos(pose[2]), np.sin(pose[2])
        seen_x = pose[0] + c * rows[:, 1] - s * rows[:, 0]; seen_y = pose[1] + s * rows[:, 1] + c * rows[:, 0]
        table = np.array(BUMPER); best = None
        for dy in np.arange(-.20, .2001, .005):
            inside = (seen_y + dy > table[0, 0]) & (seen_y + dy < table[-1, 0])
            if inside.sum() < 20:
                continue
            gap = np.interp(seen_y[inside] + dy, table[:, 0], table[:, 1]) - seen_x[inside]; dx = float(np.median(gap)); rms = float(np.sqrt(np.mean((gap - dx) ** 2)))
            if best is None or rms < best[2]:
                best = (dx, float(dy), rms, int(inside.sum()))
        return best if best is not None and best[2] < limit else None

    # ---- the last look before a pinch ---------------------------------------------------------------------
    def hand_view(self, arm, kind, tip, closing):
        """Where the thing the hand hangs above is, as the wrist camera of @arm shows it from there: base-frame xy
        for the fingers' ends, or None. kind: 'camera' (the hump on its top: both coordinates), 'racket' (its
        handle) or 'box' (the wall under the hand): for these two, the coordinate along the way the fingers close;
        the other one is kept from @tip. tip: the fingers' ends now (base frame); closing: the way they close (rad).

        From 10 cm above, the wrist camera looks straight down at the thing between the fingers. On frames taken
        with the hand put up to 3 cm off (dev/jobs/wrist_trial.py) the handle's middle comes out to 1 mm, the
        wall's to 2.5 mm and the hump's to 3 mm along the fingers' way and 6 mm across it."""
        points = camera_points(self.data[WRIST[arm] + 'depth_linear'], self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[arm]], FOCAL[arm], stride=2, max_range=1.)
        if len(points) < 100:
            return None
        tip = np.asarray(tip, float); along = np.array([np.cos(closing), np.sin(closing)]); across = np.array([-along[1], along[0]])
        z = points[:, 2] + BASE_Z; offset = points[:, :2] - tip[:2]; reach = np.linalg.norm(offset, axis=1)
        if kind == 'camera':
            body = points[(reach < .14) & (z > TABLE_TOP + .075) & (z < TABLE_TOP + .20)]
            if len(body) < 100:
                return None
            hump = body[body[:, 2] > np.percentile(body[:, 2], 99.5) - HUMP_BAND]
            a = hump[:, :2] @ along; b = hump[:, :2] @ across
            if np.ptp(a) > .085 or np.ptp(b) > .08:
                return None
            return along * (np.percentile(a, 2) + np.percentile(a, 98)) / 2 + across * (np.percentile(b, 2) + np.percentile(b, 98)) / 2
        if kind == 'racket':
            part = points[(reach < .09) & (z > TABLE_TOP + .012) & (z < TABLE_TOP + .06)]; widths = (.02, .06)
        else:
            part = points[(np.abs(offset @ across) < .07) & (np.abs(offset @ along) < .07) & (np.abs(z - BOX_SIZE[2]) < .012)]; widths = (.008, .045)
        if len(part) < 100:
            return None
        a = part[:, :2] @ along; lo, hi = np.percentile(a, 2), np.percentile(a, 98)
        if not widths[0] < hi - lo < widths[1]:
            return None
        return along * (lo + hi) / 2 + across * float(tip[:2] @ across)
