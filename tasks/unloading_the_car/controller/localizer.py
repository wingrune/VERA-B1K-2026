"""Robot localisation in the prior map from the head depth camera. (Written for the office of the fax and battery
tasks, used unchanged in the house: the constants below were measured there on the same robot.)

A horizontal "scan" is cut out of each depth image (points between knee and desk height, robot
and carried object removed) and matched against the map's likelihood field. The start pose is
unknown, so the first scans are scored over the whole free space; afterwards the pose is tracked
with commanded base motion as the prediction and a local scan-match as the correction.

Why not integrate proprioceptive base velocity: its yaw-rate channel carries a bias of up to
0.14 rad/s (measured), while the base follows velocity commands closely in free motion.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

SCAN_BAND = (.25, .90)          # heights above the floor used for matching (below the glass panels)
UPPER_BAND = (1.40, 1.85)       # second band, above the cubicle partitions: walls, windows, tall cabinets
BASE_Z = .012
LINEAR_GAIN = .75; ANGULAR_GAIN = .95        # measured here: 0.944 at half the command range; the fax runs used .97
# Measured in free motion (dev/jobs/odom_calib.py): turning on the spot the base is 0.03 rad/s slower than
# commanded, at any rate (0.90 of the command at 0.3, 0.95 at 0.6, 0.96 at 0.9); turning while it drives, 0.01
# slower (0.97 at 0.3). Its own report of its translation reads 5 mm/s too fast at any speed (6% at 0.11 m/s,
# 1.4% at 0.37 m/s) and half a degree to the left of the true direction.
TURN_DEFICIT = .03; TURN_DEFICIT_DRIVING = .009; SPEED_EXCESS = .005; REPORT_ANGLE = np.radians(.6)


def turn_rate(command, driving=False):
    """True yaw rate (rad/s) for a yaw-rate command, on the spot or while @driving. Measured from 0.3 up; below
    0.1 the deficit is taken to shrink in proportion (a dead band there is not established and would freeze
    the dead-reckoned heading)."""
    size = abs(float(command)); deficit = TURN_DEFICIT_DRIVING if driving else TURN_DEFICIT
    return float(np.sign(command) * (size - deficit if size >= .1 else size * (1. - deficit / .1)))


# The translation command is followed closely in free motion, in every direction: the base moves at 0.75 m/s per
# unit of command less a fixed 6.4 mm/s forwards or backwards and 2.3 mm/s sideways (41 legs at 0.11-0.37 m/s,
# direction within 1.2 deg). Its own report is good to a few per cent forwards but not sideways: there it read
# between 36% short and 12% long depending on what the base had done before.
COMMAND_GAIN = .75; COMMAND_DEFICIT = (.0064, .0023)


def command_velocity(command):
    """The base's translation (m/s, base frame) in free motion for a translation command."""
    c = np.asarray(command, float)[:2]; size = float(np.linalg.norm(c))
    if size < 1e-6:
        return np.zeros(2)
    unit = c / size; deficit = COMMAND_DEFICIT[0] * unit[0] ** 2 + COMMAND_DEFICIT[1] * unit[1] ** 2
    return unit * max(0., COMMAND_GAIN * size - deficit)


def true_velocity(report):
    """The base's translation (m/s, base frame) from its own report of it."""
    v = np.asarray(report, float)[:2]; speed = float(np.linalg.norm(v))
    if speed < 1e-6:
        return np.zeros(2)
    c, s = np.cos(REPORT_ANGLE), np.sin(REPORT_ANGLE)
    true = speed - SPEED_EXCESS if speed >= .05 else speed * (1. - SPEED_EXCESS / .05)
    return true / speed * np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])


def camera_points(depth, cam_rel, focal_ratio, stride=6, max_range=12.):
    """3-D points in the base frame from a z-depth image and the camera pose relative to the base."""
    x, y, z, w = cam_rel[3:]
    rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])
    h, wd = depth.shape; f = focal_ratio * wd
    v, u = np.mgrid[stride // 2:h:stride, stride // 2:wd:stride]; d = depth[v, u].astype(np.float32)
    ok = np.isfinite(d) & (d > .15) & (d < max_range)
    pc = np.stack([(u + .5 - wd / 2) / f * d, (v + .5 - h / 2) / f * d, d], -1)[ok]
    return pc @ rot.T + cam_rel[:3]


class Localizer:
    def __init__(self, scene, sigma=.08):
        self.scene = scene; self.res = scene.res; self.sigma = sigma
        # Field 0 belongs to the lower band, field 1 to the upper band. The office is a grid of
        # near-identical cubicles, so the lower band alone cannot tell the aisles apart; the outer
        # walls seen over the partitions can.
        self.likelihood = []; self.coarse = []; self.likelihood_all = []; self.coarse_all = []; self.walls = scene.upper & ~maximum_filter(~scene.upper, size=5)   # wall interiors
        low = scene.block if getattr(scene, 'sight_low', None) is None else scene.sight_low
        upper = scene.upper if getattr(scene, 'sight_upper', None) is None else scene.sight_upper
        # In the house the map has thin height slabs instead of the two bands: each lies between the heights
        # where furniture ends, so that what is in a slab's layer is what a ray at that height hits. The last
        # slab is the upper band.
        self.bands = list(getattr(scene, 'sight_slabs', None) or []) or [SCAN_BAND, UPPER_BAND]
        # Things whose shape in the map (collision hulls) is not what the camera sees (a car's rounded body, its
        # trunk lid open or shut) are no landmarks for the fix: they are taken out of the fields here, and what is
        # seen where they stand is left out of the match (correct()). With the car in, a start beside it was fixed
        # 9 cm off. For the global search they stay in (likelihood_all, coarse_all): a start beside the car has the
        # car in half of its view, and with those points counted as misses a look-alike pose at the other end of
        # the garage scored higher than the true one. So does the leaf of the door the robot is going to push open:
        # at the start it is shut, and a start next to it has it across a quarter of its view.
        door = getattr(scene, 'door', None); shut = door is not None and bool(door.any())
        sight = getattr(scene, 'top_sight', scene.top)
        self.top_start = np.where(door, np.maximum(sight, 2.), sight) if shut else sight      # the height field as it is at the start
        car = (scene.meta.get('scope') or {}).get('car.n.01_1')
        self.unlike = [] if car is None else [(car['aabb_lo'][0] - .15, car['aabb_lo'][1] - .15, car['aabb_hi'][0] + .15, car['aabb_hi'][1] + .15)]
        gx = scene.x0 + (np.arange(scene.shape[0]) + .5) * self.res; gy = scene.y0 + (np.arange(scene.shape[1]) + .5) * self.res; landmark = np.ones(scene.shape, bool)
        for x0, y0, x1, y1 in self.unlike:
            landmark &= ~(((gx > x0) & (gx < x1))[:, None] & ((gy > y0) & (gy < y1))[None, :])
        for everything in (scene.sight if getattr(scene, 'sight', None) else (low, upper)):
            for occupied, fine, coarse in ((everything & landmark, self.likelihood, self.coarse), (everything | door if shut else everything, self.likelihood_all, self.coarse_all)):
                if not self.unlike and not shut and fine is self.likelihood_all:
                    fine.append(self.likelihood[-1]); coarse.append(self.coarse[-1]); continue
                # Match against faces only. Desks and cabinets are filled areas in the map; with those
                # as targets any scan pushed into a cubicle would score perfectly.
                faces = occupied & maximum_filter(~occupied, size=3)
                field = (distance_transform_edt(~faces) * self.res).astype(np.float32)
                fine.append(np.exp(-.5 * (np.minimum(field, .6) / sigma) ** 2).astype(np.float32))
                # A smoother field for the global search, where hypotheses are coarse.
                coarse.append(np.exp(-.5 * (np.minimum(field, .9) / .22) ** 2).astype(np.float32))
        free = scene.distance > scene.base_radius - .05
        start = scene.meta.get('start_floor')
        if start is not None:
            # The task definition starts the robot on one floor (the copy room), so the unknown start
            # pose is searched for there and not in the rooms around it.
            gx = scene.x0 + (np.arange(scene.shape[0]) + .5) * self.res; gy = scene.y0 + (np.arange(scene.shape[1]) + .5) * self.res
            free = free & ((gx > start['lo'][0]) & (gx < start['hi'][0]))[:, None] & ((gy > start['lo'][1]) & (gy < start['hi'][1]))[None, :]
        self.free_cells = np.argwhere(free[::6, ::6]) * 6 + 3
        self.pose = None; self.misses = 0; self.score = 0.; self.debug = {}
        self.frozen = False                         # no corrections while set
        self.lock = None                            # unit vector in M along which corrections are not applied
        self.swing = scene.meta.get('door_swing')   # where a door leaf moves: what is seen there is not in the map
        self.clear = scene.distance_static          # chairs can be pushed aside, fixed furniture cannot
        self.travel = 0.; self.turn = 0.            # commanded motion since the last accepted correction
        self.rivals = []            # [pose, evidence] of other start hypotheses still being followed
        self.eye = (.24, 0., 1.35)  # the camera in the base frame, its height above the floor (set by the observer)
        # What a start hypothesis could not have seen, and its weight against the agreement with the map: in the
        # house, lines of sight against the map's height field in every band; in the office (two bands), as before,
        # the upper band against the walls.
        self.unseen = self.blocked_rays if getattr(scene, 'sight_slabs', None) else self.blocked_fraction; self.unseen_weight = .6
        self.evidence = 0.

    def standable(self, xy):
        i = np.floor((xy[:, 0] - self.scene.x0) / self.res).astype(int); j = np.floor((xy[:, 1] - self.scene.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.clear.shape[0]) & (j >= 0) & (j < self.clear.shape[1])
        return ok & (self.clear[np.clip(i, 0, self.clear.shape[0] - 1), np.clip(j, 0, self.clear.shape[1] - 1)] > self.scene.base_radius - .10)

    # ---- scan ----------------------------------------------------------------------------
    def scan(self, points, exclude=None):
        """Two 2-D scans (base frame), lower and upper band, from base-frame 3-D points.

        exclude = (centres, radii) of robot (and payload) spheres to cut out of the lower band.
        """
        z = points[:, 2] + BASE_Z; out = []
        for index, (lo, hi) in enumerate(self.bands):
            pts = points[(z > lo) & (z < hi)]
            if index < len(self.bands) - 1 and exclude is not None and len(pts):
                centres, radii = exclude
                near = (np.linalg.norm(pts[:, None, :] - centres[None], axis=2) < radii[None] + .05).any(axis=1)
                pts = pts[~near]
            if not len(pts):
                out.append(np.zeros((0, 2))); continue
            # Keep upright surfaces only: cells whose points span some height. Desk tops and the
            # like are horizontal and would not line up with the faces in the map.
            keys, inverse = np.unique(np.floor(pts[:, :2] / .04).astype(np.int64), axis=0, return_inverse=True)
            inverse = inverse.ravel(); low = np.full(len(keys), np.inf); high = np.full(len(keys), -np.inf)
            np.minimum.at(low, inverse, pts[:, 2]); np.maximum.at(high, inverse, pts[:, 2])
            out.append((keys[high - low > min(.18, .6 * (hi - lo))] + .5) * .04)
        return out

    @staticmethod
    def size(scan):
        return sum(len(band) for band in scan)

    def _score_one(self, field, scan, poses):
        """Per pose: (sum of field values over scan points that land inside the map, their count)."""
        c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
        gx = poses[:, 0, None] + c * scan[None, :, 0] - s * scan[None, :, 1]
        gy = poses[:, 1, None] + s * scan[None, :, 0] + c * scan[None, :, 1]
        i = np.floor((gx - self.scene.x0) / self.res).astype(np.int32); j = np.floor((gy - self.scene.y0) / self.res).astype(np.int32)
        ok = (i >= 0) & (i < field.shape[0]) & (j >= 0) & (j < field.shape[1])
        return np.where(ok, field[np.clip(i, 0, field.shape[0] - 1), np.clip(j, 0, field.shape[1] - 1)], 0.).sum(axis=1), ok.sum(axis=1)

    def _score(self, fields, scan, poses, limit=None, rng=None):
        """Agreement of @scan with the map under each pose (K,3) -> (K,), in 0..1.

        The two bands count equally however many points each has. The lower band is dense but
        repeats from aisle to aisle; the upper band (outer walls) is sparse and is what tells the
        aisles apart, so it must not be outvoted.

        Points that land outside the map say nothing: the map ends somewhere and the camera sees
        further. Counting them as misses made a true pose near the map's edge look worse than a
        look-alike in the middle of it. A band that is mostly off the map is not used.
        """
        total = np.zeros(len(poses), np.float32); weight = np.zeros(len(poses), np.float32)
        for field, band in zip(fields, scan):
            if len(band) < 6:
                continue
            if limit is not None and len(band) > limit:
                band = band[(rng or np.random.default_rng(1)).choice(len(band), limit, replace=False)]
            value, inside = self._score_one(field, band, poses)
            usable = inside >= max(6, .5 * len(band))
            total += np.where(usable, value / np.maximum(inside, 1), 0.); weight += usable
        return total / np.maximum(weight, 1.)

    # ---- global search -------------------------------------------------------------------
    def global_search(self, scans, keep=400, yaw_step=np.radians(6)):
        """Scores every free position and heading against one or more scans.

        scans: list of (scan_xy, relative_pose) where relative_pose = (dx, dy, dyaw) of the base at
        that scan with respect to the base at the first scan (from commanded motion).
        Returns the best poses (of the first scan's base) with scores, best first.
        """
        rng = np.random.default_rng(0)
        cells = self.free_cells; xy = np.c_[self.scene.x0 + cells[:, 0] * self.res, self.scene.y0 + cells[:, 1] * self.res]
        yaws = np.arange(-np.pi, np.pi, yaw_step); total = np.zeros((len(xy), len(yaws)), np.float32)
        for scan, (dx, dy, dyaw) in scans:
            if self.size(scan) < 25:
                continue
            sub = [band[rng.choice(len(band), min(len(band), 60), replace=False)] if len(band) else band for band in scan]
            for k, yaw in enumerate(yaws):
                c, s = np.cos(yaw), np.sin(yaw)
                poses = np.c_[xy[:, 0] + c * dx - s * dy, xy[:, 1] + s * dx + c * dy, np.full(len(xy), yaw + dyaw)]
                total[:, k] += self._score(self.coarse_all, sub, poses)
        flat = np.argsort(total.ravel())[::-1][:keep]; a, b = np.unravel_index(flat, total.shape)
        return np.c_[xy[a], yaws[b]], total.ravel()[flat]

    def refine(self, scan, pose, span=(.16, .16, np.radians(8)), steps=(9, 9, 9), field=None):
        """Best pose on a small grid around @pose, coarse-to-fine. Returns (pose, score)."""
        field = self.likelihood if field is None else field; best = np.asarray(pose, float); score = -1.
        for shrink in (1., .35, .12):
            gx = best[0] + np.linspace(-1, 1, steps[0]) * span[0] * shrink
            gy = best[1] + np.linspace(-1, 1, steps[1]) * span[1] * shrink
            gt = best[2] + np.linspace(-1, 1, steps[2]) * span[2] * shrink
            grid = np.stack(np.meshgrid(gx, gy, gt, indexing='ij'), -1).reshape(-1, 3)
            values = self._score(field, scan, grid, limit=300)
            # The base cannot be inside furniture. Without this a scan of two parallel walls slides
            # sideways onto the next pair of parallel faces (window wall onto cubicle partition).
            values = np.where(self.standable(grid[:, :2]), values, -1.)
            k = int(np.argmax(values)); best = grid[k]; score = float(values[k])
        best[2] = (best[2] + np.pi) % (2 * np.pi) - np.pi
        return best, score

    def blocked_fraction(self, scan, pose, origin=(.24, 0.)):
        """Share of upper-band scan points that the map says could not have been seen from @pose:
        the straight line from the camera to the point runs through a wall. (Rays of the upper band
        are almost level, at head height, so only walls and tall cabinets can stop them.)"""
        band = scan[-1]
        if len(band) < 12:
            return 0.
        c, s = np.cos(pose[2]), np.sin(pose[2]); rot = np.array([[c, -s], [s, c]])
        start = np.asarray(pose[:2]) + rot @ np.asarray(origin); ends = pose[:2] + band @ rot.T
        length = np.linalg.norm(ends - start, axis=1); count = 24
        fraction = np.linspace(.05, 1., count)[None, :] * np.maximum(length - .30, 0.)[:, None] / np.maximum(length, 1e-6)[:, None]
        px = start[0] + fraction * (ends[:, 0] - start[0])[:, None]; py = start[1] + fraction * (ends[:, 1] - start[1])[:, None]
        i = np.floor((px - self.scene.x0) / self.res).astype(int); j = np.floor((py - self.scene.y0) / self.res).astype(int)
        ok = (i >= 0) & (i < self.walls.shape[0]) & (j >= 0) & (j < self.walls.shape[1])
        hit = np.where(ok, self.walls[np.clip(i, 0, self.walls.shape[0] - 1), np.clip(j, 0, self.walls.shape[1] - 1)], False)
        return float(hit.any(axis=1).mean())

    def blocked_rays(self, scan, pose, slack=.12):
        """Share of the scan points (all bands) that the map says could not have been seen from @pose: the straight
        line from the camera to the point runs below the top of something on its way (the map is a height field:
        what stands on the floor is taken as solid up to its top). A point's height is that of the middle of its
        band; the last 0.3 m before it are not tested, and the map's tops may be @slack higher than the line.
        This is what tells the garage's look-alike corners apart: from the corner behind the car, most of what a
        start in the opposite corner sees would lie behind the car."""
        c, s = np.cos(pose[2]), np.sin(pose[2]); rot = np.array([[c, -s], [s, c]]); start = np.asarray(pose[:2], float) + rot @ np.asarray(self.eye[:2], float)
        top = self.top_start; hits = 0; count = 0
        for band, (lo, hi) in zip(scan, self.bands):
            if len(band) < 6:
                continue
            ends = np.asarray(pose[:2], float) + band @ rot.T; length = np.linalg.norm(ends - start, axis=1)
            fraction = np.linspace(.05, 1., 24)[None, :] * np.maximum(length - .30, 0.)[:, None] / np.maximum(length, 1e-6)[:, None]
            px = start[0] + fraction * (ends[:, 0] - start[0])[:, None]; py = start[1] + fraction * (ends[:, 1] - start[1])[:, None]
            height = self.eye[2] + fraction * (.5 * (lo + hi) - self.eye[2])
            i = np.floor((px - self.scene.x0) / self.res).astype(int); j = np.floor((py - self.scene.y0) / self.res).astype(int)
            ok = (i >= 0) & (i < top.shape[0]) & (j >= 0) & (j < top.shape[1])
            above = np.where(ok, top[np.clip(i, 0, top.shape[0] - 1), np.clip(j, 0, top.shape[1] - 1)], 0.) > height + slack
            hits += int(above.any(axis=1).sum()); count += len(band)
        return hits / count if count else 0.

    def localize(self, scans, shortlist=14):
        """Global localisation from scans taken at one spot.

        scans: list of (scan, yaw of the base relative to the first scan). Returns
        (pose at the first scan, score, margin to the best clearly different hypothesis).
        """
        usable = [(scan, (0., 0., dyaw)) for scan, dyaw in scans if self.size(scan) >= 30]
        if not usable:
            return None, 0., 0.
        candidates, _ = self.global_search(usable)
        results = []; seen = []
        for cand in candidates:
            if any(np.linalg.norm(cand[:2] - o[:2]) < .30 and abs((cand[2] - o[2] + np.pi) % (2 * np.pi) - np.pi) < .30 for o in seen):
                continue
            seen.append(cand)
            pose, score = self.refine(usable[0][0], cand, span=(.16, .16, np.radians(8)), field=self.likelihood_all)
            total = score - self.unseen_weight * self.unseen(usable[0][0], pose)
            for scan, (_, _, dyaw) in usable[1:]:
                # The commanded turn between scans is only approximately known: let each scan settle.
                turned, extra = self.refine(scan, np.array([pose[0], pose[1], pose[2] + dyaw]), span=(.04, .04, np.radians(8)), steps=(3, 3, 9), field=self.likelihood_all)
                total += extra - self.unseen_weight * self.unseen(scan, turned)
            results.append((total / len(usable), pose))
            if len(seen) >= shortlist:
                break
        results.sort(key=lambda item: -item[0]); best_score, best = results[0]
        others = [(sc, p) for sc, p in results[1:] if np.linalg.norm(p[:2] - best[:2]) > .5 or abs((p[2] - best[2] + np.pi) % (2 * np.pi) - np.pi) > .5]
        rival = max([sc for sc, _ in others] or [0.])
        self.shortlist = [(sc, p) for sc, p in others if sc > best_score - .15][:4]
        return best, float(best_score), float(best_score - rival)

    def start_tracking(self, pose, rivals=()):
        """Begin tracking from @pose; @rivals are other plausible start poses (the office repeats every
        four metres) that are followed in parallel until the walls further along tell them apart."""
        self.pose = np.asarray(pose, float).copy(); self.misses = 0; self.evidence = 0.
        self.rivals = [[np.asarray(p, float).copy(), 0.] for p in rivals]

    # ---- tracking ------------------------------------------------------------------------
    def predict(self, command, dt=1 / 30, measured=None):
        """Dead-reckon the tracked pose with the base velocity command of the last step.

        measured: (vx, vy) of the base in its own frame as the robot reports them; used for the translation
        when given (the command is followed only approximately, sideways in particular).
        """
        if self.pose is None:
            return
        vx, vy, wz = command[0] * LINEAR_GAIN, command[1] * LINEAR_GAIN, turn_rate(command[2], driving=np.hypot(command[0], command[1]) > .06)
        if measured is not None:
            vx, vy = float(measured[0]), float(measured[1])

        def advance(pose):
            x, y, yaw = pose; c, s = np.cos(yaw + .5 * wz * dt), np.sin(yaw + .5 * wz * dt)
            return np.array([x + (c * vx - s * vy) * dt, y + (s * vx + c * vy) * dt, yaw + wz * dt])

        self.pose = advance(self.pose)
        self.travel += float(np.hypot(vx, vy)) * dt; self.turn += abs(wz) * dt
        for rival in self.rivals:
            rival[0] = advance(rival[0])

    def correct(self, scan, min_points=40):
        """Scan-match around the predicted pose. After a few poor matches the search widens, so a
        lock that slipped (a fast turn, a featureless view) is picked up again instead of drifting."""
        if self.pose is None or self.frozen:
            return False
        if self.swing is not None:
            # A door leaf is as tall as the wall: what is seen where it swings is left out of every band.
            c, s = np.cos(self.pose[2]), np.sin(self.pose[2]); kept = []
            for band in scan:
                if len(band):
                    xy = np.c_[self.pose[0] + c * band[:, 0] - s * band[:, 1], self.pose[1] + s * band[:, 0] + c * band[:, 1]]
                    band = band[np.linalg.norm(xy - self.swing['centre'], axis=1) > self.swing['radius'] + .05]
                kept.append(band)
            scan = kept
        if self.unlike:
            c, s = np.cos(self.pose[2]), np.sin(self.pose[2]); kept = []
            for band in scan:
                if len(band):
                    x = self.pose[0] + c * band[:, 0] - s * band[:, 1]; y = self.pose[1] + s * band[:, 0] + c * band[:, 1]; inside = np.zeros(len(band), bool)
                    for x0, y0, x1, y1 in self.unlike:
                        inside |= (x > x0) & (x < x1) & (y > y0) & (y < y1)
                    band = band[~inside]
                kept.append(band)
            scan = kept
        if self.size(scan) < min_points:
            return False
        # Dead reckoning over a few steps is good to a fraction of the distance covered, so the
        # match may only move the pose by that much. A view that constrains one direction poorly
        # (two parallel walls) can then no longer walk the estimate along them.
        # The base reports its own translation to about one per cent and the turn command is followed to a few
        # per cent, so a correction is allowed a small fixed step plus a tenth of the motion since the last one.
        # With more slack the fit, next to furniture that has been pushed away from where the map has it, turned the
        # estimate step by step through a quarter turn (the room looks alike every 90 degrees).
        reach = min(.06, .012 + .10 * self.travel); swing = min(np.radians(3), np.radians(.6) + .10 * self.turn)
        # A coordinate held by dead reckoning is not searched at all: the other one and the heading are then
        # fitted for the position the robot is believed to have, not for a shifted one.
        held = (0, 0) if self.lock is None else (abs(self.lock[0]) > .5, abs(self.lock[1]) > .5)
        spans = (0. if held[0] else reach, 0. if held[1] else reach, swing); steps = (1 if held[0] else 9, 1 if held[1] else 9, 9)
        pose, score = self.refine(scan, self.pose, span=spans, steps=steps)
        # Along a direction the view does not constrain (a plain wall right in front, a long corridor), every
        # shift scores the same and the fit would wander: there the dead-reckoned coordinate is kept.
        slid = []
        for axis in (0, 1):
            if held[axis] or abs(pose[axis] - self.pose[axis]) < 1e-4:
                continue
            probe = np.repeat(pose[None], 2, axis=0); probe[0, axis] -= .06; probe[1, axis] += .06
            if score - float(np.max(self._score(self.likelihood, scan, probe, limit=300))) < .02:
                pose[axis] = self.pose[axis]; slid.append('xy'[axis])
        wide = False
        if score < .5 and self.misses >= 12 and self.lock is None:
            # Coarse field first (broad basin), then the sharp one.
            rough, _ = self.refine(scan, self.pose, span=(.30, .30, np.radians(10)), steps=(11, 11, 11), field=self.coarse)
            candidate, wide_score = self.refine(scan, rough, span=(.10, .10, np.radians(5)))
            if wide_score > score + .08:
                pose, score, wide = candidate, wide_score, True
        self.score = score
        self.debug = dict(points=[len(band) for band in scan], score=round(score, 3), shift=np.round(pose - self.pose, 3).tolist(), wide=wide, misses=self.misses, unconstrained=slid)
        accepted = score > .45
        if accepted:
            self.pose = pose; self.misses = 0; self.travel = 0.; self.turn = 0.
        else:
            self.misses += 1
        if self.rivals:
            # Compare the hypotheses on what they explain and on what they could not have seen.
            self.evidence = .92 * self.evidence + score - .6 * self.blocked_fraction(scan, self.pose)
            for rival in self.rivals:
                candidate, value = self.refine(scan, rival[0], span=(.10, .10, np.radians(5)), steps=(5, 5, 7))
                if value > .45:
                    rival[0] = candidate
                rival[1] = .92 * rival[1] + value - .6 * self.blocked_fraction(scan, rival[0])
            best = max(self.rivals, key=lambda item: item[1])
            if best[1] > self.evidence + .8:
                # A rival has explained the views clearly better for a while: it was the right start.
                self.debug['switched_to'] = np.round(best[0], 3).tolist()
                best[0], self.pose = self.pose, best[0]; best[1], self.evidence = self.evidence, best[1]; self.misses = 0
            self.rivals = [r for r in self.rivals if r[1] > self.evidence - 2.5]
        return accepted
