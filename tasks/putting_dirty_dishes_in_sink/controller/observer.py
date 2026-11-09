"""State estimation for the trash task from onboard observations only.

Inputs: the head camera's depth image, the camera poses relative to the base, proprioception, plus the static
map of the lower floor of the house. No simulator state.

  * base pose in the map: Localizer (scan matching against the map)
  * soda cans and the trash can: things of their size standing on floor that the map has empty. A soda can is
    0.076 m across and 0.127-0.135 m tall, the trash can 0.23 m across and 0.279 m tall; nothing else of
    these sizes stands loose on these floors.
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
SODA_HEIGHT = (.775, .83); SODA_WIDTH = .24    # dishes on the booth's table (top 0.75 m): bowls 12-13 cm across with rims at 0.81 m, plates 17-24 cm at 0.78 m
GLASS_BAND = (.762, .86)        # heights in which dishes are looked for, over the booth's table only
SINK_BOX = (-.25, .50, -9.80, -8.40)     # x0 x1 y0 y1 (map) of the booth's table (booth_xzrpar_2)
BIN_HEIGHT = .422; BIN_WIDTH = .38          # the trash can (ashcan cjmezk, as in the restaurant)
BIN_FIT = .03                               # mean distance of the rim points from the fitted circle (m): on start 309 the trash can fitted at 0.020-0.022
FLOOR_BAND = (.015, .47)        # (the restaurant's trash can is 0.42 m high: its rim must be in the band)        # heights above the floor in which loose things are looked for (above carpet edges and door sills)
CELL = .02
MAP_CLEAR = .10                 # loose things are looked for this far from anything the map has
LYING_TOP = (9., 9.)          # (no lying cans: the pieces of glass lie flat)        # the top of a can that lies on the floor (it is 0.076 m across)


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose2(x, y, yaw, z=0.):
    t = np.eye(4); c, s = np.cos(yaw), np.sin(yaw); t[:2, :2] = [[c, -s], [s, c]]; t[:3, 3] = [x, y, z]; return t


class Observer:
    def __init__(self, folder, scene):
        folder = Path(folder); self.scene = scene; self.loc = Localizer(scene)
        self.body = Body(folder); self.collision = RobotCollision(folder / 'robot_collision_spheres.json')
        self.steps = 0; self._cache = {}; self.rejects = []     # clusters as tall as the trash can that were not taken for it (logged)

    # ---- per-step input ------------------------------------------------------------------
    def frame(self, data):
        """data: dict of numpy arrays for this step (proprio, cam_rel_poses, depth of the three cameras)."""
        self.steps += 1; self.data = data; self._cache = {}
        self.q = from_proprio(data['robot_r1::proprio'].astype(float), self.body.names)

    def spheres(self):
        if 'spheres' not in self._cache:
            self._cache['spheres'] = (self.collision.centers(self.body.fk(self.q)[0]), self.collision.radii)
        return self._cache['spheres']

    def points(self, camera='head', stride=2):
        key = (camera, stride)
        if key not in self._cache:
            prefix = HEAD if camera == 'head' else WRIST[camera]
            self._cache[key] = camera_points(self.data[prefix + 'depth_linear'], self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[camera]], FOCAL[camera], stride=stride)
        return self._cache[key]

    def head_points(self, stride=6):
        return self.points('head', stride)

    # ---- localisation --------------------------------------------------------------------
    def scan(self):
        return self.loc.scan(self.head_points(stride=3), exclude=self.spheres())

    def localize_global(self, scans):
        return self.loc.localize(scans)

    # ---- loose things on the floor ---------------------------------------------------------
    def pixels(self, points, camera='head'):
        """Image coordinates (u, v) of base-frame @points in @camera's image, and the image size."""
        pose = self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[SLOT[camera]]; x, y, z, w = pose[3:]
        rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])
        size = self.data[(HEAD if camera == 'head' else WRIST[camera]) + 'depth_linear'].shape[0]; f = FOCAL[camera] * size
        pc = (np.asarray(points, float) - pose[:3]) @ rot; depth = np.maximum(pc[:, 2], 1e-6)
        return pc[:, 0] / depth * f + size / 2, pc[:, 1] / depth * f + size / 2, size

    @staticmethod
    def circle(points, radius, start):
        """Centre of the circle of @radius that fits @points best (an arc is enough), from @start."""
        centre = np.asarray(start, float).copy()
        for _ in range(12):
            delta = points - centre; distance = np.maximum(np.linalg.norm(delta, axis=1), 1e-6)
            # Gauss-Newton on sum (|p - c| - R)^2: the residual's gradient with respect to c is -delta / distance.
            jac = -delta / distance[:, None]; residual = distance - radius
            step = np.linalg.lstsq(jac, -residual, rcond=None)[0]; centre = centre + np.clip(step, -.05, .05)
            if np.linalg.norm(step) < 1e-4:
                break
        return centre

    def floor_things(self, base_pose, camera='head', reach=4.5):
        """Things standing on floor the map has empty, seen by @camera: list of dict(kind, xy (map), point (base
        frame xy), top (m above the floor), width (m), points, cut).

        kind is 'soda', 'bin' or None (something else, or a can that lies). cut: the thing touches the border of
        the image, so its middle is not where the visible part's middle is. Points are taken between FLOOR_BAND
        heights, away from the robot's own body, on cells the map has empty and at least MAP_CLEAR from anything
        in the map, movable furniture included (chair and table legs are of a can's width; over all distributed
        starts no can stands closer than 0.22 m to furniture and the trash can no closer than 0.16 m).
        """
        key = ('things', camera, tuple(np.round(base_pose, 4)))
        if key in self._cache:
            return self._cache[key]
        points = self.points(camera, stride=2); height = points[:, 2] + BASE_Z
        keep = (((height > FLOOR_BAND[0]) & (height < FLOOR_BAND[1])) | ((height > GLASS_BAND[0]) & (height < GLASS_BAND[1]))) & (np.hypot(points[:, 0], points[:, 1]) < reach)
        points = points[keep]; out = []
        if len(points) >= 12:
            centres, radii = self.spheres()
            points = points[~(np.linalg.norm(points[:, None, :] - centres[None], axis=2) < radii[None] + .05).any(axis=1)]
        if len(points) >= 12:
            world = pose2(*base_pose); xy = points[:, :2] @ world[:2, :2].T + world[:2, 3]; z = points[:, 2] + BASE_Z
            i, j = self.scene.cell(xy); ok = self.scene.inside(i, j)
            ic, jc = np.clip(i, 0, self.scene.shape[0] - 1), np.clip(j, 0, self.scene.shape[1] - 1)
            low = z < FLOOR_BAND[1]
            on_sink = (xy[:, 0] > SINK_BOX[0]) & (xy[:, 0] < SINK_BOX[1]) & (xy[:, 1] > SINK_BOX[2]) & (xy[:, 1] < SINK_BOX[3]) & (self.scene.top[ic, jc] < z - .010)
            ok &= np.where(low, (self.scene.top[ic, jc] < .03) & (self.scene.distance[ic, jc] > MAP_CLEAR), on_sink)
            xy, z, local = xy[ok], z[ok], points[ok]
            if len(xy) >= 12:
                u, v, size = self.pixels(local, camera); edge = (u < 8) | (u > size - 8) | (v > size - 8)
                keys = np.floor(xy / CELL).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool)
                grid[tuple((keys - origin + 1).T)] = True
                labels, count = label(grid, structure=np.ones((3, 3))); tag = labels[tuple((keys - origin + 1).T)]
                bins = []
                for index in range(1, count + 1):
                    member = tag == index; n = int(member.sum())
                    if n < 12:
                        continue
                    p, zz = xy[member], z[member]; top = float(np.percentile(zz, 98)); extent = p.max(0) - p.min(0); width = float(extent.max())
                    seen = local[member, :2]; middle = (seen.min(0) + seen.max(0)) / 2; cut = bool(edge[member].any())
                    if BIN_HEIGHT - .04 < top < BIN_HEIGHT + .03 and width < BIN_WIDTH + .07:
                        # The trash can: its rim is a circle 0.23 m across. From far off the near wall and the far
                        # inner wall are separate patches, so the circle is fitted to all rim points around here.
                        around = (np.linalg.norm(local[:, :2] - middle, axis=1) < BIN_WIDTH + .03) & (z > BIN_HEIGHT - .045) & (z < BIN_HEIGHT + .03)
                        rim = local[around, :2]
                        if len(rim) >= 12:
                            away = middle / max(float(np.linalg.norm(middle)), 1e-9)
                            centre = self.circle(rim, BIN_WIDTH / 2, (rim.min(0) + rim.max(0)) / 2 if len(rim) >= 60 else middle + away * BIN_WIDTH / 4)
                            fit = float(np.mean(np.abs(np.linalg.norm(rim - centre, axis=1) - BIN_WIDTH / 2)))
                            if fit < BIN_FIT:
                                bins.append(dict(kind='bin', point=centre, top=top, width=width, points=int(around.sum()), cut=bool(edge[around].any()), fit=round(fit, 4)))
                                continue
                            self.rejects = (self.rejects + [dict(xy=np.round(p.mean(0), 2).tolist(), top=round(top, 3), width=round(width, 3), n=n, rim=len(rim), fit=round(fit, 4))])[-12:]
                    elif .30 < top < .55:
                        self.rejects = (self.rejects + [dict(xy=np.round(p.mean(0), 2).tolist(), top=round(top, 3), width=round(width, 3), n=n)])[-12:]
                    if SODA_HEIGHT[0] - .025 < top < SODA_HEIGHT[1] + .02 and .03 < width < 2 * SODA_WIDTH + .10:
                        # A can, or two that stand together (nine training starts have a pair closer than 0.16 m).
                        # The head camera looks down on a can: the points of its top lie around its axis.
                        lid = seen[zz > top - .015]; parts = [(seen, lid)]
                        if width >= SODA_WIDTH + .05:
                            centred = lid - lid.mean(0); along = centred @ np.linalg.svd(centred, full_matrices=False)[2][0]
                            split = (along.min() + along.max()) / 2; halves = [lid[along < split], lid[along >= split]]
                            if not all(len(h) >= 5 and float(np.ptp(h, axis=0).max()) < SODA_WIDTH + .03 for h in halves):
                                out.append(dict(kind=None, point=np.asarray(middle, float), top=top, width=width, points=n, cut=cut)); continue
                            parts = [(h, h) for h in halves]
                        for body, lid in parts:
                            if len(lid) >= 6:
                                centre = (lid.min(0) + lid.max(0)) / 2 if len(lid) >= 25 else lid.mean(0)
                            else:
                                # Too far off for its top to show: the near side is seen, the axis lies half the
                                # can's width behind the middle of that, along the line of sight.
                                centre = (body.min(0) + body.max(0)) / 2; depth = float(np.linalg.norm(body.max(0) - body.min(0)))
                                centre = centre + centre / max(float(np.linalg.norm(centre)), 1e-9) * max(0., SODA_WIDTH / 2 - depth / 2) * .8
                            out.append(dict(kind='soda', point=np.asarray(centre, float), top=top, width=float(np.ptp(body, axis=0).max()), points=len(body), cut=cut, pair=len(parts) > 1))
                        continue
                    if LYING_TOP[0] < top < LYING_TOP[1] and n >= 25:
                        # A can that has fallen over: from above its points cover a strip as wide as a can and up to
                        # as long as one. Its axis is the long direction of that strip.
                        centred = seen - seen.mean(0); directions = np.linalg.svd(centred, full_matrices=False)[2]
                        along = centred @ directions[0]; across = centred @ directions[1]
                        if .085 < float(np.ptp(along)) < SODA_HEIGHT[1] + .04 and float(np.ptp(across)) < SODA_WIDTH + .03:
                            centre = seen.mean(0) + directions[0] * (along.max() + along.min()) / 2 + directions[1] * (across.max() + across.min()) / 2
                            out.append(dict(kind='lying', point=np.asarray(centre, float), axis=float(np.arctan2(directions[0][1], directions[0][0])), top=top, width=width, points=n, cut=cut))
                            continue
                    out.append(dict(kind=None, point=np.asarray(middle, float), top=top, width=width, points=n, cut=cut))
                # One trash can: patches of the same can give the same circle; the one with most rim points stays.
                for thing in sorted(bins, key=lambda b: -b['points']):
                    if not any(o['kind'] == 'bin' and np.linalg.norm(o['point'] - thing['point']) < BIN_WIDTH for o in out):
                        out.append(thing)
                # ... and whatever else was found on its walls is part of it, not a thing of its own.
                found = [o['point'] for o in out if o['kind'] == 'bin']
                out = [o for o in out if o['kind'] == 'bin' or not any(np.linalg.norm(o['point'] - c) < BIN_WIDTH / 2 + .05 for c in found)]
                for thing in out:
                    thing['xy'] = world[:2, :2] @ thing['point'] + world[:2, 3]
                    if thing['kind'] == 'lying':
                        thing['yaw'] = float(wrap(thing['axis'] + base_pose[2]))        # its axis in the map
        self._cache[key] = out
        return out

    def gap_sides(self, ahead=(.55, 1.45), band=(.05, .43), reach=.75, ignore=(), slice_length=.08):
        """The two sides of a gap between furniture straight ahead, at chassis height, as the head camera sees
        them: for each side the line y = a + b x (base frame: x ahead, y to the left) along its face, as (a, b),
        left side first; None for a side that does not show. ignore: base-frame xy of cans standing in the gap.

        Each side's face is sampled in slices across the way ahead (the point nearest to the middle in each)
        and a straight line is fitted, once more without the samples that lie off it (a corner, a leg).
        """
        points = self.points('head', stride=2); z = points[:, 2] + BASE_Z
        points = points[(z > band[0]) & (z < band[1]) & (points[:, 0] > ahead[0]) & (points[:, 0] < ahead[1]) & (np.abs(points[:, 1]) < reach)]
        if len(points):
            centres, radii = self.spheres()
            points = points[~(np.linalg.norm(points[:, None, :] - centres[None], axis=2) < radii[None] + .05).any(axis=1)]
        for xy in ignore:
            if xy is not None:
                points = points[np.linalg.norm(points[:, :2] - np.asarray(xy, float)[:2], axis=1) > SODA_WIDTH / 2 + .04]
        out = []
        for sign in (1., -1.):
            side = points[sign * points[:, 1] > 0.]; samples = []
            for start in np.arange(ahead[0], ahead[1] - 1e-6, slice_length):
                part = side[(side[:, 0] >= start) & (side[:, 0] < start + slice_length)]
                if len(part) >= 10:
                    samples.append((start + slice_length / 2, sign * float(np.percentile(sign * part[:, 1], 3))))
            line = None
            if len(samples) >= 4:
                samples = np.array(samples); keep = np.ones(len(samples), bool)
                for _ in range(2):
                    slope, offset = np.polyfit(samples[keep, 0], samples[keep, 1], 1); residual = np.abs(samples[:, 1] - (offset + slope * samples[:, 0]))
                    keep = residual < max(.012, 2. * float(np.median(residual)))
                    if keep.sum() < 4:
                        break
                else:
                    line = (float(offset), float(slope))
            out.append(line)
        return tuple(out)
