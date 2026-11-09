"""State estimation for the fax task from onboard observations only.

Inputs: head RGB-D, the camera pose relative to the base, proprioception, the robot's own last
action, plus the static priors (office map, fax geometry). No simulator state.

  * base pose in the map: Localizer (scan matching against the map)
  * fax, far away: anything a few centimetres tall standing on floor the map says is empty
  * fax, close: its 3-D model fitted to the points above the floor, with the toggle marker
    (red when off, green when on) fixing the orientation
"""
import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from localizer import Localizer, camera_points, BASE_Z
from robot_collision import RobotCollision
from wbik import Body
from kinematics import from_proprio

HEAD = 'robot_r1::robot_r1:zed_link:Camera:0::'
BUTTON = np.array([.0308, .0872, .0179])
FAX_ORIGIN_HEIGHT = .0362          # height of the fax model origin above the floor when it stands on it
FOCAL = 306. / 720.


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pose2(x, y, yaw, z=0.):
    t = np.eye(4); c, s = np.cos(yaw), np.sin(yaw); t[:2, :2] = [[c, -s], [s, c]]; t[:3, 3] = [x, y, z]; return t


class Observer:
    def __init__(self, folder, scene):
        folder = Path(folder); self.scene = scene; self.loc = Localizer(scene)
        self.body = Body(folder); self.collision = RobotCollision(folder / 'robot_collision_spheres.json')
        model = np.load(folder / 'priors' / 'facsimile' / 'model.npz')
        self.model = model['xyz'][::3].astype(float); self.tree = cKDTree(self.model); self.bounds = model['bounds'].astype(float)
        self.steps = 0; self.fax = None; self.fax_seen = -1; self.fax_stats = {}; self.far = []; self.marker = None
        self._cache = {}

    # ---- per-step input ------------------------------------------------------------------
    def frame(self, data, payload=None):
        """data: dict of numpy arrays for this step (proprio, cam_rel_poses, head rgb + depth)."""
        self.steps += 1; self.data = data; self._cache = {}
        self.q = from_proprio(data['robot_r1::proprio'].astype(float), self.body.names); self.payload = payload

    def spheres(self):
        if 'spheres' not in self._cache:
            poses = self.body.fk(self.q)[0]; centres = self.collision.centers(poses); radii = self.collision.radii
            if self.payload is not None:      # (link, points in link frame, radius) of a carried object
                link, points, radius = self.payload; t = poses[link]
                centres = np.r_[centres, points @ t[:3, :3].T + t[:3, 3]]; radii = np.r_[radii, np.full(len(points), radius)]
            self._cache['spheres'] = (centres, radii)
        return self._cache['spheres']

    def head_points(self, stride=6, with_rgb=False):
        key = ('head', stride, with_rgb)
        if key not in self._cache:
            depth = self.data[HEAD + 'depth_linear']; rel = self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[2]
            if not with_rgb:
                self._cache[key] = camera_points(depth, rel, FOCAL, stride=stride)
            else:
                x, y, z, w = rel[3:]
                rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                                [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])
                h, wd = depth.shape; f = FOCAL * wd
                v, u = np.mgrid[stride // 2:h:stride, stride // 2:wd:stride]; d = depth[v, u].astype(np.float32)
                ok = np.isfinite(d) & (d > .15) & (d < 12.)
                pc = np.stack([(u + .5 - wd / 2) / f * d, (v + .5 - h / 2) / f * d, d], -1)[ok]
                self._cache[key] = (pc @ rot.T + rel[:3], self.data[HEAD + 'rgb'][v, u][ok][:, :3].astype(np.int16))
        return self._cache[key]

    # ---- localisation --------------------------------------------------------------------
    def scan(self):
        return self.loc.scan(self.head_points(stride=3), exclude=self.spheres())

    def localize_global(self, scans):
        """scans: list of (scan, relative yaw of the base w.r.t. the first scan); the base has not translated."""
        return self.loc.localize(scans)

    # ---- fax, far ------------------------------------------------------------------------
    def detect_far(self, base_pose, stride=2):
        """Fax candidates in M: small things standing on floor that is empty in the map."""
        points, rgb = self.head_points(stride=stride, with_rgb=True)
        height = points[:, 2] + BASE_Z; sel = (height > .018) & (height < .075)
        points, rgb = points[sel], rgb[sel]
        if len(points) < 8:
            self.far = []; return []
        centres, radii = self.spheres()
        keep = ~(np.linalg.norm(points[:, None, :] - centres[None], axis=2) < radii[None] + .06).any(axis=1)
        points, rgb = points[keep], rgb[keep]
        world = pose2(*base_pose); xy = points[:, :2] @ world[:2, :2].T + world[:2, 3]
        i, j = self.scene.cell(xy); ok = self.scene.inside(i, j)
        ic, jc = np.clip(i, 0, self.scene.shape[0] - 1), np.clip(j, 0, self.scene.shape[1] - 1)
        ok &= (self.scene.top[ic, jc] < .012) & (self.scene.distance[ic, jc] > .10)
        region = self.scene.meta.get('fax_floor')
        if region is not None:
            # The task places the fax on the shared-office floor; the corridor outside is not it.
            ok &= (xy[:, 0] > region['lo'][0]) & (xy[:, 0] < region['hi'][0] + .05) & (xy[:, 1] > region['lo'][1]) & (xy[:, 1] < region['hi'][1])
        xy, rgb, points = xy[ok], rgb[ok], points[ok]
        if len(xy) < 8:
            self.far = []; return []
        cell = .03; keys = np.floor(xy / cell).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool)
        grid[tuple((keys - origin + 1).T)] = True
        labels, count = label(grid, structure=np.ones((3, 3))); tag = labels[tuple((keys - origin + 1).T)]
        found = []
        for index in range(1, count + 1):
            member = tag == index; n = int(member.sum())
            if n < 8:
                continue
            p = xy[member]; extent = np.ptp(p, axis=0); diagonal = float(np.linalg.norm(extent))
            if diagonal < .07 or diagonal > .42:
                continue
            colour = rgb[member]; saturation = colour.max(1) - colour.min(1)
            grey = float(np.mean(saturation < 60)); red = int(np.sum((colour[:, 0] > 85) & (colour[:, 0] > 2.5 * colour[:, 1] + 20) & (colour[:, 0] > 2.5 * colour[:, 2] + 20)))
            if grey < .5:
                continue
            distance = float(np.linalg.norm(points[member][:, :2].mean(0)))
            found.append(dict(xy=p.mean(0), points=n, diagonal=diagonal, grey=grey, red=red, range=distance))
        found.sort(key=lambda item: -item['points']); self.far = found
        return found

    def wrist_marker(self, arm):
        """Toggle marker seen by the wrist camera of @arm: (xyz in the base frame, 'red' | 'green', pixels) or None.

        Looking straight down from the pre-press and pre-grasp poses the marker fills a good part
        of this image, which pins the fax position to a few millimetres right before contact.
        """
        index = 0 if arm == 'left' else 1; link = f'{arm}_realsense_link'
        prefix = f'robot_r1::robot_r1:{link}:Camera:0::'
        if prefix + 'rgb' not in self.data:
            return None
        rgb = self.data[prefix + 'rgb'][..., :3].astype(np.int16); depth = self.data[prefix + 'depth_linear'].astype(np.float32)
        red = (rgb[..., 0] > 85) & (rgb[..., 0] > 2.5 * rgb[..., 1] + 20) & (rgb[..., 0] > 2.5 * rgb[..., 2] + 20)
        green = (rgb[..., 1] > 85) & (rgb[..., 1] > 2.0 * rgb[..., 0] + 20) & (rgb[..., 1] > 2.0 * rgb[..., 2] + 20)
        colour, mask = ('red', red) if red.sum() >= green.sum() else ('green', green)
        mask &= np.isfinite(depth) & (depth > .05) & (depth < .6)
        if mask.sum() < 40:
            return None
        # Keep the largest blob.
        labels, count = label(mask); sizes = np.bincount(labels.ravel()); sizes[0] = 0; mask = labels == int(np.argmax(sizes))
        if mask.sum() < 40:
            return None
        rel = self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[index]; x, y, z, w = rel[3:]
        rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])
        h, wd = depth.shape; f = 388.6639 / 480. * wd; v, u = np.nonzero(mask); d = depth[v, u]
        pc = np.stack([(u + .5 - wd / 2) / f * d, (v + .5 - h / 2) / f * d, d], -1)
        points = pc @ rot.T + rel[:3]
        return np.median(points, axis=0), colour, int(mask.sum())

    def marker_points(self):
        """Base-frame points of red and of green pixels in the head image, at full resolution.
        The marker is two centimetres across: a sub-sampled image would leave only a pixel or two of it."""
        if 'marker' in self._cache:
            return self._cache['marker']
        rgb = self.data[HEAD + 'rgb'][..., :3].astype(np.int16); depth = self.data[HEAD + 'depth_linear']
        rel = self.data['robot_r1::cam_rel_poses'].reshape(3, 7)[2]; x, y, z, w = rel[3:]
        rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])
        h, wd = depth.shape; f = FOCAL * wd; out = {}
        masks = {'red': (rgb[..., 0] > 85) & (rgb[..., 0] > 2.5 * rgb[..., 1] + 20) & (rgb[..., 0] > 2.5 * rgb[..., 2] + 20),
                 'green': (rgb[..., 1] > 85) & (rgb[..., 1] > 2.0 * rgb[..., 0] + 20) & (rgb[..., 1] > 2.0 * rgb[..., 2] + 20)}
        for name, mask in masks.items():
            v, u = np.nonzero(mask & np.isfinite(depth) & (depth > .15) & (depth < 6.))
            d = depth[v, u].astype(np.float32)
            out[name] = np.stack([(u + .5 - wd / 2) / f * d, (v + .5 - h / 2) / f * d, d], -1) @ rot.T + rel[:3] if len(v) else np.zeros((0, 3))
        self._cache['marker'] = out
        return out

    # ---- fax, close ----------------------------------------------------------------------
    def fit_fax(self, prior_xy, radius=.40):
        """Fax pose in the base frame from the head camera; @prior_xy is its rough position in the base frame.

        Returns (4x4 pose, stats) or None. The fax stands on the floor, so only x, y and yaw are free.
        """
        points, rgb = self.head_points(stride=2, with_rgb=True)
        height = points[:, 2] + BASE_Z
        near = np.linalg.norm(points[:, :2] - np.asarray(prior_xy)[None], axis=1) < radius
        centres, radii = self.spheres()
        region = near & (height > -.01) & (height < .12)
        if region.sum() < 30:
            return None
        points, rgb, height = points[region], rgb[region], height[region]
        keep = ~(np.linalg.norm(points[:, None, :] - centres[None], axis=2) < radii[None] + .03).any(axis=1)
        points, rgb, height = points[keep], rgb[keep], height[keep]
        red = (rgb[:, 0] > 85) & (rgb[:, 0] > 2.5 * rgb[:, 1] + 20) & (rgb[:, 0] > 2.5 * rgb[:, 2] + 20)
        green = (rgb[:, 1] > 85) & (rgb[:, 1] > 2.0 * rgb[:, 0] + 20) & (rgb[:, 1] > 2.0 * rgb[:, 2] + 20)
        body = (height > .014) & (height < .080) & ~red & ~green
        if body.sum() < 40:
            return None
        cloud = points[body]
        # Largest connected blob of above-floor points near the prior.
        cell = .02; keys = np.floor(cloud[:, :2] / cell).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool)
        grid[tuple((keys - origin + 1).T)] = True
        labels, count = label(grid, structure=np.ones((3, 3))); tag = labels[tuple((keys - origin + 1).T)]
        sizes = np.bincount(tag, minlength=count + 1); sizes[0] = 0; cloud = cloud[tag == int(np.argmax(sizes))]
        if len(cloud) < 40:
            return None
        extent = np.sort(np.ptp(cloud[:, :2], axis=0))
        centre = cloud[:, :2].mean(0)
        marker = None; colour = None
        for name, spots in self.marker_points().items():
            spots = spots[(np.linalg.norm(spots[:, :2] - centre[None], axis=1) < .20) & (spots[:, 2] + BASE_Z < .10) & (spots[:, 2] + BASE_Z > .02)]
            if len(spots) >= 5 and (marker is None or len(spots) > marker[1]):
                marker = (np.median(spots, axis=0), len(spots)); colour = name
        sample = cloud[np.random.default_rng(0).choice(len(cloud), min(len(cloud), 500), replace=False)]
        z = FAX_ORIGIN_HEIGHT - BASE_Z; model_centre = self.model[:, :2].mean(0)

        def residual(x):
            c, s = np.cos(x[2]), np.sin(x[2]); rot = np.array([[c, -s], [s, c]])
            local = np.c_[(sample[:, :2] - x[:2]) @ rot, sample[:, 2] - z]
            error = np.minimum(self.tree.query(local)[0], .03)
            if marker is not None:
                error = np.r_[error, 3. * (rot @ BUTTON[:2] + x[:2] - marker[0][:2])]
            return error

        best = None
        for yaw in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            c, s = np.cos(yaw), np.sin(yaw); start = np.r_[centre - np.array([[c, -s], [s, c]]) @ model_centre, yaw]
            fit = least_squares(residual, start, max_nfev=30, diff_step=1e-3)
            cost = float(np.mean(fit.fun[:len(sample)] ** 2)) + (float(np.sum(fit.fun[len(sample):] ** 2)) / 9. if marker is not None else 0.)
            if best is None or cost < best[0]:
                best = (cost, fit.x)
        cost, x = best; c, s = np.cos(x[2]), np.sin(x[2]); rot = np.array([[c, -s], [s, c]])
        local = np.c_[(sample[:, :2] - x[:2]) @ rot, sample[:, 2] - z]; distance = self.tree.query(local)[0]
        stats = dict(points=int(len(cloud)), rms=float(np.sqrt(np.mean(np.minimum(distance, .03) ** 2))), inlier=float(np.mean(distance < .008)),
                     marker=colour, marker_error=None if marker is None else float(np.linalg.norm(rot @ BUTTON[:2] + x[:2] - marker[0][:2])),
                     range=float(np.linalg.norm(x[:2])))
        # Without the marker the wedge-shaped body alone leaves a 180 degree ambiguity: reject.
        if marker is None or stats['marker_error'] > .025 or stats['inlier'] < .55 or stats['rms'] > .012:
            self.fax_stats = dict(stats, accepted=False); return None
        pose = pose2(x[0], x[1], x[2], z); self.fax_stats = dict(stats, accepted=True); self.marker = colour
        return pose, stats
