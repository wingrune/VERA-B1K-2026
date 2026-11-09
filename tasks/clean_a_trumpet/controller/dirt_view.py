"""Dirt on the bathroom floor, from the head camera's colour and depth images.

The floor is a light beige; the dirt shows on it as dark marks a few centimetres across (from 2.7 m, 150 pixels of
the 720 x 720 image for one mark). The pixels whose depth puts them on the floor (within 1.5 cm of it, inside the
bathroom's floor) are compared with the floor's own brightness round them (the median over 0.5 m squares of floor
seen in the same image: the light differs across the room); a pixel well darker than that is dirt. Dirt pixels are
grouped on a 2 cm grid of the floor; each group small enough to be a mark is one.

Everything returned is in the map frame M.
"""
import numpy as np
from scipy import ndimage

FLOOR_Z = .015                      # a pixel within this of the floor is the floor
DARK = .45                          # dirt: darker than this fraction of the floor round it (dirt pixels 0.12-0.44 of it, shadows by the fixtures 0.41-0.69)
CELL = .02                          # the grid dirt pixels are grouped on (m)
SQUARE = .5                         # the floor's brightness is taken over squares this big (m)
MARK = dict(pixels=3, size=.30)     # a group with fewer pixels is noise; one bigger across is a shadow


def pixel_points(depth, cam_rel, focal_ratio, stride=2, max_range=6.):
    """3-D points in the base frame for every @stride-th pixel of a depth image, with their pixel rows and columns."""
    x, y, z, w = cam_rel[3:]
    rot = np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                    [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]) @ np.diag([1., -1., -1.])
    h, wd = depth.shape; f = focal_ratio * wd
    v, u = np.mgrid[stride // 2:h:stride, stride // 2:wd:stride]; d = depth[v, u].astype(np.float32)
    ok = np.isfinite(d) & (d > .15) & (d < max_range)
    pc = np.stack([(u + .5 - wd / 2) / f * d, (v + .5 - h / 2) / f * d, d], -1)[ok]
    return pc @ rot.T + cam_rel[:3], v[ok], u[ok]


class DirtView:
    def __init__(self, floor_lo, floor_hi):
        self.lo = np.asarray(floor_lo, float); self.hi = np.asarray(floor_hi, float)
        self.debug = {}

    def see(self, depth, rgb, cam_rel, focal_ratio, base, stride=2, exclude=None):
        """The dirt marks this image shows: [dict(xy, pixels, range)]. @exclude: pixels (in pixel_points' order) that
        are not dirt whatever their colour."""
        pts, v, u = pixel_points(depth, cam_rel, focal_ratio, stride)
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
        xy = np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]]
        floor = (np.abs(pts[:, 2]) < FLOOR_Z) & np.all(xy > self.lo, axis=1) & np.all(xy < self.hi, axis=1)
        if exclude is not None:
            floor &= ~exclude
        self.debug = dict(floor=int(floor.sum()))
        if floor.sum() < 50:
            return []
        xy, v, u, rng = xy[floor], v[floor], u[floor], np.linalg.norm(pts[floor, :2], axis=1)
        colour = rgb[v, u, :3].astype(np.float32); bright = colour @ np.array([.299, .587, .114])
        # the floor's brightness round each pixel: the median over its square
        key = np.floor((xy - self.lo) / SQUARE).astype(int); index = key[:, 0] * 1000 + key[:, 1]
        reference = np.empty_like(bright)
        for k in np.unique(index):
            m = index == k; reference[m] = np.median(bright[m]) if m.sum() >= 20 else np.median(bright)
        dark = (bright < DARK * reference) & (reference - bright > 30.)
        self.debug.update(dark=int(dark.sum()))
        if not dark.any():
            return []
        cells = np.floor((xy[dark] - self.lo) / CELL).astype(int)
        shape = tuple(np.ceil((self.hi - self.lo) / CELL).astype(int) + 1)
        grid = np.zeros(shape, int); np.add.at(grid, (cells[:, 0], cells[:, 1]), 1)
        labels, n = ndimage.label(ndimage.binary_dilation(grid > 0))
        out = []
        for k in range(1, n + 1):
            member = labels[cells[:, 0], cells[:, 1]] == k
            if member.sum() < MARK['pixels']:
                continue
            pts_k = xy[dark][member]; extent = pts_k.max(axis=0) - pts_k.min(axis=0)
            if extent.max() > MARK['size']:
                continue
            out.append(dict(xy=pts_k.mean(axis=0), pixels=int(member.sum()), range=float(np.median(rng[dark][member]))))
        return out
