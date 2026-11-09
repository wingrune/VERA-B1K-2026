"""The scrub brush and the dirt on the bathroom floor, from the head camera's colour and depth images.

- The brush is red (bristles dark red): 15 cm long, 7 cm wide, 3 cm high, lying on the floor. Red pixels within
  6 cm of the floor are grouped on a 2 cm grid; the biggest group 8-25 cm long is the brush (its middle and the
  direction of its long side).
- Dirt: dark marks on the light floor (dirt_view.py), red ones (the brush) left out, and the fixed dark spots of
  priors/dark_spots.json (seams and corners of the floor that look like dirt) too.

Everything returned is in the map frame M.
"""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from dirt_view import DirtView, pixel_points

BRUSH = dict(height=.06, long=(.08, .25), pixels=12)
FIXED = .07                         # a mark this near to a fixed dark spot is that spot


def red(colour):
    r, g, b = colour[:, 0], colour[:, 1], colour[:, 2]
    return (r > 90) & (r > 1.6 * g) & (r > 1.6 * b)


class ScrubView(DirtView):
    def __init__(self, floor_lo, floor_hi, root):
        super().__init__(floor_lo, floor_hi)
        spots = Path(root) / 'priors' / 'dark_spots.json'
        self.fixed = np.array(json.loads(spots.read_text())['spots']) if spots.exists() else np.zeros((0, 2))

    def look(self, depth, rgb, cam_rel, focal_ratio, base, stride=2):
        """(dirt marks [dict(xy, pixels, range)], brush dict(xy, direction, length, pixels, range) or None)."""
        pts, v, u = pixel_points(depth, cam_rel, focal_ratio, stride)
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw)
        xy = np.c_[x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]]
        colour = rgb[v, u, :3].astype(np.float32)
        inside = np.all(xy > self.lo, axis=1) & np.all(xy < self.hi, axis=1)
        is_red = red(colour)
        brush = self._brush(xy[inside & is_red & (pts[:, 2] < BRUSH['height'])], np.linalg.norm(pts[inside & is_red & (pts[:, 2] < BRUSH['height']), :2], axis=1))
        marks = self.see(depth, rgb, cam_rel, focal_ratio, base, stride, exclude=is_red)
        out = []
        for m in marks:
            if len(self.fixed) and float(np.min(np.hypot(*(self.fixed - m['xy']).T))) < FIXED:
                continue
            if brush is not None and float(np.linalg.norm(m['xy'] - brush['xy'])) < .12:
                continue                # (the brush's dark bristles)
            out.append(m)
        return out, brush

    def _brush(self, xy, rng):
        if len(xy) < BRUSH['pixels']:
            return None
        cells = np.floor((xy - self.lo) / .02).astype(int)
        shape = tuple(np.ceil((self.hi - self.lo) / .02).astype(int) + 1)
        grid = np.zeros(shape, bool); grid[cells[:, 0], cells[:, 1]] = True
        labels, n = ndimage.label(ndimage.binary_dilation(grid))
        best = None
        for k in range(1, n + 1):
            member = labels[cells[:, 0], cells[:, 1]] == k
            if member.sum() < BRUSH['pixels']:
                continue
            p = xy[member]; centre = p.mean(axis=0); d = p - centre
            axis = np.linalg.svd(d, full_matrices=False)[2][0]; a = d @ axis
            length = float(np.percentile(a, 98) - np.percentile(a, 2))
            if not BRUSH['long'][0] < length < BRUSH['long'][1]:
                continue
            if best is None or member.sum() > best['pixels']:
                best = dict(xy=centre, direction=axis, length=length, pixels=int(member.sum()), range=float(np.median(rng[member])))
        return best
