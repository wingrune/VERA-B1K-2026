"""The mixing bowls on the counter from a coloured head cloud: red points in the counter's height band, without the columns where the red goes
on up the wall (the backsplash), grouped into blobs of a bowl's size; the bowl centre is the circle of the known rim radius fitted to the
blob's top points (from above the whole rim ring is seen; the walls alone are seen from one side). Priors: the bowl asset's rim radius and
rim height (params). Tracks merge sightings within BOWL_MATCH_M."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from scipy import ndimage

from .. import params as P
from ..world import BowlEst


def red_mask(col: np.ndarray) -> np.ndarray:
    r, g, b = (col[:, k].astype(float) for k in range(3))
    return (r >= P.RED_MIN) & (g <= P.RED_G_MAX * r) & (b <= P.RED_B_MAX * r)


def fit_rim(xy: np.ndarray, radius: float, start: np.ndarray, iters: int = 15) -> Optional[np.ndarray]:
    """Centre of the circle of a fixed radius through the points (Gauss-Newton from `start`)."""
    c = np.asarray(start, float).copy()
    for _ in range(iters):
        d = xy - c
        r = np.linalg.norm(d, axis=1)
        ok = r > 1e-6
        if ok.sum() < 6:
            return None
        u = d[ok] / r[ok, None]
        res = r[ok] - radius
        step, *_ = np.linalg.lstsq(-u, -res, rcond=None)
        c = c - step
        if np.linalg.norm(step) < 1e-4:
            break
    return c


def find_bowls(pts: np.ndarray, col: np.ndarray, step: int) -> List[BowlEst]:
    """Bowl sightings in one head view: pts (N, 3) odometry frame, col (N, 3) uint8."""
    red = red_mask(col)
    if red.sum() < P.BOWL_MIN_POINTS:
        return []
    rp = pts[red]
    wall = rp[rp[:, 2] > P.BACKSPLASH_ABOVE_M]
    band = rp[(rp[:, 2] > P.BOWL_BAND_M[0]) & (rp[:, 2] < P.BOWL_BAND_M[1])]
    if len(band) < P.BOWL_MIN_POINTS:
        return []
    res = 0.04
    lo = band[:, :2].min(0) - 2 * res
    if len(wall):                                          # columns with red over the counter's band: the backsplash behind the bowls
        wc = np.floor((wall[:, :2] - lo) / res).astype(int)
        bc = np.floor((band[:, :2] - lo) / res).astype(int)
        wset = {(int(a), int(b)) for a, b in wc}
        band = band[np.array([(int(a), int(b)) not in wset for a, b in bc], bool)]
    if len(band) < P.BOWL_MIN_POINTS:
        return []
    fine = 0.02
    lo = band[:, :2].min(0) - 2 * fine
    idx = np.floor((band[:, :2] - lo) / fine).astype(int)
    grid = np.zeros(idx.max(0) + 3, bool)
    grid[idx[:, 0], idx[:, 1]] = True
    lab, n = ndimage.label(ndimage.binary_closing(grid, iterations=2), structure=np.ones((3, 3)))
    groups = [np.flatnonzero(lab[idx[:, 0], idx[:, 1]] == k) for k in range(1, n + 1)]
    merged = True              # from a distance a bowl shows its near rim and its far inner wall as two blobs (the near interior is hidden):
    while merged:              # merge blobs whose union is still a bowl's size (two bowls side by side span 0.75 m or more)
        merged = False
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                u = band[np.r_[groups[a], groups[b]]]
                if np.all(u[:, :2].max(0) - u[:, :2].min(0) <= P.BOWL_MERGE_EXT_M):
                    groups[a] = np.r_[groups[a], groups[b]]
                    groups.pop(b)
                    merged = True
                    break
            if merged:
                break
    out = []
    for g in groups:
        if len(g) < P.BOWL_MIN_POINTS:
            continue
        p = band[g]
        ext = p[:, :2].max(0) - p[:, :2].min(0)
        if not (P.BOWL_EXT_M[0] <= ext.min() and ext.max() <= P.BOWL_EXT_M[1]):
            continue
        top = float(np.percentile(p[:, 2], 98))
        if not P.BOWL_RIM_BAND_M[0] <= top <= P.BOWL_RIM_BAND_M[1]:     # knobs, the microwave button, a strip of the backsplash
            continue
        rim = p[p[:, 2] > top - 0.025]
        c = fit_rim(rim[:, :2], P.BOWL_RIM_R_M, rim[:, :2].mean(0)) if len(rim) >= 10 else None
        if c is None or np.linalg.norm(c - p[:, :2].mean(0)) > 0.12:
            c = p[:, :2].mean(0)
        out.append(BowlEst(centre=np.array([c[0], c[1], top]), rim=top, seen_step=step))
    return out


def merge(tracks: List[BowlEst], seen: List[BowlEst]) -> None:
    """Update the tracks with this view's sightings (running mean of the centre, newest rim height)."""
    for s in seen:
        d = [float(np.linalg.norm(t.centre[:2] - s.centre[:2])) for t in tracks]
        if d and min(d) < P.BOWL_MATCH_M:
            t = tracks[int(np.argmin(d))]
            w = 1.0 / min(t.n + 1, 8)
            t.centre = (1 - w) * t.centre + w * s.centre
            t.rim, t.n, t.seen_step, t.last = s.rim, t.n + 1, s.seen_step, s.centre.copy()
        else:
            s.last = s.centre.copy()
            tracks.append(s)
