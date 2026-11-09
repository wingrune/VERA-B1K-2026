"""Tree trunks in head RGB-D: a narrow column standing on the ground that continues up into a canopy above 2 m.
Bushes (wide, under 2 m), fences (long) and garden lights (no canopy) fail one of the tests."""
import cv2
import numpy as np

from geometry import pointcloud

CELL = 0.08                                                                                   # m, ground grid
SPAN = 12.0                                                                                   # m, grid half width


def detect_trunks(cam, max_range=SPAN, need_canopy=True):                                     # list of trunks in the base frame
    p, rgb = pointcloud(cam, stride=2, max_range=max_range)                                  # [n, 3], [n, 3]
    n = int(2 * SPAN / CELL)
    ij = np.floor((p[:, :2] + SPAN) / CELL).astype(int)
    ok = np.all((ij >= 0) & (ij < n), axis=1)
    p, rgb, ij = p[ok], rgb[ok], ij[ok]
    band = (p[:, 2] > 0.30) & (p[:, 2] < 1.20)                                               # trunk section
    grid = np.zeros((n, n), np.uint8)
    grid[ij[band, 0], ij[band, 1]] = 1
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)
    label_of_point = labels[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, count):
        w, h = stats[k, cv2.CC_STAT_WIDTH] * CELL, stats[k, cv2.CC_STAT_HEIGHT] * CELL
        if max(w, h) > 0.70:                                                                  # bush, fence, wall
            continue
        sel = band & (label_of_point == k)
        if sel.sum() < 15:
            continue
        q = p[sel]
        if len(np.unique(np.floor((q[:, 2] - 0.30) / 0.10))) < 8:                           # continuous 0.3-1.2 m column (the basket is 0.45 m tall)
            continue
        centre = np.median(q[:, :2], axis=0)
        near = np.linalg.norm(p[:, :2] - centre, axis=1)
        canopy = (p[:, 2] > 3.0) & (near < 2.5)                                               # garden palms and lights stay under 3 m
        low = (p[:, 2] > 0.02) & (p[:, 2] < 0.30) & (near < 0.45)
        if need_canopy and canopy.sum() < 30:
            continue
        view = centre / max(np.linalg.norm(centre), 1e-6)                                    # only the near side of the trunk is seen
        side = q[:, :2] @ np.array([-view[1], view[0]])
        radius = float(np.percentile(side, 95) - np.percentile(side, 5)) / 2                  # half the visible width
        base = np.median(p[low, :2], axis=0) if low.sum() >= 5 else centre                  # trunk foot (trunks can lean)
        base = base + 0.7 * radius * view                                                     # visible surface -> axis (mean depth of a half cylinder ~ 0.7 r nearer)
        centre = centre + 0.7 * radius * view
        hsv = np.median(cv2.cvtColor(rgb[sel][None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0], axis=0)
        if not (18 <= hsv[0] <= 35 and 25 <= hsv[1] <= 90 and 70 <= hsv[2] <= 140):        # bark (301: 22-32, 32-73, 90-119); not white poles or red fence posts
            continue
        out.append({"xy": base, "section": centre, "radius": radius, "points": int(sel.sum()), "canopy": int(canopy.sum()), "range": float(np.linalg.norm(base)), "hsv": hsv.tolist()})
    return out
