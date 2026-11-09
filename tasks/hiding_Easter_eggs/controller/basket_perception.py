"""Wicker basket candidates in head RGB-D: a small non-green object standing on the lawn.
Measured on instance 301 views: basket ~0.40 m wide, ~0.43 m tall, HSV(OpenCV) median (20,100,95); lawn (37,132,148)."""
import cv2
import numpy as np

from geometry import pointcloud

CELL = 0.08                                                                                   # m, ground grid for clustering
SPAN = 16.0                                                                                   # m, grid half width (detection range)


def detect_baskets(cam, max_range=SPAN, debug=None):                                          # list of candidates in the base frame
    p, rgb = pointcloud(cam, stride=2, max_range=max_range)                                  # [n, 3], [n, 3]
    reject = (lambda k, why: debug.append((k, why))) if debug is not None else (lambda k, why: None)
    n = int(2 * SPAN / CELL)
    ij = np.floor((p[:, :2] + SPAN) / CELL).astype(int)
    inside = np.all((ij >= 0) & (ij < n), axis=1)
    p, rgb, ij = p[inside], rgb[inside], ij[inside]
    if len(p) == 0:                                                                           # sky or far field only (full_v10_312, 314 crashed in cvtColor)
        return []
    obj = (p[:, 2] > 0.08) & (p[:, 2] < 0.60)
    tall = (p[:, 2] > 0.65) & (p[:, 2] < 1.40)                                                # trunk or bush; a canopy overhanging the basket starts higher

    grid = np.zeros((n, n), np.uint8)
    grid[ij[obj, 0], ij[obj, 1]] = 1
    high = np.zeros((n, n), np.uint8)
    high[ij[tall, 0], ij[tall, 1]] = 1
    high = cv2.dilate(high, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(grid, connectivity=8)

    hsv = cv2.cvtColor(rgb[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0].astype(float)    # [n, 3]
    label_of_point = labels[ij[:, 0], ij[:, 1]]
    out = []
    for k in range(1, count):
        w, h = stats[k, cv2.CC_STAT_WIDTH] * CELL, stats[k, cv2.CC_STAT_HEIGHT] * CELL
        centre_cells = (np.array([stats[k, cv2.CC_STAT_LEFT] + stats[k, cv2.CC_STAT_WIDTH] / 2, stats[k, cv2.CC_STAT_TOP] + stats[k, cv2.CC_STAT_HEIGHT] / 2]) * CELL - SPAN)[::-1]
        if not (0.12 <= max(w, h) <= 0.64 and min(w, h) >= 0.08):
            reject(centre_cells, f"size {w:.2f}x{h:.2f}")
            continue
        cells = labels == k
        if (high & cells).any():                                                              # trunk, bush or wall above it
            reject(centre_cells, "tall above")
            continue
        sel = obj & (label_of_point == k)
        if sel.sum() < 8:
            reject(centre_cells, "few points")
            continue
        q, c = p[sel], hsv[sel]
        top = q[:, 2].max()
        hue, sat, val = np.median(c, axis=0)
        green = np.mean((c[:, 0] >= 32) & (c[:, 0] <= 48) & (c[:, 1] > 90))
        wicker = 12 <= hue <= 27 and 70 <= sat <= 135 and 45 <= val <= 175                     # basket shaded (301): (20-24, 88-108, 85-99); sunlit (308): (13-15, 108-114, 108-146)
        if not (0.30 <= top <= 0.52 and wicker and green < 0.35):                              # basket top measured 0.41-0.46
            reject(centre_cells, f"top {top:.2f} hsv {np.round([hue, sat, val])} green {green:.2f}")
            continue
        centre = q[:, :2].mean(axis=0)
        if np.linalg.norm(centre) > max_range:                                                # depth is along the optical axis; off-axis ground range can exceed it
            continue
        out.append({"xy": centre, "top": float(top), "size": (float(w), float(h)), "points": int(sel.sum()),
                    "hsv": (float(hue), float(sat), float(val)), "green": float(green), "range": float(np.linalg.norm(centre))})
    return out


def basket_rim(cam, guess, radius=0.20):                                                     # rim circle near guess (base frame): centre xy, radius, top z
    p, _ = pointcloud(cam, stride=1, max_range=4.0)
    p = p[(np.linalg.norm(p[:, :2] - guess, axis=1) < radius + 0.15) & (p[:, 2] > 0.05) & (p[:, 2] < 0.60)]
    if len(p) < 50:
        return None
    top = np.percentile(p[:, 2], 98)
    ring = p[p[:, 2] > top - 0.04]
    if len(ring) < 30:
        return None
    A = np.c_[2 * ring[:, :2], np.ones(len(ring))]                                            # algebraic circle fit: |x|^2 = 2 c.x + (r^2 - |c|^2)
    sol = np.linalg.lstsq(A, (ring[:, :2] ** 2).sum(axis=1), rcond=None)[0]
    centre = sol[:2]
    fit = float(np.sqrt(max(sol[2] + centre @ centre, 0.0)))
    return {"xy": centre, "radius": fit, "top": float(top), "points": len(ring)}
