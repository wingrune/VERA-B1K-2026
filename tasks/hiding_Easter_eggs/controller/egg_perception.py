"""Easter eggs on the lawn in head RGB-D: saturated yellow blobs whose points lie 0.02-0.20 m above the ground.
The eggs (rxwfse at ~3x scale) are 0.13-0.15 m, yellow with red and blue patterns."""
import cv2
import numpy as np

from geometry import pointcloud


def detect_eggs(cam, max_range=4.0):                                                          # list of egg centres (base frame, ground contact height)
    hsv = cv2.cvtColor(cam["rgb"].astype(np.uint8), cv2.COLOR_RGB2HSV)
    yellow = (hsv[..., 0] >= 20) & (hsv[..., 0] <= 34) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 140)
    yellow = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(yellow, connectivity=8)
    out = []
    for k in range(1, count):
        if stats[k, cv2.CC_STAT_AREA] < 30:
            continue
        p, _ = pointcloud(cam, mask=labels == k, max_range=max_range)
        if len(p) < 20:
            continue
        top = np.percentile(p[:, 2], 95)
        if not 0.04 < top < 0.22:                                                             # on the ground, not in a raised basket
            continue
        xy = np.median(p[:, :2], axis=0)
        same = [e for e in out if np.linalg.norm(e["xy"] - xy) < 0.12]                       # patterns split one egg into several blobs
        if same:
            same[0]["xy"] = (same[0]["xy"] + xy) / 2
            continue
        out.append({"xy": xy, "top": float(top), "pixels": int(stats[k, cv2.CC_STAT_AREA])})
    return out


def top_egg(cam, centre, radius=0.25):                                                        # highest egg in the basket around centre (base xy): [top x, y, z, long axis x, y, elongation] or None
    hsv = cv2.cvtColor(cam["rgb"].astype(np.uint8), cv2.COLOR_RGB2HSV)
    yellow = (hsv[..., 0] >= 20) & (hsv[..., 0] <= 34) & (hsv[..., 1] >= 150) & (hsv[..., 2] >= 100)   # value floor 100 (lawn eggs: 140): a last egg in
    p, _ = pointcloud(cam, mask=yellow, max_range=3.0)                                                   # the basket's shade sits at V ~100 (full_v32_305: 2 points at 140)
    p = p[(np.linalg.norm(p[:, :2] - centre, axis=1) < radius) & (p[:, 2] > 0.03) & (p[:, 2] < 0.32)]   # egg tops in the basket <= 0.27 m; sunlit rim
    if len(p) < 30:                                                                           # highlights at 0.35-0.41 m pass the yellow mask (full_v28_302: 3 picks aimed at the rim)
        return None
    z = np.percentile(p[:, 2], 98)
    cap = p[p[:, 2] > z - 0.03]                                                               # the top 3 cm of the highest egg,
    peak = cap[cap[:, 2] >= np.percentile(cap[:, 2], 90)][:, :2].mean(axis=0)
    cap = cap[np.linalg.norm(cap[:, :2] - peak, axis=1) < 0.06]                               # one egg: two eggs side by side put the median between them (full_v32_309)
    top = np.r_[np.median(cap[:, :2], axis=0), z]
    body = p[(np.linalg.norm(p[:, :2] - top[:2], axis=1) < 0.09) & (p[:, 2] > z - 0.06)]       # its upper 6 cm
    w, v = np.linalg.eigh(np.cov(body[:, :2].T))                                              # ascending
    axis = v[:, 1] if v[0, 1] >= 0 else -v[:, 1]
    return np.r_[top, axis, np.sqrt(w[1] / max(w[0], 1e-9))]
