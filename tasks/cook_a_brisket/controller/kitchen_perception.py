"""Kitchen detectors on one camera (base frame, z up): toggle-button markers, the oven button, the fridge front from its two
handle bars, the chopping board, brisket-coloured pixels. Colour gates from checks/colour_sampler.py on the dev scans."""
import math

import cv2
import numpy as np

from geometry import pointcloud


def hsv_of(cam):
    return cv2.cvtColor(np.ascontiguousarray(cam["rgb"]), cv2.COLOR_RGB2HSV)


def blobs(cam, mask, min_area=3):                                                            # connected pixel blobs -> base-frame centre from the median depth
    n, labels, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(labels == i)
        d = cam["depth"][ys, xs]
        d = d[np.isfinite(d) & (d > 0.06)]
        if len(d) == 0:
            continue
        z = float(np.median(d))
        u, v = centres[i]
        p = np.array([(u + 0.5 - cam["K"][0, 2]) / cam["K"][0, 0] * z, (v + 0.5 - cam["K"][1, 2]) / cam["K"][1, 1] * z, z])
        out.append({"xyz": cam["R"] @ p + cam["t"], "area": int(stats[i, cv2.CC_STAT_AREA]), "px": [float(u), float(v)], "range": z})
    return out


def markers(cam):                                                                            # red (off) and green (on) toggle markers
    hsv = hsv_of(cam)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h <= 6) | (h >= 174)) & (s >= 140) & (v >= 90)
    green = (h >= 50) & (h <= 70) & (s >= 140) & (v >= 90)
    return [dict(b, on=False) for b in blobs(cam, red)] + [dict(b, on=True) for b in blobs(cam, green)]


def oven_button(cam, z_range=(0.70, 0.84)):                                                  # the oven's marker is the only one at knob height (0.767 m)
    found = [m for m in markers(cam) if z_range[0] <= m["xyz"][2] <= z_range[1]]
    return max(found, key=lambda m: m["area"]) if found else None


def board(cam, z_range=(0.88, 0.96), min_points=150):                                         # orange wood on the counter: centre [3], extent [2]
    hsv = hsv_of(cam)
    mask = (hsv[..., 0] >= 8) & (hsv[..., 0] <= 16) & (hsv[..., 1] >= 120) & (hsv[..., 2] >= 80)
    p, _ = pointcloud(cam, mask=mask, stride=1, max_range=4.5)
    p = p[(p[:, 2] >= z_range[0]) & (p[:, 2] <= z_range[1])]
    if len(p) < min_points:
        return None
    c = np.median(p[:, :2], 0)
    p = p[np.linalg.norm(p[:, :2] - c, axis=1) < 0.35]                                         # the board is 0.32 x 0.41 m; the pan handle is wood too
    if len(p) < min_points:
        return None
    lo, hi = np.percentile(p[:, :2], 3, 0), np.percentile(p[:, :2], 97, 0)
    if min(hi - lo) < 0.15 or max(hi - lo) < 0.22 or len(p) < 400:                          # the board's flat top, not the pan's wooden handle or a held brisket:
        return None                                                                           # board views 0.19-0.35 x 0.25-0.36 m, false ones 0.03-0.06 x 0.06-0.26 (board_dumps)
    return {"centre": np.r_[(lo + hi) / 2, np.median(p[:, 2])], "extent": hi - lo, "points": len(p)}


def handle_pair(q, s, full, a, d, n):                                                         # two bars 6-10 cm apart along the face (centres 8.1 cm) -> (score, mid)
    if len(q) < 40:                                                                           # q, s: proud band points and their position along d; full: all points
        return None
    hist, edges = np.histogram(s, bins=np.arange(s.min() - 0.02, s.max() + 0.03, 0.01))
    peaks = [i for i in range(1, len(hist) - 1) if hist[i] >= max(hist[i - 1], hist[i + 1]) and hist[i] >= 12]
    dist = (full[:, :2] - a) @ n
    along = (full[:, :2] - a) @ d
    proud = (dist > 0.04) & (dist < 0.10)
    best = None
    for i in peaks:
        for j in peaks:
            if not 5 <= j - i <= 11:
                continue
            ok = True
            for k in (i, j):                                                                  # the fridge handles run z 0.71-1.69 and stop: walls, door edges and
                bar = full[proud & (np.abs(along - (edges[k] + 0.005)) < 0.015)][:, 2]       # window frames keep going, cabinet handles are short
                inside = bar[(bar > 0.72) & (bar < 1.68)]
                outside = bar[(bar < 0.62) | (bar > 1.78)]
                if len(inside) < 10 or np.ptp(inside) < 0.75 or len(outside) > 0.15 * len(inside):
                    ok = False
            if not ok:
                continue
            score = min(hist[i], hist[j])
            if best is None or score > best[0]:
                best = (int(score), (edges[i] + edges[j + 1]) / 2)
    return best


def face_checks(p, hsv, a, d, n, mid):                                                       # does a candidate front look like the fridge's: wide, grey, clear
    centre = a + mid * d
    u = np.array([n[1], -n[0]])
    dist = (p[:, :2] - a) @ n
    along = (p[:, :2] - centre) @ u
    inside = (along > -0.50) & (along < 0.38)                                                 # right door 0.58 m, left door 0.44 m from the split
    face = inside & (np.abs(dist) < 0.015)
    bins = np.histogram(along[face], bins=np.arange(-0.50, 0.381, 0.05))[0]
    coverage = float(np.mean(bins >= 3))
    sat = float(np.median(hsv[face, 1])) if face.sum() else 255.0
    proud = inside & (dist > 0.02) & (dist < 0.15) & (np.abs(along) > 0.08)                   # anything sticking out besides the two bars
    clutter = float(proud.sum() / max(face.sum(), 1))
    return {"coverage": round(coverage, 2), "sat": round(sat, 1), "clutter": round(clutter, 3)}


def fridge_front(cam, rng=np.random.default_rng(0), hypotheses=250, min_coverage=0.0, max_sat=255.0, max_clutter=99.0, prior=None):   # prior: (origin, n) base frame
    full, rgb = pointcloud(cam, stride=2, max_range=4.5)                                      # handle bars 7 cm proud of a wide grey front -> frame
    if len(full) > 60000:
        pick = rng.choice(len(full), 60000, replace=False)
        full, rgb = full[pick], rgb[pick]
    keep = (full[:, 2] > 0.85) & (full[:, 2] < 1.55)
    p, rgb = full[keep], rgb[keep]
    if len(p) < 500:
        return None
    if len(p) > 20000:
        pick = rng.choice(len(p), 20000, replace=False)
        p, rgb = p[pick], rgb[pick]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[None]), cv2.COLOR_RGB2HSV)[0]
    best = None
    for _ in range(hypotheses):                                                              # RANSAC vertical planes (lines in xy), each scored by its handle pair
        a, b = p[rng.choice(len(p), 2, replace=False)]
        d = b[:2] - a[:2]
        if np.linalg.norm(d) < 0.3:
            continue
        d = d / np.linalg.norm(d)
        n = np.array([-d[1], d[0]])
        if n @ (-a[:2]) < 0:                                                                  # normal toward the robot
            n = -n
        if prior is not None and n @ prior[1][:2] < math.cos(math.radians(12.0)):               # refining a known front: keep near its normal
            continue
        dist = (p[:, :2] - a[:2]) @ n
        on = np.abs(dist) < 0.012
        if on.sum() < 150:
            continue
        s_on = (p[on, :2] - a[:2]) @ d
        proud = (dist > 0.04) & (dist < 0.10)                                                 # in front of the face by the handle depth
        s = (p[proud, :2] - a[:2]) @ d
        inside = (s > s_on.min()) & (s < s_on.max())
        pair = handle_pair(p[proud][inside], s[inside], full, a[:2], d, n)
        if pair is None:
            continue
        if prior is not None and np.linalg.norm(a[:2] + pair[1] * d - prior[0][:2]) > 0.15:
            continue
        checks = face_checks(p, hsv, a[:2], d, n, pair[1])
        if checks["coverage"] < min_coverage or checks["sat"] > max_sat or checks["clutter"] > max_clutter:
            continue
        if best is None or pair[0] > best[0]:
            best = (pair[0], a[:2].copy(), d, n, pair[1], int(on.sum()), checks)
    if best is None:
        return None
    score, a, d, n, mid, support, checks = best
    centre = a + mid * d
    near = p[(np.abs((p[:, :2] - a) @ n) < 0.015) & (np.abs((p[:, :2] - centre) @ d) < 0.5)][:, :2]   # refit the face round the handles
    if len(near) > 50:
        c = near.mean(0)
        d2 = np.linalg.svd(near - c, full_matrices=False)[2][0]
        d = d2 if d2 @ d > 0 else -d2
        n = np.array([-d[1], d[0]]) * np.sign(np.array([-d[1], d[0]]) @ n)
        a = c
        dist = (p[:, :2] - a) @ n
        proud = (dist > 0.04) & (dist < 0.10) & (np.abs((p[:, :2] - centre) @ d) < 0.3)
        pair = handle_pair(p[proud], (p[proud, :2] - a) @ d, full, a, d, n)
        mid = pair[1] if pair is not None else (centre - a) @ d
    origin = a + mid * d
    u = np.array([n[1], -n[0]])                                                               # n x z: the robot's left when it faces the front
    return {"origin": np.r_[origin, 0.0], "n": np.r_[n, 0.0], "u": np.r_[u, 0.0], "support": support, "score": score, **checks}


def face_near(cam, frame, along=(-0.5, 0.4), z_range=(0.3, 1.9), window=0.15, rng=np.random.default_rng(0)):   # the flat front near a predicted frame:
    p, _ = pointcloud(cam, stride=2, max_range=3.0)                                           # RANSAC line (1 cm) among points within +-window of the predicted
    q = to_local(p, frame)                                                                    # face, its normal within 10 deg -> (n offset, normal) in the frame
    keep = (q[:, 0] > along[0]) & (q[:, 0] < along[1]) & (np.abs(q[:, 1]) < window) & (q[:, 2] > z_range[0]) & (q[:, 2] < z_range[1])
    q = q[keep]
    if len(q) < 300:
        return None
    best = None
    for _ in range(150):
        a, b = q[rng.choice(len(q), 2, replace=False)]
        d = b[:2] - a[:2]
        if np.linalg.norm(d) < 0.2:
            continue
        d = d / np.linalg.norm(d)
        if abs(d[1]) > math.sin(math.radians(10.0)):                                          # the face runs along u
            continue
        inliers = np.abs((q[:, :2] - a[:2]) @ np.array([-d[1], d[0]])) < 0.01
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < 200:
        return None
    f = q[best][:, :2]
    c = f.mean(0)
    d = np.linalg.svd(f - c, full_matrices=False)[2][0]
    d = d if d[0] > 0 else -d
    return {"n_offset": float(c[1] - d[1] / d[0] * c[0]), "slope": float(d[1] / d[0]), "inliers": int(best.sum())}


def bars_near(cam, frame, face_n, expect_u, half=0.12):                                       # handle bars 4-10 cm proud of the face round expect_u -> their midpoint u
    p, _ = pointcloud(cam, stride=1, max_range=3.0)
    q = to_local(p, frame)
    keep = (q[:, 1] > face_n + 0.04) & (q[:, 1] < face_n + 0.10) & (np.abs(q[:, 0] - expect_u) < half) & (q[:, 2] > 0.8) & (q[:, 2] < 1.6)
    s = q[keep][:, 0]
    if len(s) < 60:
        return None
    hist, edges = np.histogram(s, bins=np.arange(expect_u - half, expect_u + half + 0.001, 0.01))
    peaks = [i for i in range(1, len(hist) - 1) if hist[i] >= max(hist[i - 1], hist[i + 1]) and hist[i] >= 10]
    pairs = [(i, j) for i in peaks for j in peaks if 6 <= j - i <= 10]
    if not pairs:
        return None
    i, j = max(pairs, key=lambda ij: min(hist[ij[0]], hist[ij[1]]))
    centres = [s[np.abs(s - (edges[k] + 0.005)) < 0.016].mean() for k in (i, j)]                # bar centroids, not bin edges (1.5 cm median error)
    return float(np.mean(centres))


def bar_near(cam, frame, face_n, expect_u, half=0.06):                                      # one handle bar 4-10 cm proud of the face round expect_u -> its centroid u
    p, _ = pointcloud(cam, stride=1, max_range=3.0)                                           # (the left door with the right one open: no pair for bars_near)
    q = to_local(p, frame)
    keep = (q[:, 1] > face_n + 0.04) & (q[:, 1] < face_n + 0.10) & (np.abs(q[:, 0] - expect_u) < half) & (q[:, 2] > 0.8) & (q[:, 2] < 1.6)
    s = q[keep][:, 0]
    if len(s) < 60:
        return None
    hist, edges = np.histogram(s, bins=np.arange(expect_u - half, expect_u + half + 0.001, 0.01))
    runs, start = [], None                                                                    # whole bars: runs of occupied 1 cm bins not cut by the window
    for i, h in enumerate(np.r_[hist, 0]):                                                    # (a neighbour bar's side face seen edge-on is one dense bin at the edge)
        if h >= 5 and start is None:
            start = i
        if h < 5 and start is not None:
            runs.append((start, i))
            start = None
    runs = [(a, b) for a, b in runs if a > 0 and b < len(hist)]
    if not runs:
        return None
    a, b = max(runs, key=lambda r: hist[r[0]:r[1]].sum())
    return float(s[(s >= edges[a]) & (s < edges[b])].mean())


def to_local(p, frame):                                                                       # base-frame points [n, 3] -> object frame (u, n, z) [n, 3]
    q = p - frame["origin"]
    return np.c_[q @ frame["u"], q @ frame["n"], q[:, 2]]


def slab_on_shelf(cam, frame, u_range, n_front, top, min_points=60, min_sat=0, beyond=0.0):   # beyond: also past the prior shelf edge (a pushed slab overhangs it)           # brisket box on one shelf, object-frame (u, n) extents, or None:
    p, rgb = pointcloud(cam, stride=1, max_range=1.6)                                         # points 0.5-4 cm above the shelf plane inside the compartment (n > -0.49:
    if len(p) == 0:                                                                           # a back panel at n -0.51 read as a slab, dev_find2), grouped into 1 cm blobs;
        return None                                                                           # the largest blob at least 6 x 5 cm (the divider wall's foot, 1.6 cm wide, won
    q = to_local(p, frame)                                                                    # the median in full_v5_305)
    sat = cv2.cvtColor(np.ascontiguousarray(rgb[None]), cv2.COLOR_RGB2HSV)[0][:, 1]
    across = (q[:, 0] > u_range[0] + 0.01) & (q[:, 0] < u_range[1] - 0.01)
    plane = q[across & (np.abs(q[:, 2] - top) < 0.004) & (q[:, 1] > n_front - 0.15) & (q[:, 1] < n_front + 0.05)]
    edge = None                                                                               # measured shelf front edge (the prior carries the frame error): set below,
    inside = across & (q[:, 1] < n_front - 0.003 + beyond) & (q[:, 1] > -0.49)
    tall = q[inside & (q[:, 2] > top + 0.06) & (q[:, 2] < top + 0.15)]                        # walls rise above the shelf, the 2 cm slab does not (a divider foot
    q = q[inside & (q[:, 2] > top + 0.005) & (q[:, 2] < top + 0.04) & (sat >= min_sat)]       # merged with a slab touching it under a 2.5 cm frame error, find4_307)
    if len(q) < min_points:
        return None
    size = (int(round((0.0 + 0.49) / 0.01)) + 2, int(round((u_range[1] - u_range[0]) / 0.01)) + 2)
    cell = np.clip(np.floor((q[:, :2] - [u_range[0], -0.49]) / 0.01).astype(int), 0, [size[1] - 1, size[0] - 1])
    wall = np.zeros(size, np.uint8)
    if len(tall):
        tc = np.clip(np.floor((tall[:, :2] - [u_range[0], -0.49]) / 0.01).astype(int), 0, [size[1] - 1, size[0] - 1])
        wall[tc[:, 1], tc[:, 0]] = 1
        wall = cv2.dilate(wall, np.ones((3, 3), np.uint8))
    free = wall[cell[:, 1], cell[:, 0]] == 0
    q, cell = q[free], cell[free]
    if len(q) < min_points:
        return None
    grid = np.zeros(size, np.uint8)
    grid[cell[:, 1], cell[:, 0]] = 1
    count, labels = cv2.connectedComponents(cv2.dilate(grid, np.ones((3, 3), np.uint8)), connectivity=8)
    best = None
    for k in range(1, count):
        blob = q[labels[cell[:, 1], cell[:, 0]] == k]
        if len(blob) < min_points:
            continue
        lo, hi = np.percentile(blob[:, :2], 2, 0), np.percentile(blob[:, :2], 98, 0)
        if hi[0] - lo[0] < 0.06 or hi[1] - lo[1] < 0.05:
            continue
        if best is None or len(blob) > best[0]:
            best = (len(blob), lo, hi, blob)
    if best is None:
        return None
    _, lo, hi, blob = best
    mid_u = blob[(blob[:, 0] >= np.percentile(blob[:, 0], 25)) & (blob[:, 0] <= np.percentile(blob[:, 0], 75))]   # edges from the blob's middle band: a wall foot touching
    mid_n = blob[(blob[:, 1] >= np.percentile(blob[:, 1], 25)) & (blob[:, 1] <= np.percentile(blob[:, 1], 75))]   # the slab's side stays out of front / back
    lo = np.array([np.percentile(mid_n[:, 0], 2), np.percentile(mid_u[:, 1], 2)])
    hi = np.array([np.percentile(mid_n[:, 0], 98), np.percentile(mid_u[:, 1], 98)])
    beside = plane[(plane[:, 0] < lo[0] - 0.02) | (plane[:, 0] > hi[0] + 0.02)]                  # beside the slab: an overhanging slab's front face bottom sits at
    edge = float(np.percentile(beside[:, 1], 99)) if len(beside) > 200 else None               # shelf height and read as the edge (2.7 cm short, dev_relook 301)
    return {"u": float((lo[0] + hi[0]) / 2), "u_lo": float(lo[0]), "u_hi": float(hi[0]), "front": float(hi[1]), "back": float(lo[1]), "top": float(np.percentile(blob[:, 2], 90)),
            "points": int(len(blob)), "shelf_edge": edge}


def orange(cam, s_min=90, v_min=50):                                                         # brisket / board coloured pixels [H, W] bool
    hsv = hsv_of(cam)
    return (hsv[..., 0] <= 20) & (hsv[..., 1] >= s_min) & (hsv[..., 2] >= v_min) & ~(((hsv[..., 0] <= 6) | (hsv[..., 0] >= 174)) & (hsv[..., 1] >= 200) & (hsv[..., 2] >= 200))
