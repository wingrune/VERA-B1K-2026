"""Odometry-frame 2-D garden map from head depth: obstacle log-odds, observed ground, searched area; minimum-cost paths."""
import math

import cv2
import numpy as np
from skimage.graph import MCP_Geometric

from geometry import pointcloud

RES = 0.1                                                                                     # m per cell
N = 1000                                                                                      # 100 m square centred on the start pose
COARSE = 2                                                                                    # planning grid = 0.2 m


def rot2(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class GardenMap:
    def __init__(self):
        self.odds = np.zeros((N, N), np.int8)                                                # > 1: obstacle
        self.ground = np.zeros((N, N), bool)                                                 # lawn or path seen
        self.searched = np.zeros((N, N), bool)                                               # seen within basket-detector range
        self.green = np.zeros((N, N), np.int16)                                              # ground points within 6 m that look like grass / not
        self.bare = np.zeros((N, N), np.int16)
        self.keepout = []                                                                     # (xy, radius) cleared from obstacles, e.g. the basket while approaching it

    def ij(self, xy):
        return np.floor(np.asarray(xy) / RES + N / 2).astype(int)

    def xy(self, ij):
        return (np.asarray(ij) - N / 2 + 0.5) * RES

    def update(self, cam, pose, capsules=(), search_range=5.0):                              # pose = odometry (x, y, yaw); basket detection is reliable to 6 m (dev_views2_301); 7 m left baskets unseen (full_v23_305, 306)
        p, rgb = pointcloud(cam, stride=4, max_range=20.0)                                   # [n, 3] base frame, [n, 3]
        body = np.linalg.norm(p[:, :2], axis=1) < 0.50
        for a, b, radius in capsules:                                                         # arm links seen by the head
            axis = b - a
            t = np.clip((p - a) @ axis / max(axis @ axis, 1e-9), 0, 1)
            body |= np.linalg.norm(p - a - t[:, None] * axis, axis=1) < radius
        p, rgb = p[~body], rgb[~body]
        w = p[:, :2] @ rot2(pose[2]).T + pose[:2]                                            # [n, 2] odometry frame
        ij = self.ij(w)
        ok = np.all((ij >= 0) & (ij < N), axis=1)
        p, rgb, ij = p[ok], rgb[ok], ij[ok]
        if len(p) == 0:
            return
        flat = ij[:, 0] * N + ij[:, 1]
        low = np.abs(p[:, 2]) < 0.04
        obstacle = ((p[:, 2] > 0.04) & (p[:, 2] < 1.8)) | (p[:, 2] < -0.04)                  # steps both ways: full_v5_311 and full_v7_303 stuck on the patio edge; 0.04-0.07 m is 0.1-0.5% of lawn points (step_check)
        n_obs = np.bincount(flat[obstacle], minlength=N * N).reshape(N, N)
        n_low = np.bincount(flat[low], minlength=N * N).reshape(N, N)
        odds = self.odds.astype(np.int16)
        odds[n_obs >= 2] += 3
        odds[(n_low >= 3) & (n_obs == 0)] -= 1
        self.odds = np.clip(odds, -4, 10).astype(np.int8)
        self.ground |= n_low >= 1
        hsv = cv2.cvtColor(rgb[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
        grass = (hsv[:, 0] >= 30) & (hsv[:, 0] <= 55) & (hsv[:, 1] >= 60)                      # lawn HSV ~ (37, 132, 148); gravel beds are grey
        close = low & (np.linalg.norm(p[:, :2], axis=1) < 6.0)
        self.green = np.minimum(self.green + np.bincount(flat[close & grass], minlength=N * N).reshape(N, N), 1000).astype(np.int16)
        self.bare = np.minimum(self.bare + np.bincount(flat[close & ~grass], minlength=N * N).reshape(N, N), 1000).astype(np.int16)
        near = np.linalg.norm(p[:, :2], axis=1) < search_range
        seen = np.zeros(N * N, np.uint8)
        seen[flat[near & (low | obstacle)]] = 1
        self.searched |= cv2.dilate(seen.reshape(N, N), np.ones((5, 5), np.uint8)) > 0      # far ground points are sparse

    def grass_fraction(self, xy, radius=0.25):                                                # share of observed ground near xy that looks like grass (None if unseen)
        c = self.ij(xy)
        r = int(radius / RES)
        g = self.green[max(c[0] - r, 0):c[0] + r + 1, max(c[1] - r, 0):c[1] + r + 1].sum()
        b = self.bare[max(c[0] - r, 0):c[0] + r + 1, max(c[1] - r, 0):c[1] + r + 1].sum()
        return None if g + b < 20 else float(g) / float(g + b)

    def block(self, xy, radius=0.25):                                                        # mark a spot the base could not drive through
        c = self.ij(xy)
        r = int(radius / RES)
        self.odds[max(c[0] - r, 0):c[0] + r + 1, max(c[1] - r, 0):c[1] + r + 1] = 10

    def obstacles(self):
        occ = (self.odds > 1).astype(np.uint8)
        for xy, radius in self.keepout:
            c = self.ij(xy)
            cv2.circle(occ, (int(c[1]), int(c[0])), int(radius / RES), 0, -1)
        return occ > 0

    def costs(self, radius=0.35, margin=0.35):                                                # [N/COARSE, N/COARSE] planning costs
        occ = self.obstacles().reshape(N // COARSE, COARSE, N // COARSE, COARSE).any(axis=(1, 3))
        dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 5) * RES * COARSE  # m to the nearest obstacle
        known = self.ground.reshape(N // COARSE, COARSE, N // COARSE, COARSE).any(axis=(1, 3))
        cost = np.where(known, 1.0, 1.3) + 3.0 * np.clip((radius + margin - dist) / margin, 0, 1)
        cost[dist < radius] = np.inf
        return cost

    def plan(self, start, goals, radius=0.35):                                                # path [k, 2] to the cheapest reachable goal
        cost = self.costs(radius)
        a = tuple(self.ij(start) // COARSE)
        c = cost.copy()
        c[max(a[0] - 3, 0):a[0] + 4, max(a[1] - 3, 0):a[1] + 4] = np.minimum(c[max(a[0] - 3, 0):a[0] + 4, max(a[1] - 3, 0):a[1] + 4], 2.0)  # leave the current footprint
        c[~np.isfinite(c)] = -1                                                               # skimage: negative = impassable
        mcp = MCP_Geometric(c)
        total, _ = mcp.find_costs([a])
        cells = [tuple(self.ij(g) // COARSE) for g in np.atleast_2d(goals)]
        cells = [g for g in cells if 0 <= g[0] < c.shape[0] and 0 <= g[1] < c.shape[1] and np.isfinite(total[g])]
        if not cells:
            return None
        g = min(cells, key=lambda x: total[x])
        route = np.array(mcp.traceback(g))
        return (route * COARSE + COARSE / 2 - N / 2) * RES                                   # cell centres in metres

    def travel_costs(self, start, radius=0.35):                                               # [N/COARSE, N/COARSE] cost-to-go from start (inf = unreachable)
        c = self.costs(radius)
        a = tuple(self.ij(start) // COARSE)
        c[max(a[0] - 3, 0):a[0] + 4, max(a[1] - 3, 0):a[1] + 4] = np.minimum(c[max(a[0] - 3, 0):a[0] + 4, max(a[1] - 3, 0):a[1] + 4], 2.0)
        c[~np.isfinite(c)] = -1
        total, _ = MCP_Geometric(c).find_costs([a])
        return total * RES * COARSE                                                            # ~metres

    def far_reachable(self, start, visited, limit=25.0):                                      # fallback goal: reachable observed ground farthest from every visited place
        travel = self.travel_costs(start)
        ground = self.ground.reshape(N // COARSE, COARSE, N // COARSE, COARSE).any(axis=(1, 3))
        ok = ground & np.isfinite(travel) & (travel < limit) & (travel > 2.0)
        if not ok.any():
            return None
        cells = np.argwhere(ok)
        xy = (cells * COARSE + COARSE / 2 - N / 2) * RES
        spread = np.min(np.linalg.norm(xy[:, None] - np.asarray(visited)[None], axis=2), axis=1)
        return xy[int(np.argmax(spread))]

    def viewpoint(self, start, visited, view_range=5.0, keep=None, min_frontier=12):          # next exploration goal: a frontier with the most unknown area per metre of travel;
                                                                                              # stays near `keep` while a frontier within 2 m of it scores >= half the best (no oscillation)
        travel = self.travel_costs(start)
        block = lambda a: a.reshape(N // COARSE, COARSE, N // COARSE, COARSE).any(axis=(1, 3))
        occ, searched = block(self.obstacles()), block(self.searched)
        unknown = (~searched & ~occ).astype(np.uint8)
        frontier = (searched & ~occ & (cv2.dilate(unknown, np.ones((3, 3), np.uint8)) > 0)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(frontier, np.ones((3, 3), np.uint8)), connectivity=8)
        big = np.isin(labels, [k for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] >= min_frontier])   # >= ~1 m of frontier, not a wall gap
        r = int(view_range / (RES * COARSE))
        disk = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.circle(disk, (r, r), r, 1.0, -1)
        gain = cv2.filter2D(unknown.astype(np.float32), -1, disk, borderType=cv2.BORDER_CONSTANT) * (RES * COARSE) ** 2   # m^2 unknown within view range
        ok = np.ascontiguousarray((frontier > 0) & big & np.isfinite(travel) & (travel > 2.0), dtype=np.uint8)   # MCP returns a non C-ordered array
        for v in visited:                                                                      # do not return to places already scanned
            c = self.ij(v) // COARSE
            cv2.circle(ok, (int(c[1]), int(c[0])), int(4.0 / (RES * COARSE)), 0, -1)
        ok = ok > 0
        if not ok.any():
            return None, 0.0
        score = np.where(ok, gain / (travel + 3.0) ** 1.5, -1)                                 # nearest-first: full_v14_308, 310 and v13_303 left the basket's strip for the open east lawn
        g = np.unravel_index(np.argmax(score), score.shape)
        if keep is not None:
            k = self.ij(keep) // COARSE
            r2 = int(2.0 / (RES * COARSE))
            lo = np.maximum(k - r2, 0)
            window = score[lo[0]:k[0] + r2 + 1, lo[1]:k[1] + r2 + 1]
            if window.size and window.max() >= 0.5 * score[g]:
                g = tuple(np.array(np.unravel_index(np.argmax(window), window.shape)) + lo)
        return (np.array(g) * COARSE + COARSE / 2 - N / 2) * RES, float(gain[g])
