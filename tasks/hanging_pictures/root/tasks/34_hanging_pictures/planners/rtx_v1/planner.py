"""rtx_v1 parent planner for hanging_pictures on native BEHAVIOR (RTX backend), left arm, gate-clean (numpy/scipy only).

Phases (log stage = index in PHASES): search, navigate, approach (centre the picture at GRASP_SPOT), raise_hand, settle_view,
pregrasp, descend, close, lift, carry (picture upright at CARRY_AT), find_nail, mount_navigate, attach (art out, hook toward
the nail hook, press, in-plane spiral), release, verify_mount. Picture: art blue (calibrated) plus an RGB-D plane fit in the
asset frame. Nail: dark dot on a vertical plane. Kinematics: JOINTS from r1pro.urdf (b0d76760). Input: the eight OBS_CONTRACT
keys. log() gives phase, stage, reason, counters and the estimated poses. Hangs the poster in some starts (DEV_LOG.md, REPORT.md).
"""
import heapq
import math

import numpy as np
from scipy.ndimage import find_objects, grey_closing, label, maximum_filter
from scipy.spatial.transform import Rotation as R

JOINTS = (
    (0, 'base_link', 'torso_link1', 'revolute', (-0.079032, 0.0, 0.34265), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (1, 'torso_link1', 'torso_link2', 'revolute', (0.0, 0.0, 0.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (2, 'torso_link2', 'torso_link3', 'revolute', (0.0, 0.0001005, 0.3), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    (3, 'torso_link3', 'torso_link4', 'revolute', (0.0, -0.00010015, 0.09962), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (-1, 'torso_link4', 'zed_link', 'fixed', (0.065511, 0.060003, 0.47598), (-1.9199, 0.0, -1.5708), (0.0, 0.0, 0.0)),
    (-1, 'torso_link4', 'left_arm_base_link', 'fixed', (-0.00048618, 0.097234, 0.30302), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (4, 'left_arm_base_link', 'left_arm_link1', 'revolute', (0.0, 0.0735, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (5, 'left_arm_link1', 'left_arm_link2', 'revolute', (0.025012, 0.081265, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (6, 'left_arm_link2', 'left_arm_link3', 'revolute', (-0.025012, 0.0, -0.1155), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (7, 'left_arm_link3', 'left_arm_link4', 'revolute', (0.0, 0.035065, -0.1945), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (8, 'left_arm_link4', 'left_arm_link5', 'revolute', (0.0, -0.035065, -0.095), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (9, 'left_arm_link5', 'left_arm_link6', 'revolute', (0.0, -0.028, -0.16305), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (10, 'left_arm_link6', 'left_arm_link7', 'revolute', (0.0295, 0.028, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (-1, 'left_arm_link7', 'left_gripper_link', 'fixed', (-0.0295, 0.0, -0.16065), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (12, 'left_gripper_link', 'left_gripper_finger_link2', 'prismatic', (8.9046e-05, -0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
)
LIMITS = ((-1.1345, -2.7925, -1.8326, -3.0543, -4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708), (1.8326, 2.5307, 1.5708, 3.0543, 1.309, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708))
GRIP_POINT = (0.0, 0.0, -0.060)  # tool centre in left_gripper_link
CAMERAS = (("left_wrist", "left_realsense_link", 388.6639 / 480), ("right_wrist", "right_realsense_link", 388.6639 / 480),
           ("head", "zed_link", .425))
PHASES = ("search", "navigate", "approach", "raise_hand", "settle_view", "pregrasp", "descend", "close", "lift", "carry", "find_nail",
          "mount_navigate", "attach", "release", "verify_mount")
REST = (.5, -.65, -.5, 0., -1.4, .35, .2, -.7, 0., .3, 0.)  # IK rest: torso 4, left arm 7
HALF_THICKNESS = .0035  # asset origin behind the visible face, m
HAND_RAISED = (.48, .43, 1.43)  # left hand above the counter, the baseline value
POSTER_SIZE = ((.25, .45), (.38, .70))  # accepted plane patch width, height, m
POSTER_H = .442  # frame height measured on simulator fits, metres
STANDOFF_PICTURE = (.65, .73, .81, .90)  # path goal distance to the picture face, m
CARRY_AT = (.45, .05, 1.10)  # picture centre while carrying, robot frame, m
PREGRASP_BACK, PREGRASP_UP = .16, .10  # pregrasp offset out of the edge and up, m
BASE_RADIUS, BASE_RADIUS_HOLDING = .276, .34  # depth map inflation free / holding, m
NAV_DIRECT = 1.3  # nearer to the wall standoff: straight drive
PATH_MISSES = 6  # failed replans before dropping the nail
CONTACT_SETTLE, SEARCH_STEPS, SEARCH_RADIUS, SEARCH_RATE = 500, 700, .05, .05  # press steps, spiral steps, radius m, rad/step
STANDOFF_WALL = (.66, .70, .74)  # base to nail along the normal (demos .73), m
HOOK_OUT, HOOK_SIDE, HOOK_UP, PRESS = .05, .04, -.005, .02  # hook target vs dark dot: out, picture y, up (measured press bias: -4 cm side, +2 cm up); press, m
APPROACH_FROM, APPROACH_RATE = .38, .0008  # start distance from the hook, m, speed m/step
ATTACH_READY = (.05, .25)  # arm error (m, rad) that starts the press
GRASP_SPOT = (.76, -.02)  # picture centre for the pinch (IK ok: x .68-.82, y<=.06)
APPROACH_MAX = .45  # approach travel limit, m
ART_BLUE_R, ART_BLUE_G = 45, 18  # art B-R 56-66, B-G 27-31; sofa B-R 35, lamp B-G 10
NEAR_FIT = .3  # farther fit = another blue object, m
ATTACH_LOCAL = (.006816, 0., .201414)  # picture hook in the asset frame, m
HOLDING = ("close", "lift", "carry", "find_nail", "mount_navigate", "attach")  # phases with the left gripper closed
LOOK_PICTURE = ("search", "navigate", "approach", "raise_hand", "settle_view", "pregrasp", "descend", "close", "lift")  # phases that refit the picture
STALL_GAIN, STALL_STEPS = .005, 90  # arm blocked: no STALL_GAIN m gain in STALL_STEPS
NAV_RETRY, SETTLE_WAIT, GRASP_PLAN_WAIT, LIFT_CONFIRM, ATTACH_REACH_WAIT = 1500, 300, 200, 240, 450  # steps before a retry or back-off
EDGE_Y, EDGE_Z = .142, .025  # pinch point from the centre: width, up, m
LIFT_BACK, LIFT_UP = .15, .10  # lift back toward the robot, up, m
FINGERS_OPEN = .08  # finger sum of an open gripper


def rot2(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def first(x):
    """Drop the leading batch axis of size 1 that the evaluator puts in front of every observation."""
    a = np.asarray(x)
    return a[0] if a.ndim >= 2 and a.shape[0] == 1 else a


def brief(t):
    """Position and rotation vector of a 4x4 pose, rounded, for log() (None stays None)."""
    return None if t is None else np.r_[t[:3, 3], R.from_matrix(t[:3, :3]).as_rotvec()].round(3).tolist()


def pose_matrix(rotation, position):
    t = np.eye(4); t[:3, :3], t[:3, 3] = rotation, position
    return t


# ---- image helpers (replace cv2) ------------------------------------------------------------------------
def resize_nearest(d, n):
    d = np.squeeze(d); h = d.shape[0]
    if h == n: return d
    idx = np.minimum(np.floor(np.arange(n) * (h / n)).astype(int), h - 1)
    return d[idx][:, idx]


def components(mask):
    """8-connected components: (count incl. background, labels, x y w h area)."""
    labels, n = label(mask, structure=np.ones((3, 3))); stats = np.zeros((n + 1, 5), int); stats[:, 4] = np.bincount(labels.ravel(), minlength=n + 1)
    for i, sl in enumerate(find_objects(labels), 1): stats[i, :4] = sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start
    return n + 1, labels, stats


def dilate(m, size):
    return maximum_filter(m.astype(np.uint8), size=size, mode="nearest") > 0


def make_cameras(frame, size=256):
    """Camera dicts (role -> rgb, depth, K, R, t) in the robot frame. size=None keeps the native resolution."""
    rel = frame["robot_r1::cam_rel_poses"].reshape(3, 7); cams = {}
    for i, (role, link, focal) in enumerate(CAMERAS):
        prefix = f"robot_r1::robot_r1:{link}:Camera:0::"; w = frame[prefix + "rgb"].shape[1]; width = w if size is None else size
        K = np.array([[focal * width, 0, width / 2], [0, focal * width, width / 2], [0, 0, 1.]]); rot = R.from_quat(rel[i, 3:]).as_matrix() @ np.diag([1, -1, -1])
        cams[role] = {"rgb": resize_nearest(frame[prefix + "rgb"][..., :3], width), "depth": resize_nearest(frame[prefix + "depth_linear"], width), "K": K, "R": rot, "t": rel[i, :3]}
    return cams


def pointcloud(cam, mask=None, stride=1):
    d = cam["depth"]; valid = np.isfinite(d) & (d > .06) & (d < 5)
    if mask is not None: valid &= mask
    y, x = np.where(valid); y, x = y[::stride], x[::stride]; z = d[y, x]; p = np.c_[(x + .5 - cam["K"][0, 2]) / cam["K"][0, 0] * z, (y + .5 - cam["K"][1, 2]) / cam["K"][1, 1] * z, z]
    return p @ cam["R"].T + cam["t"], cam["rgb"][y, x], np.c_[x, y]


# ---- kinematics (JOINTS table instead of the URDF) --------------------------------------------------------
class Kinematics:
    """FK and Jacobians, base->zed_link and base->left gripper. q: torso 4, left arm 7, 2 fingers."""
    def __init__(self):
        self.joints = []
        for dof, parent, child, kind, xyz, rpy, axis in JOINTS:
            t = np.eye(4); t[:3, 3] = xyz; t[:3, :3] = R.from_euler("xyz", rpy).as_matrix(); self.joints.append((dof, parent, child, kind, t, np.asarray(axis)))
    def forward(self, q):
        poses, axes, anc = {"base_link": np.eye(4)}, {}, {"base_link": []}
        for dof, parent, child, kind, origin, axis in self.joints:
            t = poses[parent] @ origin; motion = np.eye(4); v = q[dof] if dof >= 0 else 0.
            if kind == "revolute": motion[:3, :3] = R.from_rotvec(axis * v).as_matrix()
            elif kind == "prismatic": motion[:3, 3] = axis * v
            axes[dof] = (t[:3, 3], t[:3, :3] @ axis, kind); poses[child] = t @ motion; anc[child] = anc[parent] + ([dof] if dof >= 0 else [])
        return poses, axes, anc
    def jacobian(self, link, q, point=None, cache=None):
        poses, axes, anc = cache or self.forward(q); t = poses[link]; xyz = t[:3, 3] if point is None else t[:3, 3] + t[:3, :3] @ point; jac = np.zeros((6, len(q)))
        for dof in anc[link]:
            pos, axis, kind = axes[dof]
            if kind == "prismatic": jac[:3, dof] = axis
            else:
                jac[:3, dof] = np.cross(axis, xyz - pos); jac[3:, dof] = axis
        return jac


def solve_lsq(fun, x, lo, hi, max_nfev=90):
    """Levenberg-Marquardt on x = mid + half * sin(u): the bounds hold by construction, so a start on a joint limit can
    still move (a clipped LM step stalls there). Forward-difference Jacobian, stands in for scipy least_squares."""
    mid, half = (hi + lo) / 2, (hi - lo) / 2; u = np.arcsin(np.clip((x - mid) / half, -.98, .98)); g = lambda v: fun(mid + half * np.sin(v))
    r = g(u); cost, lam = r @ r, 1e-2
    for _ in range(max_nfev):
        J = np.empty((len(r), len(u)))
        for k in range(len(u)):
            uk = u.copy(); uk[k] += 1e-6; J[:, k] = (g(uk) - r) / 1e-6
        A, grad = J.T @ J, J.T @ r
        while lam < 1e12:
            un = u - np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-3 * np.diag(A).max() + 1e-9), grad); rn = g(un)  # floor: no huge step on a dead joint
            if rn @ rn < cost: break
            lam *= 4
        else:
            break
        done = cost - rn @ rn < 1e-12 * cost; u, r, cost, lam = un, rn, rn @ rn, max(lam / 3, 1e-9)
        if done: break
    return mid + half * np.sin(u)


class ArmServo:
    """Left arm plus torso: IK and bounded joint steps."""
    def __init__(self):
        self.kin = Kinematics(); self.limits = np.array(LIMITS, float); self.limits[:, 3] = [-.75, .75]; self.target = None; self.point = np.asarray(GRIP_POINT)
    def pose(self, q):
        t = self.kin.forward(q)[0]["left_gripper_link"].copy(); t[:3, 3] += t[:3, :3] @ self.point
        return t
    def solve(self, q, goal, rotation):
        """Joint vector (11) reaching the tool pose, or None. The nearest of several starts is returned."""
        lo, hi = self.limits; rest = np.asarray(REST); starts = [q[:11], rest, rest + np.r_[0, 0, 0, 0, 0, .5, -.8, 0, 0, 0, 0]]; choices = []
        def residual(x):
            trial = q.copy(); trial[:11] = x; t = self.pose(trial)
            return np.r_[5 * (t[:3, 3] - goal), R.from_matrix(rotation @ t[:3, :3].T).as_rotvec(), .008 * (x - rest)]
        for start in starts:
            x = solve_lsq(residual, np.clip(start, lo + 1e-5, hi - 1e-5), lo, hi); v = residual(x)
            if np.linalg.norm(v[:3]) / 5 < .015 and np.linalg.norm(v[3:6]) < .12: choices.append((np.linalg.norm(x - q[:11]), x))
        return min(choices, key=lambda c: c[0])[1] if choices else None
    def step(self, action, q, goal, rotation, plan=False, gaze_pitch=None):
        """Write the torso and left arm command into action, return (position error, orientation error)."""
        cache = self.kin.forward(q); t = cache[0]["left_gripper_link"]; point = t[:3, 3] + t[:3, :3] @ self.point; err = goal - point; ori = R.from_matrix(rotation @ t[:3, :3].T).as_rotvec()
        if plan and self.target is None: self.target = self.solve(q, goal, rotation)
        if self.target is not None:
            delta = self.target - q[:11]; dq = np.clip(delta, -.016, .016)
            if np.max(np.abs(delta)) < .035: self.target = None
        else:
            J = self.kin.jacobian("left_gripper_link", q, self.point, cache)[:, :11]; delta = np.r_[np.clip(err * .25, -.012, .012), np.clip(ori * .2, -.03, .03)]
            if gaze_pitch is not None:
                gaze = np.zeros(11); gaze[:3] = [1., 1., -1.]; pitch = gaze_pitch - (q[0] + q[1] - q[2]); J = np.vstack([J, .5 * gaze]); delta = np.r_[delta, np.clip(pitch * .10, -.012, .012) * .5]
            dq = np.clip(J.T @ np.linalg.solve(J @ J.T + .002 * np.eye(len(delta)), delta), -.016, .016)
        command = np.clip(q[:11] + dq, self.limits[0] + 1e-5, self.limits[1] - 1e-5); action[3:7] = command[:4]; action[7:14] = command[4:]
        return float(np.linalg.norm(err)), float(np.linalg.norm(ori))


# ---- perception: poster by static colour signature and RGB-D plane ----------------------------------------------
def poster_in_view(cam, role):
    """Framed poster pose in the robot frame in the asset convention (4x4, x away from the visible face, z up) or None.

    The art has a saturated blue (thresholds calibrated on simulator frames, see ART_BLUE_R/G). Candidate blobs are validated in 3D (height, size), the plane
    comes from the art points, the patch of the plane around them gives the face centre by its bounding box."""
    rgb = cam["rgb"].astype(np.int16); depth = cam["depth"]; art = (rgb[..., 2] > rgb[..., 0] + ART_BLUE_R) & (rgb[..., 2] > rgb[..., 1] + ART_BLUE_G) & (rgb[..., 2] > 60) & (rgb.max(-1) < 215) & np.isfinite(depth) & (depth > .06) & (depth < 3.5)
    if art.sum() < 20: return None
    n, labels, stats = components(dilate(art, max(5, cam["rgb"].shape[0] // 48)))
    best = None
    for i in np.argsort(np.bincount(labels[art], minlength=n))[::-1][:3]:
        if i == 0: continue
        blob = art & (labels == i)
        if blob.sum() < 20: continue
        pts, _, pix = pointcloud(cam, blob)
        centre = np.median(pts, axis=0); pts = pts[np.linalg.norm(pts - centre, axis=1) < .3]
        if len(pts) < 20 or not .45 < centre[2] < 1.9 or np.ptp(pts, axis=0).max() > .65: continue
        x0, y0, w, h = stats[i, :4]; pad = max(w, h); window = np.zeros(blob.shape, bool); window[max(0, y0 - pad):y0 + h + pad, max(0, x0 - pad):x0 + w + pad] = True
        origin = pts.mean(0); normal = np.linalg.svd(pts - origin, full_matrices=False)[2][-1]
        if normal @ (cam["t"] - origin) < 0: normal = -normal
        patch, _, _ = pointcloud(cam, window); patch = patch[(np.abs((patch - origin) @ normal) < .02) & (np.linalg.norm(patch - origin, axis=1) < .38)]
        up = np.array([0., 0., 1.]); zax = up - (up @ normal) * normal
        zax = unit(zax) if np.linalg.norm(zax) > .3 else unit(np.array([1., 0., 0.]) - normal[0] * normal)  # a flat picture: the far edge is "up"
        yax = np.cross(zax, normal); face, size = origin, None
        if len(patch) > 120:
            u, v = (patch - origin) @ yax, (patch - origin) @ zax; lo_u, hi_u, lo_v, hi_v = np.percentile(u, 1), np.percentile(u, 99), np.percentile(v, 1), np.percentile(v, 99); size = (hi_u - lo_u, hi_v - lo_v)
            if POSTER_SIZE[0][0] < size[0] < POSTER_SIZE[0][1] and POSTER_SIZE[1][0] < size[1] < POSTER_SIZE[1][1]: face = origin + yax * (lo_u + hi_u) / 2 + zax * (lo_v + POSTER_H / 2)  # from the bottom edge, the top may merge with the wall
            else: face = origin - zax * .0525  # the art centroid sits about 5 cm above the picture centre
        score = len(pts) / (1. + np.linalg.norm(centre - cam["t"]))
        if best is None or score > best[0]: best = (score, {"pose": pose_matrix(np.column_stack([-normal, -yax, zax]), face - normal * HALF_THICKNESS), "view": role, "art": int(blob.sum()), "size": None if size is None else (float(size[0]), float(size[1]))})
    return None if best is None else best[1]


# ---- perception: the wall nail, a small dark protrusion on a vertical plane (baseline MountDetector, numpy) ----------
def detect_nails(cam, height_range=(.65, 2.45)):
    """Nail candidates in the robot frame, best first: dict point, normal, score."""
    rgb = cam["rgb"]; gray = rgb.astype(np.float32) @ np.array([.299, .587, .114], np.float32); h, w = gray.shape
    mask = ((grey_closing(gray, size=(21, 21)) - gray > 25) & (gray < 125)); n, labels, stats = components(mask); found = []; K = cam["K"]
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if not 2 <= area <= 180 or max(ww, hh) > 22 or max(ww, hh) / max(min(ww, hh), 1) > 2.8 or x < 18 or y < 18 or x + ww > w - 18 or y + hh > h - 18: continue
        inside = labels[y:y + hh, x:x + ww] == i; vv, uu = np.nonzero(inside); u, v = uu.mean() + x, vv.mean() + y
        radius = int(max(12, min(25, max(ww, hh) * 2))); x0, x1, y0, y1 = max(0, int(u) - radius), min(w, int(u) + radius + 1), max(0, int(v) - radius), min(h, int(v) + radius + 1)
        yy, xx = np.indices((y1 - y0, x1 - x0)); dist = np.hypot(xx + x0 - u, yy + y0 - v); patch = np.zeros((h, w), bool); patch[y0:y1, x0:x1] = (dist > max(ww, hh) * .85 + 2) & (dist < radius)
        pts, _, _ = pointcloud(cam, patch, stride=3)
        if len(pts) < 45: continue
        centre = np.median(pts, axis=0); pts = pts[np.linalg.norm(pts - centre, axis=1) < .2]
        if len(pts) < 30: continue
        for _ in range(3):
            centre = pts.mean(0); normal = np.linalg.svd(pts - centre, full_matrices=False)[2][-1]; keep = np.abs((pts - centre) @ normal) < .008
            if keep.sum() < 25: break
            pts = pts[keep]
        if len(pts) < 25 or abs(normal[2]) > .25 or np.median(np.abs((pts - centre) @ normal)) > .004: continue
        if normal @ (cam["t"] - centre) < 0: normal = -normal
        ray = cam["R"] @ np.array([(u + .5 - K[0, 2]) / K[0, 0], (v + .5 - K[1, 2]) / K[1, 1], 1.]); den = normal @ ray
        if abs(den) < .2: continue
        z = (normal @ (centre - cam["t"])) / den; p = cam["t"] + z * ray; diameter = max(ww, hh) * z / K[0, 0]
        if not .007 < diameter < .05 or not height_range[0] < p[2] < height_range[1] or not .3 < z < 4.: continue
        around = gray[patch]; surrounding = float(np.median(around)); contrast = surrounding - float(np.median(gray[y:y + hh, x:x + ww][inside]))
        if contrast < 35 or float(np.std(around)) > 15 or surrounding < 115: continue
        local = rgb[y:y + hh, x:x + ww][inside].astype(float)
        if np.median(local.max(1) - local.min(1)) > 60: continue
        found.append({"point": p + normal * .012, "normal": normal, "score": contrast - 300 * abs(diameter - .019)})
    return sorted(found, key=lambda c: c["score"], reverse=True)


class NailTracker:
    """Nail candidates tracked in the odometry frame (merged when closer than 8 cm with a similar normal)."""
    def __init__(self):
        self.tracks, self.latest = [], []
    def update(self, cam, base, step, held=None):
        self.latest = []
        for c in detect_nails(cam):
            if held is not None and np.all(np.abs(held[:3, :3].T @ (c["point"] - held[:3, 3])) < np.array([.055, .20, .28])): continue
            p, n = base[:3, :3] @ c["point"] + base[:3, 3], base[:3, :3] @ c["normal"]; self.latest.append({"point": p, "normal": n, "score": c["score"], "step": step})
            near = [t for t in self.tracks if np.linalg.norm(t["point"] - p) < .08 and t["normal"] @ n > .9]
            if near:
                t = min(near, key=lambda t: np.linalg.norm(t["point"] - p)); t["point"] = .7 * t["point"] + .3 * p; t["normal"] = unit(.7 * t["normal"] + .3 * n)
                t["count"] += int(step - t["last"] >= 8); t["last"] = step; t["score"] = .7 * t["score"] + .3 * c["score"]
            else: self.tracks.append({"point": p, "normal": n, "count": 1, "last": step, "score": c["score"]})
        self.tracks = [t for t in self.tracks if step - t["last"] < 1200]
    def best(self, origin):
        good = [t for t in self.tracks if t["count"] >= 3 and 1.15 < t["point"][2] < 2.15]
        return max(good, key=lambda t: t["score"] + min(t["count"], 6) * 5 - 5 * np.linalg.norm(t["point"][:2] - origin[:2])) if good else None


# ---- navigation: local depth occupancy map ------------------------------------------------------------------
class DepthMap:
    def __init__(self):
        self.res, self.n, self.radius = .06, 160, .276; self.hits = np.zeros((160, 160), np.uint16)
    def ij(self, xy):
        return np.floor(np.asarray(xy) / self.res + self.n / 2).astype(int)
    def xy(self, ij):
        return (np.asarray(ij) - self.n / 2 + .5) * self.res
    def update(self, cams, odom):
        p, _, _ = pointcloud(cams["head"], stride=5); keep = (p[:, 2] > .10) & (p[:, 2] < 1.5) & (np.linalg.norm(p[:, :2], axis=1) > self.radius + .2); ij = self.ij(p[keep][:, :2] @ rot2(odom[2]).T + odom[:2])
        ij = ij[np.all((ij >= 0) & (ij < self.n), axis=1)]; np.add.at(self.hits, (ij[:, 0], ij[:, 1]), 1)
    def occupancy(self):
        cells = self.radius / self.res; k = np.arange(-6, 7); xx, yy = np.meshgrid(k, k)
        return maximum_filter((self.hits >= 2).astype(np.uint8), footprint=xx * xx + yy * yy <= cells * cells, mode="constant") > 0
    def path(self, start, goal):
        occ = self.occupancy(); a, b = self.ij(start), self.ij(goal)
        if not np.all((b >= 0) & (b < self.n)) or occ[tuple(b)]: return None
        xs, ys = np.ogrid[:self.n, :self.n]; occ[(xs - a[0]) ** 2 + (ys - a[1]) ** 2 < 4 ** 2] = False; a, b = tuple(a), tuple(b); queue, cost, prev, expanded = [(0., 0., a)], {a: 0.}, {}, 0
        while queue and expanded < 16000:
            expanded += 1; _, g, u = heapq.heappop(queue)
            if g > cost[u] + 1e-6: continue
            if u == b:
                out = [u]
                while u != a:
                    u = prev[u]; out.append(u)
                return [self.xy(p) for p in out[::-1]]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < self.n and 0 <= v[1] < self.n) or occ[v]: continue
                if dx and dy and (occ[u[0] + dx, u[1]] or occ[u[0], u[1] + dy]): continue
                ng = g + math.hypot(dx, dy)
                if ng < cost.get(v, float("inf")):
                    cost[v], prev[v] = ng, u; heapq.heappush(queue, (ng + math.hypot(v[0] - b[0], v[1] - b[1]), ng, v))
        return None


def path_length(path):
    return sum(float(np.linalg.norm(b - a)) for a, b in zip(path, path[1:]))


# ---- policy ---------------------------------------------------------------------------------------------
class Planner:
    """One method per phase (phase_<name>); every transition goes through change()."""
    def reset(self, task_spec):
        self.servo, self.map, self.nails = ArmServo(), DepthMap(), NailTracker(); self.phase, self.reason, self.age, self.step = "search", "start", 0, 0
        self.odom = np.zeros(3); self.action = self.object = self.nav = self.heading = self.path = self.grasp = self.grip_object = self.mount = None
        self.last_seen, self.path_misses, self.pregrasp_ready, self.contact_age, self.attach_ready, self.best_err, self.best_age, self.fingers = -1000, 0, False, 0, False, 9., 0, 0.
        self.counts = {"grasp_tries": 0, "approach_fails": 0, "nails_seen": 0, "wall_retries": 0}; self.events, self.fit_info, self.debug = [], {}, {}
    def log(self):
        inv = np.linalg.inv(self.base())
        return {"stage": PHASES.index(self.phase), "phase": self.phase, "reason": self.reason, "step": self.step, "age": self.age, "odom": self.odom.round(3).tolist(),
                "counts": self.counts, "seen_ago": self.step - self.last_seen, "fit": self.fit_info, "debug": self.debug, "events": self.events[-40:],
                "picture": None if self.object is None else brief(inv @ self.object), "held": brief(self.grip_object),
                "nail": None if self.mount is None else np.r_[(inv @ np.r_[self.mount["point"], 1])[:3], inv[:3, :3] @ self.mount["normal"]].round(3).tolist(),
                "nav": None if self.nav is None else np.r_[rot2(self.odom[2]).T @ (self.nav - self.odom[:2]), wrap(self.heading - self.odom[2])].round(3).tolist()}
    def change(self, phase, reason):
        self.events = self.events[-39:] + [f"{self.step}:{self.phase}>{phase}:{reason}"]; self.phase, self.reason, self.age = phase, reason, 0; self.servo.target = None; self.best_err, self.best_age = 9., 0
    def stalled(self, err):
        """Arm blocked or at its reach: no STALL_GAIN gain in STALL_STEPS."""
        if err < self.best_err - STALL_GAIN: self.best_err, self.best_age = err, self.age
        return self.age - self.best_age > STALL_STEPS
    def base(self):
        return pose_matrix(R.from_euler("z", self.odom[2]).as_matrix(), np.r_[self.odom[:2], 0.])
    def drive(self, action, speed, gain_turn):
        """Follow self.path (odometry frame) toward self.heading with the planar base."""
        near = int(np.argmin([np.linalg.norm(p - self.odom[:2]) for p in self.path])); waypoint = self.path[min(near + 3, len(self.path) - 1)]
        action[:2] = np.clip(rot2(self.odom[2]).T @ (waypoint - self.odom[:2]) * speed * 4., -speed, speed); action[2] = np.clip(wrap(self.heading - self.odom[2]) * gain_turn, -.35, .35)
    def plan_navigation(self):
        """Path goal STANDOFF_PICTURE in front of the picture face."""
        centre = self.object[:2, 3]; normal = unit(-self.object[:2, 0]); choices = []  # asset x points away from the art
        for distance in STANDOFF_PICTURE:
            for angle in np.linspace(-1.2, 1.2, 11):
                direction = rot2(angle) @ normal; goal = centre + distance * direction; path = self.map.path(self.odom[:2], goal)
                if path: choices.append((path_length(path) + .3 * abs(angle) + distance, goal, math.atan2(-direction[1], -direction[0]), path))
        if not choices: return False
        _, self.nav, self.heading, self.path = min(choices, key=lambda c: c[0]); return True
    def plan_grasp(self, q, inverse):
        """Side-edge pinch: both edges and wrist flips, pregrasp IK nearest to q."""
        obj = inverse @ self.object; best = None
        for edge_sign in (1, -1):
            closing = obj[:3, 0]; forward = edge_sign * obj[:3, 1]; rotation = np.column_stack([np.cross(closing, forward), closing, forward]); grip = obj[:3, 3] + obj[:3, :3] @ np.array([.0018, EDGE_Y * edge_sign, EDGE_Z])
            for flip in (1, -1):
                cand = rotation @ np.diag([flip, flip, 1.]); sol = self.servo.solve(q, grip + forward * PREGRASP_BACK + np.array([0., 0., PREGRASP_UP]), cand)
                if sol is not None and (best is None or np.linalg.norm(sol - q[:11]) < best[0]): best = (np.linalg.norm(sol - q[:11]), grip, cand, obj[2, 3], sol)
        return None if best is None else {"point": best[1], "rotation": best[2], "z0": best[3], "joints": best[4]}
    def read_poster(self, cams, base):
        """Refit the picture from the head (far) or the left wrist (near) and store it in the odometry frame."""
        for role in (["head"] if self.phase in ("search", "navigate", "approach") else ["left_wrist", "head"]):
            fit = poster_in_view(cams[role], role)
            if fit is None: continue
            whole = fit["size"] is not None and POSTER_SIZE[0][0] < fit["size"][0] < POSTER_SIZE[0][1] and POSTER_SIZE[1][0] < fit["size"][1] < POSTER_SIZE[1][1]
            near = self.object is not None and np.linalg.norm((base @ fit["pose"])[:3, 3] - self.object[:3, 3]) < NEAR_FIT
            if whole and (role == "head" or near or self.object is None):  # pose only from a whole frame
                self.object = base @ fit["pose"]; self.last_seen = self.step; self.fit_info = {"view": role, "art": fit["art"], "size": fit["size"], "step": self.step}
                return
            if near: self.last_seen = self.step; return  # partial view: seen, pose kept
    def act(self, obs):
        # A closed list of the eight contract keys; nothing else can enter the control law.
        frame = {
            "robot_r1::proprio": first(obs["robot_r1::proprio"]),
            "robot_r1::cam_rel_poses": first(obs["robot_r1::cam_rel_poses"]),
            "robot_r1::robot_r1:zed_link:Camera:0::rgb": first(obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"]),
            "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": first(obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"]),
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": first(obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"]),
            "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear": first(obs["robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear"]),
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": first(obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"]),
            "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear": first(obs["robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear"]),
        }
        s = frame["robot_r1::proprio"]; self.s, self.q = s, np.r_[s[53:57], s[3:10], s[24:26]].astype(float); self.step += 1; self.age += 1
        if self.step > 1 and (self.action is None or np.abs(self.action[:3]).max() > 0):  # a still base: no velocity noise
            self.odom[:2] += rot2(self.odom[2]) @ s[:2] / 30; self.odom[2] += s[2] / 30
        if self.action is None: self.action = np.r_[np.zeros(3), s[53:57], s[3:10], 1., s[28:35], 1.]
        action = self.action.copy(); action[:3] = 0; action[22] = 1.; action[14] = -1. if self.phase in HOLDING else 1.
        base = self.base(); self.inv = np.linalg.inv(base)
        if self.step % 12 == 1 and self.phase in LOOK_PICTURE: self.read_poster(make_cameras(frame, size=None), base)
        if self.step % 15 == 1:
            self.map.radius = BASE_RADIUS_HOLDING if self.grip_object is not None else BASE_RADIUS; self.map.update(make_cameras(frame, size=256), self.odom)
        if self.step % 12 == 1 and self.phase in ("find_nail", "mount_navigate", "attach"):
            held = None if self.grip_object is None else base @ self.servo.pose(self.q) @ self.grip_object; self.nails.update(make_cameras(frame, size=None)["head"], base, self.step, held)
        error = {"search": self.phase_search, "navigate": self.phase_navigate, "approach": self.phase_approach, "raise_hand": self.phase_raise_hand,
                 "settle_view": self.phase_settle_view, "pregrasp": self.phase_grasp, "descend": self.phase_grasp, "close": self.phase_close, "lift": self.phase_lift,
                 "carry": self.phase_carry, "find_nail": self.phase_find_nail, "mount_navigate": self.phase_mount_navigate, "attach": self.phase_attach,
                 "release": self.phase_release, "verify_mount": self.phase_verify_mount}[self.phase](action)
        if error is not None: self.debug["err"] = [round(error[0], 3), round(error[1], 3)]
        self.action = action.copy(); assert np.isfinite(action).all()
        return action.astype(np.float32)
    def phase_search(self, action):
        if self.object is not None and self.step - self.last_seen < 20 and self.plan_navigation(): self.change("navigate", "picture seen and a standoff path exists")
        else:
            action[2] = .32
            if self.age > 600 and self.age % 200 < 60: action[0] = .15  # new viewpoint
    def phase_navigate(self, action):
        if self.age % 90 == 1 and self.object is not None: self.plan_navigation()
        self.drive(action, .22, .8)
        if np.linalg.norm(self.nav - self.odom[:2]) < .045 and abs(wrap(self.heading - self.odom[2])) < .14: self.approach_start = self.odom[:2].copy(); self.change("approach", "standoff pose reached")
        elif self.age > NAV_RETRY: self.change("search", "standoff not reached, replanning from a new search")
    def phase_approach(self, action):
        rel = (self.inv @ self.object)[:3, 3]; ex, ey = rel[0] - GRASP_SPOT[0], rel[1] - GRASP_SPOT[1]  # centre the picture, keep facing it
        action[0], action[1], action[2] = np.clip(ex * .8, -.10, .10), np.clip(ey * .8, -.10, .10), np.clip(math.atan2(rel[1], rel[0]) * .6, -.2, .2)
        if abs(ex) < .03 and abs(ey) < .03: self.change("raise_hand", "picture at the grasp spot in the robot frame")
        elif np.linalg.norm(self.odom[:2] - self.approach_start) > APPROACH_MAX: self.counts["approach_fails"] += 1; self.change("search", "approach moved too far, the standoff was wrong")
    def phase_raise_hand(self, action):
        error = self.servo.step(action, self.q, np.asarray(HAND_RAISED), np.eye(3), plan=self.age == 1)
        if error[0] < .035 and error[1] < .2: self.change("settle_view", "left hand raised")
        elif self.stalled(error[0]): self.change("settle_view", "raise stalled short of the pose, the pregrasp IK plans from here")
        return error
    def phase_settle_view(self, action):
        if self.age > 36 and self.step - self.last_seen < 15: self.change("pregrasp", "fresh picture fit after settling")
        elif self.age > SETTLE_WAIT: self.counts["grasp_tries"] += 1; self.change("search", "picture not seen after raising the hand")
    def phase_grasp(self, action):
        """pregrasp: joint move to the IK-checked pose, then beside the edge; descend: onto the edge."""
        if self.grasp is None and self.object is not None and self.step == self.last_seen:  # fresh fits only, each plan is up to 4 IK solves
            self.grasp = self.plan_grasp(self.q, self.inv); self.debug["grasp_feasible"] = self.grasp is not None
            if self.grasp is not None: self.servo.target = self.grasp["joints"]
        if self.grasp is None:
            if self.age > GRASP_PLAN_WAIT: self.counts["grasp_tries"] += 1; self.change("search", "no feasible grasp for the fitted picture")
            return None
        goal = self.grasp["point"].copy(); rotation = self.grasp["rotation"]
        if self.phase == "pregrasp":
            goal += rotation[:, 2] * PREGRASP_BACK
            if not self.pregrasp_ready: goal[2] += PREGRASP_UP
        error = self.servo.step(action, self.q, goal, rotation)
        if self.phase == "pregrasp" and error[0] < .045 and error[1] < .40:
            if not self.pregrasp_ready: self.pregrasp_ready = True; self.best_err = 9.
            else: self.change("descend", "pregrasp pose reached")
        elif self.phase == "descend" and error[0] < .018 and error[1] < .22: self.change("close", "grasp point reached")
        elif self.stalled(error[0] if self.servo.target is None else float(np.max(np.abs(self.servo.target - self.q[:11])))): self.release_grasp(f"{self.phase} blocked at error {error[0]:.3f} m")
        return error
    def phase_close(self, action):
        fingers = float(self.s[24:26].sum()); moving = abs(fingers - self.fingers) > 1e-4; self.fingers = fingers
        if self.age > 10 and not moving:  # fingers stopped; lift tells if on the frame
            self.close_pose = self.servo.pose(self.q).copy(); d = -self.close_pose[:3, 3].copy(); d[2] = 0
            self.lift_goal = self.close_pose[:3, 3] + unit(d) * LIFT_BACK + np.array([0., 0., LIFT_UP]); self.grip_object = np.linalg.inv(self.base() @ self.close_pose) @ self.object
            self.change("lift", f"fingers settled at {fingers:.3f}")
    def phase_lift(self, action):
        error = self.servo.step(action, self.q, self.lift_goal, self.close_pose[:3, :3])
        if error[0] < .06 or self.stalled(error[0]):
            measured = None if self.object is None or self.step - self.last_seen > 36 else (self.inv @ self.object)[2, 3]; moved = np.linalg.norm(self.servo.pose(self.q)[:3, 3] - self.close_pose[:3, 3])
            if (measured is not None and measured - self.grasp["z0"] > .06) or (float(self.s[24:26].sum()) > .004 and moved > .06 and self.age > 90): self.change("carry", "picture rose with the hand")
            elif self.age > LIFT_CONFIRM: self.release_grasp("lifted hand but the picture did not rise")
        return error
    def phase_carry(self, action):
        goal = pose_matrix(np.eye(3), np.asarray(CARRY_AT)) @ np.linalg.inv(self.grip_object)  # picture upright, art toward the robot, as it will hang
        error = self.servo.step(action, self.q, goal[:3, 3], goal[:3, :3], plan=self.age == 1, gaze_pitch=-.4)
        if error[0] < .06 and abs(self.s[53] + self.s[54] - self.s[55] + .4) < .12: self.change("find_nail", "picture held in front, head tilted to the wall")
        elif self.servo.target is None and self.stalled(error[0]): self.change("find_nail", f"carry pose blocked at error {error[0]:.3f} m, searching as held")
        return error
    def phase_find_nail(self, action):
        candidate = self.nails.best(self.odom)
        if candidate is None:
            action[2] = .30
            if self.age > 600 and self.age % 200 < 60: action[0] = .12  # new viewpoint
            return None
        self.counts["nails_seen"] += 1; self.mount = {"point": candidate["point"].copy(), "normal": candidate["normal"].copy()}; options = []; normal = unit(self.mount["normal"][:2])
        for radius in STANDOFF_WALL:
            for turn in (0., -.15, .15):  # the demonstrations face the wall squarely
                direction = rot2(turn) @ normal; goal = self.mount["point"][:2] + direction * radius; path = self.map.path(self.odom[:2], goal)
                if path: options.append((path_length(path), goal, math.atan2(-direction[1], -direction[0]), path))
        if options:
            _, self.nav, self.heading, self.path = min(options, key=lambda o: o[0]); self.nav_offset = self.nav - self.mount["point"][:2]; self.change("mount_navigate", "nail tracked and a path to its wall exists")
        else: self.mount = None; action[2] = .20
    def phase_mount_navigate(self, action):
        if np.linalg.norm(self.nav - self.odom[:2]) < NAV_DIRECT: self.path, self.path_misses = [self.nav], 0  # the last stretch is straight: the wall fills the map
        elif self.age % 60 == 1:
            path = self.map.path(self.odom[:2], self.nav)
            if path: self.path, self.path_misses = path, 0
            else: self.path_misses += 1  # keep the old path for a while, the map flickers near walls
            if self.path_misses > PATH_MISSES: self.mount = None; self.change("find_nail", "path to the nail standoff vanished"); return None
        if self.refresh_mount(): self.nav = self.mount["point"][:2] + self.nav_offset
        self.drive(action, .18, .7)
        if np.linalg.norm(self.nav - self.odom[:2]) < .06 and abs(wrap(self.heading - self.odom[2])) < .15: self.change("attach", "wall standoff reached")
    def phase_attach(self, action):
        """Art out, picture hook toward the nail hook, move in, press, spiral in-plane."""
        self.refresh_mount(); normal = self.mount["normal"].copy(); normal[2] = 0; normal = unit(normal); up = np.array([0., 0., 1.])
        out = -normal; rot = np.column_stack([out, np.cross(up, out), up])  # asset x into the wall, the art faces the room
        approach = max(HOOK_OUT - PRESS, APPROACH_FROM - self.contact_age * APPROACH_RATE)
        if self.attach_ready: self.contact_age += 1
        picture = pose_matrix(rot, self.mount["point"] + normal * approach + rot @ (self.search_offset() + np.array([0., HOOK_SIDE, HOOK_UP]) - np.asarray(ATTACH_LOCAL)))
        goal = self.inv @ picture @ np.linalg.inv(self.grip_object); error = self.servo.step(action, self.q, goal[:3, 3], goal[:3, :3], plan=self.age % 150 == 1 and not self.attach_ready)
        if not self.attach_ready and error[0] < ATTACH_READY[0] and error[1] < ATTACH_READY[1]: self.attach_ready = True; self.events.append(f"{self.step}:attach_ready")
        self.debug.update(contact_age=self.contact_age)
        if self.contact_age > CONTACT_SETTLE + SEARCH_STEPS: self.change("release", "spiral search at the wall finished")
        elif not self.attach_ready and self.age > ATTACH_REACH_WAIT: self.counts["wall_retries"] += 1; self.mount = None; self.change("find_nail", "arm cannot reach the hook pose from this stance")
        return error
    def phase_release(self, action):
        if float(self.s[24:26].sum()) > FINGERS_OPEN: self.change("verify_mount", "gripper open")
    def phase_verify_mount(self, action):
        action[0] = -.10 if self.age < 90 else 0.  # back away; the evaluator decides success
    def search_offset(self):
        """In-plane spiral (picture y, z) once pressed to the wall: the hook only catches within 5 cm of the nail."""
        t = max(0, self.contact_age - CONTACT_SETTLE); a = min(SEARCH_RADIUS, t * SEARCH_RADIUS / SEARCH_STEPS)
        return np.array([0., a * math.cos(t * SEARCH_RATE), a * math.sin(t * SEARCH_RATE)])
    def refresh_mount(self):
        """Pull the nail toward a fresh sighting: removes odometry drift."""
        fresh = [c for c in self.nails.latest if c["step"] == self.step and np.linalg.norm(c["point"] - self.mount["point"]) < .3 and c["normal"] @ self.mount["normal"] > .8]
        if not fresh: return False
        c = max(fresh, key=lambda c: c["score"]); self.mount["point"] = .5 * self.mount["point"] + .5 * c["point"]; self.mount["normal"] = unit(.5 * self.mount["normal"] + .5 * c["normal"]); self.debug["nail_seen"] = self.step
        return True
    def release_grasp(self, reason):
        """Open the gripper and look for the picture again."""
        self.counts["grasp_tries"] += 1; self.grasp = self.grip_object = None; self.pregrasp_ready = False; self.object = None; self.change("search", reason)


def entrypoint():
    return Planner()
