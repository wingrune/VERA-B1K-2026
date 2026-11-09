"""broccolini_v2: cook_broccolini (task 96). Moves the ingredients from the plate into the frying pan one by one with the left hand.

Pipeline: search (spin, flat white disc at table height in the head cloud) -> navigate (A* on a depth occupancy map) -> raise_hand -> settle_view (food
blobs that stand above the plate bottom inside the plate disc) -> pregrasp -> descend -> close (pinch across the item) -> lift -> find_pan (flat dark disc)
-> pan_navigate -> over_pan -> release -> back to the plate for the next item. The flat 2 cm plate itself cannot be held by a rim pinch (broc_v1 jobs
grasp1-5: it tilts and slips), so v2 picks the food instead of pouring. Primitives (kinematics, ArmServo, DepthMap, GroundFrame) are the modem_v11 ones.
Input: the eight keys of OBS_CONTRACT.md. Output: float32 (23,). Odometry is the proprio base velocity (yaw from the command). Not yet: pan to the stove, switch.
Every phase moves on only from what it observes, a stalled phase retries or backs off, log() gives phase, stage, reason and counters.
"""
import heapq
import math

import numpy as np
from scipy.ndimage import binary_dilation, find_objects, label, maximum_filter
from scipy.spatial.transform import Rotation as R
JOINTS = (
    (0, 'base_link', 'torso_link1', 'revolute', (-0.079032, 0.0, 0.34265), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (1, 'torso_link1', 'torso_link2', 'revolute', (0.0, 0.0, 0.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (2, 'torso_link2', 'torso_link3', 'revolute', (0.0, 0.0001005, 0.3), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)), (3, 'torso_link3', 'torso_link4', 'revolute', (0.0, -0.00010015, 0.09962), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (-1, 'torso_link4', 'zed_link', 'fixed', (0.065511, 0.060003, 0.47598), (-1.9199, 0.0, -1.5708), (0.0, 0.0, 0.0)), (-1, 'torso_link4', 'left_arm_base_link', 'fixed', (-0.00048618, 0.097234, 0.30302), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (4, 'left_arm_base_link', 'left_arm_link1', 'revolute', (0.0, 0.0735, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (5, 'left_arm_link1', 'left_arm_link2', 'revolute', (0.025012, 0.081265, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (6, 'left_arm_link2', 'left_arm_link3', 'revolute', (-0.025012, 0.0, -0.1155), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (7, 'left_arm_link3', 'left_arm_link4', 'revolute', (0.0, 0.035065, -0.1945), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (8, 'left_arm_link4', 'left_arm_link5', 'revolute', (0.0, -0.035065, -0.095), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (9, 'left_arm_link5', 'left_arm_link6', 'revolute', (0.0, -0.028, -0.16305), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (10, 'left_arm_link6', 'left_arm_link7', 'revolute', (0.0295, 0.028, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (-1, 'left_arm_link7', 'left_gripper_link', 'fixed', (-0.0295, 0.0, -0.16065), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (11, 'left_gripper_link', 'left_gripper_finger_link1', 'prismatic', (-8.8709e-05, 0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (12, 'left_gripper_link', 'left_gripper_finger_link2', 'prismatic', (8.9046e-05, -0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
    (-1, 'torso_link4', 'right_arm_base_link', 'fixed', (-0.00048706, -0.097236, 0.30302), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (13, 'right_arm_base_link', 'right_arm_link1', 'revolute', (0.0, -0.0735, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (14, 'right_arm_link1', 'right_arm_link2', 'revolute', (0.023988, -0.081265, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (15, 'right_arm_link2', 'right_arm_link3', 'revolute', (-0.023988, 0.0, -0.1155), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (16, 'right_arm_link3', 'right_arm_link4', 'revolute', (0.0, 0.035058, -0.1945), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (17, 'right_arm_link4', 'right_arm_link5', 'revolute', (0.0, -0.035058, -0.095), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (18, 'right_arm_link5', 'right_arm_link6', 'revolute', (0.0, 0.028001, -0.16305), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (19, 'right_arm_link6', 'right_arm_link7', 'revolute', (0.0295, -0.028001, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (-1, 'right_arm_link7', 'right_gripper_link', 'fixed', (-0.0295, 0.0, -0.16065), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (20, 'right_gripper_link', 'right_gripper_finger_link1', 'prismatic', (-8.8709e-05, 0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (21, 'right_gripper_link', 'right_gripper_finger_link2', 'prismatic', (8.9046e-05, -0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
)
LIMITS = ((-1.1344999074935913, -2.7924997806549072, -1.8325998783111572, -3.054299831390381, -4.4506001472473145, -0.1745000034570694, -2.3561956882476807, -2.0943996906280518, -2.3561956882476807, -1.0471980571746826, -1.5707999467849731), (1.8325998783111572, 2.5306997299194336, 1.5707999467849731, 3.054299831390381, 1.309000015258789, 3.1415998935699463, 2.3561956882476807, 0.3490999639034271, 2.3561956882476807, 1.0471980571746826, 1.5707999467849731))
TIP_LINK = 'left_gripper_finger_link2'; TIP_POINT = (-8.915719809010625e-05, 0.015747327357530594, -0.03796372190117836)

B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

CAMERAS = (("left_wrist", "left_realsense_link", 388.6639 / 480), ("right_wrist", "right_realsense_link", 388.6639 / 480), ("head", "zed_link", .425))
def rot2(yaw): return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

def wrap(x): return (x + math.pi) % (2 * math.pi) - math.pi

def first(x):
    """Drop the leading batch axis of size 1 that the evaluator puts in front of every observation."""
    a = np.asarray(x)
    return a[0] if a.ndim >= 2 and a.shape[0] == 1 else a

def resize_nearest(d, n):
    d = np.squeeze(d); h = d.shape[0]
    if h == n: return d
    idx = np.minimum(np.floor(np.arange(n) * (h / n)).astype(int), h - 1)
    return d[idx][:, idx]

def resize_area(im, n):
    """Box-overlap (INTER_AREA) resize of a square HxWx3 uint8 image to n x n."""
    h = im.shape[0]
    if h == n: return im
    s = h / n; lo = (np.arange(n) * s)[:, None]; ov = np.clip(np.minimum(lo + s, np.arange(h)[None] + 1.) - np.maximum(lo, np.arange(h)[None]), 0, None); w = (ov / s).astype(np.float32)
    t = np.tensordot(w, im.astype(np.float32), axes=(1, 0)); out = np.tensordot(t, w, axes=(1, 1)).transpose(0, 2, 1)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)

def components(mask):
    """8-connected components like cv2.connectedComponentsWithStats: (count incl. background, labels, x y w h area)."""
    labels, n = label(mask, structure=np.ones((3, 3))); stats = np.zeros((n + 1, 5), int); stats[:, 4] = np.bincount(labels.ravel(), minlength=n + 1)
    for i, sl in enumerate(find_objects(labels), 1): stats[i, :4] = sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start
    return n + 1, labels, stats

def dilate(m, size): return maximum_filter(m, size=size, mode="nearest")

def red_mask(rgb): a = rgb.astype(float); return (a[..., 0] > 85) & (a[..., 0] > 2.5 * a[..., 1] + 20) & (a[..., 0] > 2.5 * a[..., 2] + 20)

def blue_mask(rgb): a = rgb.astype(float); return (a[..., 2] > 50) & (a[..., 2] > 1.25 * a[..., 0]) & (a[..., 2] > 1.15 * a[..., 1])
def make_cameras(frame, size=256, only=None):
    """Camera dicts (role -> rgb, depth, K, R, t) in the robot frame. size None keeps the native resolution; only restricts to one role."""
    rel = frame["robot_r1::cam_rel_poses"].reshape(3, 7); cams = {}
    for i, (role, link, focal) in enumerate(CAMERAS):
        if only is not None and role != only: continue
        prefix = f"robot_r1::robot_r1:{link}:Camera:0::"; w = frame[prefix + "rgb"].shape[1]; width = w if size is None else size
        K = np.array([[focal * width, 0, width / 2], [0, focal * width, width / 2], [0, 0, 1.]]); rot = R.from_quat(rel[i, 3:]).as_matrix() @ np.diag([1, -1, -1])
        cams[role] = {"rgb": resize_area(frame[prefix + "rgb"][..., :3], width), "depth": resize_nearest(frame[prefix + "depth_linear"], width), "K": K, "R": rot, "t": rel[i, :3]}
    return cams

def pointcloud(cam, mask=None, stride=1):
    d = cam["depth"]; valid = np.isfinite(d) & (d > .06) & (d < 5)
    if mask is not None: valid &= mask
    y, x = np.where(valid); y, x = y[::stride], x[::stride]; z = d[y, x]; p = np.c_[(x + .5 - cam["K"][0, 2]) / cam["K"][0, 0] * z, (y + .5 - cam["K"][1, 2]) / cam["K"][1, 1] * z, z]
    return p @ cam["R"].T + cam["t"], cam["rgb"][y, x], np.c_[x, y]

def level_cameras(cams, T):
    """Cameras re-expressed in the gravity-aligned frame on the ground (native ground_frame.level_cameras)."""
    return {k: dict(c, R=T[:3, :3] @ c["R"], t=T[:3, :3] @ c["t"] + T[:3, 3]) for k, c in cams.items()}

def from_proprio(s):
    """q: torso 4, left arm 7, left fingers 2, right arm 7, right fingers 2."""
    return np.r_[s[53:57], s[3:10], s[24:26], s[28:35], s[49:51]].astype(float)

class Kinematics:
    """FK and geometric Jacobians over the JOINTS table (base_link root)."""
    def __init__(self):
        self.joints = []
        for dof, parent, child, kind, xyz, rpy, axis in JOINTS:
            t = np.eye(4); t[:3, 3] = xyz; t[:3, :3] = R.from_euler("xyz", rpy).as_matrix(); self.joints.append((dof, parent, child, kind, t, np.asarray(axis)))
    def forward(self, q):
        poses, axes, anc = {"base_link": np.eye(4)}, {}, {"base_link": []}
        for dof, parent, child, kind, origin, axis in self.joints:
            t = poses[parent] @ origin; motion = np.eye(4); v = q[dof] if dof >= 0 else 0.
            if kind == "revolute":
                k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0.]]); motion[:3, :3] = np.eye(3) + math.sin(v) * k + (1 - math.cos(v)) * (k @ k)
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
    """Levenberg-Marquardt on x = mid + half * sin(u): the bounds hold by construction, so a start on a joint limit can still move (a clipped
    LM step stalls there). Damping floor so a joint without effect takes no huge step. Forward-difference Jacobian (stands in for scipy)."""
    mid, half = (hi + lo) / 2, (hi - lo) / 2; u = np.arcsin(np.clip((x - mid) / half, -.98, .98)); g = lambda v: fun(mid + half * np.sin(v))
    r = g(u); cost, lam = r @ r, 1e-2
    for _ in range(max_nfev):
        J = np.empty((len(r), len(u)))
        for k in range(len(u)):
            uk = u.copy(); uk[k] += 1e-6; J[:, k] = (g(uk) - r) / 1e-6
        A, grad = J.T @ J, J.T @ r
        while lam < 1e12:
            un = u - np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-3 * np.diag(A).max() + 1e-9), grad); rn = g(un)
            if rn @ rn < cost: break
            lam *= 4
        else:
            break
        done = cost - rn @ rn < 1e-12 * cost; u, r, cost, lam = un, rn, rn @ rn, max(lam / 3, 1e-9)
        if done: break
    return mid + half * np.sin(u)

class ArmServo:
    """Static-kinematics servo of torso + left arm (q[:11]) with bounded joint actions (native controller.ArmServo)."""
    def __init__(self): self.kin = Kinematics(); self.limits = np.asarray(LIMITS).copy(); self.limits[:, 3] = [-.75, .75]; self.link, self.point, self.target, self.synchronized = "left_gripper_link", np.array([0., 0., -.060]), None, False
    def pose(self, q): t = self.kin.forward(q)[0][self.link].copy(); t[:3, 3] += t[:3, :3] @ self.point; return t
    def capsules(self, q): poses = self.kin.forward(q)[0]; return [(poses[a][:3, 3], poses[b][:3, 3], .10) for _, a, b, _, _, _ in self.kin.joints if a.startswith(ARM_PREFIX) or b.startswith(ARM_PREFIX)]
    def solve(self, q, goal, rotation):
        lo, hi = self.limits; rest = np.asarray(REST); starts = [q[:11], rest, rest + np.array([0, 0, 0, 0, 0, .5, -.8, 0, 0, 0, 0])]; choices = []
        def residual(x): trial = q.copy(); trial[:11] = x; t = self.pose(trial); return np.r_[5 * (t[:3, 3] - goal), R.from_matrix(rotation @ t[:3, :3].T).as_rotvec(), .008 * (x - rest)]
        for start in starts:
            x = solve_lsq(residual, np.clip(start, lo + 1e-5, hi - 1e-5), lo, hi, 90); v = residual(x)
            if np.linalg.norm(v[:3]) / 5 < .015 and np.linalg.norm(v[3:6]) < .12: choices.append((np.linalg.norm(x - q[:11]), x))
        return min(choices, key=lambda c: c[0])[1] if choices else None
    def step(self, action, q, goal, rotation, plan=False):
        cache = self.kin.forward(q); t = cache[0][self.link]; point = t[:3, 3] + t[:3, :3] @ self.point; err = goal - point; ori = R.from_matrix(rotation @ t[:3, :3].T).as_rotvec()
        if plan and self.target is None: self.target = self.solve(q, goal, rotation)
        if self.target is not None:
            delta = self.target - q[:11]; dq = delta * min(1., .016 / max(np.max(np.abs(delta)), 1e-9)) if self.synchronized else np.clip(delta, -.016, .016)
            if np.max(np.abs(self.target - q[:11])) < .035: self.target = None
        else:
            J = self.kin.jacobian(self.link, q, self.point, cache)[:, :11]; delta = np.r_[np.clip(err * .25, -.012, .012), np.clip(ori * .2, -.03, .03)]
            dq = np.clip(J.T @ np.linalg.solve(J @ J.T + .002 * np.eye(6), delta), -.016, .016)
        command = np.clip(q[:11] + dq, self.limits[0] + 1e-5, self.limits[1] - 1e-5); action[3:7] = command[:4]; action[7:14] = command[4:]
        return float(np.linalg.norm(err)), float(np.linalg.norm(ori))
class DepthMap:
    def __init__(self): self.res, self.n, self.radius, self.z_min, self.nz = .06, 160, .276, .10, 24; self.log_odds = np.zeros((self.n, self.n, self.nz), np.int8)
    def ij(self, xy): return np.floor(np.asarray(xy) / self.res + self.n / 2).astype(int)
    def xy(self, ij): return (np.asarray(ij) - self.n / 2 + .5) * self.res
    def voxel_indices(self, points): cells = np.c_[self.ij(points[:, :2]), np.floor((points[:, 2] - self.z_min) / self.res).astype(int)]; cells = cells[np.all((cells >= 0) & (cells < np.array([self.n, self.n, self.nz])), axis=1)]; return np.unique((cells[:, 0] * self.n + cells[:, 1]) * self.nz + cells[:, 2])
    def update(self, head, odom, capsules):
        points, _, _ = pointcloud(head, stride=5); is_self = np.linalg.norm(points[:, :2], axis=1) < .47
        for a, b, radius in capsules:
            axis = b - a; alpha = np.clip(((points - a) @ axis) / max(float(axis @ axis), 1e-10), 0, 1); is_self |= np.linalg.norm(points - a - alpha[:, None] * axis, axis=1) < radius
        points = points.copy(); points[:, :2] = points[:, :2] @ rot2(odom[2]).T + odom[:2]; origin = head["t"].copy(); origin[:2] = rot2(odom[2]) @ origin[:2] + odom[:2]
        if not len(points): return
        # Clear only voxels crossed by observed rays, so a ray above a low obstacle never erases the ground column.
        rays = points[::max(1, int(np.ceil(len(points) / 512)))] - origin; lengths = np.linalg.norm(rays, axis=1); directions = rays / np.maximum(lengths[:, None], 1e-9)
        distances = np.arange(self.res / 2, 5., self.res); visible = distances[None, :] < (lengths[:, None] - self.res * .75)
        free = self.voxel_indices((origin + directions[:, None, :] * distances[None, :, None])[visible]); hit = self.voxel_indices(points[~is_self]); flat = self.log_odds.ravel()
        flat[free] = np.maximum(-8, flat[free] - 1); flat[hit] = np.minimum(8, flat[hit] + 3)
    def occupancy(self): width = int(np.ceil(self.radius / self.res)); k = np.arange(-width, width + 1); xx, yy = np.meshgrid(k, k); return binary_dilation((self.log_odds >= 2).any(axis=2), structure=xx * xx + yy * yy <= (self.radius / self.res) ** 2)
    def path(self, start, goal):
        occ = self.occupancy(); a, b = self.ij(start), self.ij(goal)
        if not np.all((b >= 0) & (b < self.n)) or occ[tuple(b)]: return None
        xs, ys = np.ogrid[:self.n, :self.n]; occ[(xs - a[0]) ** 2 + (ys - a[1]) ** 2 < (self.radius / self.res) ** 2] = False  # the start footprint is already occupied by the robot
        a, b = tuple(a), tuple(b); queue = [(0., 0., a)]; cost = {a: 0.}; prev = {}; expanded = 0
        while queue and expanded < 16000:
            expanded += 1; _, g, u = heapq.heappop(queue)
            if g > cost[u] + 1e-6: continue
            if u == b:
                path = [u]
                while u != a: u = prev[u]; path.append(u)
                return [self.xy(p) for p in path[::-1]]
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                v = (u[0] + dx, u[1] + dy)
                if not (0 <= v[0] < self.n and 0 <= v[1] < self.n) or occ[v]: continue
                if dx and dy and (occ[u[0] + dx, u[1]] or occ[u[0], u[1] + dy]): continue
                ng = g + math.hypot(dx, dy)
                if ng < cost.get(v, float("inf")): cost[v] = ng; prev[v] = u; heapq.heappush(queue, (ng + math.hypot(v[0] - b[0], v[1] - b[1]), ng, v))
        return None
    def line_clear(self, start, goal):
        occ = self.occupancy(); a = self.ij(start); xs, ys = np.ogrid[:self.n, :self.n]; occ[(xs - a[0]) ** 2 + (ys - a[1]) ** 2 < (self.radius / self.res) ** 2] = False
        cells = self.ij(np.asarray(start)[None] + np.linspace(0, 1, max(2, int(np.linalg.norm(np.asarray(goal) - start) / self.res * 2)))[:, None] * (np.asarray(goal) - start))
        return bool(np.all((cells >= 0) & (cells < self.n)) and not occ[cells[:, 0], cells[:, 1]].any())
class GroundFrame:
    """Gravity and ground clearance from the visible RGB-D floor (native ground_frame.GroundFrameEstimator)."""
    def __init__(self): self.transform = np.eye(4)
    def update(self, camera):
        points, _, _ = pointcloud(camera, stride=5); radius = np.linalg.norm(points[:, :2], axis=1); points = points[(radius > .5) & (radius < 4) & (points[:, 2] > -.9) & (points[:, 2] < .55)]; points = points[::max(1, len(points) // 1800)]
        if len(points) < 60: return self.transform.copy()
        rng = np.random.default_rng(905); best = None
        for _ in range(240):
            a, b, c = points[rng.choice(len(points), 3, replace=False)]; normal = np.cross(b - a, c - a); norm = np.linalg.norm(normal)
            if norm < .02: continue
            normal /= norm; normal = -normal if normal[2] < 0 else normal
            if normal[2] < .80: continue
            offset = -normal @ a; intercept = -offset / normal[2]
            if not -.45 < intercept < .15: continue
            mask = np.abs(points @ normal + offset) < .012
            if mask.sum() < 100 or np.linalg.norm(np.ptp(points[mask, :2], axis=0)) < .9: continue
            score = mask.sum() - 200 * max(intercept, 0)
            if best is None or score > best[0]: best = (score, mask)
        if best is None: return self.transform.copy()
        sel = points[best[1]]; center = sel.mean(0); normal = np.linalg.svd(sel - center, full_matrices=False)[2][-1]; normal = -normal if normal[2] < 0 else normal
        if np.median(np.abs((sel - center) @ normal)) > .003: return self.transform.copy()
        cross = np.cross(normal, [0., 0., 1.]); sine = np.linalg.norm(cross); rot = np.eye(3) if sine < 1e-8 else R.from_rotvec(cross / sine * np.arctan2(sine, normal[2])).as_matrix()
        self.transform = np.eye(4); self.transform[:3, :3] = rot; self.transform[2, 3] = -np.median(sel @ rot[2])
        return self.transform.copy()
# ---- cook_broccolini specific part (broccolini_v2): plate, food and pan finders, phases ----------------------------------------------------
PHASES = ("search", "approach_table", "navigate", "raise_hand", "settle_view", "pregrasp", "descend", "close", "lift", "reopen", "find_pan", "pan_navigate",
          "over_pan", "release")
CLOSED = ("close", "lift", "find_pan", "pan_navigate", "over_pan")  # left gripper command -1 in these phases
HELD_PHASES = ("lift", "find_pan", "pan_navigate", "over_pan")
ARM_PREFIX = ("left_arm", "right_arm", "left_gripper", "right_gripper")  # links whose capsules mask the robot out of the depth map
STALL_STEPS, STALL_GAIN = 150, .005  # a servo phase whose error has not dropped by STALL_GAIN for STALL_STEPS steps retries or backs off
FINGER_STILL_STEPS, FINGER_STILL_EPS = 6, 2e-4  # gripper motion is over when the finger width changed less than EPS over STEPS
HOLD_MIN_WIDTH = .004  # finger width (sum of both joints, m) above which the fingers hold something
DROP_STEPS = 20  # fingers empty this long while carrying -> plate dropped, search again
SEARCH_YAW_RATE, SEARCH_EXPLORE_AFTER, SEARCH_CYCLE, EXPLORE_STEP = .32, 600, 780, .65  # spin rate, local exploration after a spin, cycle, move length (m)
NAV_REPLAN_EVERY, NAV_MAX_FAILS, NAV_STALL = 90, 4, 45
NAV_GAIN, NAV_VMAX, NAV_YAW_GAIN, NAV_WMAX, NAV_REACH, NAV_YAW_TOL = .9, .22, .8, .35, .045, .14
RAISE_GOAL, RAISE_POS_TOL, RAISE_ORI_TOL = (.38, .38, .90), .05, .3  # hand-raised viewing pose in the robot frame
RAISE_MAX = 210
SETTLE_MIN, SETTLE_MAX, FRESH_FIT = 36, 120, 15  # wait for a plate detection not older than FRESH_FIT steps
PREGRASP_HEIGHT = .15  # pregrasp height above the grasp point
TIP_BELOW_POINT = .015  # fingertips hang this far below the servo point (finger link z -.075 against the point at -.060)
PREGRASP_POS_TOL, PREGRASP_ORI_TOL, IK_MAX_REPLANS = .045, .40, 3
DESCEND_ORI_TOL, DESCEND_MAX_STALLS = .22, 2
DESCEND_OVERSHOOT, DESCEND_Z_TOL, DESCEND_XY_TOL = .012, .005, .015  # servo below the grasp point, close once the point is this close in z (v2a: tips stopped 1 cm high, closed on air)
CLOSE_MIN, PICKUP_MAX_FAILS, CLOSE_RAISE, CLOSE_MAX = 24, 3, .006, 60
REOPEN_MAX = 150
LIFT_BACK, LIFT_UP, LIFT_TOL, LIFT_CONFIRM = .0, .12, .06, .06
ODOM_CMD_YAW_GAIN = .9  # odometry yaw integrates the commanded yaw rate times this gain (proprio yaw rate reads 0.8-1.4x the true turn in the port)
DETECT_EVERY, DETECT_SIZE = 6, 720  # head-camera detection period (steps) and resolution
PLATE_WHITE_MIN, PLATE_WHITE_SPREAD, PLATE_Z, PLATE_EXTENT, PLATE_THICK, PLATE_MIN_PTS = 185, 40, (.80, .98), (.10, .36), .06, 12  # white flat disc at table height
PAN_BLACK_MAX, PAN_Z, PAN_EXTENT, PAN_THICK, PAN_MIN_PTS, PAN_ABOVE_TABLE = 60, (.90, 1.0), (.18, .50), .09, 40, (.025, .11)  # dark pan walls above the table top
TABLE_Z, TABLE_EXTENT, TABLE_THICK, TABLE_MIN_PTS, TABLE_RADII = (.80, .97), (.35, 3.5), .08, 150, (1.0, 1.15, 1.3)  # work table: large flat surface at table height
PLATE_RADIUS_DEFAULT = .13
SELF_RADIUS = .55  # detections closer than this to the robot base are the robot itself
DROP_STANDOFFS, DROP_HEIGHT, DROP_TOL, RELEASE_STEPS = (.52, .58, .66, .74, .82, .90), .10, .04, 40  # search nearer and farther sides of the pan
FOOD_ABOVE, FOOD_R, FOOD_MIN_PTS, FOOD_CELL, FOOD_SPLIT = .004, .135, 3, .008, .075  # include low, small, and edge ingredients while excluding the white plate
FOOD_TIP_ABOVE_TABLE = .013  # fingertip height above the table top (the plate bottom is ~4 mm above it, the items ~1.5 cm thick)
PAN_FRESH, PAN_LOST = 30, 400
CARRY_GOAL = (.32, .30, 1.08)  # carry pose of the left hand in the robot frame, above the table tops (0.87 m)
REST = (.5, -.65, -.5, 0., -1.4, .35, .2, -.7, 0., .3, 0.)


def plan_approach(depth_map, odom, object_position, radii):
    """Full-circle approach search through the occupancy map: stand at one of `radii` from the object, facing it, hand side shifted .12 m."""
    center = np.asarray(object_position)[:2]; view = np.asarray(odom)[:2] - center; view /= max(np.linalg.norm(view), 1e-8); candidates = []
    for radius in radii:
        for turn in np.linspace(-np.pi, np.pi, 33)[:-1]:
            direction = rot2(turn) @ view; heading = np.arctan2(-direction[1], -direction[0]); goal = center + radius * direction - .12 * np.array([-np.sin(heading), np.cos(heading)]); path = depth_map.path(odom[:2], goal)
            if path: candidates.append((sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])) + .10 * abs(turn) + .10 * abs(wrap(heading - odom[2])), goal, heading, path))
    if not candidates: return None
    _, goal, heading, path = min(candidates, key=lambda c: c[0])
    return goal, heading, path

def find_flat_discs(head, base, color, z_range, extent, thick, min_pts):
    """Flat horizontal blobs of one colour at a height range: (centre xyz, radius, points) in the odom frame. head is a robot-frame camera dict."""
    pts, rgb, _ = pointcloud(head, stride=2); world = pts @ base[:3, :3].T + base[:3, 3]
    keep = (np.linalg.norm(world[:, :2] - base[:2, 3], axis=1) > SELF_RADIUS) & (world[:, 2] > z_range[0]) & (world[:, 2] < z_range[1]) & color(rgb.astype(float))
    sel = world[keep]
    if len(sel) < min_pts: return []
    lo = sel[:, :2].min(0); cell = ((sel[:, :2] - lo) / .03).astype(int); img = np.zeros(cell.max(0) + 1, bool); img[cell[:, 0], cell[:, 1]] = True
    labels, n = label(dilate(img, 3), structure=np.ones((3, 3))); out = []
    for k in range(1, n + 1):
        members = sel[labels[cell[:, 0], cell[:, 1]] == k]
        if len(members) < min_pts: continue
        spread = np.ptp(members[:, :2], axis=0)
        if np.all(spread > extent[0]) and np.all(spread < extent[1]) and np.ptp(members[:, 2]) < thick: out.append((np.r_[members[:, :2].mean(0), np.percentile(members[:, 2], 80)], float(spread.max() / 2), members))
    return out

def plate_color(a): return (a.min(1) > PLATE_WHITE_MIN) & (a.max(1) - a.min(1) < PLATE_WHITE_SPREAD)

def pan_color(a): return a.max(1) < PAN_BLACK_MAX

def find_food(head, base, centre, table_z):
    """Food items on the plate: blobs of head-cloud points inside FOOD_R of the plate centre that stand FOOD_ABOVE over the table (the plate bottom is
    0-4 mm thick there). Each: centre (xy mean, median z), principal xy axis, point count, green fraction (broccolini) and top height, odom frame."""
    pts, rgb, _ = pointcloud(head); world = pts @ base[:3, :3].T + base[:3, 3]; r = np.linalg.norm(world[:, :2] - centre[:2], axis=1)
    keep = (r < FOOD_R) & (world[:, 2] > table_z + FOOD_ABOVE) & (world[:, 2] < table_z + .08) & ~plate_color(rgb.astype(float)); sel, col = world[keep], rgb[keep].astype(float)  # white points are the plate rim
    if len(sel) < FOOD_MIN_PTS: return []
    cell = ((sel[:, :2] - sel[:, :2].min(0)) / FOOD_CELL).astype(int); img = np.zeros(cell.max(0) + 1, bool); img[cell[:, 0], cell[:, 1]] = True
    labels, n = label(img, structure=np.ones((3, 3))); member = labels[cell[:, 0], cell[:, 1]]; out = []
    groups = [np.flatnonzero(member == k) for k in range(1, n + 1)]
    while groups:
        g = groups.pop()
        if len(g) < FOOD_MIN_PTS: continue
        xy = sel[g, :2]; c = xy.mean(0); axis = np.linalg.svd(xy - c, full_matrices=False)[2][0]; along = (xy - c) @ axis
        if np.ptp(along) > FOOD_SPLIT: groups += [g[along < np.median(along)], g[along >= np.median(along)]]; continue  # touching items: split at the median along the axis
        out.append({"centre": np.r_[c, np.median(sel[g, 2])], "axis": axis, "n": len(g), "green": float(np.mean(col[g, 1] > col[g, 0] + 8)), "top": float(sel[g, 2].max()), "length": float(np.ptp(along))})
    return out


class Planner:
    """Food transfer for cook_broccolini: search the ingredient plate, approach, pinch one food item with the left hand, lift, find the frying pan, drive
    to a drop stand, open over the pan, return for the next item. Every phase moves on only from what it observes; a stalled phase retries or
    backs off (reason in log()). Not in this version: carrying the pan to the stove and pressing the switch."""
    def reset(self, task_spec):
        self.servo = ArmServo(); self.ground = GroundFrame(); self.map = DepthMap(); self.step = self.age = 0; self.phase = "search"; self.odom = np.zeros(3); self.level = np.eye(4); self.action = None; self.nav_last = None; self.nav_stall = 0
        self.table = None; self.table_done = False
        self.plate = self.pan = self.grasp = self.nav = self.path = self.heading = self.lift_goal = self.lift_rotation = self.close_pose = self.grip_plate = None
        self.foods, self.food, self.food_seen, self.tried_foods = [], None, -1000, []
        self.plate_radius = PLATE_RADIUS_DEFAULT; self.table_z = None; self.last_seen = self.pan_seen = -1000; self.pregrasp_ready = False; self.not_held = 0; self.orientation_point = None
        self.reason = "start"; self.best = (1e9, 0); self.fingers = []; self.events = []
        self.counts = {"grasp_attempts": 0, "pickups_failed": 0, "ik_replans": 0, "nav_replans": 0, "drops": 0, "plate_dets": 0, "pan_dets": 0, "descend_stalls": 0, "drop_replans": 0, "table_dets": 0, "released": 0, "no_food": 0}
    def log(self):
        return {"stage": PHASES.index(self.phase), "phase": self.phase, "reason": self.reason, "step": self.step, "odom": self.odom.round(3).tolist(), "plate_seen": self.plate is not None,
                "pan_seen": self.pan is not None, "foods_seen": len(self.foods), "food": None if self.food is None else [round(float(v), 3) for v in self.food["centre"]], "finger_width": round(self.fingers[-1], 4) if self.fingers else None, **self.counts, "events": self.events[-40:]}
    def change(self, phase, reason=""):
        self.phase, self.age = phase, 0; self.servo.target = None; self.best = (1e9, 0); self.reason = reason or "entered"; self.events = self.events[-39:] + [f"{self.step}:{phase}:{reason}"]
    def stalled(self, error):
        if error < self.best[0] - STALL_GAIN: self.best = (error, self.age)
        return self.age - self.best[1] > STALL_STEPS
    def holding(self): return bool(self.fingers) and self.fingers[-1] > HOLD_MIN_WIDTH
    def fingers_still(self): return len(self.fingers) > FINGER_STILL_STEPS and abs(self.fingers[-1] - self.fingers[-1 - FINGER_STILL_STEPS]) < FINGER_STILL_EPS
    def forget_grasp(self): self.grasp = self.grip_plate = None; self.pregrasp_ready = False; self.servo.synchronized = False
    def detect(self, frame, base):
        """Fuse head detections with a close left-wrist pass when small foods are missing."""
        head = make_cameras(frame, size=DETECT_SIZE, only="head")["head"]
        if self.phase not in HELD_PHASES:
            found = find_flat_discs(head, base, plate_color, PLATE_Z, PLATE_EXTENT, PLATE_THICK, PLATE_MIN_PTS)
            if found:
                centre, radius, _ = min(found, key=lambda f: np.linalg.norm(f[0][:2] - base[:2, 3])); self.plate = np.eye(4); self.plate[:3, 3] = centre; self.plate_radius = float(np.clip(radius, .13, .16)); self.last_seen = self.step; self.counts["plate_dets"] += 1; self.table_height(head, base, centre)
                if self.table_z is not None:
                    self.foods = find_food(head, base, centre, self.table_z)
                    if self.phase in ("settle_view", "pregrasp") and len(self.foods) < 5:
                        wrist = make_cameras(frame, size=None, only="left_wrist")["left_wrist"]
                        for item in find_food(wrist, base, centre, self.table_z):
                            if not any(np.linalg.norm(item["centre"][:2] - old["centre"][:2]) < .035 for old in self.foods):
                                self.foods.append(item)
                    self.food_seen = self.step
            elif self.phase == "search" and not self.table_done:
                tables = find_flat_discs(head, base, lambda a: np.ones(len(a), bool), TABLE_Z, TABLE_EXTENT, TABLE_THICK, TABLE_MIN_PTS)
                if tables:
                    members = min(tables, key=lambda f: np.linalg.norm(f[0][:2] - base[:2, 3]))[2]; self.table = members[np.argmin(np.linalg.norm(members[:, :2] - base[:2, 3], axis=1)), :3]; self.counts["table_dets"] += 1
        else:
            z = PAN_Z if self.table_z is None else (self.table_z + PAN_ABOVE_TABLE[0], self.table_z + PAN_ABOVE_TABLE[1]); found = find_flat_discs(head, base, pan_color, z, PAN_EXTENT, PAN_THICK, PAN_MIN_PTS)  # only the raised walls: the black marble table top itself is dark (v2e dropped on the table)
            if found:
                centre, _, _ = min(found, key=lambda f: np.linalg.norm(f[0][:2] - base[:2, 3])); self.pan = centre; self.pan_seen = self.step; self.counts["pan_dets"] += 1
    def table_height(self, head, base, centre):
        """Table top height from the cloud in a ring around the plate (median z of the points 4 to 30 cm outside the plate disc)."""
        pts, _, _ = pointcloud(head, stride=3); world = pts @ base[:3, :3].T + base[:3, 3]; r = np.linalg.norm(world[:, :2] - centre[:2], axis=1)
        ring = world[(r > self.plate_radius + .04) & (r < self.plate_radius + .30) & (world[:, 2] > .75) & (world[:, 2] < centre[2])]
        if len(ring) > 40: self.table_z = float(np.median(ring[:, 2]))
    def plan_navigation(self, target, radii=(.65, .73, .81, .90)):
        """Select a collision-free standoff and reset the odometry stall detector."""
        result = plan_approach(self.map, self.odom, target, radii)
        if result is None:
            result = plan_approach(self.map, self.odom, target, tuple(radii) + (.48, .56, 1.0, 1.1))
        if result is None: return False
        self.nav, self.heading, self.path = result
        self.nav_last = self.odom.copy()
        self.nav_stall = 0
        return True
    def drive(self, gain, vmax, wgain, wmax, action):
        """Follow the path while forcing a replan after insufficient odometry progress."""
        if self.nav_last is None:
            self.nav_last = self.odom.copy()
        elif np.linalg.norm(self.odom[:2] - self.nav_last[:2]) < .004:
            self.nav_stall += 1
        else:
            self.nav_stall = 0
        self.nav_last = self.odom.copy()
        near = np.argmin([np.linalg.norm(x - self.odom[:2]) for x in self.path]); waypoint = self.path[min(near + 3, len(self.path) - 1)]
        action[:2] = np.clip(rot2(self.odom[2]).T @ (waypoint - self.odom[:2]) * gain, -vmax, vmax); action[2] = np.clip(wrap(self.heading - self.odom[2]) * wgain, -wmax, wmax)
        if self.nav_stall > NAV_STALL:
            action[:2] = np.array([-.08, 0.])
            self.path = None
    def explore(self, action, reason):
        options = []
        for theta in np.linspace(-np.pi, np.pi, 12, endpoint=False):
            goal = self.odom[:2] + EXPLORE_STEP * np.array([np.cos(theta), np.sin(theta)])
            if self.map.path(self.odom[:2], goal): options.append((abs(wrap(theta - self.odom[2])), goal))
        if options: goal = min(options, key=lambda x: x[0])[1]; action[:2] = np.clip(rot2(self.odom[2]).T @ (goal - self.odom[:2]) * .4, -.18, .18); action[2] = 0; self.reason = reason
    # ---- phases: each returns the arm error (or None) and sets self.reason ----
    def phase_search(self, action, q, cams, base, inverse):
        if self.plate is not None and self.step - self.last_seen < 20:
            if self.plan_navigation(self.plate[:3, 3]): self.change("navigate", "plate detected, approach planned"); return None
            self.plate = None; self.last_seen = -1000; self.path = None; self.nav = None
            self.reason = "plate detected, refreshing viewpoint after blocked approach"
        if self.plate is None and self.table is not None and not self.table_done:
            if self.plan_navigation(self.table, TABLE_RADII): self.change("approach_table", "plate not seen, table seen, approach planned"); return None
            self.table = None
        action[2] = SEARCH_YAW_RATE; self.reason = "spinning, plate not detected" if self.plate is None else self.reason
        if self.age > SEARCH_EXPLORE_AFTER:
            self.explore(action, "exploring, plate not detected after a full spin")
            if self.age > SEARCH_CYCLE: self.age = 0
        return None
    def phase_approach_table(self, action, q, cams, base, inverse):
        if self.plate is not None and self.step - self.last_seen < 20: self.table_done = True; self.change("search", "plate detected on the way"); return None
        if (self.age % NAV_REPLAN_EVERY == 1 or self.nav_stall > NAV_STALL) and not self.plan_navigation(self.table, TABLE_RADII):
            self.counts["nav_replans"] += 1; self.reason = "replan failed, keep last path"
            if self.counts["nav_replans"] % NAV_MAX_FAILS == 0: self.table_done = True; self.change("search", "table unreachable"); return None
        if self.path: self.drive(NAV_GAIN, NAV_VMAX, NAV_YAW_GAIN, NAV_WMAX, action)
        distance = np.linalg.norm(self.nav - self.odom[:2]); self.reason = f"driving, {distance:.2f} m to the table stand-off"
        if distance < NAV_REACH and abs(wrap(self.heading - self.odom[2])) < NAV_YAW_TOL: self.table_done = True; self.change("search", "table stand-off reached")
        return None
    def phase_navigate(self, action, q, cams, base, inverse):
        if (self.age % NAV_REPLAN_EVERY == 1 or self.nav_stall > NAV_STALL) and not self.plan_navigation(self.plate[:3, 3]):
            self.counts["nav_replans"] += 1; self.reason = "replan failed, keep last path"
            if self.counts["nav_replans"] % NAV_MAX_FAILS == 0: self.plate = None; self.change("search", "approach unreachable, search again"); return None
        if self.path: self.drive(NAV_GAIN, NAV_VMAX, NAV_YAW_GAIN, NAV_WMAX, action)
        distance = np.linalg.norm(self.nav - self.odom[:2]); self.reason = f"driving, {distance:.2f} m to the approach pose"
        if distance < NAV_REACH and abs(wrap(self.heading - self.odom[2])) < NAV_YAW_TOL: self.grasp = None; self.change("raise_hand", "approach pose reached")
        return None
    def phase_raise_hand(self, action, q, cams, base, inverse):
        """Raise for a bounded interval, then use wrist perception instead of waiting indefinitely."""
        error = self.servo.step(action, q, np.asarray(RAISE_GOAL), np.eye(3), plan=self.age == 1); self.reason = f"raising hand, error {error[0]:.3f}"
        if error[0] < RAISE_POS_TOL and error[1] < RAISE_ORI_TOL:
            self.change("settle_view", "hand raised")
        elif self.age > RAISE_MAX:
            self.servo.target = None
            self.change("settle_view", "raise budget expired, use wrist detection")
        elif self.stalled(error[0]):
            self.counts["ik_replans"] += 1
            self.servo.target = self.servo.solve(q, np.asarray(RAISE_GOAL), np.eye(3))
            self.best = (1e9, self.age)
            self.reason = "raise stalled, new IK target"
        return error
    def phase_settle_view(self, action, q, cams, base, inverse):
        """Select a fresh food blob and reset all per-grasp state before planning."""
        self.reason = "waiting for a fresh food detection on the plate"
        if self.age > SETTLE_MIN and self.step - self.food_seen < FRESH_FIT and self.foods:
            candidates = [f for f in self.foods if not any(np.linalg.norm(f["centre"][:2] - p) < .035 for p in self.tried_foods)]
            if not candidates:
                self.tried_foods = []
                candidates = self.foods
            self.food = max(candidates, key=lambda f: (f["green"] > .3, f["n"]))
            self.grasp = None
            self.orientation_point = None
            self.pregrasp_ready = False
            self.counts["no_food"] = 0
            self.counts["grasp_attempts"] += 1
            self.change("pregrasp", f"food item picked: {self.food['n']} points, length {self.food['length']:.3f}, green {self.food['green']:.2f}")
        elif self.age > SETTLE_MAX and self.plate is not None and self.step - self.food_seen < FRESH_FIT: self.counts["no_food"] += 1; self.age = SETTLE_MIN; self.reason = "plate seen, no food left on it"
        elif self.age > SETTLE_MAX: self.change("search", "plate not seen from the approach pose")
        return None
    def plan_grasp(self, q, inverse):
        """Top-down pinch across the food item: closing axis horizontal and normal to the item's long axis, fingertips just above the plate bottom."""
        f = self.food; up = np.array([0., 0., 1.]); axis = inverse[:3, :3] @ np.r_[f["axis"], 0.]; closing = np.cross(up, axis); closing /= max(np.linalg.norm(closing), 1e-6)
        tip_z = self.table_z + FOOD_TIP_ABOVE_TABLE; grip = (inverse @ np.r_[f["centre"][:2], tip_z + TIP_BELOW_POINT, 1.])[:3]
        rotation = R.from_matrix(np.column_stack([np.cross(closing, up), closing, up])).as_matrix(); choices = []
        for sign in [1, -1]:
            candidate = rotation @ np.diag([sign, sign, 1]); solution = self.servo.solve(q, grip + up * .09, candidate)
            if solution is not None: choices.append((np.linalg.norm(solution - q[:11]), candidate, solution))
        if choices: _, rotation, solution = min(choices, key=lambda x: x[0])
        self.grasp = {"point": grip, "rotation": rotation, "initial_z": grip[2], "ik": bool(choices)}
    def phase_pregrasp(self, action, q, cams, base, inverse):
        if self.grasp is None: self.plan_grasp(q, inverse)
        goal, rotation = self.grasp["point"] + self.grasp["rotation"][:, 2] * PREGRASP_HEIGHT, self.grasp["rotation"]
        if self.orientation_point is None: self.orientation_point = self.servo.pose(q)[:3, 3].copy()
        if not self.pregrasp_ready: goal = self.orientation_point  # first turn the hand in place, then move above the grasp point
        else: goal[2] = max(goal[2], self.orientation_point[2])
        error = self.servo.step(action, q, goal, rotation, plan=False); self.reason = f"{'moving above grasp' if self.pregrasp_ready else 'orienting hand'}, error {error[0]:.3f}/{error[1]:.2f}"
        if error[0] < PREGRASP_POS_TOL and error[1] < PREGRASP_ORI_TOL:
            if not self.pregrasp_ready: self.pregrasp_ready = True; self.best = (1e9, self.age)
            else: self.change("descend", "above the grasp point")
        elif self.stalled(error[0] + .1 * error[1]):
            self.counts["ik_replans"] += 1; self.grasp = None; self.pregrasp_ready = False; self.orientation_point = None
            if self.counts["ik_replans"] % IK_MAX_REPLANS == 0: self.change("settle_view", "pregrasp unreachable, detect the plate again")
            else: self.best = (1e9, self.age); self.reason = "pregrasp stalled, replanning the grasp"
        return error
    def phase_descend(self, action, q, cams, base, inverse):
        goal = self.grasp["point"] - np.array([0., 0., DESCEND_OVERSHOOT]); error = self.servo.step(action, q, goal, self.grasp["rotation"], plan=False)
        offset = self.servo.pose(q)[:3, 3] - self.grasp["point"]; self.reason = f"descending, dz {offset[2]:.3f}, dxy {np.linalg.norm(offset[:2]):.3f}"
        if offset[2] < DESCEND_Z_TOL and np.linalg.norm(offset[:2]) < DESCEND_XY_TOL and error[1] < DESCEND_ORI_TOL: self.change("close", "at the grasp point")
        elif self.stalled(error[0]) and offset[2] < 2 * DESCEND_Z_TOL and np.linalg.norm(offset[:2]) < DESCEND_XY_TOL: self.change("close", "descent stopped by contact near the grasp point")
        elif self.stalled(error[0]):
            self.counts["descend_stalls"] += 1
            if self.counts["descend_stalls"] % DESCEND_MAX_STALLS: self.pregrasp_ready = True; self.change("pregrasp", "descend stalled, back above the grasp")
            else: self.counts["pickups_failed"] += 1; self.forget_grasp(); self.change("reopen", "grasp point unreachable, detect and replan the grasp")
        return error
    def phase_close(self, action, q, cams, base, inverse):
        """Close until finger motion settles, then force a lift attempt so failures can be retried."""
        self.servo.step(action, q, self.grasp["point"] + np.array([0., 0., CLOSE_RAISE]), self.grasp["rotation"], plan=False); self.reason = "closing the gripper"
        if self.age > CLOSE_MIN and (self.fingers_still() or self.age > CLOSE_MAX):
            self.close_pose = self.servo.pose(q).copy(); self.lift_rotation = self.close_pose[:3, :3].copy(); direction = -self.close_pose[:3, 3].copy(); direction[2] = 0; direction /= max(np.linalg.norm(direction), 1e-6)
            self.lift_goal = self.close_pose[:3, 3] + direction * LIFT_BACK + np.array([0., 0., LIFT_UP]); self.grip_plate = self.close_pose.copy(); self.change("lift", f"gripper closed at width {self.fingers[-1]:.4f}")
        return None
    def phase_lift(self, action, q, cams, base, inverse):
        error = self.servo.step(action, q, self.lift_goal, self.lift_rotation, plan=False); self.reason = f"lifting, error {error[0]:.3f}"
        if error[0] < LIFT_TOL or self.stalled(error[0]):
            motion = np.linalg.norm(self.servo.pose(q)[:3, 3] - self.close_pose[:3, 3])
            if self.holding() and motion > LIFT_CONFIRM: self.pan = None; self.change("find_pan", "plate lifted (finger width)")
            else: self.tried_foods.append(self.food["centre"][:2].copy()); self.counts["pickups_failed"] += 1; self.forget_grasp(); self.change("reopen", "lift not confirmed: fingers empty")
        return error
    def phase_reopen(self, action, q, cams, base, inverse):
        """Open and retreat for a bounded interval, then retry perception or re-approach."""
        error = self.servo.step(action, q, np.asarray(RAISE_GOAL), np.eye(3), plan=self.age == 1); self.reason = f"reopening and raising, error {error[0]:.3f}"
        ready = self.fingers_still() or self.age > REOPEN_MAX or self.stalled(error[0])
        if self.age > CLOSE_MIN and ready:
            if self.counts["pickups_failed"] > 0 and self.counts["pickups_failed"] % PICKUP_MAX_FAILS == 0:
                self.plate = None
                self.change("search", "repeated pickup failures, approach again")
            else:
                self.change("settle_view", "hand open, detect and grasp again")
        return error
    def carry(self, action, q):
        """Hold the item tucked high next to the body while turning and driving (v2d: the extended arm swept the pan off the table)."""
        self.servo.step(action, q, np.asarray(CARRY_GOAL), self.lift_rotation, plan=self.age == 1)
    def phase_find_pan(self, action, q, cams, base, inverse):
        self.carry(action, q)
        if self.pan is not None and self.step - self.pan_seen < 20:
            for radii in (DROP_STANDOFFS,):
                if self.plan_navigation(self.pan, radii): self.change("pan_navigate", "pan detected, stand planned"); return None
            self.pan = None
            self.pan_seen = -1000
            self.reason = "pan stand blocked, refreshing map and searching"
        action[2] = SEARCH_YAW_RATE; self.reason = "spinning with the plate, pan not detected" if self.pan is None else self.reason
        if self.age > SEARCH_EXPLORE_AFTER:
            self.explore(action, "exploring with the plate, pan not detected after a full spin")
            if self.age > SEARCH_CYCLE: self.age = 0
        return None
    def phase_pan_navigate(self, action, q, cams, base, inverse):
        self.carry(action, q)
        if (self.age % NAV_REPLAN_EVERY == 1 or self.nav_stall > NAV_STALL) and not self.plan_navigation(self.pan, DROP_STANDOFFS):
            self.counts["nav_replans"] += 1; self.reason = "replan failed, keep last path"
            if self.counts["nav_replans"] % NAV_MAX_FAILS == 0: self.pan = None; self.change("find_pan", "stand unreachable, look for the pan again"); return None
        if self.path: self.drive(NAV_GAIN, NAV_VMAX, NAV_YAW_GAIN, NAV_WMAX, action)
        distance = np.linalg.norm(self.nav - self.odom[:2]); self.reason = f"driving with the plate, {distance:.2f} m to the drop stand"
        if distance < NAV_REACH and abs(wrap(self.heading - self.odom[2])) < NAV_YAW_TOL: self.change("over_pan", "drop stand reached")
        return None
    def phase_over_pan(self, action, q, cams, base, inverse):
        goal = (inverse @ np.r_[self.pan[:2], self.pan[2] + DROP_HEIGHT, 1.])[:3]; error = self.servo.step(action, q, goal, self.lift_rotation, plan=self.age == 1)
        self.reason = f"moving the item over the pan, error {error[0]:.3f}"
        if error[0] < DROP_TOL: self.change("release", "item over the pan")
        elif self.stalled(error[0]): self.counts["drop_replans"] += 1; self.pan = None; self.change("find_pan", "drop pose unreachable, look for another stand")
        return error
    def phase_release(self, action, q, cams, base, inverse):
        self.reason = f"opening over the pan, {self.age} steps"
        if self.age > RELEASE_STEPS: self.counts["released"] += 1; self.forget_grasp(); self.food = None; self.change("reopen", "item released over the pan")
        return None
    def act(self, obs):
        # A closed list of the eight contract keys; nothing else can enter the control law.
        frame = {
            "robot_r1::proprio": first(obs["robot_r1::proprio"]), "robot_r1::cam_rel_poses": first(obs["robot_r1::cam_rel_poses"]),
            "robot_r1::robot_r1:zed_link:Camera:0::rgb": first(obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"]), "robot_r1::robot_r1:zed_link:Camera:0::depth_linear": first(obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"]),
            "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": first(obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"]), "robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear": first(obs["robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear"]),
            "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": first(obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"]), "robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear": first(obs["robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear"]),
        }
        s = frame["robot_r1::proprio"]; q = from_proprio(s); self.step += 1; self.age += 1; servo = self.servo; phase = self.phase; self.fingers = self.fingers[-40:] + [float(s[24:26].sum())]
        self.odom[:2] += rot2(self.odom[2]) @ s[:2] / 30; self.odom[2] += (0. if self.action is None else self.action[2]) * ODOM_CMD_YAW_GAIN / 30  # xy from proprio velocity, yaw from the last yaw command
        cams = make_cameras(frame, size=None)
        if self.step % 5 == 1: self.level = self.ground.update(make_cameras(frame, size=480, only="head")["head"])
        base = np.eye(4); base[:3, :3] = R.from_euler("z", self.odom[2]).as_matrix(); base[:2, 3] = self.odom[:2]; base = base @ self.level; inverse = np.linalg.inv(base)
        if self.step % DETECT_EVERY == 1 and phase not in ("close", "lift", "release"): self.detect(frame, base)
        if self.action is None: self.action = np.r_[np.zeros(3), s[53:57], s[3:10], 1., s[28:35], 1.]
        action = self.action.copy(); action[:3] = 0; action[14] = -1 if phase in CLOSED else 1
        if self.step % 15 == 1:
            capsules = [(self.level[:3, :3] @ a + self.level[:3, 3], self.level[:3, :3] @ b + self.level[:3, 3], r) for a, b, r in servo.capsules(q)]
            self.map.update(level_cameras(make_cameras(frame, size=256, only="head"), self.level)["head"], self.odom, capsules)
        if self.grip_plate is not None and phase in HELD_PHASES[1:]:
            self.not_held = self.not_held + 1 if not self.holding() else 0
            if self.not_held > DROP_STEPS: self.counts["drops"] += 1; self.not_held = 0; self.forget_grasp(); self.plate = None; self.change("search", "plate dropped (fingers closed on nothing)"); phase = "search"
        self.dispatch(phase, action, q, cams, base, inverse)
        self.action = action.copy(); assert np.isfinite(action).all()
        return action.astype(np.float32)
    def dispatch(self, phase, action, q, cams, base, inverse):
        handlers = {"search": self.phase_search, "approach_table": self.phase_approach_table, "navigate": self.phase_navigate, "raise_hand": self.phase_raise_hand, "settle_view": self.phase_settle_view, "pregrasp": self.phase_pregrasp,
                    "descend": self.phase_descend, "close": self.phase_close, "lift": self.phase_lift, "reopen": self.phase_reopen, "find_pan": self.phase_find_pan, "pan_navigate": self.phase_pan_navigate,
                    "over_pan": self.phase_over_pan, "release": self.phase_release}
        return handlers[phase](action, q, cams, base, inverse)

def entrypoint(): return Planner()

