"""spraytrees_v1: parent planner for BEHAVIOR task 39 (spraying_fruit_trees), H100 port, RGB-D only.

Goal: cover both garden trees with pesticide from the atomizer, which stands upright on the ground somewhere in the
garden (a dark green 16 cm can, red toggle ball on its head that turns green when on, grey nozzle cone beside it).
Input: the eight OBS_CONTRACT keys. Output: float32 (23,) native action (base 0:3, torso 3:7, left arm 7:14, left grip
14, right arm 15:22, right grip 22; grip +1 open).

Phases (log phase/stage/reason; every move to the next phase is decided by an observation, none by a timeout):
 0 search_a  spin, then explore frontiers (ground seen only within DETECT_R counts as searched) until the red or green
             head of the atomizer is seen 3 times at 0.02..0.32 m height.
 1 go_a      A* to a view standoff (atomizer VIEW_X ahead), replanned on every fresh sighting that moved it.
 2 align_a   fresh head sighting, then a straight holonomic creep that puts the atomizer at REACH_XY.
 3 reach_a   pinch PRE_DZ over the marker, the right wrist RGB-D re-centres it and reads the nozzle bearing; the grasp yaw
             keeps the nozzle between the fingers; then down to the marker height.
 4 close_a   close the right gripper; fingers on the head press the toggle; fingers stopped open = held, else retry.
 5 lift_a    carry pose; fingers closed fully = dropped -> search again.
 6 press_a   the can is shown to the head camera: green marker = on; red = put down, open and close in place, show again.
 7 search_t  carry pose, spin and explore until a tree is seen (thin isolated trunk under a high canopy).
 8 go_t      A* to a standoff that puts the trunk at SPRAY_SPOT.
 9 spray     fine alignment on the trunk seen fresh, then the nozzle sideways at SPRAY_H, SPRAY_R before the trunk axis,
             yaw aimed and swept for SPRAY_HOLD steps; then the next tree.
10 done      MAX_TREES tree candidates sprayed (look-alike garden trees exist), hold.
Kinematics: r1pro.urdf table (tttonyalpha/b1k_rl_guys 0adae10), damped least squares servo as in rgbd_spray_v1.
"""
import heapq
import math

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as R

PHASES = ("search_a", "go_a", "align_a", "reach_a", "close_a", "lift_a", "press_a", "search_t", "go_t", "spray", "done")
SEARCH_A, GO_A, ALIGN_A, REACH_A, CLOSE_A, LIFT_A, PRESS_A, SEARCH_T, GO_T, SPRAY, DONE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
DT = 1.0 / 30.0
YAW_SCALE = 0.81                     # proprio base wz over the true yaw rate is about 1.23 on the port
BASE_LIMITS = (0.75, 0.75, 1.0)
FLOOR = -0.018                       # floor height in the base frame
# map, odometry frame
RES, GRID = 0.1, 400                 # 40 m square around the start
OBST_Z = (0.08, 2.0)                 # points in this height band are obstacles (plant bases are low), m
OBST_RANGE = 5.0                     # only points this close mark obstacles (far ground reads high through depth and tilt error), m
INFLATE = (0.38, 0.3)                # robot radius plus margin, tighter fallback, m
VIEW_STRIDE, MAX_DEPTH = 4, 15.0
DETECT_R = 3.0                       # the red head is resolvable this far: ground seen closer counts as searched, m
EXPLORE_RANGE = (2.0, 12.0)          # mapped frontier goals may be farther from the robot, m
FAR_RANGE = (8.0, 14.0)              # committed depth-guided search legs, m
# atomizer
ATOM_TOP_Z = (0.02, 0.32)            # height band of the toggle marker (upright 0.16, fallen 0.03), m
RED = (100, 2.0)                     # R above, R over G and over B ratio
GREEN = (90, 1.5)
ATOM_MERGE = 0.35                    # detections closer than this are the same object, m
REACH_XY = ((0.63, -0.20), (0.58, -0.16), (0.68, -0.24))  # atomizer in the base frame at the grasp, tried in order
VIEW_X = 0.9                         # go_a stops with the atomizer this far ahead (the head sees it down to 0.7 m)
CREEP_SPEED, CREEP_TOL = 0.12, 0.015 # align_a: holonomic creep from VIEW_X to REACH_XY, m/s and m
RETARGET = 0.15                      # replan go_a when a fresh sighting moves the target this much, m
REPLAN_EVERY = 45                    # go_a/go_t replan on the fresh map this often (obstacles seen on the way), steps
# grasp
GRIP_POINT = (0.0, 0.0, -0.075)      # pinch centre in the gripper_link frame
PRE_DZ, REACH_TOL, WRIST_STABLE = 0.07, 0.012, 0.006  # pregrasp over the marker, reach tolerance, wrist fix agreement, m
MARKER_DZ, MARKER_SURF = -0.08, 0.012  # pinch height relative to the marker centre; seen red surface above the centre
CLOSE_STEPS, HOLD_GAP = 30, 0.004    # wait after closing; finger opening above this means an object is held
CARRY = (0.30, -0.25, 0.62)          # carry point of the pinch, base frame
ARM_STEP, TORSO_W, ROT_W, LIFT_TOL = 0.03, 0.5, 0.5, 0.04  # joint step, torso share, rotation weight, carry tolerance
GRASP_TRIES = 3
FLIP_YAW = 1                    # grasp yaw: 0 nozzle line along gripper x with the side seen nearest the base x axis, 1 the other side
CLOSE_RAMP = 20                   # steps over which the gripper command goes from open to closed
REACH_STEPS = 300                    # steps of wrist centring before the base is realigned on a new standoff
NOZZLE_GREY, NOZZLE_DZ, NOZZLE_R = 40, (-0.037, 0.013), (0.02, 0.05)  # grey level, height and radius window of the nozzle about the ball centre
NOZZLE_LOOKS = 3                     # gripper turns of 0.8 rad to find the nozzle in the wrist view
# toggle check: the held can is shown to the head camera, marker red = off, green = on
SHOW_DIST, SHOW_DOWN, SHOW_RIGHT = 0.6, 0.6, 0.35  # marker this far from the head camera, rad below and right of its axis
SHOW_TOL, SHOW_SWEEP, SHOW_MAX = 0.05, 240, 200    # position tolerance m; steps of the gripper yaw sweep; steps to reach the pose
SHOW_YAW = 0.9                       # yaw sweep half-width about the start yaw (the wrist cannot turn a full circle), rad
MARKER_UP = 0.076                    # marker centre above the pinch along the gripper z axis, m
MARKER_NEAR, MARKER_PIX = 0.06, 8    # red/green points this close to the expected marker count; minimum to decide
PUT_UP = 0.01                        # put-down pinch height above the grasp height, m
OPEN_STEPS = 15                      # red marker: the can is put down, the fingers open this long and close again in place
RECLOSE_RAMPS = (6, 20, 40)          # closing ramps of successive re-closes (a bounce can flip the toggle twice)
RECLOSE_MAX = 2                      # bound unreliable in-place toggle retries
# trees: the targets are broad trees, bare trunk to about 2.7 m, crown up to 6 m and 5 m wide
CANOPY_Z, TRUNK_Z, TRUNK_MID = 3.2, (0.35, 2.3), 1.4   # a trunk fills both bands below and above TRUNK_MID
TCELL = 0.25                         # tree grid cell, m
CANOPY_MIN, ISOLATE_R, CANOPY_R = 80, 4, 10  # canopy points; isolation ring and canopy radius in tree cells
SPRAY_SPOT, SPOT_TOL = (0.70, -0.20), 0.05   # trunk axis in the base frame before the spray, tolerance
SPRAY_H = 0.73                       # nozzle height in the base frame (0.75 m over the floor), m
SPRAY_R, SPRAY_TOL = 0.28, 0.035     # pinch to trunk axis distance, position tolerance, m
SPRAY_HOLD, SPRAY_PERIOD = 240, 120  # steps held in tolerance, period of the yaw sweep
SPRAY_SWEEP, SPRAY_YAW = 0.6, 2.3    # yaw sweep half-width with a known nozzle side, without one
SPRAY_FAILS = 3                      # stands without reaching the aim before a tree is given up
TREE_DONE_R = 3.0
TREE_MERGE = 1.0                     # a trunk seen in the current head frame this close to the target is the same tree, m
TRACK_EVERY = 15                     # go_t re-anchors the target on the trunk in view this often, steps
CROWN_SPREAD = 3.5                   # the target crown spans at least this much in both map axes, m
MAX_TREES = 2                        # the task has exactly two target trees
# motion
SPIN_STEPS, SPIN_RATE = 330, 0.6
STALL_STEPS, STALL_MOVE = 90, 0.05
BLOCK_STEPS, BLOCK_SPEED, BLOCK_CHANGE = 30, 0.1, 0.05  # driving this long at this speed (m/s) with fewer than this share of head depth pixels changing = blocked
BLOCK_DEPTH = 8.0                    # only objects this close judge the motion, m
BLOCK_MARK = (0.6, 0.35)             # a blocked robot marks an obstacle this far ahead with this radius, m
JOINTS = (  # r1pro.urdf chains base -> both grippers and fingers (dof index into q, -1 fixed)
    (0, 'base_link', 'torso_link1', 'r', (-0.079032, 0.0, 0.34265), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (1, 'torso_link1', 'torso_link2', 'r', (0.0, 0.0, 0.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (2, 'torso_link2', 'torso_link3', 'r', (0.0, 0.000101, 0.3), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)), (3, 'torso_link3', 'torso_link4', 'r', (0.0, -0.0001, 0.09962), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (-1, 'torso_link4', 'left_arm_base_link', 'f', (-0.000486, 0.097234, 0.30302), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (4, 'left_arm_base_link', 'left_arm_link1', 'r', (0.0, 0.0735, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (5, 'left_arm_link1', 'left_arm_link2', 'r', (0.025012, 0.081265, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (6, 'left_arm_link2', 'left_arm_link3', 'r', (-0.025012, 0.0, -0.1155), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (7, 'left_arm_link3', 'left_arm_link4', 'r', (0.0, 0.035065, -0.1945), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (8, 'left_arm_link4', 'left_arm_link5', 'r', (0.0, -0.035065, -0.095), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (9, 'left_arm_link5', 'left_arm_link6', 'r', (0.0, -0.028, -0.16305), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (10, 'left_arm_link6', 'left_arm_link7', 'r', (0.0295, 0.028, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (-1, 'left_arm_link7', 'left_gripper_link', 'f', (-0.0295, 0.0, -0.16065), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), (18, 'left_gripper_link', 'left_gripper_finger_link1', 'p', (-8.9e-05, 0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (19, 'left_gripper_link', 'left_gripper_finger_link2', 'p', (8.9e-05, -0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)), (-1, 'torso_link4', 'right_arm_base_link', 'f', (-0.000487, -0.097236, 0.30302), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (11, 'right_arm_base_link', 'right_arm_link1', 'r', (0.0, -0.0735, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (12, 'right_arm_link1', 'right_arm_link2', 'r', (0.023988, -0.081265, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (13, 'right_arm_link2', 'right_arm_link3', 'r', (-0.023988, 0.0, -0.1155), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (14, 'right_arm_link3', 'right_arm_link4', 'r', (0.0, 0.035058, -0.1945), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (15, 'right_arm_link4', 'right_arm_link5', 'r', (0.0, -0.035058, -0.095), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (16, 'right_arm_link5', 'right_arm_link6', 'r', (0.0, 0.028001, -0.16305), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (17, 'right_arm_link6', 'right_arm_link7', 'r', (0.0295, -0.028001, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (-1, 'right_arm_link7', 'right_gripper_link', 'f', (-0.0295, 0.0, -0.16065), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (20, 'right_gripper_link', 'right_gripper_finger_link1', 'p', (-8.9e-05, 0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (21, 'right_gripper_link', 'right_gripper_finger_link2', 'p', (8.9e-05, -0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
)
QLO = (-1.1345, -2.7925, -1.8326, -3.0543, -4.4506, -0.1745, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708, -4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708, 0.0, 0.0, 0.0, 0.0)
QHI = (1.8326, 2.5307, 1.5708, 3.0543, 1.309, 3.1416, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708, 1.309, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708, 0.05, 0.05, 0.05, 0.05)
TORSO, LEFT, RIGHT = (0, 1, 2, 3), (4, 5, 6, 7, 8, 9, 10), (11, 12, 13, 14, 15, 16, 17)
FOCAL = (388.6639 / 480, 388.6639 / 480, 0.425)  # left wrist, right wrist, head: fx / image width
NBRS = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rot2(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def red(col):  # toggle marker off, port frames (255, 0, 0)
    return (col[:, 0] > 150) & (col[:, 1] < 40) & (col[:, 2] < 40)


def green(col):  # toggle marker on, port frames (0, 255, 0) shaded
    return (col[:, 1] > 110) & (col[:, 1] > 1.8 * np.maximum(col[:, 0], col[:, 2]))


def dijkstra(free, start, box):
    """Cost and parents over an 8-connected grid inside box = (i0, i1, j0, j1)."""
    dist = np.full(free.shape, np.inf)
    prev = np.full(free.shape + (2,), -1, dtype=int)
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        cost, (i, j) = heapq.heappop(heap)
        if cost > dist[i, j]:
            continue
        for di, dj, step in NBRS:
            a, b = i + di, j + dj
            if not (box[0] <= a < box[1] and box[2] <= b < box[3]) or not free[a, b]:
                continue
            if di and dj and not (free[i + di, j] and free[i, j + dj]):
                continue
            if cost + step < dist[a, b]:
                dist[a, b] = cost + step
                prev[a, b] = (i, j)
                heapq.heappush(heap, (cost + step, (a, b)))
    return dist, prev


class Kin:
    """Forward kinematics and geometric Jacobians of the JOINTS table (q: torso 4, left 7, right 7, fingers 4)."""

    def __init__(self):
        self.joints = []
        for dof, parent, child, kind, xyz, rpy, axis in JOINTS:
            t = np.eye(4)
            t[:3, 3], t[:3, :3] = xyz, R.from_euler('xyz', rpy).as_matrix()
            self.joints.append((dof, parent, child, kind, t, np.asarray(axis)))

    def forward(self, q):
        poses, axes, anc = {'base_link': np.eye(4)}, {}, {'base_link': []}
        for dof, parent, child, kind, origin, axis in self.joints:
            t = poses[parent] @ origin
            m = np.eye(4)
            if kind == 'r':
                m[:3, :3] = R.from_rotvec(axis * q[dof]).as_matrix()
            elif kind == 'p':
                m[:3, 3] = axis * q[dof]
            if dof >= 0:
                axes[dof] = (t[:3, 3], t[:3, :3] @ axis, kind)
            poses[child] = t @ m
            anc[child] = anc[parent] + ([dof] if dof >= 0 else [])
        return poses, axes, anc

    def jacobian(self, link, point, cache):
        poses, axes, anc = cache
        t = poses[link]
        xyz = t[:3, 3] + t[:3, :3] @ point
        jac = np.zeros((6, 22))
        for dof in anc[link]:
            pos, axis, kind = axes[dof]
            if kind == 'p':
                jac[:3, dof] = axis
            else:
                jac[:3, dof], jac[3:, dof] = np.cross(axis, xyz - pos), axis
        return jac


class Planner:
    """Phase machine of the module docstring. All state lives in this object."""

    def __init__(self):
        self.kin = Kin()
        self.reset(None)

    def reset(self, task_spec):
        self.step_no, self.phase, self.age, self.sub, self.reason = 0, 0, 0, 0, "start"
        self.pos, self.yaw = np.zeros(2), 0.0
        self.obst = np.zeros((GRID, GRID), np.float32)
        self.seen = np.zeros((GRID, GRID), bool)
        tg = int(GRID * RES / TCELL)
        self.canopy, self.low, self.mid = np.zeros((tg, tg), np.float32), np.zeros((tg, tg), np.float32), np.zeros((tg, tg), np.float32)
        self.cmd = self.rel = self.views = self.p = self.q = None
        self.atom, self.goal_atom, self.creep = None, None, None
        self.atom_hits, self.grasp_tries, self.reach_k, self.trees_done = 0, 0, 0, []
        self.grip_yaw, self.nozzle, self.wfix, self.stable, self.looks = 0.0, 0, None, 0, 0
        self.marker, self.hold, self.spray_fails, self.err = (0, 0), 0, 0, 0.0
        self.tree, self.show, self.pick, self.grasp_yaw = None, None, np.zeros(2), 0.0
        self.path, self.pi, self.goal_yaw, self.anchor = [], 0, 0.0, (np.zeros(2), 0)
        self.spin_left = SPIN_STEPS
        self.search_dir = -1
        self.marks = np.zeros((GRID, GRID), bool)
        self.depth_ref, self.pos_ref, self.drive_n, self.blocked, self.last_v = None, self.pos.copy(), 0, False, np.zeros(2)

    # ---- perception ------------------------------------------------------------------------------------------
    def cloud(self, cam, mask_fn, stride, near=0.06):
        """Base-frame points and colours of one camera (pixels kept by mask_fn, then every stride-th)."""
        rgb, depth = self.views[cam]
        w = depth.shape[1]
        rel = self.rel[cam]
        ok = np.isfinite(depth) & (depth > near) & (depth < MAX_DEPTH)
        if mask_fn is not None:
            ok &= mask_fn(rgb)
        v, u = np.nonzero(ok)
        v, u = v[::stride], u[::stride]
        z = depth[v, u]
        f = FOCAL[cam] * w
        pc = np.stack([(u + 0.5 - w / 2.0) / f * z, (v + 0.5 - w / 2.0) / f * z, z], -1)
        rot = R.from_quat(rel[3:7]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
        return pc @ rot.T + rel[:3], rgb[v, u].astype(np.float32)

    def to_odom(self, p):
        return p[:, :2] @ rot2(self.yaw).T + self.pos

    def to_base(self, xy):
        return rot2(-self.yaw) @ (np.asarray(xy) - self.pos)

    def update_map(self):
        pts, _ = self.cloud(2, None, 1)
        pts = pts[:: VIEW_STRIDE * VIEW_STRIDE]
        xy = self.to_odom(pts)
        ij = np.floor((xy + GRID * RES / 2) / RES).astype(int)
        ok = ((ij >= 0) & (ij < GRID)).all(1) & (np.hypot(pts[:, 0], pts[:, 1]) > 0.5)
        ij, z, pts = ij[ok], pts[ok, 2], pts[ok]
        close = np.hypot(pts[:, 0], pts[:, 1]) < DETECT_R
        self.seen[ij[close, 0], ij[close, 1]] = True
        hi = (z > OBST_Z[0]) & (z < OBST_Z[1]) & (np.hypot(pts[:, 0], pts[:, 1]) < OBST_RANGE)
        np.add.at(self.obst, (ij[hi, 0], ij[hi, 1]), 1.0)
        tij = np.floor((xy[ok] + GRID * RES / 2) / TCELL).astype(int)
        can, low = z > CANOPY_Z, (z > TRUNK_Z[0]) & (z < TRUNK_Z[1])
        np.add.at(self.canopy, (tij[can, 0], tij[can, 1]), 1.0)
        np.add.at(self.low, (tij[low, 0], tij[low, 1]), 1.0)
        mid = low & (z > TRUNK_MID)
        np.add.at(self.mid, (tij[mid, 0], tij[mid, 1]), 1.0)

    def find_marker(self, colour):
        """Odometry xy, mean height and base-frame centroid of red (or green) pixels in the marker height band."""
        lim, ratio = RED if colour == "red" else GREEN
        ch = 0 if colour == "red" else 1

        def mask(rgb):
            a = rgb.astype(np.float32)
            others = np.delete(a, ch, axis=-1)
            return (a[..., ch] > lim) & (a[..., ch] > ratio * others.max(-1))
        best = None
        for cam in range(3):
            pts, _ = self.cloud(cam, mask, 1)
            pts = pts[(pts[:, 2] > ATOM_TOP_Z[0]) & (pts[:, 2] < ATOM_TOP_Z[1])]
            if len(pts) >= 2 and (best is None or len(pts) > len(best)):
                best = pts
        if best is None:
            return None
        med = np.median(best, 0)
        best = best[np.linalg.norm(best - med, axis=1) < 0.08]
        return None if len(best) < 2 else (self.to_odom(best).mean(0), float(best[:, 2].mean()), best.mean(0))

    def sight(self):
        return self.find_marker("red") or self.find_marker("green")

    def wrist_fix(self):
        """Find either toggle colour and estimate the nozzle bearing from the right wrist RGB-D."""
        pts, col = self.cloud(1, None, 2, 0.03)
        m = red(col) | green(col)
        if m.sum() < 15:
            return None, None
        c = np.median(pts[m], 0)
        d = pts - c + np.array([0.0, 0.0, MARKER_SURF])
        rr = np.hypot(d[:, 0], d[:, 1])
        grey = (col.max(1) > NOZZLE_GREY) & (col.max(1) - col.min(1) < 25)   # nozzle cone: not the black fingers, not the green body
        noz = ~m & grey & (d[:, 2] > NOZZLE_DZ[0]) & (d[:, 2] < NOZZLE_DZ[1]) & (rr > NOZZLE_R[0]) & (rr < NOZZLE_R[1])
        return c, (math.atan2(*d[noz, :2].mean(0)[::-1]) if noz.sum() >= 5 else None)

    def find_tree(self):
        """Odometry xy of the trunk of a tree not sprayed yet: isolated low cell under canopy, or None."""
        occ = self.low > 3
        lab, n = ndimage.label(occ)
        best = None
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) > 2:
                continue
            ci, cj = [(s.start + s.stop - 1) / 2.0 for s in sl]
            i0, j0 = int(ci), int(cj)
            ring = occ[max(0, i0 - ISOLATE_R):i0 + ISOLATE_R + 1, max(0, j0 - ISOLATE_R):j0 + ISOLATE_R + 1].sum()
            can = self.canopy[max(0, i0 - CANOPY_R):i0 + CANOPY_R + 1, max(0, j0 - CANOPY_R):j0 + CANOPY_R + 1].sum()
            if ring - (lab == k).sum() > 0 or can < CANOPY_MIN or self.mid[sl].sum() < 2 or self.low[sl].sum() - self.mid[sl].sum() < 2:
                continue
            ci_, cj_ = np.nonzero(self.canopy[max(0, i0 - CANOPY_R):i0 + CANOPY_R + 1, max(0, j0 - CANOPY_R):j0 + CANOPY_R + 1] > 2)
            if len(ci_) == 0 or min(np.ptp(ci_), np.ptp(cj_)) * TCELL < CROWN_SPREAD:
                continue  # small crowns: look-alike garden trees
            xy = (np.array([ci, cj]) + 0.5) * TCELL - GRID * RES / 2
            if any(np.linalg.norm(xy - d) < TREE_DONE_R for d in self.trees_done):
                continue
            if best is None or can > best[1]:
                best = (xy, can)
        return None if best is None else best[0]

    # ---- arm -------------------------------------------------------------------------------------------------
    def servo(self, goal, rd=None, torso_w=TORSO_W):
        """One damped least squares step of the right pinch point to goal (base frame) and the gripper rotation to rd
        (default: approach down, fingers closing along base y). Writes torso and right arm targets into the command;
        returns the position and rotation errors."""
        idx, q = list(TORSO + RIGHT), self.q
        cache = self.kin.forward(q)
        t = cache[0]["right_gripper_link"]
        err = np.asarray(goal) - (t[:3, 3] + t[:3, :3] @ np.asarray(GRIP_POINT))
        rv = R.from_matrix((np.eye(3) if rd is None else rd) @ t[:3, :3].T).as_rotvec()
        e = np.r_[err * min(1.0, 0.03 / max(np.linalg.norm(err), 1e-9)), ROT_W * np.clip(rv, -0.2, 0.2)]
        w = np.array([torso_w if i < 4 else 1.0 for i in idx])
        jw = self.kin.jacobian("right_gripper_link", np.asarray(GRIP_POINT), cache)[:, idx] * w
        dq = w * (jw.T @ np.linalg.solve(jw @ jw.T + 1e-3 * np.eye(6), e))
        tgt = np.clip(q[idx] + np.clip(dq, -ARM_STEP, ARM_STEP), np.array(QLO)[idx], np.array(QHI)[idx])
        for k, i in enumerate(idx):
            self.cmd[3 + i if i < 4 else 4 + i] = tgt[k]
        return float(np.linalg.norm(err)), float(np.linalg.norm(rv))

    def held(self):
        return float(np.mean(self.p[49:51])) >= HOLD_GAP

    # ---- navigation ------------------------------------------------------------------------------------------
    def cell(self, xy):
        return tuple(np.clip(np.floor((np.asarray(xy) + GRID * RES / 2) / RES).astype(int), 0, GRID - 1))

    def plan(self, goals):
        """Grid path to the cheapest reachable of goals [(xy, yaw)]."""
        start = self.cell(self.pos)
        for infl in INFLATE:
            r = int(round(infl / RES))
            yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
            free = ~ndimage.binary_dilation(self.obst > 2, structure=(xx * xx + yy * yy <= r * r))
            free[max(0, start[0] - 3):start[0] + 4, max(0, start[1] - 3):start[1] + 4] = True
            cells = [self.cell(g) for g, _ in goals] + [start]
            box = (max(0, min(c[0] for c in cells) - 30), min(GRID, max(c[0] for c in cells) + 31),
                   max(0, min(c[1] for c in cells) - 30), min(GRID, max(c[1] for c in cells) + 31))
            dist, prev = dijkstra(free, start, box)
            ok = [(dist[self.cell(g)], g, y) for g, y in goals if np.isfinite(dist[self.cell(g)])]
            if ok:
                _, g, y = min(ok, key=lambda t: t[0])
                c, cells = self.cell(g), []
                while c != start:
                    cells.append((np.array(c) + 0.5) * RES - GRID * RES / 2)
                    c = tuple(prev[c])
                self.path, self.pi, self.goal_yaw = cells[::-1][:-1] + [np.asarray(g)], 0, y
                self.anchor = (self.pos.copy(), self.step_no)
                return True
        return False

    def drive(self):
        """(vx, vy, wz, arrived) along self.path, velocity in the robot frame."""
        if not self.path:
            return 0.0, 0.0, 0.0, True
        while self.pi < len(self.path) - 1 and np.linalg.norm(self.path[self.pi] - self.pos) < 0.3:
            self.pi += 1
        vec = self.path[self.pi] - self.pos
        dist = float(np.linalg.norm(vec))
        final = self.pi == len(self.path) - 1
        heading = self.goal_yaw if final and dist < 0.3 else math.atan2(vec[1], vec[0])
        err = wrap(heading - self.yaw)
        wz = float(np.clip(2.0 * err, -0.8, 0.8))
        if final and dist < 0.06:
            return 0.0, 0.0, wz, abs(err) < 0.05
        speed = min(0.6, 1.2 * dist) * (0.25 if abs(err) > 0.7 else 1.0)
        v = rot2(-self.yaw) @ (vec / max(dist, 1e-6) * speed)
        return float(v[0]), float(v[1]), wz, False

    def watch_motion(self, base):
        """Blocked test from the head depth: the base moves in a held-back pose while proprio keeps reporting speed."""
        self.last_v = base[:2].copy() if float(np.linalg.norm(base[:2])) > BLOCK_SPEED else self.last_v
        self.drive_n += int(float(np.linalg.norm(base[:2])) > BLOCK_SPEED and abs(base[2]) < 0.4)
        if self.step_no % BLOCK_STEPS:
            return
        d = self.views[2][1][:360:24, ::24]  # upper half: the ground does not change with translation
        ok = np.isfinite(d) & (d < BLOCK_DEPTH)
        if self.depth_ref is not None and self.drive_n >= 0.8 * BLOCK_STEPS:
            both = ok & self.depth_ref[1]
            changed = float(np.mean(np.abs(d - self.depth_ref[0])[both] > 0.03)) if both.sum() > 50 else 1.0
            if changed < BLOCK_CHANGE:
                self.pos = self.pos_ref.copy()  # no real displacement: undo the phantom odometry
                ahead = self.pos + rot2(math.atan2(self.last_v[1], self.last_v[0]) + self.yaw) @ np.array([BLOCK_MARK[0], 0.0])
                ij = np.array(self.cell(ahead))
                r = int(BLOCK_MARK[1] / RES)
                lo, hi = np.clip(ij - r, 0, GRID), np.clip(ij + r + 1, 0, GRID)
                self.obst[lo[0]:hi[0], lo[1]:hi[1]] += 3.0
                self.marks[lo[0]:hi[0], lo[1]:hi[1]] = True
                self.blocked = True
        self.depth_ref, self.pos_ref, self.drive_n = (d.copy(), ok), self.pos.copy(), 0

    def forget_marks(self):
        self.obst[self.marks] = 0.0
        self.marks[:] = False

    def stalled(self):
        if self.blocked:
            self.blocked, self.reason = False, "blocked_marked_ahead"
            return True
        if self.step_no - self.anchor[1] < STALL_STEPS:
            return False
        moved = float(np.linalg.norm(self.pos - self.anchor[0]))
        self.anchor = (self.pos.copy(), self.step_no)
        return moved < STALL_MOVE

    def explore(self):
        """Choose a far depth-guided leg before falling back to mapped frontier cells."""
        pts, _ = self.cloud(2, None, 6)
        rng = np.hypot(pts[:, 0], pts[:, 1])
        valid = np.isfinite(rng) & (rng > 0.6) & (rng < MAX_DEPTH)
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        scores = []
        for s in range(8):
            centre = self.yaw + s * math.pi / 4.0
            delta = np.arctan2(np.sin(angles[valid] - centre), np.cos(angles[valid] - centre))
            rr = rng[valid][np.abs(delta) < math.pi / 8.0]
            scores.append(float(np.percentile(rr, 75)) if len(rr) > 8 else 0.0)

        order = sorted(range(8), key=lambda s: (-scores[s], s == self.search_dir, s))
        for s in order:
            if len(order) > 1 and s == self.search_dir:
                continue
            centre = self.yaw + s * math.pi / 4.0
            distance = float(np.clip(scores[s] + 2.0, FAR_RANGE[0], FAR_RANGE[1]))
            goal = self.pos + rot2(centre) @ np.array([distance, 0.0])
            if self.plan([(goal, centre)]):
                self.search_dir = s
                return True

        start = self.cell(self.pos)
        edge = ndimage.binary_dilation(~self.seen, iterations=2) & self.seen & (self.obst < 1)
        ii, jj = np.nonzero(edge)
        d = np.hypot(ii - start[0], jj - start[1]) * RES
        k = np.flatnonzero((d > EXPLORE_RANGE[0]) & (d < EXPLORE_RANGE[1]))
        if len(k):
            pick = k[np.argsort(d[k])[::-1]][:: max(1, len(k) // 12)]
            goals = [((np.array([ii[t], jj[t]]) + 0.5) * RES - GRID * RES / 2, self.yaw) for t in pick]
            if self.plan(goals):
                return True

        candidates = [s for s in order if s != self.search_dir]
        if not candidates:
            candidates = order
        if candidates:
            s = candidates[0]
            centre = self.yaw + s * math.pi / 4.0
            distance = float(np.clip(scores[s] + 2.0, FAR_RANGE[0], FAR_RANGE[1]))
            goal = self.pos + rot2(centre) @ np.array([distance, 0.0])
            self.path, self.pi, self.goal_yaw = [goal], 0, centre
            self.search_dir = s
            self.anchor = (self.pos.copy(), self.step_no)
            return True
        return False

    def go(self, phase, reason, keep_path=False):
        """Enter a phase; reserve a full scan for tree-search launches only."""
        self.phase, self.age, self.sub, self.reason = phase, 0, 0, reason
        if not keep_path:
            self.path = []
        self.spin_left = SPIN_STEPS if phase == SEARCH_T else 0
        self.stable, self.wfix, self.looks, self.hold, self.show = 0, None, 0, 0, None

    def search(self, found):
        """Scan when required, then commit to new far legs without endless spin retries."""
        if self.step_no % 3 == 0 and found():
            return np.zeros(3)
        if self.spin_left > 0:
            self.spin_left -= 1
            self.reason = "spinning"
            return np.array([0.0, 0.0, SPIN_RATE])
        if not self.path:
            self.reason = "exploring" if self.explore() else "no_reachable_search_leg"
            if not self.path:
                self.search_dir = (self.search_dir + 1) % 8
                self.spin_left = 0
        vx, vy, wz, arrived = self.drive()
        if arrived:
            self.path = []
            self.spin_left = SPIN_STEPS if self.phase == SEARCH_T else 0
            self.reason = "drive_complete_next_leg"
        elif self.stalled():
            self.path = []
            if float(np.linalg.norm(self.last_v)) > BLOCK_SPEED:
                self.search_dir = int(round(math.atan2(self.last_v[1], self.last_v[0]) / (math.pi / 4.0))) % 8
            self.spin_left = 0
            self.reason = "blocked_new_bearing"
        return np.array([vx, vy, wz])

    # ---- atomizer phases -------------------------------------------------------------------------------------
    def atom_found(self):
        hit = self.sight()
        if hit is None:
            return False
        if self.atom is not None and np.linalg.norm(hit[0] - self.atom[0]) < ATOM_MERGE:
            self.atom = [0.7 * self.atom[0] + 0.3 * hit[0], hit[1] - MARKER_SURF]
            self.atom_hits += 1
        else:
            self.atom, self.atom_hits = [hit[0], hit[1] - MARKER_SURF], 1
        if self.atom_hits >= 3:
            return self.plan_atom()
        return False

    def approach_goals(self, target, spot, spread, n):
        """Standoff goals [(xy, yaw)] that put target at spot of the base frame, headings around the bearing from here."""
        b = target - self.pos
        centre = math.atan2(b[1], b[0]) - math.atan2(spot[1], spot[0])
        return [(target - rot2(a) @ np.asarray(spot), a) for a in centre + np.linspace(-spread, spread, n)]

    def plan_atom(self):
        """Path to a view standoff: the atomizer VIEW_X ahead, at the lateral offset of the grasp."""
        a = self.atom[0]
        ry = REACH_XY[self.reach_k % len(REACH_XY)][1]
        if self.plan(self.approach_goals(a, (VIEW_X, ry), 1.2, 7)) or self.plan(self.approach_goals(a, (VIEW_X, ry), math.pi, 16)):
            self.go(GO_A, "atomizer_seen", True)
            self.goal_atom = a.copy()
            return True
        self.reason = "atomizer_unreachable"
        return False

    def rebase_press(self):
        """Rebase a held-can toggle retry to the current gripper pose."""
        t = self.kin.forward(self.q)[0]["right_gripper_link"]
        pinch = t[:3, 3] + t[:3, :3] @ np.asarray(GRIP_POINT)
        self.pick = pinch[:2].copy()
        self.atom[0] = self.to_odom(pinch[None])[0]
        self.atom[1] = float(pinch[2] - MARKER_DZ - PUT_UP)
        self.show = None
        self.tree = None
        self.go(PRESS_A, "marker_red_before_spray")

    def ph_go(self, nxt):
        """Replan navigation and revalidate the held atomizer before spraying."""
        if nxt == ALIGN_A and self.step_no % 3 == 0:
            hit = self.sight()
            if hit is not None and np.linalg.norm(hit[0] - self.atom[0]) < 3 * ATOM_MERGE:
                self.atom = [hit[0], hit[1] - MARKER_SURF]  # a fresh sighting beats the odometry-carried estimate
                if np.linalg.norm(hit[0] - self.goal_atom) > RETARGET:
                    self.reason = "retarget"
                    if not self.plan_atom():
                        self.go(SEARCH_A, "retarget_unreachable")
                    return np.zeros(3)
        if nxt == SPRAY and self.age % TRACK_EVERY == 0:
            red_count, green_count = self.marker_seen()
            if red_count >= MARKER_PIX and red_count > 2 * green_count:
                self.rebase_press()
                return np.zeros(3)
            self.trunk_in_view()
        if self.age % REPLAN_EVERY == 0 and len(self.path) - self.pi > 3:
            self.reason = "periodic_replan"
            if not (self.plan_atom() if nxt == ALIGN_A else self.plan_tree()):
                self.reason = "periodic_replan_failed_keep_path"
        vx, vy, wz, arrived = self.drive()
        if arrived:
            self.go(nxt, "arrived")
            return np.zeros(3)
        if self.stalled():
            self.reason = "stalled_replan"
            ok = self.plan_atom() if nxt == ALIGN_A else self.plan_tree()
            if not ok:
                self.forget_marks()
                self.go(SEARCH_A if nxt == ALIGN_A else SEARCH_T, "replan_failed")
        return np.array([vx, vy, wz])

    def ph_align(self):
        """Fresh head sighting at the view standoff, then a straight holonomic creep that puts it at REACH_XY."""
        if self.creep is None:
            hit = self.sight()
            if hit is None:
                self.reason = "view_marker_unseen"
                if self.age > 30:
                    self.go(SEARCH_A, "lost_at_view_standoff")
                return np.zeros(3)
            self.atom = [hit[0], hit[1] - MARKER_SURF]
            rx, ry = REACH_XY[self.reach_k % len(REACH_XY)]
            self.creep = (self.pos.copy(), self.yaw, hit[2][:2] - np.array([rx, ry]))
        start, yaw0, d = self.creep
        left = d - rot2(-yaw0) @ (self.pos - start)
        self.reason = "creeping"
        if float(np.linalg.norm(left)) < CREEP_TOL:
            self.creep = None
            self.go(REACH_A, "at_grasp_standoff")
            return np.zeros(3)
        v = left / max(float(np.linalg.norm(left)), 1e-6) * min(CREEP_SPEED, 1.5 * float(np.linalg.norm(left)))
        return np.array([v[0], v[1], float(np.clip(2.0 * wrap(yaw0 - self.yaw), -0.3, 0.3))])

    def ph_reach(self):
        """sub 0: pinch PRE_DZ over the marker, the wrist camera re-centres it and reads the nozzle bearing; the grasp yaw
        puts the nozzle between the fingers. sub 1: down to the marker height."""
        self.cmd[22] = 1.0
        b = self.to_base(self.atom[0])
        rd = R.from_euler("z", self.grip_yaw).as_matrix()
        if self.sub == 0:
            err, rerr = self.servo(np.r_[b, self.atom[1] + PRE_DZ], rd)
            self.err = err
            if self.step_no % 3 == 0 and err < 0.05:
                c, ang = self.wrist_fix()
                if c is not None and np.linalg.norm(c[:2] - b) < 0.12:
                    self.stable = self.stable + 1 if self.wfix is not None and np.linalg.norm(c - self.wfix) < WRIST_STABLE else 0
                    self.wfix, self.atom = c, [self.to_odom(c[None])[0], c[2] - MARKER_SURF]
                    if ang is not None:
                        yaw = wrap(ang) if (abs(wrap(ang)) <= math.pi / 2) != bool(FLIP_YAW) else wrap(ang + math.pi)
                        self.grip_yaw, self.nozzle = yaw, (1 if abs(wrap(ang - yaw)) < 0.1 else -1)
            if err < REACH_TOL and rerr < 0.1 and self.stable >= 3 and self.nozzle == 0 and self.looks < NOZZLE_LOOKS:
                self.grip_yaw, self.looks, self.stable = wrap(self.grip_yaw + 0.8), self.looks + 1, 0
                self.reason = "nozzle_not_seen_turning"
            elif err < REACH_TOL and rerr < 0.1 and self.stable >= 3:
                self.sub, self.reason = 1, "marker_centred_descending"
            elif self.age > REACH_STEPS:
                self.reach_k += 1
                if not self.plan_atom():
                    self.go(SEARCH_A, "centring_failed_search")
            else:
                self.reason = "over_atomizer_centring"
            return
        err, rerr = self.servo(np.r_[b, self.atom[1] + MARKER_DZ], rd)
        self.err = err
        if err < REACH_TOL and rerr < 0.1:
            return self.go(CLOSE_A, "pinch_at_head")
        self.reason = "descending"

    def ph_close(self):
        self.cmd[22] = max(-1.0, 1.0 - 2.0 * self.age / max(CLOSE_RAMP, 1))
        gap = float(np.mean(self.p[49:51]))
        self.reason = "closing"
        if self.age < CLOSE_STEPS:
            return
        if gap > HOLD_GAP:
            t = self.kin.forward(self.q)[0]["right_gripper_link"]
            self.pick, self.grasp_yaw = (t[:3, 3] + t[:3, :3] @ np.asarray(GRIP_POINT))[:2], self.grip_yaw
            return self.go(LIFT_A, "fingers_stopped_open")
        self.grasp_tries += 1
        self.cmd[22] = 1.0
        if self.grasp_tries % GRASP_TRIES:
            self.go(REACH_A, "grasp_missed_retry")
        else:
            self.go(SEARCH_A, "grasp_missed_search")

    def ph_lift(self):
        self.cmd[22] = -1.0
        err, _ = self.servo(CARRY, R.from_euler("z", self.grip_yaw).as_matrix())
        self.err = err
        if not self.held():
            self.cmd[22] = 1.0
            return self.go(SEARCH_A, "dropped_while_lifting")
        if err < LIFT_TOL:
            return self.go(PRESS_A, "lifted_and_held")
        self.reason = "lifting"

    def carry(self):
        """Hold the atomizer at CARRY; False (and back to search_a) when the fingers closed fully (dropped)."""
        self.cmd[22] = -1.0
        self.servo(CARRY, R.from_euler("z", self.grip_yaw).as_matrix())
        if self.held():
            return True
        self.cmd[22] = 1.0
        self.go(SEARCH_A, "dropped_in_" + PHASES[self.phase])
        return False

    def marker_seen(self):
        """(red, green) camera points within MARKER_NEAR of the marker expected from the held pinch pose."""
        t = self.kin.forward(self.q)[0]["right_gripper_link"]
        exp = t[:3, 3] + t[:3, :3] @ (np.asarray(GRIP_POINT) + np.array([0.0, 0.0, MARKER_UP]))
        out = [0, 0]
        for cam in range(3):
            pts, col = self.cloud(cam, None, 1, 0.03)
            near = np.linalg.norm(pts - exp, axis=1) < MARKER_NEAR
            out[0] += int((near & red(col)).sum())
            out[1] += int((near & green(col)).sum())
        return out

    def show_goal(self):
        """Pinch goal and start gripper yaw that put the marker before the head camera, the finger gap facing it."""
        cam = self.rel[2]
        ax = R.from_quat(cam[3:7]).as_matrix() @ np.array([0.0, 0.0, -1.0])
        h = rot2(-SHOW_RIGHT) @ (ax[:2] / max(float(np.linalg.norm(ax[:2])), 1e-6))
        mk = cam[:3] + SHOW_DIST * np.r_[math.cos(SHOW_DOWN) * h, -math.sin(SHOW_DOWN)]
        toward = math.atan2(-h[1], -h[0]) + (math.pi if self.nozzle > 0 else 0.0)  # gripper x at the camera, nozzle away
        return mk - np.array([0.0, 0.0, MARKER_UP]), toward

    def ph_press(self):
        """The head grasp presses the toggle. sub 0: the held can is shown to the head camera while the gripper yaw sweeps
        once around (the marker hides behind a finger at some yaws); green marker = on -> trees; red = sub 1 put down
        where it was picked, sub 2 open in place, sub 3 close again (a fresh press), lift, show again."""
        if self.sub == 0:
            self.cmd[22] = -1.0
            if not self.held():
                self.cmd[22] = 1.0
                return self.go(SEARCH_A, "dropped_while_showing")
            if self.show is None:
                self.show, self.hold, self.marker = self.show_goal(), 0, (0, 0)
            goal, yaw0 = self.show
            yaw = wrap(yaw0 + SHOW_YAW * math.sin(2.0 * math.pi * self.hold / SHOW_SWEEP))
            err, _ = self.servo(goal, R.from_euler("z", yaw).as_matrix(), 0.0)
            self.err = err
            self.hold += int(self.hold > 0 or err < SHOW_TOL)  # the sweep clock runs once the pose was reached
            if err < SHOW_TOL:
                seen = self.marker_seen()
                self.marker = (self.marker[0] + seen[0], self.marker[1] + seen[1])
                self.reason = "showing_marker_sweep"
            elif self.hold == 0 and self.age > SHOW_MAX:
                self.fresh_maps()
                return self.go(SEARCH_T, "show_pose_unreachable_going_on")
            else:
                self.reason = "moving_to_show_pose"
            r, g = self.marker
            if g >= MARKER_PIX and g > 2 * r:
                self.fresh_maps()
                return self.go(SEARCH_T, "marker_green_atomizer_on")
            if r >= MARKER_PIX and r > 2 * g and self.grasp_tries >= RECLOSE_MAX:
                self.fresh_maps()
                return self.go(SEARCH_T, "marker_red_reclose_budget_spent_going_on")
            if r >= MARKER_PIX and r > 2 * g:
                self.sub, self.age, self.reason = 1, 0, "marker_red_putting_down"
                return
            if self.hold >= SHOW_SWEEP:
                self.fresh_maps()
                return self.go(SEARCH_T, "marker_unseen_going_on")
            return
        rd = R.from_euler("z", self.grasp_yaw).as_matrix()
        if self.sub == 1:
            self.cmd[22] = -1.0
            self.err = self.servo(np.r_[self.pick, self.atom[1] + MARKER_DZ + PUT_UP], rd)[0]
            self.reason = "putting_down"
            if self.err < REACH_TOL:
                self.sub, self.age = 2, 0
            return
        self.servo(np.r_[self.pick, self.atom[1] + MARKER_DZ + PUT_UP], rd)
        if self.sub == 2:  # open in place: the finger contact ends, the next close is a fresh press
            self.cmd[22], self.reason = 1.0, "opening_in_place"
            if self.age >= OPEN_STEPS:
                self.sub, self.age, self.grasp_tries = 3, 0, self.grasp_tries + 1
            return
        ramp = RECLOSE_RAMPS[self.grasp_tries % len(RECLOSE_RAMPS)]
        self.cmd[22], self.reason = max(-1.0, 1.0 - 2.0 * self.age / ramp), "reclosing_in_place"
        if self.age < CLOSE_STEPS + ramp:
            return
        self.grip_yaw = self.grasp_yaw
        if self.held():
            return self.go(LIFT_A, "reclosed_held")
        self.cmd[22], self.atom[0] = 1.0, self.to_odom(self.pick[None])[0]
        self.go(REACH_A, "reclose_missed_grasping_again")

    # ---- tree phases -----------------------------------------------------------------------------------------
    def tree_found(self):
        t = self.find_tree()
        if t is None:
            return False
        self.tree = t
        return self.plan_tree()

    def plan_tree(self):
        if self.plan(self.approach_goals(self.tree, SPRAY_SPOT, 1.2, 7)) or self.plan(self.approach_goals(self.tree, SPRAY_SPOT, math.pi, 24)):
            self.go(GO_T, "tree_seen", True)
            return True
        self.reason = "tree_unreachable"
        return False

    def refine_tree(self):
        """Trunk points of the head view near the expected trunk (arm and can stay closer than 0.45 m)."""
        pts, _ = self.cloud(2, None, VIEW_STRIDE)
        tb = self.to_base(self.tree)
        near = pts[(np.linalg.norm(pts[:, :2] - tb, axis=1) < 0.5) & (np.hypot(pts[:, 0], pts[:, 1]) > 0.45)
                   & (pts[:, 2] > TRUNK_Z[0]) & (pts[:, 2] < TRUNK_Z[1])]
        if len(near) > 5:
            self.tree = np.median(self.to_odom(near), 0)

    def trunk_in_view(self):
        """Re-anchor the target on the bare trunk band (TRUNK_MID..TRUNK_Z[1]) of the current head frame within TREE_MERGE
        of the estimate: the densest 0.25 m cell wins (odometry heading drifts, the frame does not)."""
        pts, _ = self.cloud(2, None, VIEW_STRIDE)
        tb = self.to_base(self.tree)
        rng = np.hypot(pts[:, 0], pts[:, 1])
        sel = pts[(np.linalg.norm(pts[:, :2] - tb, axis=1) < TREE_MERGE) & (rng > 0.45) & (rng < 9.0)
                  & (pts[:, 2] > TRUNK_MID) & (pts[:, 2] < TRUNK_Z[1])][:, :2]
        if len(sel) < 6:
            return
        cells = np.floor(sel / TCELL).astype(int)
        uniq, cnt = np.unique(cells, axis=0, return_counts=True)
        best = (uniq[np.argmax(cnt)] + 0.5) * TCELL
        near = sel[np.linalg.norm(sel - best, axis=1) < 0.4]
        self.tree = self.to_odom(np.r_[np.median(near, 0), 0.0][None])[0]

    def fresh_maps(self):
        """Clear maps and blacklist the bearing of the just-sprayed tree for the next search."""
        self.obst[:], self.seen[:], self.marks[:] = 0.0, False, False
        self.canopy[:], self.low[:], self.mid[:] = 0.0, 0.0, 0.0
        self.depth_ref, self.pos_ref = None, self.pos.copy()
        self.drive_n, self.blocked = 0, False
        if self.tree is None:
            self.search_dir = -1
        else:
            away = self.tree - self.pos
            if np.linalg.norm(away) > 1e-6:
                self.search_dir = int(round(wrap(math.atan2(away[1], away[0]) - self.yaw) / (math.pi / 4.0))) % 8
            else:
                self.search_dir = -1

    def ph_spray(self):
        """sub 0: fine alignment of the base on the trunk seen fresh. sub 1: nozzle sideways at SPRAY_H, SPRAY_R before the
        trunk axis, yaw aimed and swept, SPRAY_HOLD steps in tolerance."""
        self.cmd[22] = -1.0
        if not self.held():
            self.cmd[22] = 1.0
            self.go(SEARCH_A, "dropped_while_spraying")
            return np.zeros(3)
        if self.sub == 0 and (self.age <= 2 or self.age % TRACK_EVERY == 0):
            red_count, green_count = self.marker_seen()
            if red_count >= MARKER_PIX and red_count > 2 * green_count:
                self.rebase_press()
                return np.zeros(3)
        base = np.zeros(3)
        if self.sub == 0:
            self.servo(CARRY, R.from_euler("z", self.grip_yaw).as_matrix())
            if self.step_no % 3 == 0:
                self.refine_tree()
            b = self.to_base(self.tree)
            e = b - np.asarray(SPRAY_SPOT)
            base[:2] = np.clip(e * 1.2, -0.12, 0.12)
            base[2] = float(np.clip(wrap(math.atan2(b[1], b[0]) - math.atan2(SPRAY_SPOT[1], SPRAY_SPOT[0])), -0.3, 0.3))
            self.err = float(np.linalg.norm(e))
            self.reason = "aligning_on_trunk"
            if self.err < SPOT_TOL:
                self.sub, self.age = 1, 0
            return base
        p = self.to_base(self.tree)
        u = p / max(float(np.linalg.norm(p)), 1e-9)
        aim = math.atan2(u[1], u[0]) + (math.pi if self.nozzle > 0 else 0.0)  # nozzle -1: applier axis along gripper +x (v12h-s301 truth)
        span = SPRAY_SWEEP if self.nozzle else SPRAY_YAW
        yaw = wrap(aim + span * math.sin(self.hold * 2.0 * math.pi / SPRAY_PERIOD))
        err, rerr = self.servo(np.r_[p - u * SPRAY_R, SPRAY_H], R.from_euler("z", yaw).as_matrix())
        self.err = err
        if err > SPRAY_TOL or rerr > 0.3:
            self.reason = "bringing_nozzle_to_trunk"
            if self.hold == 0 and self.age > 600:
                self.spray_fails += 1
                if self.spray_fails % SPRAY_FAILS:
                    self.go(SPRAY, "aim_not_reached_realign")
                else:
                    self.trees_done.append(self.tree.copy())
                    self.go(SEARCH_T, "aim_not_reached_give_up_tree")
            return base
        self.hold += 1
        self.reason = "spraying"
        if self.hold >= SPRAY_HOLD:
            self.trees_done.append(self.tree.copy())
            if len(self.trees_done) >= MAX_TREES:
                self.go(DONE, "sweep_complete")
            else:
                self.fresh_maps()
                self.tree = None
                self.go(SEARCH_T, "sweep_complete_new_search")
        return base

    # ---- main loop -------------------------------------------------------------------------------------------
    def act(self, obs):
        p = np.asarray(obs["robot_r1::proprio"], dtype=np.float64)[0]
        self.p = p
        self.q = np.r_[p[53:57], p[3:10], p[28:35], p[24:26], p[49:51]]
        self.rel = np.asarray(obs["robot_r1::cam_rel_poses"], dtype=np.float64)[0].reshape(3, 7)
        self.views = (
            (np.asarray(obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"])[0][..., :3],
             np.asarray(obs["robot_r1::robot_r1:left_realsense_link:Camera:0::depth_linear"], dtype=np.float32)[0]),
            (np.asarray(obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"])[0][..., :3],
             np.asarray(obs["robot_r1::robot_r1:right_realsense_link:Camera:0::depth_linear"], dtype=np.float32)[0]),
            (np.asarray(obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"])[0][..., :3],
             np.asarray(obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"], dtype=np.float32)[0]))
        if self.cmd is None:
            self.cmd = np.r_[np.zeros(3), p[53:57], p[3:10], 1.0, p[28:35], 1.0]
        self.pos = self.pos + rot2(self.yaw) @ p[:2] * DT
        self.yaw = wrap(self.yaw + YAW_SCALE * p[2] * DT)
        self.step_no, self.age = self.step_no + 1, self.age + 1
        base = np.zeros(3)
        ph = self.phase
        if ph in (SEARCH_A, GO_A, SEARCH_T, GO_T) and self.step_no % 3 == 0:
            self.update_map()
        if ph in (SEARCH_T, GO_T) and not self.carry():
            ph = self.phase
        if ph == SEARCH_A:
            base = self.search(self.atom_found)
        elif ph == GO_A:
            base = self.ph_go(ALIGN_A)
        elif ph == ALIGN_A:
            base = self.ph_align()
        elif ph == REACH_A:
            self.ph_reach()
        elif ph == CLOSE_A:
            self.ph_close()
        elif ph == LIFT_A:
            self.ph_lift()
        elif ph == PRESS_A:
            self.ph_press()
        elif ph == SEARCH_T:
            base = self.search(self.tree_found)
        elif ph == GO_T:
            base = self.ph_go(SPRAY)
        elif ph == SPRAY:
            base = self.ph_spray()
        elif ph == DONE:
            self.cmd[22] = -1.0
            self.reason = "all_trees_sprayed_holding"
        self.watch_motion(base)
        self.cmd[:3] = np.clip(base / np.array(BASE_LIMITS), -1.0, 1.0)
        return self.cmd.astype(np.float32)

    def log(self):
        return {"phase": PHASES[self.phase], "stage": int(self.phase), "reason": self.reason, "step": self.step_no,
                "atom_hits": self.atom_hits, "grasp_tries": self.grasp_tries, "nozzle": self.nozzle,
                "marker_red_green": list(self.marker), "grip_yaw": round(self.grip_yaw, 2), "trees_done": len(self.trees_done), "tree_seen": self.tree is not None,
                "spray_hold": self.hold, "spray_fails": self.spray_fails, "err": round(self.err, 3),
                "odom": [round(float(self.pos[0]), 2), round(float(self.pos[1]), 2), round(self.yaw, 2)]}


def entrypoint():
    return Planner()
