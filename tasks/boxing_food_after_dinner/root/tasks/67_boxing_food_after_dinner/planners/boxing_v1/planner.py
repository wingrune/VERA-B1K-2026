"""boxing_v1, boxing_food_after_dinner (task 67) parent, OBS_CONTRACT inputs only (head RGB-D, camera poses, proprio). Carries each plate from the counter into the sink:
search, approach, grasp the rim with the right hand, carry over the sink, release. Phases = PHASES (log() gives phase, stage, reason); tunables are named constants."""
import math

import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.spatial.transform import Rotation as R

TIP_LINK = 'right_gripper_finger_link2'; TIP_POINT = (-8.868e-05, 0.0157473, -0.0379637)
JOINTS = (
    (0, 'base_link', 'torso_link1', 'revolute', (-0.079032, 0.0, 0.34265), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (1, 'torso_link1', 'torso_link2', 'revolute', (0.0, 0.0, 0.4), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (2, 'torso_link2', 'torso_link3', 'revolute', (0.0, 0.0001005, 0.3), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)), (3, 'torso_link3', 'torso_link4', 'revolute', (0.0, -0.00010015, 0.09962), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    (-1, 'torso_link4', 'zed_link', 'fixed', (0.065511, 0.060003, 0.47598), (-1.9199, 0.0, -1.5708), (0.0, 0.0, 0.0)), (-1, 'torso_link4', 'right_arm_base_link', 'fixed', (-0.00048706, -0.097236, 0.30302), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (4, 'right_arm_base_link', 'right_arm_link1', 'revolute', (0.0, -0.0735, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (5, 'right_arm_link1', 'right_arm_link2', 'revolute', (0.023988, -0.081265, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    (6, 'right_arm_link2', 'right_arm_link3', 'revolute', (-0.023988, 0.0, -0.1155), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (7, 'right_arm_link3', 'right_arm_link4', 'revolute', (0.0, 0.035058, -0.1945), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (8, 'right_arm_link4', 'right_arm_link5', 'revolute', (0.0, -0.035058, -0.095), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (9, 'right_arm_link5', 'right_arm_link6', 'revolute', (0.0, 0.028001, -0.16305), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (10, 'right_arm_link6', 'right_arm_link7', 'revolute', (0.0295, -0.028001, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (-1, 'right_arm_link7', 'right_gripper_link', 'fixed', (-0.0295, 0.0, -0.16065), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    (12, 'right_gripper_link', 'right_gripper_finger_link2', 'prismatic', (8.9046e-05, -0.013453, -0.03689), (0.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
)
LIMITS = ((-1.1345, -2.7925, -1.8326, -3.0543, -4.4506, -3.1416, -2.3562, -2.0944, -2.3562, -1.0472, -1.5708), (1.8326, 2.5307, 1.5708, 3.0543, 1.309, 0.1745, 2.3562, 0.3491, 2.3562, 1.0472, 1.5708))


def rot2(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def first(x):
    """Drop the leading batch axis of size 1."""
    a = np.asarray(x)
    return a[0] if a.ndim >= 2 and a.shape[0] == 1 else a



# ---- kinematics (JOINTS table instead of the URDF) --------------------------------------------------------
class Kinematics:
    """FK and Jacobians, base->zed_link and base->right finger tip. q: torso 4, right arm 7, 2 fingers."""
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


def from_proprio(s):
    return np.r_[s[53:57], s[28:35], s[49:51]].astype(float)


def solve_lsq(fun, x, lo, hi, max_nfev=140):
    """Bounded Levenberg-Marquardt, forward-difference Jacobian."""
    r = fun(x); cost, lam = r @ r, 1e-2
    for _ in range(max_nfev):
        J = np.empty((len(r), len(x)))
        for k in range(len(x)):
            h = 1e-6 * max(1., abs(x[k])) * (1 if x[k] + 1e-6 * max(1., abs(x[k])) <= hi[k] else -1); xk = x.copy(); xk[k] += h; J[:, k] = (fun(xk) - r) / h
        A, g = J.T @ J, J.T @ r
        while lam < 1e12:
            xn = np.clip(x - np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g), lo, hi); rn = fun(xn)
            if rn @ rn < cost: break
            lam *= 4
        else:
            return x
        done = cost - rn @ rn < 1e-12 * cost or np.linalg.norm(xn - x) < 1e-9; x, r, cost, lam = xn, rn, rn @ rn, max(lam / 3, 1e-9)
        if done: break
    return x





# ---- tunables (colours/heights calibrated on rollout snapshots: privileged poses were projected into the head image, dev only) ----
HEAD = "robot_r1::robot_r1:zed_link:Camera:0::"
PLATE_V_MIN, PLATE_S_MAX = 200, 0.06    # white ceramic: V (0-255) at least, saturation (0-1) at most
PLATE_Z = (0.86, 1.00)                  # plate rim / taco height band in the robot frame (counter top 0.89, plate top 0.92, taco 0.97)
PLATE_EXT = (0.25, 0.55)                # plate cluster length (m): plates are 0.38-0.42 across
PLATE_MIN_WIDTH, PLATE_MIN_N = 0.20, 25
PLATE_FLAT = 0.04                       # a plate top is flat: z spread (10-90 percent) of the cluster at most this (rejects the white fridge door)
PLATE_CELL = 0.03                       # clustering grid (m): smaller than the gap between a plate and its neighbours
SINK_Z = (0.66, 0.80)                   # basin floor band: below the counter top (0.89), above the stool seats (0.63); seen from the stand-off
SINK_RANGE = 2.5                        # the basin floor is only visible from close by
SINK_MIN_WIDTH, SINK_MIN_N, SINK_EXT = 0.20, 150, (0.20, 0.70)
PLATE_R = 0.19                          # plate radius (m): the hand is put this far plus PUSH_GAP from the plate centre
REACH = (0.72, -0.18)                   # plate / sink centre in the robot frame for the right-arm reach
NEAR = 1.6                              # inside this range the heading puts the target at the REACH bearing
ARRIVE_POS, ARRIVE_YAW = 0.05, 0.06
FAR_YAW = 0.15                          # far from the target the base drives only when the heading error is below this (rad)
TRACK_R = 0.45                          # a new cluster updates the estimate only within this distance of it
LOST_AGE, LOST_DIST = 450, 1.4          # steps without a sighting, and distance, after which the target counts as lost
SEARCH_TURN, SEARCH_LEG, SEARCH_FREE = 650, 150, 1.0   # turn steps, forward-leg steps, free depth (m) needed to drive ahead
WARMUP, CONFIRM = 15, 2                 # first steps ignored (renderer warm-up), consecutive sightings needed to start the approach
SETTLE = 30                             # steps standing still to refine the estimate before reaching
PUSH_Z, PUSH_PRE, PUSH_GAP = 0.902, 0.12, 0.06   # fingertip height beside the plate rim, clearance above it, gap to the rim before the push
SLIDE_SPEED, SLIDE_TOL, LIFT_Z = 0.10, 0.04, 0.20  # base strafe speed while pushing (m/s), stop distance to the sink centre (m), arm raise after the push
OVER_STEPS = 120                        # extra full-speed strafe after the sink centre estimate is reached (0.40 m at 30 Hz): the basin floor
                                        # cluster is partly hidden by the plate and the hand, so its centre is biased to the near side (H100 dev, seed 301)
HAND_YAW = 1.5707963267948966           # hand orientation about the vertical (the pose the IK reaches reliably above the counter)
RAISE_TIP = (0.40, -0.25, 1.25)         # waypoint above the counter front, in front of the body: the arm swings up here before it reaches over the counter
HAND_X_BIAS = 0.05                      # the head sees only the far part of a plate this close, so its centre estimate is this much too near (m)
HAND_Y = -0.16                          # lateral position of the hand in the robot frame (the right arm's own line)
CLOSE_FINGERS = -1.0                    # gripper command: the closed hand is the pusher
TORSO_HOME = (0.70, -0.95, -0.35, 0.0)
ARM_REST = (-1.6, 0.0, -0.7, 0.1, -0.5, 0.3, -0.6)
IK_ARM_STARTS = ((-1.0, 0.0, 0.0, -1.5, 0.0, 0.0, 0.0), (-0.5, -0.5, 0.0, -1.5, 0.0, 0.0, 0.0), (-1.2, -0.3, 0.5, -1.2, 0.0, -0.5, 0.0))
STILL_STEPS = 25                        # steps without tip motion that count as contact
PLACED_R = 0.35                         # plates released at the sink are skipped within this radius of the sink
PHASES = ("search", "approach", "push")


def hsv_v_s(rgb):
    c = rgb.astype(float); mx, mn = c.max(-1), c.min(-1)
    return mx, np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)


def head_cam(frame):
    rel = frame["robot_r1::cam_rel_poses"].reshape(3, 7); rgb, d = frame[HEAD + "rgb"][..., :3], np.squeeze(frame[HEAD + "depth_linear"]); w = d.shape[0]
    return {"rgb": rgb, "depth": d, "K": np.array([[.425 * w, 0, w / 2], [0, .425 * w, w / 2], [0, 0, 1.]]), "R": R.from_quat(rel[2, 3:]).as_matrix() @ np.diag([1, -1, -1]), "t": rel[2, :3]}


def pointcloud(cam, mask, stride=1):
    """Robot-frame points of the masked, valid head pixels."""
    d = cam["depth"]; valid = np.isfinite(d) & (d > .06) & (d < 6) & mask
    y, x = np.where(valid); y, x = y[::stride], x[::stride]; z = d[y, x]
    p = np.c_[(x + .5 - cam["K"][0, 2]) / cam["K"][0, 0] * z, (y + .5 - cam["K"][1, 2]) / cam["K"][1, 1] * z, z]
    return p @ cam["R"].T + cam["t"]


def blobs(p, cell, min_n, ext, min_width, flat=9.):
    """Clusters of p on a grid: dicts c (xy centre), z (median), zmax, len, width, n; length within ext, minor extent >= min_width."""
    if len(p) < min_n: return []
    ij = np.floor(p[:, :2] / cell).astype(int); ij -= ij.min(0); grid = np.zeros(ij.max(0) + 1, bool); grid[ij[:, 0], ij[:, 1]] = True
    lab, n = label(maximum_filter(grid, size=3), structure=np.ones((3, 3))); ids = lab[ij[:, 0], ij[:, 1]]; out = []
    for k in range(1, n + 1):
        q = p[ids == k]
        if len(q) < min_n: continue
        c = q[:, :2].mean(0); vec = np.linalg.eigh(np.cov((q[:, :2] - c).T) + 1e-9 * np.eye(2))[1]
        length, width = np.ptp((q[:, :2] - c) @ vec[:, 1]), np.ptp((q[:, :2] - c) @ vec[:, 0])
        if ext[0] <= length <= ext[1] and width >= min_width and np.ptp(np.quantile(q[:, 2], (.1, .9))) <= flat:
            out.append({"c": c, "z": float(np.median(q[:, 2])), "zmax": float(np.quantile(q[:, 2], .95)), "len": float(length), "width": float(width), "n": len(q)})
    return out


def find_plates(cam):
    v, s = hsv_v_s(cam["rgb"]); p = pointcloud(cam, (v >= PLATE_V_MIN) & (s <= PLATE_S_MAX))
    p = p[(p[:, 2] > PLATE_Z[0]) & (p[:, 2] < PLATE_Z[1]) & (np.linalg.norm(p[:, :2], axis=1) < 6)]
    return blobs(p, PLATE_CELL, PLATE_MIN_N, PLATE_EXT, PLATE_MIN_WIDTH, PLATE_FLAT)


def find_sinks(cam):
    """Basin floor: depth-only clusters of points below the counter top, close to the robot."""
    p = pointcloud(cam, np.ones(cam["depth"].shape, bool), 2)
    p = p[(p[:, 2] > SINK_Z[0]) & (p[:, 2] < SINK_Z[1]) & (np.linalg.norm(p[:, :2], axis=1) < SINK_RANGE)]
    return blobs(p, .04, SINK_MIN_N, SINK_EXT, SINK_MIN_WIDTH)


class Planner:
    """Carry each plate from the counter into the sink: search, approach, grasp the rim, carry over the sink, release."""
    def reset(self, task_spec):
        self.kin = Kinematics(); self.lim = np.asarray(LIMITS); self.phase, self.sub, self.reason = "search", "", "start"; self.step = self.age = self.sub_age = 0
        self.odom = np.zeros(3); self.arm_action = self.plan = self.goal = self.last_tip = None; self.desired = np.eye(3)
        self.plate = self.sink = self.start_y = None; self.dir = 1.; self.obj_h = None; self.sights = 0
        self.events, self.debug = [], {}; self.placed = self.still = self.search_legs = self.plates_seen = self.sink_seen = 0
    def log(self):
        return {"phase": self.phase, "sub": self.sub, "stage": PHASES.index(self.phase), "reason": self.reason, "step": self.step, "placed": self.placed,
                "search_legs": self.search_legs, "plates_seen": self.plates_seen, "sink_seen": self.sink_seen,
                "odom": self.odom.round(3).tolist(), "plate": None if self.plate is None else np.round(self.plate, 3).tolist(),
                "sink": None if self.sink is None else np.round(self.sink, 3).tolist(), "debug": self.debug, "events": self.events[-30:]}
    def advance(self, phase, why, sub=""):
        self.events = self.events[-29:] + ["%d:%s.%s:%s" % (self.step, phase, sub, why)]; self.phase, self.sub, self.reason, self.plan = phase, sub, why, None; self.age = self.sub_age = 0
        if phase == "search": self.plate = None
        if phase == "approach": self.obj_h = None
    def to_sub(self, sub, why=""):
        self.events = self.events[-29:] + ["%d:%s.%s" % (self.step, self.phase, sub)]; self.sub, self.reason, self.sub_age, self.still = sub, why or sub, 0, 0
    def to_odom(self, p):
        return rot2(self.odom[2]) @ np.asarray(p[:2]) + self.odom[:2]
    def to_robot(self, p):
        return rot2(-self.odom[2]) @ (np.asarray(p[:2]) - self.odom[:2])
    def tip_of(self, q, cache=None):
        if cache is None: q = q.copy(); q[11:13] = 0.
        pose = (cache or self.kin.forward(q))[0][TIP_LINK]
        return pose, pose[:3, 3] + pose[:3, :3] @ np.asarray(TIP_POINT)
    # ---- arm control (same pattern as the validated 44 / radio planners)
    def solve_ik(self, q, goal, desired, torso=None):
        q = q.copy(); q[11:13] = 0.; lo, hi = self.lim.copy()
        if torso is not None: lo[:4] = hi[:4] = np.asarray(torso)
        rest = np.r_[TORSO_HOME if torso is None else torso, ARM_REST]; best = None
        def residual(x):
            trial = q.copy(); trial[:11] = x; pose, tip = self.tip_of(trial)
            return np.r_[(tip - goal) * 5, R.from_matrix(desired @ pose[:3, :3].T).as_rotvec(), (x - rest) * .008]
        for start in [q[:11], rest, np.r_[1.2, -1.6, -.3, 0., ARM_REST]] + [np.r_[rest[:4], arm] for arm in IK_ARM_STARTS]:
            x = solve_lsq(residual, np.clip(start, lo + 1e-6, hi - 1e-6) if torso is None else np.clip(start, lo, hi), lo, hi); r = residual(x)
            if np.linalg.norm(r[:3]) / 5 < .02 and np.linalg.norm(r[3:6]) < .15 and (best is None or np.linalg.norm(x - q[:11]) < best[0]): best = (np.linalg.norm(x - q[:11]), x)
        return None if best is None else best[1]
    def joint_move(self, action, q, target, rate=.02):
        err = target - q[:11]; step = err * min(1., rate / max(float(np.abs(err).max()), 1e-9)); action[3:7], action[15:22] = q[:4] + step[:4], q[4:11] + step[4:]
        self.debug["jerr"] = np.round(target - q[:11], 3).tolist()
        return float(np.abs(target - q[:11]).max())
    def servo(self, action, q, goal, desired, speed=.0035):
        q = q.copy(); q[11:13] = 0.; cache = self.kin.forward(q); pose, tip = self.tip_of(q, cache); error = goal - tip; lo, hi = self.lim
        J = self.kin.jacobian(TIP_LINK, q, np.asarray(TIP_POINT), cache)[:, :11]; angular = R.from_matrix(desired @ pose[:3, :3].T).as_rotvec()
        delta = np.r_[error * min(.65, speed / max(np.linalg.norm(error), 1e-8)), np.clip(angular * .4, -.025, .025)]
        for _ in range(3):
            dq = J.T @ np.linalg.solve(J @ J.T + .001 * np.eye(6), delta); J[:, ((q[:11] < lo + .002) & (dq < 0)) | ((q[:11] > hi - .002) & (dq > 0))] = 0
        target = np.clip(q[:11] + np.clip(dq, -.035, .035), lo, hi); action[3:7], action[15:22] = target[:4], target[4:]
        self.still = self.still + 1 if self.last_tip is not None and np.linalg.norm(tip - self.last_tip) < .0006 else 0; self.last_tip = tip
        self.debug.update(err=round(float(np.linalg.norm(error)), 3), tip=tip.round(3).tolist())
        return float(np.linalg.norm(error)), tip
    # ---- perception: estimates live in the odometry frame (x, y, z)
    def track_plate(self, cam):
        cl = [c for c in find_plates(cam) if self.sink is None or np.linalg.norm(self.to_odom(c["c"]) - self.sink[:2]) > PLACED_R]
        self.plates_seen += bool(cl)
        if self.plate is not None: cl = [c for c in cl if np.linalg.norm(self.to_odom(c["c"]) - self.plate[:2]) < TRACK_R]
        if not cl: return False
        c = min(cl, key=lambda c: np.linalg.norm(c["c"])); self.plate = np.r_[self.to_odom(c["c"]), c["z"]]
        return True
    def track_sink(self, cam):
        cl = find_sinks(cam); self.sink_seen += bool(cl)
        if self.sink is not None: cl = [c for c in cl if np.linalg.norm(self.to_odom(c["c"]) - self.sink[:2]) < TRACK_R * 2]
        if not cl: return False
        c = max(cl, key=lambda c: c["n"]); self.sink = np.r_[self.to_odom(c["c"]), c["z"]]
        return True
    # ---- phases
    def act(self, obs):
        frame = {"robot_r1::proprio": first(obs["robot_r1::proprio"]), "robot_r1::cam_rel_poses": first(obs["robot_r1::cam_rel_poses"]),
                 HEAD + "rgb": first(obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"]), HEAD + "depth_linear": first(obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"])}
        s = frame["robot_r1::proprio"]; q = from_proprio(s); self.step += 1; self.age += 1; self.sub_age += 1
        if self.step > 1:
            self.odom[:2] += rot2(self.odom[2]) @ s[:2] / 30; self.odom[2] += s[2] / 30
        if self.arm_action is None: self.arm_action = np.r_[np.zeros(3), s[53:57], s[3:10], 1., s[28:35], 1.]
        action = self.arm_action.copy(); action[:3] = 0; self.debug = {"fingers": round(float(s[49] + s[50]), 4), "q": q[:11].round(2).tolist()}; look = self.step % 5 == 1
        cam = head_cam(frame) if look else None
        if cam is not None and self.phase in ("search", "approach") or (cam is not None and self.phase == "push" and self.sub == "settle"): self.track_sink(cam)
        if self.phase == "search": self.search(action, cam, frame)
        elif self.phase == "approach": self.approach(action, cam)
        else: self.push(action, cam, q)
        self.arm_action = action.copy()
        assert np.isfinite(action).all()
        return action.astype(np.float32)
    def search(self, action, cam, frame):
        """Turn in place, then a short forward leg on free depth; the first plate on the counter starts the approach."""
        if cam is not None and self.step > WARMUP:
            self.sights = self.sights + 1 if self.track_plate(cam) else 0
            if not self.sights: self.plate = None
            if self.sights >= CONFIRM: return self.advance("approach", "plate seen at %s" % np.round(self.plate, 2).tolist())
        if self.age < SEARCH_TURN: action[2] = .3; self.reason = "turning, no plate in view"
        elif self.age < SEARCH_TURN + SEARCH_LEG:
            ahead = head_cam(frame)["depth"][300:460, 240:480]; ok = ahead[np.isfinite(ahead)]
            if len(ok) and np.quantile(ok, .05) > SEARCH_FREE: action[0] = .25; self.reason = "forward leg"
            else: action[2] = .3; self.reason = "forward leg blocked, turning"
        else: self.search_legs += 1; self.age = 0
    def approach(self, action, cam):
        """Holonomic drive until the plate centre lies at REACH in the robot frame, tracking it with the head."""
        if cam is not None and not self.track_plate(cam) and self.age > LOST_AGE and np.linalg.norm(self.to_robot(self.plate)) > LOST_DIST:
            return self.advance("search", "plate lost from view")
        d = self.plate[:2] - self.odom[:2]
        if self.obj_h is None or np.linalg.norm(d) > NEAR: self.obj_h = math.atan2(d[1], d[0]) - math.atan2(REACH[1], REACH[0]) * (np.linalg.norm(d) < NEAR)
        local = rot2(-self.odom[2]) @ (self.plate[:2] - rot2(self.obj_h) @ np.asarray(REACH) - self.odom[:2]); dh = wrap(self.obj_h - self.odom[2]); dpos = float(np.linalg.norm(local))
        action[:2] = np.clip(local * 1.2, -.25, .25); action[2] = np.clip(dh * 1.5, -.4, .4)
        if np.linalg.norm(d) > NEAR: action[:2] = (.25 if abs(dh) < FAR_YAW else 0.) * np.array([1., 0.])   # far: unicycle along the bearing, no sideways drift
        self.reason = "driving, %.2f m and %.2f rad to the stand-off" % (dpos, abs(dh)); self.debug["rel"] = np.round(self.to_robot(self.plate), 3).tolist()
        if dpos < ARRIVE_POS and abs(dh) < ARRIVE_YAW:
            action[:3] = 0; self.advance("push", "arrived, plate at %s" % self.debug["rel"], "settle")
    def push(self, action, cam, q):
        """Closed hand beside the plate on the side away from the sink, then strafe the base so the hand slides the plate over the sink."""
        action[22] = CLOSE_FINGERS if self.sub in ("slide", "over", "lift") else 1.
        if self.sub == "settle":
            if cam is not None: self.track_plate(cam)
            if self.sub_age <= SETTLE: return
            if self.sink is None:
                action[1] = .12 * (1 if (self.age // 200) % 2 == 0 else -1); self.reason = "plate reached, no sink in view, strafing to look"; return
            plate, sink = self.to_robot(self.plate), self.to_robot(self.sink); self.dir = 1. if sink[1] > plate[1] else -1.
            shift = float(plate[1] - (HAND_Y + self.dir * (PLATE_R + PUSH_GAP)))   # strafe so the plate sits beside the hand, which stays on the arm's own line
            if abs(shift) > .02: action[1] = float(np.clip(shift, -.15, .15)); self.reason = "strafing %.2f m to put the plate beside the hand line" % shift; return
            self.start_y = float(plate[1]); self.goal = np.r_[plate[0] + HAND_X_BIAS, HAND_Y, PUSH_Z]; self.desired = R.from_euler("z", HAND_YAW).as_matrix()
            self.plan = self.solve_ik(q, np.asarray(RAISE_TIP), self.desired); self.to_sub("raise", "lifting the hand clear of the counter front")
            if self.plan is None: self.advance("search", "raise pose out of reach")
        elif self.sub == "raise":
            if self.joint_move(action, q, self.plan) < .01:
                self.plan = self.solve_ik(q, self.goal + np.array([0, 0, PUSH_PRE]), self.desired); self.to_sub("pre", "moving the hand beside the plate")
                if self.plan is None: self.advance("search", "side of the plate out of reach at %s" % self.goal.round(2).tolist())
        elif self.sub == "pre":
            if self.joint_move(action, q, self.plan) < .01: self.to_sub("down", "lowering the hand beside the rim")
        elif self.sub == "down":
            err, _ = self.servo(action, q, self.goal, self.desired, .0025)
            if err < .008 or self.still > STILL_STEPS: self.to_sub("slide", "pushing the plate towards the sink")
        elif self.sub == "slide":
            if cam is not None: self.track_sink(cam)
            left = float(self.to_robot(self.sink)[1] - self.start_y); self.debug["left"] = round(left, 3)
            action[1] = float(np.clip(left, -SLIDE_SPEED, SLIDE_SPEED)); self.reason = "pushing, %.2f m to the sink centre" % abs(left)
            if abs(left) < SLIDE_TOL: self.to_sub("over", "sink centre estimate reached, pushing on into the basin")
        elif self.sub == "over":
            action[1] = self.dir * SLIDE_SPEED; self.reason = "pushing on, %d steps left" % (OVER_STEPS - self.sub_age)
            if self.sub_age >= OVER_STEPS: action[:3] = 0; self.goal = self.tip_of(q)[1] + np.array([0, 0, LIFT_Z]); self.to_sub("lift", "plate over the sink, raising the hand")
        else:
            err, _ = self.servo(action, q, self.goal, self.desired)
            if err < .03 or self.still > STILL_STEPS: self.placed += 1; self.advance("search", "plate %d pushed over the sink" % self.placed)


def entrypoint():
    return Planner()
