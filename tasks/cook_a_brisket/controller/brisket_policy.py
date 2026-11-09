"""Cook a brisket controller: open the oven and switch it on, take the brisket out of the fridge, cook it in the closed oven,
put it on the chopping board. Inputs: onboard RGB-D, 61-D proprio and camera poses relative to the base. Pose is integrated
base velocity corrected by visual odometry (odometry frame = start pose). Objects are kept as frames in the odometry frame;
every manipulation is a script of hand targets in the frame of the object it acts on (checks/dev_*.py measured each one)."""
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from arm_servo import ArmServo
from geometry import cameras, pointcloud, resized
from kitchen_perception import bar_near, bars_near, board as detect_board, face_near, fridge_front, markers, oven_button, slab_on_shelf
from occupancy import GardenMap, rot2, wrap
from visual_odometry import VisualOdometry

cv2.setNumThreads(2)
DT = 1 / 30                                                                                   # s per action
MAX_V, MAX_W = 0.75, 1.0                                                                      # m/s, rad/s at action 1 (eval/r1pro.yaml base limits)
GRIP, FINGERS = 22, slice(49, 51)                                                             # right gripper action (1 open, -1 close) and finger joints (see GRIPS for either arm)
Z = np.array([0.0, 0.0, 1.0])
# oven frame: origin = toggle-button marker, n out of the front, u = n x z (the robot's left facing it), z up (dev_probe / dev_oven / dev_rack)
OVEN_HANDLE = np.array([-0.203, -0.008, -0.083])                                              # handle bar grasp point
OVEN_HINGE = np.array([-0.198, -0.035, -0.499])                                               # door hinge (door link origin); the door opens about -u
OVEN_EDGE = np.array([0.0, -0.02, 0.432])                                                     # door top edge 1.5 cm in, from the hinge, door shut
OVEN_STANCE = np.array([-0.078, 0.725])                                                       # base origin facing -n (dev_oven: handle, button, rack all reachable)
OVEN_CLOSE_BACK = 0.22                                                                        # base backs up for the open door's edge (dev_oven5: 0.15 out of reach)
RACK_U, RACK_EDGE_N, RACK_TOP_Z = -0.198, -0.116, -0.306                                      # rack 1: centre, front edge, plate top (dev_rack)
DOOR_FACE_N = -0.026                                                                          # outer door face when shut
GLASS_N = -0.0335                                                                             # the door's glass (most of its face, y -0.357..-0.36): what face_near measures
BUTTON_BIAS = np.array([-0.0035, 0.009, 0.004])                                               # detected marker - true button, oven frame (oven_frame_test, 89 frames)
# fridge frame: origin = midpoint of the two handle bars on the door face, same axis convention (dev_fridge / dev_shelves)
FRIDGE_HANDLE = np.array([-0.0405, 0.047, 1.05])                                              # right door handle bar grasp point
FRIDGE_HINGE = np.array([-0.580, -0.062])                                                     # right door hinge; opens counter-clockwise about +z
FRIDGE_STANCE = np.array([0.241, 0.619])                                                      # base origin facing -n; slides SHIFT along u while the door goes 55 -> 85 deg
FRIDGE_SHIFT = -0.30
RIGHT_SHELVES = ((-0.543, -0.041), [(0.969, -0.14), (1.314, -0.091), (1.657, -0.091)])        # compartment u range, (shelf top z, front edge n)
LOOK_STANCE = np.array([-0.17, 0.30])
LEFT_SHELVES = ((0.025, 0.414), [(0.709, -0.17), (0.965, -0.196), (1.207, -0.196), (1.449, -0.196), (1.692, -0.196)])
LEFT_HANDLE = np.array([0.041, 0.047, 1.05])                                                  # left door handle bar; the door opens clockwise about +z (axis -z)
LEFT_HINGE = np.array([0.443, -0.062])
LEFT_STANCE, LEFT_SHIFT = np.array([-0.22, 0.569]), 0.30                                      # 2 cm clear of the open right door; slides +u while the door goes 55 -> 85 deg
LOOK_LEFT = np.array([0.08, 0.32])                                                            # left-arm shelf views, 2 cm clear of the open left door
OVEN_STANCES = {"right": OVEN_STANCE, "left": np.array([-0.318, 0.725])}                     # rack centre 0.12 m to the working hand's side
GRIPS = {"right": (22, slice(49, 51)), "left": (14, slice(24, 26))}                           # gripper action, finger joints                                                         # base origin for the shelf views and the brisket grasp (dev_find4 / dev_brisket4)
# brisket handling (dev_brisket4: pitch 55 push, 6/6; dev_oven7: place, push deeper, close, cook, pinch out)
PUSH_PITCH = math.radians(55.0)
PUSH_YAW = 1.3                                                                                # torso yaw limit while pushing (servo default 0.75): the 301 top-shelf start 0.76 m ahead needs it (reach_test3)
DOOR_GAP = 0.005                                                                              # fridge doors' free edges: u = -+0.005 (gap between the two handle bars)
BACK_BIAS = 0.015                                                                             # the shelf view reads the back edge 1.4-1.7 cm short (dev_find4, 8 instances)
PUSH_GAP = 0.035                                                                              # hand starts this far behind the slab's back edge (dev_brisket4; 0.045 put the top-shelf start out of reach, full_v8_301)
PUSH_SLIP = 0.047                                                                             # slab travel = hand travel - 0.035..0.057 (dev_brisket4)
OVERHANG = 0.035                                                                              # slab front edge past the shelf edge before the pinch (4 cm + push scatter neared the 5.5 cm tipping point)
FRONT_AHEAD = 0.010                                                                           # the pinch point at most this far in front of the slab front
SLAB_GRIP = (0.012, 0.030)                                                                    # max finger joint after closing on the 2 cm slab
OVEN_PINCH_PITCH, OVEN_PINCH_BEHIND = math.radians(30.0), 0.018                                # retrieval: fingers pitched down, hand behind the slab front
OVEN_GRIP = (0.012, 0.040)                                                                    # max finger joint, pitched pinch (dev_retrieve)
DOOR_GRIP = (0.008, 0.051)                                                                    # oven handle / door edge: AG held on [0.047, 0] (full_v17_303, x3) and [0.028, 0.05] (full_v13_301)
PINCH_AHEAD = 0.022                                                                           # pinch point in front of the shelf / rack edge (the lower fingertip is 1.9 cm out)
HELD = OVERHANG - PINCH_AHEAD                                                                 # slab front edge this far toward the robot from the grasp point
OVEN_OVERHANG, OVEN_DEEPER, OVEN_PINCH_AHEAD = 0.035, 0.023, 0.020                            # place, push in (centre into the fillable volume), retrieve
PUSH_CENTRE = -0.149                                                                          # measured slab centre after the push-in (see push_to)
INSIDE_N, DOOR_INNER_N = -0.150, -0.085                                                       # oven frame: fillable volume from y -0.475 at rack height (dev_retreat), shut door's inner face
CARRY = np.array([0.40, -0.10, 0.95])                                                         # hand while driving, base frame


def allowed(obs):                                                                             # the closed input set: proprio, camera poses, RGB-D
    return {k: v.detach().cpu().numpy() for k, v in obs.items()
            if k == "robot_r1::proprio" or k == "robot_r1::cam_rel_poses" or (k.startswith("robot_r1::robot_r1:") and k.endswith(("::rgb", "::depth_linear")))}


def axis_rot(axis, angle):
    return Rotation.from_rotvec(np.asarray(axis, float) * angle).as_matrix()


class BrisketPolicy:
    def __init__(self, *args, **kwargs):
        self.output = Path(os.environ.get("POLICY_OUTPUT", "/tmp/brisket_policy"))
        self.output.mkdir(parents=True, exist_ok=True)
        self.servos = {arm: ArmServo(Path(__file__).parent, arm=arm) for arm in ("right", "left")}
        self.holders = {arm: ArmServo(Path(__file__).parent, arm=arm, torso=False) for arm in ("right", "left")}   # arm only, outside scripts
        self.episode = -1

    @property
    def servo(self):                                                                          # the working arm
        return self.servos[self.arm]

    @property
    def gi(self):
        return GRIPS[self.arm][0]

    @property
    def fi(self):
        return GRIPS[self.arm][1]

    def side(self, v):                                                                        # base-frame pose written for the right hand, mirrored for the left
        v = np.array(v, float)
        if self.arm == "left":
            v[1] = -v[1]
        return v

    def oven_stance(self):
        return OVEN_STANCES[self.arm]

    def set_action_dim(self, n):
        assert n == 23

    def reset(self):
        self.episode += 1
        self.step, self.age, self.phase = 0, 0, "scan"
        self.pose = np.zeros(3)
        self.vo = VisualOdometry(interval=5)
        self.map = GardenMap()
        self.action = None
        self.vel = np.zeros(3)
        self.still = 0
        self.turned = 0.0
        self.path, self.goal, self.heading, self.planned = None, None, None, -1000
        self.stalled, self.backup, self.blocked_dir = 0, 0, np.zeros(2)
        self.oven_hits, self.fridge_hits, self.board_hits = [], [], []
        self.oven, self.fridge, self.board = None, None, None                                 # frames {"o": [3] odom, "n": [2] odom}; board {"c": [3]}
        self.script, self.si, self.sage, self.mem = None, 0, 0, {}
        self.holding = False
        self.slab = None
        self.held = HELD
        self.arm, self.shelves, self.look_stance = "right", RIGHT_SHELVES, LOOK_STANCE
        self.pressed = False
        self.cook_start = None
        self.after = None
        self.log = {"phases": []}

    def change(self, phase):
        frames = {k: None if F is None else {kk: np.round(vv, 3).tolist() for kk, vv in F.items() if kk in ("o", "n", "c")} for k, F in (("oven", self.oven), ("fridge", self.fridge), ("board", self.board))}
        print("PHASE", self.step, self.phase, "->", phase, json.dumps({"pose": np.round(self.pose, 3).tolist(), **frames}), flush=True)
        self.log["phases"].append([self.step, phase])
        (self.output / f"log_{self.episode}.json").write_text(json.dumps(self.log, default=float))
        self.phase, self.age = phase, 0
        self.path, self.planned, self.turned, self.stalled = None, -1000, 0.0, 0
        self.servo.target = None

    # frames -----------------------------------------------------------------------------------
    def to_base3(self, p):                                                                    # odom [3] -> base [3]
        return np.r_[rot2(self.pose[2]).T @ (np.asarray(p[:2]) - self.pose[:2]), p[2]]

    def to_odom3(self, p):
        return np.r_[rot2(self.pose[2]) @ np.asarray(p[:2]) + self.pose[:2], p[2]]

    def point(self, F, a, b, c=0.0):                                                          # object-frame (u, n, z) -> odom [3]
        n = np.asarray(F["n"])
        u = np.array([n[1], -n[0]])
        xy = F["o"][:2] + a * u + b * n
        return np.r_[xy, F["o"][2] + c]

    def dirs(self, F):                                                                        # u, n as base-frame 3-vectors
        n = rot2(self.pose[2]).T @ np.asarray(F["n"])
        return np.r_[n[1], -n[0], 0.0], np.r_[n, 0.0]

    def fv(self, F, pitch=0.0):                                                               # fingers into the object (-n), closing vertical, pitched down
        u, n = self.dirs(F)
        return axis_rot(u, pitch) @ np.column_stack([-u, Z, n])

    def fs(self, F, pitch=0.0):                                                               # fingers into the object, closing sideways (wrist camera 25 deg down)
        u, n = self.dirs(F)
        return axis_rot(u, pitch) @ np.column_stack([Z, u, n])

    def stance(self, F, local):                                                               # base goal (odom xy) and heading facing -n
        g = self.point(F, local[0], local[1])[:2]
        n = np.asarray(F["n"])
        return g, math.atan2(-n[1], -n[0])

    # perception -------------------------------------------------------------------------------
    def frame_from(self, origin_b, n_b):                                                      # base-frame detection -> odom frame
        return {"o": self.to_odom3(origin_b), "n": rot2(self.pose[2]) @ (n_b[:2] / np.linalg.norm(n_b[:2]))}

    def oven_normal(self, head, button_b, rng=np.random.default_rng(0)):                      # control-panel strip round the button -> outward normal (base):
        p, _ = pointcloud(head, stride=2, max_range=4.0)                                      # RANSAC line, 5 mm inliers: 0.04 deg mean error vs 3.7 for a plain fit
        p = p[(np.linalg.norm(p[:, :2] - button_b[:2], axis=1) < 0.45) & (p[:, 2] > 0.72) & (p[:, 2] < 0.84)]   # that the knobs skew (oven_frame_test, 89 scan frames)
        if len(p) < 80:
            return None
        best = None
        for _ in range(200):
            a, b = p[rng.choice(len(p), 2, replace=False)]
            d = b[:2] - a[:2]
            if np.linalg.norm(d) < 0.15:
                continue
            d = d / np.linalg.norm(d)
            inliers = np.abs((p[:, :2] - a[:2]) @ np.array([-d[1], d[0]])) < 0.005
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or best.sum() < 40:
            return None
        q = p[best][:, :2]
        d = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
        n = np.array([-d[1], d[0]])
        if n @ (-button_b[:2]) < 0:
            n = -n
        return np.r_[n, 0.0]

    def oven_frame(self, b, n):                                                               # button detection -> oven frame; the marker reads 9 mm proud, 4 mm high (oven_frame_test)
        u = np.array([n[1], -n[0], 0.0])
        return self.frame_from(b - BUTTON_BIAS[0] * u - BUTTON_BIAS[1] * n - BUTTON_BIAS[2] * Z, n)

    def local_fridge(self, head, F, door="both"):                                             # face depth (face_near: 0.6 mm median under +-8 cm / 5 deg offsets) and
        u, n = self.dirs(F)                                                                    # the bar pair along the face (bars_near: ~1 cm), test_local_refine
        frame = {"origin": self.to_base3(F["o"]), "n": n, "u": u}
        face = face_near(head, frame) if door == "both" else face_near(head, frame, along=(0.0, 0.45))
        if face is None:
            self.log.setdefault("local", []).append([self.step, "fridge", None])
            return F
        rot = rot2(math.atan(face["slope"]))
        n2 = rot @ np.asarray(F["n"])
        o = self.point(F, 0.0, face["n_offset"], 0.0)
        G = {"o": o, "n": n2 / np.linalg.norm(n2), "votes": F.get("votes", 0)}
        u, n = self.dirs(G)
        if door == "both":
            bars = bars_near(head, {"origin": self.to_base3(G["o"]), "n": n, "u": u}, 0.0, 0.0)
        else:
            bar = bar_near(head, {"origin": self.to_base3(G["o"]), "n": n, "u": u}, 0.0, LEFT_HANDLE[0])
            bars = None if bar is None else bar - LEFT_HANDLE[0]
        if bars is not None and abs(bars) < 0.06:
            G["o"] = self.point(G, bars, 0.0, 0.0)
        self.log.setdefault("local", []).append([self.step, "fridge", round(face["n_offset"], 4), round(math.degrees(math.atan(face["slope"])), 2), None if bars is None else round(bars, 4)])
        print("LOCAL fridge", self.step, round(face["n_offset"], 4), None if bars is None else round(bars, 4), flush=True)
        return G

    def local_oven(self, head, F):                                                           # depth from the shut door's glass (2.4 mm median, test_local_refine)
        u, n = self.dirs(F)
        frame = {"origin": self.to_base3(F["o"]), "n": n, "u": u}
        face = face_near(head, frame, along=(-0.45, 0.05), z_range=(0.30, 0.65), window=0.10)
        if face is None or abs(face["n_offset"] - GLASS_N) > 0.04:
            self.log.setdefault("local", []).append([self.step, "oven", None])
            return F
        G = dict(F)
        G["o"] = self.point(F, 0.0, face["n_offset"] - GLASS_N, 0.0)
        G["o"][2] = F["o"][2]
        self.log.setdefault("local", []).append([self.step, "oven", round(face["n_offset"] - GLASS_N, 4)])
        print("LOCAL oven", self.step, round(face["n_offset"] - GLASS_N, 4), flush=True)
        return G

    def detect(self, head):                                                                   # vote for the oven, the fridge and the board in the odom frame
        b = oven_button(head)
        if b is not None and np.linalg.norm(b["xyz"][:2]) < 7.0:                              # 4.8 m from the 302 start: far sightings get a provisional normal
            n = self.oven_normal(head, b["xyz"]) if np.linalg.norm(b["xyz"][:2]) < 4.5 else None
            good = n is not None
            if n is None:
                n = np.r_[-b["xyz"][:2] / np.linalg.norm(b["xyz"][:2]), 0.0]                   # toward the robot: refined at the stance
            self.oven_hits.append(dict(self.oven_frame(b["xyz"], n), good=good, on=b.get("on", False)))
        if self.step % 15 == 0 and (self.fridge is None or self.fridge.get("votes", 0) < 6):   # RANSAC over 250 lines: the slow detector
            f = fridge_front(head)
            if f is not None:
                self.fridge_hits.append(self.frame_from(f["origin"], f["n"]))
        k = detect_board(head)
        if k is not None:
            self.board_hits.append({"c": self.to_odom3(k["centre"])})
        if not self.mem.get("lock_oven"):
            self.oven = self.vote(self.oven_hits, 0.15) or self.oven
        if not self.mem.get("lock_fridge"):
            self.fridge = self.vote(self.fridge_hits, 0.15) or self.fridge
        self.board = self.vote(self.board_hits, 0.15, key="c") or self.board

    def vote(self, hits, radius, key="o", need=2):                                            # the largest cluster; its latest member's position averaged with its mates
        if not hits:
            return None
        pts = np.array([h[key][:2] for h in hits[-60:]])
        counts = (np.linalg.norm(pts[:, None] - pts[None], axis=2) < radius).sum(1)
        k = int(np.argmax(counts))
        if counts[k] < need:
            return None
        mates = [h for h, d in zip(hits[-60:], np.linalg.norm(pts - pts[k], axis=1)) if d < radius]
        out = dict(mates[-1])
        out[key] = np.mean([h[key] for h in mates[-8:]], 0)
        if "n" in out:
            good = [h for h in mates if h.get("good", True)]
            n = np.mean([h["n"] for h in (good or mates)[-8:]], 0)
            out["n"] = n / np.linalg.norm(n)
        out["votes"] = int(counts[k])
        return out

    # navigation -------------------------------------------------------------------------------
    def drive(self, action, final_heading=None, speed=0.40, lookahead=0.7):                    # follow self.path; True at its end with the heading reached
        if self.path is None or len(self.path) == 0:
            return False
        if self.backup > 0:
            self.backup -= 1
            action[0:2] = -0.3 * self.blocked_dir
            if self.backup == 0:
                self.path, self.planned = None, -1000
            return False
        end = self.path[-1]
        dist = np.linalg.norm(end - self.pose[:2])
        k = int(np.argmin(np.linalg.norm(self.path - self.pose[:2], axis=1)))
        far = [p for p in self.path[k:] if np.linalg.norm(p - self.pose[:2]) > lookahead]
        target = far[0] if far and dist > lookahead else end
        delta = target - self.pose[:2]
        heading = math.atan2(delta[1], delta[0]) if (dist > 0.6 or final_heading is None) else final_heading
        yaw_error = wrap(heading - self.pose[2])
        v = rot2(self.pose[2]).T @ delta
        v = v / max(np.linalg.norm(v), 1e-6) * min(speed, 0.8 * dist + 0.03)
        if abs(yaw_error) > 0.9 and dist > 0.6:
            v *= 0.2
        action[0:2] = np.clip(v / MAX_V, -1, 1)
        turn = 0.35 if self.holding else 0.7
        action[2] = np.clip(1.5 * yaw_error, -turn, turn) / MAX_W
        commanded = np.linalg.norm(action[0:2]) * MAX_V
        stuck = (commanded > 0.15 and np.linalg.norm(self.vel[0:2]) < max(0.03, 0.25 * commanded)) or (abs(action[2]) * MAX_W > 0.3 and abs(self.vel[2]) < 0.05)
        self.stalled = self.stalled + 1 if stuck else 0
        if self.stalled > 45:
            d = action[0:2] / np.linalg.norm(action[0:2]) if np.linalg.norm(action[0:2]) > 0.05 else np.array([1.0, 0.0])
            self.map.block(self.to_odom3(np.r_[0.5 * d, 0.0])[:2])
            self.blocked_dir, self.backup, self.stalled = d, 25, 0
            self.log["blocked"] = self.log.get("blocked", 0) + 1
        return dist < 0.03 and (final_heading is None or abs(wrap(final_heading - self.pose[2])) < 0.03)

    def go(self, action, goal, heading):                                                       # plan (every 90 steps) and follow; True on arrival
        if self.path is None or self.age - self.planned > 90:
            path = self.map.plan(self.pose[:2], goal[None], radius=0.36)
            self.planned = self.age
            via = None
            if path is None and np.linalg.norm(goal - self.pose[:2]) >= 0.8:
                via = self.map.toward(self.pose[:2], goal, radius=0.36)
                path = None if via is None else self.map.plan(self.pose[:2], via[None], radius=0.36)
            if path is not None:
                length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
                straight = float(np.linalg.norm(goal - self.pose[:2]))
                if length > 2 * straight + 1.0 and self.mem.get("detour_dumped") != self.phase:
                    self.mem["detour_dumped"] = self.phase
                    self.log.setdefault("detours", []).append([self.step, self.phase, round(length, 2), round(straight, 2)])
                    print("DETOUR", self.step, self.phase, round(length, 2), round(straight, 2), flush=True)
                    np.savez_compressed(self.output / f"map_{self.episode}_{self.step}.npz", odds=self.map.odds, ground=self.map.ground, path=path,
                                        pose=self.pose, goal=goal, obstacles=self.map.obstacles())
                self.path = np.vstack([path, goal if via is None else via])
                self.mem["noplan"], self.mem["via"] = 0, via is not None
                if via is not None:
                    self.log.setdefault("toward", []).append([self.step, np.round(via, 2).tolist(), np.round(goal, 2).tolist()])
            elif np.linalg.norm(goal - self.pose[:2]) < 0.8:                                   # the goal cell itself may read occupied next to furniture
                self.path = np.vstack([self.pose[:2], goal])
                self.mem["via"] = False
            else:
                self.mem["noplan"] = self.mem.get("noplan", 0) + 1                             # full_v6_305: a red marker 3.9 m off in another room won the oven vote
                self.path = None
                if self.mem["noplan"] % 2 == 0:
                    self.blocked_dir, self.backup = np.array([1.0, 0.0]), 40                   # drive() reverses along blocked_dir
        if self.path is None and self.backup > 0:
            self.backup -= 1
            action[0:2] = -0.3 * self.blocked_dir
            return False
        arrived = self.drive(action, final_heading=None if self.mem.get("via") else heading)
        if arrived and self.mem.get("via"):                                                   # at the waypoint: plan to the goal again
            self.mem["via"], self.path = False, None
            return False
        return arrived

    # scripts ----------------------------------------------------------------------------------
    def start(self, steps, then):
        self.script, self.si, self.sage, self.after = steps, 0, 0, then
        self.servo.target = None
        self.mem = {k: v for k, v in self.mem.items() if not k.startswith("tries_")}
        self.yaw_limit(0.75)

    def yaw_limit(self, v):
        for servo in self.servos.values():
            servo.limits[:, 3] = [-v, v]

    def tick(self, action, q):                                                                # run the current script step for one action; True when the script is done
        while self.si < len(self.script):
            st = self.script[self.si]
            kind = st[0]
            if kind == "do":                                                                  # ("do", fn): run once
                st[1]()
                self.si, self.sage = self.si + 1, 0
                continue
            self.sage += 1
            done = False
            if kind == "move":                                                                # ("move", name, fn -> (base xyz, base R), grip, max steps[, tol, plan])
                _, name, fn, grip, limit = st[:5]
                tol = st[5] if len(st) > 5 else 0.010
                target, rot = fn()
                err = self.servo.step(action, q, target, rot, plan=(len(st) > 6 and st[6] and self.sage == 1))
                action[self.gi] = grip
                done = (err[0] < tol and err[1] < 0.08 and self.sage > 15) or self.sage >= limit
                self.mem["err"], self.mem["target"] = err, target
            elif kind == "verify":                                                            # ("verify", name, retry steps fn, tries[, (lo, hi)]): did the fingers stop on something
                lo, hi = st[4] if len(st) > 4 else (0.008, 0.045)
                held = lo < float(np.max(self.s[self.fi])) < hi                                         # (bar or slab: one finger stays >= 0.02; nothing: both close to 0)
                tries = self.mem.get("tries_" + st[1], 0)
                self.log.setdefault("verify", []).append([self.step, st[1], held, tries, np.round(self.s[self.fi], 3).tolist()])
                print("VERIFY", self.step, st[1], held, tries, np.round(self.s[self.fi], 3).tolist(), flush=True)
                if not held and tries < st[3]:
                    self.mem["tries_" + st[1]] = tries + 1
                    self.script[self.si + 1:self.si + 1] = st[2]() + [st]
                elif not held:
                    self.mem["failed"] = st[1]
                self.si, self.sage = self.si + 1, 0
                continue
            elif kind == "wait":                                                              # ("wait", name, steps, grip)
                action[self.gi] = st[3]
                done = self.sage >= st[2]
            elif kind == "arc":                                                               # ("arc", name, fn(action, q) -> done, limit)
                done = st[2](action, q) or self.sage >= st[3]
            elif kind == "drive":                                                             # ("drive", name, fn -> (odom xy goal, heading), limit)
                goal, heading = st[2]()
                delta = rot2(self.pose[2]).T @ (goal - self.pose[:2])
                action[0:2] = np.clip(np.clip(1.5 * delta, -0.12, 0.12) / MAX_V, -1, 1)
                action[2] = np.clip(1.5 * wrap(heading - self.pose[2]), -0.3, 0.3) / MAX_W
                done = (np.linalg.norm(delta) < 0.01 and abs(wrap(heading - self.pose[2])) < 0.02) or self.sage >= st[3]
                if "grip" in self.mem:
                    action[self.gi] = self.mem["grip"]
            if done:
                row = [self.step, st[1], self.sage, [round(float(e), 3) for e in self.mem.get("err", (0, 0))], round(float(self.mem.get("phi", 0.0)), 3),
                       np.round(self.mem.get("target", np.zeros(3)), 3).tolist() if kind == "move" else None]
                self.log.setdefault("steps", []).append(row)
                print("STEP", json.dumps(row), flush=True)
                self.si, self.sage = self.si + 1, 0
                self.servo.target = None
            return False
        return True

    def door_arc(self, F, hinge_local, axis_local, ref_local, ref_rot_fn, stop, lead, grip=-1.0, base_fn=None):
        """Hand on a door along its hinge arc until the angle measured from the hand reaches `stop` (rad, signed)."""
        def fn(action, q):
            u, n = self.dirs(F)
            axis = axis_local[0] * u + axis_local[1] * n + axis_local[2] * Z
            H = self.to_base3(self.point(F, *hinge_local))
            r0 = self.to_base3(self.point(F, *ref_local)) - H
            hand = self.servo.pose(q)[:3, 3]
            a0, a1 = r0 - axis * (r0 @ axis), (hand - H) - axis * ((hand - H) @ axis)
            phi = math.atan2(float(np.cross(a0, a1) @ axis), float(a0 @ a1))
            sign = 1.0 if stop > self.mem.get("arc_from", 0.0) else -1.0
            goal = phi + sign * lead
            goal = min(goal, stop) if sign > 0 else max(goal, stop)
            self.servo.step(action, q, H + axis_rot(axis, goal) @ r0, axis_rot(axis, goal) @ ref_rot_fn())
            action[self.gi] = grip
            if base_fn is not None:
                bg, bh = base_fn(phi)
                delta = rot2(self.pose[2]).T @ (bg - self.pose[:2])
                action[0:2] = np.clip(np.clip(1.5 * delta, -0.10, 0.10) / MAX_V, -1, 1)
                action[2] = np.clip(1.5 * wrap(bh - self.pose[2]), -0.3, 0.3) / MAX_W
            self.mem["phi"] = phi
            return (sign > 0 and phi > stop - 0.03) or (sign < 0 and phi < stop + 0.03)
        return fn

    # skill scripts (each mirrors a dev check) -------------------------------------------------
    def oven_open_script(self, press):                                                        # dev_oven7 steps 1 and 5
        F = self.oven
        H = OVEN_HINGE
        hw = lambda d=0.0: (self.to_base3(self.point(F, *(OVEN_HANDLE + np.array([0.0, d, 0.0])))), self.fv(F))
        s = [("move", "pre_handle", lambda: hw(0.12), 1.0, 300, 0.010, True),
             ("move", "handle", lambda: hw(), 1.0, 150, 0.006),
             ("move", "grip", lambda: hw(), -1.0, 40, 0.0),
             ("verify", "oven_handle", lambda: [("move", "regrip_off", lambda: hw(0.06), 1.0, 60, 0.01), ("move", "handle", lambda: hw(), 1.0, 150, 0.006),
                                               ("move", "grip", lambda: hw(), -1.0, 40, 0.0)], 2, DOOR_GRIP),
             ("do", lambda: self.mem.update(arc_from=0.0)),
             ("arc", "pull", self.door_arc(F, H, (-1, 0, 0), OVEN_HANDLE, lambda: self.fv(F), math.radians(35), math.radians(6)), 400),
             ("wait", "let_go", 20, 1.0),
             ("move", "back_off", lambda: self.retreat(F), 1.0, 100, 0.015),
             ("move", "lift_away", lambda: (self.side([0.40, -0.10, 0.95]), self.fv(F)), 1.0, 250, 0.02),
             ("wait", "fall", 90, 1.0)]
        if press:
            s += self.press_steps()
        return s

    def press_steps(self):
        F = self.oven
        button = lambda d: (self.to_base3(self.point(F, 0.0, d, 0.0)), self.fv(F))
        return [("move", "button_pre", lambda: button(0.10), -1.0, 250, 0.01, True),
                ("move", "press", lambda: button(0.008 - 0.006 * self.mem.get("presses", 0)), -1.0, 25, 0.006),
                ("move", "unpress", lambda: button(0.12), -1.0, 120, 0.02),
                ("move", "show", lambda: (self.side([0.40, -0.10, 0.95]), self.fv(F)), 1.0, 150, 0.02),
                ("wait", "marker_settle", 4, 1.0),
                ("do", self.check_on)]

    def check_on(self):                                                                       # the toggle marker: red off, green on (toggles once per 5 steps of contact, toggle.py)
        b = oven_button(cameras(self.data)["head"])
        near = b is not None and np.linalg.norm(self.to_odom3(b["xyz"])[:2] - self.oven["o"][:2]) < 0.15
        presses = self.mem.get("presses", 0)
        self.log.setdefault("presses", []).append([self.step, None if b is None else bool(b["on"]), bool(near), presses])
        print("PRESS", self.step, None if b is None else bool(b["on"]), near, presses, flush=True)
        if near and not b["on"] and presses < 2:
            self.mem["presses"] = presses + 1
            self.script[self.si + 1:self.si + 1] = self.press_steps()
        self.pressed = True

    def retreat(self, F):                                                                     # out along the turned finger axis from the let-go handle pose
        phi = self.mem.get("phi", 0.0)
        u, n = self.dirs(F)
        H = self.to_base3(self.point(F, *OVEN_HINGE))
        r0 = self.to_base3(self.point(F, *OVEN_HANDLE)) + 0.12 * n - H
        R = axis_rot(-u, phi)
        return H + R @ r0, R @ self.fv(F)

    def fridge_open_script(self, door="right"):                                                # dev_fridge fridge_shift_301 / fridgeL_*
        F = self.fridge
        handle, hinge2, home, shift, axis = (FRIDGE_HANDLE, FRIDGE_HINGE, FRIDGE_STANCE, FRIDGE_SHIFT, (0, 0, 1)) if door == "right" else \
            (LEFT_HANDLE, LEFT_HINGE, LEFT_STANCE, LEFT_SHIFT, (0, 0, -1))
        hw = lambda d=0.0: (self.to_base3(self.point(F, handle[0], handle[1] + d, handle[2])), self.fs(F))
        stance = lambda phi: self.stance(F, home + min(max((math.degrees(phi) - 25.0) / 60.0, 0.0), 1.0) * np.array([shift, 0.08]))
        hinge = (hinge2[0], hinge2[1], handle[2])
        sgn = 1.0 if handle[0] > hinge2[0] else -1.0                                          # along u from the hinge toward the free edge
        width = abs(-sgn * DOOR_GAP - hinge2[0])
        def door_pt(a, b):                                                                     # a along the turned door from the hinge, b out of the hinge line
            u, n = self.dirs(F)
            R = axis_rot(axis[0] * u + axis[1] * n + axis[2] * Z, self.mem.get("phi", 0.0))
            return self.to_base3(self.point(F, *hinge)) + R @ (a * sgn * u + b * n)
        return [("move", "pre_handle", lambda: hw(0.12), 1.0, 300, 0.010, True),
                ("move", "handle", lambda: hw(), 1.0, 150, 0.006),
                ("move", "grip", lambda: hw(), -1.0, 40, 0.0),
                ("verify", "fridge_handle", lambda: [("move", "regrip_off", lambda: hw(0.06), 1.0, 60, 0.01), ("move", "handle", lambda: hw(), 1.0, 150, 0.006),
                                                    ("move", "grip", lambda: hw(), -1.0, 40, 0.0)], 2),
                ("do", lambda: self.mem.update(arc_from=0.0)),
                ("arc", "pull", self.door_arc(F, hinge, axis, handle, lambda: self.fs(F), math.radians(98), math.radians(8), base_fn=stance), 500),
                ("wait", "let_go", 20, 1.0),
                ("do", lambda: self.mem.update(let_go=self.to_odom3(self.servo.pose(self.servo.q(self.s))[:3, 3]), let_go_rot=self.servo.pose(self.servo.q(self.s))[:3, :3].copy())),
                ("move", "back_off", lambda: (self.to_base3(self.mem["let_go"]) + 0.07 * self.mem["let_go_rot"][:, 2], self.mem["let_go_rot"]), 1.0, 80, 0.02),
                ("move", "around", lambda: (door_pt(width + 0.10, 0.17), self.mem["let_go_rot"]), 1.0, 150, 0.03),
                ("move", "inside", lambda: (door_pt(width + 0.10, -0.06), self.mem["let_go_rot"]), 1.0, 120, 0.03),
                ("move", "clear", lambda: (self.side([0.35, -0.25, 1.05]), self.fs(F)), 1.0, 200, 0.03)]

    def left_switch_script(self):                                                             # right compartment empty: stow the right hand, open the left door with the left arm
        F = self.fridge
        def switch():
            self.right_parked = self.action[15:22].copy()
            self.arm, self.shelves, self.look_stance = "left", LEFT_SHELVES, LOOK_LEFT
            self.slab = None
        def refine():                                                                          # in place: the door script below holds this dict
            self.fridge.update(self.local_fridge(cameras(self.data)["head"], self.fridge, door="left"))
        def search():
            self.script[self.si + 1:self.si + 1] = self.find_script()
        return [("move", "stow", lambda: (np.array([0.30, -0.30, 0.85]), self.fs(F)), 1.0, 200, 0.03),
                ("drive", "to_left_door", lambda: self.stance(F, LEFT_STANCE), 300),
                ("wait", "settle_left", 5, 1.0),
                ("do", refine),
                ("do", switch)] + self.fridge_open_script("left") + [("do", search)]

    def find_script(self):                                                                    # dev_find4: wrist camera 0.17 m above each shelf, looking 25 deg down
        F = self.fridge
        (u0, u1), shelves = self.shelves
        steps = [("drive", "to_look", lambda: self.stance(F, self.look_stance), 250)]
        for top, front in shelves:
            view = lambda top=top: (self.to_base3(self.point(F, (u0 + u1) / 2, 0.02, top + 0.12)), self.fs(F, math.radians(10)))
            steps += [("move", f"view_{top:.2f}", view, 1.0, 220, 0.012, True),
                      ("wait", f"settle_{top:.2f}", 4, 1.0),
                      ("do", lambda top=top, front=front: self.look_shelf(top, front))]
        return steps

    def dump(self, tag):                                                                      # dev aid: the allowed inputs of a look that found nothing
        try:
            np.savez_compressed(self.output / f"miss_{self.episode}_{self.step}_{tag}.npz", pose=self.pose, **self.data)
        except Exception as e:
            print("DUMP failed", e, flush=True)

    def look_shelf(self, top, front):
        if self.slab is not None:
            return
        wrist = cameras(self.data)[self.arm + "_wrist"]
        F = self.fridge
        u, n = self.dirs(F)
        frame = {"origin": self.to_base3(self.point(F, 0.0, 0.0, 0.0)), "n": n, "u": u}
        det = slab_on_shelf(wrist, frame, self.shelves[0], front, top)
        self.log.setdefault("looks", []).append([top, None if det is None else {k: (round(v, 3) if v is not None else None) for k, v in det.items()}])
        if det is not None and det["points"] >= 150:
            measured = det["shelf_edge"] is not None and abs(det["shelf_edge"] - front) < 0.05
            edge = det["shelf_edge"] if measured else front
            self.slab = dict(det, shelf=top, edge=edge, measured=measured, back=det["back"] - BACK_BIAS, pushes=0, depth=det["front"] - det["back"] + BACK_BIAS)

    def grasp_script(self):                                                                   # dev_brisket4 push, then look again and pinch against the measured edges
        return self.push_steps() + self.relook_steps()

    def push_steps(self):                                                                     # pitch 55 closed hand from behind the slab, Cartesian moves only (dev_brisket4)
        F, sl = self.fridge, self.slab
        top, edge = sl["shelf"], sl["edge"]
        start = sl["back"] - PUSH_GAP
        travel = edge + OVERHANG - sl["front"] if sl.get("measured", True) else min(edge + OVERHANG - sl["front"], 0.02)   # unmeasured edge: the prior in a drifted
        end = start + travel + PUSH_SLIP + (PUSH_GAP - 0.035)                                          # frame pushed full_v19_304's slab off the shelf; 2 cm, then the relook
        push = lambda n_, z_: (self.to_base3(self.point(F, sl["u"], n_, z_)), self.fv(F, PUSH_PITCH))
        steps = [("do", lambda: self.yaw_limit(PUSH_YAW)),
                 ("move", "entry", lambda: push(edge + 0.10, top + 0.07), -1.0, 300, 0.012, True),
                 ("move", "over", lambda: push(start, top + 0.06), -1.0, 300),
                 ("move", "down", lambda: push(start, top + 0.022), -1.0, 150)]
        k = max(1, int(math.ceil((end - start) / 0.02)))
        for i in range(1, k + 1):
            steps.append(("move", f"drag{i}", lambda i=i: push(start + (end - start) * i / k, top + 0.022), -1.0, 40))
        steps += [("move", "up", lambda: push(end, top + 0.09), -1.0, 120),
                  ("move", "back", lambda: push(edge + 0.12, top + 0.09), -1.0, 200),
                  ("do", lambda: self.yaw_limit(0.75))]
        return steps

    def relook_steps(self):                                                                   # the shelf view again: where did the push leave the slab, where is the edge
        F, sl = self.fridge, self.slab
        (u0, u1), _ = self.shelves
        view = lambda: (self.to_base3(self.point(F, min(max(sl["u"], u0 + 0.08), u1 - 0.08), 0.02, sl["shelf"] + 0.12)), self.fs(F, math.radians(10)))
        return [("move", "relook", view, 1.0, 200, 0.012), ("wait", "relook_settle", 4, 1.0), ("do", self.after_push)]

    def after_push(self):                                                                     # pinch if the slab overhangs the measured edge by >= 2 cm, else push again
        F, sl = self.fridge, self.slab
        u, n = self.dirs(F)
        frame = {"origin": self.to_base3(self.point(F, 0.0, 0.0, 0.0)), "n": n, "u": u}
        det = slab_on_shelf(cameras(self.data)[self.arm + "_wrist"], frame, self.shelves[0], sl["edge"], sl["shelf"], beyond=0.08)
        if det is None:
            self.dump("relook")
        self.log.setdefault("relooks", []).append([self.step, None if det is None else {k: (round(v, 3) if v is not None else None) for k, v in det.items()}])
        if det is not None:
            if not sl.get("measured", True) and det["shelf_edge"] is not None and abs(det["shelf_edge"] - sl["edge"]) < 0.05:
                sl.update(edge=det["shelf_edge"], measured=True)                                # no edge before the push: the relook's, the only one there is
            sl.update(u=det["u"], front=det["front"], back=det["back"] - BACK_BIAS)             # the edge stays the one measured before the push: over the slab the relook read it
                                                                                                # 2.8-3.5 cm shallow (full_v16_303 / 307: a needless re-push tipped both onto the floor)
        overhang = sl["front"] - sl["edge"] if det is not None else OVERHANG
        lo, hi = sl["edge"] + PINCH_AHEAD, sl["front"] + FRONT_AHEAD
        sl["pinch"] = (lo + hi) / 2 if det is not None and hi > lo else lo
        self.held = min(max(sl["front"] - sl["pinch"], 0.0), 0.04) if det is not None else HELD  # slab front edge toward the robot from the grasp point
        print("RELOOK", self.step, None if det is None else round(overhang, 3), sl["pushes"], flush=True)
        if det is not None and overhang < 0.02 and sl["pushes"] < 2:
            sl["pushes"] += 1
            more = self.push_steps() + self.relook_steps()
        else:
            more = self.pinch_steps()
        self.script[self.si + 1:self.si + 1] = more

    def pinch_steps(self):                                                                    # horizontal fingers under the overhang, 2.2 cm in front of the measured edge
        F, sl = self.fridge, self.slab
        top, edge = sl["shelf"], sl["edge"]
        pinch = lambda dn=0.0, dz=0.0: (self.to_base3(self.point(F, sl["u"], sl.get("pinch", edge + PINCH_AHEAD) + dn, top + 0.01 + dz)), self.fv(F))
        return [("move", "turn", lambda: pinch(0.10, 0.08), 1.0, 200),
                ("move", "lower", lambda: pinch(0.10), 1.0, 200),
                ("move", "reach", lambda: pinch(), 1.0, 150, 0.006),
                ("move", "close", lambda: pinch(), -1.0, 40, 0.0),
                ("verify", "slab_pinch", lambda: [("move", "reopen", lambda: pinch(0.08), 1.0, 80, 0.01), ("move", "reach", lambda: pinch(0.004), 1.0, 150, 0.006),
                                                 ("move", "close", lambda: pinch(0.004), -1.0, 40, 0.0)], 2, SLAB_GRIP),
                ("do", lambda: setattr(self, "holding", True)),
                ("move", "lift", lambda: pinch(0.0, 0.03), -1.0, 120),
                ("move", "out", lambda: pinch(0.35, 0.03), -1.0, 300),
                ("move", "carry", lambda: (self.side(CARRY), self.fv({"o": np.r_[self.pose[:2], 0.0], "n": -np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])})), -1.0, 200, 0.02)]

    def oven_look_steps(self, tag):                                                           # wrist camera 0.17 m above the rack, outside the cavity, looking 35 deg down
        F = self.oven
        view = lambda: (self.to_base3(self.point(F, RACK_U, 0.06, RACK_TOP_Z + 0.17)), self.fs(F, math.radians(10)))
        return [("move", f"look_{tag}", view, 1.0, 250, 0.012, True), ("wait", f"look_settle_{tag}", 4, 1.0), ("do", lambda: self.oven_measure(tag))]

    def oven_measure(self, tag):
        F = self.oven
        u, n = self.dirs(F)
        frame = {"origin": self.to_base3(F["o"]), "n": n, "u": u}
        det = slab_on_shelf(cameras(self.data)[self.arm + "_wrist"], frame, (RACK_U - 0.19, RACK_U + 0.19), RACK_EDGE_N, RACK_TOP_Z, beyond=0.06)
        if det is None:
            self.dump("oven_" + tag)
        self.log.setdefault("oven_looks", []).append([self.step, tag, None if det is None else {k: (round(v, 3) if v is not None else None) for k, v in det.items()}])
        print("OVENLOOK", self.step, tag, None if det is None else {k: (round(v, 3) if v is not None else None) for k, v in det.items()}, flush=True)
        self.mem["look_ok"] = det is not None
        if det is None:
            return
        if det["shelf_edge"] is not None and abs(det["shelf_edge"] - RACK_EDGE_N) < 0.03:
            self.mem["rack_edge"] = det["shelf_edge"]
        self.mem["oven_slab"] = det

    def place_script(self):                                                                   # dev_oven7 steps 3 and 4
        F = self.oven
        at = lambda a, b, c, rot=None: (self.to_base3(self.point(F, a, b, c)), self.fv(F) if rot is None else rot)
        place_n = RACK_EDGE_N + OVEN_OVERHANG - self.held
        depth = self.slab["depth"] if self.slab is not None else 0.13
        z_place = RACK_TOP_Z + 0.013
        def push_to():                                                                        # (front now, front target): the slab centre to PUSH_CENTRE, front clear of the shut door
            det = self.mem.get("oven_slab")
            if det is not None:
                front_now = det["front"]
                target = front_now - max((det["front"] + det["back"]) / 2 - PUSH_CENTRE, 0.0)
            else:
                front_now = place_n + self.held
                target = PUSH_CENTRE + depth / 2
            target = min(target, DOOR_INNER_N - 0.003)
            self.mem["push_front"] = (front_now, target)
            return front_now, target
        world_z = lambda z: z - F["o"][2]                                                     # absolute heights into the oven frame
        def edge_at(phi):                                                                     # door top edge grasp pose at door angle phi
            u, n = self.dirs(F)
            H = self.to_base3(self.point(F, *OVEN_HINGE))
            r0 = self.to_base3(self.point(F, *(OVEN_HINGE + OVEN_EDGE))) - H
            R = axis_rot(-u, phi)
            return H + R @ r0, R @ axis_rot(-u, -math.pi / 2) @ self.fv(F)
        close_base = lambda phi: self.stance(F, self.oven_stance() + np.array([0.0, OVEN_CLOSE_BACK * min(max(phi, 0.0) / (math.pi / 2), 1.0)]))
        return [("move", "oven_entry", lambda: at(RACK_U, 0.405, world_z(0.60), self.fv(F, math.radians(15))), -1.0, 300, 0.012, True),   # planned: full_v16_310 stalled 0.16 m off;
                                                                                                # below the cavity ceiling (z 0.602): from 0.68 full_v19_301's slab hung at z 0.62
                ("move", "oven_in", lambda: at(RACK_U, place_n, world_z(0.54), self.fv(F, math.radians(15))), -1.0, 250),
                ("move", "level", lambda: at(RACK_U, place_n, world_z(0.52)), -1.0, 150),
                ("move", "lower", lambda: at(RACK_U, place_n, z_place), -1.0, 150),
                ("wait", "release", 30, 1.0),
                ("do", lambda: setattr(self, "holding", False)),
                ("move", "retract", lambda: at(RACK_U, place_n + 0.10, z_place), 1.0, 120)] + self.oven_look_steps("placed") + [
                ("do", lambda: self.mem.update(push_extra=0.0, pushes_in=0))] + self.push_in_steps(at, push_to) + [
                ("move", "clear", lambda: at(RACK_U, 0.475, world_z(0.68)), 1.0, 250, 0.02),
                ("drive", "back", lambda: self.stance(F, self.oven_stance() + np.array([0.0, OVEN_CLOSE_BACK])), 220),
                ("move", "edge_pre", lambda: (edge_at(math.pi / 2)[0] + 0.10 * self.dirs(F)[1], edge_at(math.pi / 2)[1]), 1.0, 300, 0.012),
                ("move", "edge", lambda: edge_at(math.pi / 2), 1.0, 150, 0.006),
                ("move", "edge_grip", lambda: edge_at(math.pi / 2), -1.0, 40, 0.0),
                ("verify", "door_edge", lambda: [("move", "edge_off", lambda: (edge_at(math.pi / 2)[0] + 0.06 * self.dirs(F)[1], edge_at(math.pi / 2)[1]), 1.0, 60, 0.01),
                                                ("move", "edge", lambda: edge_at(math.pi / 2), 1.0, 150, 0.006), ("move", "edge_grip", lambda: edge_at(math.pi / 2), -1.0, 40, 0.0)], 2, DOOR_GRIP),
                ("do", lambda: self.mem.update(arc_from=math.pi / 2)),
                ("arc", "close", self.door_arc(F, OVEN_HINGE, (-1, 0, 0), OVEN_HINGE + OVEN_EDGE, lambda: axis_rot(-self.dirs(F)[0], -math.pi / 2) @ self.fv(F),
                                               math.radians(22), math.radians(10), base_fn=close_base), 500),
                ("wait", "let_go2", 15, 1.0),
                ("move", "up", lambda: (edge_at(self.mem.get("phi", 0.4))[0] + 0.10 * self.dirs(F)[1] + 0.06 * Z, edge_at(self.mem.get("phi", 0.4))[1]), 1.0, 40, 0.02),
                ("move", "face", lambda: at(RACK_U, DOOR_FACE_N + 0.10 + math.tan(self.mem.get("phi", 0.4) + 0.05) * (world_z(0.55) - OVEN_HINGE[2]), world_z(0.55)), -1.0, 80, 0.015),
                ("move", "push_shut", lambda: at(RACK_U, DOOR_FACE_N + 0.019 - 0.004, world_z(0.55)), -1.0, 120),
                ("wait", "hold_shut", 10, -1.0),
                ("move", "off_door", lambda: at(RACK_U, 0.22, world_z(0.62)), -1.0, 100, 0.02),
                ("drive", "to_stance", lambda: self.stance(F, self.oven_stance()), 150),
                ("move", "rest", lambda: (self.side([0.30, -0.25, 0.85]), self.fv(F)), 1.0, 150, 0.03),
                ("do", lambda: self.mem.update(presses=0)),
                ("wait", "marker_settle2", 4, 1.0),
                ("do", self.check_on)]

    def push_in_steps(self, at, push_to):                                                    # fist, push, look; again by the shortfall
        tip = lambda: push_to()[1] + 0.019 - self.mem.get("push_extra", 0.0)
        return [("wait", "close_fist", 15, -1.0),
                ("move", "fist", lambda: at(RACK_U, push_to()[0] + 0.05, RACK_TOP_Z + 0.016), -1.0, 120),
                ("move", "push", lambda: at(RACK_U, tip(), RACK_TOP_Z + 0.016), -1.0, 120),
                ("move", "unpush", lambda: at(RACK_U, tip() + 0.05, RACK_TOP_Z + 0.016), -1.0, 80),
                ("move", "unpush_up", lambda: at(RACK_U, push_to()[1] + 0.10, RACK_TOP_Z + 0.04), -1.0, 120)] + self.oven_look_steps("pushed") + [
                ("do", lambda: self.check_push(at, push_to))]

    def check_push(self, at, push_to):
        front, target = push_to()
        short = front - target
        self.log.setdefault("push_in", []).append([self.step, round(front, 4), round(target, 4), self.mem.get("look_ok"), round(self.mem.get("push_extra", 0.0), 4)])
        print("PUSHIN", self.step, round(front, 4), round(target, 4), self.mem.get("look_ok"), round(self.mem.get("push_extra", 0.0), 4), flush=True)
        if self.mem.get("look_ok") and short > 0.003 and self.mem.get("pushes_in", 0) < 2:   # small re-pushes only: a 4 cm one wedged under the slab (full_v15_309)
            self.mem["pushes_in"] = self.mem.get("pushes_in", 0) + 1
            self.mem["push_extra"] = min(self.mem.get("push_extra", 0.0) + short + 0.002, 0.015)
            self.script[self.si + 1:self.si + 1] = self.push_in_steps(at, push_to)

    def retrieve_script(self):                                                                # dev_oven7 step 7
        F = self.oven
        at = lambda a, b, c, rot=None: (self.to_base3(self.point(F, a, b, c)), self.fv(F) if rot is None else rot)
        world_z = lambda z: z - F["o"][2]
        def pinch_n():                                                                         # behind the slab front measured by the "cooked" look (2 mm), else by the rack edge
            det = self.mem.get("oven_slab")
            return det["front"] - OVEN_PINCH_BEHIND if det is not None and self.mem.get("look_ok") else RACK_EDGE_N + 0.005
        slab_u = lambda: self.mem["oven_slab"]["u"] if self.mem.get("oven_slab") is not None else RACK_U
        c, s_ = math.cos(OVEN_PINCH_PITCH), math.sin(OVEN_PINCH_PITCH)
        pinch = lambda back=0.0, dn=0.0, dz=0.0: at(slab_u(), pinch_n() + dn + back * c, RACK_TOP_Z + 0.010 + dz + back * s_, self.fv(F, OVEN_PINCH_PITCH))   # back: out along the fingers
        return self.oven_open_script(press=False) + [("do", lambda: self.mem.update(oven_slab=None, look_ok=False))] + self.oven_look_steps("cooked") + [
            ("move", "pinch_entry", lambda: at(slab_u(), 0.405, world_z(0.62), self.fv(F, math.radians(15))), 1.0, 300, 0.012, True),
            ("move", "pinch_pre", lambda: pinch(0.10, dz=0.03), 1.0, 300),
            ("move", "pinch_reach", lambda: pinch(), 1.0, 150, 0.004),
            ("move", "pinch_close", lambda: pinch(), -1.0, 40, 0.0),
            ("verify", "oven_pinch", lambda: [("move", "pinch_reopen", lambda: pinch(0.08), 1.0, 80, 0.01),
                                             ("move", "pinch_reach", lambda: pinch(dn=-0.003), 1.0, 150, 0.004),
                                             ("move", "pinch_close", lambda: pinch(dn=-0.003), -1.0, 40, 0.0)], 2, OVEN_GRIP),
            ("do", lambda: setattr(self, "holding", True)),
            ("move", "pinch_lift", lambda: pinch(dz=0.03), -1.0, 100),
            ("move", "pinch_out", lambda: pinch(dn=0.35, dz=0.06), -1.0, 300),
            ("move", "carry", lambda: (self.side(CARRY), self.fv(F)), -1.0, 200, 0.02)]

    def board_script(self):                                                                   # hand over the board centre, slab 3 cm above it, let go
        c = self.board["c"]
        fwd = lambda: np.array([1.0, 0.0, 0.0])
        rot = lambda: self.fv({"o": np.r_[self.pose[:2], 0.0], "n": -np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])})
        over = lambda dz: (self.to_base3(c) - 0.05 * fwd() + np.array([0.0, 0.0, dz]), rot())
        return [("move", "over_board", lambda: over(0.12), -1.0, 300, 0.012),
                ("move", "lower_board", lambda: over(0.04), -1.0, 150),
                ("wait", "drop", 30, 1.0),
                ("do", lambda: setattr(self, "holding", False)),
                ("move", "away", lambda: over(0.20), 1.0, 100, 0.02),
                ("move", "home", lambda: (self.side(CARRY), rot()), 1.0, 150, 0.03)]

    # main loop ----------------------------------------------------------------------------------
    def forward(self, obs):
        data = allowed(obs)
        self.data = data
        s = data["robot_r1::proprio"].reshape(-1)                                          # [61]
        self.s = s
        self.step += 1
        self.age += 1
        commanded = self.action is not None and float(np.max(np.abs(self.action[0:3]))) > 1e-3
        self.still = 0 if commanded else self.still + 1
        if self.still < 15:                                                                   # zero-velocity update: full_v17_305 read -0.011 m/s for 240 steps of arm motion with the
            self.pose[:2] += rot2(self.pose[2]) @ s[0:2] * DT                                # base truly still (8.8 cm of odometry drift, the door edge missed); base_qvel is in
            w_cmd = 0.0 if self.action is None else self.action[2] * MAX_W                    # the base frame (robots/robot.py:1636)
            w = s[2] if (abs(w_cmd) < 0.05 or w_cmd * s[2] <= 0) else math.copysign(min(abs(s[2]), abs(w_cmd)), s[2])   # proprio reads in-place turns 5-15% high (hiding_Easter_eggs odom_cal2_301)
            self.pose[2] = wrap(self.pose[2] + w * DT)
        self.vel = s[0:3]
        if self.step % 10 == 0:
            self.log.setdefault("odom", []).append([self.step, *np.round(self.pose, 4).tolist(), *np.round(s[0:3], 4).tolist(), self.phase, self.vo.stats.get("reason")])
        cams = cameras(data)
        head = cams["head"]
        q = self.servo.q(s)
        driving = self.action is not None and np.linalg.norm(self.action[0:2]) * MAX_V > 0.02
        turning = self.action is not None and abs(self.action[2]) * MAX_W > 0.05
        scripted = self.phase in ("oven_open", "fridge_open", "place", "retrieve", "grasp_run", "board_place")
        if self.step % 5 == 1 and (driving or turning) and not scripted:                      # standing still at the oven (arm and door moving in view) VO drifted the pose
            small = resized(head, 480)                                                        # 6 cm and turning in place 0.24 m (full_v3); yaw-only driving drifted 0.19 m (full_v5_302)
            small["moving_object_mask"] = self.arm_mask(head, q, 480)
            corrected = self.vo.update(small, self.step, self.pose)
            if driving:
                self.pose = corrected
            self.pose[2] = wrap(corrected[2])
        elif self.step % 5 == 1:
            self.vo.anchor = None
        if self.action is None:
            self.action = np.r_[np.zeros(3), s[53:57], s[3:10], 1.0, s[28:35], 1.0]        # [23] trunk, left arm, left grip, right arm, right grip
            self.left = s[3:10].copy()
            self.torso0 = s[53:57].copy()
        action = self.action.copy()
        action[0:3] = 0.0
        if self.arm == "right":
            action[7:14] = self.left
        else:
            action[15:22] = self.right_parked
        if self.step % 6 == 1:
            self.map.update(head, self.pose, self.attached(q))
        if self.phase in ("scan", "go_oven", "go_fridge", "go_oven2", "go_board") and self.step % 3 == 0:
            self.detect(head)

        if self.phase == "scan":                                                              # turn in place until the oven and the fridge are both known
            action[2] = 0.5 / MAX_W
            self.turned += abs(self.vel[2]) * DT
            if self.oven is not None and (self.fridge is not None or self.turned > 2 * math.pi) and self.turned > 1.0:
                self.change("go_oven")
            elif self.turned > 2 * math.pi + 0.3:
                self.mem["views"] = self.mem.get("views", 0) + 1
                if self.oven is None and self.fridge is not None and not self.mem.get("seek_done"):   # 302: the oven hides behind the dining table; it is 1.7 m from the fridge
                    self.mem["seek_oven"] = True
                    self.change("go_fridge")
                else:
                    self.change("explore" if self.mem["views"] <= 4 else ("go_oven" if self.oven is not None else "done"))
        elif self.phase == "explore":                                                         # next frontier viewpoint (hiding_Easter_eggs map), then scan again
            if self.path is None or self.age - self.planned > 150:
                visited = self.mem.setdefault("visited", [np.zeros(2)])
                goal, _ = self.map.viewpoint(self.pose[:2], visited, view_range=4.0)
                self.planned = self.age
                if goal is not None:
                    path = self.map.plan(self.pose[:2], goal[None], radius=0.36)
                    self.path = None if path is None else np.vstack([path, goal])
            arrived = self.drive(action) if self.path is not None else False
            if arrived or self.age > 900 or (self.oven is not None and self.fridge is not None):
                self.mem["visited"].append(self.pose[:2].copy())
                self.change("scan")
        elif self.phase in ("go_oven", "go_oven2"):
            goal, heading = self.stance(self.oven, self.oven_stance())
            if self.mem.get("noplan", 0) >= 5 and not self.holding:                         # unreachable: drop that oven cluster and look again
                self.oven_hits = [h for h in self.oven_hits if np.linalg.norm(h["o"][:2] - self.oven["o"][:2]) > 0.3]
                self.log.setdefault("dropped", []).append(["oven_unreachable", np.round(self.oven["o"], 3).tolist()])
                self.oven, self.mem["lock_oven"], self.mem["noplan"] = None, False, 0
                self.change("scan")
                self.action = action
                return torch.as_tensor(action[None, :], dtype=torch.float32)
            if self.go(action, goal, heading) or (self.age > 2500):
                self.change("refine_oven")
                self.mem["refine"] = []
        elif self.phase == "refine_oven":                                                     # fresh button + panel normal from the stance before touching anything
            if self.age % 3 == 0:
                b = oven_button(head)
                if b is not None:
                    n = self.oven_normal(head, b["xyz"])
                    if n is not None:
                        self.mem["refine"].append(self.oven_frame(b["xyz"], n))
            if self.age >= 30:
                ref = [r for r in self.mem["refine"] if np.linalg.norm(r["o"][:2] - self.oven["o"][:2]) < (1.0 if self.mem.get("lock_oven") else 0.35)]
                self.log.setdefault("refines", []).append([self.step, "oven", len(self.mem["refine"]), len(ref), self.holding, self.arm])
                print("REFINE oven", self.step, len(self.mem["refine"]), len(ref), self.holding, self.arm, flush=True)
                if not ref:                                                                   # full_v14_304 (left arm, holding): no button seen, a 7.6 cm stale frame kept
                    self.dump("refine_oven")
                if not ref and (self.oven.get("votes", 0) >= 3 or self.mem.get("lock_oven")): # keep a well-voted or already refined oven when the stance view misses the button
                    ref = [{"o": self.oven["o"], "n": self.oven["n"]}]
                if not ref:                                                                   # no button at the stance: a false cluster; drop it and scan again
                    self.oven_hits = [h for h in self.oven_hits if np.linalg.norm(h["o"][:2] - self.oven["o"][:2]) > 0.3]
                    self.log.setdefault("dropped", []).append(["oven", np.round(self.oven["o"], 3).tolist()])
                    self.oven, self.mem["lock_oven"] = None, False
                    self.change("scan")
                    self.action = action
                    return torch.as_tensor(action[None, :], dtype=torch.float32)
                n = np.median([r["n"] for r in ref], 0)
                self.oven = {"o": np.median([r["o"] for r in ref], 0), "n": n / np.linalg.norm(n), "votes": max(len(ref), self.oven.get("votes", 0))}
                self.mem["lock_oven"] = True
                if not self.holding:                                                          # the door is shut here (first visit, and after cooking)
                    self.oven = self.local_oven(head, self.oven)
                goal, heading = self.stance(self.oven, self.oven_stance())
                if np.linalg.norm(goal - self.pose[:2]) > 0.05 and self.mem.get("rego", 0) < 2:
                    self.mem["rego"] = self.mem.get("rego", 0) + 1
                    self.change("go_oven2" if self.holding else "go_oven")
                else:
                    self.mem["rego"] = 0
                    if self.holding:
                        self.start(self.place_script(), "cook")
                        self.change("place")
                    elif self.cook_start is not None:
                        self.start(self.retrieve_script(), "go_board")
                        self.change("retrieve")
                    else:
                        self.start(self.oven_open_script(press=True), "go_fridge")
                        self.change("oven_open")
        elif self.phase == "go_fridge":
            if self.mem.get("seek_oven") and self.oven is not None:                           # the oven came into view on the way: it goes first
                self.mem["seek_oven"] = False
                self.change("go_oven")
            elif self.fridge is None:
                action[2] = 0.4 / MAX_W
            else:
                goal, heading = self.stance(self.fridge, FRIDGE_STANCE)
                if self.mem.get("noplan", 0) >= 5:                                            # unreachable fridge cluster: drop it and keep turning
                    self.fridge_hits = [h for h in self.fridge_hits if np.linalg.norm(h["o"][:2] - self.fridge["o"][:2]) > 0.3]
                    self.log.setdefault("dropped", []).append(["fridge_unreachable", np.round(self.fridge["o"], 3).tolist()])
                    self.fridge, self.mem["lock_fridge"], self.mem["noplan"] = None, False, 0
                    self.action = action
                    return torch.as_tensor(action[None, :], dtype=torch.float32)
                if self.go(action, goal, heading) or self.age > 2500:
                    if self.mem.get("seek_oven"):                                             # at the fridge without the oven: look round once more
                        self.mem["seek_oven"], self.mem["seek_done"] = False, True
                        self.change("scan")
                    else:
                        self.change("refine_fridge")
                        self.mem["refine"] = []
        elif self.phase == "refine_fridge":
            if self.age % 3 == 0:
                f = fridge_front(head)                                                       # close up the fridge fills the view: no prior (a 12 deg / 0.15 m gate dropped
                if f is not None:                                                             # the true fridge after 0.16 m of odometry drift, full_v5_302)
                    self.mem["refine"].append(self.frame_from(f["origin"], f["n"]))
            if self.age >= 30:
                ref = [r for r in self.mem["refine"] if np.linalg.norm(r["o"][:2] - self.fridge["o"][:2]) < 0.25 and r["n"] @ self.fridge["n"] > math.cos(math.radians(10))]
                if len(ref) < 3 and (self.fridge.get("votes", 9) >= 3 or self.mem.get("lock_fridge")):   # close-up detections can be wrong (23 deg off on full_v6_301):
                    ref = [{"o": self.fridge["o"], "n": self.fridge["n"]}]                     # keep the vote, or a frame refined on an earlier visit (full_v11_301 dropped it)
                if not ref:                                                                   # nothing fridge-like at the stance: a false cluster; drop it and search on
                    self.fridge_hits = [h for h in self.fridge_hits if np.linalg.norm(h["o"][:2] - self.fridge["o"][:2]) > 0.3]
                    self.log.setdefault("dropped", []).append(["fridge", np.round(self.fridge["o"], 3).tolist()])
                    self.fridge, self.mem["lock_fridge"] = None, False
                    self.change("go_fridge")
                    self.action = action
                    return torch.as_tensor(action[None, :], dtype=torch.float32)
                n = np.median([r["n"] for r in ref], 0)
                self.fridge = {"o": np.median([r["o"] for r in ref], 0), "n": n / np.linalg.norm(n), "votes": max(len(ref), self.fridge.get("votes", 0))}
                self.mem["lock_fridge"] = True
                self.fridge = self.local_fridge(head, self.fridge)
                goal, heading = self.stance(self.fridge, FRIDGE_STANCE)
                if np.linalg.norm(goal - self.pose[:2]) > 0.05 and self.mem.get("rego", 0) < 2:
                    self.mem["rego"] = self.mem.get("rego", 0) + 1
                    self.change("go_fridge")
                else:
                    self.mem["rego"] = 0
                    self.start(self.fridge_open_script() + self.find_script(), "grasp")
                    self.change("fridge_open")
        elif self.phase in ("oven_open", "fridge_open", "place", "retrieve", "grasp_run", "board_place"):
            if self.tick(action, q):
                nxt = self.after
                failed = self.mem.pop("failed", None)
                if failed is not None:
                    self.log.setdefault("failed", []).append([self.step, self.phase, failed])
                    self.mem["redo_" + self.phase] = self.mem.get("redo_" + self.phase, 0) + 1
                if failed == "slab_pinch" and self.mem["redo_grasp_run"] <= 2:                  # empty hand: look at the shelf again and redo the push and pinch
                    self.slab, self.holding = None, False
                    for k in [k for k in self.mem if k.startswith("tries_")]:
                        del self.mem[k]
                    self.start(self.find_script(), "grasp")
                    self.change("fridge_open")
                    self.action = action
                    return torch.as_tensor(action[None, :], dtype=torch.float32)
                if failed == "oven_handle" and self.phase == "oven_open" and self.mem["redo_oven_open"] <= 1:
                    for k in [k for k in self.mem if k.startswith("tries_")]:
                        del self.mem[k]
                    self.change("refine_oven")
                    self.mem["refine"] = []
                    self.action = action
                    return torch.as_tensor(action[None, :], dtype=torch.float32)
                if self.phase == "fridge_open":
                    if self.slab is None:
                        self.mem["finds"] = self.mem.get("finds", 0) + 1
                        if self.arm == "right":                                               # 304 / 316 / 319: the brisket is behind the left door
                            self.start(self.left_switch_script(), "grasp")
                            self.change("fridge_open")
                            self.action = action
                            return torch.as_tensor(action[None, :], dtype=torch.float32)
                        if self.mem["finds"] <= 3:                                            # look again (the frame refines between tries)
                            self.start(self.find_script(), "grasp")
                            self.change("fridge_open")
                            self.action = action
                            return torch.as_tensor(action[None, :], dtype=torch.float32)
                        nxt = "done"
                    else:
                        self.start(self.grasp_script(), "go_oven2")
                        nxt = "grasp_run"
                if self.phase == "place":
                    self.cook_start = self.step
                if self.phase == "board_place":
                    nxt = "done"
                self.change(nxt)
        elif self.phase == "cook":                                                            # the oven heats the brisket ~90 steps once shut; wait 150
            if self.step - self.cook_start > 150:
                self.change("refine_oven")
                self.mem["refine"] = []
        elif self.phase == "go_board":
            if self.board is None:
                action[2] = 0.3 / MAX_W
            else:
                goal, heading = self.board_stance()
                if goal is not None and (self.go(action, goal, heading) or self.age > 2500):
                    self.change("refine_board")
                    self.mem["refine"] = []
        elif self.phase == "refine_board":
            if self.age % 3 == 0:
                k = detect_board(head)
                if k is not None and np.linalg.norm(self.to_odom3(k["centre"])[:2] - self.board["c"][:2]) < 1.0:
                    self.mem["refine"].append(self.to_odom3(k["centre"]))
            if self.age >= 30:
                if len(self.mem["refine"]) >= 3:
                    self.board = {"c": np.median(self.mem["refine"], 0), "votes": len(self.mem["refine"])}
                    self.board_hits = [{"c": c} for c in self.mem["refine"]]
                self.log.setdefault("board_refine", []).append([self.step, len(self.mem["refine"]), np.round(self.board["c"], 3).tolist()])
                print("BOARD", self.step, len(self.mem["refine"]), np.round(self.board["c"], 3).tolist(), flush=True)
                goal, heading = self.board_stance()
                if goal is not None and np.linalg.norm(goal - self.pose[:2]) > 0.10 and self.mem.get("rego_board", 0) < 1:
                    self.mem["rego_board"] = 1
                    self.change("go_board")
                else:
                    self.start(self.board_script(), "done")
                    self.change("board_place")
        elif self.phase == "done":
            pass
        if self.phase not in ("oven_open", "fridge_open", "place", "retrieve", "grasp_run", "board_place"):
            q = self.servo.q(s)
            hold = self.side(CARRY if self.holding else [0.30, -0.25, 0.85])
            action[3:7] = s[53:57] + np.clip(self.torso0 - s[53:57], -0.016, 0.016)
            if self.phase not in ("refine_oven", "refine_fridge", "refine_board", "cook"):
                self.holders[self.arm].step(action, q, hold, self.fv({"o": np.r_[self.pose[:2], 0.0], "n": -np.array([math.cos(self.pose[2]), math.sin(self.pose[2])])}))
            action[self.gi] = -1.0 if self.holding else 1.0
        self.action = action
        if self.step % 300 == 0 or self.phase == "done" and self.age == 1:
            (self.output / f"log_{self.episode}.json").write_text(json.dumps(self.log, default=float))
        return torch.as_tensor(action[None, :], dtype=torch.float32)

    def board_stance(self):                                                                   # free spot 0.62 m from the board facing it, nearest to the robot
        c = self.board["c"][:2]
        angles = np.linspace(-math.pi, math.pi, 16, endpoint=False)
        goals = c + 0.62 * np.c_[np.cos(angles), np.sin(angles)]
        order = np.argsort(np.linalg.norm(goals - self.pose[:2], axis=1))[:8]
        path = self.map.plan(self.pose[:2], goals[order], radius=0.36)
        g = goals[order[0]] if path is None else goals[order][np.argmin(np.linalg.norm(goals[order] - path[-1], axis=1))]
        d = c - g
        return g, math.atan2(d[1], d[0])

    def attached(self, q):                                                                    # capsules (a, b, radius) moving with the robot: arm links and a held slab
        caps = self.servo.capsules(q)
        if self.holding:
            hand = self.servo.pose(q)
            caps = caps + [(hand[:3, 3], hand[:3, 3] + hand[:3, :3] @ np.array([0.0, 0.0, -0.2]), 0.12)]
        return caps

    def arm_mask(self, head, q, size):                                                        # [size, size] pixels showing the arms (moving with the robot)
        tiny = resized(head, 120)
        valid = np.isfinite(tiny["depth"]) & (tiny["depth"] > 0.06) & (tiny["depth"] < 10.0)
        p, _ = pointcloud(tiny, stride=1, max_range=10.0)
        moving = np.zeros(len(p), bool)
        for a, b, radius in self.attached(q):
            axis = b - a
            t = np.clip((p - a) @ axis / max(axis @ axis, 1e-9), 0, 1)
            moving |= np.linalg.norm(p - a - t[:, None] * axis, axis=1) < radius + 0.05
        mask = np.zeros(valid.shape, np.uint8)
        mask[valid] = moving
        return cv2.dilate(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST), np.ones((7, 7), np.uint8)) > 0
