"""Hiding Easter eggs controller: search the garden for the wicker basket, then move its eggs next to a tree.
Inputs: onboard RGB-D, 61-D proprio and camera poses relative to the base. Pose is integrated base velocity (odometry frame = start pose)."""
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from scipy.spatial.transform import Rotation

from arm_servo import ArmServo
from basket_perception import basket_rim, detect_baskets
from egg_perception import top_egg
from geometry import cameras, ground_tilt, pointcloud, project, resized
from occupancy import COARSE, N, RES, GardenMap, rot2, wrap
from tree_perception import detect_trunks
from visual_odometry import VisualOdometry

cv2.setNumThreads(2)                                                                          # shared host: keep OpenCV off the other users' cores
DT = 1 / 30                                                                                   # s per action
MAX_V, MAX_W = 0.75, 1.0                                                                      # m/s, rad/s at action 1 (eval/r1pro.yaml base limits)
GRIP, FINGERS = 14, slice(24, 26)                                                             # left gripper action and finger joints (1 open, -1 close)
UP = np.array([0.0, 0.0, 1.0])
RIM_RADIUS = 0.165                                                                            # grasp point from the fitted rim centre (sweep_rim_301)
RIM_DEPTH = 0.07                                                                              # servo point below the fitted rim top: the top 4 cm of the rim has no collider (sweep_rim_301)
GRASP_ANGLE = -math.pi / 2                                                                    # grasp point around the rim from the robot-facing point: the left rim (pour3_301)
CARRY_Z = 0.95                                                                                # hand height while carrying, base frame (full_v8_312 knocked the basket at 0.85)
POUR_Z = 0.55                                                                                 # hand height while tipping: eggs within 0.18 m of their centroid (pour5_301), 0.8 m spread from 0.85
POUR_DEG = 160                                                                                # tilt about the rim tangent, mouth away from the hand: 3/3 eggs out in pour3_301
SHAKE_DEG = 15
EGG_DEPTH = 0.035                                                                             # right-hand pinch: servo point below the egg's top (egg_d035_301 2/3 vs egg_d055_301 1/3)
RGRIP, RFINGERS = 22, slice(49, 51)                                                           # right gripper action and finger joints
SWEET_PICK = np.array([0.60, -0.08])                                                          # where the base puts a pick target for the right hand (base frame)
TRUNK_DIR = math.radians(-65)                                                                 # basket -> trunk direction (base frame) at the pick pose: static IK reaches every spot (place_reach)
EGG_RING = 0.22                                                                               # egg spots this far from the trunk surface, 40 degrees apart round it
VIEW_ROT = Rotation.from_rotvec([0.0, math.radians(-25.0), 0.0]).as_matrix()                 # egg view: the wrist camera sits at 25 deg in the gripper frame (cam_mount, 6 frames);
                                                                                              # pitched so it looks straight down (unpitched it lost an egg at the image edge: full_v23_301)
LEFT_TUCK = np.array([0.504, 0.127, 0.218, -0.688, 0.031, -0.797, 0.205])                    # left arm joints during the egg phases, hand behind the shoulder: in the basket zone in
                                                                                              # 8/173 egg-phase torso postures vs 30/173 at the let_go park (tuck_search; a Cartesian hold diverged)
PLACE_Z = 0.70                                                                                # hand height while carrying an egg to its spot: the egg hangs ~0.11 m below the hand, the rim top is ~0.40 m;
                                                                                              # a planned move down to 0.45 dragged the egg over the rim and tipped the basket (dev22_301, dev22_302)
SET_GAP = 0.45                                                                                # trunk surface beyond the set-down basket's edge
RELEASE_Z = 0.15 - EGG_DEPTH                                                                  # servo height that puts the egg's bottom ~1 cm above the lawn
TRUNK_GAP = 0.35                                                                              # trunk surface ahead of the hanging basket's centre when tipping (basket radius 0.20 + 0.15 clearance):
                                                                                              # full_v3_306 (0.35 m centre to centre) and full_v18_306 (0.30 + radius 0.12) jammed the basket


def allowed(obs):                                                                             # the closed input set: proprio, camera poses, RGB-D
    return {k: v.detach().cpu().numpy() for k, v in obs.items()
            if k == "robot_r1::proprio" or k == "robot_r1::cam_rel_poses" or (k.startswith("robot_r1::robot_r1:") and k.endswith(("::rgb", "::depth_linear")))}


class EasterPolicy:
    def __init__(self, *args, **kwargs):
        self.output = Path(os.environ.get("POLICY_OUTPUT", "/tmp/easter_policy"))
        self.output.mkdir(parents=True, exist_ok=True)
        self.servo = ArmServo(Path(__file__).parent, arm="left")
        self.rservo = ArmServo(Path(__file__).parent, arm="right")
        self.episode = -1

    def set_action_dim(self, n):
        assert n == 23

    def reset(self):
        self.episode += 1
        self.step, self.age, self.phase = 0, 0, "scan"
        self.pose = np.zeros(3)                                                               # x, y, yaw in the start frame
        self.vo = VisualOdometry(interval=5)                                                  # head ORB + depth; proprio yaw rate reads ~14% high while turning (search_v1_301)
        self.map = GardenMap()
        self.candidates = []                                                                  # basket hypotheses in the start frame
        self.basket = None
        self.target = None                                                                    # the candidate being approached
        self.approach_angle = None
        self.visited = [np.zeros(2)]
        self.goal, self.heading, self.path, self.planned = None, None, None, -1000
        self.turned = 0.0
        self.action = None
        self.stuck = 0
        self.trees = []                                                                       # trunk hypotheses in the start frame
        self.rims = []
        self.grasp = None                                                                     # base-frame grasp point and hand rotation
        self.attempts = 0
        self.holding = False
        self.tree = None
        self.hand = None
        self.gap = TRUNK_GAP
        self.stalled, self.backup, self.blocked_dir, self.vel = 0, 0, np.zeros(2), np.zeros(3)
        self.tilt = 0.0
        self.backoffs = 0
        self.placed, self.egg, self.spots_w, self.views, self.fails, self.basket_w = 0, None, [], [], 0, None
        self.ring = False                                                                     # approach leg to a point 2.5 m from a far candidate
        self.reposition, self.view_seen = 0, False                                            # egg views that never saw the basket: base moves so far
        self.log = {}

    def change(self, phase):
        print("PHASE", self.step, self.phase, "->", phase, flush=True)
        self.phase, self.age = phase, 0
        self.path, self.planned, self.turned, self.stuck, self.ring, self.view_seen = None, -1000, 0.0, 0, False, False
        self.servo.target = None
        self.rservo.target, self.rservo.synchronized = None, False
        if phase == "approach":
            self.approach_angle = None

    # perception -------------------------------------------------------------------------------
    def to_world(self, xy):
        return rot2(self.pose[2]) @ np.asarray(xy) + self.pose[:2]

    def to_base(self, xy):
        return rot2(self.pose[2]).T @ (np.asarray(xy) - self.pose[:2])

    def toward_cell(self, xy):                                                                # reachable planning cell nearest to xy, if >= 1 m nearer than the robot
        travel = self.map.travel_costs(self.pose[:2])
        ii, jj = np.nonzero(np.isfinite(travel))
        if len(ii) == 0:
            return None
        cells = (np.c_[ii, jj] * COARSE + COARSE / 2 - N / 2) * RES                             # [k, 2] odometry frame
        d = np.linalg.norm(cells - xy, axis=1)
        k = int(np.argmin(d))
        return cells[k] if d[k] < np.linalg.norm(self.pose[:2] - xy) - 1.0 else None

    def update_candidates(self, head):                                                        # far detections only nominate; detections within 4.5 m confirm
        found = detect_baskets(head, max_range=10.0)                                          # beyond ~8 m the detector is unreliable (dev_views2_301, full_v2_303 false positive at 16.5 m)
        for d in found:
            w = self.to_world(d["xy"])
            near = [c for c in self.candidates if np.linalg.norm(c["xy"] - w) < 0.6]
            if near:
                c = near[0]
                k = min(c["hits"], 8)
                c["xy"] = (k * c["xy"] + w) / (k + 1)
                c["hits"] += 1
                c["last"] = self.step
                c["close"] += int(d["range"] < 4.5)
            else:
                self.candidates.append({"xy": w, "hits": 1, "misses": 0, "close": int(d["range"] < 4.5), "close_misses": 0, "last": self.step, "dead": False})
        for c in self.candidates:                                                             # in view, close, not detected: count a miss
            b = self.to_base(c["xy"])
            if c["last"] == self.step or not 1.0 < np.linalg.norm(b) < 8.0:
                continue
            uv, z = project(head, np.array([[b[0], b[1], 0.25]]))
            h, w = head["depth"].shape
            if z[0] > 0 and 20 < uv[0, 0] < w - 20 and 20 < uv[0, 1] < h - 20:
                u, v = int(uv[0, 0]), int(uv[0, 1])
                if np.nanmedian(head["depth"][v - 4:v + 5, u - 4:u + 5]) < z[0] - 0.5:         # something stands in front of it: a bush killed the real basket (full_v23_309)
                    continue
                c["misses"] += 1
                c["close_misses"] += int(np.linalg.norm(b) < 4.5)
        for c in self.candidates:                                                             # misses rule out a candidate only while they outnumber the hits: the real basket, killed
            c["missed"] = (c["misses"] >= 8 and c["misses"] > 2 * c["hits"]) or (c["close_misses"] >= 6 and c["close_misses"] > 2 * c["close"])   # by 35 close misses, then
        good = [c for c in self.candidates if c["hits"] >= 2 and not c["dead"] and not c["missed"] and not (c.get("parked") and c["close"] < 2)]   # got 177 close hits (frozen_full_v34_305)
        return max(good, key=lambda c: (c["close"] >= 2, c["hits"] - c["misses"])) if good else None

    def update_trees(self, head, need_canopy=True):                                           # hits: with a canopy above 3 m; near_hits: bark column within 4 m
        for d in detect_trunks(head, need_canopy=need_canopy):
            if not need_canopy and d["range"] > 4.0:
                continue
            w = self.to_world(d["xy"])
            near = [t for t in self.trees if np.linalg.norm(t["xy"] - w) < 0.6]
            if near:
                t = near[0]
                k = min(t["hits"] + t["near_hits"], 8)
                t["xy"] = (k * t["xy"] + w) / (k + 1)
                t["radius"] = (k * t["radius"] + d["radius"]) / (k + 1)
                t["hits" if need_canopy else "near_hits"] += 1
                t["last"] = self.step
            else:
                self.trees.append({"xy": w, "hits": int(need_canopy), "near_hits": int(not need_canopy), "radius": d["radius"], "last": self.step})

    def refine_tree(self, head):
        b = self.to_base(self.tree["xy"])
        near = [d for d in detect_trunks(head, need_canopy=False) if np.linalg.norm(d["xy"] - b) < 0.7]
        if near:
            d = min(near, key=lambda d: np.linalg.norm(d["xy"] - b))
            self.tree["xy"] = 0.5 * self.tree["xy"] + 0.5 * self.to_world(d["xy"])
            self.tree["radius"] = 0.5 * self.tree["radius"] + 0.5 * d["radius"]
            self.log["tree_refined"] = self.step

    def attached(self, q):                                                                    # capsules (a, b, radius) in the base frame that move with the robot
        capsules = self.servo.capsules(q)
        if self.holding:                                                                      # the basket hangs below the hand, its rim centre RIM_RADIUS from the grasp
            centre = self.servo.pose(q)[:3, 3] - RIM_RADIUS * self.grasp["closing"]
            capsules = capsules + [(centre + [0, 0, 0.1], centre - [0, 0, 0.45], 0.30)]
        return capsules

    def attached_mask(self, head, q, size):                                                   # [size, size] pixels showing the robot or what it carries
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

    def after_fail(self):                                                                     # next phase after a missed or blocked egg pick: look again from the same base pose
        return "egg_view" if self.fails < 3 else "done"

    def choose_tree(self, limit=7.0):                                                         # confirmed trunk nearest to the basket; public baskets are <= 5.55 m from a tree
        good = [t for t in self.trees if (t["hits"] >= 2 or t["near_hits"] >= 3) and np.linalg.norm(t["xy"] - self.basket) < limit and not t.get("unreachable")]
        fresh = [t for t in good if self.step - t["last"] < 900]                              # older trunk estimates carry the odometry's heading drift: full_v35_302 set the basket
        good = fresh or good                                                                  # down 2 m from any tree (11 deg heading error over a 7.6 m drive)
        return min(good, key=lambda t: np.linalg.norm(t["xy"] - self.basket)) if good else None

    def pour_pose(self, heading=None):                                                        # base goal and heading that put the trunk TRUNK_GAP + radius ahead of the basket centre,
        centre = self.hand[:2] - RIM_RADIUS * self.grasp["closing"][:2]                       # the basket's centre once set down, base frame
        trunk = centre + np.array([0.20 + SET_GAP + self.tree["radius"], 0.0])
        headings = np.linspace(-math.pi, math.pi, 24, endpoint=False) if heading is None else np.array([heading])
        goals = np.array([self.tree["xy"] - rot2(a) @ trunk for a in headings])
        if heading is None:                                                                   # egg spots on grass: gmzozb trees stand in gravel beds (dev_views_307, dev_spots)
            grass = [min((self.map.grass_fraction(sp) or 0.0) for sp in self.spot_ring(g + rot2(a) @ centre)) for g, a in zip(goals, headings)]
            keep = [i for i, f in enumerate(grass) if f > 0.75]
            self.log["spot_grass"] = [round(f, 2) for f in grass]
            if keep:
                goals, headings = goals[keep], headings[keep]
        path = self.map.plan(self.pose[:2], goals, radius=0.55)                              # the carried basket hangs ~0.9 m ahead of the base centre
        if path is None:                                                                      # cluttered trunk (full_v10_316): accept a tighter footprint
            path = self.map.plan(self.pose[:2], goals, radius=0.40)
        if path is None:
            return None, None, None
        k = int(np.argmin(np.linalg.norm(goals - path[-1], axis=1)))
        return goals[k], float(headings[k]), np.vstack([path, goals[k]])

    def spot_ring(self, basket_w):                                                            # three egg spots round the trunk on the basket side, world frame
        t, r = self.tree["xy"], self.tree["radius"]
        toward = (basket_w - t) / max(np.linalg.norm(basket_w - t), 1e-6)
        return [t + (r + EGG_RING) * (rot2(a) @ toward) for a in np.radians([0, 40, -40])]

    def basket_lifted(self, wrist):                                                           # wicker above 0.5 m in the wrist view: 506 points with the basket hanging on the egg
        hsv = cv2.cvtColor(wrist["rgb"].astype(np.uint8), cv2.COLOR_RGB2HSV)               # (dev25_302), 0-44 in 19 held-egg frames with the basket on the lawn (lift_check)
        wicker = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 27) & (hsv[..., 1] >= 70) & (hsv[..., 1] <= 135) & (hsv[..., 2] >= 45) & (hsv[..., 2] <= 175)
        p, _ = pointcloud(wrist, mask=wicker, max_range=2.0)
        return int((p[:, 2] > 0.5).sum()) >= 200

    def pick_pose(self):                                                                      # base pose with the basket centre at SWEET_PICK and the trunk along TRUNK_DIR from it
        d = self.tree["xy"] - self.basket_w
        heading = wrap(math.atan2(d[1], d[0]) - TRUNK_DIR)
        return self.basket_w - rot2(heading) @ SWEET_PICK, heading

    def around(self, goal, radius=0.62):                                                      # path to goal on an arc round the set-down basket, the side away from the trunk
        s, g, t = self.pose[:2] - self.basket_w, goal - self.basket_w, self.tree["xy"] - self.basket_w
        a0, a1, at = math.atan2(s[1], s[0]), math.atan2(g[1], g[0]), math.atan2(t[1], t[0])
        turn = wrap(a1 - a0)
        if 0 < wrap(at - a0) * np.sign(turn) < abs(turn):                                    # the short way passes the trunk
            turn -= math.copysign(2 * math.pi, turn)
        angles = a0 + np.linspace(0, turn, max(2, int(abs(turn) * radius / 0.1) + 1))
        return np.vstack([self.pose[:2], self.basket_w + radius * np.c_[np.cos(angles), np.sin(angles)], goal])

    # motion -------------------------------------------------------------------------------------
    def drive(self, action, final_heading=None, speed=0.45, lookahead=0.8):                    # follow self.path; True when at its end
        if self.path is None or len(self.path) == 0:
            return False
        if self.backup > 0:                                                                   # recovering from a blocked base: reverse, then replan
            self.backup -= 1
            action[0:2] = -0.4 * self.blocked_dir
            if self.backup == 0:
                self.path, self.planned = None, -1000
            return False
        end = self.path[-1]
        dist = np.linalg.norm(end - self.pose[:2])
        k = int(np.argmin(np.linalg.norm(self.path - self.pose[:2], axis=1)))
        far = [p for p in self.path[k:] if np.linalg.norm(p - self.pose[:2]) > lookahead]
        target = far[0] if far and dist > lookahead else end
        delta = target - self.pose[:2]
        heading = math.atan2(delta[1], delta[0]) if (dist > 1.0 or final_heading is None) else final_heading
        yaw_error = wrap(heading - self.pose[2])
        v = rot2(self.pose[2]).T @ delta
        v = v / max(np.linalg.norm(v), 1e-6) * min(speed, 0.9 * dist + 0.05)
        if abs(yaw_error) > 0.9 and dist > 1.0:
            v *= 0.2
        action[0:2] = np.clip(v / MAX_V, -1, 1)
        turn = 0.3 if self.holding else 0.8                                                   # a 0.8 rad/s turn with the basket in hand flung it to z 1.75 m and threw the eggs out (full_v37_305)
        action[2] = np.clip(1.5 * yaw_error, -turn, turn) / MAX_W
        commanded = np.linalg.norm(action[0:2]) * MAX_V
        stuck = (commanded > 0.2 and np.linalg.norm(self.vel[0:2]) < 0.05) or (abs(action[2]) * MAX_W > 0.3 and abs(self.vel[2]) < 0.05)   # full_v7_303: turn commanded, no rotation
        self.stalled = self.stalled + 1 if stuck else 0
        if self.tilt > 6.0 and np.linalg.norm(action[0:2]) > 0.05:                            # driving onto an edge or a slope: treat it as blocked
            self.stalled = 46
            self.tilt = 0.0
            self.log["tilt_events"] = self.log.get("tilt_events", 0) + 1
        if self.stalled > 45:                                                                 # pushing against something the map does not hold
            d = action[0:2] / np.linalg.norm(action[0:2]) if np.linalg.norm(action[0:2]) > 0.05 else np.array([1.0, 0.0])
            self.map.block(self.to_world(0.55 * d))
            self.blocked_dir, self.backup, self.stalled = d, 30, 0
            self.log["blocked"] = self.log.get("blocked", 0) + 1
        done = dist < 0.05 and (final_heading is None or abs(wrap(final_heading - self.pose[2])) < 0.05)
        return done

    def standoff(self, basket, distance=0.75):                                                # base goal and heading that face the basket
        angles = np.linspace(-math.pi, math.pi, 16, endpoint=False)
        if self.approach_angle is not None:                                                   # keep the chosen side while it stays reachable (full_v3_305 zig-zagged)
            angles = self.approach_angle + np.array([0.0, -0.39, 0.39])
        goals = basket + distance * np.c_[np.cos(angles), np.sin(angles)]
        order = np.argsort(np.linalg.norm(goals - self.pose[:2], axis=1))[:8]
        path = self.map.plan(self.pose[:2], goals[order])
        if path is None:
            self.approach_angle = None
            return None, None, None
        k = order[np.argmin(np.linalg.norm(goals[order] - path[-1], axis=1))]
        self.approach_angle = angles[k]
        g = goals[k]
        path = np.vstack([path, g])
        d = basket - g
        return g, math.atan2(d[1], d[0]), path

    # main loop ----------------------------------------------------------------------------------
    def forward(self, obs):
        data = allowed(obs)
        s = data["robot_r1::proprio"].reshape(-1)                                          # [61]
        self.step += 1
        self.age += 1
        self.pose[:2] += rot2(self.pose[2]) @ s[0:2] * DT                                    # base_qvel is in the base frame (robots/robot.py:1636)
        w_cmd = 0.0 if self.action is None else self.action[2] * MAX_W
        w = s[2] if (abs(w_cmd) < 0.05 or w_cmd * s[2] <= 0) else math.copysign(min(abs(s[2]), abs(w_cmd)), s[2])   # proprio reads in-place turns 5-15% high, the command <= 4% (odom_cal2_301); blocked turns read low
        self.pose[2] = wrap(self.pose[2] + w * DT)
        self.vel = s[0:3]
        cams = cameras(data)
        head = cams["head"]
        if self.step % 10 == 0:                                                               # base tilt from the visible ground plane (full_v13_303 rolled 29 deg on a garden edge)
            self.tilt = ground_tilt(head) or 0.0
            self.log["tilt"] = round(self.tilt, 1)
        if self.step % 5 == 1 and self.tilt < 4.0:                                            # visual correction of the proprio prediction (planar: not while tilted)
            small = resized(head, 480)
            small["moving_object_mask"] = self.attached_mask(head, self.servo.q(s), 480)    # arms and the carried basket move with the robot
            self.pose = self.vo.update(small, self.step, self.pose)
            self.pose[2] = wrap(self.pose[2])
            self.log["vo"] = self.vo.stats.get("reason")
        if self.action is None:
            self.action = np.r_[np.zeros(3), s[53:57], s[3:10], 1.0, s[28:35], 1.0]       # [23] trunk, left arm, left grip, right arm, right grip
        action = self.action.copy()
        action[0:3] = 0.0
        q = self.servo.q(s)

        if self.step % 6 == 1:
            self.map.update(head, self.pose, self.attached(q))
        if self.step % 3 == 1 and self.phase in ("scan", "explore", "approach"):
            self.target = self.update_candidates(head)
            self.basket = None if self.target is None else self.target["xy"].copy()
        if self.step % 9 == 4 and self.phase in ("scan", "explore", "approach", "goto_tree"):
            self.update_trees(head)
        if self.step % 9 == 1 and self.phase in ("approach", "at_basket", "goto_tree"):       # close trunks: the canopy is above the head's view (full_v8_312)
            self.update_trees(head, need_canopy=False)
        if self.step % 9 == 7 and self.phase == "goto_tree" and self.tree is not None and np.linalg.norm(self.to_base(self.tree["xy"])) < 3.5:
            self.refine_tree(head)                                                            # close trunk: the canopy is above the head's view

        if self.phase == "scan":                                                              # turn once in place
            action[2] = 0.8 / MAX_W
            self.turned += abs(s[2]) * DT
            if self.basket is not None:
                self.change("approach")
            elif self.turned > 2 * math.pi:
                self.change("explore")
        elif self.phase == "explore":
            if self.basket is not None:
                self.change("approach")
            else:
                if self.path is None or self.step - self.planned > 90:
                    strong = [c for c in self.candidates if c.get("parked") and not c["dead"] and not c.get("missed") and c["hits"] >= 20 and c["misses"] <= 2]
                    lure = self.toward_cell(max(strong, key=lambda c: c["hits"])["xy"]) if strong else None   # v3.9.3 305: the real basket parked at 91 hits / 0 misses,
                    if lure is not None:                                                  # unplannable from 10 m; frontiers elsewhere took the rest of the run
                        self.goal, gain = lure, -2.0
                        self.log["lure"] = self.log.get("lure", 0) + 1
                    else:
                        self.goal, gain = self.map.viewpoint(self.pose[:2], self.visited, keep=self.goal)
                    if self.goal is None:                                                     # no frontier left (full_v4_303 turned in place 15 m from the basket)
                        self.goal, gain = self.map.viewpoint(self.pose[:2], [], min_frontier=3)
                    if self.goal is None:
                        self.goal, gain = self.map.far_reachable(self.pose[:2], self.visited), -1.0
                    self.path = None if self.goal is None else self.map.plan(self.pose[:2], self.goal[None])
                    self.planned = self.step
                    self.log["viewpoint_gain"] = gain
                if self.path is None:
                    action[2] = 0.5 / MAX_W                                                   # nothing to go to: keep turning
                elif self.drive(action) or (self.goal is not None and np.linalg.norm(self.goal - self.pose[:2]) < 0.4):
                    self.visited.append(self.pose[:2].copy())
                    self.change("scan")
        elif self.phase == "approach":
            if self.target is None:                                                           # candidate rejected on the way
                self.change("explore")
            else:
                if self.path is None or (self.step - self.planned > 60 and not self.ring):     # a ring leg runs to its end (or a blocked-base backup)
                    self.goal, self.heading, self.path = self.standoff(self.basket)
                    self.ring = False
                    near = np.linalg.norm(self.basket - self.pose[:2])
                    if self.path is None and self.target["close"] >= 2 and near < 4.0:      # confirmed and close: straight at it (full_v10_315: drift smeared the map round the basket)
                        away = (self.pose[:2] - self.basket) / max(near, 1e-6)
                        self.goal = self.basket + 0.75 * away
                        self.heading = math.atan2(-away[1], -away[0])
                        self.path = np.array([self.pose[:2], self.goal])
                        self.log["direct_approach"] = self.step
                    elif self.path is None and near > 3.0 and not self.target.get("ringed"):  # far, no standoff path: first get within 2.5 m, where it can be confirmed (full_v31_305
                        a = np.linspace(-math.pi, math.pi, 16, endpoint=False)                # parked the real basket seen from 9 m and never came back within 4.5 m)
                        path = self.map.plan(self.pose[:2], self.basket + 2.5 * np.c_[np.cos(a), np.sin(a)])
                        if path is not None:
                            d = self.basket - path[-1]
                            self.path, self.goal, self.heading, self.ring = path, path[-1], math.atan2(d[1], d[0]), True
                            self.target["ringed"] = True                                      # one leg, counted at its start: an unreachable leg looped 7800 steps (frozen_full_v34_306)
                            self.log["ring_approach"] = self.log.get("ring_approach", 0) + 1
                    self.planned = self.step
                if self.age > 1200 and self.target["close"] < 2:                              # no approach finishes an unconfirmed candidate this late (frozen_full_v34_306: 7800 steps)
                    self.target["parked"] = True
                    self.change("explore")
                elif self.path is None:
                    action[2] = 0.3 / MAX_W
                    self.stuck += 1
                    if self.stuck > 90 and self.target["close"] < 2:                         # unreachable and unconfirmed: park it until seen from within 4.5 m (full_v2_303 spun 9000 steps;
                        self.target["parked"] = True                                          # killing it lost the real basket in full_v26_305 / 306, then seen close 73 / 90 times)
                        self.change("explore")
                elif self.drive(action, self.heading, speed=0.45 if np.linalg.norm(self.basket - self.pose[:2]) > 2.5 else 0.25):
                    if self.ring:                                                             # 2.5 m from it: look, then plan the standoff or go straight at it; one leg per
                        self.path, self.ring = None, False                                    # candidate (full_v33_305 spent 885 steps of legs on one false candidate)
                    elif self.target["close"] >= 2:
                        self.change("at_basket")
                    else:                                                                     # standing next to it but never confirmed up close
                        self.target["dead"] = True
                        self.change("explore")
        elif self.phase == "at_basket":                                                       # fit the rim from a still base
            if self.age > 10 and self.age % 3 == 0:
                rim = basket_rim(head, self.to_base(self.basket))
                if rim is not None:
                    self.rims.append(rim)
            if len(self.rims) >= 5 and np.median([r["top"] for r in self.rims]) > 0.50:          # a trunk's foot: rim tops 0.38-0.41 on baskets, 0.58-0.60 on trunks (full_v14_310, v18_307, v20_305, v21_304)
                self.target["dead"] = True
                self.log["rim_rejected"] = self.log.get("rim_rejected", 0) + 1
                self.rims = []
                self.change("explore")
            elif len(self.rims) >= 5:
                c = np.median([r["xy"] for r in self.rims], axis=0)
                top = float(np.median([r["top"] for r in self.rims]))
                self.basket = self.to_world(c)
                radial = rot2(GRASP_ANGLE) @ (-c / np.linalg.norm(c))
                closing = np.r_[radial, 0.0]
                self.grasp = {"point": np.r_[c + RIM_RADIUS * radial, top - RIM_DEPTH], "rotation": np.column_stack([np.cross(closing, UP), closing, UP]),
                              "closing": closing, "top": top, "centre": c}
                self.log["grasp"] = {k: np.round(v, 3).tolist() for k, v in self.grasp.items()}
                self.rims = []
                self.change("pregrasp")
            elif self.age > 120:                                                              # rim not seen: re-approach once, then give up on this candidate
                self.target["rim_fails"] = self.target.get("rim_fails", 0) + 1
                if self.target["rim_fails"] >= 2:
                    self.target["dead"] = True
                    self.change("explore")
                else:
                    self.change("approach")
        elif self.phase in ("pregrasp", "descend", "close", "lift"):
            g = self.grasp
            goal = g["point"] + {"pregrasp": [0, 0, 0.18], "descend": [0, 0, 0], "close": [0, 0, 0], "lift": [0, 0, 0.25]}[self.phase]
            err = self.servo.step(action, q, goal, g["rotation"], plan=(self.phase == "pregrasp" and self.age == 1))
            action[GRIP] = 1.0 if self.phase in ("pregrasp", "descend") else -1.0
            self.log["arm_error"] = [round(e, 4) for e in err]
            if self.phase == "pregrasp" and ((err[0] < 0.012 and err[1] < 0.08 and self.age > 20) or self.age > 360):
                self.change("descend")
            elif self.phase == "descend" and ((err[0] < 0.012 and self.age > 10) or self.age > 240):
                self.change("close")
            elif self.phase == "close" and self.age > 45:
                self.change("lift")
            elif self.phase == "lift" and ((err[0] < 0.02 and self.age > 20) or self.age > 240):
                self.holding = float(s[FINGERS].sum()) > 0.004                               # fingers stopped by the rim
                self.log["fingers_at_lift"] = s[FINGERS].round(4).tolist()
                self.attempts += 1
                if self.holding:
                    self.hand = self.servo.pose(q)[:3, 3].copy()
                    self.change("carry_up")
                elif self.attempts < 3:
                    self.change("release")
                else:
                    self.change("done")
        elif self.phase == "release":                                                         # open, back off upward, measure again
            action[GRIP] = 1.0
            self.servo.step(action, q, self.grasp["point"] + [0, 0, 0.25], self.grasp["rotation"])
            if self.age > 60:
                self.change("at_basket")
        elif self.phase == "carry_up":
            g = self.grasp
            err = self.servo.step(action, q, np.r_[g["point"][:2], CARRY_Z], g["rotation"])
            action[GRIP] = -1.0
            if (err[0] < 0.02 and self.age > 20) or self.age > 240:
                self.hand = self.servo.pose(q)[:3, 3].copy()
                self.change("goto_tree")
        elif self.phase == "goto_tree":
            action[GRIP] = -1.0
            if self.tree is None:
                self.tree = self.choose_tree(7.0 if self.age < 700 else 50.0)                 # a full turn at 0.3 rad/s is ~630 steps
            if self.tree is None:
                action[2] = 0.3 / MAX_W                                                       # look around for a trunk
            else:
                if self.path is None or self.step - self.planned > 90:                        # heading chosen once, goal follows the refined trunk estimate
                    self.goal, heading, self.path = self.pour_pose(self.heading if self.age > 1 else None)
                    self.heading = heading if heading is not None else self.heading
                    self.planned = self.step
                if self.path is None:
                    action[2] = 0.3 / MAX_W
                    self.stuck += 1
                    if self.stuck > 120:                                                      # no pour pose around this trunk: try the next tree
                        self.tree["unreachable"] = True
                        self.log["trees_unreachable"] = self.log.get("trees_unreachable", 0) + 1
                        self.tree, self.heading, self.stuck = None, None, 0
                        if not self.choose_tree(50.0):                                        # every tree excluded: allow them again
                            for t in self.trees:
                                t["unreachable"] = False
                elif self.drive(action, self.heading, speed=0.35):
                    self.change("set_down")
        elif self.phase == "pour":                                                            # lower, tip the mouth away from the hand, shake
            g = self.grasp
            tilt = max(0, self.age - 60)
            full = int(POUR_DEG / 0.9)
            angle = math.radians(min(POUR_DEG, 0.9 * tilt))
            replan = tilt == full + 1 and self.log.get("pour_error", 0.0) > 0.35                # incremental IK stalled short of the full tilt (full_v12_312, pour5_302)
            if replan:
                self.servo.target, self.servo.synchronized = None, True
                self.log["pour_replanned"] = self.step
            if tilt > full and self.servo.target is None:                                     # shake once the planned move (if any) is done
                angle -= math.radians(SHAKE_DEG) * abs(math.sin((tilt - full) / 8.0))
            rot = Rotation.from_rotvec(-np.cross(UP, g["closing"]) * angle).as_matrix() @ g["rotation"]
            err = self.servo.step(action, q, np.r_[self.hand[:2], POUR_Z], rot, plan=replan)
            self.log["pour_error"] = round(err[1], 3)
            if self.servo.target is None:
                self.servo.synchronized = False
            action[GRIP] = -1.0
            if 60 < tilt <= full and err[1] > 0.8 * angle and self.backoffs < 2:              # the hand is not turning at all: basket jammed on the trunk (full_v18_306)
                self.backoffs += 1
                self.log["pour_backoffs"] = self.backoffs
                self.change("pour_backoff")
            elif tilt > full + 300:
                self.change("done")
        elif self.phase == "pour_backoff":                                                    # untilt, reverse 0.2 m, pour again
            self.servo.step(action, q, np.r_[self.hand[:2], POUR_Z], self.grasp["rotation"])
            action[GRIP] = -1.0
            if 45 < self.age <= 75:
                action[0] = -0.27 / MAX_V                                                     # 0.27 m/s for 1 s
            if self.age > 90:
                self.change("pour")
        elif self.phase == "set_down":                                                        # lower the basket onto the lawn
            g = self.grasp
            err = self.servo.step(action, q, np.r_[self.hand[:2], g["point"][2] + 0.02], g["rotation"])
            action[GRIP] = -1.0
            if (err[0] < 0.02 and self.age > 30) or self.age > 240:
                self.basket_w = self.to_world(self.servo.pose(q)[:2, 3] - RIM_RADIUS * g["closing"][:2])
                self.spots_w = self.spot_ring(self.basket_w)
                self.log["spots_w"] = [np.round(x, 2).tolist() for x in self.spots_w]
                self.change("let_go")
        elif self.phase == "let_go":                                                          # release the basket (demo: it stands alone), left arm up and aside
            g = self.grasp
            action[GRIP] = 1.0
            self.holding = False
            if self.age <= 40:
                pass
            elif self.age <= 160:
                self.servo.step(action, q, np.r_[self.hand[:2], g["point"][2] + 0.30], g["rotation"])
            else:
                self.servo.step(action, q, np.array([0.30, 0.45, 1.0]), g["rotation"], plan=self.age == 161)
            if self.age > 360:
                self.change("go_basket")
        elif self.phase == "go_basket":                                                       # one base move; every egg is then picked and placed from this pose
            action[GRIP] = -1.0 if self.holding else 1.0
            if self.path is None:
                self.goal, self.heading = self.pick_pose()
                self.path = self.around(self.goal)
                self.log["pick_pose"] = np.round(np.r_[self.goal, self.heading], 2).tolist()
            if self.drive(action, self.heading, speed=0.2, lookahead=0.3) or self.age > 400:
                self.change("egg_view")
        elif self.phase == "egg_view":                                                        # right wrist above the basket, looking down into it (demo episode 60000)
            centre = self.to_base(self.basket_w)
            self.rservo.step(action, q, np.r_[centre, 0.80], VIEW_ROT, plan=self.age == 1 and self.placed == 0 and self.fails == 0)   # planned only from the hanging arm
            action[RGRIP] = 1.0
            uv, depth = project(cams["right_wrist"], np.array([[centre[0], centre[1], 0.25]]))
            width = cams["right_wrist"]["depth"].shape[1]
            down = cams["right_wrist"]["R"][2, 2] < -0.95                                     # camera within 18 deg of straight down: pitched views -0.98..-1.0, a view through the wicker -0.945 (dev22_301)
            if down and depth[0] > 0 and np.linalg.norm(uv[0] - width / 2) < 0.3 * width and self.age > 20:   # basket centre in view: the arm may settle 5-12 cm off
                                                                                              # the view pose and still see the whole basket (view_proj: dev28_306 0.06 off-centre, egg found)
                self.view_seen = True
                rim = basket_rim(cams["right_wrist"], centre)                                 # the basket may have been dragged: refit its rim from above (rim_refit_test: ~1 cm on
                refit = rim is not None and 0.15 < rim["radius"] < 0.23 and 0.36 < rim["top"] < 0.44 and np.linalg.norm(rim["xy"] - centre) < 0.3   # 29 views; bad fits had
                if refit:                                                                     # radius 0.04-0.12 or top 0.48-0.58)
                    self.basket_w, centre = self.to_world(rim["xy"]), rim["xy"]
                    self.log["basket_refit"] = np.round(rim["xy"], 3).tolist()
                t = top_egg(cams["right_wrist"], centre, radius=0.15 if refit else 0.25)      # egg centres <= 0.134 m from the basket centre (egg_radius_check, 34k rows), rim 0.19 m:
                                                                                              # tilted-rim highlights at z 0.31 were taken for eggs (frozen_full_v34_301)
                if t is not None and t[5] > 4.5:                                              # a streak, not an egg: eggs 1.1-3.8, wicker highlights 5.3-19.3
                    t = None                                                                  # (frozen_full_v33_302, full_v28_302 picked the basket up)
                if t is not None:
                    if not self.views:
                        np.savez_compressed(self.output / f"view_{self.step:05d}.npz", **data)
                    self.views.append(t)
            if len(self.views) >= 3:
                t = np.median(self.views, axis=0)
                self.views = []
                radial = t[:2] - centre
                if t[5] > 1.3:                                                                # lying egg (0.15 m long, fingers open 0.127 m): close across it (dev25_301 missed one)
                    closing = np.r_[-t[4], t[3], 0.0] / max(np.linalg.norm(t[3:5]), 1e-6)
                elif np.linalg.norm(radial) > 0.03:
                    closing = np.r_[-radial[1], radial[0], 0.0] / np.linalg.norm(radial)
                else:
                    closing = np.array([0.0, 1.0, 0.0])
                self.egg = {"top": t[:3], "rotation": np.column_stack([np.cross(closing, UP), closing, UP]), "from": "basket"}
                self.log["egg_target"] = np.round(t, 3).tolist()
                self.change("egg_pre")
            elif self.age > 200:                                                              # nothing visible in the basket
                self.views = []
                if not self.view_seen and self.reposition < 2:                                # the basket never came into view (final v39 301: 1.2 m off the hand): move the base
                    self.reposition += 1
                    self.log["reposition"] = self.reposition
                    self.change("go_basket")
                else:
                    self.log["view_empty"] = self.step
                    self.change("done")
        elif self.phase in ("egg_pre", "egg_descend", "egg_close", "egg_lift", "egg_carry", "egg_lower", "egg_release", "egg_retract", "egg_return", "egg_putback", "egg_letgo"):
            e = self.egg
            spot = self.to_base(self.spots_w[min(self.placed, 2)])
            goal = {"egg_pre": np.r_[e["top"][:2], 0.60], "egg_descend": np.r_[e["top"][:2], e["top"][2] - EGG_DEPTH],
                    "egg_close": np.r_[e["top"][:2], e["top"][2] - EGG_DEPTH], "egg_lift": np.r_[e["top"][:2], 0.70], "egg_carry": np.r_[spot, PLACE_Z],
                    "egg_lower": np.r_[spot, RELEASE_Z], "egg_release": np.r_[spot, RELEASE_Z], "egg_retract": np.r_[spot, PLACE_Z],
                    "egg_return": np.r_[self.to_base(self.basket_w), PLACE_Z],
                    "egg_putback": np.r_[e["top"][:2], e["top"][2] - EGG_DEPTH + 0.02], "egg_letgo": np.r_[e["top"][:2], e["top"][2] - EGG_DEPTH + 0.02]}[self.phase]
            err = self.rservo.step(action, q, goal, e["rotation"], plan=False)   # straight moves only: a planned move to the egg ran below the rim and tipped the basket (full_v23_302)
            action[GRIP] = -1.0 if self.holding else 1.0
            action[RGRIP] = -1.0 if self.phase in ("egg_close", "egg_lift", "egg_carry", "egg_lower", "egg_putback") else 1.0
            settled = self.rservo.target is None and err[0] < 0.012 and self.age > 15
            z = self.rservo.pose(q)[2, 3]
            if self.phase == "egg_pre" and (settled or self.age > 300):
                self.change("egg_descend")
            elif self.phase == "egg_descend" and (settled or self.age > 200):
                if err[0] > 0.06:                                                             # blocked above the egg (egg_d035_301 third egg)
                    self.fails += 1
                    self.log["egg_blocked"] = self.log.get("egg_blocked", 0) + 1
                    self.change(self.after_fail())
                else:
                    self.change("egg_close")
            elif self.phase == "egg_close" and self.age > 45:
                self.change("egg_lift")
            elif self.phase == "egg_lift" and (settled or self.age > 220):                    # straight up: a planned joint move dragged the basket 0.3 m (dev21_301)
                if float(s[RFINGERS].sum()) > 0.004 and self.basket_lifted(cams["right_wrist"]):  # a wedged egg brought the basket up (dev25_302): put it back, let go, look again
                    self.fails += 1
                    self.log["basket_lifted"] = self.log.get("basket_lifted", 0) + 1
                    self.change("egg_putback")
                elif float(s[RFINGERS].sum()) > 0.004:                                        # fingers stopped by the egg (they stay near 0.05/0.05)
                    self.change("egg_carry")
                else:
                    self.fails += 1
                    self.log["egg_missed"] = self.log.get("egg_missed", 0) + 1
                    self.change(self.after_fail())
            elif self.phase == "egg_putback" and (settled or self.age > 220):
                self.change("egg_letgo")
            elif self.phase == "egg_letgo" and self.age > 30:
                self.change(self.after_fail())
            elif self.phase == "egg_carry" and (settled or self.age > 300):
                self.change("egg_lower")
            elif self.phase == "egg_lower" and (settled or z < RELEASE_Z + 0.02 or self.age > 200):
                self.change("egg_release")
            elif self.phase == "egg_release" and self.age > 30:
                self.placed += 1
                self.fails = 0
                self.log["eggs_placed"] = self.placed
                self.change("egg_retract")
            elif self.phase == "egg_retract" and (settled or z > PLACE_Z - 0.05 or self.age > 150):
                self.change("done" if self.placed >= 3 else "egg_return")
            elif self.phase == "egg_return" and (settled or err[0] < 0.05 or self.age > 150):   # back over the basket the way the egg came: the servo stalled
                self.change("egg_view")                                                       # 0.18 m short going straight from the spot to the view (dev27_306)
        elif self.phase == "done":
            action[GRIP] = -1.0 if self.holding else 1.0

        if self.phase == "go_basket" or self.phase.startswith("egg_"):                       # the parked left hand followed the torso into the basket
            action[7:14] += np.clip(LEFT_TUCK - action[7:14], -0.03, 0.03)                    # (link_clearance: frozen_full_v34_301 at the rim, full_v34_309 3-7 cm)
        if self.step % 15 == 0:
            row = {"step": self.step, "phase": self.phase, "pose": self.pose.round(3).tolist(), "basket": None if self.basket is None else self.basket.round(3).tolist(),
                   "candidates": [{"xy": c["xy"].round(2).tolist(), "hits": c["hits"], "misses": c["misses"], "close": c["close"], "close_misses": c["close_misses"], "dead": c["dead"], "missed": c.get("missed", False), "parked": c.get("parked", False)}
                                  for c in sorted(self.candidates, key=lambda c: -c["hits"])[:6]],
                   "goal": None if self.goal is None else np.round(self.goal, 2).tolist(), "path": None if self.path is None else len(self.path),
                   "action": action[:3].round(3).tolist(), "vel": s[0:3].round(3).tolist(), "fingers": s[FINGERS].round(4).tolist(), "holding": self.holding,
                   "trees": [{"xy": t["xy"].round(2).tolist(), "hits": t["hits"], "near": t["near_hits"], "r": round(t["radius"], 2)} for t in sorted(self.trees, key=lambda t: -t["hits"] - t["near_hits"])[:6]],
                   "tree": None if self.tree is None else self.tree["xy"].round(2).tolist(), "rhand": self.rservo.pose(q)[:3, 3].round(3).tolist(), "rfingers": s[RFINGERS].round(4).tolist(), **self.log}
            with open(self.output / "trace.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        if self.step % 300 == 0:
            np.savez_compressed(self.output / f"step_{self.step:05d}.npz", **data)
        if self.step % 900 == 0:                                                              # map snapshots for offline diagnosis
            np.savez_compressed(self.output / f"map_{self.step:05d}.npz", odds=self.map.odds, searched=self.map.searched, ground=self.map.ground,
                                pose=self.pose, visited=np.array(self.visited), goal=np.array([np.nan, np.nan]) if self.goal is None else self.goal)
        assert np.isfinite(action).all()
        self.action = action.copy()
        return torch.as_tensor(action[None, :], dtype=torch.float32)
