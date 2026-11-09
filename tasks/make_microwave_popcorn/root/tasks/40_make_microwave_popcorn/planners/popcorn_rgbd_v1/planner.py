"""Task 40 make_microwave_popcorn, popcorn_rgbd_v1: hand-written RGB-D planner, 23D action, obs = the 8 whitelisted keys only.

Grown from popcorn_rgbd_v0/planner_pc.py (odometry, head depth, A* on a depth grid, damped least squares IK of torso + right arm,
cloud templates kept with their checksum). Perception: port colour masks (body, door, button, bag) on the aligned head cloud.
Static priors are object-model geometry only (hinge, pinch points, button, cavity in the microwave front-face frame), no world
coordinates, no instance branching. Phases (log: phase, stage, reason, counters): search_mw, goto_mw, open_door, search_bag,
goto_bag, grasp_bag, goto_mw_loaded, place_bag, close_door, press, wait, done. Each phase checks its result from the observation
and on failure retries or backs off (FALLBACK), never advances on a timeout: door open = door pixels gone from the face plane;
bag held = finger width; bag inside = bag pixels in the cavity box; door closed = door pixels back; press = waypoints reached.
The episode ends on success, so still being called after the wait means not cooked: the press is repeated.
"""
import heapq
import math

import numpy as np
from scipy.ndimage import label

HEAD_FOCAL = 0.425
WRIST_FOCAL = 0.80971
HEAD_W = 720
WRIST_W = 480
DT = 0.03333333333333333
SCALE = (0.75, 0.75, 1.0)
VMAX = 0.5
YAW_MAX = 0.8
GRID_N = 150
GRID_RES = 0.08
INFLATE = 3
LOOKAHEAD = 0.45
CYAN_MIN = 140              # microwave body: R above this and G, B 10 above R (windows are bluer, R about 86)
MW_MIN_PX = 8              # body pixels (head, stride 3) to count a microwave sighting (about 50 at 3.7 m)
BODY_Z = (0.85, 1.25)      # microwave body height band (on the counter)
NORMAL_AGREE = 0.95        # door plane normal must agree with the estimate (an open door is turned)
DOOR_MIN_PX = 60           # door pixels near the body to fit the front plane
BAG_MIN_PX = 12            # bag pixels in the counter height band
BAG_Z = (0.91, 1.30)       # height band of the bag on the counter (m, base frame)
BAG_DN = 0.50              # bag within this of the counter line (0.1 m behind the face)
BAG_SIDE = (0.5, 2.6)      # bag lateral distance from the microwave along the counter
BAG_Q = 0.05               # quantile trimmed from each side of the bag pixel box
BAG_HALF_H = 0.075         # bag half height: centre = visible top minus this
MIN_HITS = 3               # sightings before a search phase ends
WP_BUDGET = 160            # steps per waypoint before the waypoint counts as failed
SOFT_STEPS = 30            # steps per sweep point (door arc, door push) before moving on
# microwave model geometry in the front-face frame: dn along the outward normal, dl along l = normal turned +90 deg (robot's
# right when facing the microwave), dz from the face centre height; from the scene asset (dev geometry dump, DEV_LOG.md)
DOOR_DL = -0.061           # door centre (the door spans dl -0.246 .. 0.124, hinge on the -l side)
DOOR_FRONT = 0.024         # face reference (handle and panel front) ahead of the door glass plane
HINGE = (-0.028, -0.247)   # hinge axis (dn, dl)
DOOR_T = 0.027             # door front surface ahead of the hinge axis
HANDLE_DLS = (0.11, 0.094, 0.12)  # pinch points on the door front near the free edge, one per open attempt
BUTTON = (-0.015, 0.191, -0.092)  # start button, used when the red button is not seen
CAVITY = (-0.18, -0.057, 0.0)     # cavity centre; the bag centre must end inside +-CAV_HALF
CAV_HALF = (0.13, 0.15, 0.08)
FACE_ZC_TOP = 0.133        # face centre below the top of the body
THETA_OPEN = 1.6           # door angle the arc pulls to (rad, beyond the 1.571 stop: the door ends fully open)
STAND_OPEN = (0.86, 0.0)   # base (dn, dl) for opening, from the demos
STAND_PLACE = 0.90         # insert point this far ahead of the base, 0.25 m to its right
CARRY = (0.55, -0.25, 1.15)  # fingertip in the base frame while the bag is carried
TRANSIT = 0.45             # carry to a pose this much further out first, then straight in: a sideways pass shuts the open door
STAND_CLOSE = (0.75, -0.45)  # base for pushing the door shut from its outer side
STAND_PRESS = (0.90, -0.25)
GRASP_AT = (0.75, -0.25)   # bag centre ahead / left of the base, the base faces the pinched bag wall (right-arm reach)
TURN_PREF = 0.5            # of the two bag walls, pinch the one whose facing turn is nearest this (left, away from the counter corner)
BAG_TOL = 0.12             # arrival tolerance at the bag (m)
STALL_OK = 0.12            # after one stuck back-off, a goal this close counts as reached
GRASP_OFF = (0.01, 0.06)   # pinch point: inside the near bag wall (along its normal) and above the bag centre
GRASP_AXIS = (0.0, -1.0)   # finger axis straight down: a turn about it only yaws the bag (a tilted axis tilts it)
INSERT_DL = -0.057         # centre the bag in the cavity, clear of the right control panel
INSERT_DN = -0.11          # bag centre target of the shallow insert (just inside the cavity box)
INSERT_DZS = (0.01, 0.025, -0.005)  # bag centre height above the face centre, one per place attempt (opening 0.92-1.13 m)
INSERT_OK_DN = 0.03        # a blocked insert still releases if the bag centre is past this (it rests on the cavity lip)
LIFT_AFTER = 0.06          # lift the open hand out of the bag rim before backing away
PUSH_TO = -0.10            # closed fingertips push the bag front to this depth
PLACE_AXIS = GRASP_AXIS    # same axis as the grasp: the carry offset stays valid (AG holds the bag rigidly)
PUSH_RHO = 0.25            # door push point distance from the hinge
HOLD_WIDTH = (0.01, 0.08)  # finger width sum while a bag is held (empty closed hand reads 0.0)
DOOR_CLOSED_FRAC = 0.85    # door pixels on the face vs the closed reference: closed above this fraction (0.3 rad ajar reads 0.6)
BTN_GATE = 0.15            # red pixels this close to the button prior are the button
BTN_RELOC = 0.2            # per sighting, share of the button offset moved into the face centre (late phases)
DOOR_OPEN_FRAC = 0.25      # open below this fraction (fully open still shows 0.05-0.17, ajar at 0.2 rad about 0.35)
CAV_BAG_PX = 15            # bag pixels inside the cavity box that mean the bag is in
PRESS_DEPTH = 0.005        # press this far behind the button surface
PRESS_HOLD = 12            # steps held at the button (the toggle needs 5)
PRESS_TRIALS = ((0, 0), (0.012, 0), (-0.012, 0), (0, 0.012), (0, -0.012))  # (dl, dz) offsets, one per press attempt
WAIT_STEPS = 110           # cooking needs about 77 steps; still running after this means not cooked
CHECK_SETTLE = 25          # steps the arm returns to the start pose before a door, cavity or press check
MAX_TRIES = 3              # attempts of a manipulation before backing off to the previous phase
BUDGET = (450, 400, 1100, 250, 300, 650, 300, 600, 450, 230, WAIT_STEPS + 10)  # per phase, sum 4850 < horizon 4856
FALLBACK = (0, 0, 1, 3, 3, 3, 6, 6, 8, 9, 9)  # phase to back off to when the budget runs out (same phase = retry)
NAMES = ("search_mw", "goto_mw", "open_door", "search_bag", "goto_bag", "grasp_bag", "goto_mw_loaded",
         "place_bag", "close_door", "press", "wait", "done")
# (offset in parent frame, rotation axis) for torso 1-4 then right arm 1-7, from seed/r1pro.urdf.
CHAIN = (
    ((-0.079032, 0.0, 0.34265), (0.0, 1.0, 0.0)), ((0.0, 0.0, 0.4), (0.0, 1.0, 0.0)), ((0.0, 0.0001005, 0.3), (0.0, -1.0, 0.0)),
    ((0.0, -0.00010015, 0.09962), (0.0, 0.0, 1.0)), ((-0.00048706, -0.170736, 0.30302), (0.0, 1.0, 0.0)), ((0.023988, -0.081265, 0.0),
    (1.0, 0.0, 0.0)), ((-0.023988, 0.0, -0.1155), (0.0, 0.0, 1.0)), ((0.0, 0.035058, -0.1945), (0.0, 1.0, 0.0)), ((0.0, -0.035058, -0.095),
    (0.0, 0.0, 1.0)), ((0.0, 0.028001, -0.16305), (0.0, 1.0, 0.0)), ((0.0295, -0.028001, 0.0), (1.0, 0.0, 0.0)),
)
TCP = (-0.0295, 0.0, -0.22065)
LIMITS = (
    (-1.1345, 1.8326), (-2.7925, 2.5307), (-1.8326, 1.5708), (-3.0543, 3.0543), (-4.4506, 1.309), (-3.1416, 0.1745), (-2.3562, 2.3562),
    (-2.0944, 0.349), (-2.3562, 2.3562), (-1.0472, 1.0472), (-1.5708, 1.5708),
)
READY = (0.40, -0.25, 1.10)  # fingertip point in the base frame every arm task starts from (above the counter)
HOME_RATE = 0.03           # rad per step when torso and arm return to the start pose between arm tasks
REST = (0.80, -1.06, -0.30, 0.0, -1.4, -0.35, -0.2, -0.7, 0.0, 0.3, 0.0)  # torso as the demo operators hold it for arm work


def quat_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of an xyzw quaternion."""
    x, y, z, w = q / max(float(np.linalg.norm(q)), 1e-9)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def rot_axis(axis: tuple, th: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    a = np.asarray(axis, dtype=np.float64)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)

def fk(q: np.ndarray) -> tuple:
    """Fingertip point, finger axis and finger closing direction (unit, base frame) of the right arm, 11 joints torso+arm."""
    rot, pos = np.eye(3), np.zeros(3)
    for (off, ax), th in zip(CHAIN, q):
        pos = pos + rot @ np.asarray(off)
        rot = rot @ rot_axis(ax, float(th))
    return pos + rot @ np.asarray(TCP), -rot[:, 2], rot[:, 1]

def ik_step(q: np.ndarray, target: np.ndarray, axis: object, side: object = None, lock_torso: bool = True) -> np.ndarray:
    """One damped least squares step on position, finger axis and (optionally) the finger closing direction `side`, with a
    pull toward REST in the null space. Without `side` the hand may roll about its axis (a held bag then turns with it)."""
    def res(v: np.ndarray) -> np.ndarray:
        p, a, sd = fk(v)
        r = np.r_[p - target, 0.15 * (a - axis)] if axis is not None else p - target
        return np.r_[r, 0.15 * (sd - side)] if side is not None and axis is not None else r
    raw = res(q)
    jac = np.stack([(res(q + np.eye(11)[i] * 1e-4) - raw) / 1e-4 for i in range(11)], axis=1)
    if side is not None and lock_torso:  # keep the proven posture unless a grasp is unreachable
        jac[:, 3] = 0.0
    r0 = raw * min(1.0, 0.03 / max(float(np.linalg.norm(raw[:3])), 1e-9))
    pinv = jac.T @ np.linalg.inv(jac @ jac.T + 0.004 * np.eye(len(r0)))
    null = (np.asarray(REST) - q) * 0.05
    null[3] = 0.0
    dq = -pinv @ r0 + (np.eye(11) - pinv @ jac) @ null
    dq = dq * min(1.0, 0.03 / max(float(np.abs(dq).max()), 1e-9))
    lim = np.asarray(LIMITS)
    return np.clip(q + dq, lim[:, 0] + 1e-4, lim[:, 1] - 1e-4)

def cloud(depth: np.ndarray, pose: np.ndarray, focal: float, width: int, stride: int) -> np.ndarray:
    """Valid strided depth pixels to robot-base points. pose is xyz + xyzw of the camera, USD to OpenCV flip."""
    n = depth.shape[0]
    u = ((np.arange(n) * stride + 0.5) - width / 2) / (focal * width)
    ok = np.isfinite(depth) & (depth > 0.06) & (depth < 5.0)
    d = np.where(ok, depth, 0.0)
    pts = np.stack([u[None, :] * d, u[:, None] * d, d], axis=-1)[ok]
    return pts @ (quat_mat(pose[3:]) @ np.diag([1.0, -1.0, -1.0])).T + pose[:3]

def inflate(occ: np.ndarray, r: int) -> np.ndarray:
    """Binary dilation by a disc of r cells (np.roll, edges are free space anyway)."""
    out = occ.copy()
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx * dx + dy * dy <= r * r:
                out |= np.roll(np.roll(occ, dx, axis=0), dy, axis=1)
    return out

def to_cell(xy: np.ndarray) -> tuple:
    """World xy (odom frame) to grid cell."""
    c = np.floor(np.asarray(xy) / GRID_RES + GRID_N / 2).astype(int)
    return int(np.clip(c[0], 0, GRID_N - 1)), int(np.clip(c[1], 0, GRID_N - 1))

def astar(occ: np.ndarray, start: np.ndarray, goal: np.ndarray) -> list:
    """8-connected A* between world points. Start disc is cleared, an occupied goal moves to the nearest free cell."""
    a, b = to_cell(start), to_cell(goal)
    occ = occ.copy()
    xs, ys = np.ogrid[:GRID_N, :GRID_N]
    occ[(xs - a[0]) ** 2 + (ys - a[1]) ** 2 <= 36] = False
    if occ[b]:
        free = np.argwhere(~occ)
        if len(free) == 0:
            return []
        b = tuple(int(v) for v in free[np.argmin(((free - np.array(b)) ** 2).sum(axis=1))])
    queue, cost, prev, done = [(0.0, a)], {a: 0.0}, {}, 0
    while queue and done < 25000:
        _, u = heapq.heappop(queue)
        done += 1
        if u == b:
            path = [u]
            while u != a:
                u = prev[u]
                path.append(u)
            return [(np.array(p) - GRID_N / 2 + 0.5) * GRID_RES for p in path[::-1]]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < GRID_N and 0 <= v[1] < GRID_N) or occ[v]:
                continue
            g = cost[u] + math.hypot(dx, dy)
            if g < cost.get(v, 1e18):
                cost[v], prev[v] = g, u
                heapq.heappush(queue, (g + math.hypot(v[0] - b[0], v[1] - b[1]), v))
    return []

def wrap(x: float) -> float:
    """Angle to (-pi, pi]."""
    return (x + math.pi) % (2 * math.pi) - math.pi

def rz(yaw: float) -> np.ndarray:
    """3x3 rotation about z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

# ---- source of the clouds: 97dd00b00e946cc6dcbd9d7cf11344a31d32fb815bb27c6467b5bd0523de4a2c (popcorn_task.usdc), npz sha256 microwave 651343af478f26de5a4e64a612cb79b1df03348bc5c4f9c87ed9f1b162318346, bag c3f42b44199d2e527632d22d01d6091c252c8f7ee7df8f4c85265792f5a6a84f; priors/cloud_scene/ORIGIN.json ----
CLOUD_MICROWAVE_LO = (-0.14163388311862946, -0.24172855913639069, -0.12126163393259048); CLOUD_MICROWAVE_SPAN = (0.34249570965766907, 0.5168908983469009, 0.26394376903772354)
CLOUD_MICROWAVE_COUNTS = (28, 5, 52, 353, 1, 53, 8)
CLOUD_MICROWAVE_CHECK = (7, 1521, 23065084)  # palette size, blob bytes, weighted checksum
# >>> FROZEN BEGIN cloud_microwave_blob
CLOUD_MICROWAVE_BLOB = (
    'JCQkSEhIbW1tttrattr/2ra22tra4Qto8lPl6qDc6nDc8m3h8g7H7Kjc8rja6wY48kcO8lvp8m8Y8qcZ45/x4UTx430O8orj6C3x8hBz7Jsj8j7w45vc4YUX4biE627x8rfi8pDk4brK9LVR+bql/7aT/LOR+bWD7KBv54ST56Yq7HS37KtH7EOG5xGS56us7Eye7J817Cq553jb51BD5x6k5y6Q7Cyo7Gsy7INX5zNu53XB7KVf50Ci52Zw50Cq7I8k555x6kAj5xZ+7BN454iM5xZ/7CJO50BA54tT549F5yib7ENN7Gg55zlN7GHN55Vf7GF95zh97G6C7JWX56Mq5y5h504x55GJ53bU7JvF55tN3BcTb8QASP7145sDhJDs8uiBZVQT5f2Y4QIj7tZWUpf/Q0oTAAl8uABm4QT1AEkScrqYBHp6yEITAAGlOLrbBDrCXLprKgBMALKa7dXCZCkTAGF9jzTsuJsABFpGABDbR2vsf34ADABKL/9uTqcTftj/O3UAqVf/ok0AlFD/uQmDJf9Rerz/ewDXMv/Tj4f/JT3sABvIF7qohP+psWoAALxYm1ETRwrOoSXsxgCOt1ETFAkmshfsd1oT0QDZtrouALBc7teH5bACOT3/dy/sO/+1HDUAwhoT4AquZ/wE7uVMGqgT4QPq7efd5A0DYxAT4Jz/qMv/xGAA4QCgBA/YersAhr8ATf+UALkRAEHObYbsBFNlBGskAEjdzABi3asTnooABBzBc2Psjlf/JgA2lvwDEQAX5Tv/uJoAau//BGCeJuAAX7poAOJ6Dun/pwL6hwA+wIjsAA9NjPAAlCfsz4QAAK7iApf/BLeuiKkAf17/E7qB8sVGBOr/RABUBDpQsZnskP9/aTcTD3wTQsn/BI9E4XX2N0vsAFhyVTj/h7q8rf92eG/sUqrs5LL/TEHschz/NgCSn4H/8skzddj/MwCtdAlBAOEbwMD/hKv/kXAT1qTsZBX/LLpUAE6Bp7pXfAlDuWbsxDj/OqfsNv/Rl33sWYr/nSfsBKzsBDHpL+H/aQH3fC0AOyfsygCIwOH/h7oZAJLubP8cPxEAkgBgUlsTAHwpWQDP4f28cn//nbqAMwAdpyLsvgkm1Lq2AKjdBCCjl5f/ZQB62Er/Jwk2VQARlACZgaPsAOrKtLriY2jsA/kBz7rABKAgMBMTtbcAnAv/SQCV3AmkVGbsAJRM8tFKKz8Apf/rBArokGLs45v/w7pY2d8Ankj/W7qQsxgAny4A0v81oErs4QY0ZFz/BEVjsP8VBBk57PicRbpP8sdpBIZuviYTy+H/N2wAGWnsQADkUXT/KErsuLorAJI2ACj+8ePmuH4AD6cAEy7sCQkZL0YA03r/4Q0KmKrsJ//fAIozZArJoQDpBv9mBCbL78wz0yD/ulYADIUTzADALfwEn/9kSlb/JQmbSbrmxLpIL5D/ppj/oQCVHwDXxv9OZdP/5UD8AE7fDUkATiD/SEfs8sNeREjscIUT4ZgAP1wTff9Ka2b/nLnsPRsTnLrNALuj1B8TYQBDsGwT3rrMD3zs8uJ7EzATHglWAFPMEP+9t7qPqrpz3f9ZMrpRDAsTHQ3sLgquGn0ACSn/rrp9KLpta08AM2nsBB1aSQCPcyX/oYn/2v+40ZgTCELsef+3ANgmtS/sBf9NuDv/BC1tVv/XbsX/FABgAGvsV+QAV0UAi3/sohYTwmQAomITb7rKBAx3WwDAPpUTS3rsy8IAADlYlwkTMq8Tt4wTHwIGBDuHswAjnADejJkjjz0joCQobk0jRS4mI2cmZTkjSjojYTEmPmUmhi4jSkYjlVcjuHkjil0mVGQjOZEmuzYmJEYafoImJTsjVk8jC3InCGcmi50YVk0mL3AjWzwmSZ8rw0sZl3cjYGMmcKAmRDQjqnojjHcjiXQmLpMjTi4jL4MmSisjREAjImomlUwjWnUmrSkoeqAjeT8jaZImPZ0rT0ojoT0mlJoj8ufX8NUz8c4a7teo7t2X8NJX7Ocw7NMe'
)
# <<< FROZEN END cloud_microwave_blob
CLOUD_BAG_LO = (-0.09997282922267914, -0.13266631960868835, -0.08674554526805878); CLOUD_BAG_SPAN = (0.22844670712947845, 0.2714153677225113, 0.17191462218761444)
CLOUD_BAG_COUNTS = (1, 7, 1, 1, 10, 133, 3, 23, 66, 42, 2, 11)
CLOUD_BAG_CHECK = (12, 936, 13822958)  # palette size, blob bytes, weighted checksum
CLOUD_BAG_BLOB = (
    'JCQkbUgkbUhIbW1IkW0kkW1IkZFItm1ItpFItpFt2rZt/////3lXz1QSTOrmD5imuzSzCGDHGkf0F8KiVemeQ+CIxHHKxE/xivSieAjB11rXodKHIsl9xEb7KyLHy4DWkQThxqtg25s0tHYDzlIbyZTamtpBceqaVuPXfiICx9HBKXUDH7t7oyDrVRvlJTnBC3ButTgyow70W3cDiPbeSeSuGz/3Qt7gMSwhv7+mCrjJea8DD5VLTeRDU/GWAoTnUOYJbhH/FZjBXfLSR5IFSyLNXBwE2pPSYRdc3YYVLTndYOZXLSDI12KBb8YLcdUFhN1eznvrBG/kTOedURw2C3lSPu61XI0Fe/AG2JW+ry8pyYmUHbiyhCEEjhvZIqmk12G/iQ4IJy5kXAjQtcQMz0q3DnF/qd4jzKlKhhMGnBVAy4fEGYuqz3D5jA4uL8PRdgbsWOkExTz+vsOjLc5PihR9lQ+616kZTReKpRyVKSlGlRZRhhX+LSdfqs5uYuXR2o+vK5EDazkDqS7F2WDw1po9P+jjKrAFotkSU+e/jAPpJ8l2We5lRh3Phg1KjeIWdC0CaxOXAZnVFbwZfBNcHUc+pOjZVvrIRcoFaeoAbBYBOSoMMtJKzZlkDlZYEGFfNtyqxJDMItSq11v0bf+6PyfIXAj1OBbr14OnJXwDzns/WBwlMyg+25PU2n7zEoHEmg8QELzdRJoDUfGWE0S9Kz+LMdgzApMbLZIDIdPYxFqLDpFVPWgDiTAD14g8EKdtnSHcT+AnBJErcNoH3ZPlDVx1xlWbHpnIpU0DpQ3YKuHT0cbVA2bqWejlqhHQIl8EC6fmpOKYjN5JnnUFqpMFQ+FhpiYgbfXCGbdIukUeNM2lHUNdXBWVwjZcGz1JyGCpFWHGMDMIj7QFbRfDMNJl03SuqLkGqUIH0ra7jO+ZUeunF2qM2mp72V/bz4mRe9zEeQDqYhPDQeN/tTm2IzwUtyJ4SB6oDHzxAIffh6sDdJEEF0siK89ELs9qqyVNFbzkc+ziOo4DqRPB0WINtTACbhNbDINPlteZvMdFF6u6WO1uFGiGJ929zXRfQ/C6z55UGmHhNyebx5fMQCevnAvAyoEkhvrLuJ4G0WOvEEvCJ93DoNJlbekoa+y3dQfBEo6HRNgLcBd+PiFVbhU12aHjxLhRohk9Fzfrylc1rh6JDoCCyo18WQnM95817Ld0579O+U9f70hH5j2D+VJu8a5d+JFG8bBp6bpB'
)
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
CLOUDS = (  # lower-case name -> (LO, SPAN, COUNTS, CHECK, BLOB, resting orientations as xyz Euler, grasp point in the model frame)
    ("microwave", (CLOUD_MICROWAVE_LO, CLOUD_MICROWAVE_SPAN, CLOUD_MICROWAVE_COUNTS, CLOUD_MICROWAVE_CHECK, CLOUD_MICROWAVE_BLOB, ((0, 0, 0),), (0.2009, 0.0167, 0.0107))),
    ("bag", (CLOUD_BAG_LO, CLOUD_BAG_SPAN, CLOUD_BAG_COUNTS, CLOUD_BAG_CHECK, CLOUD_BAG_BLOB, ((0, 0, 0),), (0.0034, 0.0013, -0.0021))),
)

def _entry(name): return dict(CLOUDS)[name]

def check_cloud(name, counts, blob, check):
    """Raise a clear error when the packed cloud is damaged (truncated, edited, or COUNTS not matching the point count)."""
    n_colours, n_bytes, expected = check
    data = np.concatenate([blob, (counts % 256).astype(np.uint8), (counts // 256).astype(np.uint8)]).astype(np.uint64)
    got = int((data * (np.arange(len(data), dtype=np.uint64) % 251 + 1)).sum() % 4294967291)
    if len(blob) != n_bytes or len(counts) != n_colours or counts.sum() * 3 + n_colours * 3 != n_bytes or got != expected:
        raise ValueError(f"cloud {name} is damaged: blob bytes {len(blob)} (expected {n_bytes}), colours {len(counts)} (expected {n_colours}), checksum {got} (expected {expected}).")

def load_cloud(name):
    """(xyz (N,3) metres in the model frame, rgb (N,3) uint8) of a template; decoded on every call (a few ms)."""
    lo, span, counts, check, text = _entry(name)[:5]; counts = np.asarray(counts, np.int64)
    codes = np.frombuffer(B64.encode("ascii"), np.uint8); order = np.argsort(codes)
    v = order[np.searchsorted(codes[order], np.frombuffer(text.encode("ascii"), np.uint8))].reshape(-1, 4).astype(np.uint32)
    word = (v[:, 0] << 18) | (v[:, 1] << 12) | (v[:, 2] << 6) | v[:, 3]
    blob = np.stack([word >> 16, (word >> 8) & 255, word & 255], axis=1).astype(np.uint8).reshape(-1); check_cloud(name, counts, blob, check)
    palette = blob[:3 * len(counts)].reshape(-1, 3); q = blob[3 * len(counts):].reshape(-1, 3).astype(float)
    return np.asarray(lo) + q / 255 * np.asarray(span), palette[np.repeat(np.arange(len(counts)), counts)]

def pixels3d(depth: np.ndarray, pose: np.ndarray, focal: float, width: int, stride: int) -> np.ndarray:
    """(H, W, 3) robot-base points of a strided depth image, NaN where the depth is invalid (mask-aligned cloud)."""
    n = depth.shape[0]
    u = ((np.arange(n) * stride + 0.5) - width / 2) / (focal * width)
    d = np.where(np.isfinite(depth) & (depth > 0.06) & (depth < 6.0), depth, np.nan)
    pts = np.stack([u[None, :] * d, u[:, None] * d, d], axis=-1)
    return pts @ (quat_mat(pose[3:]) @ np.diag([1.0, -1.0, -1.0])).T + pose[:3]

def colour_masks(rgb: np.ndarray) -> dict:
    """Port colours calibrated on head frames of the port (dev geometry dump): microwave body light cyan, door dark teal,
    button red, bag brown (the wooden floor is brown too, so the bag is also gated by height)."""
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    return {"body": (r > CYAN_MIN) & (g > r + 10) & (b > r + 10), "door": (np.maximum(np.maximum(r, g), b) < 110),
            "red": (r > 170) & (g < 70) & (b < 70), "bag": (r > b + 30) & (r > g + 12) & (g > b + 10) & (r > 50) & (r < 235)}


class Planner:
    """Phase machine over head RGB-D. State lives on self and is rebuilt in reset()."""

    def reset(self, task_spec: dict) -> None:
        """Start a fresh episode. task_spec carries only the task name."""
        self.task_name = task_spec["task_name"]
        self.t, self.t_stage, self.stage = 0, 0, 0
        self.odom, self.grid = np.zeros(3), np.zeros((GRID_N, GRID_N))
        self.previous_base_action = np.zeros(3)
        self.free_torso = False
        self.panel_normal = None
        self.mw, self.bag, self.btn, self.hits = None, None, None, [0, 0]
        self.wps, self.wi, self.wt, self.dw, self.wfail, self.goal = [], 0, 0, 0, False, None
        self.path, self.trail, self.recover, self.stuck, self.settle, self.spin = [], [], 0, 0, 0, 0.0
        self.grip, self.hold_tgt, self.reason, self.notes = 1.0, None, "start", []
        self.tries, self.backoffs, self.presses, self.door_ref, self.door_px, self.cav_px = 0, 0, 0, 0, 0, 0
        self.s, self.q, self.carry, self.home, self.bag_near = np.zeros(61), np.asarray(REST, dtype=np.float64), None, None, 0.11
        self.bag_psi = 0.0
        self.pos_t = 0

    def act(self, obs: dict) -> np.ndarray:
        """One control step: perceive, run the phase handler, return the 23D action (float32, shape (23,))."""
        self.s = np.asarray(obs["robot_r1::proprio"], dtype=np.float64).reshape(-1)
        rel = np.asarray(obs["robot_r1::cam_rel_poses"], dtype=np.float64).reshape(3, 7)
        self.q = np.r_[self.s[53:57], self.s[28:35]]
        self.t, self.t_stage = self.t + 1, self.t_stage + 1
        self._odometry()
        depth = np.asarray(obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"], dtype=np.float64)[0]
        self._grid(cloud(depth[::10, ::10], rel[2], HEAD_FOCAL, HEAD_W, 10))
        if self.t % 3 == 0:
            self._perceive(np.asarray(obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"])[0, ::3, ::3, :3], depth[::3, ::3], rel[2])
        if self.mw is not None and self.stage == 6:
            if self.t % 15 == 0:
                self._refine_panel_normal(np.asarray(obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"])[0, ..., :3], depth, rel[2])
            if self.panel_normal is not None and self.btn is not None:
                self.mw["n"] = self.panel_normal.copy()
                self.mw["l"] = np.array([-self.panel_normal[1], self.panel_normal[0]])
                self.mw["c"] = self.btn[:2] - self.mw["n"] * BUTTON[0] - self.mw["l"] * BUTTON[1]
        vel, target, axis = np.zeros(3), self.hold_tgt, None
        handler = (self._search_mw, self._goto_mw, self._manip_phase, self._search_bag, self._goto_bag, self._manip_phase,
                   self._goto_loaded, self._manip_phase, self._manip_phase, self._manip_phase, self._wait, self._done)[self.stage]
        out = handler()
        if out is not None:
            vel, target, axis = out
        self._budget()
        action = self._compose(vel, target, axis)
        self.previous_base_action = action[:3].copy()
        return action

    def log(self) -> dict:
        """Diagnostics for the mutator: phase name, stage index, reason of the current wait/retry/back-off, counters."""
        return {"phase": NAMES[self.stage], "stage": self.stage, "reason": self.reason, "steps": self.t, "tries": self.tries,
                "backoffs": self.backoffs, "presses": self.presses, "mw_hits": self.hits[0], "bag_hits": self.hits[1],
                "door_px": [self.door_px, self.door_ref], "cav_bag_px": self.cav_px, "width": round(self._width(), 4),
                "wp": [self.wi, len(self.wps)], "stuck": self.stuck, "notes": self.notes[-8:]}

    # frames and perception ------------------------------------------------
    def _odometry(self) -> None:
        # R1Pro proprioceptive yaw velocity has a measured bias. Use the
        # calibrated command model already validated by Task 87 a023/localizer.py.
        # Integrate the action actually issued on the preceding physics step.
        command = self.previous_base_action
        size = float(np.linalg.norm(command[:2]))
        xy = np.zeros(2)
        if size > 1e-6:
            unit = command[:2] / size
            deficit = 0.0064 * unit[0] ** 2 + 0.0023 * unit[1] ** 2
            xy = unit * max(0.0, 0.75 * size - deficit)
        # Command prediction is invalid when a collision prevents translation.
        # Detect that directly from allowed measured base velocity, retaining
        # measured lateral slip instead of integrating motion through a wall.
        predicted_speed = float(np.linalg.norm(xy))
        measured_xy = self.s[:2]
        if predicted_speed > 0.01 and float(measured_xy @ xy) < 0.5 * predicted_speed ** 2:
            xy = measured_xy.copy()
        w = float(command[2])
        deficit = 0.009 if size > 0.01 else 0.03
        rate = np.sign(w) * (abs(w) - deficit if abs(w) >= 0.1 else abs(w) * (1.0 - deficit / 0.1))
        v = np.r_[xy, rate]
        mid = self.odom[2] + v[2] * DT / 2
        c, sn = math.cos(mid), math.sin(mid)
        self.odom = self.odom + DT * np.array([c * v[0] - sn * v[1], sn * v[0] + c * v[1], v[2]])

    def _to_base(self, p: np.ndarray) -> np.ndarray:
        return (p - np.r_[self.odom[:2], 0.0]) @ rz(self.odom[2])

    def _to_odom(self, p: np.ndarray) -> np.ndarray:
        return p @ rz(self.odom[2]).T + np.r_[self.odom[:2], 0.0]

    def _grid(self, base: np.ndarray) -> None:
        m = (np.linalg.norm(base[:, :2], axis=1) > 0.45) & (base[:, 2] > 0.12) & (base[:, 2] < 1.5)
        ij = np.floor(self._to_odom(base[m])[:, :2] / GRID_RES + GRID_N / 2).astype(int)
        ok = ((ij >= 0) & (ij < GRID_N)).all(axis=1)
        self.grid *= 0.995
        np.add.at(self.grid, (ij[ok, 0], ij[ok, 1]), 1.0)
        self.grid = np.minimum(self.grid, 5.0)

    def _width(self) -> float:
        return float(self.s[49] + self.s[50])

    def _refine_panel_normal(self, rgb, depth, pose):
        """Estimate the appliance's fixed front plane around its tracked red marker.

        Local surface-normal voting excludes the curved button and horizontal
        counter. It uses only the calibrated RGB-D observation.
        """
        pts = pixels3d(depth, pose, HEAD_FOCAL, HEAD_W, 1)
        prior = self._to_base(self.btn if self.btn is not None else self._face(*BUTTON))
        red = colour_masks(rgb)["red"] & np.isfinite(pts).all(-1)
        red &= np.linalg.norm(pts - prior, axis=-1) < 0.20
        if int(red.sum()) < 8:
            return
        center = np.median(pts[red], axis=0)
        yy, xx = np.nonzero(red)
        y0, y1 = max(0, int(yy.min()) - 65), min(depth.shape[0], int(yy.max()) + 66)
        x0, x1 = max(0, int(xx.min()) - 65), min(depth.shape[1], int(xx.max()) + 66)
        crop = pts[y0:y1, x0:x1]
        colors = rgb[y0 + 1:y1 - 1, x0 + 1:x1 - 1].astype(float)
        pp = crop[1:-1, 1:-1]
        normals = np.cross(crop[1:-1, 2:] - crop[1:-1, :-2], crop[2:, 1:-1] - crop[:-2, 1:-1])
        length = np.linalg.norm(normals, axis=-1)
        normals /= np.maximum(length[..., None], 1e-12)
        good = (colors.min(-1) > 100) & (colors.max(-1) - colors.min(-1) < 65)
        good &= (np.linalg.norm(pp - center, axis=-1) < 0.15) & (np.abs(normals[..., 2]) < 0.15)
        good &= np.isfinite(normals).all(-1) & (length > 1e-10)
        vectors = normals[good, :2]
        if len(vectors) < 100:
            return
        vectors *= np.where((vectors * -pp[good, :2]).sum(1) > 0, 1, -1)[:, None]
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        hist, edges = np.histogram(angles, bins=180, range=(-np.pi, np.pi))
        k = int(hist.argmax()); peak = 0.5 * (edges[k] + edges[k + 1])
        select = np.abs((angles - peak + np.pi) % (2 * np.pi) - np.pi) < 0.0524
        if int(select.sum()) < 100 or float(select.mean()) < 0.5:
            return
        normal = np.median(vectors[select], axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-9)
        world_normal = rz(self.odom[2])[:2, :2] @ normal
        if float(world_normal @ self.mw["n"]) < 0.7:
            return
        self.panel_normal = world_normal

    def _perceive(self, rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray) -> None:
        """Colour masks + aligned head cloud: microwave front plane (door pixels), button, bag, door and cavity checks."""
        pts = self._to_odom(pixels3d(depth, pose, HEAD_FOCAL, HEAD_W, 3).reshape(-1, 3)).reshape(depth.shape + (3,))
        m = colour_masks(rgb)
        ok = np.isfinite(pts[..., 0])
        # Native RTX changes material brightness/hue. Anchor the appliance on its
        # visible red toggle marker plus a vertical dark door of the asset's size.
        # No ground-truth object poses, segmentation or instance-dependent coordinates.
        red_mask = m["red"] & ok & (pts[..., 2] > BODY_Z[0]) & (pts[..., 2] < BODY_Z[1])
        labels, count = label(red_mask)
        candidates = []
        for component in range(1, count + 1):
            pixels = labels == component
            if int(pixels.sum()) < 2:
                continue
            button = np.median(pts[pixels], axis=0)
            near = m["door"] & ok & (np.linalg.norm(pts[..., :2] - button[:2], axis=-1) < 0.55)
            near &= (pts[..., 2] > button[2] - 0.015) & (pts[..., 2] < button[2] + 0.24)
            door = pts[near]
            if len(door) < 12:
                continue
            delta = door[:, :2] - button[:2]
            toward = self.odom[:2] - button[:2]
            angle = math.atan2(toward[1], toward[0])
            best = None
            for theta in angle + np.radians(np.arange(-75, 76, 5)):
                n = np.array([math.cos(theta), math.sin(theta)])
                lv = np.array([-n[1], n[0]])
                dn, dl = delta @ n, delta @ lv
                keep = (dn > -0.12) & (dn < 0.035) & (dl > -0.47) & (dl < -0.04)
                score = int(keep.sum())
                if best is None or score > best[0]:
                    best = (score, n, keep)
            if best[0] < 12:
                continue
            n, keep = best[1], best[2]
            for _ in range(2):
                xy = door[keep, :2]; xy = xy - xy.mean(axis=0)
                values, vectors = np.linalg.eigh(xy.T @ xy)
                n = vectors[:, 0]
                if n @ toward < 0:
                    n = -n
                lv = np.array([-n[1], n[0]])
                dn, dl = delta @ n, delta @ lv
                keep = (dn > -0.12) & (dn < 0.04) & (dl > -0.47) & (dl < -0.04)
                if int(keep.sum()) < 12:
                    break
            if int(keep.sum()) < 12 or values[0] > 0.15 * max(values[1], 1e-9):
                continue
            lateral = delta[keep] @ lv
            z = door[keep, 2]
            if np.ptp(np.quantile(lateral, [.05,.95])) < .10 or np.ptp(np.quantile(z,[.05,.95])) < .06:
                continue
            center = button[:2] - n * BUTTON[0] - lv * BUTTON[1]
            # The native microwave has a light control panel coplanar with
            # its red button. A dark TV with a red marker has no such panel.
            delta3 = pts - button
            white = (rgb.min(axis=-1) > 100) & (np.ptp(rgb.astype(float), axis=-1) < 65) & ok
            panel = white & (np.abs(delta3[..., :2] @ lv) < 0.055) & (delta3[..., 2] > 0.02) & (delta3[..., 2] < 0.18)
            panel &= np.abs(delta3[..., :2] @ n) < 0.025
            if int(panel.sum()) < 4:
                continue
            candidates.append((int(keep.sum()), button, {'c':center,'n':n,'l':lv,'zc':float(button[2]-BUTTON[2]),'fine':True}))
        if candidates and self.stage <= 1:
            _, button, new = max(candidates, key=lambda item:item[0])
            if self.mw is None or np.linalg.norm(self.mw['c']-new['c']) > .4:
                self.mw = new
            else:
                self.mw = {k: .8*self.mw[k]+.2*new[k] for k in ('c','n','l','zc')}
                self.mw['n'] /= np.linalg.norm(self.mw['n'])
                self.mw['l'] = np.array([-self.mw['n'][1],self.mw['n'][0]])
                self.mw['fine'] = True
            self.btn = button
            self.hits[0] += 1
        if self.mw is not None:
            red = pts[m['red'] & ok]
            red = red[np.linalg.norm(red-self._face(*BUTTON),axis=1)<BTN_GATE]
            if len(red)>=2:
                self.btn=np.median(red,axis=0)
                if self.stage>=6:
                    self.mw['c'] += BTN_RELOC*(self.btn[:2]-self._face(*BUTTON)[:2])
            self.door_px=int(self._in_face(pts[m['door'] & ok],(-.12,.05),(-.26,.14),.13).sum())
            self.cav_px=int(self._in_face(pts[m['bag'] & ok],(CAVITY[0]-CAV_HALF[0],CAVITY[0]+CAV_HALF[0]),
                                        (CAVITY[1]-CAV_HALF[1],CAVITY[1]+CAV_HALF[1]),CAV_HALF[2]).sum())
        if self.mw is not None and (self.stage in (3, 4) or (self.stage == 5 and self.goal is not None and getattr(self, "grasp_refine", 0) >= 20)):  # bag: brown, at counter height, on the counter beside the microwave
            rel = pts[..., :2] - self.mw["c"]
            sel = m["bag"] & ok & (pts[..., 2] > BAG_Z[0]) & (pts[..., 2] < BAG_Z[1]) & (np.abs(rel @ self.mw["n"] + 0.1) < BAG_DN) \
                & (np.abs(rel @ self.mw["l"]) > BAG_SIDE[0]) & (np.abs(rel @ self.mw["l"]) < BAG_SIDE[1])
            lab, k = label(sel)
            sizes = np.bincount(lab.ravel(), minlength=k + 1)[1:]
            bag = pts[lab == 1 + int(np.argmax(sizes))] if k else np.zeros((0, 3))  # largest blob only (other brown wood)
        else:
            bag = np.zeros((0, 3))
        frozen = self.stage == 4 and self.hits[1] >= MIN_HITS and self.bag is not None \
            and float(np.linalg.norm(self.bag[:2] - self.odom[:2])) < 1.2  # near the bag: estimate frozen, goal stays fixed
        if len(bag) >= BAG_MIN_PX and (self.stage in (3, 4) or (self.stage == 5 and self.goal is not None and getattr(self, "grasp_refine", 0) >= 20)) and self.mw is not None and not frozen:
            ext = lambda v: np.quantile(v, 1 - BAG_Q) - np.quantile(v, BAG_Q)
            psi = min(np.radians(np.arange(-45, 45, 5)), key=lambda th: ext(bag[:, :2] @ (rz(th)[:2, :2] @ self.mw["n"]))
                      * ext(bag[:, :2] @ (rz(th + math.pi / 2)[:2, :2] @ self.mw["n"])))  # bag yaw: smallest box
            n, lv = rz(psi)[:2, :2] @ self.mw["n"], rz(psi + math.pi / 2)[:2, :2] @ self.mw["n"]
            a, b = bag[:, :2] @ n, bag[:, :2] @ lv
            mid = lambda v: 0.5 * (np.quantile(v, BAG_Q) + np.quantile(v, 1 - BAG_Q))
            c = np.r_[n * mid(a) + lv * mid(b), np.quantile(bag[:, 2], 1 - BAG_Q) - BAG_HALF_H]  # box centre, top minus half height
            self.bag_psi = float(min((psi, psi + math.pi / 2), key=lambda v: abs(v - TURN_PREF)))  # wall to pinch, robot faces it
            w = a if self.bag_psi == float(psi) else b
            self.bag_near = float(np.quantile(w, 1 - BAG_Q) - mid(w))  # that wall from the bag centre
            if self.bag is None or float(np.linalg.norm(self.bag[:2] - c[:2])) > 0.3:
                self.bag, self.hits[1] = c, 1
            else:
                self.bag, self.hits[1] = 0.7 * self.bag + 0.3 * c, self.hits[1] + 1

    def _face(self, dn: float, dl: float, dz: float) -> np.ndarray:
        """Face-frame offset to an odom point."""
        m = self.mw
        return np.r_[m["c"] + m["n"] * dn + m["l"] * dl, m["zc"] + dz]

    def _in_face(self, p: np.ndarray, dn: tuple, dl: tuple, dz: float) -> np.ndarray:
        m = self.mw
        a, b, z = (p[:, :2] - m["c"]) @ m["n"], (p[:, :2] - m["c"]) @ m["l"], p[:, 2] - m["zc"]
        return (a > dn[0]) & (a < dn[1]) & (b > dl[0]) & (b < dl[1]) & (np.abs(z) < dz)

    # navigation -----------------------------------------------------------
    def _spin(self) -> tuple:
        self.spin += 0.9 * DT
        if self.spin > 2 * math.pi + 0.3:  # nothing in a full turn: drive 1.5 m ahead on the grid, then turn again
            self.spin, self.goal = 0.0, (self.odom[:2] + 1.5 * np.array([math.cos(self.odom[2]), math.sin(self.odom[2])]), self.odom[2])
        if self.goal is not None:
            vel, ok = self._nav(*self.goal)
            self.goal = None if ok or self.t_stage % 300 == 0 else self.goal
            return vel, None, None
        return np.array([0.0, 0.0, 0.9]), None, None

    def _nav(self, goal: np.ndarray, yaw: float, tol: float = 0.06) -> tuple:
        """A* path on the inflated depth grid, lookahead following, stuck backoff. Returns (vel, arrived)."""
        pos = self.odom[:2]
        dist = float(np.linalg.norm(goal - pos))
        if self.t % 30 == 0:
            self.trail = (self.trail + [pos.copy()])[-4:]
        if self.recover > 0:
            self.recover -= 1
            return np.array([-0.3, 0.3 if self.stuck % 2 else -0.3, 0.0]), False
        if len(self.trail) == 4 and dist > 0.25 and float(np.linalg.norm(self.trail[0] - pos)) < 0.05:
            self.stuck, self.recover, self.trail, self.path = self.stuck + 1, 45, [], []
        if self.t % 10 == 0 or not self.path:
            self.path = astar(inflate(self.grid >= 1.5, INFLATE + min(self.stuck, 3)), pos, goal)
        tgt = goal
        for p in self.path:
            if float(np.linalg.norm(p - pos)) > LOOKAHEAD:
                tgt = p
                break
        e = tgt - pos
        v = e / max(float(np.linalg.norm(e)), 1e-6) * min(VMAX, 1.2 * dist)
        face = math.atan2(e[1], e[0]) if dist > 1.0 else yaw
        wz = float(np.clip(1.5 * wrap(face - self.odom[2]), -YAW_MAX, YAW_MAX))
        c, sn = math.cos(self.odom[2]), math.sin(self.odom[2])
        return np.array([c * v[0] + sn * v[1], -sn * v[0] + c * v[1], wz]), (dist < tol or self.stuck >= 1 and dist < STALL_OK) \
            and abs(wrap(yaw - self.odom[2])) < 0.1

    def _arrive(self, goal: tuple, tol: float = 0.06) -> tuple:
        """Drive to goal (xy, yaw); True once settled there for 5 steps (or stuck close to it, see STALL_OK)."""
        vel, ok = self._nav(*goal, tol)
        self.settle = self.settle + 1 if ok else 0
        return vel, self.settle >= 5

    def _stand(self, point: np.ndarray, ahead: float, left: float, turn: float = 0.0) -> tuple:
        """Base pose with `point` at (ahead, left) in the base frame, yaw = facing the microwave (-n) plus `turn`."""
        yaw = math.atan2(-self.mw["n"][1], -self.mw["n"][0]) + turn
        return point[:2] - ahead * np.array([math.cos(yaw), math.sin(yaw)]) - left * np.array([-math.sin(yaw), math.cos(yaw)]), yaw

    # phases ---------------------------------------------------------------
    def _search_mw(self) -> tuple:
        self.reason = "turning, microwave not seen" if self.mw is None else "microwave sighted"
        if self.hits[0] >= MIN_HITS or (self.mw is not None and self.t_stage > 90):
            self._next(1, "seen")
        return self._spin()

    def _goto_mw(self) -> tuple:
        vel, ok = self._arrive(self._stand(self.mw["c"], *STAND_OPEN))
        self.reason = "driving to the front of the microwave"
        if ok and (self.door_px >= DOOR_MIN_PX or self.door_ref):
            self.door_ref = self.door_ref or self.door_px  # closed-door reference, measured once (on a later visit the door is open)
            self._next(2, "at_front_door_seen")
        elif ok:
            self.reason = "at the front but the door is not seen"
        return vel, None, None

    def _search_bag(self) -> tuple:
        self.reason = "turning, bag not seen"
        if self.hits[1] >= MIN_HITS:
            self._next(4, "seen")
        if self.goal is None and self.t_stage < 60 and self.mw is not None:
            self.goal = self._stand(self.mw["c"] + self.mw["l"] * 0.8, 1.3, 0.0)
        if self.goal is not None and self.t_stage < 140:
            vel, ok = self._nav(*self.goal, 0.15)
            if ok:
                self.goal = None
            return vel, None, None
        return self._spin()

    def _goto_bag(self) -> tuple:
        vel, ok = self._arrive(self._stand(self.bag, *GRASP_AT, self.bag_psi), BAG_TOL)
        self.reason = "driving to the bag"
        if ok:
            self._next(5, "at_bag")
        return vel, None, None

    def _goto_loaded(self) -> tuple:
        if not HOLD_WIDTH[0] < self._width() < HOLD_WIDTH[1]:
            self._next(3, "bag_dropped_while_carrying")
            return None
        vel, ok = self._arrive(self._stand(self._face(INSERT_DN, INSERT_DL, 0.0), STAND_PLACE + TRANSIT, -0.10), 0.10)
        self.reason = "carrying the bag to the microwave"
        if ok:
            self._next(7, "at_cavity")
        return vel, self.hold_tgt, self._axis(GRASP_AXIS) @ rz(self.odom[2])

    def _wait(self) -> tuple:
        self.reason = "waiting for the heating"
        if self.t_stage >= WAIT_STEPS:  # the episode ends on success: still being called means the bag is not cooked
            if self.door_px < DOOR_CLOSED_FRAC * self.door_ref:
                self._next(8, "not_cooked_door_open")
            else:
                self._next(9, "not_cooked_repress")
        return None

    def _done(self) -> tuple:
        self.reason = "done"
        return None

    def _axis(self, a: tuple) -> np.ndarray:
        v = np.r_[self.mw["n"] * a[0], a[1]]
        return v / np.linalg.norm(v)

    # manipulation ---------------------------------------------------------
    def _manip_phase(self) -> tuple:
        """Approach (if the phase has a stand pose), then track the waypoint list, then check the result."""
        if self.goal is not None:
            if self.stage == 5 and getattr(self, "grasp_refine", 0):
                self.grasp_refine += 1
                if self.grasp_refine == 20:
                    self.bag, self.hits[1] = None, 0
                self.reason = "settling and reobserving bag before grasp"
                if self.grasp_refine < 35 or self.hits[1] < MIN_HITS:
                    return np.zeros(3), None, None
                vel, ok = np.zeros(3), True
            else:
                vel, ok = self._arrive(self.goal, BAG_TOL if self.stage == 5 else 0.06)
                if self.stage == 5 and ok:
                    self.grasp_refine = 1
                    return np.zeros(3), None, None
            self.pos_t += 1
            if self.pos_t > 200 and float(np.linalg.norm(self.goal[0] - self.odom[:2])) < 0.35:
                ok = True  # positioning stalled near the stand point: accept it
            self.reason = "positioning the base for " + NAMES[self.stage]
            if ok:
                wps = self._waypoints()
                if self.stage != 7:  # from the start pose the arm would sweep under the counter: lift it in front of the chest first
                    wps = [(self._to_odom(np.asarray(READY)), wps[0][1], wps[0][2], -0.05, 1)] + wps
                self.goal, self.wps, self.wi, self.wt, self.dw, self.wfail = None, wps, 0, 0, 0, False
            return vel, self.hold_tgt, None
        if self.wi >= len(self.wps) or self.wfail:
            self.dw += 1  # arm back to the start pose first (it would hide the door from the head), except with a bag in hand
            if self.stage in (5, 7) and self.wfail or self.stage == 5 or self.dw > CHECK_SETTLE:
                self.dw = 0
                self._check()
            return np.zeros(3), None, None
        pos, ax, self.grip, tol, dwell = self.wps[self.wi]
        tgt = self._to_base(pos)
        self.wt += 1
        if self.stage == 5 and self.wi in (1, 2) and self.wt >= WP_BUDGET // 2 and not self.free_torso \
                and float(np.linalg.norm(fk(self.q)[0] - tgt)) > 0.04:
            # The right arm alone may saturate laterally. Release the artificial
            # torso-yaw lock only after observing an actual failure to converge.
            self.free_torso = True
            self.wt = 0
            self.notes.append("grasp_ik_recovery: permit torso yaw")
        if float(np.linalg.norm(fk(self.q)[0] - tgt)) < abs(tol):
            self.dw += 1
        if self.dw >= dwell or tol < 0 and self.wt > SOFT_STEPS:  # negative tolerance: a sweep point, passed after SOFT_STEPS
            if self.stage == 5 and self.wi == 3:
                self.grasp_tip_z = float(fk(self.q)[0][2])
            self.wi, self.wt, self.dw = self.wi + 1, 0, 0
        elif self.wt > WP_BUDGET and self.stage == 7 and self.wi == 2 and self._bag_dn() < INSERT_OK_DN:
            self.wi, self.wt, self.dw = self.wi + 1, 0, 0  # blocked, but the bag centre is past the lip: release here
            self.notes.append(f"insert_blocked_at_dn {self._bag_dn():.3f}")
        elif self.wt > WP_BUDGET:
            self.wfail, self.reason = True, f"waypoint {self.wi} of {NAMES[self.stage]} not reached"
        self.hold_tgt = tgt
        return np.zeros(3), tgt, ax @ rz(self.odom[2])

    def _bag_dn(self) -> float:
        """Depth of the carried bag centre along the normal: fingertip (FK) plus the carry offset measured at the grasp."""
        off = self.carry if self.carry is not None else np.r_[-self.mw["n"] * GRASP_OFF[0], -GRASP_OFF[1]]
        return float((self._to_odom(fk(self.q)[0])[:2] + off[:2] - self.mw["c"]) @ self.mw["n"])

    def _door_pt(self, th: float, rho: float, out: float) -> np.ndarray:
        """Point on the door front surface at angle th, radius rho from the hinge, `out` ahead of the surface."""
        h = DOOR_T + out
        return self._face(HINGE[0] + rho * math.sin(th) + h * math.cos(th), HINGE[1] + rho * math.cos(th) - h * math.sin(th), 0.0)

    def _door_out(self, th: float) -> np.ndarray:
        return np.r_[-math.sin(th) * self.mw["l"] + math.cos(th) * self.mw["n"], 0.0]

    def _waypoints(self) -> list:
        """Waypoints (odom point, odom finger axis, grip, tolerance, dwell) of the current manipulation phase; a negative
        tolerance marks a sweep point that is passed after SOFT_STEPS."""
        return {2: self._wp_open_door, 5: self._wp_grasp_bag, 7: self._wp_place_bag, 8: self._wp_close_door, 9: self._wp_press}[self.stage]()

    def _wp_open_door(self) -> list:
        """Pinch the door front near the free edge (assisted grasp), pull along the hinge arc, release, back off."""
        r = HANDLE_DLS[self.tries % len(HANDLE_DLS)] - HINGE[1]
        ax = lambda th: -self._door_out(th)
        wps = [(self._door_pt(0, r, 0.12), ax(0), 1.0, 0.02, 1), (self._door_pt(0, r, -0.005), ax(0), 1.0, 0.015, 3),
               (self._door_pt(0, r, -0.005), ax(0), -1.0, 9.0, 40)]
        wps += [(self._door_pt(th, r, -0.005), ax(th), -1.0, -0.03, 1) for th in np.linspace(0.1, THETA_OPEN, 14)]
        return wps + [(self._door_pt(THETA_OPEN, r, -0.005), ax(THETA_OPEN), 1.0, 9.0, 15), (self._door_pt(THETA_OPEN, r, 0.15), ax(THETA_OPEN), 1.0, 0.04, 1)]

    def _wp_grasp_bag(self) -> list:
        """Vertical pinch across the bag wall facing the robot, close, lift."""
        a, up = self._axis(GRASP_AXIS), np.array([0.0, 0.0, 1.0])
        u = rz(self.bag_psi)[:2, :2] @ self.mw["n"]  # normal of the bag wall facing the robot
        g = self.bag + np.r_[u * (self.bag_near - GRASP_OFF[0]), GRASP_OFF[1]]
        return [(g - a * 0.15, a, 1.0, 0.03, 1), (g, a, 1.0, 0.020, 3), (g, a, -1.0, 9.0, 30), (g + up * 0.15, a, -1.0, 0.03, 1)]

    def _wp_place_bag(self) -> list:
        """Shallow insert (the hand stays out of the cavity), release, lift out of the rim, back off, push the bag in."""
        a, n3, up = self._axis(PLACE_AXIS), np.r_[self.mw["n"], 0.0], np.array([0.0, 0.0, 1.0])
        off = self.carry if self.carry is not None else np.r_[-self.mw["n"] * GRASP_OFF[0], -GRASP_OFF[1]]
        goal = self._face(INSERT_DN, INSERT_DL, INSERT_DZS[self.tries % len(INSERT_DZS)]) - off
        p = lambda dn: self._face(dn, INSERT_DL, 0.0)
        return [(goal + n3 * 0.30, a, -1.0, 0.025, 1), (goal + n3 * 0.10, a, -1.0, 0.02, 1), (goal, a, -1.0, 0.02, 3), (goal, a, 1.0, 9.0, 20),
                (goal + up * LIFT_AFTER, a, 1.0, -0.03, 1), (goal + up * LIFT_AFTER + n3 * 0.25, a, 1.0, 0.04, 1), (p(0.20), -n3, -1.0, 0.025, 1),
                (p(0.06), -n3, -1.0, -0.02, 1), (p(PUSH_TO), -n3, -1.0, -0.02, 1), (p(0.25), -n3, -1.0, 0.04, 1)]

    def _wp_close_door(self) -> list:
        """Closed hand on the outer face of the open door, sweep the arc back to closed, back off."""
        n3 = np.r_[self.mw["n"], 0.0]
        wps = [(self._door_pt(1.55, PUSH_RHO, 0.15) + n3 * 0.10, -self._door_out(1.55), -1.0, 0.06, 1),
               (self._door_pt(1.55, PUSH_RHO, 0.06), -self._door_out(1.55), -1.0, 0.04, 1)]
        wps += [(self._door_pt(th, PUSH_RHO, 0.0), -self._door_out(th), -1.0, -0.03, 1) for th in np.linspace(1.4, -0.05, 14)]
        shut = self._door_pt(0.0, PUSH_RHO + 0.06, -0.02)  # press the door front near the free edge to latch it shut
        return wps + [(shut, -n3, -1.0, -0.02, 1), (self._door_pt(0.0, PUSH_RHO, 0.15), -n3, -1.0, 0.04, 1)]

    def _wp_press(self) -> list:
        """Fingertip into the red button (seen, else the model offset) along -normal, hold, back off."""
        n3 = np.r_[self.mw["n"], 0.0]
        du, dz = PRESS_TRIALS[self.presses % len(PRESS_TRIALS)]
        b = (self.btn if self.btn is not None else self._face(*BUTTON)) + np.r_[self.mw["l"] * du, dz]
        return [(b + n3 * 0.10, -n3, -1.0, 0.02, 1), (b - n3 * PRESS_DEPTH, -n3, -1.0, 0.015, PRESS_HOLD), (b + n3 * 0.12, -n3, -1.0, 0.04, 1)]

    def _check(self) -> None:
        """Judge the finished (or failed) waypoint list of the phase from the observation: advance, retry or back off."""
        st = self.stage
        if st == 2:
            ok, why = self.door_px < DOOR_OPEN_FRAC * self.door_ref, f"door pixels {self.door_px} of {self.door_ref}"
        elif st == 5:
            ok, why = HOLD_WIDTH[0] < self._width() < HOLD_WIDTH[1] and not self.wfail, f"finger width {self._width():.3f}"
            if ok:  # bag centre seen before the grasp, relative to the tip; the hand turns by -psi for the carry, so does the offset
                self.carry = self.bag - self._to_odom(fk(self.q)[0])
                self.carry[:2] = rz(-self.bag_psi)[:2, :2] @ self.carry[:2]
                self.carry[2] = self.bag[2] - getattr(self, "grasp_tip_z", fk(self.q)[0][2] - 0.15)
                self.hold_tgt = np.asarray(CARRY)  # bag close to the body while driving (held further out it hits the microwave)
        elif st == 7:  # released (past the release waypoint) and bag pixels in the cavity box
            ok, why = self.cav_px >= CAV_BAG_PX and self.wi > 3, f"bag pixels in the cavity {self.cav_px}, waypoint {self.wi}"
            if not ok and not HOLD_WIDTH[0] < self._width() < HOLD_WIDTH[1] and self.tries >= 1:
                self._next(3, "bag_not_in_cavity_and_not_held")
                return
        elif st == 8:
            ok, why = self.door_px >= DOOR_CLOSED_FRAC * self.door_ref, f"door pixels {self.door_px} of {self.door_ref}"
        else:
            ok, why = not self.wfail, "press waypoints " + ("reached" if not self.wfail else "not reached")
            self.presses += 1
        if ok:
            self._next(st + 1, "ok: " + why)
            return
        self.tries += 1
        self.notes.append(f"{NAMES[st]}_retry{self.tries}: {why}")
        if self.tries >= MAX_TRIES and st != 9:
            self.backoffs += 1
            self._next(FALLBACK[st], f"gave_up after {self.tries} tries: {why}")
        else:
            self.reason = f"retry {self.tries}: {why}"
            self._enter(st)

    def _next(self, st: int, why: str) -> None:
        self.notes.append(f"{NAMES[self.stage]}->{NAMES[st]}:{why}@{self.t}")
        self.reason = why
        if st != self.stage:
            self.tries = 0
        self.stage, self.t_stage, self.settle, self.path, self.spin, self.goal = st, 0, 0, [], 0.0, None
        if st in (3, 0):
            self.hits[1 if st == 3 else 0] = 0
        self._enter(st)

    def _enter(self, st: int) -> None:
        """Set the stand pose of a manipulation phase; its waypoints are built on arrival."""
        self.pos_t = 0
        self.grasp_refine = 0
        if st == 7:
            self.free_torso = True
        self.wps, self.wi, self.wt, self.dw, self.wfail, self.carry = [], 0, 0, 0, False, None if st == 5 else self.carry
        stands = {2: (self.mw["c"] if self.mw else None, STAND_OPEN), 5: (self.bag, GRASP_AT + (self.bag_psi,)), 8: (self.mw["c"] if self.mw else None, STAND_CLOSE),
                  9: (self.mw["c"] if self.mw else None, STAND_PRESS), 7: (self._face(INSERT_DN, INSERT_DL, 0.0) if self.mw else None, (STAND_PLACE, -0.10))}
        if st in stands:
            self.goal = self._stand(stands[st][0], *stands[st][1])
        if st not in (6, 7):
            self.hold_tgt = None  # park the arm unless a bag is carried

    def _budget(self) -> None:
        """Out of budget: back off to the FALLBACK phase (never forward)."""
        if self.stage >= 11 or self.t_stage <= BUDGET[self.stage]:
            return
        self.backoffs += 1
        fb = FALLBACK[self.stage]
        if fb in (1, 2, 4, 6, 7, 8, 9) and self.mw is None or fb in (4, 5) and self.bag is None:
            fb = 0 if self.mw is None else 3
        self._next(fb, f"budget of {NAMES[self.stage]} spent")

    def _compose(self, vel: np.ndarray, target: object, axis: object) -> np.ndarray:
        if self.home is None:
            self.home = self.q.copy()  # start torso and arm: the head camera looks as at the start (IK bends the torso)
        if target is None:  # no arm task: return torso and arm to the start pose at a bounded joint speed
            qc = self.q + np.clip(self.home - self.q, -HOME_RATE, HOME_RATE)
        else:
            side = None
            if axis is not None and abs(float(axis[2])) > 0.9 and self.stage in (5, 6, 7) and self.mw is not None:
                yaw = self.bag_psi if self.stage == 5 else 0.0  # grasp: across the bag wall; carry and place: wall turned to the face
                side = np.r_[rz(yaw)[:2, :2] @ self.mw["n"], 0.0] @ rz(self.odom[2])
                side = side if float(side @ fk(self.q)[2]) >= 0 else -side
            qc = ik_step(self.q, target, axis, side, lock_torso=not self.free_torso)
        a = np.zeros(23)
        a[:3] = np.clip(vel / np.asarray(SCALE), -1.0, 1.0)
        a[3:7], a[7:14], a[14], a[15:22], a[22] = qc[:4], self.s[3:10], 1.0, qc[4:], self.grip
        return a.astype(np.float32)


def entrypoint() -> Planner:
    """Return a fresh planner."""
    return Planner()
