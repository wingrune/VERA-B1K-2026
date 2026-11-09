"""rtx_v3, chopping_wood (task 44) parent, RTX backend, OBS_CONTRACT inputs only (head RGB-D, camera poses, proprio). SlicingRule cuts a
log on ANY contact with the axe while it is active (re-activates 2 s after the last contact): no force, block or upright log needed.
Phases = PHASES (log() gives phase, stage, reason); tunables are the named constants after the templates. Kinematics, CLOUD_* from rtx_v2."""
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


def dilate(m, size):
    return maximum_filter(m, size=size, mode="nearest")


def pointcloud(cam, mask=None, stride=1):
    d = cam["depth"]; valid = np.isfinite(d) & (d > .06) & (d < 5)
    if mask is not None: valid &= mask
    y, x = np.where(valid); y, x = y[::stride], x[::stride]; z = d[y, x]; p = np.c_[(x + .5 - cam["K"][0, 2]) / cam["K"][0, 0] * z, (y + .5 - cam["K"][1, 2]) / cam["K"][1, 1] * z, z]
    return p @ cam["R"].T + cam["t"], cam["rgb"][y, x], np.c_[x, y]



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




CLOUD_AXE_LO = (-0.33258238434791565, -0.07387106120586395, -0.015257981605827808); CLOUD_AXE_SPAN = (0.5319108814001083, 0.203074112534523, 0.04045627731829882)
CLOUD_AXE_COUNTS = (398, 1, 1, 1, 9, 38, 138, 2, 233, 48, 25, 5, 1)
CLOUD_AXE_CHECK = (13, 2739, 39956299)  # palette size, blob bytes, weighted checksum
# >>> FROZEN BEGIN cloud_axe_blob
CLOUD_AXE_BLOB = (
    'AAAAJAAAJCQASCQASCQkbSQkbUgkbUhIkUgkkUhIkW1Itm1Itm1t1QBK1lqb5M9c0aYzzEO7y1Bxx4B70BZF1luayJJz00w6xzg/zQQl3p85zDmvvMckzibQ2NYp6eYg5rRq3n6D0CEN1wGm2G5BymRMyMRetsYi6q1Zst5V17sv2utd2Dpftc5wzLos1C4X24M/y4x20eQMyYY868hhzFU9z2iN0Fmb46Q5yY51xvAX5spf0DXbvKdoymqJymaF29wKzkYzy01u2NJaz9hV10Y32RvYzy0U0v0p1ka20D7G79ppr+goxvBO1DYiychd1Ih7zj1x1QG1zM0n2BK9seMb0wCY4tVZ985w9sYk4alvy7pk0fke5J902Kpt06sy2S7KyhGi88M72vhHvq8ry9pj0h8yygjyrdUXtvEpyCoP8cR+77tozQwT0NoJzA9x0eIKzvlE1YI+989u2N5zzz1tv6ouvNETssNayRMq395w2u0X1lee0pc5v5thwswkyeBqy1qDvttp/thP01A80ulhy1Fc2FRAzDyUvKVIzV6U1OgRuLpg3ZJ4yB0dxfsryCoP+Mwn1RjHzgXoykJd0fNQ1hVKzRZHsspz0VOi56xvxNYHwMVc7eYmwNNWt+89tdEJwPEZyRUNw5U1zulg1+Vo1wUb07Qwz/FS0tYnv+IO2mNl29pm4M9c2upf1xbE4Ks2z78rqtxJu6c+379k0F2W/9RPtMkFzTQ5zxO02SwzyRQnx/VI0iAO1P8x1Q6cykdXvvAa0ONp2SAQxJ5u9ski1wyQ8+VR0oJ+2irRxZk11F5B5KE6/NBN3n1h2cdf/tJD1fVNzvZL2izQ35J41+4W0OcP1CQwwt9n7+1F+MpH1Sbzq91J2ENGza8wzRGm0I47ufMzteYZuN5d8sFO0BGqv7Zj9eNRw5Vxtbkzyx41wcdczl9B03g/xy7nyp00x4s68sFiu9Jqte470ztmvqotyvlCv78nzjyS1+dl39Ur7L1n4pc9zuEJ+9w/ts5y0lWgrNQ017dn0RsN7dsdzAI3ubJkvrgpzA9/uc5YzDYeygOLubMp1GpBw+VfzYd62BEizqtqt+ob7t4iyIk7zdkF7+sv3HNVxdlt1f8wzRAPxn1Q3t1y16ts8MoWyR40zu4UzCvOy0xtynw+1wqi1als1Txx2og+1WVC399vy15G0wBz2Ngouc0F4dUO1SDR4rxm1TLF4a1t6e0m0YM93sYuzizOzQ+LveFf0Ved2DnRzxsM46ZxvaFrx4p22DdF+9UyyuNm4tgMt8ohseso1aBw2EEy290KyDkf7L1nx+ILs95Ws8xx8sFY7Lk20LZmvOoXzlk/zEUx25k61Q5+6KhL6blpyg361zxy8sFjyRwMtulJzXiC0FE82NUp3Ngd2LMxz+9Wtu8k0oE+6dhz3Lkx68Jk0KA169oZreJCx3pTrdcXwPEa1OMM2Iw99MRVyCTPuN1f1gLS1gA/2DhT5NFbwuFk4dUs1vge2BYPzkAtrdET25h12DI+6Mtf0AXq0BGrzBX80zTc7+892DY0v5tUzBwMzDMa2vZLy0Ux0SvQ1RH/8+FZ6bVrvOUTyn5+sMsL3eZkzD1m6u5OtMlZ6sBlyRvky1dYyJF00S0x39dxADFmAB150ikuWkww6iiZCC+dKRstbSmULRs2bCeL6hWVACFwMCMYMScQWyN20CYutTszhC2uRSmhDTEHiSJ/CjmB6hReXC2mxRCXGxmIWlBdVCukfCykuEKHUEKjFimmzhnFCxycAxOHBBaSDTskHxZ0aVJavS7Jp0aTrB6mbTCpU0156CXWmkt5BjGYbCVizi7KZyVNIR+axD1oNyOXkC0kGBiGFSykDT5nhDoWohxu3ziWoSSvfjW1SiiegywmlSy4szotAC5PAxudb0cXSzetV0kiDRFgTDYKCReRXSmbrSS6aykygyN13Ro75jaZwhGakzgaliGWdUESIiSjNxxrMxxBLCKbJzaiph+hgCNrjEs73xk8AjkqoSa0XSJtRi2oZy0gFjWaLiKZzja2oDTBWkcbiiRJfDQYCSEDriO4BhF55zNVZS2lLjCpay0jzyQuUi8T0SQvtjcrJzYIwT9wARE7oR5SnSkqZCyiXFBrQ0Kb6iTFKigKlzK/jEIdvD5KlR9mr0E82Q+pWFBajyWfBhkPGiumTimfTikcbUyYyBZFbjGrYU0xSx9ptTi86ipZsBhywDU2U0OiKR+VjE1BRCsULDIF4TdmYkYXISMNuCA0pSUv6ix/hTC0hSN66Bs3uz1GaCV5ABWCJy2oOiSZChsOjy63ryMybEsgaD8PYDkOHTGkZ0ISiDy4sTTEKTSmISelUyod2ijSOR17AjcjCS+cZi4emUUvAykAqjq95zVkqhp0CCak2DuJRyB3RC2oT0QXRiSUziQuWTAV3Q1tYjIUaTKtKkZlkSc0FS4DuR42FEBek0OvvD5PHx4ZVTmuaEgaPSihjjy6e0KyREeK0jtb5i7EOjgIxCwsnUqBPEATWUiYRkxZOjepyy4xOyA02BrO1jlWpiq/shtLbDYTEjMIJxhwnx2BKCypT0iRpDcfhVBZJzKmTCF6dD4QMUc92juGSEYdAx2g3ivNVzgMRUWT6S2+HyOibTMYbysv0iDQxy/Ha0mgIisFzhCd4RC9i0yR4SA0UyBmDD1pbDsQIkAj4QxpWSeZPDIIJS0FJx4fHyqnOSsPo0EuvRao4yMzMCoKpCI8qSS2YFBJKEQwYUcaBTeGliM9SCMwsSA7OTapcy8ihEstRTmrg004HTClkCY4aFB/OkplODeonSi3fj612Q+nP0SR4hJMMkhmHj8jL0Y3Qx971TZHPSqmcjYVejwSHS8D6g55GSEMIkRHnj8km0c5ZTIXABh+0BJfgzcW1xVI4Rc/iSU+rinDazGsojfAxz16LkY0qCvBKEImdik6QkUc6RvTWyoiclA1aSueeTAguyYptEOPhiwn2hvQZiiRnzMeRiqltUBFsyvIez21xT2Csh6zHRVZnDUdGycFoS4gcFRbJzKmlCFQWjUOsheCHyYIkUqWdE+SPkCdyD1oXkSlwj6Jczaxzh01LkdaeSZP4DSyMEKLUDeupCcr1hFbWygoIykG3So1zw+OKj4W5AuoKCmmflFLvhm56SXM1Tx0mExdvyTLwSwq3h3UPyKNDykBTSByuSonq0VI6RLFiDYYUE0/pim+yThG0jA0rEhyzR7METx5fkQZfTEelyq2bSg46iVF5Sc3Bh0JmUWmTiBUwjmt5ivM0TDHYk81HRgwcTcSdjEeNkhvtBaCljUbaTUTiz0YFykDeVE+2Dt7xzS8XUwugCoxfEmks0RYQ0p4xjdDxCHNjkgt3yk1fE+ISSB2uT2qQUMX2zDHkiY13TVPtjozWyJwNkKRY04zlEuLyxJaXSweM0dz2RFX5Q9Tzj1+M0KOpj23DRFOmTMdHSGhmTUc3yHWYlFvEDaSoUQzg09ChUkpj0s+2zhYqS/FqxpnyC/GZ0ij1BGumR+RwD5b5C3JzibQHi6myDxjxjxgGS2lwja34Q9VUjauzB/NviPKtCnHzz12qzgkHS6lQzarND0Odk6WcE0mJyAakTi8hUcirTu6GxY3Xjuw2y442DlaSU1cclN2MEEa'
)
# <<< FROZEN END cloud_axe_blob
CLOUD_BLOCK_LO = (-0.14316870272159576, -0.13553525507450104, -0.049996308982372284); CLOUD_BLOCK_SPAN = (0.2984517216682434, 0.26897260546684265, 0.09999998658895493)
CLOUD_BLOCK_COUNTS = (1, 2, 14, 107, 27, 121, 12, 2, 6, 55, 7, 3, 31, 9, 1, 74, 9, 1, 6, 2, 10)
CLOUD_BLOCK_CHECK = (21, 1563, 24966491)  # palette size, blob bytes, weighted checksum
# >>> FROZEN BEGIN cloud_block_blob
CLOUD_BLOCK_BLOB = (
    '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'
)
# <<< FROZEN END cloud_block_blob
CLOUD_LOG_LO = (-0.14670369029045105, -0.04901760071516037, -0.04812190681695938); CLOUD_LOG_SPAN = (0.2946804165840149, 0.09873294830322266, 0.09653322398662567)
CLOUD_LOG_COUNTS = (1, 4, 2, 41, 48, 2, 20, 43, 4, 80, 39, 12, 3, 2, 14, 70, 18, 8, 1, 17, 27, 14, 4, 1, 20, 4, 1)
CLOUD_LOG_CHECK = (27, 1581, 23992096)  # palette size, blob bytes, weighted checksum
# >>> FROZEN BEGIN cloud_log_blob
CLOUD_LOG_BLOB = (
    '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'
)
# <<< FROZEN END cloud_log_blob
# ---- object cloud TEMPLATES (offline from the training asset, see priors/ORIGIN.json): base64 palette + 8-bit xyz in the model frame.
# Do not edit: check_cloud raises if a blob, CHECK or COUNTS is damaged. Use: load_cloud(name) -> (xyz, rgb).
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
CLOUDS = (  # lower-case name -> (LO, SPAN, COUNTS, CHECK, BLOB, resting orientations as xyz Euler, grasp point in the model frame)
    ("axe", (CLOUD_AXE_LO, CLOUD_AXE_SPAN, CLOUD_AXE_COUNTS, CLOUD_AXE_CHECK, CLOUD_AXE_BLOB, ((0, 0, 0), (3.141592653589793, 0, 0)), (-0.2014, -0.0322, 0.0003))),
    ("block", (CLOUD_BLOCK_LO, CLOUD_BLOCK_SPAN, CLOUD_BLOCK_COUNTS, CLOUD_BLOCK_CHECK, CLOUD_BLOCK_BLOB, ((0, 0, 0),), (0.0012, 0.0009, 0.05))),
    ("log", (CLOUD_LOG_LO, CLOUD_LOG_SPAN, CLOUD_LOG_COUNTS, CLOUD_LOG_CHECK, CLOUD_LOG_BLOB, ((0, 0, 0), (0, 1.5707963267948966, 0)), (-0.0011, 0.0007, 0.0004))),
)


def _entry(name):
    return dict(CLOUDS)[name]


def check_cloud(name, counts, blob, check):
    """Raise a clear error when the packed cloud is damaged (truncated, edited, or COUNTS not matching the point count)."""
    n_colours, n_bytes, expected = check
    data = np.concatenate([blob, (counts % 256).astype(np.uint8), (counts // 256).astype(np.uint8)]).astype(np.uint64)
    got = int((data * (np.arange(len(data), dtype=np.uint64) % 251 + 1)).sum() % 4294967291)
    if len(blob) != n_bytes or len(counts) != n_colours or counts.sum() * 3 + n_colours * 3 != n_bytes or got != expected:
        raise ValueError(f"cloud {name} is damaged: blob bytes {len(blob)} (expected {n_bytes}), colours {len(counts)} (expected {n_colours}), checksum {got} (expected {expected}).")


def load_cloud(name):
    """(xyz (N,3) metres in the model frame, rgb (N,3) uint8) of a template."""
    lo, span, counts, check, text = _entry(name)[:5]; counts = np.asarray(counts, np.int64)
    codes = np.frombuffer(B64.encode("ascii"), np.uint8); order = np.argsort(codes)
    v = order[np.searchsorted(codes[order], np.frombuffer(text.encode("ascii"), np.uint8))].reshape(-1, 4).astype(np.uint32)
    word = (v[:, 0] << 18) | (v[:, 1] << 12) | (v[:, 2] << 6) | v[:, 3]
    blob = np.stack([word >> 16, (word >> 8) & 255, word & 255], axis=1).astype(np.uint8).reshape(-1); check_cloud(name, counts, blob, check)
    palette = blob[:3 * len(counts)].reshape(-1, 3); q = blob[3 * len(counts):].reshape(-1, 3).astype(float)
    return np.asarray(lo) + q / 255 * np.asarray(span), palette[np.repeat(np.arange(len(counts)), counts)]


# ---- tunables (colours calibrated on rollout snapshots: priv poses projected into the head image; HSV on PIL's 0-255 scale) -----
AXE_HSV = ((0, 22), (110, 256), (90, 256))  # orange handle, median RGB (157, 90, 62)
LOG_HSV = ((6, 22), (45, 111), (70, 256))   # bark, median RGB (131, 103, 91); fence and garage share the hue but stand upright
GROUND_Z = (.004, .14)        # height band of ground objects (m, robot frame)
AXE_MAX_Z = .05               # the axe lies flat (handle top ~0.03, zmax < 2x): log bark tops sit at ~0.08
AXE_LEN = (.25, .60)          # handle cluster length (axe 0.53 m)
LOG_LEN = (.20, .42)          # whole log 0.29 m; halves (0.15) are skipped
LOG_ELONG = 1.5               # length/width: rejects the round chopping block
LOG_MAX_Z = .13
WALL_Z, WALL_R = .25, .30     # a candidate with anything taller than WALL_Z within WALL_R is fence/garage foot
TOUCHED_R = .30               # a chopped spot is skipped within this radius
TRACK_R = .50                 # once locked, only clusters this close to the estimate update it
REACH = (.72, -.18)           # object position in the robot frame for the right-arm reach (still visible to the head)
NEAR = 1.6                    # within this range the heading puts the object at REACH bearing
ARRIVE_POS, ARRIVE_YAW = .05, .06
GO_SLACK_AGE, GO_SLACK_POS = 900, .15   # accept a looser stand-off after this many steps
GO_GIVE_UP = 1300             # per approach attempt
GO_RETRIES = 3                # alternate approach headings before final skip
LOST_AGE, LOST_DIST = 450, .85
SCAN_W, SCAN_ARC = .4, 6.1    # in-place scan: yaw rate and arc (rad) before a leg is chosen
NBIN = 12                     # heading bins of the scan clearance map
CORRIDOR, CLEAR_MAX, CLEAR_Z = .40, 3., .65   # leg corridor half-width, range cap, top of the clearance band (carried axe sits at ~0.78)
LEG_MAX, LEG_MIN, LEG_MARGIN, LEG_STOP, LEG_V = 2.2, .4, 1.2, .9, .25   # leg length cap and floor, stand-off kept from an obstacle, stop range, speed
FAR_R = 6.                    # legs ending farther than this from the start are penalised
LOOSE_AFTER = 4               # scans without an axe before the axe filters are relaxed
AXE_LEN_LOOSE, WALL_R_LOOSE = (.18, .65), .18
BAD_R = .35                   # a spot with two failed grasps is not detected as an axe again
MEM_R, MEM_NEAR, MEM_AGE = 1., 1.9, 600   # remembered log: first-sighting gate, range at which it must be seen, drive budget
EMPTY_FINGERS, EMPTY_STEPS = .002, 30     # finger qpos sum of an empty closed hand, steps before the axe counts as lost
SETTLE = 30                   # steps standing still to refine the estimate before reaching
PRE_GRASP_Z, GRASP_Z = .15, .008
CLOSE_STEPS, LIFT_Z = 45, .25
HELD_FINGERS = (.006, .09)    # finger qpos sum of a closed hand around the handle (open 0.10, empty 0.0)
CHOP_PRE_Z, CHOP_PRESS = .22, .06  # above the bark top, then press this far below it (contact stops the arm)
HOLD_STEPS, UP_MIN = 20, 60   # 60 steps = 2 s off the log: the slicer is active again
STILL_GRASP, STILL_CHOP = 25, 12   # steps without tip motion that count as contact
TORSO_HOME = (.70, -.95, -.35, 0.)
ARM_REST = (-1.6, 0., -.7, .1, -.5, .3, -.6)
CARRY_TIP = (.42, -.30, .78)  # torso at home puts the shoulder at 1.34 m: a downward tip is only reachable above ~0.7 m
IK_ARM_STARTS = ((-1., 0., 0., -1.5, 0., 0., 0.), (-.5, -.5, 0., -1.5, 0., 0., 0.), (-1.2, -.3, .5, -1.2, 0., -.5, 0.))
PHASES = ("search", "go_axe", "grasp", "find_log", "go_log", "chop")
HEAD = "robot_r1::robot_r1:zed_link:Camera:0::"


def hsv255(rgb):
    c = rgb.astype(float); mx, mn = c.max(-1), c.min(-1); d = np.maximum(mx - mn, 1e-6); r, g, b = c[..., 0], c[..., 1], c[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6 * 255
    return np.where(mx - mn < 1e-6, 0, h), np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6) * 255, 0), mx


def ground_clusters(cam, band, min_n, ext, min_elong=0., hsv=None):
    """Ground-band clusters of band-coloured head pixels: dicts c (xy), z, zmax, axis, len, n. hsv: optional precomputed hsv255 of the image."""
    h, s, v = hsv if hsv is not None else hsv255(cam["rgb"]); m = np.ones(h.shape, bool)
    for x, (lo, hi) in zip((h, s, v), band): m &= (x >= lo) & (x < hi)
    p, _, _ = pointcloud(cam, m); p = p[(p[:, 2] > GROUND_Z[0]) & (p[:, 2] < GROUND_Z[1]) & (np.linalg.norm(p[:, :2], axis=1) < 6)]
    if len(p) < min_n: return []
    ij = np.floor(p[:, :2] / .04).astype(int); ij -= ij.min(0); grid = np.zeros(ij.max(0) + 1, bool); grid[ij[:, 0], ij[:, 1]] = True
    lab, n = label(dilate(grid, 2), structure=np.ones((3, 3))); ids = lab[ij[:, 0], ij[:, 1]]; out = []
    for k in range(1, n + 1):
        q = p[ids == k]
        if len(q) < min_n: continue
        c = q[:, :2].mean(0); vec = np.linalg.eigh(np.cov((q[:, :2] - c).T) + 1e-9 * np.eye(2))[1]; axis = vec[:, 1]
        proj = (q[:, :2] - c) @ axis; length = proj.max() - proj.min(); width = np.ptp((q[:, :2] - c) @ vec[:, 0])
        if ext[0] <= length <= ext[1] and length >= min_elong * max(width, .02):
            out.append({"c": c, "z": float(np.median(q[:, 2])), "zmax": float(np.quantile(q[:, 2], .95)), "axis": axis, "len": float(length), "n": len(q)})
    return out


def head_cam(frame, st=2):
    """Head camera at pixel stride st (2 = 360 px, 1 = full 720 px); intrinsics follow the image width."""
    rel = frame["robot_r1::cam_rel_poses"].reshape(3, 7); rgb, d = frame[HEAD + "rgb"][::st, ::st, :3], np.squeeze(frame[HEAD + "depth_linear"])[::st, ::st]; w = d.shape[0]
    return {"rgb": rgb, "depth": d, "K": np.array([[.425 * w, 0, w / 2], [0, .425 * w, w / 2], [0, 0, 1.]]), "R": R.from_quat(rel[2, 3:]).as_matrix() @ np.diag([1, -1, -1]), "t": rel[2, :3]}


class Planner:
    """Take the axe in the right hand, then touch every log with it."""
    def reset(self, task_spec):
        for name in ("axe", "block", "log"): load_cloud(name)  # raises if a template is damaged
        self.kin = Kinematics(); self.lim = np.asarray(LIMITS); self.phase, self.sub, self.reason = "search", "", "start"; self.step = self.age = self.sub_age = 0
        self.odom = np.zeros(3); self.arm_action = self.obj = self.obj_h = self.plan = self.goal = self.target = self.last_tip = None; self.desired = np.eye(3)
        self.touched, self.events, self.debug = [], [], {}; self.chops = self.grasp_tries = self.still = self.search_legs = self.skipped = 0
        self.logs, self.visited, self.bad_axe, self.bad_h, self.clear = [], [], [], [], {}; self.confirmed, self.held = True, False
        self.clear_now, self.turned, self.dyaw, self.hd, self.leg_len = CLEAR_MAX, 0., 0., 0., 0.; self.leg0 = np.zeros(2); self.empty = self.lost_ax = 0
        self.go_retries, self.go_offset = 0, 0.
    def log(self):
        return {"phase": self.phase, "sub": self.sub, "stage": PHASES.index(self.phase), "reason": self.reason, "step": self.step, "chops": self.chops,
                "grasp_tries": self.grasp_tries, "skipped": self.skipped, "search_legs": self.search_legs, "odom": self.odom.round(3).tolist(),
                "scans": len(self.visited), "mem_logs": len(self.logs), "bad_axe": len(self.bad_axe), "lost_ax": self.lost_ax,
                "obj": None if self.obj is None else np.round(self.obj, 3).tolist(), "debug": self.debug, "events": self.events[-30:]}
    def advance(self, phase, why, sub=""):
        self.events = self.events[-29:] + ["%d:%s.%s:%s" % (self.step, phase, sub, why)]; self.phase, self.sub, self.reason, self.plan = phase, sub, why, None; self.age = self.sub_age = 0
        if phase in ("search", "find_log"): self.obj = None
        if phase in ("go_axe", "go_log"):
            self.obj_h = None; self.go_retries = 0; self.go_offset = 0.
    def to_sub(self, sub, why=""):
        self.events = self.events[-29:] + ["%d:%s.%s" % (self.step, self.phase, sub)]; self.sub, self.reason, self.sub_age, self.still = sub, why or sub, 0, 0
    def to_odom(self, p):
        return rot2(self.odom[2]) @ p[:2] + self.odom[:2]
    def to_robot(self, p):
        return rot2(-self.odom[2]) @ (np.asarray(p[:2]) - self.odom[:2])
    def tip_of(self, q, cache=None):  # fingers at zero: TIP_POINT is on finger 2, 5 cm off the grasp centre when open
        if cache is None: q = q.copy(); q[11:13] = 0.
        pose = (cache or self.kin.forward(q))[0][TIP_LINK]
        return pose, pose[:3, 3] + pose[:3, :3] @ np.asarray(TIP_POINT)
    # ---- arm control (radio iter15 pattern)
    def solve_ik(self, q, goal, desired, torso=None):
        """Joint target (11) for tip `goal`, orientation `desired`, torso fixed if given; None if unreachable."""
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
        step = np.clip(target - q[:11], -rate, rate); action[3:7], action[15:22] = q[:4] + step[:4], q[4:11] + step[4:]
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
    def carry_plan(self, q):
        self.plan = self.solve_ik(q, np.asarray(CARRY_TIP), self.desired, TORSO_HOME)
    # ---- perception
    def banned(self, c):
        """True when two failed grasps or give-ups were logged within BAD_R of this axe candidate (odometry frame)."""
        p = self.to_odom(c["c"]); return sum(float(np.linalg.norm(p - b)) < BAD_R for b in self.bad_axe) >= 2
    def observe(self, frame, kinds, st):
        """Head detections {kind: clusters, nearest first} at pixel stride st; every log seen is remembered; clear_now = free range ahead of the base."""
        cam = head_cam(frame, st); hsv = hsv255(cam["rgb"]); tall, _, _ = pointcloud(cam, stride=3 if st == 2 else 6); tall = tall[(tall[:, 2] > WALL_Z) & (tall[:, 2] < 2.)]
        band = tall[(tall[:, 2] < CLEAR_Z) & (tall[:, 0] > .55) & (np.abs(tall[:, 1]) < CORRIDOR)]
        self.clear_now = float(min(CLEAR_MAX, band[:, 0].min())) if len(band) else CLEAR_MAX
        loose = len(self.visited) >= LOOSE_AFTER; out = {}
        free = lambda c, r: not (np.linalg.norm(tall[:, :2] - c["c"], axis=1) < r).any()
        if "axe" in kinds:
            out["axe"] = sorted([c for c in ground_clusters(cam, AXE_HSV, 6, AXE_LEN_LOOSE if loose else AXE_LEN, 0., hsv) if c["z"] < AXE_MAX_Z and c["zmax"] < 2 * AXE_MAX_Z
                                 and free(c, WALL_R_LOOSE if loose else WALL_R) and not self.banned(c)], key=lambda c: np.linalg.norm(c["c"]))
        if "log" in kinds:
            out["log"] = sorted([c for c in ground_clusters(cam, LOG_HSV, 8, LOG_LEN, LOG_ELONG, hsv) if c["zmax"] < LOG_MAX_Z and free(c, WALL_R)
                                 and all(np.linalg.norm(self.to_odom(c["c"]) - t) > TOUCHED_R for t in self.touched)], key=lambda c: np.linalg.norm(c["c"]))
            for c in out["log"]: self.remember(c)
        return out
    def remember(self, c):
        """Log memory in the odometry frame (x, y, z, axis yaw); a sighting within 0.4 m updates the same entry."""
        ax = rot2(self.odom[2]) @ c["axis"]; e = np.r_[self.to_odom(c["c"]), c["z"], math.atan2(ax[1], ax[0])]
        for i, m in enumerate(self.logs):
            if np.linalg.norm(m[:2] - e[:2]) < .4: self.logs[i] = e; return
        self.logs.append(e)
    def recall(self):
        """Nearest remembered log that is not a chopped spot, else None."""
        self.logs = [m for m in self.logs if all(np.linalg.norm(m[:2] - t) > TOUCHED_R for t in self.touched)]
        return min(self.logs, key=lambda m: np.linalg.norm(m[:2] - self.odom[:2])) if self.logs else None
    def forget(self, p):
        """Drop remembered logs within 0.6 m of p (stale sighting)."""
        self.logs = [m for m in self.logs if np.linalg.norm(m[:2] - p[:2]) > .6]
    def lock(self, c):
        """Set obj (odometry frame: x, y, z, axis yaw) from a cluster; the target counts as seen."""
        ax = rot2(self.odom[2]) @ c["axis"]; self.obj = np.r_[self.to_odom(c["c"]), c["z"], math.atan2(ax[1], ax[0])]; self.confirmed = True
    def track(self, frame, kind):
        """Refresh obj from the head view; a remembered, not yet seen target uses the wider MEM_R gate."""
        cl = self.observe(frame, (kind,), 2)[kind]
        if self.obj is not None: cl = [c for c in cl if np.linalg.norm(self.to_odom(c["c"]) - self.obj[:2]) < (TRACK_R if self.confirmed else MEM_R)]
        if not cl: return False
        self.lock(cl[0]); return True
    def object_robot(self):
        """Robot-frame (x, y, z) and axis yaw in [-pi/2, pi/2] (the grasp yaw)."""
        yaw = wrap(self.obj[3] - self.odom[2]); yaw = yaw - math.pi if yaw > math.pi / 2 else yaw + math.pi if yaw < -math.pi / 2 else yaw
        return np.r_[self.to_robot(self.obj), self.obj[2]], yaw
    # ---- phases
    def act(self, obs):
        frame = {"robot_r1::proprio": first(obs["robot_r1::proprio"]), "robot_r1::cam_rel_poses": first(obs["robot_r1::cam_rel_poses"]),
                 HEAD + "rgb": first(obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"]), HEAD + "depth_linear": first(obs["robot_r1::robot_r1:zed_link:Camera:0::depth_linear"])}
        s = frame["robot_r1::proprio"]; q = from_proprio(s); self.step += 1; self.age += 1; self.sub_age += 1
        self.dyaw = 0.
        if self.step > 1:
            self.dyaw = float(s[2]) / 30; self.odom[:2] += rot2(self.odom[2]) @ s[:2] / 30; self.odom[2] += self.dyaw
        if self.arm_action is None: self.arm_action = np.r_[np.zeros(3), s[53:57], s[3:10], 1., s[28:35], 1.]
        action = self.arm_action.copy(); action[:3] = 0; self.debug = {"fingers": round(float(s[49] + s[50]), 4)}; look = self.step % (8 if self.phase == "search" else 5) == 1
        f = float(s[49] + s[50]); self.empty = self.empty + 1 if self.phase in ("find_log", "go_log", "chop") and f < EMPTY_FINGERS else 0
        if self.empty > EMPTY_STEPS:  # closed on nothing: the axe slipped; stow the arm (grasp.carry), then search for it again
            self.empty = 0; self.lost_ax += 1; self.held = False; self.advance("grasp", "axe lost from the hand, fingers %.3f" % f, "carry"); self.carry_plan(q)
        if self.phase in ("search", "find_log"): self.search(action, frame, look)
        elif self.phase in ("go_axe", "go_log"): self.go(action, frame, look)
        elif self.phase == "grasp": self.grasp(action, frame, q, s, look)
        else: self.chop(action, frame, q, look)
        self.arm_action = action.copy()
        assert np.isfinite(action).all()
        return action.astype(np.float32)
    def search(self, action, frame, look):
        """Scan 360 deg in place (free range per heading bin, full-res ax and log detection), aim at the freest least-visited heading, drive a leg, repeat. Remembered logs skip the scan."""
        kind = "axe" if self.phase == "search" else "log"; action[3:7] += np.clip(np.asarray(TORSO_HOME) - action[3:7], -.008, .008)
        if kind == "axe": action[22] = 1.
        if self.sub == "" and kind == "log":
            m = self.recall()
            if m is not None:
                self.obj, self.confirmed = m.copy(), False
                return self.advance("go_log", "remembered log at " + str(np.round(m[:2], 2).tolist()))
        if self.sub == "":
            self.visited.append(self.odom[:2].copy()); self.clear = {}; self.turned = 0.; self.to_sub("scan", "scanning for " + kind)
        if look:
            det = self.observe(frame, ("axe", "log") if kind == "axe" else ("log",), 1 if kind == "axe" else 2)
            if self.sub == "scan":
                b = int(np.round(wrap(self.odom[2]) / (2 * math.pi / NBIN))) % NBIN; self.clear[b] = min(self.clear.get(b, CLEAR_MAX), self.clear_now)
            if det[kind]:
                self.lock(det[kind][0]); return self.advance("go_" + kind, "%s seen at %s" % (kind, np.round(self.obj, 2).tolist()))
        if self.sub == "scan":
            action[2] = SCAN_W; self.turned += abs(self.dyaw)
            if self.turned > SCAN_ARC or self.sub_age > 900 or (self.sub_age > 180 and self.turned < .15): self.plan_leg()
        elif self.sub == "aim":
            dh = wrap(self.hd - self.odom[2]); action[2] = float(np.clip(dh * 1.5, -.4, .4))
            if abs(dh) < .08 or self.sub_age > 400: self.leg0 = self.odom[:2].copy(); self.to_sub("leg", "forward leg of %.1f m" % self.leg_len)
        else:
            dh = wrap(self.hd - self.odom[2]); action[0] = LEG_V; action[2] = float(np.clip(dh * 1.5, -.3, .3)); moved = float(np.linalg.norm(self.odom[:2] - self.leg0))
            if moved > self.leg_len or (look and self.clear_now < LEG_STOP and len(self.bad_h) < 2) or self.sub_age > 400:
                self.bad_h = self.bad_h + [self.hd] if moved < .2 else []; self.search_legs += 1; self.advance(self.phase, "leg done, moved %.1f m" % moved)
    def plan_leg(self):
        """Score clear, novel headings and choose an unblocked fallback when all candidates are short."""
        best = None; step = 2 * math.pi / NBIN
        for b in range(NBIN):
            h = b * step; length = min(LEG_MAX, (CLEAR_MAX if len(self.bad_h) >= 2 else self.clear.get(b, 0.)) - LEG_MARGIN)
            if length < LEG_MIN or any(abs(wrap(h - x)) < .5 for x in self.bad_h): continue
            p = self.odom[:2] + length * np.array([math.cos(h), math.sin(h)])
            score = min([float(np.linalg.norm(p - v)) for v in self.visited] + [3.]) + .5 * length - .3 * max(0., float(np.linalg.norm(p)) - FAR_R)
            if best is None or score > best[0]: best = (score, h, length)
        if best is None:
            choices = [k for k in range(NBIN) if not any(abs(wrap(k * step - x)) < .5 for x in self.bad_h)]
            if not choices:
                self.bad_h = []; choices = list(range(NBIN))
            k = max(choices, key=lambda j: self.clear.get(j, CLEAR_MAX))
            length = min(LEG_MAX, max(LEG_MIN, self.clear.get(k, CLEAR_MAX) - LEG_MARGIN))
            best = (0., k * step, length)
        self.hd, self.leg_len = best[1], best[2]; self.to_sub("aim", "aiming %.0f deg for a %.1f m leg" % (math.degrees(wrap(self.hd - self.odom[2])), self.leg_len))
    def go(self, action, frame, look):
        """Drive to a continuously recomputed stand-off pose, retrying blocked approaches from alternate headings."""
        kind = self.phase[3:]; back = "search" if kind == "axe" else "find_log"; seen = look and self.track(frame, kind)
        rb = self.to_robot(self.obj); dist = float(np.linalg.norm(rb))
        if not self.confirmed and self.age > 6 and (self.age > MEM_AGE or (dist < MEM_NEAR and abs(math.atan2(rb[1], rb[0])) < .35)):
            self.forget(self.obj); return self.advance(back, "remembered %s not found" % kind)  # facing it within range and still unseen: stale memory
        if look and not seen and self.age > LOST_AGE and dist > LOST_DIST:
            return self.advance(back, "%s lost from view" % kind)
        if self.age > GO_GIVE_UP:
            if self.go_retries < GO_RETRIES:
                self.go_retries += 1
                self.age = self.sub_age = 0
                self.go_offset = (math.pi / 2 if self.go_retries == 1 else
                                  -math.pi / 2 if self.go_retries == 2 else math.pi)
                self.reason = "replanning approach %d" % self.go_retries
                action[0] = -.18; action[1] = 0.
                action[2] = -.35 if self.go_retries == 1 else .35
                return
            p = self.obj[:2].copy(); self.skipped += 1
            if kind == "axe": self.bad_axe += [p, p]
            else: self.touched.append(p)
            return self.advance(back, "%s unreachable after retries, skipped" % kind)
        if kind == "log": action[3:7] += np.clip(np.asarray(TORSO_HOME) - action[3:7], -.008, .008)
        d = self.obj[:2] - self.odom[:2]
        dist = float(np.linalg.norm(d))
        off = math.atan2(REACH[1], REACH[0]) if dist < NEAR else 0.
        self.obj_h = math.atan2(d[1], d[0]) - off + self.go_offset
        local = rot2(-self.odom[2]) @ (self.obj[:2] - rot2(self.obj_h) @ np.asarray(REACH) - self.odom[:2]); dh = wrap(self.obj_h - self.odom[2]); dpos = float(np.linalg.norm(local))
        action[:2] = np.clip(local * 1.2, -.25, .25) * (.3 if dpos > .5 and abs(dh) > .5 else 1.); action[2] = np.clip(dh * 1.5, -.4, .4)
        self.reason = "driving, %.2f m and %.2f rad to the stand-off" % (dpos, abs(dh)); self.debug["rel"] = np.round(self.to_robot(self.obj), 3).tolist()
        if (dpos < ARRIVE_POS and abs(dh) < ARRIVE_YAW) or (self.age > GO_SLACK_AGE and dpos < GO_SLACK_POS):
            action[:3] = 0; self.advance("grasp" if kind == "axe" else "chop", "arrived, %s at %s" % (kind, self.debug["rel"]), "settle")
    def grasp(self, action, frame, q, s, look):
        """Top-down right-hand grasp of the handle centre: settle, pre (joint plan above), down (servo), close, lift, check, carry."""
        action[22] = 1. if self.sub in ("settle", "pre", "down") else -1.
        if self.sub == "settle":
            if look: self.track(frame, "axe")
            if self.sub_age > SETTLE:
                p, yaw = self.object_robot(); self.desired = R.from_euler("z", yaw).as_matrix(); self.goal = p + np.array([0, 0, PRE_GRASP_Z])
                self.plan = self.solve_ik(q, self.goal, self.desired); self.to_sub("pre", "moving above the handle")
                if self.plan is None: self.bad_axe.append(self.obj[:2].copy()); self.advance("search", "handle out of reach at %s" % p.round(2).tolist())
        elif self.sub == "pre":
            if self.joint_move(action, q, self.plan) < .01 or self.sub_age > 400:
                p, _ = self.object_robot(); self.goal = np.r_[p[:2], GRASP_Z]; self.to_sub("down", "lowering onto the handle")
        elif self.sub == "down":
            err, _ = self.servo(action, q, self.goal, self.desired, .0025)
            if err < .008 or self.still > STILL_GRASP or self.sub_age > 300: self.to_sub("close", "closing the hand")
        elif self.sub == "close":
            self.servo(action, q, self.goal, self.desired, .0015)
            if self.sub_age > CLOSE_STEPS: self.goal = self.tip_of(q)[1] + np.array([0, 0, LIFT_Z]); self.to_sub("lift", "lifting")
        elif self.sub == "lift":
            err, _ = self.servo(action, q, self.goal, self.desired)
            if err < .02 or self.sub_age > 250:
                f = float(s[49] + s[50])
                if HELD_FINGERS[0] < f < HELD_FINGERS[1]: self.held = True; self.carry_plan(q); self.to_sub("carry", "axe in hand, fingers %.3f" % f)
                else: self.grasp_tries += 1; self.held = False; self.bad_axe.append(self.obj[:2].copy()); self.carry_plan(q); self.to_sub("carry", "grasp missed, fingers %.3f (try %d), stowing" % (f, self.grasp_tries))
        elif self.plan is None or self.joint_move(action, q, self.plan, .015) < .01 or self.sub_age > 400:
            self.advance("find_log" if self.held else "search", "axe held, looking for logs" if self.held else "arm stowed, no axe in hand")
    def chop(self, action, frame, q, look):
        """Lower the held axe onto the log: settle, pre (joint plan above), down (servo through the bark top), hold, up (>= 2 s), carry."""
        action[22] = -1.
        if self.sub == "settle":
            if look: self.track(frame, "log")
            if self.sub_age > SETTLE:
                self.target = self.object_robot()[0]; self.goal = self.target + np.array([0, 0, CHOP_PRE_Z]); self.plan = self.solve_ik(q, self.goal, self.desired)
                self.to_sub("pre", "moving the axe above the log" if self.plan is not None else "no joint plan above the log, servoing")
        elif self.sub == "pre":
            if self.plan is None or self.joint_move(action, q, self.plan) < .01 or self.sub_age > 400:
                self.goal = np.r_[self.target[:2], max(self.target[2] - CHOP_PRESS, .02)]; self.to_sub("down", "pressing the axe onto the log")
        elif self.sub == "down":
            err, _ = self.servo(action, q, self.goal, self.desired, .004)
            if err < .01 or self.still > STILL_CHOP or self.sub_age > 250: self.to_sub("hold", "axe on the log (contact)" if self.still > STILL_CHOP else "axe at depth")
        elif self.sub == "hold":
            self.servo(action, q, self.goal, self.desired, .002)
            if self.sub_age > HOLD_STEPS: self.goal = self.tip_of(q)[1] + np.array([0, 0, LIFT_Z]); self.to_sub("up", "lifting, slicer reactivation wait")
        elif self.sub == "up":
            err, _ = self.servo(action, q, self.goal, self.desired)
            if (err < .02 and self.sub_age > UP_MIN) or self.sub_age > 200:
                self.chops += 1; self.touched.append(self.obj[:2].copy()); self.carry_plan(q); self.to_sub("carry", "chop %d done" % self.chops)
        elif self.plan is None or self.joint_move(action, q, self.plan, .015) < .01 or self.sub_age > 300:
            self.advance("find_log", "chop %d done, next log" % self.chops)


def entrypoint():
    return Planner()
