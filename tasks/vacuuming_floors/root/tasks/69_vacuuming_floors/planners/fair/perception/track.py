"""Tracking of the marker, the tilt of the held vacuum and the dust memory for the fair planner of task 69 (functions of the Perception object p)."""
from __future__ import annotations

import numpy as np

from core.skills_lib import rgbd_geometry as G

from .. import models as MD
from .. import params as P
from ..world import Dust
from . import cloud as C
from .cloud import odom_T
from . import dust as D
from . import marker as M


def marker(p, rgbd, cam_rel, base, vac_odom, q) -> None:
    T_robot = np.linalg.inv(odom_T(base)) @ vac_odom
    pred = T_robot[:3, :3] @ MD.MARKER + T_robot[:3, 3]
    best = None
    window = P.MARKER_WINDOW_M if p.held is not None else P.MARKER_STAND_WINDOW_M      # a fitted standing vacuum is off by centimetres
    for cam in ('head', 'left_wrist', 'right_wrist'):
        v = M.look(rgbd[cam][0], rgbd[cam][1], cam, cam_rel, pred, window_m=window)
        if v is not None and v['state'] is not None and (best is None or v['red'] + v['green'] > best['red'] + best['green']):
            best = v
    mk = p.marker
    if best is None:
        if p.held is not None and mk.get('state') == 'on':
            tilt_check(p, rgbd, cam_rel, pred)
        return
    p.marker_lost = 0
    mk['seen'] += 1
    mk['last'] = dict(state=best['state'], cam=best['cam'], red=best['red'], green=best['green'], step=p.step)
    if best['state'] == mk['pending']:
        mk['votes'] += 1
    else:
        mk['pending'], mk['votes'] = best['state'], 1
    if best['state'] == 'on' and mk['votes'] >= 2:
        mk['state'], mk['step'] = 'on', p.step                       # an on reading is believed at once (a press toggles once)
    elif best['state'] == 'off' and mk['votes'] >= (P.OFF_VOTES if mk['state'] == 'on' else P.MARKER_VOTES):
        mk['state'], mk['step'] = 'off', p.step
    elif mk['state'] is None:
        mk['state'], mk['step'] = best['state'], p.step
    if p.held is None and p.vacuum is not None and best['centre'] is not None and P.MARKER_STAND_FIX:
        delta = best['centre'] - pred                      # the model fit is good to centimetres, the marker blob to millimetres
        if np.linalg.norm(delta) < P.MARKER_STAND_WINDOW_M:
            T = p.vacuum.T.copy()
            T[:2, 3] += (odom_T(base)[:2, :2] @ delta[:2])
            if abs(delta[2]) < 0.03:
                T[2, 3] += delta[2] * P.MARKER_Z_GAIN
            p.vacuum.T = T
            p.counts['marker_fix_stand'] = p.counts.get('marker_fix_stand', 0) + 1
            mk['fix_stand'] = np.round(delta, 4).tolist()
    if (p.marker_fix and p.held is not None and (best['cam'] == 'head' or P.MARKER_FIX_WRIST) and best['centre'] is not None
            and np.linalg.norm(best['centre'] - pred) < P.MARKER_REFINE_M):
        delta = best['centre'] - pred                      # robot frame: the held pose was off by this at the marker
        Tr = np.eye(4)
        Tr[:3, 3] = 0.5 * delta
        fixed = odom_T(base) @ Tr @ T_robot
        p.held['T_mid_vac'] = np.linalg.inv(p.mid_T(q, p.held['side'])) @ fixed
        p.counts['marker_fix'] += 1
        mk['fix'] = np.round(delta, 4).tolist()


def tilt_check(p, rgbd, cam_rel, pred) -> None:
    """The marker is not at the place predicted from the hand.

    The marker is not at the place predicted from the hand: look around it in the head image; a green marker found far from the
    prediction means the held vacuum has turned in the hand or the arm has given way (it caught on something)."""
    v = M.look(rgbd['head'][0], rgbd['head'][1], 'head', cam_rel, pred, window_m=P.MARKER_WIDE_M, green_only=True)
    if v is not None and v['state'] is not None and v['centre'] is not None and np.linalg.norm(v['centre'] - pred) > P.TILT_OFFSET_M:
        p.tilt_alarm = p.step
        p.marker['tilt_offset'] = round(float(np.linalg.norm(v['centre'] - pred)), 3)
        return
    # no green marker anywhere near: when nothing stands in front of the predicted place (the free arm, the holding arm) the vacuum is not there
    pix, dep = G.project(np.asarray(pred, float)[None], 'head', cam_rel)
    r, c = int(round(pix[0, 0])), int(round(pix[0, 1]))
    if dep[0] > 0.3 and P.DUST_EDGE_PX <= r < 720 - P.DUST_EDGE_PX and P.DUST_EDGE_PX <= c < 720 - P.DUST_EDGE_PX:
        seen = float(np.median(np.asarray(rgbd['head'][1], float)[r - 2:r + 3, c - 2:c + 3]))
        if (not np.isfinite(seen)) or seen > dep[0] - P.MARKER_FREE_TOL_M:
            p.marker_lost += 1
            if p.marker_lost >= P.MARKER_LOST_UPDATES:
                p.tilt_alarm = p.step
                p.marker['tilt_offset'] = 'lost'
            return
    p.marker_lost = 0


def relook(p, d) -> None:
    """Floor near flake d counts as not looked at again."""
    if P.GONE_RELOOK_M <= 0:
        return
    near = np.argwhere(p.floor_seen)
    xy = near * P.OBST_RES_M - P.OBST_HALF_M + 0.5 * P.OBST_RES_M
    cells = near[np.hypot(xy[:, 0] - d.xy[0], xy[:, 1] - d.xy[1]) < P.GONE_RELOOK_M]
    p.dust_cov[cells[:, 0], cells[:, 1]] = False
    p.counts['relook'] = p.counts.get('relook', 0) + 1


def dust(p, head, cam_rel, q, base, held_robot) -> None:
    rgb, depth = head

    def not_self(pts: np.ndarray) -> np.ndarray:
        ok = C.self_keep(pts, q, p.floor_z)
        if held_robot is not None:
            ok &= C.held_keep(pts, held_robot, 0.10)
        if p.vacuum is not None:
            Tr = np.linalg.inv(odom_T(base)) @ p.vacuum.T
            ok &= C.held_keep(pts, Tr, 0.10)
        return ok

    blobs = D.dust_blobs(rgb, depth, 'head', cam_rel, p.floor_z, ignore=not_self)
    seen = set()
    for b in blobs:
        xy = G.robot_to_odom(b['centre'][None], base)[0, :2]
        p.counts['dust_seen'] += 1
        near = [(float(np.linalg.norm(d.xy - xy)), i) for i, d in enumerate(p.dust) if not d.gone]
        near = [t for t in near if t[0] < P.DUST_GATE_M]
        if near:
            i = min(near)[1]
            d = p.dust[i]
            d.xy, d.n, d.seen_step, d.misses = xy, d.n + 1, p.step, 0
            if d.visit_step >= 0 and p.step > d.visit_step + 5:
                d.reseen += 1
            seen.add(i)
        else:
            p.dust.append(Dust(xy=xy, n=1, seen_step=p.step, pid=p.next_pid))
            p.next_pid += 1
            p.counts['dust_new'] += 1
    # absence: a remembered flake whose floor spot is in view (nothing nearer in front of it) and shows no blob
    for i, d in enumerate(p.dust):
        if d.gone or i in seen or d.n < P.DUST_CONFIRM:
            continue
        pr = G.odom_to_robot(np.array([[d.xy[0], d.xy[1], p.floor_z]]), base)
        pix, dep = G.project(pr, 'head', cam_rel)
        r, c = int(round(pix[0, 0])), int(round(pix[0, 1]))
        if dep[0] < 0.5 or not (P.DUST_EDGE_PX <= r < 720 - P.DUST_EDGE_PX and P.DUST_EDGE_PX <= c < 720 - P.DUST_EDGE_PX):
            continue
        seen_d = depth[r - 2:r + 3, c - 2:c + 3]
        if not np.all(np.isfinite(seen_d)) or abs(float(np.median(seen_d)) - float(dep[0])) > P.DUST_VISIBLE_DEPTH_TOL_M:
            continue                               # something in front of the spot (the robot, the vacuum head): no evidence
        d.misses = getattr(d, 'misses', 0) + 1
        if d.misses >= P.DUST_GONE_MISSES:
            d.gone = True
            p.counts['dust_gone'] += 1
            if d.visit_step < 0:      # gone without a pass over it: it was placed wrong (odometry drift), look at that floor again
                relook(p, d)

