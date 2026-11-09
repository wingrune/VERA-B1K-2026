"""Pinch planning of the fair planner of task 69: the pinch of the handle loop from its -x side (fingertip midpoint DEEP_M inside the face) on
the fitted vacuum model, the stands that reach it (GPU reachable_stands masks, both arms, rings around the handle), the free hand's press of a
handle face from the same stand (GPU masks first, then an arm-only solve with the trunk the grasp will have), and the candidates' costs."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import models as MD
from .. import params as P

PRESS_TRUNK_COST = 1.0


def frames_of(vac_T: np.ndarray) -> List[Dict[str, Any]]:
    return G.pinch_frames(vac_T, G.handle_point(P.DEEP_M), P.PITCHES, from_side=-1.0, yaws=P.APPROACH_YAWS, rolls=P.ROLLS)


def press_scan(vac_T, stands, faces, base_z=0.0) -> Dict[str, Any]:
    """GPU reachability of every press face.

    GPU reachability of every press face (pre-press and full push poses) from every stand, for both arms."""
    goals = []
    for f in faces:
        for off in (P.PRESS_PRE_M, f['final']):
            fr = G.press_frame(vac_T, f['point'], f['normal'], pre_m=off, close_body=f['close'])
            goals.append(G.pose_T(fr['rot'], fr['pre']))
    out = dict(faces=faces)
    for arm in ('left', 'right'):
        out[arm] = ik.scan(arm, goals, stands, base_z)
    return out


def press_pick(press, arm, i, q_hold) -> Optional[Tuple[Dict[str, Any], float]]:
    """The press face the free arm reaches from stand i.

    The press face the free arm reaches from stand i (pre-press and full push) with the trunk closest to the holding solution's."""
    masks, qs = press[arm]
    best = None
    for k, f in enumerate(press['faces']):
        if not (masks[2 * k, i] and masks[2 * k + 1, i]):
            continue
        d = float(np.max(np.abs(qs[2 * k + 1, i][:4] - q_hold[:4])))
        if best is None or d < best[1]:
            best = (f, d)
    if best is None or best[1] > P.TRUNK_MAX:
        return None
    return best


def candidates(c, w, q, stands=None, rings=True) -> List[Dict[str, Any]]:
    """Grasp candidates (stand, arm.

    Grasp candidates (stand, arm, frame, pre-grasp and grasp joint solutions, press face), cheapest first."""
    vac_T = w.vacuum.T
    frames = frames_of(vac_T)
    centre = vac_T[:3, :3] @ (0.5 * (MD.HANDLE_LO + MD.HANDLE_HI)) + vac_T[:3, 3]
    if stands is None:
        stands = G.ring_stands(centre[:2], P.GRASP_RADII, P.N_BEARINGS, P.YAW_OFFSETS)
        keep = [i for i, st in enumerate(stands) if np.linalg.norm(st[:2] - vac_T[:2, 3]) >= P.VAC_STAND_MIN_M and c.free_pose(st[:2])]
        stands = stands[keep]
    out: List[Dict[str, Any]] = []
    if not len(stands):
        return out
    faces = G.press_faces(P.PRESS_TOUCH_M, P.PRESS_IN_M)
    goals = []
    for f in frames:
        goals += [G.pose_T(f['rot'], f['pre']), G.pose_T(f['rot'], f['point'])]
    grasp = {}
    ok_any = np.zeros(len(stands), bool)
    for side in ('left', 'right'):                       # the stands that reach the pinch with either arm
        masks, qs = ik.scan(side, goals, stands)
        c.counts['scans'] += 1
        grasp[side] = (masks, qs)
        ok_any |= np.any(masks[0::2] & masks[1::2], axis=0)
    idx = np.flatnonzero(ok_any)
    if not len(idx):
        return out
    press = press_scan(vac_T, stands[idx], [f for f in faces if f['name'][-1] in '01'])        # the free hand's press only from those (two finger turns per face)
    pos = {int(i): k for k, i in enumerate(idx)}
    for side in ('left', 'right'):
        masks, qs = grasp[side]
        for j, f in enumerate(frames):
            for i in np.flatnonzero(masks[2 * j] & masks[2 * j + 1]):
                st = stands[i]
                if float(np.linalg.norm(f['pre'][:2] - st[:2])) > float(np.linalg.norm(f['point'][:2] - st[:2])) - P.SIDE_MARGIN_M:
                    continue                      # the hand would reach across the vacuum to the far side of the loop
                pick = press_pick(press, 'right' if side == 'left' else 'left', pos[int(i)], qs[2 * j + 1, i])
                nopress = pick is None
                if nopress and not P.NOPRESS_ON:
                    continue
                if nopress:
                    pick = (faces[0], P.TRUNK_MAX)         # the free arm reaches no face from here: grasp anyway, the press search moves the held vacuum
                yaw_err = abs(G.wrap(st[2] - w.base[2]))
                ecost = float(np.linalg.norm(st[:2] - w.base[:2]))
                cost = ecost + 0.3 * yaw_err + 0.2 * float(np.linalg.norm(qs[2 * j + 1, i] - K.rest(side))) + P.PITCH_COST * f['pitch'] \
                    + P.TILT_COST * (abs(f['roll']) + abs(f['yaw'])) + PRESS_TRUNK_COST * pick[1] + (P.NOPRESS_COST if nopress else 0.)
                out.append(dict(stand=st.copy(), side=side, frame=f, cost=cost, ecost=ecost, q_pre=qs[2 * j, i].copy(),
                                q_grasp=qs[2 * j + 1, i].copy(), press_face=pick[0], press_dtrunk=pick[1], nopress=nopress))
    out.sort(key=lambda x: x['cost'])
    return approach_check(c, w, banned_out(c, press_check(c, w, out)))


def frame_key(f) -> tuple:
    return (round(float(f['pitch']), 2), float(f['sign']), round(float(f['roll']), 2), round(float(f['yaw']), 2))


def banned_out(c, cands) -> List[Dict[str, Any]]:
    """Without the (arm, stand, pinch frame) of failed presses."""
    if c.cand_relax >= 1:
        return cands
    return [x for x in cands if not any(b['side'] == x['side'] and b['key'] == frame_key(x['frame']) and float(np.linalg.norm(b['stand'] - x['stand'][:2])) < P.BAN_RADIUS_M
                                        for b in c.banned)]


def approach_mode(cand: Dict[str, Any]) -> Optional[str]:
    """Finger approach mode.

    How the fingers get from the pre-grasp point to the handle from the candidate's stand: 'joint' (one solve_near at the grasp point, the joint
    walk keeps the finger midpoint within JOINT_PATH_TOL of the straight line), 'line' (a chain of solve_near along the line without a branch
    jump), or None (neither). The measurements go into cand['approach_info']."""
    f, side, stand, x0 = cand['frame'], cand['side'], cand['stand'], cand['q_pre']
    info = cand['approach_info'] = {}
    xp = ik.hand_near(side, G.pose_T(f['rot'], f['point']), stand, x0)
    info['point_solved'] = xp is not None
    if xp is not None:
        info['dq_point'] = round(float(np.max(np.abs(xp - x0))), 3)
        a, ab = f['pre'], f['point'] - f['pre']
        dev = 0.
        for lam in np.linspace(0.0, 1.0, 9)[1:-1]:
            v = x0 + (xp - x0) * lam
            p = ik.hand_fk(side, v[:4], v[4:], stand)[:3, 3]
            t = float(np.clip((p - a) @ ab / max(float(ab @ ab), 1e-12), 0., 1.))
            dev = max(dev, float(np.linalg.norm(p - (a + t * ab))))
        info['dev'] = round(dev, 4)
        if info['dq_point'] <= P.JOINT_PATH_MAX and dev <= P.JOINT_PATH_TOL:
            cand['xp'] = xp                     # the joints at the handle: the press was checked with their trunk
            return 'joint'
    x, n, chain = x0, 12, []
    for k in range(1, n + 1):
        nxt = ik.hand_near(side, G.pose_T(f['rot'], f['pre'] + (f['point'] - f['pre']) * k / n), stand, x)
        if nxt is None:
            info['chain_fail'] = f'no solve at {k}/{n}'
            return None
        jump = float(np.max(np.abs(nxt - x)))
        info['chain_max'] = max(info.get('chain_max', 0.), round(jump, 3))
        if jump > P.JUMP_MAX:
            info['chain_fail'] = f'jump {jump:.2f} at {k}/{n}'
            return None
        x = nxt
        chain.append(nxt)
    cand['chain'] = chain                  # the planned joint solutions along the line: the seeds of the execution
    return 'line'


def arm_path_clear(c, w, cand) -> bool:
    """The arm walk to the pre-grasp stays clear of the vacuum body.

    The joint walk of the trunk and the arm from the travel posture to the pre-grasp solution keeps the arm links (link 3 to the fingertips) out of the
    vacuum's body box grown by PATH_GROW_M (forward kinematics at the candidate stand, PATH_SAMPLES points); a walk through the vacuum knocks it over."""
    side, st = cand['side'], cand['stand']
    x0 = c.rest_pose(side)
    inv = np.linalg.inv(w.vacuum.T)
    lo, hi = MD.VERTS.min(0) - P.PATH_GROW_M, MD.VERTS.max(0) + P.PATH_GROW_M
    names = [f'{side}_{n}' for n in ('arm_link3', 'arm_link4', 'arm_link5', 'arm_link6', 'arm_link7', 'gripper_link', 'gripper_finger_link1', 'gripper_finger_link2')]
    cz, sz = math.cos(st[2]), math.sin(st[2])
    qq = np.zeros(22)
    for a in np.linspace(0., 1., P.PATH_SAMPLES + 1)[1:]:
        v = x0 + (cand['q_pre'] - x0) * a
        qq[list(K.TRUNK_DOFS)] = v[:4]
        qq[list(K.ARM_DOFS[side])] = v[4:]
        poses = K.kinematics().forward(qq)[0]
        pts = np.array([poses[n][:3, 3] for n in names])
        pts = np.vstack([pts, 0.5 * (pts[:-1] + pts[1:])])
        world = np.c_[st[0] + cz * pts[:, 0] - sz * pts[:, 1], st[1] + sz * pts[:, 0] + cz * pts[:, 1], pts[:, 2]]
        body = world @ inv[:3, :3].T + inv[:3, 3]
        if np.any(np.all((body >= lo) & (body <= hi), axis=1)):
            return False
    return True


def approach_check(c, w, cands) -> List[Dict[str, Any]]:
    """Keep the candidates whose approach to the handle is...

    Keep the candidates whose approach to the handle is continuous in joint space (see approach_mode); the mode goes with them."""
    kept, bad = [], 0
    for cand in cands:
        mode = approach_mode(cand) if (not P.PATH_CHECK or c.cand_relax >= 1 or arm_path_clear(c, w, cand)) else None
        cand['mode'] = mode
        if mode is None:
            bad += 1
            continue
        cand['mode'] = mode
        kept.append(cand)
    c.approach_check_log = dict(of=len(cands), kept=len(kept), bad=bad, joint=sum(1 for x in kept if x['mode'] == 'joint'),
                                info=[x.get('approach_info') for x in cands[:6]])
    return kept


def press_check(c, w, cands, side_filter=True) -> List[Dict[str, Any]]:
    """Cheapest candidates whose free arm reaches a press.

    Keep the cheapest grasp candidates whose free arm reaches a press face (pre-press and full push) with the trunk the candidate's own
    grasp will have (solve_near at the grasp point from the pre-grasp solution), arm-only cuRobo at the candidate stand; the face and its
    joints go with them. The free hand presses from the robot's side of the face, the x+ face first."""
    vac_T = w.vacuum.T
    faces = sorted(G.press_faces(P.PRESS_TOUCH_M, P.PRESS_IN_M), key=lambda f: 0 if f['name'].startswith('x+') else 1)
    kept, tried = [], 0
    for cand in cands:
        if tried >= P.PRESS_CHECKS or len(kept) >= P.ROUTE_CHECKS:
            break
        tried += 1
        free = 'right' if cand['side'] == 'left' else 'left'
        g = ik.hand_near(cand['side'], G.pose_T(cand['frame']['rot'], cand['frame']['point']), cand['stand'], cand['q_pre'])
        if g is None:
            continue
        trunk = g[:4]
        cand['q_grasp_pred'] = g
        if cand.get('nopress'):
            kept.append(cand)
            continue
        for f in faces:
            fp = vac_T[:3, :3] @ f['point'] + vac_T[:3, 3]
            fpre = fp + vac_T[:3, :3] @ f['normal'] * P.PRESS_PRE_M
            if side_filter and float(np.linalg.norm(fpre[:2] - cand['stand'][:2])) > float(np.linalg.norm(fp[:2] - cand['stand'][:2])) - P.SIDE_MARGIN_M:
                continue
            qs, q0 = [], K.rest(free)[4:]
            for off in (P.PRESS_PRE_M, f['final']):
                fr = G.press_frame(vac_T, f['point'], f['normal'], pre_m=off, close_body=f['close'])
                x = ik.arm_near(free, G.pose_T(fr['rot'], fr['pre']), cand['stand'], trunk, q0)
                if x is None:
                    break
                qs.append(x)
                q0 = x
            if len(qs) == 2 and P.PRESS_PLAN_MARGIN_M > 0:      # the push must also be solvable deeper than planned
                fr = G.press_frame(vac_T, f['point'], f['normal'], pre_m=f['final'] - P.PRESS_PLAN_MARGIN_M, close_body=f['close'])
                if ik.arm_near(free, G.pose_T(fr['rot'], fr['pre']), cand['stand'], trunk, qs[1]) is None:
                    qs = []
            if len(qs) == 2 and P.PRESS_ROBUST_M > 0:      # a press that is feasible only at this exact geometry fails after the stand is reached
                for d in ((P.PRESS_ROBUST_M, 0.), (-P.PRESS_ROBUST_M, 0.), (0., P.PRESS_ROBUST_M), (0., -P.PRESS_ROBUST_M)):
                    moved = vac_T.copy()
                    moved[:2, 3] += d
                    for off, q1 in zip((P.PRESS_PRE_M, f['final']), qs):
                        fr = G.press_frame(moved, f['point'], f['normal'], pre_m=off, close_body=f['close'])
                        if ik.arm_near(free, G.pose_T(fr['rot'], fr['pre']), cand['stand'], trunk, q1) is None:
                            qs = []
                            break
                    if not qs:
                        break
            if len(qs) == 2:
                cand.update(press_face=f, press_q=qs)
                kept.append(cand)
                break
    c.press_check_log = dict(tried=tried, kept=len(kept), of=len(cands), side_filter=side_filter)
    if not kept and side_filter and P.PRESS_RELAX:      # no stand has a face on the robot's side: any face the free arm reaches
        return press_check(c, w, cands, side_filter=False)
    return kept
