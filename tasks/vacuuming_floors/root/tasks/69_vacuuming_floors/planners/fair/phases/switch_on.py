"""Switch-on phases of the fair planner of task 69: the hand that does not hold the vacuum, fingers closed, pushes a handle face at the ToggledOn
marker (the robot-side face, +x first) up to 5 cm inside; the arm drive presses the fingertip past the contact far enough to enter the marker's
1.87 cm sphere for the native five controls (a pinch alone stays 2 cm away). The free arm is solved alone with the trunk held (arm-only cuRobo),
so the held vacuum does not move; it tucks to its rest pose afterwards. The switch state is read from the marker: red off, green on (cameras)."""
from __future__ import annotations

import numpy as np

from core.skills_lib import r1pro_kinematics as K

from .. import geometry as G
from .. import ik
from .. import params as P
from .base import Walker
from .grasp import regrasp


def press_solve(c, w, offset: float):
    """Joint command (18 = trunk 4.

    Joint command (18 = trunk 4, holding arm 7, free arm 7) for the press target at `offset` along the face normal, the trunk held."""
    f = G.press_frame(w.vac_T, c.press_face['point'], c.press_face['normal'], pre_m=offset, close_body=c.press_face['close'])
    q0 = c.other_cmd
    planned = (c.grasp or {}).get('press_q')
    if planned is not None and (c.grasp.get('press_face') or {}).get('name') == c.press_face['name'] \
            and float(np.max(np.abs(c.other_cmd - c.free_start))) < 1e-6:
        q0 = planned[0]                  # the first solve of the planned face: from the planning-time solution
    x7 = ik.arm_near(c.free_side, G.pose_T(f['rot'], f['pre']), w.base, c.arm_cmd[:4], q0)
    if x7 is None and (offset < P.PRESS_TOUCH_M or P.PRESS_LOOSE_PRE or getattr(c, 'press_loose', False)) and P.PRESS_LOOSE:      # a push past the contact: the loose solver (the orientation is not critical)
        x7 = ik.arm_near(c.free_side, G.pose_T(f['rot'], f['pre']), w.base, c.arm_cmd[:4], q0, loose=True)
    if x7 is None and float(np.max(np.abs(q0 - K.rest(c.free_side)[4:]))) > 1e-6:        # the planning seeded with the rest pose of the free arm
        x7 = ik.arm_near(c.free_side, G.pose_T(f['rot'], f['pre']), w.base, c.arm_cmd[:4], K.rest(c.free_side)[4:])
    if x7 is not None and offset < P.PRESS_PRE_M - 0.011 and float(np.max(np.abs(x7 - c.other_cmd))) > P.PRESS_JUMP_MAX:
        c.two_arm_why = f'arm-only IK jumps {float(np.max(np.abs(x7 - c.other_cmd))):.2f} rad (another branch)'
        return None
    if x7 is None:
        c.two_arm_why = 'arm-only IK (trunk held)'
        return None
    return np.r_[c.arm_cmd, x7]


def dual_walk(c, target, step: float = P.JOINT_STEP, settle: int = P.WALK_SETTLE) -> None:
    c.walker = Walker(np.r_[c.arm_cmd, c.other_cmd], target, step, settle)


def dual_next(c) -> bool:
    v, done = c.walker.next()
    c.arm_cmd, c.other_cmd = v[:11], v[11:]
    return done


def press_plan(c, action, q, w) -> None:
    new_key = getattr(c, '_press_key', None) != id(c.grasp)
    if not getattr(c, 'press_faces', None) and (not c.press_log or new_key):
        c._press_key = id(c.grasp)            # a new grasp: the faces, the search and the closer pre-press are tried again
        c.searched = False
        c._pre_fb = False
        c.press_pre_used = P.PRESS_PRE_M
        c.press_loose = False
        faces = G.press_faces(P.PRESS_TOUCH_M, P.PRESS_IN_M)
        here = c.hand(q, c.free_side)[:3, 3]
        for f in faces:
            fr = G.press_frame(w.vac_T, f['point'], f['normal'], P.PRESS_PRE_M, f['close'])
            f['dist'] = float(np.linalg.norm(fr['pre'] - here)) + (1.0 if f['name'].startswith('x+') else 0.0) + 0.05 * int(f['name'][-1])
        c.press_faces = sorted(faces, key=lambda f: f['dist'])
        chosen = (c.grasp or {}).get('press_face')
        if chosen is not None:              # the face the planning chose first
            c.press_faces = [f for f in c.press_faces if f['name'] == chosen['name']] + [f for f in c.press_faces if f['name'] != chosen['name']]
        c.free_start = c.other_cmd.copy()
        c._press_all = list(c.press_faces)
        c.counts['press_tries'] += 1
    while c.press_faces:
        c.press_face = c.press_faces.pop(0)
        x = press_solve(c, w, P.PRESS_PRE_M)
        if x is None:
            pred = (c.grasp or {}).get('q_grasp_pred')
            c.press_log.append(dict(step=c.step, face=c.press_face['name'], what='no arm-only IK for the pre-press: ' + c.two_arm_why,
                                    dtrunk=None if pred is None else np.round(c.arm_cmd[:4] - pred[:4], 3).tolist()))
            continue
        c.grip[c.free_side] = K.GRIPPER_CLOSED
        c.press_pre_used = P.PRESS_PRE_M
        c.press_loose = False
        dual_walk(c, x, P.PRE_STEP)
        c.change('press_reach', f"free {c.free_side} hand to the {c.press_face['name']} face")
        return
    if not getattr(c, '_pre_fb', True):
        c._pre_fb = True                    # no face at the 0.10 m pre-press: a closer pre-press, then the loose solver, from the current and the rest pose
        rest7 = K.rest(c.free_side)[4:]
        for loose in (False, True):
            for pre in (0.07, 0.045, 0.02):
                for f in getattr(c, '_press_all', []):
                    fr = G.press_frame(w.vac_T, f['point'], f['normal'], pre_m=pre, close_body=f['close'])
                    for q0 in (c.other_cmd, rest7):
                        x7 = ik.arm_near(c.free_side, G.pose_T(fr['rot'], fr['pre']), w.base, c.arm_cmd[:4], q0, loose=loose)
                        if x7 is None:
                            continue
                        c.press_face = f
                        c.press_pre_used = pre
                        c.press_loose = loose
                        c.free_start = c.other_cmd.copy()
                        c.grip[c.free_side] = K.GRIPPER_CLOSED
                        c.press_log.append(dict(step=c.step, face=f['name'], what='closer pre-press', pre=pre, loose=loose))
                        dual_walk(c, np.r_[c.arm_cmd, x7], P.PRE_STEP)
                        c.change('press_reach', f"free {c.free_side} hand to the {f['name']} face at a pre-press of {pre:.3f} m (loose {loose})")
                        return
    if not c.searched:
        c.searched = True
        found = press_search(c, w)
        if found is not None:
            c.press_faces = [found['face']]
            c.walker = Walker(c.arm_cmd, found['x'], 0.5 * P.JOINT_STEP, 4)
            c.press_log.append(dict(step=c.step, what='moves the held vacuum', delta=found['delta'], face=found['face']['name']))
            c.change('press_pose', f"moving the held vacuum by {found['delta']} so that the free hand reaches the {found['face']['name']} face")
            return
    regrasp(c, q, w, f'no face to press reached; press log {c.press_log[-2:]}')


def press_search(c, w):
    """No face is reachable by the free arm.

    No face is reachable by the free arm (arm-only, trunk held) at the pose the vacuum is held in: move and turn the held vacuum (solve_near for the
    holding hand, the vacuum follows rigidly) until the free arm reaches the pre-press and the full push of some face. Returns the joints of the
    holding arm, the move and the face."""
    T0 = ik.hand_fk(c.side, c.arm_cmd[:4], c.arm_cmd[4:], w.base)
    faces = sorted(G.press_faces(P.PRESS_TOUCH_M, P.PRESS_IN_M), key=lambda f: 0 if f['name'].startswith('x+') else 1)
    fallback = None
    moves = sorted(P.PRESS_SEARCH, key=lambda d: float(np.linalg.norm(d[:3])) + 0.1 * abs(d[3]))
    for dx, dy, dz, dyaw in moves:
        T = T0.copy()
        T[:3, :3] = G.rot_yaw(dyaw) @ T0[:3, :3]
        T[:3, 3] = T0[:3, 3] + np.array([dx, dy, dz])
        x = ik.hand_near(c.side, T, w.base, c.arm_cmd)
        if x is None or float(np.max(np.abs(x - c.arm_cmd))) > 0.6:
            continue
        vac = T @ np.linalg.inv(T0) @ w.vac_T                  # the vacuum moves with the hand
        for f in faces:
            q0, ok = c.other_cmd, True
            for off in (P.PRESS_PRE_M, f['final'], f['final'] - P.PRESS_MARGIN_M):      # the push must also reach a little deeper than planned
                fr = G.press_frame(vac, f['point'], f['normal'], pre_m=off, close_body=f['close'])
                q0 = ik.arm_near(c.free_side, G.pose_T(fr['rot'], fr['pre']), w.base, x[:4], q0)
                if q0 is None:
                    ok = False
                    if off != f['final'] - P.PRESS_MARGIN_M and off == f['final']:
                        break
                    if off == P.PRESS_PRE_M:
                        break
                    if fallback is None:
                        fallback = dict(x=x, delta=(dx, dy, dz, dyaw), face=f)       # reaches the planned depth, not the margin
                    break
            if ok:
                return dict(x=x, delta=(dx, dy, dz, dyaw), face=f)
    return fallback


def press_pose(c, action, q, w) -> None:
    """Walk the holding arm to the pose found by press_search.

    Walk the holding arm to the pose found by press_search (the vacuum moves with the hand), then press."""
    c.arm_cmd, done = c.walker.next()
    c.reason = f'moving the held vacuum, joint gap {c.walker.gap:.3f}'
    if done or c.age > 400:
        c.walker = None
        c.hold_arm = c.arm_cmd.copy()
        c.change('press_plan', 'the held vacuum is moved; pressing')


def press_reach(c, action, q, w) -> None:
    if dual_next(c):
        c.walker = None
        touch, final = c.press_face['touch'], c.press_face['final']
        pre_used = float(getattr(c, 'press_pre_used', P.PRESS_PRE_M))
        c.press_offsets = list(np.arange(pre_used - 0.01, touch - 1e-9, -0.01)) + [touch] + list(np.arange(touch - P.PRESS_STEP_M, final - 1e-9, -P.PRESS_STEP_M))
        c.press_i, c.press_skips = 0, 0
        c.change('press_in', f"at the {c.press_face['name']} pre-press")
    elif c.age > 500:
        c.press_log.append(dict(step=c.step, face=c.press_face['name'], what='pre-press walk timed out'))
        c.change('press_out', 'pre-press walk timed out')


def press_in(c, action, q, w) -> None:
    name = c.press_face['name']
    if w.toggled:
        c.press_log.append(dict(step=c.step, face=name, what='switched on', offset=round(float(c.press_offsets[min(c.press_i, len(c.press_offsets) - 1)]), 4)))
        c.change('press_out', f'vacuum switched on by the {name} press (the marker is green)')
        return
    if c.walker is None:
        if c.press_i >= len(c.press_offsets):
            c.change('press_hold', f'pushed to {c.press_face["final"] * 100:+.1f} cm of the {name} target')
            return
        x = press_solve(c, w, float(c.press_offsets[c.press_i]))
        if x is None and c.press_i < len(c.press_offsets) - 1:        # a marginal waypoint: the end of the push may still be solvable (joint-space walk)
            x = press_solve(c, w, float(c.press_offsets[-1]))
            if x is not None:
                c.press_i = len(c.press_offsets) - 1
        if x is None:                       # a marginal waypoint: skip to the next one (a few in a row end the press)
            c.press_skips += 1
            c.press_log.append(dict(step=c.step, face=name, what='no arm-only IK on the press line: ' + c.two_arm_why,
                                    offset=round(float(c.press_offsets[c.press_i]), 4)))
            if c.press_skips > P.PRESS_SKIPS:
                c.change('press_out', f'no IK at {c.press_offsets[c.press_i]:+.3f} m of the {name} face')
            else:
                c.press_i += 1
            return
        c.press_skips = 0
        dual_walk(c, x, P.PRESS_WALK * P.JOINT_STEP, 2)
    c.reason = f"pressing the {name} face, offset {c.press_offsets[min(c.press_i, len(c.press_offsets) - 1)]:+.3f} m, marker {w.toggled}"
    if dual_next(c):
        c.walker = None
        c.press_i += 1


def press_hold(c, action, q, w) -> None:
    name = c.press_face['name']
    c.reason = f'holding the {name} press, marker {w.toggled}'
    if w.toggled:
        c.press_log.append(dict(step=c.step, face=name, what='switched on while holding'))
        c.change('press_out', f'vacuum switched on by the {name} press')
    elif c.age > P.PRESS_HOLD:
        c.press_log.append(dict(step=c.step, face=name, what='no switch after the hold', marker=w.toggled))
        c.change('press_out', f'the {name} press did not switch the vacuum on')


def press_out(c, action, q, w) -> None:
    if c.walker is None:
        x = press_solve(c, w, P.PRESS_PRE_M)
        if x is None:                       # back off the free arm alone along its joints to where it started
            x = np.r_[c.arm_cmd, c.free_start]
        dual_walk(c, x, P.PRESS_OUT_STEP, 2)
    if dual_next(c):
        c.walker = None
        if w.toggled:
            dual_walk(c, np.r_[c.arm_cmd, K.rest(c.free_side)[4:]], P.PRE_STEP)
            c.change('press_home', 'switched on; the free arm goes back')
        else:
            c.change('press_plan', 'next face')


def press_home(c, action, q, w) -> None:
    if dual_next(c):
        c.walker = None
        c.grip[c.free_side] = K.GRIPPER_CLOSED
        c.lift_round = 0
        c.line = None
        c.change('lift', 're-levelling the cube after the press')


PHASES = dict(press_pose=press_pose, press_plan=press_plan, press_reach=press_reach, press_in=press_in, press_hold=press_hold, press_out=press_out, press_home=press_home)
