"""Sweep phases of the fair planner of task 69: the vacuum is held (arm and trunk frozen), on, and its projection cube is levelled with the
floor; the base drives so that the cube centre visits every flake believed on the floor (nearest neighbour + 2-opt order, a base yaw per flake
with the base and the head footprint free in the perceived grid). The flake position is refreshed while it is in view during the approach (the
odometry drifts, the head camera sees the floor from about 0.9 m); a flake seen again after a pass was not removed and gets another pass at
another hand height. Subgoals first_removed, half_removed, not_covered (the harness measures them)."""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from .. import params as P
from ..perception.track import relook
from .grasp import floor_w, regrasp
from .target import cube_in_base, pending, target_pose, unchecked


def plan_sweep(c, action, q, w) -> None:
    c.arm_cmd = c.hold_arm.copy()
    c.lift_s = 0.
    if not c.holds(q, c.side):
        c.change('failed', 'the vacuum is no longer held')
        return
    todo = pending(c, w)
    if not todo:
        if c.hard and c.age > 2:
            c.hard.clear()               # every flake left failed once: retry them with the other yaws, and with less strict tests
            c.relax = min(c.relax + 1, 2)
            c.sweep_log.append(dict(step=c.step, what='retry the given-up flakes', relax=c.relax))
            return
        c.reason = f'no flake to visit ({len(w.dust)} believed, {len(c.hard)} given up)'
        if c.age > P.IDLE_STEPS:
            if unchecked(c, w):
                c.vantage = None
                c.change('check_flake', 'no flake left to visit; looking at the visited ones')
            else:
                c.expl = None
                c.change('explore', 'no flake left to visit; looking at the floor not looked at yet')
        return
    centre, _, _ = w.cube()
    c.build_grid(w, [d.xy for d in w.dust] + [centre[:2]])
    order = G.order_greedy(centre[:2], {d.pid: d.xy for d in todo})
    by_pid = {d.pid: d for d in todo}
    for pid in order:
        d = by_pid[pid]
        t = target_pose(c, w, d) or target_pose(c, w, d, tight=True)
        if t is None:
            c.hard.add(pid)
            c.counts['hard'] += 1
            c.sweep_log.append(dict(step=c.step, pid=pid, what='no base pose', why=getattr(c, 'target_why', None)))
            continue
        t['p'] = d.xy.copy()
        t['dz'] = P.REVISIT_DZ[min(d.visits, len(P.REVISIT_DZ) - 1)]
        c.target = t
        c.counts['targets'] += 1
        c.drive_start = w.base.copy()
        c.change('adjust' if abs(t['dz'] - getattr(c, 'hand_dz', 0.)) > 1e-4 else 'drive',
                 f"flake {pid} ({len(w.dust)} believed, pass {d.visits + 1}): base {np.round(t['xy'], 2).tolist()} yaw {t['yaw']:.2f}")
        return
    c.reason = 'no base pose for any flake'
    if c.age > 30:
        if c.nopose >= P.MAX_NOPOSE:
            c.change('failed', f'no base pose reaches the {len(w.dust)} flakes believed')
            return
        c.nopose += 1               # the flakes believed are misplaced or out of reach: forget them and look at that floor again
        for d in todo:
            d.gone = True
            c.perc.counts['dust_gone'] += 1
            relook(c.perc, d)
        c.expl = None
        c.change('explore', f'no base pose reaches the {len(todo)} flakes believed; looking at the floor again')


def adjust(c, action, q, w) -> None:
    """Move the hand up or down (the cube with it) to the pass...

    Move the hand up or down (the cube with it) to the pass height of this flake."""
    c.arm_cmd = c.hold_arm.copy()
    if c.line is None:
        mid = c.hand(q)
        want = c.target['dz'] - getattr(c, 'hand_dz', 0.)
        c.start_line(mid[:3, 3], mid[:3, 3] + G.UP * want, mid[:3, :3], step=P.LIFT_STEP_M)
        c.want_dz = c.target['dz']
    r = c.follow_line(w)
    if r is False:
        c.line = None
        give_up(c, w, 'no IK for the height change')
    elif r:
        c.line = None
        c.hand_dz = c.want_dz
        c.hold_arm = c.arm_cmd.copy()
        c.raise_pose(w)
        c.change('drive', f'hand at the pass height {c.hand_dz:+.3f} m')


def drive(c, action, q, w) -> None:
    c.arm_cmd = c.hold_arm.copy()
    t = c.target
    if not c.holds(q, c.side):
        c.change('failed', 'the vacuum fell out of the hand while sweeping')
        return
    pulled = c.hand_pulled(q, w)
    t['pull_max'] = max(t.get('pull_max', 0.), pulled)
    if w.tilted or pulled > P.PULL_ALARM_M:      # the head caught on something (the hand is pulled off where it was commanded): back off
        t['why_alarm'] = 'marker' if w.tilted else f'the hand is {pulled:.3f} m off its command'
        give_up(c, w, 'the vacuum is tilted: ' + t['why_alarm'], to='recover')
        return
    d = next((x for x in w.dust if x.pid == t['pid']), None)
    if d is None:
        c.sweep_log.append(dict(step=c.step, pid=t['pid'], what='gone en route'))
        c.change('plan_sweep', f"flake {t['pid']} no longer believed")
        return
    if float(np.linalg.norm(d.xy - t['p'])) > P.REFRESH_TARGET_M and w.step - d.seen_step < 10:   # seen again, elsewhere: follow it
        c_b, _ = cube_in_base(w)
        t['xy'] = d.xy - G.rot_yaw(t['yaw'])[:2, :2] @ c_b
        t['p'] = d.xy.copy()
        c.path = None
    if not t.get('looked') and dist_to(w, t) < P.LOOK_BEFORE_M and dist_to(w, t) > 0.5:
        c.pause = P.LOOK_PAUSE                                 # stand still and look: the flake is placed without the odometry lag
        t['looked'] = True
    if c.pause > 0:
        c.pause -= 1
        c.reason = f"looking at flake {t['pid']} standing"
        return
    vfar = P.SWEEP_VFAR * (1.25 if dist_to(w, t) > 0.8 else 1.0)
    dist, yaw_err = c.navigate(action, w, t['xy'], t['yaw'], vfar, P.SWEEP_VNEAR, P.SWEEP_WMAX, tight=t.get('tight', False), gain=P.SWEEP_GAIN)
    c.arm_cmd = c.transit(dist)
    c.err = (dist, yaw_err)
    c.reason = f"to flake {t['pid']}: {dist:.2f} m, yaw err {yaw_err:.2f}, floor w {floor_w(w):+.2f}"
    if t.get('best') is None or dist < t['best'] - P.SWEEP_PROGRESS_M:      # the head caught on something: the base creeps on and the stuck check never fires
        t['best'], t['best_age'] = dist, c.age
    if dist < P.SWEEP_POS_TOL and abs(yaw_err) < P.SWEEP_YAW_TOL:
        c.change('level' if abs(floor_w(w) - P.CUBE_W_TARGET) > P.LEVEL_W_TOL else 'verify', f"over flake {t['pid']}, floor w {floor_w(w):+.2f}")
    elif P.SWEEP_STALL_STEPS > 0 and c.age - t['best_age'] > P.SWEEP_STALL_STEPS:
        give_up(c, w, f'no progress for {P.SWEEP_STALL_STEPS} controls, {dist:.2f} m from the place')
    elif c.counts['nav_stuck'] >= P.NAV_MAX_STUCK or c.age > P.SWEEP_TIMEOUT:
        c.counts['nav_stuck'] = 0
        give_up(c, w, f'not reached ({dist:.2f} m)')


def dist_to(w, t) -> float:
    return float(np.linalg.norm(t['xy'] - w.base[:2]))


def level(c, action, q, w) -> None:
    """Over the flake: move the hand by the cube height error, hold.

    Over the flake but the cube is off its level (the arm sags or is pulled on a drive): move the hand up or down by the height error, then hold."""
    c.arm_cmd = c.hold_arm.copy() if c.line is None else c.arm_cmd
    if c.line is None:
        c.lvl_dz = float(np.clip((floor_w(w) - P.CUBE_W_TARGET) * float(w.cube()[2][2]), -0.02, 0.02))
        mid = c.hand(q)
        c.start_line(mid[:3, 3], mid[:3, 3] + G.UP * c.lvl_dz, mid[:3, :3], step=P.LIFT_STEP_M)
    r = c.follow_line(w)
    c.reason = f'levelling the cube over the flake: dz {c.lvl_dz:+.4f} m, floor w {floor_w(w):+.2f}'
    if r is not None or c.age > 120:
        c.line = None
        c.hold_arm = c.arm_cmd.copy()
        c.change('verify', f'cube levelled, floor w {floor_w(w):+.2f}')


def give_up(c, w, why: str, to: str = 'plan_sweep') -> None:
    t = c.target
    c.tries[t['pid']] = c.tries.get(t['pid'], 0) + 1
    c.bad_yaws.setdefault(t['pid'], []).append(t['yaw'])
    if c.tries[t['pid']] >= P.MAX_TARGET_TRIES:
        c.hard.add(t['pid'])
        c.counts['hard'] += 1
    c.sweep_log.append(dict(step=c.step, pid=t['pid'], what=why, tries=c.tries[t['pid']]))
    c.change(to, f"flake {t['pid']}: {why}")


def verify(c, action, q, w) -> None:
    """Hold the cube over the flake for a few controls.

    Hold the cube over the flake for a few controls (the native remover deletes a flake inside the cube once per control, vacuum on)."""
    c.arm_cmd = c.hold_arm.copy()
    t = c.target
    if w.tilted:
        give_up(c, w, 'the vacuum is tilted (the marker is off its place)', to='recover')
        return
    c.reason = f"holding over flake {t['pid']}, marker {w.toggled}"
    if c.age < P.VERIFY_STEPS:
        return
    d = next((x for x in w.dust if x.pid == t['pid']), None)
    for x in c.perc.dust:
        if x.pid == t['pid']:
            x.visits += 1
            x.visit_step, x.reseen, x.checked = c.step, 0, False
    c.bad_yaws.setdefault(t['pid'], []).append(t['yaw'])      # the next pass over it comes from another direction
    c.sweep_log.append(dict(step=c.step, pid=t['pid'], what='passed', visits=0 if d is None else d.visits + 1, floor_w=round(floor_w(w), 2), marker=w.toggled,
                            pull_max=round(t.get('pull_max', 0.), 3)))
    c.change('plan_sweep', f"passed over flake {t['pid']}")


def recover(c, action, q, w) -> None:
    """Back to where the last drive started.

    Back to where the last drive started (base only, slowly), then level the cube again."""
    c.arm_cmd = c.hold_arm.copy()
    b = c.drive_start
    dist, yaw_err = c.navigate(action, w, b[:2], b[2], P.SWEEP_VNEAR, P.SWEEP_VNEAR, P.SWEEP_WMAX)
    c.reason = f'backing off: {dist:.2f} m, tilted {w.tilted}'
    if dist < 0.05 and w.tilted and c.unjam < P.MAX_UNJAM:       # back where the drive began and still tilted: the head is jammed, pull it back along its own axis
        c_b, _ = cube_in_base(w)
        tgt = w.base[:2] - P.UNJAM_M * G.rot_yaw(w.base[2])[:2, :2] @ G.unit(c_b)
        c.unjam += 1
        if c.free_pose(tgt):
            c.drive_start = np.array([tgt[0], tgt[1], w.base[2]])
        return
    if (dist < 0.05 and not w.tilted) or c.age > P.RECOVER_STEPS:
        c.unjam = 0
        c.counts['recoveries'] += 1
        c.alarms += 1
        if c.alarms >= P.TILT_REGRASP_AFTER and w.toggled:      # the vacuum keeps tilting in the hand (it slips under the drag): put it down, pinch it upright again, still on
            c.alarms = 0
            n0 = c.counts['tilt_regrasps']
            regrasp(c, q, w, 'the held vacuum keeps tilting in the hand', press=False)
            if c.counts['tilt_regrasps'] > n0:
                return
        c.lift_round, c.line = 0, None
        c.change('lift', f'backed off ({dist:.2f} m): levelling the cube again')


PHASES = dict(plan_sweep=plan_sweep, adjust=adjust, drive=drive, level=level, verify=verify, recover=recover)
