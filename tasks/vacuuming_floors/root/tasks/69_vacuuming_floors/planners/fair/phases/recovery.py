"""Recovery phases of the fair planner of task 69: look around with the vacuum in hand when no flake is left to visit but the goal is not met
(a flake was missed or never seen), and the terminal phases."""
from __future__ import annotations

import math

import numpy as np

from .. import geometry as G
from .. import params as P
from .target import cube_in_base, swept_free, unchecked


def check_flake(c, action, q, w) -> None:
    """Drive to a vantage point 1.25 m from a visited flake.

    Drive to a vantage point 1.25 m from a visited flake, face it and stand still: the head camera then sees whether it is still on the
    floor (seen again: another pass), or not (it goes out of the memory)."""
    c.arm_cmd = c.hold_arm.copy()
    if c.age == 1 or getattr(c, 'vantage', None) is None:
        todo = unchecked(c, w)
        if not todo:
            c.vantage = None
            c.change('plan_sweep', 'every visited flake looked at')
            return
        d = min(todo, key=lambda x: float(np.linalg.norm(x.xy - w.base[:2])))
        c.build_grid(w, [d.xy, w.base[:2]])
        best = None
        for k in range(8):
            a = 2 * math.pi * k / 8
            xy = d.xy + P.VANTAGE_M * np.array([math.cos(a), math.sin(a)])
            if not c.free_pose(xy) or not c.head_free(d.xy, math.atan2(*(d.xy - xy)[::-1])):
                continue
            r = c.route_len(w, xy)
            if math.isfinite(r) and (best is None or r < best[0]):
                best = (r, xy)
        if best is None:
            d.checked = True
            c.sweep_log.append(dict(step=c.step, pid=d.pid, what='no vantage point'))
            c.vantage = None
            c.age = 0
            return
        xy = best[1]
        c.vantage = dict(pid=d.pid, xy=xy, yaw=math.atan2(d.xy[1] - xy[1], d.xy[0] - xy[0]), phase='drive')
        c.path = None
    v = c.vantage
    d = next((x for x in c.perc.dust if x.pid == v['pid']), None)
    if v['phase'] == 'drive':
        dist, yaw_err = c.navigate(action, w, v['xy'], v['yaw'], P.SWEEP_VFAR, P.SWEEP_VNEAR, P.SWEEP_WMAX)
        c.reason = f"to the vantage point of flake {v['pid']}: {dist:.2f} m"
        if c.arrived(dist, yaw_err, 0.05, 0.08):
            v['phase'], v['t0'] = 'look', c.age
        elif c.counts['nav_stuck'] >= P.NAV_MAX_STUCK or c.age > P.SWEEP_TIMEOUT:
            c.counts['nav_stuck'] = 0
            if d is not None:
                d.checked = True
            c.vantage = None
    else:
        c.reason = f"looking at flake {v['pid']} from the vantage point"
        if c.age - v['t0'] > P.VANTAGE_PAUSE:
            if d is not None:
                d.checked = True
            c.sweep_log.append(dict(step=c.step, pid=v['pid'], what='looked', gone=None if d is None else d.gone, reseen=None if d is None else d.reseen))
            c.vantage = None
            c.change('plan_sweep', f"looked at flake {v['pid']}")


def reset(c, why: str) -> None:
    """Forget what was looked at and look again, or give up."""
    if c.explore_resets >= P.EXPLORE_RESETS:
        c.change('failed', why)
        return
    c.explore_resets += 1
    c.explores = 0
    c.perc.dust_cov[:] = False
    c.sweep_log.append(dict(step=c.step, what='explore reset', why=why))


def explore(c, action, q, w) -> None:
    """Drive, the vacuum in hand, to where the unlooked floor is seen.

    Nothing left to visit but the goal is not met: drive (the vacuum in hand) to the place from which the head looks at the most floor not yet looked at for flakes,
    stand still there and look."""
    c.arm_cmd = c.hold_arm.copy()
    if c.expl is None:
        centre, _, _ = w.cube()
        todo = c.perc.uncovered_floor(w.base[:2], P.EXPLORE_RADIUS_M)
        c.explores += 1
        if len(todo) < P.EXPLORE_MIN_CELLS or c.explores > P.MAX_EXPLORES:
            reset(c, f'nothing left to look at ({len(todo)} cells, {c.explores - 1} looks)')
            return
        c.build_grid(w, [w.base[:2]] + [t for t in todo[::8]])
        c_b, phi = cube_in_base(w)
        best = None
        for xy in todo[np.random.default_rng(c.explores).permutation(len(todo))[:60]]:
            for ang in np.arange(0., 2 * math.pi, math.pi / 4):
                for d in (1.2, 1.7):
                    b = xy - d * np.array([math.cos(ang), math.sin(ang)])
                    yaw = ang
                    if not c.free_pose(b) or not c.head_free(b + G.rot_yaw(yaw)[:2, :2] @ c_b, yaw + phi) or not swept_free(c, w, b, yaw, c_b, phi, False):
                        continue
                    rel = todo - b
                    r, bear = np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])
                    gain = int(np.sum((r > P.COVER_RANGE_M[0]) & (r < P.COVER_RANGE_M[1]) & (np.abs((bear - yaw + math.pi) % (2 * math.pi) - math.pi) < 0.6)))
                    cost = c.route_len(w, b)
                    if math.isfinite(cost) and gain > 0 and (best is None or gain / (1. + 0.4 * cost) > best[0]):
                        best = (gain / (1. + 0.4 * cost), b, yaw, gain)
        if best is None:
            reset(c, 'no standing place sees the floor left to look at')
            return
        rel = todo - best[1]
        r, bear = np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])
        seen = (r > P.COVER_RANGE_M[0]) & (r < P.COVER_RANGE_M[1]) & (np.abs((bear - best[2] + math.pi) % (2 * math.pi) - math.pi) < 0.6)
        c.expl = dict(xy=best[1], yaw=best[2], gain=best[3], phase='drive', cells=todo[seen])
        c.path = None
        c.sweep_log.append(dict(step=c.step, what='explore', xy=np.round(best[1], 2).tolist(), cells=best[3], todo=len(todo)))
    e = c.expl
    if e['phase'] == 'drive':
        dist, yaw_err = c.navigate(action, w, e['xy'], e['yaw'], P.SWEEP_VFAR, P.SWEEP_VNEAR, P.SWEEP_WMAX)
        c.reason = f"to a standing place that looks at {e['gain']} floor cells, {dist:.2f} m"
        if c.arrived(dist, yaw_err, 0.05, 0.08) or c.counts['nav_stuck'] >= P.NAV_MAX_STUCK or c.age > P.SWEEP_TIMEOUT:
            c.counts['nav_stuck'] = 0
            e['phase'], e['t0'] = 'look', c.age
    else:
        c.reason = 'looking at the floor from here'
        if c.age - e['t0'] > P.VANTAGE_PAUSE:
            cells = c.perc._cells(e['cells'])                  # looked at: never again (the head may not resolve them, e.g. behind the vacuum)
            c.perc.dust_cov[cells[:, 0], cells[:, 1]] = True
            c.expl = None
            c.change('plan_sweep', 'looked at the floor from a new place')


def done(c, action, q, w) -> None:
    if c.hold_arm is not None:
        c.arm_cmd = c.hold_arm.copy()
    c.reason = 'done'


def failed(c, action, q, w) -> None:
    if c.hold_arm is not None and c.arm_cmd is not None:
        c.arm_cmd = c.hold_arm.copy()


PHASES = dict(check_flake=check_flake, explore=explore, done=done, failed=failed)
