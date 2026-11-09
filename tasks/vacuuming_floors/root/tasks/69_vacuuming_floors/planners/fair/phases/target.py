"""Target geometry of the sweep of the fair planner of task 69: where the cube sits relative to the base, the base pose that puts the cube centre on a
flake (the base and the head footprint free), the swept path of a move, which flakes still need a visit or a look. Imported by sweep.py and recovery.py."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from .. import geometry as G
from .. import params as P


def cube_in_base(w) -> tuple:
    """Cube centre (base frame xy) and the yaw of its long...

    Cube centre (base frame xy) and the yaw of its long axis relative to the base heading."""
    centre, rot, _ = w.cube()
    c_b = G.to_base(w.base, centre)
    return c_b, G.wrap(math.atan2(rot[1, 0], rot[0, 0]) - w.base[2])


SWATH_HALF_M = 0.11    # swept half-band credited to a drive: the 30 cm cube long axis across the travel, minus a margin
SWATH_COST = 0.6       # cost (m) of driving with the cube long axis along the travel (a 12 cm swath) instead of across it
COVER_BONUS = 0.45     # cost (m) subtracted per other pending flake the drive's swept band passes over


def swath_pen(axis_yaw: float, travel_yaw: float) -> float:
    """0 when the cube long axis lies across the travel direction (the widest swath), 1 when it lies along it.

    0 when the cube long axis lies across the travel direction (the widest swath), 1 when it lies along it."""
    return abs(math.cos(G.wrap(axis_yaw - travel_yaw)))


def swath_covered(others, p0, p1) -> int:
    """Other pending flakes whose xy lies within the swept band of the cube-centre segment from p0 to p1.

    Other pending flakes whose xy lies within the swept band of the cube-centre segment from p0 to p1: the cube centre moves from its
    start pose (rigid in the hand) to the target flake, and a flake inside SWATH_HALF_M of that segment is removed by the same drive."""
    seg = np.asarray(p1, float) - p0
    L2 = float(seg @ seg)
    n = 0
    for e in others:
        t = 0.0 if L2 < 1e-9 else float(np.clip(float((np.asarray(e.xy, float) - p0) @ seg) / L2, 0.0, 1.0))
        if float(np.linalg.norm(np.asarray(e.xy, float) - (p0 + t * seg))) < SWATH_HALF_M:
            n += 1
    return n


def target_pose(c, w, d, tight: bool = False) -> Optional[Dict[str, Any]]:
    """Best base pose putting the cube centre on the flake.

    Best base pose putting the cube centre on the flake: the base and the head footprint free, the move there clear (relax level 0); level 1
    drops the swept-path test, level 2 the head test too. Cheapest by distance + yaw change.
    The cost also turns the cube's long axis across the travel direction (a 30 cm swath instead of 12 cm: a believed xy off by dust_err
    still puts the true flake under the head on the first pass) and pays a bonus for every other pending flake the swept band passes over:
    those flakes are then removed by the same drive instead of a dedicated drive later, which the horizon of the lost seeds runs out of."""
    c_b, phi = cube_in_base(w)
    others = [e for e in pending(c, w) if e.pid != d.pid]
    p0 = w.base[:2] + G.rot_yaw(w.base[2])[:2, :2] @ c_b
    best, why = None, dict(bad_yaw=0, base=0, head=0, swept=0)
    for k in range(P.SWEEP_YAWS):
        th = G.wrap(w.base[2] + 2 * math.pi * k / P.SWEEP_YAWS)
        if any(abs(G.wrap(th - b)) < 0.2 for b in c.bad_yaws.get(d.pid, [])):
            why['bad_yaw'] += 1
            continue
        b = d.xy - G.rot_yaw(th)[:2, :2] @ c_b
        if not c.free_pose(b, tight):
            why['base'] += 1
            continue
        if c.relax < 2 and not c.head_free(d.xy, th + phi):
            why['head'] += 1
            continue
        if c.relax < 1 and not swept_free(c, w, b, th, c_b, phi, tight):
            why['swept'] += 1
            continue
        travel = math.atan2(b[1] - w.base[1], b[0] - w.base[0])
        cost = (float(np.linalg.norm(b - w.base[:2]))
                + P.SWEEP_YAW_COST * abs(G.wrap(th - w.base[2]))
                + SWATH_COST * swath_pen(th + phi, travel)
                - COVER_BONUS * swath_covered(others, p0, d.xy))
        if best is None or cost < best['cost']:
            best = dict(pid=d.pid, xy=b, yaw=th, cost=cost, tight=tight)
    if best is None:
        c.target_why = dict(pid=d.pid, xy=np.round(d.xy, 2).tolist(), tight=tight, relax=c.relax, **why)
    return best


def swept_free(c, w, b, th, c_b, phi, tight) -> bool:
    """The base and the vacuum head stay clear of the mapped...

    The base and the vacuum head stay clear of the mapped obstacles along the move from the base pose to (b, th): the head swings on an arc
    when the base turns with the vacuum held out in front."""
    dyaw = G.wrap(th - w.base[2])
    for k in range(1, P.SWEEP_PATH_SAMPLES):
        f = k / P.SWEEP_PATH_SAMPLES
        xy = w.base[:2] + f * (b - w.base[:2])
        yaw = w.base[2] + f * dyaw
        if not c.free_pose(xy, tight) or not c.head_free(xy + G.rot_yaw(yaw)[:2, :2] @ c_b, yaw + phi):
            return False
    return True


def unchecked(c, w):
    """Flakes the vacuum head stood on whose removal nobody saw.

    Flakes the vacuum head stood on whose removal nobody saw (the head hides the floor under it and the base drove on): not looked at yet."""
    return [d for d in w.dust if d.visits >= 1 and not d.checked and d.reseen < P.RESEEN_MIN]


def pending(c, w):
    """The flakes still to visit: never visited, or seen again after a pass.

    The flakes still to visit: never visited, or seen again after a pass; not given up."""
    out = []
    for d in w.dust:
        if d.pid in c.hard or d.visits >= P.MAX_VISITS:
            continue
        if d.visits == 0 or d.reseen >= P.RESEEN_MIN:
            out.append(d)
    return out
