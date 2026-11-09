"""Phases that find the bowls and the vegetables: settle, scan (turn in place a full turn, stopping now and then so the views are taken
standing), survey (drive to the middle of the free floor seen and scan again when something is still missing)."""
from __future__ import annotations

import math

import numpy as np

from core.skills_lib import nav_grid

from .. import ik
from .. import params as P
from ..world import World


def step_settle(c, action, q, w: World) -> None:
    worst = c.to_rest(action, q)
    if c.age == 1:
        ik.start()            # IK_BACKEND gpu|cpu printed once the worker is ready; raises without a GPU solver (BIMBO_IK_REQUIRE_GPU=1)
    if c.age >= P.SETTLE_STEPS and (worst < 0.05 or c.age > P.SETTLE_MAX):
        c.scan_yaw0, c.turned = float(w.base[2]), 0.
        c.change('scan', 'settled')


def missing(c, w: World) -> list:
    from .pick import distinct_bowls
    out = []
    if len(distinct_bowls(w)) < 3:
        out.append('bowls')
    if len([b for b in w.baskets if b.n >= P.BASKET_MIN_SIGHTINGS]) < 2:
        out.append('baskets')
    return out


def step_scan(c, action, q, w: World) -> None:
    """Turn in place in steps: SCAN_TURN_CTRL controls turning, SCAN_PAUSE_CTRL standing (detection runs only while the base stands)."""
    c.to_rest(action, q)
    if c.scan_yaw0 is None:
        c.scan_yaw0, c.turned = float(w.base[2]), 0.
    c.turned += abs(nav_grid.wrap(float(w.base[2]) - c.scan_yaw0))
    c.scan_yaw0 = float(w.base[2])
    phase_ctrl = c.age % (P.SCAN_TURN_CTRL + P.SCAN_PAUSE_CTRL)
    if phase_ctrl < P.SCAN_TURN_CTRL:
        action[2] = P.SCAN_RATE / nav_grid.BASE_TURN
    gone = missing(c, w)
    early = not gone and c.variant != 'scan' and phase_ctrl == P.SCAN_TURN_CTRL + P.SCAN_PAUSE_CTRL - 1   # everything found: stop at a pause end
    if c.turned < P.SCAN_TURN_RAD and not early:
        return
    c.counts['scans'] = c.counts.get('scans', 0) + 1
    if c.variant == 'scan':
        c.change('done', f'scan only: {len(w.bowls)} bowls, {len(w.vegs)} vegetables, {len(w.baskets)} baskets')
    elif not gone:
        c.change('plan_veg', f'scanned: {len(w.bowls)} bowls, {len(w.vegs)} vegetables, {len(w.baskets)} baskets')
    elif c.counts['scans'] > P.MAX_SCANS:
        c.change('plan_veg', f'still missing {gone} after {c.counts["scans"]} scans')
    else:
        c.change('survey', f'missing {gone}')


def survey_goal(c, w: World):
    """The free floor cell seen farthest from the places scanned from, within SURVEY_MAX_M of the base and with a route."""
    c.build_grid(w, [w.base[:2] + P.SURVEY_MAX_M, w.base[:2] - P.SURVEY_MAX_M])
    seen = c.perc.vox.floor_seen
    best, far = None, -1.
    g = c.inflated
    for i in range(0, g.shape[0], 4):
        for j in range(0, g.shape[1], 4):
            if g.blocked[i, j]:
                continue
            xy = g.world((i, j))
            k = c.perc.vox.cell2(xy)
            if not seen[k]:
                continue
            d = c.route(xy)
            if not np.isfinite(d) or d > P.SURVEY_MAX_M or d < P.SURVEY_MIN_M:
                continue
            gap = min(float(np.linalg.norm(xy - v)) for v in c.scanned) if c.scanned else d
            if gap > far:
                best, far = xy, gap
    return best


def step_survey(c, action, q, w: World) -> None:
    c.to_rest(action, q)
    if c.age == 1:
        c.scanned.append(w.base[:2].copy())
        c.survey_to = survey_goal(c, w)
        if c.survey_to is None:            # scan again where it stands (failing here ended 301 at 9114 with half the time left, t20f19)
            c.scan_yaw0, c.turned = float(w.base[2]), 0.
            c.change('scan', 'nowhere new to survey from: scan here')
            return
    goal = c.survey_to
    yaw = math.atan2(goal[1] - w.base[1], goal[0] - w.base[0])
    dist, yaw_err = c.navigate(action, w, goal, yaw if np.linalg.norm(goal - w.base[:2]) > 0.3 else float(w.base[2]))
    if dist < 0.3 or c.age > P.NAV_TIMEOUT:
        c.scan_yaw0, c.turned = float(w.base[2]), 0.
        c.change('scan', f'at the survey point {np.round(goal, 2).tolist()}')


def view_stand(c, w: World, centre: np.ndarray):
    """A base pose VIEW_RINGS from a basket centre facing it (the head, 40 degrees down, then sees into it from 60-65 degrees above), free,
    clear of the obstacles, with the shortest route; None without one."""
    best = None
    for r in P.VIEW_RINGS:
        for bearing in np.arange(P.N_BEARINGS) * 2 * math.pi / P.N_BEARINGS:
            xy = centre[:2] + r * np.array([math.cos(bearing), math.sin(bearing)])
            if not c.free_pose(xy) or c.grid_raw.clearance(xy) < P.VEG_STAND_CLEAR_M:
                continue
            d = c.route(xy)
            stand = np.array([xy[0], xy[1], nav_grid.wrap(bearing + math.pi)])
            if np.isfinite(d) and d < P.VIEW_MAX_ROUTE_M and (best is None or d < best[0]) and c.sees(stand, np.r_[centre[:2], 0.10]):
                # (the island hid basket 1, t20f5; a view stand 7 m round was not reached in 1350 controls, t20f16)
                best = (d, stand)
    return None if best is None else best[1]


def step_plan_view(c, action, q, w: World) -> None:
    c.to_rest(action, q)
    unseen = [k for k, b in enumerate(w.baskets) if k not in c.looked and b.n >= P.BASKET_MIN_SIGHTINGS]
    if not unseen:
        c.change('plan_veg', 'every basket looked into')
        return
    unseen.sort(key=lambda k: float(np.linalg.norm(w.baskets[k].centre - w.base[:2])))
    k = unseen[0]
    c.build_grid(w, [w.baskets[k].centre])
    c.view = (k, view_stand(c, w, w.baskets[k].centre))
    if c.view[1] is None:
        c.looked.add(k)
        c.change('plan_veg', f'no stand to look into basket {k} from')
        return
    c.change('go_view', f'to look into basket {k} at {np.round(w.baskets[k].centre, 2).tolist()} from {np.round(c.view[1], 2).tolist()}')


def step_go_view(c, action, q, w: World) -> None:
    c.to_rest(action, q)
    k, stand = c.view
    dist, yaw_err = c.navigate(action, w, stand[:2], stand[2])
    if c.arrived(dist, yaw_err) or c.age > c.nav_budget(stand[:2]):
        c.change('look_basket', f'at the view stand of basket {k}, {dist:.2f} m off')


def step_look_basket(c, action, q, w: World) -> None:
    """Stand still: detections every control (the planner perceives every control in the look phases)."""
    c.to_rest(action, q)
    if c.age < P.LOOK_STEPS * 2:
        return
    k, _ = c.view
    c.looked.add(k)
    kinds = w.baskets[k].kinds if k < len(w.baskets) else {}
    c.change('plan_veg', f'basket {k} holds {kinds}')


PHASES = {'settle': step_settle, 'scan': step_scan, 'survey': step_survey, 'plan_view': step_plan_view, 'go_view': step_go_view,
          'look_basket': step_look_basket}
