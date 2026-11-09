"""putting_dishes_away_after_cleaning: the eight plates from the bar and the countertop onto the shelves of the pantry's
east bay (bottom_cabinet_fancyy_0 = cabinet.n.01_2), one plate at a time.

    localise (trash_observed.localize); open the pantry's east door (pantry.open_door, the hinge skill);
    look at the bar and the countertop from the aisle between them (plates = things 2-5 cm high and 0.1-0.4 m
    across on those surfaces, observer.floor_things);
    each plate, nearest the pantry first: drive to the aisle's middle opposite it, straight in to a stance 0.36 m
    from the surface's edge, look again, pinch the near rim (fingers down), lift, draw in level, straight back to the
    aisle's middle; drive to the pantry; set it down on a shelf of the east bay (a pile per shelf), back out.

Poses are (x, y, yaw) in the map frame M. Nothing here reads simulator state.
"""
import time
import numpy as np
from manip import hold, follow, move_to, drive, pose2, BASE_Z, wrap
from skills import interpolate_cartesian
from trash_skills import tuck
import dishes as D
from trash_observed import turn_in_place, localize
import pantry as P

AISLE_Y = -.945                     # the middle of the aisle between the bar (y -0.305) and the countertop (-1.585)
AISLE_X = (4.95, 8.45)              # where on it the base may stand
BACK = .36                          # base centre this far from a surface's edge when a plate is pinched
LATERALS = (0., .10, -.10, .20, -.20, .30, -.30)
SURVEY_X = (5.00, 6.20, 7.40, 8.30)   # (from the aisle's middle the head camera shows a plate on the bar or the countertop within about 0.7 m to either side)
MERGE = .12                         # sightings this close are one plate
PLACE_STANCES = ((8.50, -1.15, -1.05), (8.55, -1.10, -1.31), (8.55, -1.20, -1.57))   # west of the open door's leaf (offline: arm planner with the leaf at 57 deg)
PLATES = 8
# Where the k-th plate goes: (shelf, pile under it, how far the hand pushes it in beyond where it was let go, tried in
# order). Two plates a shelf, one behind the other: the first pushed 0.22 m in (to the back), the next let go in
# front of it. A plate let go over another one at the bay's front tipped forwards off the shelf (r3 301, twice).
SLOTS = ((1.162, 0., (.22, .15, .07)), (1.162, 0., (.07,)), (.849, 0., (.22, .15, .07)), (.849, 0., (.07,)),
         (1.479, 0., (.22, .15, .07)), (1.479, 0., (.07,)), (1.162, .03, (.07,)), (.849, .03, (.07,)))
TURN_RATE = .45                     # turns on the spot at this commanded rate
LOCK_X = 6.40
FREEZE_EAST = 7.60                  # east of this the map pose is kept by dead reckoning alone: there the scans matched the map
                                    # poorly (score 0.6 against 0.98 elsewhere) and walked the pose 0.3-0.8 m off (r1 301)


def freeze_east(rt):
    loc = rt.ob.loc; original = loc.correct

    def correct(scan, min_points=40):
        if loc.pose is not None and (loc.pose[0] > FREEZE_EAST or not -1.12 < loc.pose[1] < -.78):
            return False
        # between LOCK_X and FREEZE_EAST the view along the aisle does not hold x: x is dead-reckoned there, y and
        # the heading are corrected (arriving at the door the pose was 0.17 m off along the aisle, r7 301)
        loc.lock = np.array([1., 0.]) if loc.pose is not None and loc.pose[0] > LOCK_X else None
        return original(scan, min_points)
    loc.correct = correct
STOP_AT = 15600                     # no new trip after this step (horizon 16430)


class Plates:
    """Plates seen on the bar and the countertop, in the map: list of dict(xy, r, surface, n, tries)."""

    def __init__(self):
        self.items = []; self.gone = []         # places plates were taken from (nothing is looked for there again)

    def add(self, xy, width, surface):
        xy = np.asarray(xy, float)
        if any(float(np.linalg.norm(xy - g)) < MERGE for g in self.gone):
            return
        for it in self.items:
            if float(np.linalg.norm(it['xy'] - xy)) < MERGE:
                it['xy'] = (it['xy'] * it['n'] + xy) / (it['n'] + 1); it['n'] += 1; it['w'] = max(it['w'], width); return
        self.items.append(dict(xy=xy, w=width, surface=surface, n=1, tries=0))

    def take(self, item):
        self.gone.append(np.asarray(item['xy'], float).copy())
        self.items = [i for i in self.items if i is not item]


def surface_of(xy):
    for name, S in D.SURFACES.items():
        if S['x'][0] - .05 < xy[0] < S['x'][1] + .05 and S['y'][0] - .05 < xy[1] < S['y'][1] + .05:
            return name
    return None


def radius_of(item):
    return float(np.clip(item['w'] / 2., .115, .16))


def sightings(rt, frames=4):
    """What the head camera shows on the bar and the countertop now: [(xy, width, surface)]."""
    ctx, est = rt.ctx, rt.est; out = []
    for _ in range(frames):
        yield from hold(ctx, 3)
        for t in rt.ob.floor_things(est.base()):
            if t['kind'] == 'soda' and not t['cut']:
                where = surface_of(t['xy'])
                if where is not None:
                    out.append((np.asarray(t['xy'], float), float(t['width']), where))
                    rt.last_near = getattr(rt, 'last_near', [])
                    rt.last_near.append((np.asarray(t['xy'], float), np.asarray(t.get('near', t['xy']), float), np.asarray(est.base()[:2], float)))
    return out


RELOCALIZE_AT = (5.60, -.945)
RETRY_SHIFTS = (-.02, .025)          # after a missed pinch
AISLE_SLIDE = False                 # (off: with it the pose went 0.16-0.29 m and 20 deg off at the east end, r5 302, r6 301)
TURN_X = (5.00, 7.40)               # along the aisle's middle the base turns only between these x (the map pose is corrected there)


def go(rt, xy, yaw, tolerance=.04):
    """Drive to (xy, yaw) along a planned route; a large turn on the spot is made first and slowly, where the base
    stands (the map pose kept better). From the aisle's middle to the aisle's middle: turn west of TURN_X[1] (where
    the scans correct the pose), then slide sideways (east of 7.6 the pose is dead reckoning only: a turn there put
    it 8-10 cm off, and two plates were let go short of the shelf and fell, r4 301)."""
    ctx = rt.ctx; here = rt.est.base(); xy = np.asarray(xy, float)
    if AISLE_SLIDE and abs(here[1] - AISLE_Y) < .07 and abs(xy[1] - AISLE_Y) < .07 and AISLE_X[0] - .2 < xy[0] < 8.7 and 4.7 < here[0] < 8.7:
        if abs(wrap(yaw - here[2])) > .15:
            turn_x = float(np.clip(here[0], TURN_X[0], TURN_X[1]))
            if abs(here[0] - turn_x) > .03:
                yield from straight(rt, (turn_x, AISLE_Y), here[2], speed=.3, tolerance=.03)
            yield from turn_in_place(rt, float(wrap(yaw - rt.est.base()[2])), rate=TURN_RATE)
        status = yield from straight(rt, xy, yaw, speed=.3, tolerance=max(tolerance, .02))
        return status
    if float(np.linalg.norm(xy - here[:2])) < 1.0 and abs(wrap(yaw - here[2])) > .6 and -1.585 + .55 < here[1] < -.305 - .55:
        yield from turn_in_place(rt, float(wrap(yaw - here[2])), rate=TURN_RATE)
    status = yield from drive(ctx, xy, yaw, speed=.4, tolerance=tolerance, yaw_tolerance=.04, max_steps=1500)
    return status


def straight(rt, xy, yaw, speed=.15, tolerance=.012):
    far = float(np.linalg.norm(np.asarray(xy, float) - np.asarray(rt.est.base()[:2], float)))
    status = yield from drive(rt.ctx, np.asarray(xy, float), yaw, speed=speed, direct=True, tolerance=tolerance, yaw_tolerance=.02, min_speed=.03, min_turn=.05, face_travel=False, max_steps=int(400 + 40 * far / speed))
    return status


def to_aisle(rt, x=None, yaw=None):
    """Straight back to the aisle's middle (keeping the heading), so that the base can turn."""
    x0, y0, yaw0 = rt.est.base()
    if abs(y0 - AISLE_Y) > .05 and -1.585 < y0 < -.305:
        yield from straight(rt, (x0, AISLE_Y), yaw0, speed=.2, tolerance=.03)


def survey(rt, plates, enough=PLATES):
    """Look at the bar and the countertop from the aisle's middle until @enough plates are known (or taken): along
    the bar facing north from the nearer end, one turn, and back along the countertop facing south. Between lookouts
    the base slides sideways (each turn on the spot at the aisle's east end put the map pose 0.3 m off along the
    aisle, r1 301)."""
    ctx, est = rt.ctx, rt.est; seen = 0
    xs = list(SURVEY_X); x0 = est.base()[0]
    if abs(x0 - xs[-1]) < abs(x0 - xs[0]):
        xs = xs[::-1]
    for yaw, row in ((np.pi / 2, xs), (-np.pi / 2, xs[::-1])):
        for k, x in enumerate(row):
            if len(plates.items) + len(plates.gone) >= enough:
                return seen
            yield from to_aisle(rt)
            here = est.base()
            if abs(here[1] - AISLE_Y) < .06 and abs(wrap(here[2] - yaw)) < .1 and abs(here[0] - x) < 1.5:
                status = yield from straight(rt, (x, AISLE_Y), yaw, speed=.3, tolerance=.03)
            else:
                status = yield from go(rt, (x, AISLE_Y), yaw)
            yield from hold(ctx, 6)
            found = yield from sightings(rt)
            for xy, w, where in found:
                plates.add(xy, w, where)
            seen += len(found)
            ctx.event('look', at=[x, round(float(yaw), 2)], status=status, sightings=len(found), plates=[np.round(i['xy'], 3).tolist() for i in plates.items])
    return seen


def stance_for(ctx, item, here=None):
    """Base pose and arm to pinch @item's near rim from: (pose, arm, pitch) or None (arm planner over the map)."""
    S = D.SURFACES[item['surface']]; c = item['xy']; r = radius_of(item)
    y = S['face'] - BACK if item['surface'] == 'bar' else S['face'] + BACK
    for lat in LATERALS:
        pose = (float(c[0] + lat), float(y), float(S['yaw']))
        if not (4.85 < pose[0] < 8.55) or not ctx.scene.base_free(np.array(pose[:2]), margin=-.05):
            continue
        # the hand on the plate's side first (facing north the right hand is east)
        side = (c[0] - pose[0]) * (1. if item['surface'] == 'bar' else -1.)
        for arm in (('right', 'left') if side > 0 else ('left', 'right')):
            ctx.arm = arm
            for pitch in D.PITCHES:
                point, rot = D.rim_grip(pose, c, r, S['top'], pitch)
                if D.plan_pinch(ctx, point, rot, base=pose2(*pose)) is not None:
                    return pose, arm, pitch
    return None


def fetch(rt, plates, item):
    """Take @item from its surface. Returns (arm, held offset in the tool frame) or (None, why)."""
    ctx, est = rt.ctx, rt.est
    item['tries'] += 1
    found = stance_for(ctx, item)
    if found is None:
        ctx.event('fetch', ok=False, reason='no stance', plate=np.round(item['xy'], 3).tolist()); return None, 'no_stance'
    pose, arm, pitch = found
    ctx.event('stance', plate=np.round(item['xy'], 3).tolist(), r=round(radius_of(item), 3), pose=np.round(pose, 3).tolist(), arm=arm, pitch=pitch)
    yield from to_aisle(rt)
    status = yield from go(rt, (pose[0], AISLE_Y), pose[2])
    if status != 'arrived':
        ctx.event('fetch', ok=False, reason='no way to the aisle', status=status); return None, 'no_way'
    # a fresh look from the aisle's middle (the plate 0.9-1.1 m off, whole in the image): its place relative to the
    # base as it stands now, whatever the map pose's error
    yield from hold(ctx, 6)
    rt.last_near = []
    near = yield from sightings(rt, frames=3)
    close = [(xy, w) for xy, w, where in near if float(np.linalg.norm(xy - item['xy'])) < .25]
    if close:
        best = min(close, key=lambda c: float(np.linalg.norm(c[0] - item['xy'])))
        group = [(xy, w) for xy, w in close if float(np.linalg.norm(xy - best[0])) < .06]
        item['w'] = float(np.median([w for _, w in group]))
        # the middle from the plate's near edge (well seen) and its radius: the patch's middle came out 4.5 cm
        # towards the camera on a countertop plate, and the pinch missed its rim (r3 301)
        edges = [(nr, bxy) for xy, nr, bxy in rt.last_near if float(np.linalg.norm(xy - best[0])) < .06]
        if edges:
            nr = np.mean([e for e, _ in edges], axis=0); bxy = edges[0][1]; u = (nr - bxy) / max(float(np.linalg.norm(nr - bxy)), 1e-6)
            centre = nr + radius_of(item) * u
        else:
            centre = np.mean([xy for xy, _ in group], axis=0)
        moved = centre - item['xy']
        item['xy'] = centre
        ctx.event('aisle_look', moved=np.round(moved, 3).tolist(), seen=len(group))
        if float(np.linalg.norm(moved)) > .04:
            again = stance_for(ctx, item)
            if again is not None:
                pose, arm, pitch = again
                yield from straight(rt, (pose[0], AISLE_Y), pose[2], speed=.2, tolerance=.02)
    status = yield from straight(rt, pose[:2], pose[2])
    yield from hold(ctx, 6)
    close = []      # (no second look from the stance: from 0.5 m the patch's middle is biased towards the camera)
    S = D.SURFACES[item['surface']]; base = est.base(); plan = None
    for a in (arm, 'left' if arm == 'right' else 'right'):
        ctx.arm = a
        for p in (pitch,) + tuple(x for x in D.PITCHES if x != pitch):
            point, rot = D.rim_grip(base, item['xy'], radius_of(item), S['top'], p)
            plan = D.plan_pinch(ctx, point, rot)
            if plan is not None:
                arm = a; break
        if plan is not None:
            break
    ctx.event('relook', seen=len(close), plate=np.round(item['xy'], 3).tolist(), w=round(item['w'], 3), plan=plan is not None)
    if plan is None:
        yield from to_aisle(rt); return None, 'no_plan'
    ctx.arm = arm
    ok = yield from D.pinch(ctx, plan)
    used = 0.
    for shift in RETRY_SHIFTS:
        if ok:
            break
        # missed (fingers closed on nothing: 4 of 12 pinches in r4 301, all on countertop plates): again from where
        # the base stands, the plate's middle taken this much further away (+) or nearer (-)
        here = est.base(); u = item['xy'] - np.asarray(here[:2], float); u = u / max(float(np.linalg.norm(u)), 1e-6)
        point, rot = D.rim_grip(here, item['xy'] + shift * u, radius_of(item), S['top'], D.PITCHES[0])
        again = D.plan_pinch(ctx, point, rot)
        if again is None:
            continue
        ok = yield from D.pinch(ctx, again); used = shift
        ctx.event('repinch', shift=shift, ok=ok)
    if not ok:
        yield from tuck(ctx, (arm,))
        yield from to_aisle(rt)
        return None, 'missed'
    item['xy'] = item['xy'] + used * (item['xy'] - np.asarray(est.base()[:2], float)) / max(float(np.linalg.norm(item['xy'] - np.asarray(est.base()[:2], float))), 1e-6)
    tip = ctx.tip(); centre = np.r_[D.in_base(est.base(), item['xy']), S['top'] + .012 + D.LIFT - BASE_Z]
    held = tip[:3, :3].T @ (centre - tip[:3, 3])
    yield from D.carry(ctx, arm)
    yield from to_aisle(rt)
    if not D.holds(ctx, arm):
        ctx.set_grip(1., arm); yield from tuck(ctx, (arm,))
        ctx.event('fetch', ok=False, reason='dropped'); return None, 'dropped'
    plates.take(item)
    ctx.event('fetch', ok=True, arm=arm, held=np.round(held, 3).tolist())
    return arm, held


def deliver(rt, arm, held, count):
    """Set the plate in @arm down on a shelf of the east bay. @count plates were set down before."""
    ctx, est = rt.ctx, rt.est; ctx.arm = arm
    shelf, pile, push = SLOTS[min(count, len(SLOTS) - 1)]
    for k, stance in enumerate(PLACE_STANCES):
        yield from to_aisle(rt)
        status = yield from go(rt, (stance[0], AISLE_Y), stance[2])
        status = yield from straight(rt, stance[:2], stance[2])
        yield from hold(ctx, 6)
        ok = yield from D.place(ctx, est, arm, held, shelf + pile, push=push)
        if ok:
            return True
        if not D.holds(ctx, arm):
            break
        yield from D.carry(ctx, arm)
    ctx.set_grip(1.); yield from tuck(ctx, (arm,))
    ctx.event('deliver', ok=False)
    return False


def door_seen(rt):
    """Head-camera points where the east door's leaf stands when it is open (about 90 deg: along x 9.3 north of the
    hinge) and where it stands shut (across the bay's front): dict(open=count, closed=count). Taken from the push
    stance, facing the pantry."""
    ctx, est = rt.ctx, rt.est
    yield from hold(ctx, 8)
    pts = rt.ob.points('head', stride=2); z = pts[:, 2] + BASE_Z
    pts = pts[(z > .30) & (z < 1.80)]
    world = pose2(*est.base()); xy = pts[:, :2] @ world[:2, :2].T + world[:2, 3]
    i, j = rt.ob.scene.cell(xy); ok = rt.ob.scene.inside(i, j)
    top = rt.ob.scene.top[np.clip(i, 0, rt.ob.scene.shape[0] - 1), np.clip(j, 0, rt.ob.scene.shape[1] - 1)]
    xy = xy[ok & (top < .10)]                   # (walls and furniture the map has are not the leaf)
    open_zone = (xy[:, 0] > 9.16) & (xy[:, 0] < 9.45) & (xy[:, 1] > -1.45) & (xy[:, 1] < -1.05)
    shut_zone = (xy[:, 0] > 8.80) & (xy[:, 0] < 9.20) & (xy[:, 1] > -1.62) & (xy[:, 1] < -1.50)
    return dict(open=int(open_zone.sum()), closed=int(shut_zone.sum()))


def full_task(rt):
    ctx, est = rt.ctx, rt.est
    freeze_east(rt)
    yield from tuck(ctx)
    plates = Plates()
    yield from survey(rt, plates)
    # A fresh global fix at the aisle's west part, where the start fixes were right (the survey's turn at the east
    # end once left the pose 0.16 m off along the aisle for good, r5 302)
    yield from to_aisle(rt)
    yield from go(rt, RELOCALIZE_AT, np.pi / 2)
    before = tuple(np.round(est.base(), 3))
    ok = yield from localize(rt)
    ctx.event('relocalize', ok=bool(ok), before=list(before), after=list(np.round(est.base(), 3)))
    opened = 0.
    for attempt in range(2):
        yield from to_aisle(rt)
        opened = yield from P.open_door(ctx, est, go=lambda xy, yaw: go(rt, xy, yaw))
        if opened > np.radians(15.):
            break
    # the hand on the handle leads the door to 23-54 deg only: then it is pushed open from inside
    yield from to_aisle(rt)
    pushed = yield from P.push_open(ctx, est, go=lambda xy, yaw: go(rt, xy, yaw), led=opened if opened > 0 else None)
    seen = yield from door_seen(rt)
    ctx.event('door', led=round(float(np.degrees(opened)), 1), pushed=round(float(np.degrees(pushed)), 1), seen=seen)
    if seen['closed'] > seen['open']:
        # the door did not follow the hand (r6 301: the handle held, the lead went to 68 deg, the door stayed shut):
        # once more, both stages
        yield from to_aisle(rt)
        opened = yield from P.open_door(ctx, est, go=lambda xy, yaw: go(rt, xy, yaw))
        yield from to_aisle(rt)
        pushed = yield from P.push_open(ctx, est, go=lambda xy, yaw: go(rt, xy, yaw), led=opened if opened > 0 else None)
        seen = yield from door_seen(rt)
        ctx.event('door', again=True, led=round(float(np.degrees(opened)), 1), pushed=round(float(np.degrees(pushed)), 1), seen=seen)
    yield from to_aisle(rt)
    placed = 0; looks = 1
    while placed < PLATES and ctx.steps < STOP_AT:
        todo = [i for i in plates.items if i['tries'] < 2]
        if not todo:
            if looks >= 3:
                break
            looks += 1
            yield from survey(rt, plates)
            continue
        here = np.asarray(est.base()[:2], float)
        item = min(todo, key=lambda i: float(np.linalg.norm(i['xy'] - here)) + .3 * i['tries'])
        arm, held = yield from fetch(rt, plates, item)
        if arm is None:
            continue
        ok = yield from deliver(rt, arm, held, placed)
        placed += int(bool(ok))
        ctx.event('progress', placed=placed, known=len(plates.items))
    return 'done'
