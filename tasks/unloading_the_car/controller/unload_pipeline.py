"""Unloading the car as a sequence of skills, driven by an estimator.

The robot starts in the garage. A briefcase and a satchel lie in the car's trunk, which is shut. At the end both
must be next to the sofa in the living room.

  1. through the door to the corridor and back, with empty hands: the chassis pushes the door open and it stays
     open (a bag carried in front of the chest would meet the leaf before the chassis does);
  2. open the trunk lid (as for loading the car);
  3. for each bag: stand behind the car, look into the trunk, take the bag by its near side, draw it back from
     under the parcel shelf and lift it out; carry it to the living room; let it down on the floor east of the
     sofa's chaise.

"Next to the sofa", as the simulator decides it: the gap between the two bounding boxes is at most a sixth of the
mean of their sides (0.36 m for the briefcase, 0.39 m for the satchel), and a level ray from the middle of one, in
one of sixteen directions, hits the other. A bag on the floor within 0.3 m east of the chaise has both.

The estimator supplies base() = (x, y, yaw) in M, hold_axis(axis), settle(flag), fix_at_lid(), fix_at_bumper(),
lid_shows_closed(), and what it knows of the bags (see bags.py).
"""
import time
import numpy as np
from manip import hold, follow, move_to, pose2, BASE_Z, TALL_TRUNK
from skills import interpolate_cartesian
from car_skills import tuck, held, LID_MIDDLE, UPRIGHT
from car_pipeline import route, straight, settle, open_trunk, leave_garage, back_to_garage, LID_STAGING, ENTRY, NORTH
from trunk_view import floor_height
import bags

# Behind the car (map frame). The base stands square to the car, its front 8 cm from the bumper, and is moved along
# the car so that a shoulder faces the edge the hand is going to take.
STANCE_X = -7.17
STEP_BACK = .22                     # ... and this much further from the car for a bag that lies on the sill
NEAR_SILL = -6.62                   # a bag whose middle is nearer to the robot than this (map x) lies on the sill or against it
SHOULDER = .19                      # a shoulder's distance from the base's middle line
ALONG = .60                         # how far from the car's middle line the stance may be
SHIFTS = (0., .12, -.12, .24, -.24, .36, -.36)

# Fixed places of the scene (map frame).
SOFA_EDGE = 2.96                    # the chaise's east face (the sofa's bounding box ends there)
SOFA_STANCE = (3.90, 6.70, np.pi)   # facing the chaise from the east, its edge 0.7 m ahead of the chassis; a wall shelf (0.87-1.30 m up) hangs north of y = 7.22
SOFA_STAGING = (4.50, 6.70)         # 0.6 m behind it: room to turn with a bag in the hand
LET_DOWN_AHEAD = .58                # the fingers' ends reach this far ahead of the base before the bag goes down: the bag then lies clear of the chassis, 0.1-0.3 m from the sofa
WIDE = .36                          # half the width a bag may take in front of the chest on the way (the chassis is 0.345; the doorway is 0.41 at chassis height and 0.375 at the door handle's)


def open_door(ctx, est):
    """Push the door between garage and corridor open with the chassis and come back into the garage. Returns
    'in_garage' when the base is back."""
    status = yield from leave_garage(ctx, est)
    if status != 'in_corridor':
        return status
    return (yield from back_to_garage(ctx, est))


def to_sofa(ctx, est):
    """From the corridor to the stance east of the sofa's chaise."""
    status = yield from route(ctx, est, ENTRY, NORTH, tolerance=.10)
    status = yield from route(ctx, est, SOFA_STAGING, SOFA_STANCE[2], tolerance=.05)
    if status != 'arrived':
        return 'sofa_' + status
    yield from settle(ctx, est, 8)
    status = yield from straight(ctx, SOFA_STANCE[:2], SOFA_STANCE[2], speed=.2, tolerance=.03)
    return status


def let_down(ctx, est, arm):
    """Reach forward with the bag @arm's hand holds, lower it as the hand holds it until its low end is just above
    the floor, and let it go; the hand then draws back along its fingers before the arm folds. Returns True when
    the hand has let go (from where it was, if the arm found no way down: the bag then falls).

    (Let down close in front of the chassis and with the arm folded straight after, the satchel was dragged 0.4 m
    back from the sofa by the hand, which still had its fingers astride the bag's edge.)"""
    ctx.arm = arm; check = ctx.validator(relax_from=ctx.q); tip = ctx.tip(); here = tip[:3, 3].copy(); rotation = tip[:3, :3].copy()
    high = bags.low_end(ctx, arm) if arm in ctx.payload else .3; way = None
    for ahead in (LET_DOWN_AHEAD, LET_DOWN_AHEAD - .07, LET_DOWN_AHEAD - .14, here[0]):
        forward = np.array([max(ahead, here[0]), here[1], here[2]])
        out = interpolate_cartesian(ctx.ik, ctx.q, here, forward, rotation, validator=check, step=.03) if forward[0] > here[0] + .01 else []
        if out is None:
            continue
        full = ctx.q.copy()
        if out:
            full[ctx.ik.columns] = out[-1]
        for share in (1., .8, .6, .4):
            drop = max(0., high - .05) * share
            down = interpolate_cartesian(ctx.ik, full, forward, forward - [0., 0., drop], rotation, validator=check, step=.03) if drop > .03 else None
            if down is not None:
                way = out + down; break
        if way is not None:
            break
    status = 'no_way'
    if way is not None:
        status = yield from follow(ctx, way, rate=.02, max_steps=700)
        yield from hold(ctx, 6)
    low = bags.low_end(ctx, arm) if arm in ctx.payload else None; reached = ctx.tip()[:3, 3].copy()
    ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
    yield from hold(ctx, 30)
    # Out of the bag's edge the way the fingers point, then up, before the arm folds.
    tip = ctx.tip(); back = interpolate_cartesian(ctx.ik, ctx.q, tip[:3, 3], tip[:3, 3] + .10 * tip[:3, 2] + [0., 0., .06], tip[:3, :3], step=.03)
    if back is not None:
        yield from follow(ctx, back, rate=.02, max_steps=200)
    ctx.event('let_down', ok=status == 'done', arm=arm, way=status, from_height=round(high, 3), low_end_above_floor=None if low is None else round(low, 3), tip=np.round(reached, 3).tolist())
    yield from tuck(ctx, (arm,))
    return True


# ---- at the car ---------------------------------------------------------------------------------------------
def to_car(ctx, est):
    """To the staging point behind the car, the pose taken from the bumper there (the lid is open); from here to
    the stance and back the position is kept by dead reckoning."""
    status = yield from route(ctx, est, LID_STAGING, 0., tolerance=.05)
    if status != 'arrived':
        return 'car_' + status
    yield from settle(ctx, est, 16)
    shift = est.fix_at_bumper()
    ctx.event('bumper_fix', shift=None if shift is None else np.round(shift, 4).tolist())
    est.hold_axis('xy')
    return 'arrived'


def look_into_trunk(ctx, est, where):
    """A look into the trunk. From the staging point the whole trunk is in view: what was known before is dropped
    (a bag that was taken out, or that moved when the other one was, is not where it was)."""
    yield from hold(ctx, 4)
    if where == 'staging':
        est.bag.clear()
    found = est.look_into_trunk()
    # A bag that has been taken out is not in the trunk, whatever the look makes of what is left there (what the
    # satchel's shape left unexplained was taken for the briefcase on 3 looks of 380 after the briefcase had gone:
    # as an obstacle beside the satchel it could have stood in the hand's way).
    for name in getattr(ctx, 'gone', ()):
        est.bag.pop(name, None); found[name] = None
    ctx.event('trunk_look', where=where, **{name: None if known is None else dict(at=np.round(known['origin'], 3).tolist(), how=known['how'], tilt=round(float(np.degrees(known['tilt']))), on=known['on'], share=round(known['share'], 2)) for name, known in found.items()})
    return found


def in_frame_of(stance, found):
    """An edge (bags.edges, in M) in the base frame of a base standing at @stance = (x, y, yaw)."""
    inverse = np.linalg.inv(pose2(*stance)); turn = inverse[:3, :3]
    return dict(found, point=turn @ found['point'] + inverse[:3, 3], top=turn @ found['top'], side=turn @ found['side'], inward=float((found['inward'] - stance[2] + np.pi) % (2 * np.pi) - np.pi))


SIDE_WALLS = (6.33, 7.47)           # of the trunk (map y)


SEARCH_SECONDS = 40.                # the longest one search for a stance may hold up a step of the policy


def plan_bag(ctx, est, name, tried=(), hopeless=None, spent=()):
    """How bag @name is taken from behind the car: dict(arm, stance (x, y, yaw in M), edge (which of the bag's
    edges: the id bags.edges gives it)) for the first of the bag's edges, the nearest first, for which the arm planner
    has the slanted pinch from a stance there; None if it has none. tried: keys (arm, edge, shift, stance x) to leave
    out. hopeless: a set that takes the keys for which the planner had nothing, and leaves them out the next
    time (a search that finds nothing for an edge takes the arm planner a minute). spent: (arm, edge) from which the
    hand did not get the bag before: those come after all the others. A search that takes longer than SEARCH_SECONDS
    stops (a satchel with no stance took 100 s in one step): what it had no time for is left for the next visit."""
    known = est.bag[name]; started = time.time()
    # Among the edges a hand comes onto askew, the one with most room to the trunk's side walls first (on the end by
    # the wall the hand was stopped on its way 2 times of 4, on the other end it held 2 of 2).
    edges = sorted(bags.edges(name, known['origin'], known['turn']), key=lambda found: (abs(found['inward']) > 1.9, bool(found['skew']),
                   -min(found['point'][1] - SIDE_WALLS[0], SIDE_WALLS[1] - found['point'][1]) if found['skew'] else found['point'][0] + .25 * abs(found['inward'])))
    # (The hand points across the edge into the bag: to the right of straight ahead it is the left hand's reach, to the
    # left the right hand's: a left hand that reached round to point left shoved the bag away.)
    # A bag that lies on the sill or just behind it is too near for a hand from the stance at the bumper: from STEP_BACK
    # further back then.
    rows = (STANCE_X - STEP_BACK, STANCE_X) if known['origin'][0] < NEAR_SILL else (STANCE_X,)
    # The stance where a shoulder faces the edge first, for every edge and hand; then the stances to its sides (3 of
    # 280 starts want one of those; gone through edge by edge they took a minute before a hand askew got its turn).
    # Each edge with the hand it is for before any edge with the other hand.
    order = [(found, bags.arms_for(found['inward'])[other], x, shift) for again in (False, True) for aside in (False, True) for other in (0, 1) for found in edges for x in rows
             for shift in (SHIFTS[:3] if found['skew'] or x != STANCE_X else SHIFTS) if ((bags.arms_for(found['inward'])[other], found['id']) in spent) == again and (shift != 0.) == aside]
    for found, arm, x, shift in order:
        y = float(np.clip(found['point'][1] - (SHOULDER if arm == 'left' else -SHOULDER) + shift, LID_MIDDLE - ALONG, LID_MIDDLE + ALONG)); stance = (x, y, 0.)
        key = (arm, found['id'], shift, x)
        if key in tried or (hopeless is not None and key in hopeless):
            continue
        if time.time() - started > SEARCH_SECONDS:
            return None
        edge = in_frame_of(stance, found); ctx.arm = arm
        if bags.plan_pinch(ctx, edge['point'], edge['top'], edge['side'], base=pose2(*stance), below=edge['below'], behind=edge['behind'], skew=edge['skew']) is not None:
            return dict(arm=arm, stance=stance, edge=found['id'], inward=found['inward'], shift=shift, key=key)
        if hopeless is not None:
            hopeless.add(key)
    return None


def bags_as_obstacles(ctx, est, taking=None):
    """What is known of the bags goes into the collision model: the arm must not sweep through a bag on its way to
    an edge (a forearm that did shoved the satchel 13 cm away). The hand may touch the bag it is to take."""
    ctx.obstacles = {}
    for name, known in est.bag.items():
        # (A briefcase that lies on the satchel has the satchel's top a finger's breadth under its rim: the hand that
        # takes it may come as near to the satchel as to the briefcase.)
        points, radii = est.shapes.obstacle(name, known['origin'], known['turn']); ctx.obstacles[name] = (points, radii, name == taking or (taking == 'briefcase' and on_the_satchel(est)))


def edge_in_base(est, name, which):
    """Edge @which (its id in bags.edges) of a bag, where the bag is believed to lie, in the base frame; None if the
    bag as it is seen now has no such edge (a briefcase that leant and now lies flat, or the other way round).
    (By its id, not by its place in the list: the order of two edges that are about as near changed between the
    look from the staging point and the one from the stance, and the hand was sent to the other edge.)"""
    known = est.bag[name]
    for found in bags.edges(name, known['origin'], known['turn']):
        if found['id'] == which:
            return in_frame_of(est.base(), found)
    return None


def carry_bag(ctx):
    """The bag the active hand holds to a posture it travels in: the trunk upright, the bag hanging ahead of and
    below the hand in front of the chest, no wider than the chassis. Returns the status of the move, or 'no_pose'."""
    side = 1. if ctx.arm == 'left' else -1.; free = ctx.validator(relax_from=ctx.q); arm = ctx.arm

    def check(full, poses):
        if not free(full, poses):
            return False
        points, size = ctx.carried(poses, arm)
        return float(np.max(np.abs(points[:, 1]) + size)) <= WIDE
    for turn, spin in bags.CARRY_ATTITUDES:
        rotation = bags.carry_rotation(np.radians(turn), np.radians(side * spin))
        for x, y, z in bags.CARRY_SPOTS:
            goal = ctx.ik.solve(ctx.q, np.array([x, side * y, z - BASE_Z]), rotation, count=4, validator=check, locked=UPRIGHT)
            if goal is None:
                continue
            status = yield from move_to(ctx, goal, rate=.03, label='carry_bag', max_steps=700)
            if status == 'done':
                ctx.event('carry_bag', ok=True, arm=arm, turned=turn, spun=spin, spot=[x, y, z], low_end=round(bags.low_end(ctx, arm), 2))
                return status
    ctx.event('carry_bag', ok=False, reason='no pose', arm=ctx.arm)
    yield from move_to(ctx, np.r_[TALL_TRUNK, ctx.chain[4:]], rate=.03, label='carry_bag')
    return 'no_pose'


def take_bag(ctx, est, name):
    """From behind the car (pose fixed at the staging point, kept by dead reckoning), bag @name having been seen:
    stand where a shoulder faces the bag's nearest edge, take the bag by it, draw it out and hold it for
    travelling; back to the staging point. Returns True when the hand holds the bag, False if it was not taken."""
    got = False; tried = set(); arm = None; spent = ctx.spent.setdefault(name, set())
    # What the planner had nothing for is remembered from visit to visit while the bag lies where it lay.
    origin, turn = est.bag[name]['origin'], est.bag[name]['turn']; before = ctx.hopeless.get(name)
    if before is None or np.linalg.norm(origin - before[0]) > .04 or np.abs(turn - before[1]).max() > .25:
        ctx.hopeless[name] = (origin.copy(), turn.copy(), set())
    hopeless = ctx.hopeless[name][2]
    # Up to three stances: a hand that finds no way to the edge from where the base came to stand (the plan was made
    # for the stance as it should be, and for the bag as it showed from further away) goes on from another one.
    for stand in range(3):
        bags_as_obstacles(ctx, est, taking=name)
        plan = plan_bag(ctx, est, name, tried, hopeless, spent)
        if plan is None:
            ctx.obstacles = {}
            ctx.event('bag', ok=False, bag=name, reason='no stance', stand=stand); break
        arm = ctx.arm = plan['arm']; tried.add(plan['key'])
        ctx.event('bag_plan', bag=name, arm=arm, stance=np.round(plan['stance'], 3).tolist(), edge=plan['edge'], inward=round(plan['inward'], 2), at=np.round(est.bag[name]['origin'], 3).tolist(), how=est.bag[name]['how'], stand=stand)
        yield from straight(ctx, plan['stance'][:2], plan['stance'][2], speed=.2, tolerance=.015)
        yield from hold(ctx, 6)
        # From here the camera shows the bag from above and nearer: the hand goes by this look.
        seen = yield from look_into_trunk(ctx, est, 'stance')
        at_bag = False
        # Two tries: the second a little deeper into the bag and higher, in case the lower finger closed on air.
        for attempt, higher in enumerate((0., .005)):
            if attempt:
                yield from tuck(ctx, (arm,))                    # out of the camera's view of the trunk
                seen = yield from look_into_trunk(ctx, est, 'stance')
                if name not in est.bag:
                    break
            bags_as_obstacles(ctx, est, taking=name)
            edge = edge_in_base(est, name, plan['edge'])
            if edge is None:
                ctx.obstacles = {}; break
            ok = yield from bags.pinch(ctx, edge['point'], edge['top'], edge['side'], label=name, below=edge['below'] - higher, behind=edge['behind'] - attempt * edge['deeper'], skew=edge['skew'])
            ctx.obstacles = {}
            if not ok:
                if ok is None:
                    break
                at_bag = True; continue
            at_bag = True
            ctx.attach(arm, *bags.spheres(name, *est.bag_in_base(name))); ctx.holding[arm] = name
            way = bags.lift_out(ctx)
            if way is not None:
                yield from follow(ctx, way, rate=.015, max_steps=900)
            yield from hold(ctx, 6)
            # A hand that holds the bag has its fingers apart on it; one that came away without it has closed.
            got = held(ctx)
            ctx.event('bag_out', bag=name, arm=arm, attempt=attempt, way=way is not None, held=got, fingers=ctx.fingers().round(4).tolist(), tip=np.round(ctx.tip()[:3, 3], 3).tolist())
            if got:
                break
            ctx.set_grip(1.); ctx.detach(arm); ctx.holding.pop(arm, None)
            yield from hold(ctx, 10)
        if not got:
            spent.add((arm, plan['edge']))                      # another edge or the other hand first, the next time
        if got or at_bag or name not in est.bag:
            break                                               # (a hand that was at the bag may have moved it: the next visit starts with a look from the staging point)
        yield from tuck(ctx, (arm,))
    if not got and arm is not None:
        yield from tuck(ctx, (arm,))
    # Back from the car with the bag held up over the sill as it came out: in front of the chest it would be in the
    # bumper. The travelling posture is taken at the staging point.
    yield from straight(ctx, LID_STAGING, 0., speed=.2)
    if got:
        status = yield from carry_bag(ctx)
        ctx.event('bag_carried', bag=name, status=status)
        est.bag.pop(name, None); ctx.gone.add(name)
    return got


def visit(ctx, est, left, failed=None, streak=0):
    """One visit to the trunk, from the staging point behind the car with the pose fixed there: look, choose which
    of the bags @left (a list, put in order here) to take, and take it. failed, streak: the bag that the visit
    before this one did not get, and how many visits in a row did not get it. Returns (name, True if the hand
    holds it, False if it was not taken, None if it did not show)."""
    yield from look_into_trunk(ctx, est, 'staging')
    if len(left) == 2 or left[0] not in est.bag:
        # A look from the car's middle line, nearer and from above. With both bags in the trunk the order matters,
        # and a briefcase that leans on the satchel does not show from the staging point (start 103: the satchel
        # was taken first and swept the briefcase out of the trunk, onto the floor); a bag that was not made out
        # from there may show from here.
        yield from straight(ctx, (STANCE_X, LID_MIDDLE), 0., speed=.2, tolerance=.015)
        yield from hold(ctx, 6)
        yield from look_into_trunk(ctx, est, 'stance')
    order, in_the_way = order_of_bags(est, left)
    # After a visit that did not get a bag the other one is tried, then this one again; but a briefcase that is in
    # the satchel's way out is tried on four visits (by other edges and the other hand each time) before the satchel
    # is, which then may sweep it out of the trunk: it did on start 104 after two visits, each with two tries that
    # closed on nothing on a briefcase leaning by 68 degrees.
    if len(order) == 2 and failed == order[0] and (streak >= 4 or not in_the_way):
        order = order[::-1]
    left[:] = order; name = left[0]
    if name not in est.bag:
        yield from straight(ctx, LID_STAGING, 0., speed=.2)
        return name, None
    got = yield from take_bag(ctx, est, name)
    return name, got


ON_TOP = .21                        # the middle of a briefcase that lies on the satchel is this high above the trunk's floor at least (0.24-0.28 on the six starts that have it; at most 0.08 lying on the floor, 0.17 leaning)
IN_THE_WAY = dict(flat=.12, leaning=.10)      # how much nearer to the sill than the satchel a briefcase is that lies in its way out


def order_of_bags(est, left):
    """Which of the bags still in the trunk to take first, and whether that is because the briefcase is in the
    satchel's way: the satchel (a briefcase that leans on it then comes to lie flat), unless the briefcase lies
    between the satchel and the sill, where the satchel is drawn out. (The satchel swept a briefcase out of the
    trunk that lay flat there, and two that leant on it from there: on starts 103 and 221, 0.13 and 0.17 m nearer
    to the sill; with briefcases that leant on it 0.05-0.09 m nearer to the sill, or from a side, it came out and
    they stayed in the trunk: starts 303, 17, 46, 316.)"""
    if 'satchel' in left and 'briefcase' in left and 'satchel' in est.bag and 'briefcase' in est.bag:
        s = est.bag['satchel']['origin']; b = est.bag['briefcase']['origin']
        nearer = IN_THE_WAY['leaning' if est.bag['briefcase']['tilt'] > bags.LEANING else 'flat']
        if (b[0] < s[0] - nearer and abs(b[1] - s[1]) < .42) or on_the_satchel(est):
            return ['briefcase', 'satchel'], True
    return sorted(left, key=lambda name: name != 'satchel'), False


def on_the_satchel(est):
    """Does the briefcase lie on top of the satchel? (6 of 280 starts. The satchel drawn out from under it took it along,
    and it fell to the floor behind the bumper: start 34.)"""
    if 'satchel' not in est.bag or 'briefcase' not in est.bag:
        return False
    s = est.bag['satchel']['origin']; b = est.bag['briefcase']['origin']
    return bool(b[2] - floor_height(b[0]) > ON_TOP and np.hypot(b[0] - s[0], b[1] - s[1]) < .30)


# ---- the whole task ---------------------------------------------------------------------------------------
def full_task(ctx, est):
    ctx.holding = {}; ctx.spent = {}; ctx.hopeless = {}; ctx.gone = set()
    yield from tuck(ctx)
    opened = yield from open_trunk(ctx, est)
    ctx.event('stage', done='trunk', ok=bool(opened))
    status = yield from open_door(ctx, est)
    ctx.event('stage', done='door', status=status)
    delivered = []; left = ['satchel', 'briefcase']; failed = None; streak = 0
    for attempt in range(8):
        if not left:
            break
        status = yield from to_car(ctx, est)
        if status != 'arrived':
            ctx.event('stage', done='to_car', status=status); continue
        name, got = yield from visit(ctx, est, left, failed, streak)
        streak = 0 if got else streak + 1 if failed == name else 1; failed = None if got else name
        est.hold_axis(None)
        ctx.event('stage', done='bag_taken', bag=name, ok=got)
        if not got:
            continue
        arm = next(a for a, held_name in ctx.holding.items() if held_name == name)
        status = yield from leave_garage(ctx, est)
        if status == 'in_corridor':
            status = yield from to_sofa(ctx, est)
        ctx.event('stage', done='at_sofa', bag=name, status=status)
        if status == 'arrived':
            done = yield from let_down(ctx, est, arm)
            if done:
                delivered.append(name)
            yield from straight(ctx, SOFA_STAGING, SOFA_STANCE[2], speed=.25)
        left.remove(name)
        ctx.event('stage', done='bag_delivered', bag=name, ok=name in delivered)
        if left:
            status = yield from back_to_garage(ctx, est)
            if status != 'in_garage':
                return status
    return 'unloaded' if len(delivered) == 2 else 'one_bag' if delivered else 'nothing'


# ---- parts of the task, for development runs (privileged set-up: see dev/jobs/chain_observed.py) ----------------
def stage_bags(ctx, est):
    """From the staging point behind the car with the lid open: each bag out of the trunk and down on the garage
    floor behind the robot."""
    results = {}; left = ['satchel', 'briefcase']; failed = None; streak = 0; ctx.spent = {}; ctx.hopeless = {}; ctx.gone = set()
    for turn in range(8):
        if not left:
            break
        shift = est.fix_at_bumper(); est.hold_axis('xy')
        ctx.event('bumper_fix', shift=None if shift is None else np.round(shift, 4).tolist())
        name, got = yield from visit(ctx, est, left, failed, streak)
        streak = 0 if got else streak + 1 if failed == name else 1; failed = None if got else name
        est.hold_axis(None); results[name] = got
        ctx.event('stage', done='bag_taken', bag=name, ok=got)
        if not got:
            yield from settle(ctx, est, 12)
        else:
            left.remove(name)
            arm = next(a for a, held_name in ctx.holding.items() if held_name == name)
            spot = (-8.25, 5.6 if name == 'satchel' else 4.6)
            yield from route(ctx, est, spot, np.pi / 2 + 1.2, tolerance=.08)
            yield from let_down(ctx, est, arm)
            est.bag.clear()                             # what was seen of the trunk before is stale
            yield from route(ctx, est, LID_STAGING, 0., tolerance=.05)
            yield from settle(ctx, est, 12)
    return 'both' if len(results) == 2 and all(results.values()) else str(results)


def stage_sofa(ctx, est):
    """From the corridor with the door open: to the sofa and back, with empty hands (the way and the stance)."""
    status = yield from to_sofa(ctx, est)
    ctx.event('stage', done='at_sofa', status=status)
    yield from straight(ctx, SOFA_STAGING, SOFA_STANCE[2], speed=.25)
    return status
