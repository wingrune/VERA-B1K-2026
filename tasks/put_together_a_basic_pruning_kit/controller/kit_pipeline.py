"""Putting the pruning kit together: the task as a sequence of skills.

    find the toolbox, the pruner and the shears on the garage floor (look round, then from lookouts);
    learn which long side of the toolbox is its front (a look at a long side from in front of it);
    for each tool, the nearer one first:
        choose the stance at the toolbox and with it the hand that opens the lid there: the tool goes into the
        other hand;
        stand where that hand reaches the tool, the way in chosen from a look at it; take it (kit.py); hold it in
        front of the chest;
        stand at the toolbox's hinge side or at one of its ends (at its front, where those have no room); the free
        hand opens the lid by its handle and holds it, the other lets the tool go over the open box; the lid is led
        back and let go.

Poses are (x, y, yaw) in the map frame M.
"""
import time
import numpy as np
from manip import hold, drive, pose2
from car_skills import tuck, park, plan_top_pinch, top_pinch, down
from car_pipeline import straight, settle, stance_options, gap, FOOTPRINT, TURN_CLEAR
import kit

wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
RADIUS = dict(toolbox=.345, pruner=.06, shears=.08, lying=.13)       # of the disc round each thing that the chassis keeps out of (lying: the pruner knocked over)
BOX_DISCS = ((-.19, 0., .19), .17)  # the toolbox for the chassis: discs along its length (where along, their radius)
ROOM = .10                          # room the chassis keeps to a thing it passes (m), where the way allows
REACH = .56; ASIDE = .16            # where a tool lies in the base frame when a hand takes it: ahead, and to the hand's side
SLANT = np.radians(45.)             # the hand comes onto the pruner's handles slanting down by this
VIEW = 1.10                         # from this far in front of a long side the toolbox's latches show (or do not)
# Lookouts in the garage (x, y) from which the base looks round when a thing has not shown yet: the shears show from
# 2.5 m at most.
LOOKOUTS = ((-8.05, 4.0), (-8.05, 6.6), (-5.6, 3.9), (-3.2, 4.0), (-7.75, 8.25))
# Where the hand that holds a tool is held for the way: ahead, to the hand's side, height above the floor.
CARRY_SPOTS = ((.36, .16, .82), (.40, .14, .78), (.34, .20, .86), (.42, .10, .90))
# Stances at the toolbox, in its own frame (x across it, its front at +x; y along it): where the base stands and
# which way it faces, by side. From its hinge side the lid opens towards the robot; from an end, sideways. Each
# side's first stances have the handle on the side of the hand that takes it there (SIDES).
BOX_STANCES = dict(back=[(bx, by, 0.) for bx in (-.60, -.66, -.54) for by in (.12, 0., -.12)], front=[(bx, by, np.pi) for bx in (.45, .50, .55) for by in (0., .12, -.12)],
                   **{'end-': [(bx, by, np.pi / 2) for by in (-.74, -.80, -.70, -.66) for bx in (.08, 0., -.08)], 'end+': [(bx, by, -np.pi / 2) for by in (.74, .80, .70, .66) for bx in (.08, 0., -.08)]})
# Where the base turns and looks before it moves into a stance at the box: how far from the stance, and in which
# direction off straight behind it (the base moves any way without turning: where a wall is behind the stance it
# comes in from the side).
BOX_STAGING = tuple((length, turn) for turn in (0., np.pi / 2, -np.pi / 2, .8, -.8) for length in (.45, .60, .80, 1.00))


# ---- the way ------------------------------------------------------------------------------------------------
def obstacles(est, but=()):
    """The things on the floor as discs: [(xy, radius)], without those named in @but."""
    out = []
    for name, known in est.things.items():
        if name in but:
            continue
        xy = np.asarray(known['xy'], float)
        if name == 'toolbox':
            along = np.array([-np.sin(known['yaw']), np.cos(known['yaw'])]); out += [(xy + d * along, BOX_DISCS[1]) for d in BOX_DISCS[0]]
        else:
            out.append((xy, RADIUS['lying' if known.get('lying') else name]))
    return out


def keepouts(est, here, goal, extra, but=()):
    """Discs the base's middle stays out of on a way from @here to @goal: each thing's disc with the chassis's half
    width and @extra round it; narrowed, not dropped, where the way starts or ends nearer than that."""
    return [(xy, max(0., min(radius + .345 + extra, float(np.linalg.norm(here - xy)) - .03, float(np.linalg.norm(goal - xy)) - .03))) for xy, radius in obstacles(est, but)]


def go(ctx, est, goal, yaw, but=(), **options):
    """drive() to @goal round the things on the floor; the way is planned again when a thing shows that was not
    known, or shows elsewhere. Returns drive's status, or 'no_route'."""
    goal = np.asarray(goal, float)[:2]
    for again in range(5):
        here = np.asarray(est.base()[:2], float); known = {name: np.asarray(k['xy'], float).copy() for name, k in est.things.items()}; keep = None
        for extra in (ROOM, .04, 0.):
            keep = keepouts(est, here, goal, extra, but)
            if ctx.scene.path(here, goal, keepout=keep) is not None:
                break
            keep = None
        if keep is None:
            ctx.event('route', ok=False, goal=np.round(goal, 2).tolist()); return 'no_route'
        steps = drive(ctx, goal, yaw, keepout=keep, lookahead=.20 if keep else .35, **options)
        while True:
            try:
                action = next(steps)
            except StopIteration as stop:
                return stop.value
            yield action
            if again < 4 and (set(est.things) != set(known) or any(float(np.linalg.norm(est.things[name]['xy'] - xy)) > .10 for name, xy in known.items())):
                break
        steps.close()
        ctx.event('route', again=True, goal=np.round(goal, 2).tolist(), things={name: np.round(k['xy'], 2).tolist() for name, k in est.things.items()})
    return 'no_route'


def clear(est, spot, but=()):
    """May the base stand and turn at map @spot without touching a thing on the floor? (On the way to a look at the
    toolbox's front it turned on the shears once, and they were not where it had seen them any more.)"""
    return all(float(np.linalg.norm(np.asarray(spot, float)[:2] - xy)) > TURN_CLEAR + radius for xy, radius in obstacles(est, but))


def turn_to(ctx, est, yaw):
    """Turn on the spot to heading @yaw."""
    here = est.base()
    status = yield from drive(ctx, here[:2], yaw, direct=True, face_travel=False, tolerance=.08, yaw_tolerance=.05, max_steps=400)
    return status


def look(ctx, est, steps=4):
    yield from hold(ctx, steps)
    return est.look()


# ---- finding the things ------------------------------------------------------------------------------------------
def missing(est, names):
    return [name for name in names if name not in est.things and name not in est.gone]


def look_round(ctx, est, names):
    """Looks a quarter of a turn apart, all the way round, until the things @names are all known."""
    start = est.base()[2]
    for quarter in range(4):
        if quarter:
            yield from turn_to(ctx, est, wrap(start + quarter * np.pi / 2))
        yield from look(ctx, est)
        if not missing(est, names):
            return True
    return not missing(est, names)


def find(ctx, est, names):
    """Know where the things @names are: a look round where the base stands, then from the lookouts, the nearest
    first. Returns the names still not found."""
    yield from look_round(ctx, est, names)
    ctx.event('find', where='start', known=sorted(est.things), missing=missing(est, names))
    left = list(LOOKOUTS)
    while missing(est, names) and left:
        here = np.asarray(est.base()[:2], float); left.sort(key=lambda p: float(np.linalg.norm(np.asarray(p) - here))); spot = left.pop(0)
        if float(np.linalg.norm(np.asarray(spot) - here)) < .8 or not ctx.scene.base_free(np.asarray(spot), margin=.02) or not clear(est, spot):
            continue
        status = yield from go(ctx, est, spot, float(np.arctan2(spot[1] - here[1], spot[0] - here[0])), tolerance=.10)
        if status in ('no_route',):
            continue
        yield from look_round(ctx, est, names)
        ctx.event('find', where=list(spot), status=status, known=sorted(est.things), missing=missing(est, names))
    return missing(est, names)


# ---- a tool --------------------------------------------------------------------------------------------------
TURNED = (0., .3, -.3, .6, -.6, .9, -.9)        # how far the base may stand turned from the way the hand comes onto the pruner
FAR_TURNED = 2.                     # ... turned by 0.9 only where nothing nearer has a stance: so turned the hand knocked the pruner over 2 times of 9 (0 of 12 up to 0.6)
GAP = dict(pruner=.012, shears=.008)        # fingers further apart than this (sum of both, m) hold the tool (the pruner's handles keep them 2.9 cm apart, the shears' blades 1.3-1.7 cm)
SHEARS_BACK = .006                  # the fingers' ends go this far from the shears' middle towards their handles
LYING_GRIP = dict(flat=.010, edge=.030)     # height of the fingers' ends on a pruner that lies on a broad side, on a narrow one
# The shears in the hand, for the arm planner: 13 cm long, held at their middle, any way round. (Taken for a ball
# round the fingers' ends, and hanging from their blades, they were torn out of the hand on 2 of 17 visits from an
# end of the box, where the trunk bends furthest, and lay on the robot's own chassis.)
HANGING = (0., -.06, -.12)
other = lambda arm: 'right' if arm == 'left' else 'left'


def tool_target(est, name):
    """Where the fingers' ends go on tool @name: map x, y, height above the floor."""
    known = est.things[name]; xy = np.asarray(known['xy'], float)
    if name == 'pruner':
        return np.r_[xy, LYING_GRIP[known['lying']] if known.get('lying') else kit.PRUNER_GRIP]
    # At their middle, where the blades cross (a touch towards the handles if the look has told which end those are):
    # there the fingers stayed 1.5-1.8 cm apart and the shears where they were in the hand through the lift, the way
    # held in front of the chest, the trunk's turns and a far reach (6 of 6). Taken 1.5-2.5 cm nearer the blades'
    # points, where they were taken before, they turned out from between the fingers and hung by the grasp, the
    # fingers shut (the hand then lost them on about one visit to the box in ten), or fell (2 of 6).
    if known.get('axis') is not None:
        xy = xy - SHEARS_BACK * np.array([np.cos(known['axis']), np.sin(known['axis'])])
    return np.r_[xy, kit.SHEARS_GRIP]


def pruner_ways(known):
    """The ways a hand may come onto the pruner (map headings of its fingers), in order of choice; None: any way.
    Along its pair of handles, so that the fingers close across their 2.9 cm with 3 cm to spare on a side (across
    the 6.7 cm the other way the hand knocked it over 2 times of 8), and from the end the blades lean away from, if a
    look has told which that is: from the other end, which comes last here, it tilts in the hand or falls (held 7
    times of 13 in a trial, whatever the slant, the height and how far past the middle; 4 of 7 in whole runs)."""
    if known.get('way') is not None:
        return [wrap(known['way'] + flip + turn) for flip in (0., np.pi) for turn in (0., .2, -.2)]
    if known.get('axis') is not None:
        return [wrap(known['axis'] + flip + turn) for flip in (0., np.pi) for turn in (0., .2, -.2)]
    return None


def tool_options(ctx, est, name, arms, here):
    """Stances from which a hand takes tool @name: [(cost, arm, (x, y, yaw), staging, hand)] in order of choice.
    hand: for the pruner the map heading of the hand's fingers (the base may stand turned from it), for the shears
    and for a pruner that lies flat the direction along them (the hand comes from straight above and closes across).
    arms: the hands that may take it, the first one for choice."""
    known = est.things[name]; target = tool_target(est, name); toward = float(np.arctan2(target[1] - here[1], target[0] - here[0]))
    round_about = [wrap(toward + (k + 1) // 2 * (1 if k % 2 else -1) * 2 * np.pi / 16) for k in range(16)]
    if name == 'pruner' and not known.get('lying'):
        ways = pruner_ways(known)
        last = 4. if known.get('way') is not None else 0.          # (the end the blades lean to: only where the other has no stance)
        pairs = [(yaw, yaw, 0.) for yaw in round_about] if ways is None else [(wrap(way + turn), way, .4 * abs(turn) + (FAR_TURNED if abs(turn) > .7 else 0.) + (0. if index % 3 == 0 else .15) + (last if index >= 3 else 0.)) for index, way in enumerate(ways) for turn in TURNED]
    else:
        pairs = [(yaw, known['along'], 0.) for yaw in round_about]
    floor = obstacles(est, but=(name,)); thing = (target[:2], RADIUS['lying' if known.get('lying') else name]); out = []
    for order, arm in enumerate(arms):
        offset = (REACH, ASIDE if arm == 'left' else -ASIDE)
        for yaw, hand, extra in pairs:
            out += [(cost + extra + 50. * order, arm, pose[:3], pose[3], hand) for cost, pose in stance_options(ctx, here, target[:2], offset, [yaw], floor, turn_clear=[thing])]
    return sorted(out, key=lambda item: item[0])


def tool_plan(ctx, slant, target, hand, pose=None):
    """The active hand's pinch of a tool from base pose @pose (None: where the base stands), or None. slant: the
    hand comes slanting down along map heading @hand (the pruner standing); else from straight above, closing
    across map direction @hand."""
    base = ctx.est.base() if pose is None else pose; point = kit.in_base(base, target); world = None if pose is None else pose2(*pose)
    if slant:
        return kit.plan_slant_pinch(ctx, point, SLANT, heading=wrap(hand - base[2]), base=world)
    return plan_top_pinch(ctx, point, wrap(hand + np.pi / 2 - base[2]), base=world)


def tool_spheres(ctx, name):
    """What the hand holds, as spheres in the base frame (for the collision model and for cutting it out of the
    camera's points): round the fingers' ends, and for the pruner up and down its length."""
    tip = ctx.tip()[:3, 3]
    if name == 'lying':                             # (taken from the floor by its handles: it hangs from the fingers, its blades down, or stays level)
        return np.array([tip + [0., 0., dz] for dz in (-.14, -.07, 0.)]), np.full(3, .06)
    if name == 'pruner':
        return np.array([tip + [0., 0., dz] for dz in (-.06, 0., .07, .14)]), np.full(4, .05)
    return np.array([tip + [0., 0., dz] for dz in HANGING]), np.full(len(HANGING), .06)


def held_shape(known, name):
    """What a hand will hold once it has taken tool @name as it stands or lies (@known), as spheres in the hand's
    tool frame (points, radii): tool_spheres() as they will be then, for plans made before the tool is taken. (The
    pruner standing is taken with the hand slanted 45 degrees: it stands 45 degrees off the fingers.)"""
    if name == 'pruner' and not known.get('lying'):
        return np.array([[.707 * along, 0., .707 * along] for along in (-.06, 0., .07, .14)]), np.full(4, .05)
    if name == 'pruner':
        return np.array([[0., 0., dz] for dz in (-.14, -.07, 0.)]), np.full(3, .06)
    return np.array([[0., 0., dz] for dz in HANGING]), np.full(len(HANGING), .06)


def carry(ctx, arm):
    """The hand with a tool to where it is held on the way: in front of the chest, the trunk upright."""
    side = 1. if arm == 'left' else -1.; ctx.arm = arm
    status = yield from park(ctx, [(x, side * y, z) for x, y, z in CARRY_SPOTS], [down(0.), down(np.pi / 2), kit.pitched(np.radians(60.)), kit.pitched(np.radians(35.))], label='carry')
    return status


def holds(ctx, arm, name):
    """Is tool @name still in @arm's hand, as far as its fingers tell? The pruner keeps them 3 cm apart, the shears,
    taken where their blades cross, 1.5 cm. (A hand whose fingers have shut is opened before anything else is
    done with it: shears taken by their blades could hang on by the grasp alone with the fingers shut, and were
    still there when the hand went for the pruner.)"""
    return float(np.sum(ctx.fingers(arm))) > GAP[name]


def told(known):
    """What a look has told of a tool's attitude, for knowing when a nearer look has told more."""
    return (known.get('way') is not None, known.get('axis') is not None, known.get('lying'))


def take_tool(ctx, est, name, arms):
    """Stand where a hand of @arms reaches tool @name, take it and hold it for the way. Returns (the arm that holds
    it or None, what stopped it: 'no_stance' (no hand of @arms has a stance and a way), 'lost' (it is not where it
    was known and has not shown elsewhere), 'failed')."""
    failed = []; pinches = 0; empty = False
    for attempt in range(7):
        if name not in est.things:
            return None, 'lost'
        if pinches >= 3:
            break
        before = dict(est.things[name]); target = tool_target(est, name); here = np.asarray(est.base()[:2], float); slant = name == 'pruner' and not before.get('lying')
        options = tool_options(ctx, est, name, arms, here); chosen = None; started = time.time()
        for cost, arm, pose, staging, hand in options:
            if any(np.hypot(pose[0] - f[0], pose[1] - f[1]) < .10 and abs(wrap(pose[2] - f[2])) < .25 for f in failed) or time.time() - started > 40.:
                continue
            if ctx.scene.path(here, staging, keepout=keepouts(est, here, staging, .04)) is None:
                continue
            ctx.arm = arm
            if tool_plan(ctx, slant, target, hand, pose) is not None:
                chosen = (arm, pose, staging, hand); break
        if chosen is None:
            ctx.event('stance', ok=False, thing=name, options=len(options), failed=len(failed), arms=list(arms)); return None, 'failed' if failed else 'no_stance'
        arm, (x, y, yaw), staging, hand = chosen; ctx.arm = arm
        if slant and before.get('way') is not None and np.cos(hand - before['way']) < 0:
            target = target + .01 * np.array([np.cos(hand), np.sin(hand), 0.])            # (from the other end: as far past the middle of the handles that way)
        ctx.event('stance', thing=name, arm=arm, pose=np.round([x, y, yaw], 3).tolist(), staging=np.round(staging, 3).tolist(), target=np.round(target, 3).tolist(), hand=round(float(hand), 2), told=list(told(before)), lying=before.get('lying'), options=len(options))
        status = yield from go(ctx, est, staging, yaw, tolerance=.04)
        if status != 'arrived':
            failed.append((x, y, yaw)); continue
        yield from settle(ctx, est, 8)
        yield from look(ctx, est)
        if name not in est.things:
            return None, 'lost'
        moved = np.asarray(est.things[name]['xy'], float) - before['xy']
        if float(np.linalg.norm(moved)) > .03 or told(est.things[name]) != told(before):
            continue                                    # seen better from here: plan again for where it is and how it stands
        target = target + np.r_[moved, 0.]
        status = yield from straight(ctx, (x, y), yaw, speed=.2, tolerance=.02)
        if status != 'arrived':
            yield from straight(ctx, staging, yaw, speed=.2)
            failed.append((x, y, yaw)); continue
        yield from hold(ctx, 6)
        plan = None
        for turn in (0., .1, -.1, .2, -.2) if slant else (0.,):
            plan = tool_plan(ctx, slant, target, hand + turn)
            if plan is not None:
                break
        if plan is None:
            ctx.event(name, ok=False, reason='no way from the stance', arm=arm)
            yield from straight(ctx, staging, yaw, speed=.2)
            failed.append((x, y, yaw)); continue
        pinches += 1
        if slant:
            ok = yield from kit.slant_pinch(ctx, plan, label=name, gap=GAP[name], lift=.12)
        else:
            # (The fingers' gap is asked while they are still closed: a hand that has closed on nothing is opened again
            # at once, and its fingers then stand apart.)
            base = est.base()
            ok = yield from top_pinch(ctx, kit.in_base(base, target), wrap(hand + np.pi / 2 - base[2]), label=name, approach=.12, gap=GAP[name])
        if ok:
            ctx.attach(arm, *tool_spheres(ctx, 'lying' if before.get('lying') else name)); ctx.holding[arm] = name; ctx.by_grasp[arm] = before.get('lying') != 'flat'; est.taken(name)
            status = yield from carry(ctx, arm)
            # A tool was where the map pose had it and the hand closed on it: the start pose was the right one, and
            # the other start poses that the localiser still followed are dropped. (In unloading_the_car one of them
            # took over late, while the camera was full of something the map does not have.)
            followed = len(est.ob.loc.rivals); est.ob.loc.rivals = []
            ctx.event('tool_taken', tool=name, arm=arm, carry=status, fingers=ctx.fingers().round(4).tolist(), start_poses_dropped=followed)
            yield from straight(ctx, staging, yaw, speed=.2)
            return arm, 'taken'
        yield from tuck(ctx, (arm,))
        yield from straight(ctx, staging, yaw, speed=.2)
        yield from hold(ctx, 6)
        seen = yield from look(ctx, est)                    # where is it now?
        failed.append((x, y, yaw))
        if name in est.things and float(np.linalg.norm(np.asarray(est.things[name]['xy'], float) - before['xy'])) < .03 and ((empty and ok is False) or (seen.get(name) is None and est.in_view(est.things[name]['xy'], est.base(), nearest=.86))):
            # Not where it was: this look does not show it there, or the hand has closed on nothing there twice (a
            # pruner that a hand knocked over lies flat and does not show as one; shears that the chassis or a hand
            # pushed lie a hand's breadth further).
            ctx.event('lost', thing=name, at=np.round(est.things[name]['xy'], 3).tolist(), shows=seen.get(name) is not None); est.forget(name); return None, 'lost'
        empty = empty or ok is False
    return None, 'failed'


# ---- the toolbox ------------------------------------------------------------------------------------------------
def learn_front(ctx, est):
    """Look at a long side of the toolbox from in front of it until a look tells which side is the front. Returns
    True when it is known."""
    tried = []; part = 0
    for attempt in range(4):
        box = est.things.get('toolbox')
        if box is None:
            return False
        here = np.asarray(est.base()[:2], float); xy = np.asarray(box['xy'], float); out = np.array([np.cos(box['yaw']), np.sin(box['yaw'])]); spots = []
        for sign in (1., -1.):
            for distance in (VIEW, VIEW + .2, VIEW - .15, VIEW + .4):
                for aside in (0., .25, -.25, .5, -.5, .8, -.8):
                    if np.hypot(distance, aside) > 1.42:        # (from further a face without latches is not told from one whose latches do not show)
                        continue
                    spot = xy + sign * distance * out + aside * np.array([-out[1], out[0]])
                    if all(float(np.linalg.norm(spot - other)) > .3 for other in tried) and ctx.scene.base_free(spot, margin=.02) and clear(est, spot) and ctx.scene.path(here, spot, keepout=keepouts(est, here, spot, .04)) is not None:
                        spots.append((float(np.linalg.norm(spot - here)) + .1 * abs(aside), sign, spot))
        for cost, sign, spot in sorted(spots, key=lambda item: item[0]):
            yaw = float(np.arctan2(xy[1] - spot[1], xy[0] - spot[0]))
            status = yield from go(ctx, est, spot, yaw, tolerance=.06)
            if status != 'arrived':
                continue
            yield from settle(ctx, est, 6)
            yield from look(ctx, est)
            told, saw = est.look_front()
            ctx.event('front', told=told, **saw, at=np.round(spot, 2).tolist(), front=None if est.front is None else round(est.front, 2))
            if told:
                return True
            tried.append(spot)
            if saw.get('whole') is False and float(np.linalg.norm(spot - xy)) < 1.7 and (abs(saw['long'] - .608) > .15 or abs(saw['short'] - .243) > .10):
                # Three looks from near that show nothing of a lid's size there: it is not the toolbox (the foot of the
                # door was taken for it once, from 3.5 m). The search goes on without it.
                part += 1
                if part >= 3:
                    ctx.event('not_the_toolbox', at=np.round(xy, 2).tolist()); est.disown('toolbox'); return False
            break                                           # (the box may show elsewhere now: its sides are worked out again, and another spot taken)
    return est.front is not None


# Which side each hand opens the lid from (the hand goes onto the handle the one way round that opens the lid, and the
# lid swings to the hinge side): the right hand from the hinge side and from the end at which that is on its right,
# the left hand from the other end. (Over every side offline the other pairs had a plan at a few stances only, and
# said so after a minute.)
SIDES = dict(right=('back', 'end+', 'front'), left=('end-', 'front'))
# From the front, where the hinge side and both ends have no room for the chassis (a box that stands with its back to
# a wall: 13 of 264 starts): the lid opens away from the robot, and the hand leads it less far (LOW), so that the arm
# that reaches over the box stays clear of what is behind it; the tool is let go nearer the front wall. From the
# other sides the same is the second choice: a box 0.45 m from a wall had a plan at its end for the first tool and,
# 2 cm and 2 degrees further on after that visit, none for the second with the lid led all the way.
LOW = dict(held_at=1.05, across=(.045, .06, .03))        # (let go 1.5 cm nearer the lid than that, under a lid led to 1.05 rad, the shears slid down its inner side and stood on edge against the hinge wall, where they did not count as inside)
FULL = {}                           # (the lid led to kit.HELD_AT, the tool let go where kit.DROP has it)
LAST = 3.                           # a stance at the front is chosen only where no other has a plan


def hows(side):
    """The ways a visit from @side may be made, in order of choice: what is different about each, for
    kit.plan_visit."""
    return [LOW] if side == 'front' else [FULL, LOW]


def one_of(pose, others):
    """Is base pose @pose one of @others, to a hand's breadth?"""
    return any(np.hypot(pose[0] - f[0], pose[1] - f[1]) < .08 and abs(wrap(pose[2] - f[2])) < .3 for f in others)


def box_stances(ctx, est, place, near, lid_arm, floor=()):
    """Stances at the toolbox standing at @place for @lid_arm's hand on the lid, each with the point behind it where
    the base turns and looks: [(cost, side, (x, y, yaw), staging)], in order of choice (the nearer to @near first).
    floor: [(xy, radius)] of what else stands on the floor (a pruner next to the box was run over on the way in
    once)."""
    x, y, yaw = place; c, s = np.cos(yaw), np.sin(yaw); out = []; box = (np.array([x, y]), RADIUS['toolbox'])
    for order, side in enumerate(SIDES[lid_arm]):
        group = BOX_STANCES[side]
        for rank, (bx, by, turn) in enumerate(group):
            pose = (x + c * bx - s * by, y + s * bx + c * by, wrap(yaw + turn))
            if not ctx.scene.footprint_free(*pose, FOOTPRINT, .025, .012) or any(gap(pose, centre, r) < .03 for centre, r in floor):
                continue
            for length, off in BOX_STAGING:
                away = np.array([np.cos(pose[2] + np.pi + off), np.sin(pose[2] + np.pi + off)]); staging = np.array(pose[:2]) + length * away
                if not ctx.scene.base_free(staging, margin=-.02) or float(np.linalg.norm(staging - box[0])) < TURN_CLEAR + .16 or any(float(np.linalg.norm(staging - centre)) < TURN_CLEAR + r for centre, r in floor):
                    continue
                swept = [np.array(pose[:2]) + d * away for d in np.arange(.05, length, .05)]
                if all(ctx.scene.footprint_free(q[0], q[1], pose[2], FOOTPRINT, .02, .008) for q in swept) and not any(gap((q[0], q[1], pose[2]), centre, r) < .03 for q in swept for centre, r in floor):
                    out.append((float(np.linalg.norm(staging - near)) + .03 * rank + .5 * order + .3 * abs(off) + (LAST if side == 'front' else 0.), side, pose, staging)); break
    return sorted(out, key=lambda item: item[0])


def visit_plan(ctx, est, lid_arms, floor, near=None, but=(), carried=(), budget=90., quiet=False):
    """A stance at the toolbox with a plan for the two hands (kit.plan_visit), the lid's hand one of @lid_arms:
    dict(lid_arm, side, pose, staging) for the first stance in order of choice that has a route and a plan, or None.
    Every stance is asked the quick way first; only if none has a plan that way, the long way for @budget seconds.
    near: where the base comes from (map xy; None: where it stands). but: stances not to choose again. carried:
    the tools that will not be on the floor then (the one the visit is for), for the route."""
    place = est.box_place(); here = np.asarray(est.base()[:2], float); near = here if near is None else np.asarray(near, float)[:2]; stances = []
    for lid_arm in lid_arms:
        stances += [(cost, lid_arm, side, pose, staging) for cost, side, pose, staging in box_stances(ctx, est, place, near, lid_arm, floor)]
    # (a stance from which a visit has put a tool in comes first)
    worked = getattr(ctx, 'worked', []); stances = [(cost - (1.5 if one_of(pose, [w for arm, w in worked if arm == lid_arm]) else 0.), lid_arm, side, pose, staging) for cost, lid_arm, side, pose, staging in stances]
    stances = [item for item in sorted(stances, key=lambda item: item[0]) if not one_of(item[3], but)]
    routed = [item for item in stances if ctx.scene.path(here, item[4], keepout=keepouts(est, here, item[4], 0., carried)) is not None]
    started = time.time(); asked = 0
    for quick in (True, False):
        for level in (0, 1):
            for cost, lid_arm, side, pose, staging in routed:
                if level >= len(hows(side)) or (not quick and time.time() - started > budget):
                    continue
                asked += 1
                if kit.plan_visit(ctx, kit.box_in_base(pose, place), lid_arm, base=pose2(*pose), quick=quick, **hows(side)[level]) is not None:
                    return dict(lid_arm=lid_arm, side=side, pose=pose, staging=staging, how=hows(side)[level], cost=cost + (5. if not quick else 0.) + level, room=len(stances), way=len(routed), asked=asked, seconds=round(time.time() - started))
    if not quiet:
        ctx.event('box_stance', ok=False, place=np.round(place, 3).tolist(), lid_arms=list(lid_arms), room=len(stances), way=len(routed), asked=asked, avoided=len(but), seconds=round(time.time() - started))
    return None


def beside_box(est, xy, margin=.05):
    """Does map point @xy lie clear of the toolbox's outline by more than @margin? What a look shows nearer than that
    is the box itself. (Looked at from 0.85 m off its end, with that end at the image's lower border, a piece of the
    end wall was taken for the shears lying there, 1 cm from the wall; the hand went for it, opened the lid a crack
    and turned the box. The five tools that did lie next to the box lay 9 cm or more from its outline.)"""
    place = est.box_place()
    if place is None:
        return True
    x, y, yaw = place; c, s = np.cos(yaw), np.sin(yaw); dx, dy = float(xy[0]) - x, float(xy[1]) - y; across, along = c * dx + s * dy, -s * dx + c * dy
    return bool(across > kit.FACES[0] + margin or across < kit.FACES[1] - margin or abs(along) > kit.HALF_LENGTH + margin)


def put_at(ctx, name, along):
    """Note where along the box tool @name was let go into it (None: it is not in the box after all). ctx.in_box, which
    the drop planner reads, lists the places of the tools that lie in the box, the latest last."""
    ctx.put = dict(getattr(ctx, 'put', {})); ctx.put.pop(name, None)
    if along is not None:
        ctx.put[name] = float(along)
    ctx.in_box = list(ctx.put.values())


def into_box(ctx, est, tool_arm, name):
    """With tool @name in @tool_arm's hand: stand at the toolbox, open it with the other hand, let the tool go into
    it, shut it. Returns 'in', or what stopped it ('lost': the tool is not in the hand any more; 'out': it was let
    go and lies on the floor next to the box). A hand whose fingers have shut on the way goes to the box all the
    same if the simulator's grasp may still hold the tool (every grip but the squeeze of the pruner lying flat),
    and is opened over it."""
    lid_arm = other(tool_arm); failed = []; visits = 0
    for attempt in range(7):
        if not holds(ctx, tool_arm, name) and not getattr(ctx, 'by_grasp', {}).get(tool_arm):
            ctx.event('tool_lost', tool=name, arm=tool_arm, fingers=ctx.fingers(tool_arm).round(4).tolist())
            ctx.detach(tool_arm); ctx.holding.pop(tool_arm, None); est.gone.discard(name)
            ctx.set_grip(1., tool_arm)
            yield from hold(ctx, 20)
            yield from tuck(ctx, (tool_arm,))
            return 'lost'
        if est.box_place() is None:
            return 'no_box'
        if visits >= 4:
            break
        place = est.box_place(); chosen = visit_plan(ctx, est, (lid_arm,), obstacles(est, but=('toolbox',)), but=failed)
        if chosen is None:
            return 'no_stance'
        side, pose, staging = chosen['side'], chosen['pose'], chosen['staging']
        ctx.event('box_stance', side=side, pose=np.round(pose, 3).tolist(), staging=np.round(staging, 3).tolist(), place=np.round(place, 3).tolist(), lid_arm=lid_arm, held_at=chosen['how'].get('held_at', kit.HELD_AT), asked=chosen['asked'], seconds=chosen['seconds'])
        status = yield from go(ctx, est, staging, pose[2], tolerance=.04)
        if status != 'arrived':
            failed.append(pose); continue
        yield from settle(ctx, est, 8)
        here = est.base(); toward = float(np.arctan2(place[1] - here[1], place[0] - here[0]))
        if abs(wrap(toward - here[2])) > .35:               # (come in from the side: the look is taken facing the box)
            yield from turn_to(ctx, est, toward)
            yield from look(ctx, est)
            yield from turn_to(ctx, est, pose[2])
            yield from settle(ctx, est, 6)
        else:
            yield from look(ctx, est)
        now = est.box_place()
        if now is None:
            return 'no_box'
        if np.hypot(now[0] - place[0], now[1] - place[1]) > .03 or abs(wrap(now[2] - place[2])) > .06:
            continue                                        # seen better from here: the stance again for where it stands
        status = yield from straight(ctx, pose[:2], pose[2], speed=.2, tolerance=.02)
        if status != 'arrived':
            yield from straight(ctx, staging, pose[2], speed=.2)
            failed.append(pose); continue
        yield from hold(ctx, 6)
        box = kit.box_in_base(est.base(), now); visit = None; how = chosen['how']
        for how in [chosen['how']] + [option for option in hows(side) if option is not chosen['how']]:
            visit = kit.plan_visit(ctx, box, lid_arm, quick=True, **how) or kit.plan_visit(ctx, box, lid_arm, **how)
            if visit is not None:
                break
        if visit is None:
            ctx.event('put_in', ok=False, reason='no way from the stance')
            yield from straight(ctx, staging, pose[2], speed=.2)
            failed.append(pose); continue
        visits += 1
        last = bool(getattr(ctx, 'in_box', []))            # (the other tool is in the box: the hand shuts the lid)
        result = yield from kit.put_in(ctx, visit, gap=GAP[name], shut=last)
        if result == 'grip':
            # Both fingers stopped on the handle: once more from here, the hand further to the hinge side.
            visit = kit.plan_visit(ctx, box, lid_arm, quick=True, aim=kit.AIM + kit.AIM_AGAIN, **how) or kit.plan_visit(ctx, box, lid_arm, aim=kit.AIM + kit.AIM_AGAIN, **how)
            if visit is not None:
                result = yield from kit.put_in(ctx, visit, gap=GAP[name], shut=last)
        if result in ('in', 'let_go'):
            ctx.holding.pop(tool_arm, None); ctx.worked = getattr(ctx, 'worked', []) + [(lid_arm, pose)]; put_at(ctx, name, getattr(ctx, 'let_go_at', 0.))
        yield from tuck(ctx, (lid_arm,) if result not in ('in', 'let_go') else (lid_arm, tool_arm))
        yield from straight(ctx, staging, pose[2], speed=.2)
        if result == 'let_go':
            # The hand had shut before it was opened over the box: the tool is in the box, or it fell on the way and is
            # looked for (it counts among what may lie in the box: the lid is shut by the hand from now on).
            ctx.event('tool_lost', tool=name, arm=tool_arm, over_box=True)
            est.gone.discard(name); est.things.pop(name, None)
            yield from settle(ctx, est, 6)
            yield from look(ctx, est)
            if name in est.things and not beside_box(est, est.things[name]['xy']):
                est.things.pop(name)                        # (a piece of the box)
            return 'lost'
        if result == 'in':
            # Is it in? A tool that came down on the lid's edge or the front wall lies on the floor next to the box
            # (a pruner let go across the box did): a look from here shows it there, unless the box hides it.
            est.gone.discard(name); est.things.pop(name, None)
            yield from settle(ctx, est, 6)
            yield from look(ctx, est)
            lies = est.things.get(name)
            if lies is not None and not beside_box(est, lies['xy']):
                ctx.event('not_a_tool', tool=name, lies=np.round(lies['xy'], 3).tolist()); est.things.pop(name, None); lies = None
            if lies is not None and float(np.hypot(lies['xy'][0] - now[0], lies['xy'][1] - now[1])) < .9:
                ctx.event('not_in', tool=name, lies=np.round(lies['xy'], 3).tolist(), how=lies.get('lying')); put_at(ctx, name, None); return 'out'
            est.taken(name)
            return result
        if tool_arm in ctx.holding and holds(ctx, tool_arm, name):
            yield from carry(ctx, tool_arm)
        failed.append(pose); est.moved('toolbox')
        yield from look(ctx, est)
    return 'not_in'


# ---- the whole task ---------------------------------------------------------------------------------------
def pinch_rank(ctx, est, name, arm, here):
    """How @arm's hand would come onto the standing pruner from the stance take_tool would choose for it: 0 the way
    that holds it, 1 a way that often knocks it over (from the end its blades lean to, or with the base turned more
    than 0.7 rad from the hand: 5 of 13 such pinches in whole runs knocked it over, against 1 of 39 of the others),
    2 no stance. For the shears and a pruner that lies: 0."""
    known = est.things[name]
    if name != 'pruner' or known.get('lying'):
        return 0
    target = tool_target(est, name); started = time.time(); arm_before = ctx.arm
    try:
        for cost, arm_, pose, staging, hand in tool_options(ctx, est, name, (arm,), here):
            if time.time() - started > 20.:
                break
            if ctx.scene.path(here, staging, keepout=keepouts(est, here, staging, .04)) is None:
                continue
            ctx.arm = arm
            if tool_plan(ctx, True, target, hand, pose) is not None:
                return int((known.get('way') is not None and np.cos(hand - known['way']) < 0) or abs(wrap(pose[2] - hand)) > .7)
    finally:
        ctx.arm = arm_before
    return 2


def choose(ctx, est, tools, misses):
    """Which tool next, and into which hand: the nearer tool first (one that a hand failed at, last), into the hand
    that is not the one to open the box (the stance at the box is chosen as for the visit that follows, with the
    other tool still on the floor); of the two hands, the one that has the better way onto the tool (pinch_rank)
    if the other can open the box. Returns (name, the lid's arm or None if no stance at the box has a plan)."""
    here = np.asarray(est.base()[:2], float); order = sorted(tools, key=lambda n: (misses[n], float(np.linalg.norm(np.asarray(est.things[n]['xy']) - here))))
    for name in order:
        chosen = None
        for lid_arm in ('right', 'left'):
            # The plan is asked with the tool as the other hand will hold it (a plan made with empty hands was not
            # there any more once the pruner was in the hand, at a box in a corner).
            points, radii = held_shape(est.things[name], name); ctx.payload[other(lid_arm)] = (points, radii)
            try:
                plan = visit_plan(ctx, est, (lid_arm,), obstacles(est, but=('toolbox', name)), near=est.things[name]['xy'], carried=(name,), quiet=True)
            finally:
                ctx.payload.pop(other(lid_arm), None)
            if plan is None:
                continue
            plan['pinch'] = pinch_rank(ctx, est, name, other(lid_arm), here)
            if chosen is None or (plan['pinch'], plan['cost']) < (chosen['pinch'], chosen['cost']):
                chosen = plan
        if chosen is not None:
            ctx.event('choice', tool=name, lid_arm=chosen['lid_arm'], side=chosen['side'], asked=chosen['asked'], seconds=chosen['seconds'], pinch=chosen['pinch'])
            return name, chosen['lid_arm']
    ctx.event('choice', tool=order[0], lid_arm=None)
    return order[0], None


def look_about_box(ctx, est, names):
    """Looks at the floor round the toolbox from its other sides, the far one first, until one of the tools @names
    has shown. (From where the base stood for a visit the box hides the floor up to 0.4 m beyond it: shears that
    came down on the front wall lay 10 cm in front of the box, unseen.)"""
    place = est.box_place()
    if place is None:
        return
    x, y, yaw = place; here = np.asarray(est.base()[:2], float); bearing = float(np.arctan2(here[1] - y, here[0] - x))
    for way in sorted([yaw + k * np.pi / 2 for k in range(4)], key=lambda a: np.cos(a - bearing)):
        for far in (1.30, 1.50):
            spot = np.array([x + far * np.cos(way), y + far * np.sin(way)])
            if float(np.linalg.norm(spot - here)) < .5 or not ctx.scene.base_free(spot, margin=.02) or not clear(est, spot):
                continue
            status = yield from go(ctx, est, spot, wrap(way + np.pi), tolerance=.10)
            if status == 'no_route':
                continue
            yield from turn_to(ctx, est, wrap(way + np.pi))
            yield from settle(ctx, est, 6)
            yield from look(ctx, est)
            for name in names:
                if name in est.things and not beside_box(est, est.things[name]['xy']):
                    est.things.pop(name)                    # (a piece of the box)
            ctx.event('look_about', at=np.round(spot, 2).tolist(), status=status, known=sorted(est.things))
            if any(name in est.things for name in names):
                return
            here = np.asarray(est.base()[:2], float)
            break


def full_task(ctx, est):
    ctx.holding = {}; ctx.by_grasp = {}; ctx.worked = []; ctx.in_box = []; ctx.put = {}
    yield from tuck(ctx)
    lost = yield from find(ctx, est, ('toolbox', 'pruner', 'shears'))
    ctx.event('stage', done='find', missing=lost, things={name: np.round(k['xy'], 2).tolist() for name, k in est.things.items()})
    done = []; searches = 0; misses = dict(pruner=0, shears=0); checks = 0
    for turn in range(12):
        left = [name for name in ('pruner', 'shears') if name not in done]
        if not left and checks < 2 and 'toolbox' in est.things:
            # Both tools were let go over the box, and the episode goes on: one of them is not in it. (It ends at once
            # when the kit is together.) Where the box hid the floor from the look after the visit, a tool may lie:
            # it is looked for round the box, and taken again.
            checks += 1; tools = ('pruner', 'shears')
            yield from hold(ctx, 45)
            est.gone.difference_update(tools)
            for name in tools:
                est.things.pop(name, None)
            ctx.event('stage', done='not_over', check=checks)
            yield from look_about_box(ctx, est, tools)
            found = [name for name in tools if name in est.things]; est.gone.update(name for name in tools if name not in found)
            for name in found:
                put_at(ctx, name, None)
            ctx.event('stage', done='looked_about', found=found)
            done = [name for name in done if name not in found]; left = found
        if not left:
            break
        tools = [name for name in left if name in est.things]
        if not tools or 'toolbox' not in est.things:
            if searches >= 2:
                break
            searches += 1
            lost = yield from find(ctx, est, ['toolbox'] + left)
            ctx.event('stage', done='find_again', missing=lost, things={name: np.round(k['xy'], 2).tolist() for name, k in est.things.items()})
            continue
        if est.front is None:
            known = yield from learn_front(ctx, est)
            if not known:
                ctx.event('stage', done='front', ok=False)
                if 'toolbox' in est.things:
                    break
                continue                                    # (what was taken for the toolbox was not one: it is looked for again)
        name, lid_arm = choose(ctx, est, tools, misses)
        arm, why = yield from take_tool(ctx, est, name, ('left', 'right') if lid_arm is None else (other(lid_arm),))
        if arm is None and why == 'no_stance' and lid_arm is not None and name in est.things:
            # No stance for that hand at the tool: the other hand, if the first can open the box.
            if visit_plan(ctx, est, (other(lid_arm),), obstacles(est, but=('toolbox', name)), near=est.things[name]['xy'], carried=(name,), quiet=True) is not None:
                arm, why = yield from take_tool(ctx, est, name, (lid_arm,))
        ctx.event('stage', done='tool', tool=name, arm=arm, why=why)
        if arm is None:
            misses[name] += 1
            if name in est.things and (misses[name] >= 2 or len(tools) == 1):
                est.forget(name)                            # (it is looked for afresh)
            continue
        result = yield from into_box(ctx, est, arm, name)
        ctx.event('stage', done='box', tool=name, result=result)
        if result == 'in':
            done.append(name)
        elif result not in ('lost', 'out'):
            break
    return 'kit' if len(done) == 2 else 'one_tool' if done else 'nothing'
