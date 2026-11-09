"""v02: the hardback from the coffee table into a bookcase.

The hardback (0.30 x 0.25 x 0.047 m) lies flat on the coffee table, wider than the open fingers (10 cm). The
simulator's assisted grasp (AG) holds a thing when both fingers touch it and the ray between the fingers' grasp
points (0.4-3.4 cm behind the fingertips) crosses it. So:
1. it is seen from the table's north side (head camera: points 3-6.5 cm over the table's top, the largest patch);
2. it is pulled over the table's nearest reachable edge (east end, or north side; the sofa closes the south and west):
   the hand, pointing down and closed, comes down just beyond its far face and is drawn toward the edge, in up to
   three strokes, until it overhangs by about 6 cm;
3. the base backs away and comes to a stance beside, and the hand, level and pointing at the table, its fingers
   closing up and down, takes the part that overhangs (fingertips 1 cm out from the table's edge): AG holds it;
4. the hand, still holding it, is brought to a pose 0.45 m straight ahead of the base at the height of a bookcase's
   middle shelf, the base drives to stand facing that compartment and comes forward 0.30 m: the hardback slides in
   over the shelf; the hand opens.
Oracle trials (dev/jobs/pull_trial.py, start 301): pulled 14-15 cm in two strokes, AG held it, carried it.
"""
import numpy as np
from scipy.ndimage import label
from manip import hold, follow, move_to, drive, pose2, BASE_Z
from trash_skills import tuck, least_loaded
from skills import interpolate_cartesian
from observer import HEAD, SLOT, FOCAL

TABLE = dict(x0=-4.413, x1=-2.956, y0=-2.823, y1=-2.023, top=.356)
HB = dict(thick=.047, band=(.030, .065))       # the hardback: thickness, heights of its top-face points over the table
LOOKS = ((-3.55, -1.35, -np.pi / 2), (-2.30, -2.40, np.pi))   # where the table is looked at from: north of it facing south, east of it facing west
OVERHANG = .075                                # (at 4 cm the pinch closed on nothing, at 7 cm AG held it; the book's middle is 14 cm in)
BEHIND = .06
STROKES = (1.0, .5)                            # (the first stroke's share of the pull; the next ones what is seen to be left)
STROKE_GAIN = (.7, .8, .8)                     # how much of a stroke's length the book is assumed to come (when it is not seen)
STAND = .46                                    # pull stance: this far beyond the thing's middle (along the pull)
RESTAND = .60; LATERAL = .36                   # pinch stance: this far out from the pinch point, and to the side
TIP_OUT = .01                                  # fingertips this far out from the table's edge at the pinch
CASE = dict(x=-3.112, shelf=.394, front=-.812)  # bookcase_otwukr_2's east compartment, middle shelf
CARRY_AHEAD = .45
ACROSS_MAX = .275
CARRY_Z = .80                                  # fingertips' height while the book is carried


def hand_frame(approach, closing):
    z = -np.asarray(approach, float); z = z / np.linalg.norm(z)
    y = np.asarray(closing, float); y = y - np.dot(y, z) * z; y = y / np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def head_points_map(ob, base_pose, stride=2):
    pts = ob.points('head', stride=stride)
    world = pose2(*base_pose); m = pts @ world[:3, :3].T + world[:3, 3]; m[:, 2] += BASE_Z
    return m


def hardback_seen(ob, base_pose):
    """The hardback's box on the table as the head camera sees its top face: dict(lo (x, y), hi (x, y), n) or None."""
    m = head_points_map(ob, base_pose, stride=1)
    t = TABLE
    # The table's top as this view has it: seen from 1-1.5 m it came out 2 cm low (the head camera's pitch), and the
    # hardback's top fell out of a band set from the map's height (run a on 301)
    rect = (m[:, 0] > t['x0']) & (m[:, 0] < t['x1']) & (m[:, 1] > t['y0']) & (m[:, 1] < t['y1'])
    tops = m[rect & (m[:, 2] > t['top'] - .06) & (m[:, 2] < t['top'] + .02), 2]
    top = float(np.median(tops)) if len(tops) > 200 else t['top']
    z0, z1 = top + HB['band'][0] - .005, top + HB['band'][1]
    # where the table's east end and north side are in this view (the table is not fixed: nudged by the base it moved
    # 11-12 cm, runs e and h on 301): its top's points, a little wider than the map's outline
    near_top = (np.abs(m[:, 2] - top) < .012) & (m[:, 0] > t['x0'] - .20) & (m[:, 0] < t['x1'] + .20) & (m[:, 1] > t['y0'] - .20) & (m[:, 1] < t['y1'] + .20)
    tp = m[near_top]
    edges = None
    if len(tp) > 300:
        mid_y = (tp[:, 1] > t['y0'] + .08) & (tp[:, 1] < t['y1'] - .08); mid_x = (tp[:, 0] > t['x0'] + .15) & (tp[:, 0] < t['x1'] - .15)
        # (the near edges are under the image's lower edge from a pull stance: the far ones, plus the table's size)
        ex = float(np.percentile(tp[mid_y, 0], .5)) + (t['x1'] - t['x0']) if mid_y.sum() > 100 else None
        ey = float(np.percentile(tp[mid_x, 1], .5)) + (t['y1'] - t['y0']) if mid_x.sum() > 100 else None
        edges = (ex, ey)
    keep = (m[:, 0] > t['x0'] + .005) & (m[:, 0] < t['x1'] + .10) & (m[:, 1] > t['y0'] + .005) & (m[:, 1] < t['y1'] + .10) & (m[:, 2] > z0) & (m[:, 2] < z1)
    m = m[keep]
    if len(m) < 30:
        return None
    # (2.5 cm cells: seen from 1.3-2 m at a slant the points lie 1.5-2 cm apart over the table, and 1 cm cells broke
    # the hardback into small pieces that were all thrown away, run a on 301)
    keys = np.floor(m[:, :2] / .025).astype(int); origin = keys.min(0)
    grid = np.zeros(keys.max(0) - origin + 3, bool); grid[tuple((keys - origin + 1).T)] = True
    labels, count = label(grid, structure=np.ones((3, 3))); tag = labels[tuple((keys - origin + 1).T)]
    best = None
    for k in range(1, count + 1):
        q = m[tag == k]
        if len(q) < 30:
            continue
        lo = np.percentile(q[:, :2], 1, axis=0); hi = np.percentile(q[:, :2], 99, axis=0); ext = hi - lo
        if not (.15 < ext.max() < .40 and .12 < ext.min() < .36):
            continue
        if best is None or len(q) > best['n']:
            best = dict(lo=lo, hi=hi, n=int(len(q)), edges=edges)
    return best


def choose_pull(box):
    """(u, need) for the pull: toward the table's east end or its north side, whichever the hardback is nearer to and
    the hand reaches behind it; None if neither."""
    lo, hi = box['lo'], box['hi']
    options = []
    d_east = TABLE['x1'] - hi[0]; d_north = TABLE['y1'] - hi[1]
    # The book goes into a compartment 0.306 m wide with the edge the hand holds across it: only a pull whose held
    # edge is under ACROSS_MAX long (held by its 0.30 m edge it met the compartment's side and fell, runs g on
    # 302/303)
    if d_east < .16 and (hi[1] - lo[1]) < ACROSS_MAX:
        options.append((d_east, np.array([1., 0.]), d_east + OVERHANG, (hi[0] - lo[0])))
    if d_north < .16 and (hi[0] - lo[0]) < ACROSS_MAX:
        options.append((d_north, np.array([0., 1.]), d_north + OVERHANG, (hi[1] - lo[1])))
    if not options:
        return None
    options.sort(key=lambda o: o[0])
    return options[0][1], options[0][2], options[0][3]


def to_base(est, pt, rot_map):
    inv = np.linalg.inv(pose2(*est.base())); return (inv @ np.r_[pt[0], pt[1], pt[2] - BASE_Z, 1.])[:3], inv[:3, :3] @ rot_map


def straight(ctx, est, target_map, rot_map, rate=.008, max_steps=500):
    p, r = to_base(est, target_map, rot_map)
    way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], p, r, validator=None, step=.012)
    if way is None:
        return False
    yield from follow(ctx, way, rate=rate, max_steps=max_steps)
    return True


def hardback_task(rt, go):
    """The whole hardback phase. @go: generator function(xy, yaw) bringing the base there. Returns a short outcome."""
    ctx, est, ob = rt.ctx, rt.est, rt.ob
    yield from tuck(ctx)
    box = None
    for look in LOOKS:
        status = yield from go(look[:2], look[2])
        yield from hold(ctx, 12)
        box = hardback_seen(ob, est.base())
        ctx.event('hardback_seen', status=status, at=list(look[:2]), box=None if box is None else [np.round(box['lo'], 3).tolist(), np.round(box['hi'], 3).tolist(), box['n']])
        if box is not None:
            break
    if box is None:
        return 'not seen'
    pick = choose_pull(box)
    if pick is None:
        ctx.event('hardback', ok=False, reason='no edge within reach'); return 'far from the edges'
    u, need, extent_u = pick; side = np.array([-u[1], u[0]]); yaw = float(np.arctan2(-u[1], -u[0]))
    centre = (box['lo'] + box['hi']) / 2; half = extent_u / 2
    # 1. the pull
    stance = centre + (STAND + need) * u
    # (the coffee table is not fixed: driven to the stance by the route, the chassis brushed its corner and the table
    # with the book on it moved 8 cm, run c on 301; so to a point 0.35 m further out first, facing the table, then
    # straight in)
    way_pt = stance + .25 * u + .30 * side          # (straight out east of it the armchair shuts the way, run d on 301)
    status = yield from go(way_pt, yaw)
    if float(np.linalg.norm(np.asarray(est.base()[:2], float) - way_pt)) > .30:
        # (never straight across from afar: the robot drove through the table, run d on 301)
        ctx.event('hardback', ok=False, reason=f'no way to the pull stance ({status})'); return 'no route'
    # seen again from here, 0.4-0.8 m off (the table is not fixed and is nudged now and then; from here the book's
    # box came out right to 1-2 cm in an oracle check)
    yield from hold(ctx, 10)
    again = hardback_seen(ob, est.base())
    if again is not None:
        box = again; lo, hi = box['lo'], box['hi']
        centre = (lo + hi) / 2
        extent_u = float(np.dot(hi - lo, np.abs(u))); half = extent_u / 2
        need = float((TABLE['x1'] - hi[0]) if u[0] > .5 else (TABLE['y1'] - hi[1])) + OVERHANG
        stance = centre + (STAND + need) * u
    ctx.event('hardback_seen_near', box=None if again is None else [np.round(again['lo'], 3).tolist(), np.round(again['hi'], 3).tolist(), again['n']], need=round(need, 3))
    yield from drive(ctx, stance, yaw, speed=.10, direct=True, tolerance=.02, yaw_tolerance=.02, min_speed=.03, min_turn=.05, face_travel=False, max_steps=400)
    yield from hold(ctx, 6)
    ref = hardback_seen(ob, est.base()) or box          # (its far face from here: the reference the pulls are measured by)
    # From here the book and the table's edge are seen again and the pull planned from that: the table (with the
    # book) moved 11 cm on the way in, run h on 301. The near face is cut by the image's edge from here: it is put
    # the book's length (seen from further off) from the far face.
    edge_c = TABLE['x1'] if u[0] > .5 else TABLE['y1']
    if ref is not box:
        far = float(np.dot(ref['lo'], u)); centre_new = (ref['lo'] + ref['hi']) / 2
        across = centre_new - float(np.dot(centre_new, u)) * u
        centre = across + (far + half) * u
        seen_edge = ref.get('edges')
        if seen_edge is not None:
            e = seen_edge[0] if u[0] > .5 else seen_edge[1]
            if e is not None and abs(e - edge_c) < .25:
                edge_c = e
        need = edge_c - (far + extent_u) + OVERHANG
    ctx.event('hardback_stance', at=[round(float(x), 3) for x in est.base()], ref=[np.round(ref['lo'], 3).tolist(), np.round(ref['hi'], 3).tolist()],
              edge=round(float(edge_c), 3), need=round(float(need), 3))
    if need > .26 or need < .02:
        ctx.event('hardback', ok=False, reason='pull out of range', need=round(float(need), 3)); yield from tuck(ctx); return 'out of range'
    ctx.arm = 'left'
    rot_down = hand_frame(np.array([0., 0., -1.]), np.r_[side, 0.])
    moved = 0.
    book = np.r_[box['lo'] - .02, TABLE['top'], box['hi'] + .02, TABLE['top'] + HB['thick'] + .03]
    for stroke, share in enumerate(STROKES + (1.,)):
        # Two strokes, the second for the part the first is assumed to have left (in trials the first brought it
        # 70-100 % of the way; seen again from the pull stance, with the arm over it, its box was wrong and the next
        # stroke threw it off the table, run b on 301). Overshooting by up to 40 % leaves it on the table.
        left_to_go = share * need if stroke == 0 else min(need - moved, .12)
        # (the far face is seen up to 5 cm short of where it is: the hand comes down 7 cm beyond it)
        behind = centre + (moved - half - BEHIND) * u
        A = np.r_[behind, TABLE['top'] + HB['thick'] + .13]; B = np.r_[behind, TABLE['top'] + .012]
        C = B + np.r_[u * (left_to_go + BEHIND + .02), 0.]; D = C + np.array([0., 0., .10])
        a, ra = to_base(est, A, rot_down)
        ctx.guard = np.array([book + np.r_[moved * u, 0., moved * u, 0.]])     # (the arm's way over it keeps off it)
        sols = least_loaded(ctx, ctx.q, a, [ra], ctx.validator(hand_margin=-.05), seeds=2)
        if not sols:
            ctx.guard = np.zeros((0, 6)); ctx.event('hardback_pull', ok=False, reason='no way over it', stroke=stroke); break
        ctx.set_grip(-1.)
        st = yield from move_to(ctx, sols[0][2], rate=.04, label='pull_pre')
        ctx.guard = np.zeros((0, 6))
        ctx.event('hardback_over', stroke=stroke, status=st)
        if st != 'done':
            yield from follow(ctx, [sols[0][2]], rate=.02, max_steps=300)
        ok = True
        for target in (B, C, D):
            ok = yield from straight(ctx, est, target, rot_down)
            if not ok:
                break
        if not ok:
            ctx.event('hardback_pull', ok=False, reason='no straight way', stroke=stroke); break
        # how far it came: seen again with the arm folded (with the arm over it the box came out wrong, run b), by
        # its far face (seen right to 1-2 cm from here; the near one is cut by the image's lower edge)
        guess = moved + STROKE_GAIN[stroke] * left_to_go
        yield from tuck(ctx, ('left',))
        yield from hold(ctx, 8)
        now = hardback_seen(ob, est.base())
        far0 = float(np.dot(ref['lo'] if (u[0] > .5 or u[1] > .5) else ref['hi'], u))
        if now is not None:
            far1 = float(np.dot(now['lo'] if (u[0] > .5 or u[1] > .5) else now['hi'], u))
            seen_moved = far1 - far0
            moved = seen_moved if -.02 < seen_moved < 1.6 * need else guess
        else:
            moved = guess
        ctx.event('hardback_pull', ok=True, stroke=stroke, moved=round(moved, 3), guess=round(guess, 3), seen=now is not None, need=round(need, 3))
        if need - moved < .015:
            break
    # 2. clear: straight up, fold, back off, to the pinch stance
    tip = ctx.tip()[:3, 3]; rotation = ctx.tip()[:3, :3]
    way = interpolate_cartesian(ctx.ik, ctx.q, tip, tip + np.array([0., 0., max(0., .65 - BASE_Z - tip[2])]), rotation, step=.02)
    if way is not None:
        yield from follow(ctx, way, rate=.01, max_steps=300)
    yield from tuck(ctx, ('left',))
    here = np.asarray(est.base()[:2], float)
    yield from drive(ctx, here + .25 * u, yaw, speed=.12, direct=True, tolerance=.04, face_travel=False, max_steps=300)
    along_tip = edge_c + TIP_OUT
    pinch_xy = centre + (along_tip - float(np.dot(centre, u))) * u
    P = np.r_[pinch_xy, TABLE['top'] + HB['thick'] / 2]
    st_xy = pinch_xy + RESTAND * u + LATERAL * side
    yield from drive(ctx, st_xy, yaw, speed=.15, direct=True, tolerance=.02, yaw_tolerance=.02, min_speed=.03, min_turn=.05, face_travel=False, max_steps=500)
    yield from hold(ctx, 6)
    # 3. the pinch of what overhangs
    rot_level = hand_frame(np.r_[-u, 0.], np.array([0., 0., 1.]))
    p, rp = to_base(est, P, rot_level); pre = p + .10 * rp[:, 2]
    # (the book, as far as it is assumed to have come, is a box for the arm's way to the pose before the pinch: the
    # arm swinging there knocked it off the table, run f on 302)
    guard = book + np.r_[moved * u, 0., moved * u, 0.]
    # (only the part on the table: the hand comes to the part that overhangs; with the whole book as a box the pose
    # before the pinch was inside it and had no plan, run v1 on 301)
    k = 0 if u[0] > .5 else 1
    guard[3 + k] = min(guard[3 + k], edge_c - .01)
    ctx.guard = np.array([guard]) if guard[3 + k] > guard[k] else np.zeros((0, 6))
    sols = least_loaded(ctx, ctx.q, pre, [rp], ctx.validator(hand_margin=-.05), seeds=2)
    if not sols:
        ctx.guard = np.zeros((0, 6)); ctx.event('hardback_pinch', ok=False, reason='no way'); return 'no pinch plan'
    ctx.set_grip(1.)
    st = yield from move_to(ctx, sols[0][2], rate=.04, label='hb_pinch_pre')
    ctx.guard = np.zeros((0, 6))
    if st != 'done':
        yield from follow(ctx, [sols[0][2]], rate=.02, max_steps=300)
    way = interpolate_cartesian(ctx.ik, ctx.q, ctx.tip()[:3, 3], p, rp, validator=None, step=.008)
    if way is None:
        sols2 = least_loaded(ctx, ctx.q, p, [rp], None, seeds=2)
        way = [sols2[0][2]] if sols2 else None
    if way is None:
        ctx.event('hardback_pinch', ok=False, reason='no way in'); return 'no way in'
    yield from follow(ctx, way, rate=.008, max_steps=300)
    yield from hold(ctx, 6)
    world = pose2(*est.base()); tip = ctx.tip()
    ctx.event('hardback_at_pinch', tip=[round(float(x), 3) for x in (world[:3, :3] @ tip[:3, 3] + world[:3, 3] + [0., 0., BASE_Z])], wanted=np.round(P, 3).tolist())
    ctx.set_grip(-1.)
    yield from hold(ctx, 40)
    fingers = ctx.fingers().copy(); held = float(np.sum(fingers)) > .01
    ctx.event('hardback_pinch', ok=held, fingers=np.round(fingers, 4).tolist())
    if not held:
        ctx.set_grip(1.); yield from hold(ctx, 10); yield from tuck(ctx, ('left',))
        return 'missed'
    # 4. carry and into the bookcase
    tip = ctx.tip()[:3, 3]; rotation = ctx.tip()[:3, :3]
    way = interpolate_cartesian(ctx.ik, ctx.q, tip, tip + np.array([0., 0., .05]), rotation, step=.01)
    if way is not None:
        yield from follow(ctx, way, rate=.01, max_steps=200)
    here = np.asarray(est.base()[:2], float)
    yield from drive(ctx, here + .45 * u, yaw, speed=.12, direct=True, tolerance=.04, face_travel=False, max_steps=400)
    # carried high in front of the base (turning with the book at shelf height by the table, the book was knocked out
    # of the hand and fell, runs g on 302/303), lowered to the shelf's height at the bookcase
    tipz = CASE['shelf'] + .015 + HB['thick'] / 2
    carry_r = hand_frame(np.array([1., 0., 0.]), np.array([0., 0., 1.]))
    sols = least_loaded(ctx, ctx.q, np.array([CARRY_AHEAD - .05, 0., CARRY_Z - BASE_Z]), [carry_r], None, seeds=2)
    if sols:
        yield from follow(ctx, [sols[0][2]], rate=.012, max_steps=500)
    yield from hold(ctx, 10)
    stance_b = (CASE['x'], CASE['front'] - .01 - (CARRY_AHEAD + .30))
    status = yield from go(stance_b, np.pi / 2)
    if float(np.linalg.norm(np.asarray(est.base()[:2], float) - np.asarray(stance_b))) > .30:
        ctx.event('hardback', ok=False, reason=f'no way to the bookcase ({status})')
        ctx.set_grip(1.); yield from hold(ctx, 20); yield from tuck(ctx)
        return 'no route to the bookcase'
    yield from drive(ctx, stance_b, np.pi / 2, speed=.12, direct=True, tolerance=.02, yaw_tolerance=.02, min_speed=.03, min_turn=.05, face_travel=False, max_steps=300)
    sols = least_loaded(ctx, ctx.q, np.array([CARRY_AHEAD, 0., tipz - BASE_Z]), [carry_r], None, seeds=2)
    if sols:
        yield from follow(ctx, [sols[0][2]], rate=.01, max_steps=500)
    yield from hold(ctx, 10)
    here = np.asarray(est.base()[:2], float)
    yield from drive(ctx, here + np.array([0., .30]), np.pi / 2, speed=.08, direct=True, tolerance=.02, yaw_tolerance=.03, face_travel=False, max_steps=400)
    yield from hold(ctx, 10)
    ctx.set_grip(1.)
    yield from hold(ctx, 40)
    here = np.asarray(est.base()[:2], float)
    yield from drive(ctx, here - np.array([0., .35]), np.pi / 2, speed=.1, direct=True, tolerance=.03, face_travel=False, max_steps=300)
    yield from tuck(ctx)
    ctx.event('hardback', ok=True, status=status)
    return 'into the bookcase'
