"""Sweeping the garage: the broom, how it is held, and the plan of straight sweeps that takes the sand and the dust away.

A particle of sand or dust on the floor goes when it is inside the broom's bounding box (axis-aligned in the scene, its
margin 2 cm): the broom need only pass over it a centimetre above the floor (dev/jobs/probe_sweep.py, broom_trial.py:
driven 2 m with the broom held so, four of the particles under its way went). The broom is held by the left hand near
the end of its handle, lying flat a centimetre and a half over the floor, across the base's heading and to its left:
then it sweeps a strip 1.35 m wide beside the base as the base drives. With the base's heading along the scene's axes
the broom's box is as wide as the broom is long.

Garage floor: x -9.27 to -1.47, y 2.26 to 8.75; the car stands on x -6.85 to -1.78, y 5.82 to 7.97. The sand and the
dust lie on the floor round the car: in an L, the strip x -9.27 to -6.85 the garage's whole depth (up to y 8.25) and
the band y 2.75 to 5.5 in front of the car.
"""
import numpy as np

FOOTPRINT = (-.40, .24, .345)       # chassis: rear, front and half width in the base frame
MARGIN = .02                        # the particle remover's margin round the broom's box
CAR = ((-6.87, 5.80), (-1.76, 7.99))         # the car's box on the floor (map), with 2 cm round it
# The broom in the base frame as the left hand holds it (taken 1.08 m from the middle of its head, the hand beside the
# base's middle, 0.55 m to its left, the trunk squatting upright over the base): its handle along +y from the
# stub behind the hand (0.13 m: clear of the chassis' side) to the head, its head 0.30 deep along the handle and 0.45
# wide across it.
# (Held 0.5 m ahead of the base the trunk bent far over: the head camera 1.0 m ahead, meeting the walls at the legs'
# ends, which shook the broom off the floor.)
# Where the base has no room beside the handle's end, the hand takes the broom from in front of it instead (the
# forward hold: 0.5 m ahead and 0.16 m to the side, the trunk bent far over) and only to carry it to where it is put
# down again in the open (sweep_pipeline.relocate).
# along: how far from the head's middle the hand takes the handle (the handle's end is 1.21 m from it).
HOLDS = dict(
    side=dict(hand=(0., .55), along=1.08, handle=((-.02, .42), (.02, 1.48)), head=((-.225, 1.48), (.225, 1.78)), torso=(-.45, .46, .25)),
    forward=dict(hand=(.50, .16), along=1.00, handle=((.48, -.04), (.52, 1.01)), head=((.275, 1.01), (.725, 1.31)), torso=(.24, 1.06, .35)),
)
# (taken nearer the head where the handle's end lies by the car: the trunk then stays clear of it, 304)
for _along in (.8, .6, .4):
    HOLDS[f'forward_{int(_along * 10)}'] = dict(hand=(.50, .16), along=_along, handle=((.48, .16 - (1.21 - _along)), (.52, .16 + _along - .15)),
                                                head=((.275, .16 + _along - .15), (.725, .16 + _along + .15)), torso=(.24, 1.06, .35))
# (along: where neither has room, between the car and the garage door, the base stands beyond the handle's end facing
# along the broom, the hand takes the handle there and the broom lies ahead of the base: it is carried out so)
HOLDS['along'] = dict(hand=(.50, .10), along=1.00, handle=((.29, .08), (1.35, .12)), head=((1.35, -.125), (1.65, .325)), torso=(.24, 1.06, .35))
HAND = HOLDS['side']['hand']
HANDLE = HOLDS['side']['handle']             # the handle's box (x0, y0), (x1, y1) in the base frame
HEAD = HOLDS['side']['head']                 # the head's box


def boxes(side='left', hold='side'):
    """The broom's two boxes in the base frame for the hand on @side (the right hand holds it mirrored)."""
    h = HOLDS[hold]
    if side == 'left':
        return (h['handle'], h['head'])
    return tuple(((x0, -y1), (x1, -y0)) for (x0, y0), (x1, y1) in (h['handle'], h['head']))


def outline(pose, side='left', hold='side'):
    """The broom's two boxes' corners in the map frame for the base at @pose: (8, 2)."""
    x, y, yaw = pose; c, s = np.cos(yaw), np.sin(yaw); out = []
    for (x0, y0), (x1, y1) in boxes(side, hold):
        for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            out.append((x + c * u - s * v, y + s * u + c * v))
    return np.array(out)


def broom_box(pose, side='left', hold='side'):
    """The broom's axis-aligned box on the floor (x0, y0, x1, y1) for the base at @pose."""
    o = outline(pose, side, hold); return np.r_[o.min(axis=0), o.max(axis=0)]


def broom_free(scene, pose, height=.03, step=.04, side='left', hold='side'):
    """Does the broom, a centimetre and a half over the floor, keep clear of what the map has (anything higher than
    @height above the floor) for the base at @pose?"""
    x, y, yaw = pose; c, s = np.cos(yaw), np.sin(yaw); pts = []
    for (x0, y0), (x1, y1) in boxes(side, hold):
        for u in np.arange(x0, x1 + 1e-6, step):
            for v in np.arange(y0, y1 + 1e-6, step):
                pts.append((x + c * u - s * v, y + s * u + c * v))
    pts = np.array(pts); i, j = scene.cell(pts); ok = scene.inside(i, j)
    if not ok.all():
        return False
    # (the map has the car as hulls over a cleared height field: its body and wheels are kept out of here by its box)
    (cx0, cy0), (cx1, cy1) = CAR
    if np.any((pts[:, 0] > cx0) & (pts[:, 0] < cx1) & (pts[:, 1] > cy0) & (pts[:, 1] < cy1)):
        return False
    return bool(np.all(scene.top[i, j] < height))


# The body over the base while the hand holds the broom (HOLDS' torso: from, to (along the heading) and half width
# in the base frame, at 0.6 m and above); in the side hold take_broom keeps the posture so (POSTURE). (In the forward
# hold the head camera stood 1.0 m ahead of the base's middle and met the wall at every leg's end of run 301;
# squatting it leans back to 0.56 m behind the middle and touched the garage door at legs' ends, 306, without harm
# to the broom: the box lets it come 0.1 m into what the map has behind, for legs that reach the door.)
POSTURE = dict(trunk=.45, arm=.55, rear=-.60)  # how far ahead of (behind) the base's middle the trunk and the arms may reach (m)


def torso_free(scene, pose, height=.55, step=.04, hold='side'):
    """Does the body over the broom keep clear of what the map has higher than @height, and of the car?"""
    x, y, yaw = pose; c, s = np.cos(yaw), np.sin(yaw); front, reach, half = HOLDS[hold]['torso']
    u, v = np.meshgrid(np.r_[np.arange(front, reach, step), reach], np.r_[np.arange(-half, half, step), half], indexing='ij')
    pts = np.c_[x + c * u.ravel() - s * v.ravel(), y + s * u.ravel() + c * v.ravel()]
    i, j = scene.cell(pts)
    if not scene.inside(i, j).all():
        return False
    (cx0, cy0), (cx1, cy1) = CAR
    if np.any((pts[:, 0] > cx0) & (pts[:, 0] < cx1) & (pts[:, 1] > cy0) & (pts[:, 1] < cy1)):
        return False
    return bool(np.all(scene.top[i, j] < height))


E, N, W, S = 0., np.pi / 2, np.pi, -np.pi / 2
# Legs: (heading, from, to) of the base's middle, each driven either way along its heading; the broom lies to the
# left (right hand: right) of the base. Found by dev/plan_sweep.py --side left (right) --margin -0.02 --min-gain 2
# --max-legs 14 over the starts not held out (the margin less than the remover's keeps neighbouring legs overlapping:
# a particle 4 cm inside a leg's edge was left, 306).
PLANS = dict(
    default=[(W, (-8.35, 4.60), (-2.05, 4.60)), (E, (-8.35, 3.75), (-2.05, 3.75)), (W, (-8.35, 7.45), (-7.35, 7.45)), (E, (-8.35, 6.50), (-7.35, 6.50)), (E, (-8.35, 4.30), (-7.10, 4.30)), (W, (-8.35, 4.25), (-2.05, 4.25)), (E, (-8.35, 3.95), (-2.05, 3.95)), (W, (-8.30, 5.10), (-2.05, 5.10))],
    right=[(E, (-8.35, 4.60), (-2.05, 4.60)), (W, (-8.35, 3.75), (-2.05, 3.75)), (E, (-8.35, 7.45), (-7.35, 7.45)), (W, (-8.35, 6.50), (-7.35, 6.50)), (W, (-8.35, 4.30), (-7.10, 4.30)), (E, (-8.35, 4.30), (-2.05, 4.30)), (W, (-8.35, 3.95), (-2.05, 3.95)), (W, (-8.30, 2.85), (-2.05, 2.85))],
)


def trimmed(scene, legs, side='left', step=.05):
    """@legs cut to the longest stretch of each where the chassis and the broom keep clear of the map."""
    out = []
    for heading, a, b in legs:
        a, b = np.asarray(a, float), np.asarray(b, float); n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
        ok = [scene.footprint_free(*(a + t * (b - a)), heading, FOOTPRINT, .02) and broom_free(scene, (*(a + t * (b - a)), heading), side=side) for t in np.linspace(0., 1., n)]
        best = None; k = 0
        while k < n:
            if not ok[k]:
                k += 1; continue
            j = k
            while j + 1 < n and ok[j + 1]:
                j += 1
            if best is None or j - k > best[1] - best[0]:
                best = (k, j)
            k = j + 1
        if best is not None and best[1] > best[0]:
            t0, t1 = best[0] / (n - 1), best[1] / (n - 1)
            out.append((heading, tuple(a + t0 * (b - a)), tuple(a + t1 * (b - a))))
    return out
