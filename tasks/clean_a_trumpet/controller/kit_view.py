"""The toolbox, the pruner and the shears on the garage floor, from the head camera.

Depth: points that stand above the floor where the map has free floor are grouped; a group is one of the three
things by its height and size (the toolbox 0.35 high and 0.62 long, the pruner 0.23 high and a hand's breadth
across, the shears 2.5 cm high and 13 cm long). Colour is used for one thing only: which long side of the toolbox
is its front (two black latches on red; the hinge side is plain red, and the two sides are the same in depth).

Everything returned is in the map frame M.
"""
import numpy as np
from scipy.ndimage import label, distance_transform_edt
from manip import BASE_Z

CELL = .02
ABOVE_FLOOR = .012                  # points this far above the floor belong to something (the shears are 2.5 cm high)
BAND_TOP = .45
CLEAR = .10                         # ... and this far from anything the map has (walls, the car, the door's leaf)
BESIDE = (.02, .045)                # round the toolbox where it is known to stand: its own points reach this far beyond its outline; points higher than the shears within the second are its too
REACH = 4.5                         # points further from the base than this are not used
BOX = dict(long=.608, short=.243, plate=(.265, .30), top=(.25, .40))            # the toolbox: its lid's plate as the camera shows it, heights above the floor
PRUNER = dict(top=(.17, .25), across=.11, blades=(.15, .205), tip=.185, foot=.09)       # the pruner: height of its top, widest extent, the band of its blades, the height its handles show down to
# Its pair of handles is 6.8 cm long and stands under the blades off the middle: the blades lean to one end of the
# pair (their top 2 cm beyond the pair's middle). A hand that comes along the pair from the other end, the way the
# blades lean, held it upright 4 times of 4; from the end the blades lean to it tilted in the hand or fell (2 of 9).
HANDLES = dict(band=(.02, .12), half=.0335, deeper=.005, told=.006, skew=np.radians(4.5))       # the handles: the heights they show at, half their length, how far past their middle the fingers' ends go, the least lean that tells its way, how far the blades' flat sides are turned from the pair's line
# (A piece of the toolbox at the image's border was taken for the pruner once: a group that touches the border is not a
# tool, and the pruner shows from its top down to its handles.)
SHEARS = dict(top=(.014, .05), long=(.07, .17), across=.08)                    # the shears lying shut
# The pruner knocked over lies on a broad side: 3 cm high, 21-22 cm long, 6 cm across at its handles (up to 9 cm
# where its blades' joint has given), 4 cm at its blades. A hand takes it across its handles, 2 cm from the middle
# of its length: next to where its weight is. (The pair of handles is 6.3 cm across and the fingers open 10: one
# finger shoves it against the other, which stands at the end of its travel, and it is held by that squeeze
# alone. Taken there it hung nearly level and stayed in the hand through a shake 4 times of 4; taken 3 cm further
# towards the handles' ends it hung blades down and fell 3 times of 6.)
LYING = dict(top=(.02, .06), long=(.19, .25), across=(.04, .12), grip=.020, ends=.010)
# ... or on a narrow side, one handle on the other (so a hand that came along the pair leaves it): 7 cm high, 21-22 cm
# long, 3 cm across. A hand takes it across those 3 cm at the middle of its length, as it takes the pruner standing
# (6 of 6, anywhere from 4 cm towards the handles' ends to 2 cm towards the blades, the fingers' ends 2 to 4.5 cm up).
EDGE = dict(top=(.055, .09), long=(.19, .25), across=.06)
LATCH = dict(along=(.15, .28), height=(.10, .25), face=.03, near=1.5, full=250)                     # where the latches are on the front face (own |y|, height), and how near to the face a point counts (3 cm: a box known from a far look is turned a few degrees from where it stands, and with 1.2 cm a look from 1.4 m had 39 points on the face instead of 600)


def rectangle(points, step=np.radians(1.)):
    """The smallest rectangle round @points (N, 2): centre, angle of its long side, long, short."""
    best = None
    for angle in np.arange(0., np.pi / 2, step):
        c, s = np.cos(angle), np.sin(angle); u = c * points[:, 0] + s * points[:, 1]; v = -s * points[:, 0] + c * points[:, 1]
        lo = np.array([np.percentile(u, .5), np.percentile(v, .5)]); hi = np.array([np.percentile(u, 99.5), np.percentile(v, 99.5)]); area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, angle, lo, hi)
    area, angle, lo, hi = best; c, s = np.cos(angle), np.sin(angle); mu, mv = .5 * (lo + hi)
    centre = np.array([c * mu - s * mv, s * mu + c * mv]); size = hi - lo
    return (centre, angle, float(size[0]), float(size[1])) if size[0] >= size[1] else (centre, angle + np.pi / 2, float(size[1]), float(size[0]))


def main_axis(points):
    """Direction (rad, modulo half a turn) along which @points (N, 2) spread most, and the spreads along and across it."""
    centred = points - points.mean(axis=0); _, s, vt = np.linalg.svd(centred, full_matrices=False)
    return float(np.arctan2(vt[0, 1], vt[0, 0])), float(s[0] / np.sqrt(len(points))), float(s[1] / np.sqrt(len(points)))


class KitView:
    def __init__(self, scene):
        self.scene = scene
        # The door to the corridor stands shut in this task, and the map keeps its leaf apart from what blocks the
        # base (other tasks push it open): for what stands on the floor it is a wall. (Its foot, seen from 3.5 m, was
        # taken for the toolbox once, and the search ended there.)
        self.door = distance_transform_edt(~scene.door) * scene.res if scene.door is not None and scene.door.any() else None

    def groups(self, ob, base_pose, stride=1, box=None):
        """Groups of points above the free floor: list of dict(points (N, 3 in M), top, centre (xy), n, cut (the group
        touches the image's border)).
        box: dict(xy, yaw) of the toolbox where it is known to stand. Its points are then one group, and what lies
        next to it is grouped apart from it (a pruner that fell beside the box lay 5 cm from its wall and was one
        group with it: not seen)."""
        local = ob.points('head', stride); height = local[:, 2] + BASE_Z
        local = local[(height > ABOVE_FLOOR) & (height < BAND_TOP) & (np.hypot(local[:, 0], local[:, 1]) < REACH)]
        if len(local) < 5:
            return []
        local = ob.clear_of_robot(local)
        x, y, yaw = base_pose; c, s = np.cos(yaw), np.sin(yaw)
        xy = np.c_[x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]]; z = local[:, 2] + BASE_Z
        i, j = self.scene.cell(xy); ok = self.scene.inside(i, j); ic, jc = np.clip(i, 0, self.scene.shape[0] - 1), np.clip(j, 0, self.scene.shape[1] - 1)
        ok &= (self.scene.top[ic, jc] < .03) & (self.scene.distance[ic, jc] > CLEAR)
        if self.door is not None:
            ok &= self.door[ic, jc] > CLEAR
        xy, z, local = xy[ok], z[ok], local[ok]
        if len(xy) < 5:
            return []
        out = []
        if box is not None:
            cb, sb = np.cos(box['yaw']), np.sin(box['yaw']); d = xy - np.asarray(box['xy'], float); across = np.abs(cb * d[:, 0] + sb * d[:, 1]); along = np.abs(-sb * d[:, 0] + cb * d[:, 1])
            its = (across < .5 * BOX['short'] + BESIDE[0]) & (along < .5 * BOX['long'] + BESIDE[0])
            if its.sum() >= 5:
                out.append(dict(points=np.c_[xy[its], z[its]], top=float(np.percentile(z[its], 98)), centre=xy[its].mean(axis=0), n=int(its.sum()), cut=bool(ob.at_border(local[its]))))
            # (what is left within a finger's breadth of the box's outline is the box's own: its handle's shadow, a wall seen askew)
            rest = ~its & ~((across < .5 * BOX['short'] + BESIDE[1]) & (along < .5 * BOX['long'] + BESIDE[1]) & (z > SHEARS['top'][1]))
            xy, z, local = xy[rest], z[rest], local[rest]
            if len(xy) < 5:
                return out
        keys = np.floor(xy / CELL).astype(int); origin = keys.min(0); grid = np.zeros(keys.max(0) - origin + 3, bool); grid[tuple((keys - origin + 1).T)] = True
        tags, count = label(grid, structure=np.ones((3, 3))); tag = tags[tuple((keys - origin + 1).T)]
        for index in range(1, count + 1):
            member = tag == index
            if member.sum() < 5:
                continue
            out.append(dict(points=np.c_[xy[member], z[member]], top=float(np.percentile(z[member], 98)), centre=xy[member].mean(axis=0), n=int(member.sum()), cut=bool(ob.at_border(local[member]))))
        return out

    def see(self, ob, base_pose, stride=1, box=None):
        """The three things as this look shows them: {name: dict or None}.
        toolbox: xy (its middle), yaw (of its long side, modulo half a turn: which long side is the front is not
        known from depth), long, short, whole (its lid's plate shows whole), n.
        pruner: xy (where the fingers' ends go on its pair of handles; of its blades if the look does not tell the
        way), axis (rad: along the blades' flat sides and the pair of handles, modulo half a turn; None if too few
        points show it), way (rad: the direction along the axis in which the blades lean, which is the way a hand
        comes onto the handles; None if the look does not tell), top, n. Knocked over: lying 'flat' (on a broad
        side) or 'edge' (on a narrow one), xy (where a hand takes it from above; for 'flat' across its handles, or
        at the middle of its length if the look does not tell its ends apart), along (rad: its length, modulo half
        a turn), axis (rad: from its handles to its blades, or None).
        shears: xy (their middle), axis (rad: from the handles to the blades' points; None if the ends cannot be told
        apart), long, n.
        box: where the toolbox is known to stand (see groups)."""
        out = dict(toolbox=None, pruner=None, shears=None)
        for group in self.groups(ob, base_pose, stride, box):
            points = group['points']; top = group['top']; extent = np.ptp(points[:, :2], axis=0)
            if BOX['top'][0] < top < BOX['top'][1] and max(extent) > .18:
                plate = points[(points[:, 2] > BOX['plate'][0]) & (points[:, 2] < BOX['plate'][1])]
                if len(plate) < 30:
                    continue
                centre, angle, long, short = rectangle(plate[:, :2])
                if not group['cut'] and abs(short - BOX['short']) < .035 and BOX['long'] - CLEAR - .02 < long < BOX['long'] - .035:
                    # A box that stands with an end towards a wall, as near as the starts have it (its middle 0.30 m
                    # from the wall, its end at it), loses up to 10 cm of that end of its lid to the margin kept round
                    # what the map has. Its other end shows: the lid is taken to its full length from there. (Left as it
                    # showed, 0.54 m long, it was never "whole", and the look for its front never told: start 305.)
                    along = np.array([np.cos(angle), np.sin(angle)]); ends = [centre + sign * .5 * long * along for sign in (1., -1.)]
                    near = [float(self.scene.clearance(end + sign * .03 * along)) for end, sign in zip(ends, (1., -1.))]
                    if min(near) < CLEAR + .04 and max(near) > CLEAR + .08:
                        cut = int(np.argmin(near)); centre = ends[1 - cut] + (1. if cut == 0 else -1.) * .5 * BOX['long'] * along; long = BOX['long']
                whole = abs(long - BOX['long']) < .035 and abs(short - BOX['short']) < .035 and not group['cut']
                thing = dict(xy=centre, yaw=float((angle + np.pi / 2) % np.pi), long=long, short=short, whole=bool(whole), n=group['n'])       # yaw: of the box's own x, across it
                if out['toolbox'] is None or (thing['whole'], thing['n']) > (out['toolbox']['whole'], out['toolbox']['n']):
                    out['toolbox'] = thing
            elif PRUNER['top'][0] < top < PRUNER['top'][1] and max(extent) < PRUNER['across'] and not group['cut'] and group['n'] >= 9 and float(points[:, 2].min()) < PRUNER['foot']:
                blades = points[(points[:, 2] > PRUNER['blades'][0]) & (points[:, 2] < PRUNER['blades'][1])]; tip = points[points[:, 2] > PRUNER['tip']]
                where = blades if len(blades) >= 6 else points
                axis = main_axis(tip[:, :2])[0] if len(tip) >= 8 else None
                thing = dict(xy=where[:, :2].mean(axis=0), axis=axis, way=None, top=top, n=group['n'])
                handles = points[(points[:, 2] > HANDLES['band'][0]) & (points[:, 2] < HANDLES['band'][1])]
                if axis is not None and len(handles) >= 6:
                    # Along the blades, away from the camera: the near end of the pair of handles shows whatever the
                    # side it is seen from; its middle is half its length further on.
                    along = np.array([np.cos(axis), np.sin(axis)]); away = thing['xy'] - np.asarray(base_pose[:2], float)
                    if along @ away < 0:
                        along = -along
                    middle = float(np.percentile(handles[:, :2] @ along, 3)) + HANDLES['half']; lean = float(np.mean(tip[:, :2] @ along)) - middle
                    if abs(lean) > HANDLES['told']:
                        sign = 1. if lean > 0 else -1.; across = np.array([-along[1], along[0]])
                        thing.update(way=float(np.arctan2(sign * along[1], sign * along[0]) - HANDLES['skew']), xy=(middle + sign * HANDLES['deeper']) * along + float(np.mean(tip[:, :2] @ across)) * across)
                if out['pruner'] is None or out['pruner'].get('lying') or thing['n'] > out['pruner']['n']:
                    out['pruner'] = thing
            elif SHEARS['top'][0] < top < SHEARS['top'][1] and group['n'] >= 6 and not group['cut']:
                centre, angle, long, short = rectangle(points[:, :2]) if group['n'] >= 12 else (points[:, :2].mean(axis=0), 0., float(max(extent)), float(min(extent)))
                if LYING['top'][0] < top < LYING['top'][1] and LYING['long'][0] < long < LYING['long'][1] and LYING['across'][0] < short < LYING['across'][1] and group['n'] >= 12:
                    # The pruner lying flat. Its handles are the wider end.
                    c, s = np.cos(angle), np.sin(angle); u = (points[:, :2] - centre) @ np.array([c, s]); v = (points[:, :2] - centre) @ np.array([-s, c])
                    ends = [float(np.ptp(v[side])) if side.sum() > 5 else 0. for side in (u < -.25 * long, u > .25 * long)]; axis = None; xy = centre
                    if abs(ends[0] - ends[1]) > LYING['ends']:
                        axis = float(angle if ends[0] > ends[1] else angle + np.pi); xy = centre - LYING['grip'] * np.array([np.cos(axis), np.sin(axis)])       # axis: from the handles to the blades
                    thing = dict(xy=xy, axis=axis, way=None, along=float(angle % np.pi), lying='flat', long=long, short=short, top=top, n=group['n'])
                    if out['pruner'] is None or (out['pruner'].get('lying') and thing['n'] > out['pruner']['n']):
                        out['pruner'] = thing
                    continue
                if not (SHEARS['long'][0] < long < SHEARS['long'][1]) or short > SHEARS['across']:
                    continue
                axis = None
                if group['n'] >= 40:
                    # The handles are the wider and higher end.
                    c, s = np.cos(angle), np.sin(angle); u = (points[:, :2] - centre) @ np.array([c, s]); v = (points[:, :2] - centre) @ np.array([-s, c])
                    ends = [(np.ptp(v[side]) if side.sum() > 5 else 0., side.sum()) for side in (u < -.25 * long, u > .25 * long)]
                    if abs(ends[0][0] - ends[1][0]) > .012:
                        axis = float(angle if ends[0][0] > ends[1][0] else angle + np.pi)                # from the wide end to the narrow one
                thing = dict(xy=centre, axis=axis, along=float(angle % np.pi), long=long, short=short, n=group['n'])
                if out['shears'] is None or thing['n'] > out['shears']['n']:
                    out['shears'] = thing
            elif EDGE['top'][0] < top < EDGE['top'][1] and group['n'] >= 12 and not group['cut']:
                centre, angle, long, short = rectangle(points[:, :2])
                if EDGE['long'][0] < long < EDGE['long'][1] and short < EDGE['across']:
                    thing = dict(xy=centre, axis=None, way=None, along=float(angle % np.pi), lying='edge', long=long, short=short, top=top, n=group['n'])
                    if out['pruner'] is None or (out['pruner'].get('lying') and thing['n'] > out['pruner']['n']):
                        out['pruner'] = thing
        return out

    def front(self, ob, image, base_pose, box):
        """Which long side of the toolbox is its front, from the head camera's colour image @image (H x W x 3, as the
        depth image): +1 if the side its yaw points at (own +x) shows the latches, -1 if the other one does, 0 if
        this look cannot tell (the near long side does not show, or shows from too far).
        box: dict(xy, yaw) as see() gives it."""
        local = ob.points('head', 1); u, v, size = ob.pixels(local, 'head'); x, y, yaw = base_pose; c, s = np.cos(yaw), np.sin(yaw)
        xy = np.c_[x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]]; z = local[:, 2] + BASE_Z
        cb, sb = np.cos(box['yaw']), np.sin(box['yaw']); d = xy - np.asarray(box['xy']); own_x = cb * d[:, 0] + sb * d[:, 1]; own_y = -sb * d[:, 0] + cb * d[:, 1]
        robot = np.array([x, y]) - np.asarray(box['xy']); side = 1. if cb * robot[0] + sb * robot[1] > 0 else -1.
        face = (np.abs(side * own_x - .1215) < LATCH['face']) & (np.abs(own_y) > LATCH['along'][0]) & (np.abs(own_y) < LATCH['along'][1]) & (z > LATCH['height'][0]) & (z < LATCH['height'][1])
        if face.sum() < 150:
            return 0, dict(points=int(face.sum()))
        scale = image.shape[0] / size
        rgb = image[np.clip((v[face] * scale).astype(int), 0, image.shape[0] - 1), np.clip((u[face] * scale).astype(int), 0, image.shape[1] - 1)].astype(int)
        grey = (rgb[:, 0] < 2.2 * (rgb[:, 1] + rgb[:, 2]) / 2 + 12) & (rgb.max(axis=1) < 110)
        share = float(grey.mean()); told = dict(points=int(face.sum()), latches=round(share, 3), range=round(float(np.linalg.norm(robot)), 2))
        # Latches seen: this is the front, from any range. None seen: the hinge side, if the look is near and full
        # enough to have shown them (on frames from 1.6 and 2.6 m the front's latches did not show, 2 of 26).
        near = told['range'] < LATCH['near'] and told['points'] >= LATCH['full']
        return (int(side) if share > .05 else int(-side) if share < .012 and near else 0), told
