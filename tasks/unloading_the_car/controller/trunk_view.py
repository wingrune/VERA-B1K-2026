"""The two bags in the open trunk, from the head camera's depth image.

What the camera shows inside the trunk, above its floor, is matched against the shape of each bag (priors/bags.npz:
the height of a bag's upper surface over its own x, y, from its collision hulls): the place and heading at which
most of the points lie on that surface. The satchel first (it is the taller one, and a wedge: its heading shows
in its profile), then the briefcase on what the satchel does not explain.

Everything here is in the map frame M; the trunk is where the map has it (the car does not move).
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull

# The inside of the trunk (map frame): where things can lie, and its floor as the car's collision shape has it (the
# bags rest on that; the carpet the camera sees lies 3 to 6 cm lower).
TRUNK_X = (-6.90, -5.70); TRUNK_Y = (6.37, 7.41); TRUNK_TOP = 1.05       # (the camera shows the side walls at y 6.35 and 7.42, the far wall from x -5.68)
# (From x -6.90: over the sill and the bumper too. A briefcase was left lying across the sill, half outside, when the
# satchel was drawn out past it, on start 216; with the points cut off at the sill it was taken to lie 17 cm further
# in, and the hand closed on nothing six times.)
SILL_TOP = .770; SILL_FROM = -6.69  # the car's body is nowhere higher than this from there to the bumper (its collision shape: 0.765-0.770 at the sill, falling to the bumper's end at x -6.84)
FLOOR_AT_SILL = .761; FLOOR_SLOPE = -.061; SILL_X = -6.75
ABOVE_FLOOR = .025                  # points this far above the floor belong to something lying on it
# The trunk's walls as the car's collision shape has them (a bag cannot lie beyond them): the side walls stand 2 to
# 4 cm outside what the camera shows of them, so a bag against a wall is seen cut off there. (Over the training starts
# the outlines of bags lying flat reach y 6.298 and 7.483.) It is a bag's outline that is held against them, not the
# box round it: the box of a satchel that lies askew against a wall has a corner up to 7 cm beyond the wall, on one
# start in eight; with the box the true place was ruled out, and on one start neither bag was found.
WALLS_X = (-6.77, -5.64); WALLS_Y = (6.285, 7.495)
ON_SURFACE = .015                   # a point this near to a bag's upper surface counts for that bag
SLAB = (.300, .222, .089)           # the briefcase: its broad face's long and short side, and its thickness
SLAB_SHOWS = (.285, .170)           # ... and how much of each side shows at least when none of it is hidden (one face is domed across: 0.19 of it lies in a plane)
BODY = dict(satchel=np.zeros(3), briefcase=np.array([.0005, -.001, -.0065]))     # where a bag's frame (see Bags.see) lies in its asset's frame
FLAT_SHARE = .60                    # a briefcase that matches its shape lying flat with at least this share is taken as lying flat
# (0.73-0.86 on the frames where it does; one that lay against the trunk's wall, partly out of view and with a side on
# the wheel arch's step, matched with 0.48 and 4.6 cm off, and the hand missed it twice: the plane of its face has it to 1 cm.)
# The slab's faces: (name, its two sides, how far the slab goes on behind it, how much of each side shows at least when
# none of it is hidden).
FACES = (('broad', SLAB[:2], SLAB[2], SLAB_SHOWS), ('long side', (SLAB[0], SLAB[2]), SLAB[1], (SLAB_SHOWS[0], .075)), ('short side', (SLAB[1], SLAB[2]), SLAB[0], (.200, .075)))
SLAB_SEEN = .45                     # least share of the points round the slab that must lie on its surface
SHARE = dict(satchel=.30, briefcase=.25)        # least share of the points given to a match that must lie on the bag's surface
# (On the recorded frames: satchel 0.37-0.61; briefcase lying flat 0.50-0.86, upside down 0.31; a briefcase that
# leans on the satchel matched somewhere with 0.07-0.19.)


def floor_height(x):
    """What a bag at map x lies on: the trunk's floor, or the sill."""
    x = np.asarray(x, float)
    return np.where(x < SILL_FROM, SILL_TOP, FLOOR_AT_SILL + FLOOR_SLOPE * (x - SILL_X))


class Bags:
    def __init__(self, folder):
        data = np.load(Path(folder) / 'priors' / 'bags.npz'); meta = json.loads(str(data['meta'])); self.cell = float(meta['cell']); self.models = {}
        for name in ('briefcase', 'satchel'):
            top = data[name + '_top'].astype(float); origin = data[name + '_origin'].astype(float); bottom = float(data[name + '_bottom'])
            surface = data[name + '_surface'].astype(float); lo, hi = surface[:, :2].min(0), surface[:, :2].max(0)
            # rest: height of the bag's origin above what it lies on; corners: of its outline seen from above (its own x, y)
            self.models[name] = dict(top=top, origin=origin, rest=-bottom, surface=surface, corners=surface[ConvexHull(surface[:, :2]).vertices][:, :2])

    def obstacle(self, name, origin, turn, cell=.04, radius=.022):
        """Spheres (centres N x 3, radii N) that stand for bag @name whose frame is at @origin, @turn (see known()):
        points of its surface, thinned to one per @cell, each with @radius."""
        model = self.models[name]
        if 'thin' not in model:
            surface = model['surface'] - BODY[name]; keys = np.floor(surface / cell).astype(np.int64); _, first = np.unique(keys, axis=0, return_index=True); model['thin'] = surface[first]
        points = model['thin'] @ np.asarray(turn, float).T + np.asarray(origin, float)
        return points, np.full(len(points), radius)

    def in_trunk(self, points):
        """Of map-frame @points (N, 3): those inside the trunk that lie above its floor."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        keep = (x > TRUNK_X[0]) & (x < TRUNK_X[1]) & (y > TRUNK_Y[0]) & (y < TRUNK_Y[1]) & (z < TRUNK_TOP) & (z - floor_height(x) > ABOVE_FLOOR)
        return points[keep]

    def _count(self, name, points, x0, y0, yaw):
        """For hypotheses of a bag lying flat with its origin over (x0, y0) (arrays K) turned by @yaw (one angle):
        how many of @points lie on its upper surface, and how many lie above it, per hypothesis."""
        model = self.models[name]; top = model['top']; c, s = np.cos(yaw), np.sin(yaw)
        dx = points[None, :, 0] - x0[:, None]; dy = points[None, :, 1] - y0[:, None]
        u = c * dx + s * dy; v = -s * dx + c * dy
        i = np.floor((u - model['origin'][0]) / self.cell).astype(int); j = np.floor((v - model['origin'][1]) / self.cell).astype(int)
        inside = (i >= 0) & (i < top.shape[0]) & (j >= 0) & (j < top.shape[1])
        height = np.where(inside, top[np.clip(i, 0, top.shape[0] - 1), np.clip(j, 0, top.shape[1] - 1)], np.nan)
        expected = floor_height(points[None, :, 0]) + model['rest'] + height
        residual = points[None, :, 2] - expected
        on = (np.abs(residual) < ON_SURFACE).sum(axis=1); above = (residual > 2 * ON_SURFACE).sum(axis=1)
        # A bag does not reach through the trunk's walls: such places and headings are out.
        corners = model['corners']; cx = x0[:, None] + c * corners[None, :, 0] - s * corners[None, :, 1]; cy = y0[:, None] + s * corners[None, :, 0] + c * corners[None, :, 1]
        inside_walls = (cx.min(axis=1) > WALLS_X[0]) & (cx.max(axis=1) < WALLS_X[1]) & (cy.min(axis=1) > WALLS_Y[0]) & (cy.max(axis=1) < WALLS_Y[1])
        return np.where(inside_walls, on, 0), np.where(inside_walls, above, 10 ** 6)

    def fits(self, name, points, guess, reach=.14, limit=1500, rng=None, count=1):
        """Places and headings of bag @name lying flat that explain most of @points (map frame, already cut to the
        trunk), the best first, up to @count of them that differ (by 30 degrees or 6 cm): each dict(place=(x, y, z of
        its origin, yaw), on (points on its surface), share, above). guess: (x, y) to search round, within @reach."""
        if len(points) < 40:
            return []
        rng = rng or np.random.default_rng(0)
        sample = points if len(points) <= limit else points[rng.choice(len(points), limit, replace=False)]; scale = len(points) / len(sample)
        offsets = np.arange(-reach, reach + 1e-9, .02); gx, gy = np.meshgrid(guess[0] + offsets, guess[1] + offsets, indexing='ij'); gx, gy = gx.ravel(), gy.ravel(); coarse = []
        for yaw in np.arange(-np.pi, np.pi, np.radians(6)):
            on, above = self._count(name, sample, gx, gy, yaw); value = on - .5 * above; k = int(np.argmax(value))
            coarse.append((float(value[k]), float(gx[k]), float(gy[k]), float(yaw), int(on[k]), int(above[k])))
        chosen = []
        for row in sorted(coarse, reverse=True):
            if all(abs((row[3] - other[3] + np.pi) % (2 * np.pi) - np.pi) > np.radians(30) or np.hypot(row[1] - other[1], row[2] - other[2]) > .06 for other in chosen):
                chosen.append(row)
            if len(chosen) == count:
                break
        out = []
        for best in chosen:
            offsets = np.arange(-.02, .02 + 1e-9, .005); gx, gy = np.meshgrid(best[1] + offsets, best[2] + offsets, indexing='ij'); gx, gy = gx.ravel(), gy.ravel()
            for yaw in best[3] + np.arange(-6, 6.01, 1) * np.radians(1):
                on, above = self._count(name, sample, gx, gy, yaw); value = on - .5 * above; k = int(np.argmax(value))
                if value[k] > best[0]:
                    best = (float(value[k]), float(gx[k]), float(gy[k]), float(yaw), int(on[k]), int(above[k]))
            value, x, y, yaw, on, above = best
            # A bag that lies flat and in view has half of the points round it on its surface; a match with a fifth of
            # them is something else (a bag that leans, the other bag, a hand).
            if on >= 60 and on / len(sample) >= SHARE[name]:
                out.append(dict(place=(x, y, float(floor_height(x)) + self.models[name]['rest'], float((yaw + np.pi) % (2 * np.pi) - np.pi)), on=int(on * scale), above=int(above * scale), share=on / len(sample)))
        return out

    def fit(self, name, points, guess, **options):
        """The best of fits(), or None."""
        found = self.fits(name, points, guess, **options)
        return found[0] if found else None

    def explained(self, name, points, place, slack=.03):
        """Which of @points belong to bag @name lying at @place: inside its outline and not above its surface."""
        model = self.models[name]; top = model['top']; x0, y0, _, yaw = place; c, s = np.cos(yaw), np.sin(yaw)
        dx = points[:, 0] - x0; dy = points[:, 1] - y0; u = c * dx + s * dy; v = -s * dx + c * dy
        i = np.floor((u - model['origin'][0]) / self.cell).astype(int); j = np.floor((v - model['origin'][1]) / self.cell).astype(int); pad = int(round(slack / self.cell))
        inside = (i >= -pad) & (i < top.shape[0] + pad) & (j >= -pad) & (j < top.shape[1] + pad)
        height = np.nan_to_num(top[np.clip(i, 0, top.shape[0] - 1), np.clip(j, 0, top.shape[1] - 1)], nan=float(np.nanmax(top)))
        return inside & (points[:, 2] < floor_height(points[:, 0]) + model['rest'] + height + slack)

    @staticmethod
    def _one_face(face, reach=.23):
        """Of points in one plane, those that can belong to one face of the briefcase: within its half diagonal (and
        4 cm) of their middle. A few points elsewhere in the same plane (the rim of the other bag, a wall) made the
        outline too long for a briefcase, and it was not found in 3 of 4 tries on a recorded frame."""
        middle = np.median(face, axis=0)
        for _ in range(2):
            kept = face[np.linalg.norm(face - middle, axis=1) < reach]
            if len(kept) < 3:
                return face
            middle = np.median(kept, axis=0)
        return kept

    def carried_at(self, xy, satchel=None):
        """Height (map z) of what carries a thing over map points @xy (N, 2): the trunk's floor or the sill, or the
        satchel's upper surface where it lies (@satchel: its place, or None)."""
        xy = np.asarray(xy, float); height = np.asarray(floor_height(xy[:, 0]), float).copy()
        if satchel is not None:
            model = self.models['satchel']; top = model['top']; x0, y0, _, yaw = satchel; c, s = np.cos(yaw), np.sin(yaw)
            dx = xy[:, 0] - x0; dy = xy[:, 1] - y0; u = c * dx + s * dy; v = -s * dx + c * dy
            i = np.floor((u - model['origin'][0]) / self.cell).astype(int); j = np.floor((v - model['origin'][1]) / self.cell).astype(int)
            inside = (i >= 0) & (i < top.shape[0]) & (j >= 0) & (j < top.shape[1])
            over = np.where(inside, top[np.clip(i, 0, top.shape[0] - 1), np.clip(j, 0, top.shape[1] - 1)], np.nan)
            height = np.where(np.isnan(over), height, height + model['rest'] + np.nan_to_num(over))
        return height

    def fit_slab(self, points, rng=None, near=.006, satchel=None):
        """The briefcase as a slab, however it lies or stands, from map-frame @points (what is left above the trunk's
        floor when the satchel's points are taken out). The plane with most points on it is one of the slab's faces:
        a broad one (SLAB[0] x SLAB[1]) or, of a briefcase that stands on a side, a narrow one. Every way of laying
        the slab against that plane that the plane's outline allows is held against all the points round it: those
        on the slab's surface count for it, those that would lie inside it against it. Returns dict(centre (of the
        slab, xyz), normal (of a broad face: the one that looks up or, of faces that stand upright, at the robot),
        long, short (unit vectors along the broad face's sides), tilt (of the broad face from level, rad), on
        (points on the slab's surface), seen (their share of the points round the slab), face (which face the plane
        was)) or None. satchel: where the satchel lies, if it does: the slab may rest on it, and not reach into it.
        (A briefcase that lay on the satchel was taken for one that stood on the floor beside it, 10 to 15 cm off:
        the slab's lowest corner was held against the floor alone.)"""
        if len(points) < 60:
            return None
        rng = rng or np.random.default_rng(0); best = None
        for _ in range(150):
            a, b, c = points[rng.choice(len(points), 3, replace=False)]; normal = np.cross(b - a, c - a); size = float(np.linalg.norm(normal))
            if size < 1e-6:
                continue
            normal = normal / size; count = int((np.abs((points - a) @ normal) < near).sum())
            if best is None or count > best[0]:
                best = (count, normal, a)
        count, normal, origin = best
        if count < 60:
            return None
        face = self._one_face(points[np.abs((points - origin) @ normal) < near])
        centre = face.mean(axis=0); _, _, vt = np.linalg.svd(face - centre); normal = vt[2]
        if normal[2] < 0 or (abs(normal[2]) < .1 and normal[0] > 0):
            normal = -normal                             # out of the slab: upwards, or towards the camera for a face that stands upright
        face = self._one_face(points[np.abs((points - centre) @ normal) < near])
        level = np.cross([0., 0., 1.], normal); level = level / np.linalg.norm(level) if np.linalg.norm(level) > .05 else np.array([1., 0., 0.]); up = np.cross(normal, level)
        a = (face - centre) @ level; b = (face - centre) @ up
        round_it = points[np.linalg.norm(points - np.median(face, axis=0), axis=1) < .36]      # (the slab's diagonal is 0.38)
        # For each face the plane may be, and each way round: the heading at which the outline of what shows is smallest.
        outlines = {}
        for angle in np.radians(np.arange(0., 180., 3.)):
            c, s = np.cos(angle), np.sin(angle); u = c * a + s * b; v = -s * a + c * b
            lo = np.array([np.percentile(u, 1), np.percentile(v, 1)]); hi = np.array([np.percentile(u, 99), np.percentile(v, 99)]); extent = hi - lo
            for number, (kind, sides, depth, shows) in enumerate(FACES):
                for swap in (False, True):
                    size = np.array(sides[::-1] if swap else sides)
                    # (What shows of a narrow face is up to 4 cm wider than the face: points of the faces beside it lie in its plane too.)
                    if np.all(extent <= size + np.where(size < .1, .04, .025)) and ((number, swap) not in outlines or extent[0] * extent[1] < outlines[number, swap][0]):
                        outlines[number, swap] = (float(extent[0] * extent[1]), angle, lo, hi)
        found = None; corner_signs = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
        for (number, swap), (area, angle, lo, hi) in outlines.items():
            kind, sides, depth, shows = FACES[number]; size = np.array(sides[::-1] if swap else sides); least = np.array(shows[::-1] if swap else shows); extent = hi - lo
            c, s = np.cos(angle), np.sin(angle); axis_u = c * level + s * up; axis_v = -s * level + c * up; axes = np.column_stack([axis_u, axis_v, normal])
            # A side that shows shorter than it is has an end out of view (beyond the strip the camera's points are cut
            # to along the trunk's walls and above its floor, or behind the other bag): the outline lies against one
            # end of what shows or against the other.
            places = [[.5 * (lo[k] + hi[k])] if extent[k] >= least[k] else [lo[k] + size[k] / 2, hi[k] - size[k] / 2] for k in (0, 1)]
            for mu in places[0]:
                for mv in places[1]:
                    middle = centre + axis_u * mu + axis_v * mv - normal * depth / 2; half = np.r_[size, depth] / 2
                    out_of = (np.abs((round_it - middle) @ axes) - half).max(axis=1)
                    on = int((np.abs(out_of) < .010).sum()); inside = int((out_of < -.02).sum()); beside = int((out_of >= .010).sum())
                    corners = middle + (corner_signs * half) @ axes.T
                    if corners[:, 1].min() < WALLS_Y[0] - .02 or corners[:, 1].max() > WALLS_Y[1] + .02:
                        continue                         # (through a side wall; over the sill it may reach)
                    lowest = float((corners[:, 2] - self.carried_at(corners[:, :2], satchel)).min())
                    # (An end that is out of view is the one nearer to a wall: there the points are cut off.)
                    room = sum(min(end[1] - TRUNK_Y[0], TRUNK_Y[1] - end[1], TRUNK_X[1] - end[0]) for k, (m, axis) in enumerate(((mu, axis_u), (mv, axis_v))) if extent[k] < least[k]
                               for end in [centre + axis_u * mu + axis_v * mv + axis * (size[k] / 2 if m > .5 * (lo[k] + hi[k]) else -size[k] / 2)])
                    value = (on - 2. * inside - .3 * beside) / len(round_it) - 4. * min(max(abs(lowest) - .02, 0.), .05) - .2 * room
                    if found is None or value > found[0]:
                        found = (value, kind, swap, axis_u, axis_v, middle, on, on / len(round_it), lowest)
        if found is None:
            return None
        value, kind, swap, axis_u, axis_v, middle, on, share, lowest = found
        first, second = (axis_v, axis_u) if swap else (axis_u, axis_v)                  # along the face's sides as FACES names them
        if kind == 'broad':
            long, short, broad = first, second, normal
        elif kind == 'long side':
            long, broad, short = first, second, normal
        else:
            short, broad, long = first, second, normal
        if broad[2] < -.1 or (abs(broad[2]) <= .1 and (broad[0] > .1 or (abs(broad[0]) <= .1 and (broad[1] > 0) != (middle[1] < .5 * (TRUNK_Y[0] + TRUNK_Y[1]))))):
            broad = -broad                               # up; of upright faces the one that looks at the robot, or else at the trunk's middle
        return dict(centre=middle, normal=broad, long=long, short=short, tilt=float(np.arccos(np.clip(broad[2], -1., 1.))), on=on, seen=share, face=kind, lowest=lowest)

    def see(self, points):
        """Both bags from map-frame @points of one look into the trunk: ({name: what is known of it, or None}, how
        many points lie above the trunk's floor). Known of a bag: dict(origin, turn: where its frame is (xyz, 3x3;
        the satchel's frame is its asset's, the briefcase's lies in the middle of its body with z out of the broad
        face that looks up), how ('shape': matched lying flat; 'slab': the briefcase from the plane of its broad
        face, however it lies), tilt (rad), on, share)."""
        things = self.in_trunk(points); out = dict(satchel=None, briefcase=None)
        if len(things) < 40:
            return out, len(things)
        rise = things[:, 2] - floor_height(things[:, 0])
        tall = things[rise > .15]                       # only the satchel is that high when both lie flat
        # The satchel may match in more than one way when the briefcase leans on it (start 103: turned by 60 degrees
        # and 12 cm off, with as large a share of the points as where it was): each way is taken with the briefcase
        # that what it leaves over shows, and the pair that explains most points is the one.
        ways = self.fits('satchel', things, np.median(tall[:, :2], axis=0), count=3) if len(tall) >= 30 else []
        best = None
        for fit in ways + [None]:
            rest = things if fit is None else things[~self.explained('satchel', things, fit['place'])]
            briefcase = self._briefcase(rest, None if fit is None else fit['place']); total = (0 if fit is None else fit['on']) + (0 if briefcase is None else briefcase['on'])
            if best is None or total > best[0]:
                best = (total, fit, briefcase)
        total, fit, briefcase = best
        if fit is not None:
            x, y, z, yaw = fit['place']; out['satchel'] = dict(origin=np.array([x, y, z]), turn=turn_z(yaw), how='shape', tilt=0., on=fit['on'], share=fit['share'])
        out['briefcase'] = briefcase
        return out, len(things)

    def _briefcase(self, rest, satchel=None):
        """The briefcase from the points that the satchel (lying at @satchel, if it is there) does not explain: what is
        known of it (see see()) or None."""
        low = rest[(rest[:, 2] - floor_height(rest[:, 0]) > .04)]
        if len(low) < 40:
            return None
        fit = self.fit('briefcase', rest, np.median(low[:, :2], axis=0))
        if fit is not None and fit['share'] >= FLAT_SHARE:
            x, y, z, yaw = fit['place']; turn = turn_z(yaw)
            return dict(origin=np.array([x, y, z]) + turn @ BODY['briefcase'], turn=turn, how='shape', tilt=0., on=fit['on'], share=fit['share'])
        # Not lying flat the right way up: from the plane of a face that shows.
        slab = self.fit_slab(rest, satchel=satchel)
        if slab is None or slab['seen'] < SLAB_SEEN:
            return None
        short = slab['short'] if np.dot(np.cross(slab['short'], slab['long']), slab['normal']) > 0 else -slab['short']
        return dict(origin=slab['centre'], turn=np.column_stack([short, slab['long'], slab['normal']]), how='slab', tilt=slab['tilt'], on=slab['on'], share=slab['seen'], face=slab['face'])


def turn_z(angle):
    c, s = np.cos(angle), np.sin(angle); return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
