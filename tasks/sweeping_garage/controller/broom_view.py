"""The broom on the garage floor, from the head camera's depth image.

The broom is 1.36 m long: a head 0.45 m wide, 0.30 m deep and 0.16 m high (its bristles), and a thin handle that runs
from the head's top down to the floor at its end. Points above the free floor are grouped as for the pruning kit's
things (kit_view.py); the broom is the group that is long (1.0 to 1.6 m along its main direction) and low (under 0.2 m).
Its head is the end where the points stand highest.

Everything returned is in the map frame M.
"""
import numpy as np
from kit_view import KitView

BROOM = dict(long=(.45, 1.6), top=(.10, .21), head_depth=.30)        # (from 2.8 m the handle's low end did not show: the group was the head and half the handle)
GRIP_ALONG = 1.00                   # where the hand takes the handle: this far from the head's middle towards the handle's end


class BroomView(KitView):
    def broom(self, ob, base_pose, stride=1):
        """The broom as this look shows it: dict(head (xy of the head's middle), direction (unit xy from the head
        towards the handle's end), length, grip (xy), grip_z (the handle's height there), n, cut, range) or None."""
        best = None
        groups = self.groups(ob, base_pose, stride)
        # (from 3 m the head and the handle show as two groups a few centimetres apart: the head 0.15 high and 0.44
        # long, the handle 0.10 high and 1.0 long; such pairs are taken together)
        merged = []
        for h in groups:
            if not .12 < h['top'] < .21 or h['n'] < 20:
                continue
            for g in groups:
                if g is h or g['top'] > .12 or g['n'] < 10:
                    continue
                gap = float(np.min(np.linalg.norm(h['points'][::3, None, :2] - g['points'][None, ::3, :2], axis=2)))
                if gap < .20:
                    merged.append(dict(points=np.vstack([h['points'], g['points']]), top=h['top'], n=h['n'] + g['n'], cut=h['cut'] or g['cut']))
        for g in groups + merged:
            pts = g['points']
            if g['n'] < 30 or not BROOM['top'][0] < g['top'] < BROOM['top'][1]:
                continue
            c = pts[:, :2].mean(axis=0); d = pts[:, :2] - c
            axis = np.linalg.svd(d, full_matrices=False)[2][0]; a = d @ axis
            lo, hi = np.percentile(a, 1), np.percentile(a, 99); length = float(hi - lo)
            if not BROOM['long'][0] < length < BROOM['long'][1]:
                continue
            # the head's end: where the points stand highest
            low_end = pts[a < lo + .3, 2]; high_end = pts[a > hi - .3, 2]
            if len(low_end) < 5 or len(high_end) < 5:
                continue
            if np.percentile(high_end, 90) < np.percentile(low_end, 90):
                axis = -axis; a = -a; lo, hi = -hi, -lo
            # (now the head is at the +axis end)
            head = c + (hi - BROOM['head_depth'] / 2) * axis
            direction = -axis
            # (the main direction of all the points leans towards the head's 0.44 m across the handle: seen from its
            # side after a put-down it was 15-26 degrees off, and the hand went 0.3 m wide of the handle, 302, 317. The
            # handle's own points, 0.3 m and more from the head's end, give its line; the head's points its middle)
            handle = pts[a < hi - .30, :2]; heads = pts[a > hi - .25, :2]
            if len(handle) >= 15 and len(heads) >= 10:
                hc = handle.mean(axis=0); hax = np.linalg.svd(handle - hc, full_matrices=False)[2][0]
                if hax @ direction < 0:
                    hax = -hax
                if float(np.ptp((handle - hc) @ hax)) > .3 and float(np.degrees(np.arccos(np.clip(hax @ direction, -1, 1)))) < 35:
                    middle = heads.mean(axis=0)
                    # (the head's middle on the handle's line, at the head's own middle along it)
                    head = hc + float((middle - hc) @ hax) * hax
                    direction = hax
            grip = head + GRIP_ALONG * direction
            near = np.linalg.norm(pts[:, :2] - grip, axis=1) < .06
            grip_z = float(np.median(pts[near, 2])) if near.sum() >= 3 else None
            rng = float(np.linalg.norm(head - np.asarray(base_pose[:2], float)))
            thing = dict(head=head, direction=direction, length=length, grip=grip, grip_z=grip_z, n=g['n'], cut=g['cut'], range=rng)
            if best is None or g['n'] > best['n']:
                best = thing
        return best
