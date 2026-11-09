"""Putting the pruning kit together from what the robot's own cameras show: the estimator and the entry point.

The base pose and the per-step runtime are those of the controllers for the car tasks (car_observed.py). What is
new here is what is known of the three things on the garage floor (kit_view.py): where the toolbox, the pruner and
the shears are, which way they are turned, and which long side of the toolbox is its front.
"""
import numpy as np
from manip import TALL_TRUNK
from car_observed import ObservedEstimator, localize
from observer import HEAD
from kit_view import KitView

NAMES = ('toolbox', 'pruner', 'shears')
NEAR = 1.6                          # a look from nearer than this replaces what was known from further away
MOVED = dict(toolbox=.06, pruner=.04, shears=.04)       # a thing seen this far from where it was known has been moved: the new look counts


class KitEstimator(ObservedEstimator):
    def __init__(self, observer, scene, folder):
        super().__init__(observer, scene)
        self.view = KitView(scene); self.things = {}; self.gone = set(); self.looks = 0
        self.phantoms = []          # places where something was taken for the toolbox that is not one
        self.switched = None
        self.front = None           # map direction (rad) out of the toolbox's front, once a look has told

    # (Nothing of the living room is looked for in this task.)
    def see(self, what=()):
        return {}

    def predict(self, s, last_action, dt=1 / 30):
        super().predict(s, last_action, dt)
        # (The localiser has taken another start pose for the right one: said in the trace.)
        switched = self.ob.loc.debug.get('switched_to') if getattr(self.ob.loc, 'debug', None) else None
        if switched is not None and switched != self.switched:
            self.news.append(dict(start_pose_switched_to=switched))
        self.switched = switched
        # On the way, with the trunk upright and not in a fast turn, every sixth frame is looked at for the things.
        if self.ob.loc.pose is not None and self.ob.steps % 6 == 0 and abs(float(last_action[2])) < .3 and float(np.max(np.abs(np.asarray(s, float)[53:57] - TALL_TRUNK))) < .06:
            self.look()

    def where(self, name):
        known = self.things.get(name)
        return None if known is None else np.asarray(known['xy'], float)

    def look(self):
        """Take what the head camera shows of the three things. Returns {name: what this look showed or None}."""
        base = self.base(); box = self.things.get('toolbox')
        # (the toolbox where a look from near has it whole is set apart from what lies beside it)
        seen = self.view.see(self.ob, base, box=box if box is not None and box['whole'] and box['range'] < 2.5 else None); self.looks += 1
        for name, new in seen.items():
            if new is None or name in self.gone or (name == 'toolbox' and any(float(np.hypot(*(new['xy'] - at))) < .6 for at in self.phantoms)):
                continue
            new = dict(new, range=float(np.hypot(new['xy'][0] - base[0], new['xy'][1] - base[1])), step=self.ob.steps); old = self.things.get(name)
            if old is None:
                better = True
            elif float(np.hypot(*(new['xy'] - old['xy']))) <= MOVED[name]:
                better = new['range'] < max(old['range'], NEAR)
            else:
                # Somewhere else than it was known: taken only if the place it was known at is in view now (and it is
                # not there), or if it was known from a look from far off. (A piece of the toolbox was taken for the
                # pruner once, 3 m from where the pruner stood, and the hand went for it eight times.)
                better = self.in_view(old['xy'], base, nearest=.86) or (old['range'] > 3. and new['range'] < 2.)
            if name == 'toolbox' and old is not None and old['whole'] and old['range'] < 2. and not new['whole'] and float(np.hypot(*(new['xy'] - old['xy']))) < .15:
                better = False                              # (its lid's plate cut off by the image's border: the whole view stays)
            if not better:
                continue
            # What this look could not tell (the pruner's blades and the way they lean, which end of the shears) is kept
            # from the look before; the pruner's place with it, which is on its handles only when the way is told.
            if old is not None and float(np.hypot(*(new['xy'] - old['xy']))) < MOVED[name] and new.get('lying') == old.get('lying'):
                if name == 'pruner' and not new.get('lying') and new.get('way') is None and old.get('way') is not None:
                    continue
                if new.get('axis') is None and old.get('axis') is not None:
                    new['axis'] = old['axis']
            self.things[name] = new
            if old is None:
                self.news.append(dict(seen=name, xy=np.round(new['xy'], 3).tolist(), range=round(new['range'], 2)))
        return seen

    @staticmethod
    def in_view(xy, base, nearest=.95):
        """Does the head camera show the floor at map @xy from base pose @base? (It shows the floor from 0.86 m ahead of
        the base, over 99 degrees.)"""
        x, y, yaw = base; c, s = np.cos(yaw), np.sin(yaw); d = np.asarray(xy, float) - [x, y]; ahead = c * d[0] + s * d[1]; aside = -s * d[0] + c * d[1]
        return bool(nearest < ahead < 3.0 and abs(np.arctan2(aside, ahead - .24)) < np.radians(40.))

    def look_front(self):
        """Which long side of the toolbox is its front, from this look's colour image: sets self.front (map direction
        out of the front) when the look tells. Returns (+1, -1 or 0 as KitView.front, what it saw)."""
        box = self.things.get('toolbox')
        if box is None:
            return 0, {}
        if not box['whole']:                # (where the box stands is known to a hand's breadth only: the windows on its face would be elsewhere)
            return 0, dict(whole=False, long=round(float(box['long']), 2), short=round(float(box['short']), 2))
        sign, told = self.view.front(self.ob, self.ob.data[HEAD + 'rgb'][..., :3], self.base(), box)
        if sign:
            self.front = float(box['yaw'] if sign > 0 else box['yaw'] + np.pi)
        return sign, told

    def box_place(self):
        """(x, y, yaw) of the toolbox in M, yaw out of its front; None while the box or its front is not known."""
        box = self.things.get('toolbox')
        if box is None or self.front is None:
            return None
        # the box's long axis as the latest look has it, the front on the side an earlier look told
        yaw = box['yaw'] if np.cos(box['yaw'] - self.front) > 0 else box['yaw'] + np.pi
        return float(box['xy'][0]), float(box['xy'][1]), float((yaw + np.pi) % (2 * np.pi) - np.pi)

    def forget(self, name):
        self.things.pop(name, None)

    def disown(self, name):
        """What was taken for the toolbox is not one (looks from near do not show its lid whole): nothing there is
        taken for it again."""
        known = self.things.pop(name, None)
        if known is not None:
            self.phantoms.append(np.asarray(known['xy'], float).copy())

    def moved(self, name):
        """A hand has been on the thing and it may stand somewhere else now: the next look that shows it counts, and it
        is looked for afresh, not where it was. (A box that came up with the hand on its handle was set down 7 cm from
        where it had stood; the look after that, with the box set apart where it was known, moved it 5 mm, and the
        hand closed three times on nothing.)"""
        known = self.things.get(name)
        if known is not None:
            known['range'] = 9.

    def taken(self, name):
        """The thing is in a hand or in the box: it is not looked for any more."""
        self.gone.add(name); self.things.pop(name, None)


def observed_pipeline(rt):
    from kit_pipeline import full_task
    located = yield from localize(rt)
    if not located:
        return 'not_localized'
    return (yield from full_task(rt.ctx, rt.est))
