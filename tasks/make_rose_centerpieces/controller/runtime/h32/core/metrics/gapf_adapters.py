"""Distance adapters of GAPF by predicate type (not by task): d >= 0 is the violation of a NECESSARY condition of the
native OmniGibson predicate, computed from world AABBs of the collision meshes (core.env.predicates.aabb). d = 0 does not
mean the predicate holds (the ray and contact parts are not measured); the goal atom itself always decides.

  nextto(a, b)    NextTo needs the axis-aligned gap norm <= (sum of the six AABB extents of a and b) / 3 / 6
                  (core.env.predicates.next_to): d = max(0, gap norm - that threshold)
  inside(a, b)    Inside needs the centre of a's AABB inside b's AABB: d = distance of that centre to b's AABB
  ontop(a, b)     OnTop needs a resting on b from above: d = hypot(distance of a's xy centre to b's xy footprint,
                  |a.min_z - b.max_z|)
  touching(a, b)  Touching needs the AABBs to meet: d = axis-aligned gap norm
Atoms whose objects have no box, negated atoms and other predicates get no distance (no shaping), never a guess.
A box is (lo, hi), two length-3 arrays.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence, Tuple

import numpy as np

from .gapf import Atom, GoalSpec

Box = Tuple[np.ndarray, np.ndarray]
NEXT_TO_RATIO = 1.0 / 6.0  # core.env.predicates.NEXT_TO_RATIO (native NextTo)


def _gap(a: Box, b: Box) -> np.ndarray:
    (lo_a, hi_a), (lo_b, hi_b) = a, b
    return np.maximum(0.0, np.maximum(lo_a, lo_b) - np.minimum(hi_a, hi_b))


def _point_to_box(p: np.ndarray, box: Box) -> float:
    lo, hi = box
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))))


def nextto(a: Box, b: Box) -> float:
    thr = float(np.sum(a[1] - a[0]) + np.sum(b[1] - b[0])) / 3.0 * NEXT_TO_RATIO
    return max(0.0, float(np.linalg.norm(_gap(a, b))) - thr)


def inside(a: Box, b: Box) -> float:
    return _point_to_box((a[0] + a[1]) / 2.0, b)


def ontop(a: Box, b: Box) -> float:
    c = (a[0] + a[1]) / 2.0
    xy = _point_to_box(c[:2], (b[0][:2], b[1][:2]))
    return float(np.hypot(xy, abs(float(a[0][2]) - float(b[1][2]))))


def touching(a: Box, b: Box) -> float:
    return float(np.linalg.norm(_gap(a, b)))


REGISTRY: Dict[str, Callable[[Box, Box], float]] = {"nextto": nextto, "inside": inside, "ontop": ontop, "touching": touching}


def distances(spec: GoalSpec, boxes: Mapping[str, Box]) -> Dict[str, float]:
    """{atom key: d} for every positive binary atom of the spec with an adapter and a box for both objects."""
    out = {}
    for a in spec.atoms():
        f = REGISTRY.get(a.pred)
        if f is None or a.negated or len(a.args) != 2 or not all(x in boxes for x in a.args):
            continue
        bx = [tuple(np.asarray(v, np.float64) for v in boxes[x]) for x in a.args]
        out[a.key] = f(bx[0], bx[1])
    return out


def literals(atoms: Sequence[Atom], truth: Mapping[str, bool]) -> Dict[str, bool]:
    """Milestone literals the harness can evaluate: a negated atom is the negation of its positive form."""
    out = {}
    for a in atoms:
        pos = Atom(a.pred, a.args).key
        if pos in truth and truth[pos] is not None:
            out[a.key] = (not truth[pos]) if a.negated else bool(truth[pos])
    return out
