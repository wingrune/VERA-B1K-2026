"""Goal-Aligned Progress Fitness (GAPF v1.0): one fitness for every BEHAVIOR-1K task, from the BDDL goal.

The fitness of an episode on one consistent goal option b (a grounded alternative of the BDDL goal):
    F_b = 0.60 * Q_b + 0.30 * M_b + 0.10 * D_b - 0.05 * C
    Q_b  newly satisfied goal atoms of b / atoms of b (official BEHAVIOR Q: true at the end and false at reset)
    M_b  0.70 * G_b + 0.30 * P_b, or G_b when b has no milestones
         G_b = clip((newly satisfied - lost initially true) / atoms of b, 0, 1)
         P_b = weight of the milestones reached / weight of the milestones of b that were not already true at reset
    D_b  mean over the positive atoms of b that were false at reset and have a distance adapter:
         clip((d(0) - d(T)) / max(d(0), EPS_M), 0, 1) (a gain under DEADBAND_M counts 0), 1 when the atom holds at the end
    C    constraint penalty in [0, 1], fixed severities, each kind counted once
F = 1.0 on official success, otherwise clip(max_b F_b, 0, 0.99): any success ranks above any failure. All components of
F_b come from the same option (no mixing of OR branches). The weights are constants of the evaluator version, equal for
all tasks and fixed during an evolution run; the planner never sees or changes them.

Inputs per step come from the harness (privileged state, never from the planner):
    option_truth   [[bool]] in the order of spec.options (what get_goal_option_satisfaction returns)
    distances      {atom key: d >= 0}: violation of a necessary condition of the atom (gapf_adapters), only for atoms
                   whose objects the harness can measure; a missing key means no shaping for that atom
    literals       {atom key: bool | None}: truth of the literals of state milestones (None = cannot evaluate)
    events         {milestone id: bool}: event milestones the env measures itself (e.g. the toggle marker engaged)
    constraints    kinds of constraint events seen at this step (CONSTRAINT_SEVERITY)
Coverage level: 1 goal only, 2 with a distance adapter on some atom, 3 with validated milestones. Unsupported atoms are
listed, never filled with a guess.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

EVALUATOR_VERSION = "gapf-1.0"
W_Q, W_M, W_D, W_C = 0.60, 0.30, 0.10, 0.05
W_G, W_P = 0.70, 0.30
FAIL_CAP = 0.99
EPS_M = 0.05                 # metres, floor of the D normaliser (numerical stability for atoms that start almost satisfied)
DEADBAND_M = 0.005           # a distance gain below this is physics jitter of a still scene, not progress (null policy on H100: 1e-6)
SE_PENALTY = 0.5             # selection score = mean fitness - SE_PENALTY * standard error (internal ranking only)
CONSTRAINT_SEVERITY = {      # fixed before a run; the planner can not change them
    "goal_object_lost": 1.0,     # an object of the goal left the reachable scene (fell through the floor, out of bounds)
    "safety": 0.5,               # an explicit safety constraint of the evaluator was violated
}
# Predicates that are binary by nature in BEHAVIOR (no geometric distance to the satisfying set); they get no D.
BINARY_PREDICATES = frozenset({"toggled_on", "real", "future", "cooked", "frozen", "burnt", "folded", "unfolded", "sliced",
                               "diced", "hot", "on_fire", "saturated", "covered", "filled", "contains", "empty", "broken",
                               "assembled", "attached", "draped", "overlaid", "insource", "inroom", "open"})

_POS = re.compile(r"^\(\s*([A-Za-z_][\w-]*)((?:\s+[^()\s]+)*)\s*\)$")
_NEG = re.compile(r"^\(\s*not\s+(\(.*\))\s*\)$")


@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[str, ...]
    negated: bool = False

    @property
    def key(self) -> str:
        body = "(" + " ".join((self.pred,) + self.args) + ")"
        return f"(not {body})" if self.negated else body

    @classmethod
    def parse(cls, text: str) -> "Atom":
        """'(nextto a b)' or '(not (open c))', the string form of a grounded BDDL atom (bddl HEAD body)."""
        text = text.strip()
        neg = _NEG.match(text)
        m = _POS.match(neg.group(1).strip() if neg else text)
        if not m or m.group(1) == "not":
            raise ValueError(f"not a grounded BDDL atom: {text!r}")
        return cls(m.group(1), tuple(m.group(2).split()), bool(neg))


@dataclass
class GoalSpec:
    """Grounded goal options of a task instance. complete=False when the goal could not be expanded into a finite set of
    options (the evaluator then reports it instead of pretending a full evaluation)."""
    options: List[List[Atom]]
    complete: bool = True
    source: str = ""

    @classmethod
    def from_strings(cls, options: Sequence[Sequence[str]], source: str = "") -> "GoalSpec":
        return cls([[Atom.parse(s) for s in opt] for opt in options], True, source)

    def atoms(self) -> List[Atom]:
        seen, out = set(), []
        for opt in self.options:
            for a in opt:
                if a.key not in seen:
                    seen.add(a.key)
                    out.append(a)
        return out


@dataclass(frozen=True)
class Milestone:
    """A validated intermediate stage. kind 'state': all `literals` hold at one step (with the prerequisites reached before or
    at that step); kind 'event': the env reports it (events[id]). `leads_to` names the goal atoms it is a confirmed
    prerequisite of: it counts for an option only if the option contains one of them."""
    id: str
    leads_to: Tuple[str, ...]
    kind: str = "state"
    literals: Tuple[Atom, ...] = ()
    prereqs: Tuple[str, ...] = ()
    weight: float = 1.0
    source: str = ""
    validated: bool = True


def toggle_milestones(spec: GoalSpec) -> List[Milestone]:
    """The ToggledOn rule of OmniGibson: the state flips only after a robot finger overlaps the object's toggle marker for the
    dwell time. Engaging the marker is therefore a necessary stage of every positive toggled_on goal atom (an event the env
    measures from its toggle history)."""
    out = []
    for a in spec.atoms():
        if a.pred == "toggled_on" and not a.negated:
            out.append(Milestone(f"toggle_engaged:{a.args[0]}", (a.key,), "event",
                                 source="OmniGibson ToggledOn rule: robot finger overlaps the toggle marker"))
    return out


@dataclass
class _MState:
    initially: bool = False
    reached_step: Optional[int] = None
    evaluable: bool = True


class GoalAlignedFitness:
    def __init__(self, spec: GoalSpec, milestones: Iterable[Milestone] = ()):
        self.spec = spec
        self.milestones = [m for m in milestones if m.validated]
        self.rejected = [m.id for m in milestones if not m.validated]
        self.reset_done = False

    # ----------------------------------------------------------------- rollout
    def reset(self, option_truth: Sequence[Sequence[bool]], distances: Mapping[str, float] = None,
              literals: Mapping[str, Optional[bool]] = None, events: Mapping[str, bool] = None) -> None:
        self._check(option_truth)
        self.init_truth = [list(map(bool, o)) for o in option_truth]
        self.last_truth = [list(o) for o in self.init_truth]
        self.d0 = {k: float(v) for k, v in (distances or {}).items() if v is not None and math.isfinite(v)}
        self.d_last = dict(self.d0)
        self.ms = {m.id: _MState() for m in self.milestones}
        self.constraints: Dict[str, int] = {}
        self.steps = 0
        for m in self.milestones:
            s = self.ms[m.id]
            now = self._milestone_now(m, literals or {}, events or {})
            if now is None:
                s.evaluable = False
            elif now:
                s.initially = True           # true at reset: no positive progress, but it opens the stages after it
        self.reset_done = True

    def update(self, option_truth: Sequence[Sequence[bool]], distances: Mapping[str, float] = None,
               literals: Mapping[str, Optional[bool]] = None, events: Mapping[str, bool] = None,
               constraints: Iterable[str] = ()) -> None:
        assert self.reset_done, "reset() first"
        self._check(option_truth)
        self.steps += 1
        self.last_truth = [list(map(bool, o)) for o in option_truth]
        for k, v in (distances or {}).items():
            if k in self.d0 and v is not None and math.isfinite(v):
                self.d_last[k] = float(v)
        for m in self.milestones:  # in list order; a stage reached at this step can open a later stage at the same step
            s = self.ms[m.id]
            if not s.evaluable or s.initially or s.reached_step is not None:
                continue
            if not all(self._open(p) for p in m.prereqs):
                continue
            now = self._milestone_now(m, literals or {}, events or {})
            if now is None:
                s.evaluable = False
            elif now:
                s.reached_step = self.steps  # kept in the history even if the state changes later (a finished stage)
        for c in constraints:
            if c not in CONSTRAINT_SEVERITY:
                raise ValueError(f"unknown constraint kind {c!r}")
            self.constraints.setdefault(c, self.steps)

    def _open(self, mid: str) -> bool:
        s = self.ms.get(mid)
        return s is not None and (s.initially or s.reached_step is not None)

    def _milestone_now(self, m: Milestone, literals, events) -> Optional[bool]:
        if m.kind == "event":
            v = events.get(m.id)
            return None if v is None else bool(v)
        vals = [literals.get(a.key) for a in m.literals]
        if any(v is None for v in vals):
            return None
        return all(vals)

    def _check(self, option_truth) -> None:
        if len(option_truth) != len(self.spec.options) or any(len(t) != len(o) for t, o in zip(option_truth, self.spec.options)):
            raise ValueError("option truth does not match the goal spec")

    # ----------------------------------------------------------------- result
    def option_components(self, b: int) -> Dict[str, float]:
        atoms, t0, tT = self.spec.options[b], self.init_truth[b], self.last_truth[b]
        n = len(atoms)
        newly = sum(1 for i0, iT in zip(t0, tT) if iT and not i0)
        lost = sum(1 for i0, iT in zip(t0, tT) if i0 and not iT)
        q = newly / n if n else 0.0
        g = min(1.0, max(0.0, (newly - lost) / n)) if n else 0.0
        keys = {a.key for a in atoms}
        mine = [m for m in self.milestones if keys.intersection(m.leads_to) and self.ms[m.id].evaluable and not self.ms[m.id].initially]
        wsum = sum(m.weight for m in mine)
        p = sum(m.weight for m in mine if self.ms[m.id].reached_step is not None) / wsum if wsum else None
        mval = W_G * g + W_P * p if p is not None else g
        dj = []
        for a, i0, iT in zip(atoms, t0, tT):
            if a.negated or i0 or a.key not in self.d0:
                continue                      # no shaping for negated atoms, atoms true at reset, atoms without an adapter
            if iT:
                dj.append(1.0)
                continue
            d0, dT = self.d0[a.key], self.d_last.get(a.key, self.d0[a.key])
            gain = d0 - dT if d0 - dT > DEADBAND_M else 0.0
            dj.append(min(1.0, max(0.0, gain / max(d0, EPS_M))))
        d = sum(dj) / len(dj) if dj else 0.0
        c = min(1.0, sum(CONSTRAINT_SEVERITY[k] for k in self.constraints))
        f = W_Q * q + W_M * mval + W_D * d - W_C * c
        return dict(q=q, goal=g, milestone=(p if p is not None else 0.0), m=mval, distance=d, constraint=c, f=f,
                    n_distance_atoms=len(dj), n_milestones=len(mine))

    def finalize(self, success: bool, official_q: Optional[float] = None) -> Dict:
        comps = [self.option_components(b) for b in range(len(self.spec.options))]
        b = max(range(len(comps)), key=lambda i: (comps[i]["f"], comps[i]["q"], -i)) if comps else None
        best = comps[b] if b is not None else dict(q=0.0, goal=0.0, milestone=0.0, m=0.0, distance=0.0, constraint=0.0, f=0.0,
                                                   n_distance_atoms=0, n_milestones=0)
        q_official = 1.0 if success else max((c["q"] for c in comps), default=0.0)
        fitness = 1.0 if success else min(FAIL_CAP, max(0.0, best["f"]))
        with_d = {a.key for a in self.spec.atoms() if a.key in self.d0}
        positive = [a for a in self.spec.atoms() if not a.negated]
        unsupported = sorted({a.pred for a in positive if a.key not in with_d and a.pred not in BINARY_PREDICATES})
        level = 3 if any(s.evaluable for s in self.ms.values()) else 2 if with_d else 1
        return dict(
            fitness=round(fitness, 6), success=bool(success), q_score=q_official,
            q_matches_official=(None if official_q is None else abs(q_official - official_q) < 1e-9),
            goal_progress=best["goal"], milestone_progress=best["milestone"], distance_progress=best["distance"],
            constraint_penalty=best["constraint"], goal_branch=b, coverage_level=level, complete_goal=self.spec.complete,
            achieved_milestones={m.id: self.ms[m.id].reached_step for m in self.milestones if self.ms[m.id].reached_step is not None},
            initially_true_milestones=[m.id for m in self.milestones if self.ms[m.id].initially],
            unevaluable_milestones=[m.id for m in self.milestones if not self.ms[m.id].evaluable] + self.rejected,
            distance_atoms=sorted(with_d), unsupported_predicates=unsupported,
            binary_predicates=sorted({a.pred for a in positive if a.pred in BINARY_PREDICATES}), constraints=dict(self.constraints),
            evaluator_version=EVALUATOR_VERSION)


# --------------------------------------------------------------------- aggregation over episodes
def aggregate(results: Sequence[Mapping]) -> Dict[str, float]:
    """Mean over the episodes of one candidate. mean_fitness is the dashboard's Best Fitness; selection_score is for internal
    ranking only (mean - 0.5 * standard error)."""
    n = len(results)
    if not n:
        return dict(num_episodes=0, mean_fitness=0.0, fitness_std=0.0, selection_score=0.0)
    f = [float(r["fitness"]) for r in results]
    mean = sum(f) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in f) / (n - 1)) if n > 1 else 0.0
    se = std / math.sqrt(n)

    def m(k):
        return sum(float(r.get(k) or 0.0) for r in results) / n
    return dict(num_episodes=n, mean_fitness=mean, fitness_std=std, selection_score=mean - SE_PENALTY * se,
                mean_q_score=m("q_score"), success_rate=sum(bool(r.get("success")) for r in results) / n,
                mean_goal_progress=m("goal_progress"), mean_milestone_progress=m("milestone_progress"),
                mean_distance_progress=m("distance_progress"), mean_constraint_penalty=m("constraint_penalty"))
