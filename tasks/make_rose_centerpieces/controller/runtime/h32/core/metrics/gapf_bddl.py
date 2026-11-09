"""GAPF from the official BDDL definitions: goal options, confirmed milestones from transition rules, coverage of B100.

  BDDLGoalExtractor            bddl.activity grounding (the same get_ground_goal_state_options BehaviorTask uses) ->
                               GoalSpec; :objects, :init and the future instances (products of transition rules)
  TransitionDependencyExtractor  milestones of goal atoms on a future instance from the transition map of bddl
                               (generated_data/transition_map/tm_jsons) and the heatSource/slicer annotations
                               (generated_data/propagated_annots_params.json). A milestone is a NECESSARY state of the rule
                               (a conjunction of atoms that must hold at one step), never an order the rule does not impose.
                               A product with no feasible rule, or several, gets no milestones and a note.
  coverage CLI                 python -m core.metrics.gapf_bddl [--tasks core/metrics/b100_tasks.txt] [--out report.json]
                               grounds every task, reports options, atoms, predicates, adapters, milestones, the coverage
                               level the evaluator can reach with a full geometry provider, and the gaps.
Needs the bddl package of BEHAVIOR-1K (bd049de, bddl3); nothing here imports a simulator.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .gapf import BINARY_PREDICATES, Atom, GoalSpec, Milestone, toggle_milestones
from .gapf_adapters import REGISTRY

RULE_FILES = ("heat_cook", "substance_cooking", "substance_watercooking", "melting", "single_toggleable_machine", "slicing",
              "dicing", "mixing_stick")  # washer rules have no product of a goal (they remove particles)


def synset_of(inst: str) -> str:
    return inst.rsplit("_", 1)[0]


def _atom_str(b) -> str:
    if isinstance(b, (list, tuple)):
        return "(" + " ".join(_atom_str(x) for x in b) + ")"
    return str(b).lstrip("?")


@lru_cache(maxsize=1)
def _bddl_dir() -> Path:
    import bddl
    return Path(os.path.dirname(bddl.__file__))


@lru_cache(maxsize=1)
def properties() -> Dict[str, dict]:
    return json.loads((_bddl_dir() / "generated_data" / "propagated_annots_params.json").read_text())


@lru_cache(maxsize=1)
def rules() -> List[dict]:
    out = []
    for name in RULE_FILES:
        for r in json.loads((_bddl_dir() / "generated_data" / "transition_map" / "tm_jsons" / f"{name}.json").read_text()):
            out.append({**r, "type": name})
    return out


class BDDLGoalExtractor:
    def __init__(self, activity: str, definition: int = 0):
        from bddl import activity as act
        self.activity, self.definition = activity, definition
        self.conds = act.Conditions(activity, definition, "behavior-1k")
        self.objects: Dict[str, List[str]] = {k: list(v) for k, v in self.conds.parsed_objects.items()}
        scope = {k: k for k in act.create_scope(self.conds.parsed_objects)}
        gc = act.get_goal_conditions(self.conds, scope)
        try:
            opts = act.get_ground_goal_state_options(self.conds, scope, gc)
            self.spec = GoalSpec([[Atom.parse(_atom_str(h.body)) for h in o] for o in opts], True, f"bddl:{activity}/{definition}")
        except AssertionError as e:  # no consistent finite grounding: say so instead of pretending a full evaluation
            self.spec = GoalSpec([], False, f"bddl:{activity}/{definition}: {e}")
        self.init = [_atom_str(c) for c in self.conds.parsed_initial_conditions]

    def instances(self) -> List[str]:
        return [i for v in self.objects.values() for i in v if not i.endswith("_*")]

    def future(self) -> List[str]:
        return [Atom.parse(s).args[0] for s in self.init if s.startswith("(future ")]


def _is_substance(synset: str) -> bool:
    return "substance" in properties().get(synset, {})


def _instances_of(objects: Dict[str, List[str]], synset: str) -> List[str]:
    return [i for i in objects.get(synset, []) if not i.endswith("_*")]


class TransitionDependencyExtractor:
    def __init__(self, ex: BDDLGoalExtractor):
        self.ex = ex
        self.notes: List[str] = []

    def _container_of(self, inst: str) -> Optional[str]:
        """The container a substance starts in, from :init (filled C X) or (contains C X)."""
        for s in self.ex.init:
            a = Atom.parse(s) if not s.startswith("(not") else None
            if a and a.pred in ("filled", "contains") and len(a.args) == 2 and a.args[1] == inst:
                return a.args[0]
        return None

    def _heat_literals(self, container: str, heat: str) -> List[Atom]:
        hs = properties().get(synset_of(heat), {}).get("heatSource", {})
        lits = [Atom("inside" if hs.get("requires_inside") else "ontop", (container, heat))]
        if hs.get("requires_closed"):
            lits.append(Atom("open", (heat,), True))
        if hs.get("requires_toggled_on"):
            lits.append(Atom("toggled_on", (heat,)))
        return lits

    def _in(self, item: str, container: str) -> Atom:
        return Atom("contains", (container, item)) if _is_substance(synset_of(item)) else Atom("inside", (item, container))

    def _feasible(self, prod: str, have: set, depth: int = 0) -> List[Tuple[dict, List[dict]]]:
        """Rules producing `prod` from task synsets, through at most 3 stages of generated intermediates (dicing a half that
        slicing makes). Returns (rule, the rules of the earlier stages it needs)."""
        out = []
        for r in rules():
            if prod not in r.get("output_synsets", {}):
                continue
            if not (set(r.get("container") or {}) | set(r.get("heat_source") or {}) | set(r.get("machine") or {})) <= have:
                continue
            chain, ok = [], True
            for s in r.get("input_synsets") or {}:
                if s in have:
                    continue
                sub = self._feasible(s, have, depth + 1) if depth < 2 else []
                if len(sub) != 1:
                    ok = False
                    break
                chain += [sub[0][0]] + sub[0][1]
            if ok:
                out.append((r, chain))
        return out

    def milestones(self) -> List[Milestone]:
        ex, out = self.ex, []
        future = set(ex.future())
        have = {k for k, v in ex.objects.items() if _instances_of(ex.objects, k)}
        for atom in ex.spec.atoms():
            for x in atom.args:
                if x not in future or atom.negated:
                    continue
                prod = synset_of(x)
                feas = self._feasible(prod, have)
                if len(feas) != 1:
                    self.notes.append(f"{prod}: {len(feas)} feasible rules, no milestones")
                    continue
                rule, chain = feas[0]
                stages = [rule] + chain   # milestones only for stages whose inputs are task instances (the first ones)
                for r in stages:
                    if all(s in have for s in (r.get("input_synsets") or {})):
                        ms = self._from_rule(r, atom)
                        out += [m for m in ms if m.id not in {o.id for o in out}]
                    else:
                        self.notes.append(f"{r['rule_name']}: inputs are generated objects, no milestone")
        return out

    def _one(self, synset: str) -> Optional[str]:
        inst = _instances_of(self.ex.objects, synset)
        return inst[0] if len(inst) == 1 else None

    def _from_rule(self, r: dict, goal: Atom) -> List[Milestone]:
        t, ex = r["type"], self.ex
        per_synset = [_instances_of(ex.objects, s) for s in (r.get("input_synsets") or {})]
        if not all(per_synset):
            self.notes.append(f"{r['rule_name']}: an input has no task instance")
            return []
        inputs = [i for v in per_synset for i in v]   # every instance of an input synset in the task takes part
        src = f"transition rule {t}:{r['rule_name']}"
        lead = (goal.key,)
        if t in ("heat_cook", "substance_cooking", "substance_watercooking", "melting"):
            if t == "heat_cook":
                container, heat = self._one(next(iter(r["container"]))), self._one(next(iter(r["heat_source"])))
            else:
                cs = {self._container_of(i) for i in inputs}
                container = cs.pop() if len(cs) == 1 else None
                heats = [i for i in ex.instances() if "heatSource" in properties().get(synset_of(i), {})]
                heat = heats[0] if len(heats) == 1 else None
            if not container or not heat:
                self.notes.append(f"{r['rule_name']}: container or heat source not unique in the task")
                return []
            inside = [self._in(i, container) for i in inputs]
            place = self._heat_literals(container, heat)
            return [Milestone(f"inputs_in:{container}", lead, literals=tuple(inside), source=src),
                    Milestone(f"placed:{container}@{heat}", lead, literals=tuple(a for a in place if a.pred != "toggled_on"), source=src),
                    Milestone(f"heating:{container}@{heat}", lead, literals=tuple(inside + place), source=src, weight=2.0)]
        if t == "single_toggleable_machine":
            m = self._one(next(iter(r["machine"])))
            if not m:
                self.notes.append(f"{r['rule_name']}: machine not unique")
                return []
            inside = [self._in(i, m) for i in inputs]
            return [Milestone(f"inputs_in:{m}", lead, literals=tuple(inside), source=src),
                    Milestone(f"running:{m}", lead, literals=tuple(inside + [Atom("toggled_on", (m,))]), source=src, weight=2.0)]
        if t in ("slicing", "dicing"):
            slicers = [i for i in ex.instances() if "slicer" in properties().get(synset_of(i), {})]
            if len(slicers) != 1:
                self.notes.append(f"{r['rule_name']}: {len(slicers)} slicers in the task")
                return []
            return [Milestone(f"cut_contact:{slicers[0]}>{i}", lead, literals=(Atom("touching", (slicers[0], i)),), source=src) for i in inputs]
        self.notes.append(f"{r['rule_name']}: rule type {t} has no milestone builder")
        return []


def task_report(activity: str) -> dict:
    ex = BDDLGoalExtractor(activity)
    tr = TransitionDependencyExtractor(ex)
    ms = tr.milestones() + toggle_milestones(ex.spec)
    atoms = ex.spec.atoms()
    preds = Counter(a.pred + ("(neg)" if a.negated else "") for a in atoms)
    shaped = [a.key for a in atoms if not a.negated and a.pred in REGISTRY]
    unsupported = sorted({a.pred for a in atoms if not a.negated and a.pred not in REGISTRY and a.pred not in BINARY_PREDICATES})
    level = 3 if ms else 2 if shaped else 1
    return dict(task=activity, complete=ex.spec.complete, options=len(ex.spec.options), atoms=len(atoms), predicates=dict(preds),
                future=ex.future(), distance_atoms=len(shaped), milestones=[dict(id=m.id, kind=m.kind, literals=[a.key for a in m.literals],
                                                                                   source=m.source) for m in ms],
                level=level, unsupported_predicates=unsupported, notes=tr.notes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="GAPF coverage over a task list (BDDL only, no simulator)")
    ap.add_argument("--tasks", default=str(Path(__file__).with_name("b100_tasks.txt")))
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    names = [l.strip() for l in Path(a.tasks).read_text().splitlines() if l.strip() and not l.startswith("#")]
    rows, errors = [], {}
    for n in names:
        try:
            rows.append(task_report(n))
        except Exception as e:  # noqa: BLE001 - a task that does not ground is reported, the others go on
            errors[n] = f"{type(e).__name__}: {e}"[:300]
    summary = dict(tasks=len(names), grounded=len(rows), errors=errors, levels=dict(Counter(r["level"] for r in rows)),
                   incomplete=[r["task"] for r in rows if not r["complete"]],
                   predicates=dict(sum((Counter(r["predicates"]) for r in rows), Counter()).most_common()),
                   unsupported=dict(Counter(p for r in rows for p in r["unsupported_predicates"])))
    text = json.dumps(dict(summary=summary, tasks=rows), indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(json.dumps(summary, indent=1))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
