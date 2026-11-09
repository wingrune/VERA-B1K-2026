"""Episode trace written by the harness, independent of the planner.

The harness feeds `Trace` from privileged env state (subgoal predicates, object poses,
contacts). The planner only contributes optional phase names and reasons. Failure causes are
classified here from env measurements, never from planner-reported stages.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

FAILURE_CAUSES = ("not_reached", "not_grasped", "dropped", "blocked", "exception", "no_progress", "infra")


@dataclass
class Trace:
    task: str
    seed: int
    horizon: int
    subgoals: list[str]
    events: list[dict[str, Any]] = field(default_factory=list)
    _phase: str | None = None
    _phase_start: int = 0
    _reached: dict[str, int] = field(default_factory=dict)
    _lost: dict[str, int] = field(default_factory=dict)

    # --- planner side (phase names are advisory; harness keeps its own clock) ---
    def phase(self, step: int, name: str, reason: str = "") -> None:
        if name == self._phase:
            return
        if self._phase is not None:
            self.events.append({"t": step, "ev": "phase_end", "phase": self._phase,
                                "dur": step - self._phase_start, "next": name, "reason": reason})
        self.events.append({"t": step, "ev": "phase_start", "phase": name})
        self._phase, self._phase_start = name, step

    def exception(self, step: int, text: str) -> None:
        self.events.append({"t": step, "ev": "exception", "text": text[:300], "phase": self._phase})

    # --- harness side ---
    def subgoal_state(self, step: int, state: dict[str, bool]) -> None:
        for g, ok in state.items():
            if ok and g not in self._reached:
                self._reached[g] = step
                self.events.append({"t": step, "ev": "subgoal_reached", "goal": g})
            elif not ok and g in self._reached and g not in self._lost:
                self._lost[g] = step
                self.events.append({"t": step, "ev": "subgoal_lost", "goal": g, "since": self._reached[g]})
            elif ok and g in self._lost:
                del self._lost[g]
                self.events.append({"t": step, "ev": "subgoal_regained", "goal": g})

    def perception_error(self, step: int, obj: str, err_m: float) -> None:
        self.events.append({"t": step, "ev": "perception_err", "obj": obj, "err_m": round(err_m, 3)})

    def held_goals(self) -> list[str]:
        return [g for g in self.subgoals if g in self._reached and g not in self._lost]

    def progress_env(self) -> float:
        return len(self.held_goals()) / max(1, len(self.subgoals))

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e, ensure_ascii=False) for e in self.events)


@dataclass
class EpisodeFacts:
    """Env measurements collected over the episode, used to classify failure."""
    success: bool
    min_dist_to_target: float          # best approach to the key object, metres
    reach_radius: float                # distance counted as "reached"
    ever_grasped: bool
    held_at_end: bool
    dropped_after_grasp: bool
    base_displacement_last_window: float
    window_steps: int
    exceptions: int
    infra_error: bool = False
    subgoals_done: int = 0


def classify_failure(f: EpisodeFacts) -> str | None:
    """Cause of failure from env measurements. None for success."""
    if f.success:
        return None
    if f.infra_error:
        return "infra"
    if f.exceptions and f.subgoals_done == 0:
        return "exception"
    if f.min_dist_to_target > f.reach_radius:
        return "blocked" if f.base_displacement_last_window < 0.02 else "not_reached"
    if not f.ever_grasped:
        return "not_grasped"
    if f.dropped_after_grasp and not f.held_at_end:
        return "dropped"
    return "no_progress"


def compress(tr: Trace, cause: str | None, max_events: int = 40) -> str:
    """Short text for the mutation prompt (about 150-400 tokens per episode)."""
    lines = [f"seed={tr.seed} progress={tr.progress_env():.2f} cause={cause or 'success'} "
             f"goals_held={','.join(tr.held_goals()) or '-'}"]
    keep = [e for e in tr.events if e["ev"] != "phase_start"]
    if len(keep) > max_events:
        keep = keep[: max_events // 2] + keep[-max_events // 2:]
    for e in keep:
        k = e["ev"]
        if k == "phase_end":
            lines.append(f"{e['t']}: {e['phase']} -> {e['next']} after {e['dur']} steps ({e['reason']})")
        elif k == "exception":
            lines.append(f"{e['t']}: EXC in {e['phase']}: {e['text']}")
        elif k.startswith("subgoal"):
            lines.append(f"{e['t']}: {k} {e['goal']}")
        elif k == "perception_err":
            lines.append(f"{e['t']}: perception {e['obj']} err={e['err_m']}m")
    return "\n".join(lines)
