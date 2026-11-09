"""User-authorized v36 overlay: finite search counts, no wall-clock cutoffs.

Applied only when importing the hash-verified frozen controller. Real clocks
remain available for diagnostics; candidate order, collision checks, solver
iteration limits and simulated-step limits are unchanged. The seconds/deadline
arguments remain for compatibility but cannot terminate a search.
"""
import importlib.abc
import importlib.util
from pathlib import Path

VARIANT = "v36_identity_fast_no_search_deadlines"
REPLACEMENTS = {
    "time.monotonic()>deadline": "False",
    "time.monotonic()>=deadline": "False",
    "time.monotonic()<deadline": "True",
    "remaining=deadline-time.monotonic()": "remaining=float('inf')",
    "time.monotonic()-started>=3.": "False",
    "if seconds<=0:": "if False:",
}
# Counts are specific to the pinned 9e8004e v36 source, not a general rewriter.
EXPECTED = {
    "compost_policy": {"remaining=deadline-time.monotonic()": 1},
    "dual_sequence": {"time.monotonic()>deadline": 3},
    "held_vase_view": {"time.monotonic()>deadline": 1},
    "joint_path": {"time.monotonic()>deadline": 1},
    "paired_grasp_planner": {"time.monotonic()>=deadline": 2},
    "pose_navigation": {"time.monotonic()<deadline": 1},
    "pose_search": {"remaining=deadline-time.monotonic()": 3},
    "reach_planner": {"time.monotonic()>deadline": 2},
    "released_pose_fit": {"time.monotonic()>deadline": 2, "if seconds<=0:": 1},
    "rose_geometry": {"time.monotonic()>deadline": 2},
    "rose_perception": {"time.monotonic()-started>=3.": 1},
    "rose_policy": {"time.monotonic()>deadline": 1},
}


def transform(name, source):
    expected = EXPECTED.get(name, {})
    for old, count in expected.items():
        if source.count(old) != count:
            raise ValueError(f"Search-time overlay does not match pinned {name}: {old!r}")
        source = source.replace(old, REPLACEMENTS[old])
    return source


class _Loader(importlib.abc.Loader):
    def __init__(self, name, path):
        self.name, self.path = name, path

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # Preserve __file__ so canonical priors resolve beside frozen sources.
        code = compile(transform(self.name, self.path.read_text()), str(self.path), "exec")
        exec(code, module.__dict__)


class SearchTimeFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root):
        self.root = Path(root)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in EXPECTED:
            return None
        source = self.root / (fullname + ".py")
        if not source.is_file():
            raise ImportError(f"Missing pinned controller module {fullname}")
        return importlib.util.spec_from_file_location(fullname, source, loader=_Loader(fullname, source))
