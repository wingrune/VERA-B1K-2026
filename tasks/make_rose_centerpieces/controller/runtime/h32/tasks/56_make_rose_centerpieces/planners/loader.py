"""Load the task planner as a package, without global legacy module aliases."""
import importlib.util
from pathlib import Path
import sys


def load_planner(kind="fair"):
    if kind != "fair":
        raise ValueError("This adaptation only provides the frozen fair controller")
    name = "bimbo_task_56_frozen_v36_fair"
    if name not in sys.modules:
        folder = Path(__file__).resolve().parent / "fair"
        spec = importlib.util.spec_from_file_location(name, folder / "__init__.py", submodule_search_locations=[str(folder)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    sys.modules.pop(key, None)
            raise
    return sys.modules[name]
