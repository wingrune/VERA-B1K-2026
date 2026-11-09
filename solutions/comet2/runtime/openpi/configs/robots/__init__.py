from pathlib import Path
import importlib

from .base_config import ROBOT_REGISTRY


_current_dir = Path(__file__).parent
_robot_modules = [
    f.stem for f in _current_dir.glob("*.py")
    if f.stem not in ("__init__", "base_config")
]

for _module_name in _robot_modules:
    importlib.import_module(f".{_module_name}", package=__name__)

__all__ = ["ROBOT_REGISTRY"]
