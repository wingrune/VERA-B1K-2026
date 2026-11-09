from lerobot.__version__ import __version__


available_extras: dict[str, list[str]] = {
    "dataset": ["lerobot-dataset-viz", "lerobot-imgtransform-viz", "lerobot-edit-dataset"],
    "training": ["lerobot-train"],
    "hardware": [
        "lerobot-calibrate",
        "lerobot-find-port",
        "lerobot-find-cameras",
        "lerobot-find-joint-limits",
        "lerobot-setup-motors",
    ],
    "core_scripts": ["lerobot-record", "lerobot-replay", "lerobot-teleoperate"],
    "evaluation": ["lerobot-eval"],
}

__all__ = ["__version__", "available_extras"]
