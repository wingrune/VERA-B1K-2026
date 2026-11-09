"""Task-local native harness adapter. Privileged predicates stay here."""
import argparse
import json
import os
import inspect
from pathlib import Path
import sys
import subprocess
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))


def recording_compatibility():
    # Existing Docker image's LeRobot predates the pinned source's recording API.
    # Evaluation/video uses the native harness, not LeRobot recording.
    stub = types.ModuleType("omnigibson.envs.lerobot_data_wrapper")
    class UnavailableRecording:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Use native harness video, not the unavailable LeRobot recorder")
    stub.LeRobotDataWrapper = stub.LeRobotPlaybackWrapper = UnavailableRecording
    sys.modules[stub.__name__] = stub


def make_adapter():
    from core.harness import native_runner as runner
    import loader

    class RoseAdapter(runner.TaskAdapter):
        task = "make_rose_centerpieces"
        subgoals = ("inside_rose_1", "inside_rose_2", "inside_rose_3", "vase_on_table")

        def load_planner(self, kind):
            return loader.load_planner(kind)

        def planner_dir(self, kind):
            return HERE / "fair"

        def bind(self, env, instance):
            from omnigibson.object_states import Inside, OnTop
            self.env = env
            scope = env.acc.object_scope
            rose_keys = sorted(k for k in scope if k.startswith("rose.n.01_"))
            if rose_keys != [f"rose.n.01_{i}" for i in range(1, 4)]:
                raise ValueError(f"Unexpected runtime rose scope: {rose_keys}")
            unwrap = lambda obj: getattr(obj, "wrapped_obj", obj)
            self.roses = [unwrap(scope[k]) for k in rose_keys]
            self.vase = unwrap(scope["vase.n.01_1"])
            self.table = unwrap(scope["coffee_table.n.01_1"])
            self.Inside, self.OnTop = Inside, OnTop
            options = env.goal_options()
            if len(options) != 1 or len(options[0]) != 4:
                raise ValueError(f"Unexpected native grounded goal size: {options}")

        def read_state(self, step):
            return {**{f"inside_rose_{i}": bool(rose.states[self.Inside].get_value(self.vase))
                       for i, rose in enumerate(self.roses, 1)},
                    "vase_on_table": bool(self.vase.states[self.OnTop].get_value(self.table))}

        def evaluate(self, state):
            return state

        def sample(self, state, log):
            return {"native_predicates": state}

        def dump(self, path, obs, state, step):
            import numpy as np
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **{k: runner.to_numpy(v) for k, v in obs.items()})

        def extra_result(self):
            return {"adaptation_only": True, "source_variant": "v36_identity_fast",
                    "source_commit": "9e8004ebd7fdbbf39821b01b0a80ee5b3074c3b6",
                    "frozen_source_files": 192}
    return RoseAdapter()


def native_env(task, mode="public_test", seed=None):
    from core.harness import native_runner
    class DockerNativeEnv(native_runner.NativeEnv):
        def commit(self):
            import omnigibson
            package = Path(inspect.getfile(omnigibson)).resolve().parent
            root = package.parent.parent
            # Docker runs as root; the read-only host checkout has another owner.
            # Trust only this mounted repository, for this command, not all Git paths.
            return subprocess.check_output(["git", "-c", f"safe.directory={root}", "-C", str(package),
                                            "rev-parse", "HEAD"], text=True).strip()
    return DockerNativeEnv(task, mode, seed)


def main():
    recording_compatibility()
    import torch
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    from core.harness import native_runner
    adapter = make_adapter()
    # NativeEnv explicitly verifies live CPU PhysX; no simulator state reaches act().
    return native_runner.main(adapter, env_factory=native_env)


if __name__ == "__main__":
    main()
