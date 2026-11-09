from __future__ import annotations

import importlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypedDict, TypeVar, cast

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvAction, EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.hub import HubMixin

from .converters import batch_to_transition, create_transition, transition_to_batch


TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ProcessorStepRegistry:


    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str | None = None):


        def decorator(step_class: type) -> type:

            registration_name = name if name is not None else step_class.__name__

            if registration_name in cls._registry:
                raise ValueError(
                    f"Processor step '{registration_name}' is already registered. "
                    f"Use a different name or unregister the existing one first."
                )

            cls._registry[registration_name] = step_class

            step_class._registry_name = registration_name
            return step_class

        return decorator

    @classmethod
    def get(cls, name: str) -> type:

        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(
                f"Processor step '{name}' not found in registry. "
                f"Available steps: {available}. "
                f"Make sure the step is registered using @ProcessorStepRegistry.register()"
            )
        return cls._registry[name]

    @classmethod
    def unregister(cls, name: str) -> None:

        cls._registry.pop(name, None)

    @classmethod
    def list(cls) -> list[str]:

        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:

        cls._registry.clear()


class ProcessorStep(ABC):


    _current_transition: EnvTransition | None = None

    @property
    def transition(self) -> EnvTransition:

        if self._current_transition is None:
            raise ValueError("Transition is not set. Make sure to call the step with a transition first.")
        return self._current_transition

    @abstractmethod
    def __call__(self, transition: EnvTransition) -> EnvTransition:

        return transition

    def get_config(self) -> dict[str, Any]:

        return {}

    def state_dict(self) -> dict[str, torch.Tensor]:

        return {}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:

        return None

    def reset(self) -> None:

        return None

    @abstractmethod
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        return features


class ProcessorKwargs(TypedDict, total=False):


    to_transition: Callable[[dict[str, Any]], EnvTransition] | None
    to_output: Callable[[EnvTransition], Any] | None
    name: str | None
    before_step_hooks: list[Callable[[int, EnvTransition], None]] | None
    after_step_hooks: list[Callable[[int, EnvTransition], None]] | None


class ProcessorMigrationError(Exception):


    def __init__(self, model_path: str | Path, migration_command: str, original_error: str):
        self.model_path = model_path
        self.migration_command = migration_command
        self.original_error = original_error
        super().__init__(
            f"Model '{model_path}' requires migration to processor format. "
            f"Run: {migration_command}\n\nOriginal error: {original_error}"
        )


@dataclass
class DataProcessorPipeline(Generic[TInput, TOutput], HubMixin):


    steps: Sequence[ProcessorStep] = field(default_factory=list)
    name: str = "DataProcessorPipeline"

    to_transition: Callable[[TInput], EnvTransition] = field(
        default_factory=lambda: cast(Callable[[TInput], EnvTransition], batch_to_transition), repr=False
    )
    to_output: Callable[[EnvTransition], TOutput] = field(
        default_factory=lambda: cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        repr=False,
    )

    before_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)
    after_step_hooks: list[Callable[[int, EnvTransition], None]] = field(default_factory=list, repr=False)

    def __call__(self, data: TInput) -> TOutput:

        transition = self.to_transition(data)
        transformed_transition = self._forward(transition)
        return self.to_output(transformed_transition)

    def _forward(self, transition: EnvTransition) -> EnvTransition:

        for idx, processor_step in enumerate(self.steps):

            for hook in self.before_step_hooks:
                hook(idx, transition)

            transition = processor_step(transition)


            for hook in self.after_step_hooks:
                hook(idx, transition)
        return transition

    def step_through(self, data: TInput) -> Iterable[EnvTransition]:

        transition = self.to_transition(data)


        yield transition

        for processor_step in self.steps:
            transition = processor_step(transition)
            yield transition

    def _save_pretrained(self, save_directory: Path, **kwargs):

        config_filename = kwargs.pop("config_filename", None)


        sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())

        if config_filename is None:
            config_filename = f"{sanitized_name}.json"

        config: dict[str, Any] = {
            "name": self.name,
            "steps": [],
        }


        for step_index, processor_step in enumerate(self.steps):
            registry_name = getattr(processor_step.__class__, "_registry_name", None)

            step_entry: dict[str, Any] = {}

            if registry_name:
                step_entry["registry_name"] = registry_name
            else:
                step_entry["class"] = (
                    f"{processor_step.__class__.__module__}.{processor_step.__class__.__name__}"
                )


            if hasattr(processor_step, "get_config"):
                step_entry["config"] = processor_step.get_config()


            if hasattr(processor_step, "state_dict"):
                state = processor_step.state_dict()
                if state:

                    cloned_state = {key: tensor.clone() for key, tensor in state.items()}


                    if registry_name:
                        state_filename = f"{sanitized_name}_step_{step_index}_{registry_name}.safetensors"
                    else:
                        state_filename = f"{sanitized_name}_step_{step_index}.safetensors"

                    save_file(cloned_state, os.path.join(str(save_directory), state_filename))
                    step_entry["state_file"] = state_filename

            config["steps"].append(step_entry)


        with open(os.path.join(str(save_directory), config_filename), "w") as file_pointer:
            json.dump(config, file_pointer, indent=2)

    def save_pretrained(
        self,
        save_directory: str | Path | None = None,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        config_filename: str | None = None,
        **push_to_hub_kwargs,
    ):

        if save_directory is None:

            sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.name.lower())
            save_directory = HF_LEROBOT_HOME / "processors" / sanitized_name


        if not push_to_hub and config_filename is not None:

            save_directory = Path(save_directory)
            save_directory.mkdir(parents=True, exist_ok=True)
            self._save_pretrained(save_directory, config_filename=config_filename)
            return None


        if config_filename is not None:
            push_to_hub_kwargs["config_filename"] = config_filename


        return super().save_pretrained(
            save_directory=save_directory,
            repo_id=repo_id,
            push_to_hub=push_to_hub,
            card_kwargs=card_kwargs,
            **push_to_hub_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        config_filename: str,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict[str, str] | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        overrides: dict[str, Any] | None = None,
        to_transition: Callable[[TInput], EnvTransition] | None = None,
        to_output: Callable[[EnvTransition], TOutput] | None = None,
        **kwargs,
    ) -> DataProcessorPipeline[TInput, TOutput]:

        model_id = str(pretrained_model_name_or_path)
        hub_download_kwargs = {
            "force_download": force_download,
            "resume_download": resume_download,
            "proxies": proxies,
            "token": token,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "revision": revision,
        }


        loaded_config, base_path = cls._load_config(model_id, config_filename, hub_download_kwargs)


        cls._validate_loaded_config(model_id, loaded_config, config_filename)


        steps, validated_overrides = cls._build_steps_with_overrides(
            loaded_config, overrides or {}, model_id, base_path, hub_download_kwargs
        )


        cls._validate_overrides_used(validated_overrides, loaded_config)


        return cls(
            steps=steps,
            name=loaded_config.get("name", "DataProcessorPipeline"),
            to_transition=to_transition or cast(Callable[[TInput], EnvTransition], batch_to_transition),
            to_output=to_output or cast(Callable[[EnvTransition], TOutput], transition_to_batch),
        )

    @classmethod
    def _load_config(
        cls,
        model_id: str,
        config_filename: str,
        hub_download_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], Path]:

        model_path = Path(model_id)

        if model_path.is_dir():

            config_path = model_path / config_filename
            if not config_path.exists():

                if cls._should_suggest_migration(model_path):
                    cls._suggest_processor_migration(model_id, f"Config file '{config_filename}' not found")
                raise FileNotFoundError(
                    f"Config file '{config_filename}' not found in directory '{model_id}'"
                )

            with open(config_path) as f:
                return json.load(f), model_path

        elif model_path.is_file():

            with open(model_path) as f:
                return json.load(f), model_path.parent

        else:

            try:
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename=config_filename,
                    repo_type="model",
                    **hub_download_kwargs,
                )

                with open(config_path) as f:
                    return json.load(f), Path(config_path).parent

            except Exception as e:
                raise FileNotFoundError(
                    f"Could not find '{config_filename}' on the HuggingFace Hub at '{model_id}'"
                ) from e

    @classmethod
    def _validate_loaded_config(
        cls, model_id: str, loaded_config: dict[str, Any], config_filename: str
    ) -> None:


        if not cls._is_processor_config(loaded_config):
            if Path(model_id).is_dir() and cls._should_suggest_migration(Path(model_id)):
                cls._suggest_processor_migration(
                    model_id,
                    f"Config file '{config_filename}' is not a valid processor configuration",
                )
            raise ValueError(
                f"Config file '{config_filename}' is not a valid processor configuration. "
                f"Expected a config with 'steps' field, but got: {list(loaded_config.keys())}"
            )

    @classmethod
    def _build_steps_with_overrides(
        cls,
        loaded_config: dict[str, Any],
        overrides: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        hub_download_kwargs: dict[str, Any],
    ) -> tuple[list[ProcessorStep], set[str]]:

        steps: list[ProcessorStep] = []
        override_keys = set(overrides.keys())

        for step_entry in loaded_config["steps"]:

            step_class, step_key = cls._resolve_step_class(step_entry)


            step_instance = cls._instantiate_step(step_entry, step_class, step_key, overrides)


            cls._load_step_state(step_instance, step_entry, model_id, base_path, hub_download_kwargs)


            if step_key in override_keys:
                override_keys.discard(step_key)

            steps.append(step_instance)

        return steps, override_keys

    @classmethod
    def _resolve_step_class(cls, step_entry: dict[str, Any]) -> tuple[type[ProcessorStep], str]:

        if "registry_name" in step_entry:
            try:
                step_class = ProcessorStepRegistry.get(step_entry["registry_name"])
                return step_class, step_entry["registry_name"]
            except KeyError as e:
                raise ImportError(f"Failed to load processor step from registry. {str(e)}") from e
        else:

            full_class_path = step_entry["class"]
            module_path, class_name = full_class_path.rsplit(".", 1)

            try:
                module = importlib.import_module(module_path)
                step_class = getattr(module, class_name)
                return step_class, class_name
            except (ImportError, AttributeError) as e:
                raise ImportError(
                    f"Failed to load processor step '{full_class_path}'. "
                    f"Make sure the module '{module_path}' is installed and contains class '{class_name}'. "
                    f"Consider registering the step using @ProcessorStepRegistry.register() for better portability. "
                    f"Error: {str(e)}"
                ) from e

    @classmethod
    def _instantiate_step(
        cls,
        step_entry: dict[str, Any],
        step_class: type[ProcessorStep],
        step_key: str,
        overrides: dict[str, Any],
    ) -> ProcessorStep:

        try:
            saved_cfg = step_entry.get("config", {})
            step_overrides = overrides.get(step_key, {})
            merged_cfg = {**saved_cfg, **step_overrides}
            return step_class(**merged_cfg)
        except Exception as e:
            step_name = step_entry.get("registry_name", step_entry.get("class", "Unknown"))
            raise ValueError(
                f"Failed to instantiate processor step '{step_name}' with config: {step_entry.get('config', {})}. "
                f"Error: {str(e)}"
            ) from e

    @classmethod
    def _load_step_state(
        cls,
        step_instance: ProcessorStep,
        step_entry: dict[str, Any],
        model_id: str,
        base_path: Path | None,
        hub_download_kwargs: dict[str, Any],
    ) -> None:

        if "state_file" not in step_entry or not hasattr(step_instance, "load_state_dict"):
            return

        state_filename = step_entry["state_file"]


        if base_path and (base_path / state_filename).exists():
            state_path = str(base_path / state_filename)
        else:

            state_path = hf_hub_download(
                repo_id=model_id,
                filename=state_filename,
                repo_type="model",
                **hub_download_kwargs,
            )

        step_instance.load_state_dict(load_file(state_path))

    @classmethod
    def _validate_overrides_used(
        cls, remaining_override_keys: set[str], loaded_config: dict[str, Any]
    ) -> None:

        if not remaining_override_keys:
            return

        available_keys = [
            step.get("registry_name") or step["class"].rsplit(".", 1)[1] for step in loaded_config["steps"]
        ]

        raise KeyError(
            f"Override keys {list(remaining_override_keys)} do not match any step in the saved configuration. "
            f"Available step keys: {available_keys}. "
            f"Make sure override keys match exact step class names or registry names."
        )

    @classmethod
    def _should_suggest_migration(cls, model_path: Path) -> bool:

        json_files = list(model_path.glob("*.json"))
        if len(json_files) == 0:
            return False


        for json_file in json_files:
            try:
                with open(json_file) as f:
                    config = json.load(f)

                if cls._is_processor_config(config):
                    return False

            except (json.JSONDecodeError, OSError):

                continue


        return True

    @classmethod
    def _is_processor_config(cls, config: dict) -> bool:


        if not isinstance(config.get("steps"), list):
            return False

        steps = config["steps"]
        if len(steps) == 0:
            return True


        for step in steps:
            if not isinstance(step, dict):
                return False
            if not ("class" in step or "registry_name" in step):
                return False

        return True

    @classmethod
    def _suggest_processor_migration(cls, model_path: str | Path, original_error: str) -> None:

        migration_command = (
            f"python src/lerobot/processor/migrate_policy_normalization.py --pretrained-path {model_path}"
        )

        raise ProcessorMigrationError(model_path, migration_command, original_error)

    def __len__(self) -> int:

        return len(self.steps)

    def __getitem__(self, idx: int | slice) -> ProcessorStep | DataProcessorPipeline[TInput, TOutput]:

        if isinstance(idx, slice):

            return DataProcessorPipeline(
                steps=self.steps[idx],
                name=self.name,
                to_transition=self.to_transition,
                to_output=self.to_output,
                before_step_hooks=self.before_step_hooks.copy(),
                after_step_hooks=self.after_step_hooks.copy(),
            )
        return self.steps[idx]

    def register_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):

        self.before_step_hooks.append(fn)

    def unregister_before_step_hook(self, fn: Callable[[int, EnvTransition], None]):

        try:
            self.before_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"Hook {fn} not found in before_step_hooks. Make sure to pass the exact same function reference."
            ) from None

    def register_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):

        self.after_step_hooks.append(fn)

    def unregister_after_step_hook(self, fn: Callable[[int, EnvTransition], None]):

        try:
            self.after_step_hooks.remove(fn)
        except ValueError:
            raise ValueError(
                f"Hook {fn} not found in after_step_hooks. Make sure to pass the exact same function reference."
            ) from None

    def reset(self):

        for step in self.steps:
            if hasattr(step, "reset"):
                step.reset()

    def __repr__(self) -> str:

        step_names = [step.__class__.__name__ for step in self.steps]

        if not step_names:
            steps_repr = "steps=0: []"
        elif len(step_names) <= 3:
            steps_repr = f"steps={len(step_names)}: [{', '.join(step_names)}]"
        else:

            displayed = f"{step_names[0]}, {step_names[1]}, ..., {step_names[-1]}"
            steps_repr = f"steps={len(step_names)}: [{displayed}]"

        parts = [f"name='{self.name}'", steps_repr]

        return f"DataProcessorPipeline({', '.join(parts)})"

    def __post_init__(self):

        for i, step in enumerate(self.steps):
            if not isinstance(step, ProcessorStep):
                raise TypeError(f"Step {i} ({type(step).__name__}) must inherit from ProcessorStep")

    def transform_features(
        self, initial_features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = deepcopy(initial_features)

        for _, step in enumerate(self.steps):
            out = step.transform_features(features)
            features = out
        return features


    def process_observation(self, observation: RobotObservation) -> RobotObservation:

        transition: EnvTransition = create_transition(observation=observation)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.OBSERVATION]

    def process_action(
        self, action: PolicyAction | RobotAction | EnvAction
    ) -> PolicyAction | RobotAction | EnvAction:

        transition: EnvTransition = create_transition(action=action)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.ACTION]

    def process_reward(self, reward: float | torch.Tensor) -> float | torch.Tensor:

        transition: EnvTransition = create_transition(reward=reward)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.REWARD]

    def process_done(self, done: bool | torch.Tensor) -> bool | torch.Tensor:

        transition: EnvTransition = create_transition(done=done)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.DONE]

    def process_truncated(self, truncated: bool | torch.Tensor) -> bool | torch.Tensor:

        transition: EnvTransition = create_transition(truncated=truncated)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.TRUNCATED]

    def process_info(self, info: dict[str, Any]) -> dict[str, Any]:

        transition: EnvTransition = create_transition(info=info)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.INFO]

    def process_complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:

        transition: EnvTransition = create_transition(complementary_data=complementary_data)
        transformed_transition = self._forward(transition)
        return transformed_transition[TransitionKey.COMPLEMENTARY_DATA]


RobotProcessorPipeline = DataProcessorPipeline[TInput, TOutput]
PolicyProcessorPipeline = DataProcessorPipeline[TInput, TOutput]


class ObservationProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def observation(self, observation: RobotObservation) -> RobotObservation:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(observation, dict):
            raise ValueError("ObservationProcessorStep requires an observation in the transition.")

        processed_observation = self.observation(observation.copy())
        new_transition[TransitionKey.OBSERVATION] = processed_observation
        return new_transition


class ActionProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def action(
        self, action: PolicyAction | RobotAction | EnvAction
    ) -> PolicyAction | RobotAction | EnvAction:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("ActionProcessorStep requires an action in the transition.")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class RobotActionProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def action(self, action: RobotAction) -> RobotAction:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, dict):
            raise ValueError(f"Action should be a RobotAction type (dict), but got {type(action)}")

        processed_action = self.action(action.copy())
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class PolicyActionProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def action(self, action: PolicyAction) -> PolicyAction:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type (tensor), but got {type(action)}")

        processed_action = self.action(action)
        new_transition[TransitionKey.ACTION] = processed_action
        return new_transition


class RewardProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def reward(self, reward) -> float | torch.Tensor:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        reward = new_transition.get(TransitionKey.REWARD)
        if reward is None:
            raise ValueError("RewardProcessorStep requires a reward in the transition.")

        processed_reward = self.reward(reward)
        new_transition[TransitionKey.REWARD] = processed_reward
        return new_transition


class DoneProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def done(self, done) -> bool | torch.Tensor:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        done = new_transition.get(TransitionKey.DONE)
        if done is None:
            raise ValueError("DoneProcessorStep requires a done flag in the transition.")

        processed_done = self.done(done)
        new_transition[TransitionKey.DONE] = processed_done
        return new_transition


class TruncatedProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def truncated(self, truncated) -> bool | torch.Tensor:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        truncated = new_transition.get(TransitionKey.TRUNCATED)
        if truncated is None:
            raise ValueError("TruncatedProcessorStep requires a truncated flag in the transition.")

        processed_truncated = self.truncated(truncated)
        new_transition[TransitionKey.TRUNCATED] = processed_truncated
        return new_transition


class InfoProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def info(self, info) -> dict[str, Any]:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        info = new_transition.get(TransitionKey.INFO)
        if info is None or not isinstance(info, dict):
            raise ValueError("InfoProcessorStep requires an info dictionary in the transition.")

        processed_info = self.info(info.copy())
        new_transition[TransitionKey.INFO] = processed_info
        return new_transition


class ComplementaryDataProcessorStep(ProcessorStep, ABC):


    @abstractmethod
    def complementary_data(self, complementary_data) -> dict[str, Any]:

        ...

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None or not isinstance(complementary_data, dict):
            raise ValueError("ComplementaryDataProcessorStep requires complementary data in the transition.")

        processed_complementary_data = self.complementary_data(complementary_data.copy())
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = processed_complementary_data
        return new_transition


class IdentityProcessorStep(ProcessorStep):


    def __call__(self, transition: EnvTransition) -> EnvTransition:

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        return features
