from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from lerobot.configs import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, PolicyAction, TransitionKey

if TYPE_CHECKING:
    from lerobot.datasets import LeRobotDataset

from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry, RobotObservation


@dataclass
class _NormalizationMixin:


    features: dict[str, PolicyFeature]
    norm_map: dict[FeatureType, NormalizationMode]
    stats: dict[str, dict[str, Any]] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    eps: float = 1e-8
    normalize_observation_keys: set[str] | None = None

    _tensor_stats: dict[str, dict[str, Tensor]] = field(default_factory=dict, init=False, repr=False)
    _stats_explicitly_provided: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):


        self._stats_explicitly_provided = self.stats is not None and bool(self.stats)

        if self.features:
            first_val = next(iter(self.features.values()))
            if isinstance(first_val, dict):
                reconstructed = {}
                for key, ft_dict in self.features.items():
                    reconstructed[key] = PolicyFeature(
                        type=FeatureType(ft_dict["type"]), shape=tuple(ft_dict["shape"])
                    )
                self.features = reconstructed


        if self.norm_map and all(isinstance(k, str) for k in self.norm_map):
            reconstructed = {}
            for ft_type_str, norm_mode_str in self.norm_map.items():
                reconstructed[FeatureType(ft_type_str)] = NormalizationMode(norm_mode_str)
            self.norm_map = reconstructed


        self.stats = self.stats or {}
        if self.dtype is None:
            self.dtype = torch.float32
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        self._reshape_visual_stats()

    def _reshape_visual_stats(self) -> None:

        for key, feature in self.features.items():
            if feature.type != FeatureType.VISUAL:
                continue
            if key not in self._tensor_stats:
                continue
            for stat_name, stat_tensor in self._tensor_stats[key].items():
                if not isinstance(stat_tensor, Tensor) or stat_tensor.ndim != 1:
                    continue
                self._tensor_stats[key][stat_name] = stat_tensor.reshape(-1, 1, 1)

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> _NormalizationMixin:

        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
        self._reshape_visual_stats()
        return self

    def state_dict(self) -> dict[str, Tensor]:

        flat: dict[str, Tensor] = {}
        for key, sub in self._tensor_stats.items():
            for stat_name, tensor in sub.items():
                flat[f"{key}.{stat_name}"] = tensor.cpu()
        return flat

    def load_state_dict(self, state: dict[str, Tensor]) -> None:


        if self._stats_explicitly_provided and self.stats is not None:


            self._tensor_stats = to_tensor(self.stats, device=self.device, dtype=self.dtype)
            self._reshape_visual_stats()
            return


        self._tensor_stats.clear()
        for flat_key, tensor in state.items():
            key, stat_name = flat_key.rsplit(".", 1)

            self._tensor_stats.setdefault(key, {})[stat_name] = tensor.to(
                dtype=torch.float32, device=self.device
            )
        self._reshape_visual_stats()


        self.stats = {}
        for key, tensor_dict in self._tensor_stats.items():
            self.stats[key] = {}
            for stat_name, tensor in tensor_dict.items():

                self.stats[key][stat_name] = from_tensor_to_numpy(tensor)

    def get_config(self) -> dict[str, Any]:

        config = {
            "eps": self.eps,
            "features": {
                key: {"type": ft.type.value, "shape": ft.shape} for key, ft in self.features.items()
            },
            "norm_map": {ft_type.value: norm_mode.value for ft_type, norm_mode in self.norm_map.items()},
        }
        if self.normalize_observation_keys is not None:
            config["normalize_observation_keys"] = sorted(self.normalize_observation_keys)
        return config

    def _normalize_observation(self, observation: RobotObservation, inverse: bool) -> dict[str, Tensor]:

        new_observation = dict(observation)
        for key, feature in self.features.items():
            if self.normalize_observation_keys is not None and key not in self.normalize_observation_keys:
                continue
            if feature.type != FeatureType.ACTION and key in new_observation:

                tensor = torch.as_tensor(new_observation[key])
                new_observation[key] = self._apply_transform(tensor, key, feature.type, inverse=inverse)
        return new_observation

    def _normalize_action(self, action: Tensor, inverse: bool) -> Tensor:


        processed_action = self._apply_transform(action, ACTION, FeatureType.ACTION, inverse=inverse)
        return processed_action

    def _apply_transform(
        self, tensor: Tensor, key: str, feature_type: FeatureType, *, inverse: bool = False
    ) -> Tensor:

        norm_mode = self.norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if norm_mode == NormalizationMode.IDENTITY or key not in self._tensor_stats:
            return tensor

        if norm_mode not in (
            NormalizationMode.MEAN_STD,
            NormalizationMode.MIN_MAX,
            NormalizationMode.QUANTILES,
            NormalizationMode.QUANTILE10,
        ):
            raise ValueError(f"Unsupported normalization mode: {norm_mode}")


        if self._tensor_stats and key in self._tensor_stats:
            first_stat = next(iter(self._tensor_stats[key].values()))
            if first_stat.device != tensor.device or first_stat.dtype != tensor.dtype:
                self.to(device=tensor.device, dtype=tensor.dtype)

        stats = self._tensor_stats[key]

        if norm_mode == NormalizationMode.MEAN_STD:
            mean = stats.get("mean", None)
            std = stats.get("std", None)
            if mean is None or std is None:
                raise ValueError(
                    "MEAN_STD normalization mode requires mean and std stats, please update the dataset with the correct stats"
                )

            mean, std = stats["mean"], stats["std"]

            denom = std + self.eps
            if inverse:
                return tensor * std + mean
            return (tensor - mean) / denom

        if norm_mode == NormalizationMode.MIN_MAX:
            min_val = stats.get("min", None)
            max_val = stats.get("max", None)
            if min_val is None or max_val is None:
                raise ValueError(
                    "MIN_MAX normalization mode requires min and max stats, please update the dataset with the correct stats"
                )

            min_val, max_val = stats["min"], stats["max"]
            denom = max_val - min_val


            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:

                return (tensor + 1) / 2 * denom + min_val

            return 2 * (tensor - min_val) / denom - 1

        if norm_mode == NormalizationMode.QUANTILES:
            q01 = stats.get("q01", None)
            q99 = stats.get("q99", None)
            if q01 is None or q99 is None:
                raise ValueError(
                    "QUANTILES normalization mode requires q01 and q99 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                )

            denom = q99 - q01

            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q01
            return 2.0 * (tensor - q01) / denom - 1.0

        if norm_mode == NormalizationMode.QUANTILE10:
            q10 = stats.get("q10", None)
            q90 = stats.get("q90", None)
            if q10 is None or q90 is None:
                raise ValueError(
                    "QUANTILE10 normalization mode requires q10 and q90 stats, please update the dataset with the correct stats using the `augment_dataset_quantile_stats.py` script"
                )

            denom = q90 - q10

            denom = torch.where(
                denom == 0, torch.tensor(self.eps, device=tensor.device, dtype=tensor.dtype), denom
            )
            if inverse:
                return (tensor + 1.0) * denom / 2.0 + q10
            return 2.0 * (tensor - q10) / denom - 1.0


        return tensor


@dataclass
@ProcessorStepRegistry.register(name="normalizer_processor")
class NormalizerProcessorStep(_NormalizationMixin, ProcessorStep):


    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        normalize_observation_keys: set[str] | None = None,
        eps: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> NormalizerProcessorStep:

        return cls(
            features=features,
            norm_map=norm_map,
            stats=dataset.meta.stats,
            normalize_observation_keys=normalize_observation_keys,
            eps=eps,
            device=device,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()


        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(
                observation, inverse=False
            )


        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            return new_transition

        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=False)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register(name="unnormalizer_processor")
class UnnormalizerProcessorStep(_NormalizationMixin, ProcessorStep):


    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        features: dict[str, PolicyFeature],
        norm_map: dict[FeatureType, NormalizationMode],
        *,
        device: torch.device | str | None = None,
    ) -> UnnormalizerProcessorStep:

        return cls(features=features, norm_map=norm_map, stats=dataset.meta.stats, device=device)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        new_transition = transition.copy()


        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is not None:
            new_transition[TransitionKey.OBSERVATION] = self._normalize_observation(observation, inverse=True)


        action = new_transition.get(TransitionKey.ACTION)

        if action is None:
            return new_transition
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        new_transition[TransitionKey.ACTION] = self._normalize_action(action, inverse=True)

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def hotswap_stats(
    policy_processor: PolicyProcessorPipeline, stats: dict[str, dict[str, Any]]
) -> PolicyProcessorPipeline:

    rp = deepcopy(policy_processor)
    for step in rp.steps:
        if isinstance(step, _NormalizationMixin):
            step.stats = stats

            step._tensor_stats = to_tensor(stats, device=step.device, dtype=step.dtype)
    return rp
