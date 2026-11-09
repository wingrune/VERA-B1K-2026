from dataclasses import dataclass
from typing import Any

import torch

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, PolicyAction, TransitionKey
from lerobot.utils.device_utils import get_safe_torch_device

from .pipeline import ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("device_processor")
@dataclass
class DeviceProcessorStep(ProcessorStep):


    device: str = "cpu"
    float_dtype: str | None = None

    DTYPE_MAPPING = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "half": torch.float16,
        "float": torch.float32,
        "double": torch.float64,
    }

    def __post_init__(self):

        self.tensor_device: torch.device = get_safe_torch_device(self.device)

        self.device = self.tensor_device.type
        self.non_blocking = "cuda" in str(self.device)


        if self.float_dtype is not None:
            if self.float_dtype not in self.DTYPE_MAPPING:
                raise ValueError(
                    f"Invalid float_dtype '{self.float_dtype}'. Available options: {list(self.DTYPE_MAPPING.keys())}"
                )
            self._target_float_dtype = self.DTYPE_MAPPING[self.float_dtype]
        else:
            self._target_float_dtype = None

    def _process_tensor(self, tensor: torch.Tensor) -> torch.Tensor:


        if tensor.is_cuda and self.tensor_device.type == "cuda":


            target_device = tensor.device
        else:


            target_device = self.tensor_device


        if target_device.type == "mps" and tensor.dtype == torch.float64:
            tensor = tensor.to(dtype=torch.float32)


        if tensor.device != target_device:
            tensor = tensor.to(target_device, non_blocking=self.non_blocking)


        if self._target_float_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=self._target_float_dtype)

        return tensor

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)

        if action is not None and not isinstance(action, PolicyAction):
            raise ValueError(f"If action is not None should be a PolicyAction type got {type(action)}")

        simple_tensor_keys = [
            TransitionKey.ACTION,
            TransitionKey.REWARD,
            TransitionKey.DONE,
            TransitionKey.TRUNCATED,
        ]

        dict_tensor_keys = [
            TransitionKey.OBSERVATION,
            TransitionKey.COMPLEMENTARY_DATA,
        ]


        for key in simple_tensor_keys:
            value = transition.get(key)
            if isinstance(value, torch.Tensor):
                new_transition[key] = self._process_tensor(value)


        for key in dict_tensor_keys:
            data_dict = transition.get(key)
            if data_dict is not None:
                new_data_dict = {
                    k: self._process_tensor(v) if isinstance(v, torch.Tensor) else v
                    for k, v in data_dict.items()
                }
                new_transition[key] = new_data_dict

        return new_transition

    def get_config(self) -> dict[str, Any]:

        return {"device": self.device, "float_dtype": self.float_dtype}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        return features
