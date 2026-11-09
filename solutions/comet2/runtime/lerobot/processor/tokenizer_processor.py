from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, RobotObservation, TransitionKey
from lerobot.utils.constants import (
    ACTION_TOKEN_MASK,
    ACTION_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_ATTENTION_MASK,
    OBS_LANGUAGE_SUBTASK_TOKENS,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.utils.import_utils import _transformers_available

from .pipeline import ActionProcessorStep, ObservationProcessorStep, ProcessorStepRegistry


if TYPE_CHECKING or _transformers_available:
    from transformers import AutoProcessor, AutoTokenizer
else:
    AutoProcessor = None
    AutoTokenizer = None


@dataclass
@ProcessorStepRegistry.register(name="tokenizer_processor")
class TokenizerProcessorStep(ObservationProcessorStep):


    tokenizer_name: str | None = None
    tokenizer: Any | None = None
    max_length: int = 512
    task_key: str = "task"
    padding_side: str = "right"
    padding: str = "max_length"
    truncation: bool = True


    input_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):

        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use TokenizerProcessorStep."
            )

        if self.tokenizer is not None:

            self.input_tokenizer = self.tokenizer
        elif self.tokenizer_name is not None:
            if AutoTokenizer is None:
                raise ImportError("AutoTokenizer is not available")
            self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            raise ValueError(
                "Either 'tokenizer' or 'tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

    def get_task(self, transition: EnvTransition) -> list[str] | None:

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            raise ValueError("Complementary data is None so no task can be extracted from it")

        task = complementary_data[self.task_key]
        if task is None:
            raise ValueError("Task extracted from Complementary data is None")


        if isinstance(task, str):
            return [task]
        elif isinstance(task, (list, tuple)) and all(isinstance(t, str) for t in task):
            return list(task)

        return None

    def get_subtask(self, transition: EnvTransition) -> list[str] | None:

        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA)
        if complementary_data is None:
            return None

        subtask = complementary_data.get("subtask")
        if subtask is None:
            return None


        if isinstance(subtask, str):
            return [subtask]
        elif isinstance(subtask, list) and all(isinstance(t, str) for t in subtask):
            return subtask

        return None

    def observation(self, observation: RobotObservation) -> RobotObservation:

        task = self.get_task(self.transition)
        if task is None:
            raise ValueError("Task cannot be None")


        tokenized_prompt = self._tokenize_text(task)


        target_device = self._detect_device(self.transition)


        if target_device is not None:
            tokenized_prompt = {
                k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                for k, v in tokenized_prompt.items()
            }


        new_observation = dict(observation)


        new_observation[OBS_LANGUAGE_TOKENS] = tokenized_prompt["input_ids"]
        new_observation[OBS_LANGUAGE_ATTENTION_MASK] = tokenized_prompt["attention_mask"].to(dtype=torch.bool)


        subtask = self.get_subtask(self.transition)
        if subtask is not None:
            tokenized_subtask = self._tokenize_text(subtask)


            if target_device is not None:
                tokenized_subtask = {
                    k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                    for k, v in tokenized_subtask.items()
                }


            new_observation[OBS_LANGUAGE_SUBTASK_TOKENS] = tokenized_subtask["input_ids"]
            new_observation[OBS_LANGUAGE_SUBTASK_ATTENTION_MASK] = tokenized_subtask["attention_mask"].to(
                dtype=torch.bool
            )

        return new_observation

    def _detect_device(self, transition: EnvTransition) -> torch.device | None:


        observation = transition.get(TransitionKey.OBSERVATION)
        if observation:
            for value in observation.values():
                if isinstance(value, torch.Tensor):
                    return value.device


        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            return action.device

        return None

    def _tokenize_text(self, text: str | list[str]) -> dict[str, torch.Tensor]:

        return self.input_tokenizer(
            text,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
            padding_side=self.padding_side,
            return_tensors="pt",
        )

    def get_config(self) -> dict[str, Any]:

        config = {
            "max_length": self.max_length,
            "task_key": self.task_key,
            "padding_side": self.padding_side,
            "padding": self.padding,
            "truncation": self.truncation,
        }


        if self.tokenizer_name is not None and self.tokenizer is None:
            config["tokenizer_name"] = self.tokenizer_name

        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:


        if OBS_LANGUAGE_TOKENS not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_TOKENS] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )


        if OBS_LANGUAGE_ATTENTION_MASK not in features[PipelineFeatureType.OBSERVATION]:
            features[PipelineFeatureType.OBSERVATION][OBS_LANGUAGE_ATTENTION_MASK] = PolicyFeature(
                type=FeatureType.LANGUAGE, shape=(self.max_length,)
            )

        return features


@dataclass
@ProcessorStepRegistry.register(name="action_tokenizer_processor")
class ActionTokenizerProcessorStep(ActionProcessorStep):


    action_tokenizer_name: str | None = None
    action_tokenizer_input_object: Any | None = None
    trust_remote_code: bool = True
    max_action_tokens: int = 256
    fast_skip_tokens: int = 128
    paligemma_tokenizer_name: str = "google/paligemma-3b-pt-224"

    action_tokenizer: Any = field(default=None, init=False, repr=False)
    _paligemma_tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):

        if not _transformers_available:
            raise ImportError(
                "The 'transformers' library is not installed. "
                "Please install it with `pip install 'lerobot[transformers-dep]'` to use ActionTokenizerProcessorStep."
            )

        if self.action_tokenizer_input_object is not None:
            self.action_tokenizer = self.action_tokenizer_input_object

        elif self.action_tokenizer_name is not None:
            if AutoProcessor is None:
                raise ImportError("AutoProcessor is not available")
            self.action_tokenizer = AutoProcessor.from_pretrained(
                self.action_tokenizer_name, trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(
                "Either 'action_tokenizer' or 'action_tokenizer_name' must be provided. "
                "Pass a tokenizer object directly or a tokenizer name to auto-load."
            )

        self._paligemma_tokenizer = AutoTokenizer.from_pretrained(
            self.paligemma_tokenizer_name,
            trust_remote_code=self.trust_remote_code,
            add_eos_token=True,
            add_bos_token=False,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is None:

            return new_transition


        tokens, mask = self._tokenize_action(action)


        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if complementary_data is None:
            complementary_data = {}
        complementary_data[ACTION_TOKEN_MASK] = mask
        complementary_data[ACTION_TOKENS] = tokens
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def _act_tokens_to_paligemma_tokens(self, tokens: torch.Tensor) -> torch.Tensor:

        return self._paligemma_tokenizer.vocab_size - 1 - self.fast_skip_tokens - tokens

    def _tokenize_action(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if action is None:
            raise ValueError("Action cannot be None")


        device = action.device if isinstance(action, torch.Tensor) else None


        single_sample = action.dim() == 1
        if single_sample:
            action = action.unsqueeze(0)

        batch_size = action.shape[0]


        tokens_list = []
        masks_list = []

        for i in range(batch_size):

            action_cpu = action[i : i + 1].cpu()
            tokens = self.action_tokenizer(action_cpu)


            if isinstance(tokens, list) or not isinstance(tokens, torch.Tensor):
                tokens = torch.tensor(tokens, dtype=torch.long, device=action.device)
            else:

                tokens = tokens.to(device=action.device)


            if tokens.dim() > 1:
                tokens = tokens.flatten()

            bos_id = self._paligemma_tokenizer.bos_token_id

            tokens = torch.cat(
                [
                    torch.tensor([bos_id], device=action.device),
                    torch.tensor(
                        self._paligemma_tokenizer.encode("Action: ", add_special_tokens=False),
                        device=action.device,
                    ),
                    self._act_tokens_to_paligemma_tokens(tokens),
                    torch.tensor(self._paligemma_tokenizer.encode("|"), device=action.device),
                ]
            )


            if len(tokens) > self.max_action_tokens:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self.max_action_tokens}), truncating. "
                    "Consider increasing the `max_action_tokens` in your model config if this happens frequently."
                )
                tokens = tokens[: self.max_action_tokens]
                mask = torch.ones(self.max_action_tokens, dtype=torch.bool, device=action.device)
            else:
                mask = torch.cat(
                    [
                        torch.ones(len(tokens), dtype=torch.bool, device=action.device),
                        torch.zeros(
                            self.max_action_tokens - len(tokens), dtype=torch.bool, device=action.device
                        ),
                    ]
                )

                tokens = torch.nn.functional.pad(tokens, (0, self.max_action_tokens - len(tokens)), value=0)

            tokens_list.append(tokens)
            masks_list.append(mask)


        tokens_batch = torch.stack(tokens_list, dim=0)
        masks_batch = torch.stack(masks_list, dim=0)


        if single_sample:
            tokens_batch = tokens_batch.squeeze(0)
            masks_batch = masks_batch.squeeze(0)


        if device is not None:
            tokens_batch = tokens_batch.to(device)
            masks_batch = masks_batch.to(device)

        return tokens_batch, masks_batch

    def action(self, action: torch.Tensor) -> torch.Tensor:

        tokens, _ = self._tokenize_action(action)
        return tokens

    def get_config(self) -> dict[str, Any]:

        config = {
            "trust_remote_code": self.trust_remote_code,
            "max_action_tokens": self.max_action_tokens,
        }


        if self.action_tokenizer_name is not None and self.action_tokenizer_input_object is None:
            config["action_tokenizer_name"] = self.action_tokenizer_name

        return config

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        return features
