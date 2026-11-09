from __future__ import annotations

from collections.abc import Sequence
from functools import singledispatch
from typing import Any

import numpy as np
import torch

from lerobot.types import EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey
from lerobot.utils.constants import ACTION, DONE, INFO, OBS_PREFIX, REWARD, TRUNCATED


@singledispatch
def to_tensor(
    value: Any,
    *,
    dtype: torch.dtype | None = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:

    raise TypeError(f"Unsupported type for tensor conversion: {type(value)}")


@to_tensor.register(torch.Tensor)
def _(value: torch.Tensor, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:

    if dtype is not None:
        value = value.to(dtype=dtype)
    if device is not None:
        value = value.to(device=device)
    return value


@to_tensor.register(np.ndarray)
def _(
    value: np.ndarray,
    *,
    dtype=torch.float32,
    device=None,
    **kwargs,
) -> torch.Tensor:


    if value.ndim == 0:

        scalar_value = value.item()
        return torch.tensor(scalar_value, dtype=dtype, device=device)


    tensor = torch.from_numpy(value)


    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)

    return tensor


@to_tensor.register(int)
@to_tensor.register(float)
@to_tensor.register(np.integer)
@to_tensor.register(np.floating)
def _(value, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:

    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(list)
@to_tensor.register(tuple)
def _(value: Sequence, *, dtype=torch.float32, device=None, **kwargs) -> torch.Tensor:

    return torch.tensor(value, dtype=dtype, device=device)


@to_tensor.register(dict)
def _(value: dict, *, device=None, **kwargs) -> dict:

    if not value:
        return {}

    result = {}
    for key, sub_value in value.items():
        if sub_value is None:
            continue

        if isinstance(sub_value, dict):

            result[key] = to_tensor(
                sub_value,
                device=device,
                **kwargs,
            )
            continue


        result[key] = to_tensor(
            sub_value,
            device=device,
            **kwargs,
        )
    return result


def from_tensor_to_numpy(x: torch.Tensor | Any) -> np.ndarray | float | int | Any:

    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.detach().cpu().numpy()
    return x


def _extract_complementary_data(batch: dict[str, Any]) -> dict[str, Any]:

    pad_keys = {k: v for k, v in batch.items() if "_is_pad" in k}
    task_key = {"task": batch["task"]} if "task" in batch else {}
    subtask_key = {"subtask": batch["subtask"]} if "subtask" in batch else {}
    index_key = {"index": batch["index"]} if "index" in batch else {}
    task_index_key = {"task_index": batch["task_index"]} if "task_index" in batch else {}
    episode_index_key = {"episode_index": batch["episode_index"]} if "episode_index" in batch else {}

    return {**pad_keys, **task_key, **subtask_key, **index_key, **task_index_key, **episode_index_key}


def create_transition(
    observation: RobotObservation | None = None,
    action: PolicyAction | RobotAction | None = None,
    reward: float = 0.0,
    done: bool = False,
    truncated: bool = False,
    info: dict[str, Any] | None = None,
    complementary_data: dict[str, Any] | None = None,
) -> EnvTransition:

    return {
        TransitionKey.OBSERVATION: observation,
        TransitionKey.ACTION: action,
        TransitionKey.REWARD: reward,
        TransitionKey.DONE: done,
        TransitionKey.TRUNCATED: truncated,
        TransitionKey.INFO: info if info is not None else {},
        TransitionKey.COMPLEMENTARY_DATA: complementary_data if complementary_data is not None else {},
    }


def robot_action_observation_to_transition(
    action_observation: tuple[RobotAction, RobotObservation],
) -> EnvTransition:

    if not isinstance(action_observation, tuple):
        raise ValueError("action_observation should be a tuple type with an action and observation")

    action, observation = action_observation

    if action is not None and not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type got {type(action)}")

    if observation is not None and not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation type got {type(observation)}")

    return create_transition(action=action, observation=observation)


def robot_action_to_transition(action: RobotAction) -> EnvTransition:

    if not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type got {type(action)}")
    return create_transition(action=action)


def observation_to_transition(observation: RobotObservation) -> EnvTransition:

    if not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation type got {type(observation)}")
    return create_transition(observation=observation)


def transition_to_robot_action(transition: EnvTransition) -> RobotAction:

    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, dict):
        raise ValueError(f"Action should be a RobotAction type (dict) got {type(action)}")
    return transition.get(TransitionKey.ACTION)


def transition_to_policy_action(transition: EnvTransition) -> PolicyAction:

    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    action = transition.get(TransitionKey.ACTION)
    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return action


def transition_to_observation(transition: EnvTransition) -> RobotObservation:

    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    observation = transition.get(TransitionKey.OBSERVATION)
    if not isinstance(observation, dict):
        raise ValueError(f"Observation should be a RobotObservation (dict) type got {type(observation)}")
    return observation


def policy_action_to_transition(action: PolicyAction) -> EnvTransition:

    if not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")
    return create_transition(action=action)


def batch_to_transition(batch: dict[str, Any]) -> EnvTransition:


    if not isinstance(batch, dict):
        raise ValueError(f"EnvTransition must be a dictionary. Got {type(batch).__name__}")

    action = batch.get(ACTION)
    if action is not None and not isinstance(action, PolicyAction):
        raise ValueError(f"Action should be a PolicyAction type got {type(action)}")


    observation_keys = {k: v for k, v in batch.items() if k.startswith(OBS_PREFIX)}
    complementary_data = _extract_complementary_data(batch)

    return create_transition(
        observation=observation_keys if observation_keys else None,
        action=batch.get(ACTION),
        reward=batch.get(REWARD, 0.0),
        done=batch.get(DONE, False),
        truncated=batch.get(TRUNCATED, False),
        info=batch.get("info", {}),
        complementary_data=complementary_data if complementary_data else None,
    )


def transition_to_batch(transition: EnvTransition) -> dict[str, Any]:

    if not isinstance(transition, dict):
        raise ValueError(f"Transition should be a EnvTransition type (dict) got {type(transition)}")

    batch = {
        ACTION: transition.get(TransitionKey.ACTION),
        REWARD: transition.get(TransitionKey.REWARD, 0.0),
        DONE: transition.get(TransitionKey.DONE, False),
        TRUNCATED: transition.get(TransitionKey.TRUNCATED, False),
        INFO: transition.get(TransitionKey.INFO, {}),
    }


    comp_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
    if comp_data:
        batch.update(comp_data)


    observation = transition.get(TransitionKey.OBSERVATION)
    if isinstance(observation, dict):
        batch.update(observation)

    return batch


def identity_transition(transition: EnvTransition) -> EnvTransition:

    return transition
