import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torchvision.transforms.functional as F

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.utils import TeleopEvents

if TYPE_CHECKING:
    from lerobot.teleoperators.teleoperator import Teleoperator

from lerobot.types import EnvTransition, PolicyAction, TransitionKey

from .pipeline import (
    ComplementaryDataProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"


@runtime_checkable
class HasTeleopEvents(Protocol):


    def get_teleop_events(self) -> dict[str, Any]:

        ...


TeleopWithEvents = TypeVar("TeleopWithEvents", bound="Teleoperator")


def _check_teleop_with_events(teleop: "Teleoperator") -> None:

    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ComplementaryDataProcessorStep):


    teleop_device: "Teleoperator"

    def complementary_data(self, complementary_data: dict) -> dict:

        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):


    teleop_device: TeleopWithEvents

    def __post_init__(self):

        _check_teleop_with_events(self.teleop_device)

    def info(self, info: dict) -> dict:

        new_info = dict(info)

        teleop_events = self.teleop_device.get_teleop_events()
        new_info.update(teleop_events)
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):


    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:

        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)


        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device

            if device.type == "mps":
                image = image.cpu()

            if self.crop_params_dict is not None and key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:

        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):


    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated: bool) -> bool:

        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True

        return truncated

    def get_config(self) -> dict[str, Any]:

        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:

        self.current_step = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("gym_hil_adapter_processor")
class GymHILAdapterProcessorStep(ProcessorStep):


    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        if TELEOP_ACTION_KEY in info:
            complementary_data[TELEOP_ACTION_KEY] = info[TELEOP_ACTION_KEY]

        if DISCRETE_PENALTY_KEY in info:
            complementary_data[DISCRETE_PENALTY_KEY] = info[DISCRETE_PENALTY_KEY]

        if "is_intervention" in info:
            info[TeleopEvents.IS_INTERVENTION] = info["is_intervention"]

        transition[TransitionKey.INFO] = info
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ProcessorStep):


    penalty: float = -0.02
    max_gripper_pos: float = 30.0
    open_threshold: float = 0.1
    closed_threshold: float = 0.9

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        raw_joint_positions = complementary_data.get("raw_joint_positions")
        if raw_joint_positions is None:
            return new_transition

        current_gripper_pos = raw_joint_positions.get(f"{GRIPPER_KEY}.pos", None)
        if current_gripper_pos is None:
            return new_transition


        if action is None:
            return new_transition


        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos


        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos


        is_open = gripper_state_normalized < self.open_threshold
        is_closed = gripper_state_normalized > self.closed_threshold
        cmd_close = gripper_action_normalized > self.closed_threshold
        cmd_open = gripper_action_normalized < self.open_threshold
        gripper_penalty_bool = (is_open and cmd_close) or (is_closed and cmd_open)

        gripper_penalty = self.penalty * int(gripper_penalty_bool)


        new_complementary_data = dict(complementary_data)
        new_complementary_data[DISCRETE_PENALTY_KEY] = gripper_penalty
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:

        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
            "open_threshold": self.open_threshold,
            "closed_threshold": self.closed_threshold,
        }

    def reset(self) -> None:

        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):


    use_gripper: bool = False
    terminate_on_success: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")


        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success = info.get(TeleopEvents.SUCCESS, False)
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()


        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict):

                action_list = [
                    teleop_action.get("delta_x", 0.0),
                    teleop_action.get("delta_y", 0.0),
                    teleop_action.get("delta_z", 0.0),
                ]
                if self.use_gripper:
                    action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
            elif isinstance(teleop_action, np.ndarray):
                action_list = teleop_action.tolist()
            else:
                action_list = teleop_action

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor


        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)


        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info


        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:

        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):


    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):

        if self.pretrained_path is not None:
            from lerobot.rewards.classifier.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition


        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition


        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(images, threshold=self.success_threshold)

        classifier_frequency = 1 / (time.perf_counter() - start_time)


        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True


        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated


        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:

        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
