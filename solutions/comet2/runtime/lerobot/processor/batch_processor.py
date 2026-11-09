from dataclasses import dataclass, field

from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, PolicyAction
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE

from .pipeline import (
    ComplementaryDataProcessorStep,
    ObservationProcessorStep,
    PolicyActionProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TransitionKey,
)


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_action")
class AddBatchDimensionActionStep(PolicyActionProcessorStep):


    def action(self, action: PolicyAction) -> PolicyAction:

        if action.dim() != 1:
            return action
        return action.unsqueeze(0)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_observation")
class AddBatchDimensionObservationStep(ObservationProcessorStep):


    def observation(self, observation: dict[str, Tensor]) -> dict[str, Tensor]:


        for state_key in [OBS_STATE, OBS_ENV_STATE]:
            if state_key in observation:
                state_value = observation[state_key]
                if isinstance(state_value, Tensor) and state_value.dim() == 1:
                    observation[state_key] = state_value.unsqueeze(0)


        if OBS_IMAGE in observation:
            image_value = observation[OBS_IMAGE]
            if isinstance(image_value, Tensor) and image_value.dim() == 3:
                observation[OBS_IMAGE] = image_value.unsqueeze(0)


        for key, value in observation.items():
            if key.startswith(f"{OBS_IMAGES}.") and isinstance(value, Tensor) and value.dim() == 3:
                observation[key] = value.unsqueeze(0)
        return observation

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor_complementary_data")
class AddBatchDimensionComplementaryDataStep(ComplementaryDataProcessorStep):


    def complementary_data(self, complementary_data: dict) -> dict:


        if "task" in complementary_data:
            task_value = complementary_data["task"]
            if isinstance(task_value, str):
                complementary_data["task"] = [task_value]


        if "index" in complementary_data:
            index_value = complementary_data["index"]
            if isinstance(index_value, Tensor) and index_value.dim() == 0:
                complementary_data["index"] = index_value.unsqueeze(0)


        if "task_index" in complementary_data:
            task_index_value = complementary_data["task_index"]
            if isinstance(task_index_value, Tensor) and task_index_value.dim() == 0:
                complementary_data["task_index"] = task_index_value.unsqueeze(0)
        return complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        return features


@dataclass
@ProcessorStepRegistry.register(name="to_batch_processor")
class AddBatchDimensionProcessorStep(ProcessorStep):


    to_batch_action_processor: AddBatchDimensionActionStep = field(
        default_factory=AddBatchDimensionActionStep
    )
    to_batch_observation_processor: AddBatchDimensionObservationStep = field(
        default_factory=AddBatchDimensionObservationStep
    )
    to_batch_complementary_data_processor: AddBatchDimensionComplementaryDataStep = field(
        default_factory=AddBatchDimensionComplementaryDataStep
    )

    def __call__(self, transition: EnvTransition) -> EnvTransition:

        if transition[TransitionKey.ACTION] is not None:
            transition = self.to_batch_action_processor(transition)
        if transition[TransitionKey.OBSERVATION] is not None:
            transition = self.to_batch_observation_processor(transition)
        if transition[TransitionKey.COMPLEMENTARY_DATA] is not None:
            transition = self.to_batch_complementary_data_processor(transition)
        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:


        return features
