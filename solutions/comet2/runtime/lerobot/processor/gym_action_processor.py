from dataclasses import dataclass

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvAction, EnvTransition, PolicyAction, TransitionKey

from .converters import to_tensor
from .hil_processor import TELEOP_ACTION_KEY
from .pipeline import ActionProcessorStep, ProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register("torch2numpy_action_processor")
@dataclass
class Torch2NumpyActionProcessorStep(ActionProcessorStep):


    squeeze_batch_dim: bool = True

    def action(self, action: PolicyAction) -> EnvAction:
        if not isinstance(action, PolicyAction):
            raise TypeError(
                f"Expected PolicyAction or None, got {type(action).__name__}. "
                "Use appropriate processor for non-tensor actions."
            )

        numpy_action = action.detach().cpu().numpy()


        if (
            self.squeeze_batch_dim
            and numpy_action.shape
            and len(numpy_action.shape) > 1
            and numpy_action.shape[0] == 1
        ):
            numpy_action = numpy_action.squeeze(0)

        return numpy_action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("numpy2torch_action_processor")
@dataclass
class Numpy2TorchActionProcessorStep(ProcessorStep):


    def __call__(self, transition: EnvTransition) -> EnvTransition:

        self._current_transition = transition.copy()
        new_transition = self._current_transition

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            if not isinstance(action, EnvAction):
                raise TypeError(
                    f"Expected np.ndarray or None, got {type(action).__name__}. "
                    "Use appropriate processor for non-tensor actions."
                )
            torch_action = to_tensor(action, dtype=None)
            new_transition[TransitionKey.ACTION] = torch_action

        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        if TELEOP_ACTION_KEY in complementary_data:
            teleop_action = complementary_data[TELEOP_ACTION_KEY]
            if isinstance(teleop_action, EnvAction):
                complementary_data[TELEOP_ACTION_KEY] = to_tensor(teleop_action)
            new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
