from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ComplementaryDataProcessorStep, ProcessorStepRegistry


@ProcessorStepRegistry.register(name="smolvla_new_line_processor")
class NewLineTaskProcessorStep(ComplementaryDataProcessorStep):


    def complementary_data(self, complementary_data):
        if "task" not in complementary_data:
            return complementary_data

        task = complementary_data["task"]
        if task is None:
            return complementary_data

        new_complementary_data = dict(complementary_data)


        if isinstance(task, str):

            if not task.endswith("\n"):
                new_complementary_data["task"] = f"{task}\n"
        elif isinstance(task, list) and all(isinstance(t, str) for t in task):

            new_complementary_data["task"] = [t if t.endswith("\n") else f"{t}\n" for t in task]


        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
