from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs import PipelineFeatureType, PolicyFeature

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="rename_observations_processor")
class RenameObservationsProcessorStep(ObservationProcessorStep):


    rename_map: dict[str, str] = field(default_factory=dict)

    def observation(self, observation):
        processed_obs = {}
        for key, value in observation.items():
            if key in self.rename_map:
                processed_obs[self.rename_map[key]] = value
            else:
                processed_obs[key] = value

        return processed_obs

    def get_config(self) -> dict[str, Any]:
        return {"rename_map": self.rename_map}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = features.copy()
        new_features[PipelineFeatureType.OBSERVATION] = {
            self.rename_map.get(k, k): v for k, v in features[PipelineFeatureType.OBSERVATION].items()
        }
        return new_features


def rename_stats(stats: dict[str, dict[str, Any]], rename_map: dict[str, str]) -> dict[str, dict[str, Any]]:

    if not stats:
        return {}
    renamed: dict[str, dict[str, Any]] = {}
    for old_key, sub_stats in stats.items():
        new_key = rename_map.get(old_key, old_key)
        renamed[new_key] = deepcopy(sub_stats) if sub_stats is not None else {}
    return renamed
