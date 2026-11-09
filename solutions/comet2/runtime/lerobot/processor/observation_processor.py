from dataclasses import dataclass

import einops
import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGE, OBS_IMAGES, OBS_STATE, OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="observation_processor")
class VanillaObservationProcessorStep(ObservationProcessorStep):


    def _process_single_image(self, img: np.ndarray) -> Tensor:


        img_tensor = torch.from_numpy(img)


        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)


        _, h, w, c = img_tensor.shape
        if not (c < h and c < w):
            raise ValueError(f"Expected channel-last images, but got shape {img_tensor.shape}")

        if img_tensor.dtype != torch.uint8:
            raise ValueError(f"Expected torch.uint8 images, but got {img_tensor.dtype}")


        img_tensor = einops.rearrange(img_tensor, "b h w c -> b c h w").contiguous()


        img_tensor = img_tensor.type(torch.float32) / 255.0

        return img_tensor

    def _process_observation(self, observation):


        processed_obs = observation.copy()

        if "pixels" in processed_obs:
            pixels = processed_obs.pop("pixels")

            if isinstance(pixels, dict):
                imgs = {f"{OBS_IMAGES}.{key}": img for key, img in pixels.items()}
            else:
                imgs = {OBS_IMAGE: pixels}

            for imgkey, img in imgs.items():
                processed_obs[imgkey] = self._process_single_image(img)

        if "environment_state" in processed_obs:
            env_state_np = processed_obs.pop("environment_state")
            env_state = torch.from_numpy(env_state_np).float()
            if env_state.dim() == 1:
                env_state = env_state.unsqueeze(0)
            processed_obs[OBS_ENV_STATE] = env_state

        if "agent_pos" in processed_obs:
            agent_pos_np = processed_obs.pop("agent_pos")
            agent_pos = torch.from_numpy(agent_pos_np).float()
            if agent_pos.dim() == 1:
                agent_pos = agent_pos.unsqueeze(0)
            processed_obs[OBS_STATE] = agent_pos

        return processed_obs

    def observation(self, observation):
        return self._process_observation(observation)

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:


        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {ft: {} for ft in features}

        exact_pairs = {
            "pixels": OBS_IMAGE,
            "environment_state": OBS_ENV_STATE,
            "agent_pos": OBS_STATE,
        }

        prefix_pairs = {
            "pixels.": f"{OBS_IMAGES}.",
        }


        for src_ft, bucket in features.items():
            for key, feat in list(bucket.items()):
                handled = False


                for old_prefix, new_prefix in prefix_pairs.items():
                    prefixed_old = f"{OBS_STR}.{old_prefix}"
                    if key.startswith(prefixed_old):
                        suffix = key[len(prefixed_old) :]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                    if key.startswith(old_prefix):
                        suffix = key[len(old_prefix) :]
                        new_key = f"{new_prefix}{suffix}"
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                if handled:
                    continue


                for old, new in exact_pairs.items():
                    if key == old or key == f"{OBS_STR}.{old}":
                        new_key = new
                        new_features[src_ft][new_key] = feat
                        handled = True
                        break

                if handled:
                    continue


                new_features[src_ft][key] = feat

        return new_features
