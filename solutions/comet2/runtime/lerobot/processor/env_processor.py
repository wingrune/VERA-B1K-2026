from dataclasses import dataclass

import torch

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.utils.constants import OBS_IMAGES, OBS_PREFIX, OBS_STATE, OBS_STR

from .pipeline import ObservationProcessorStep, ProcessorStepRegistry


@dataclass
@ProcessorStepRegistry.register(name="libero_processor")
class LiberoProcessorStep(ObservationProcessorStep):


    def _process_observation(self, observation):

        processed_obs = observation.copy()
        for key in list(processed_obs.keys()):
            if key.startswith(f"{OBS_IMAGES}."):
                img = processed_obs[key]


                img = torch.flip(img, dims=[2, 3])

                processed_obs[key] = img

        observation_robot_state_str = OBS_PREFIX + "robot_state"
        if observation_robot_state_str in processed_obs:
            robot_state = processed_obs.pop(observation_robot_state_str)


            eef_pos = robot_state["eef"]["pos"]
            eef_quat = robot_state["eef"]["quat"]
            gripper_qpos = robot_state["gripper"]["qpos"]


            eef_axisangle = self._quat2axisangle(eef_quat)

            state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)


            state = state.float()
            if state.dim() == 1:
                state = state.unsqueeze(0)

            processed_obs[OBS_STATE] = state
        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        new_features: dict[PipelineFeatureType, dict[str, PolicyFeature]] = {}


        for ft, feats in features.items():
            if ft != FeatureType.STATE:
                new_features[ft] = feats.copy()


        state_feats = {}


        state_feats[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(8,),
        )

        new_features[FeatureType.STATE] = state_feats

        return new_features

    def observation(self, observation):
        return self._process_observation(observation)

    def _quat2axisangle(self, quat: torch.Tensor) -> torch.Tensor:


        if not isinstance(quat, torch.Tensor):
            raise TypeError(f"_quat2axisangle expected a torch.Tensor, got {type(quat)}")

        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"_quat2axisangle expected shape (B, 4), got {tuple(quat.shape)}")

        quat = quat.to(dtype=torch.float32)
        device = quat.device
        batch_size = quat.shape[0]

        w = quat[:, 3].clamp(-1.0, 1.0)

        den = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))

        result = torch.zeros((batch_size, 3), device=device)

        mask = den > 1e-10

        if mask.any():
            angle = 2.0 * torch.acos(w[mask])
            axis = quat[mask, :3] / den[mask].unsqueeze(1)
            result[mask] = axis * angle.unsqueeze(1)

        return result


@dataclass
@ProcessorStepRegistry.register(name="isaaclab_arena_processor")
class IsaaclabArenaProcessorStep(ObservationProcessorStep):


    state_keys: tuple[str, ...]


    camera_keys: tuple[str, ...]

    def _process_observation(self, observation):

        processed_obs = {}

        if f"{OBS_STR}.camera_obs" in observation:
            camera_obs = observation[f"{OBS_STR}.camera_obs"]

            for cam_name, img in camera_obs.items():
                if cam_name not in self.camera_keys:
                    continue

                img = img.permute(0, 3, 1, 2).contiguous()
                if img.dtype == torch.uint8:
                    img = img.float() / 255.0
                elif img.dtype != torch.float32:
                    img = img.float()

                processed_obs[f"{OBS_IMAGES}.{cam_name}"] = img


        if f"{OBS_STR}.policy" in observation:
            policy_obs = observation[f"{OBS_STR}.policy"]


            state_components = []
            for key in self.state_keys:
                if key in policy_obs:
                    component = policy_obs[key]

                    if component.dim() > 2:
                        batch_size = component.shape[0]
                        component = component.view(batch_size, -1)
                    state_components.append(component)

            if state_components:
                state = torch.cat(state_components, dim=-1)
                state = state.float()
                processed_obs[OBS_STATE] = state

        return processed_obs

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:

        return features

    def observation(self, observation):
        return self._process_observation(observation)
