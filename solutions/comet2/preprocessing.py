from dataclasses import dataclass
import numpy as np
from openpi_client.image_tools import resize_with_pad

PROPRIO_KEY = 'robot_r1::proprio'
CAMERA_KEYS = (
    'robot_r1::robot_r1:zed_link:Camera:0::rgb',
    'robot_r1::robot_r1:left_realsense_link:Camera:0::rgb',
    'robot_r1::robot_r1:right_realsense_link:Camera:0::rgb',
)
MODEL_CAMERA_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')

STATE_GROUPS = ((list(range(0, 3)), False), (list(range(53, 57)), False),
                (list(range(3, 10)), False), (list(range(24, 26)), True),
                (list(range(28, 35)), False), (list(range(49, 51)), True))

def extract_state_from_proprio(proprio):
    proprio = np.asarray(proprio)
    if proprio.shape[-1] != 61:
        raise ValueError(f'Expected R1Pro 61D proprio, got {proprio.shape}')
    return np.concatenate([proprio[..., ids].sum(axis=-1, keepdims=True) if eef
                           else proprio[..., ids] for ids, eef in STATE_GROUPS], axis=-1)

@dataclass(frozen=True)
class ObservationProcessor:
    image_size: tuple[int, int] = (224, 224)

    def __call__(self, obs, prompt):
        proprio = np.asarray(obs[PROPRIO_KEY])
        if proprio.ndim == 1:
            proprio = proprio[None, :]
        if proprio.shape != (1, 61):
            raise ValueError('Use --num-envs 1: independent policy state per episode')
        cameras = []
        for key in CAMERA_KEYS:
            image = np.asarray(obs[key])[..., :3]
            if image.ndim == 3:
                image = image[None, ...]
            if image.shape[0] != 1:
                raise ValueError('Camera and proprio batches must agree')
            cameras.append(resize_with_pad(image, *self.image_size))
        images = np.stack(cameras, axis=1)
        return {'observation/image_0': images[0, 0], 'observation/image_1': images[0, 1],
                'observation/image_2': images[0, 2], 'observation/state': proprio[0], 'prompt': prompt}

def parse_rgb(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image

@dataclass(frozen=True)
class B1KInputs:

    def __call__(self, data):
        result = {'state': extract_state_from_proprio(data['observation/state']),
                  'image': {name: parse_rgb(data[f'observation/image_{i}'])
                            for i, name in enumerate(MODEL_CAMERA_KEYS)},
                  'image_mask': {name: np.True_ for name in MODEL_CAMERA_KEYS}}
        for key in ('actions', 'prompt'):
            if key in data:
                result[key] = data[key]
        return result

@dataclass(frozen=True)
class B1KOutputs:
    action_dim: int = 23
    def __call__(self, data):
        return {'actions': np.asarray(data['actions'][:, :self.action_dim])}

@dataclass(frozen=True)
class CheckpointNormalize:

    norm_stats: object
    def __call__(self, data):
        from openpi.transforms import Normalize
        return Normalize(self.norm_stats, use_quantiles=False)(data)

@dataclass(frozen=True)
class CheckpointUnnormalize:
    norm_stats: object
    def __call__(self, data):
        from openpi.transforms import Unnormalize
        return Unnormalize(self.norm_stats, use_quantiles=False)(data)
