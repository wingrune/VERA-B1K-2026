from typing import Any

import numpy as np

from lerobot.configs import FeatureType, PolicyFeature

from .constants import ACTION, DEFAULT_FEATURES, OBS_ENV_STATE, OBS_STR


def _validate_feature_names(features: dict[str, dict]) -> None:

    invalid_features = {name: ft for name, ft in features.items() if "/" in name}
    if invalid_features:
        raise ValueError(f"Feature names should not contain '/'. Found '/' in '{invalid_features}'.")


def hw_to_dataset_features(
    hw_features: dict[str, type | tuple], prefix: str, use_video: bool = True
) -> dict[str, dict]:

    features = {}
    joint_fts = {
        key: ftype
        for key, ftype in hw_features.items()
        if ftype is float or (isinstance(ftype, PolicyFeature) and ftype.type != FeatureType.VISUAL)
    }

    cam_fts = {key: shape for key, shape in hw_features.items() if isinstance(shape, tuple)}

    if joint_fts and prefix == ACTION:
        features[prefix] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    if joint_fts and prefix == OBS_STR:
        features[f"{prefix}.state"] = {
            "dtype": "float32",
            "shape": (len(joint_fts),),
            "names": list(joint_fts),
        }

    for key, shape in cam_fts.items():
        dtype = "video" if use_video else "image"
        if len(shape) == 3 and shape[2] in (1, 3):
            features[f"{prefix}.images.{key}"] = {
                "dtype": dtype,
                "shape": shape,
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": shape[2] == 1},
            }
        else:
            raise ValueError(
                f"Camera feature '{key}' has shape {shape}. "
                f"Expected a 3-tuple (H, W, C), e.g. (480, 640, 3) for RGB or (480, 640, 1) for depth."
            )

    _validate_feature_names(features)
    return features


def build_dataset_frame(
    ds_features: dict[str, dict], values: dict[str, Any], prefix: str
) -> dict[str, np.ndarray]:

    frame = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
        elif ft["dtype"] in ["image", "video"]:
            frame[key] = values[key.removeprefix(f"{prefix}.images.")]

    return frame


def dataset_to_policy_features(features: dict[str, dict]) -> dict[str, PolicyFeature]:


    policy_features = {}
    for key, ft in features.items():
        shape = ft["shape"]
        if ft["dtype"] in ["image", "video"]:
            type = FeatureType.VISUAL
            if len(shape) != 3:
                raise ValueError(f"Number of dimensions of {key} != 3 (shape={shape})")
            else:
                names = ft["names"]

                if names[2] in ["channel", "channels"]:
                    shape = (shape[2], shape[0], shape[1])
        elif key == OBS_ENV_STATE:
            type = FeatureType.ENV
        elif key.startswith(OBS_STR):
            type = FeatureType.STATE
        elif key.startswith(ACTION):
            type = FeatureType.ACTION
        else:
            continue

        policy_features[key] = PolicyFeature(
            type=type,
            shape=shape,
        )

    return policy_features


def combine_feature_dicts(*dicts: dict) -> dict:

    out: dict = {}
    for d in dicts:
        for key, value in d.items():
            if not isinstance(value, dict):
                out[key] = value
                continue

            dtype = value.get("dtype")
            shape = value.get("shape")
            is_vector = (
                dtype not in ("image", "video", "string")
                and isinstance(shape, tuple)
                and len(shape) == 1
                and "names" in value
            )

            if is_vector:

                target = out.setdefault(key, {"dtype": dtype, "names": [], "shape": (0,)})

                if "dtype" in target and dtype != target["dtype"]:
                    raise ValueError(f"dtype mismatch for '{key}': {target['dtype']} vs {dtype}")


                seen = set(target["names"])
                for n in value["names"]:
                    if n not in seen:
                        target["names"].append(n)
                        seen.add(n)

                target["shape"] = (len(target["names"]),)
            else:

                out[key] = value
    return out
