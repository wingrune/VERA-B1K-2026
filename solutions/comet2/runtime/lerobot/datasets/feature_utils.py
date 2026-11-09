from pprint import pformat

import datasets
import numpy as np
from PIL import Image as PILImage

from lerobot.configs.video import VIDEO_ENCODER_INFO_KEYS
from lerobot.utils.constants import DEFAULT_FEATURES
from lerobot.utils.utils import is_valid_numpy_dtype_string

from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    DatasetInfo,
)


def get_hf_features_from_features(features: dict) -> datasets.Features:

    hf_features = {}
    for key, ft in features.items():
        if ft["dtype"] == "video":
            continue
        elif ft["dtype"] == "image":
            hf_features[key] = datasets.Image()
        elif ft["shape"] == (1,):
            hf_features[key] = datasets.Value(dtype=ft["dtype"])
        elif len(ft["shape"]) == 1:
            hf_features[key] = datasets.Sequence(
                length=ft["shape"][0], feature=datasets.Value(dtype=ft["dtype"])
            )
        elif len(ft["shape"]) == 2:
            hf_features[key] = datasets.Array2D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 3:
            hf_features[key] = datasets.Array3D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 4:
            hf_features[key] = datasets.Array4D(shape=ft["shape"], dtype=ft["dtype"])
        elif len(ft["shape"]) == 5:
            hf_features[key] = datasets.Array5D(shape=ft["shape"], dtype=ft["dtype"])
        else:
            raise ValueError(f"Corresponding feature is not valid: {ft}")

    return datasets.Features(hf_features)


def create_empty_dataset_info(
    codebase_version: str,
    fps: int,
    features: dict,
    use_videos: bool,
    robot_type: str | None = None,
    chunks_size: int | None = None,
    data_files_size_in_mb: int | None = None,
    video_files_size_in_mb: int | None = None,
) -> DatasetInfo:

    return DatasetInfo(
        codebase_version=codebase_version,
        fps=fps,
        features=features,
        robot_type=robot_type,
        chunks_size=chunks_size or DEFAULT_CHUNK_SIZE,
        data_files_size_in_mb=data_files_size_in_mb or DEFAULT_DATA_FILE_SIZE_IN_MB,
        video_files_size_in_mb=video_files_size_in_mb or DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        data_path=DEFAULT_DATA_PATH,
        video_path=DEFAULT_VIDEO_PATH if use_videos else None,
    )


def features_equal_for_merge(features_a: dict[str, dict], features_b: dict[str, dict]) -> bool:


    def _without_encoder_info_keys(feature: dict) -> dict:
        filtered = dict(feature)
        filtered_info = filtered.get("info")
        if isinstance(filtered_info, dict):
            filtered["info"] = {
                info_key: info_value
                for info_key, info_value in filtered_info.items()
                if info_key not in VIDEO_ENCODER_INFO_KEYS
            }
        return filtered

    if set(features_a) != set(features_b):
        return False
    for key in features_a:
        fa_key = features_a[key]
        fb_key = features_b[key]
        if fa_key.get("dtype") != fb_key.get("dtype"):
            return False
        if fa_key.get("dtype") != "video":
            if fa_key != fb_key:
                return False
            continue

        if _without_encoder_info_keys(fa_key) != _without_encoder_info_keys(fb_key):
            return False
    return True


def check_delta_timestamps(
    delta_timestamps: dict[str, list[float]], fps: int, tolerance_s: float, raise_value_error: bool = True
) -> bool:

    outside_tolerance = {}
    for key, delta_ts in delta_timestamps.items():
        within_tolerance = [abs(ts * fps - round(ts * fps)) / fps <= tolerance_s for ts in delta_ts]
        if not all(within_tolerance):
            outside_tolerance[key] = [
                ts for ts, is_within in zip(delta_ts, within_tolerance, strict=True) if not is_within
            ]

    if len(outside_tolerance) > 0:
        if raise_value_error:
            raise ValueError(
                f"""
                The following delta_timestamps are found outside of tolerance range.
                Please make sure they are multiples of 1/{fps} +/- tolerance and adjust
                their values accordingly.
                \n{pformat(outside_tolerance)}
                """
            )
        return False

    return True


def get_delta_indices(delta_timestamps: dict[str, list[float]], fps: int) -> dict[str, list[int]]:

    delta_indices = {}
    for key, delta_ts in delta_timestamps.items():
        delta_indices[key] = [round(d * fps) for d in delta_ts]

    return delta_indices


def validate_frame(frame: dict, features: dict) -> None:


    expected_features = set(features) - set(DEFAULT_FEATURES)
    actual_features = set(frame)


    if "task" not in actual_features:
        raise ValueError("Feature mismatch in `frame` dictionary:\nMissing features: {'task'}\n")


    actual_features_for_validation = actual_features - {"task"}

    error_message = validate_features_presence(actual_features_for_validation, expected_features)

    common_features = actual_features_for_validation & expected_features
    for name in common_features:
        error_message += validate_feature_dtype_and_shape(name, features[name], frame[name])

    if error_message:
        raise ValueError(error_message)


def validate_features_presence(actual_features: set[str], expected_features: set[str]) -> str:

    error_message = ""
    missing_features = expected_features - actual_features
    extra_features = actual_features - expected_features

    if missing_features or extra_features:
        error_message += "Feature mismatch in `frame` dictionary:\n"
        if missing_features:
            error_message += f"Missing features: {missing_features}\n"
        if extra_features:
            error_message += f"Extra features: {extra_features}\n"

    return error_message


def validate_feature_dtype_and_shape(
    name: str, feature: dict, value: np.ndarray | PILImage.Image | str
) -> str:

    expected_dtype = feature["dtype"]
    expected_shape = feature["shape"]
    if is_valid_numpy_dtype_string(expected_dtype):
        return validate_feature_numpy_array(name, expected_dtype, expected_shape, value)
    elif expected_dtype in ["image", "video"]:
        return validate_feature_image_or_video(name, expected_shape, value)
    elif expected_dtype == "string":
        return validate_feature_string(name, value)
    else:
        raise NotImplementedError(f"The feature dtype '{expected_dtype}' is not implemented yet.")


def validate_feature_numpy_array(
    name: str, expected_dtype: str, expected_shape: list[int], value: np.ndarray
) -> str:

    error_message = ""
    if isinstance(value, np.ndarray):
        actual_dtype = value.dtype
        actual_shape = value.shape

        if actual_dtype != np.dtype(expected_dtype):
            error_message += f"The feature '{name}' of dtype '{actual_dtype}' is not of the expected dtype '{expected_dtype}'.\n"

        if actual_shape != expected_shape:
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{expected_shape}'.\n"
    else:
        error_message += f"The feature '{name}' is not a 'np.ndarray'. Expected type is '{expected_dtype}', but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_image_or_video(
    name: str, expected_shape: list[str], value: np.ndarray | PILImage.Image
) -> str:


    error_message = ""
    if isinstance(value, np.ndarray):
        actual_shape = value.shape
        c, h, w = expected_shape
        if len(actual_shape) != 3 or (actual_shape != (c, h, w) and actual_shape != (h, w, c)):
            error_message += f"The feature '{name}' of shape '{actual_shape}' does not have the expected shape '{(c, h, w)}' or '{(h, w, c)}'.\n"
    elif isinstance(value, PILImage.Image):
        pass
    else:
        error_message += f"The feature '{name}' is expected to be of type 'PIL.Image' or 'np.ndarray' channel first or channel last, but type '{type(value)}' provided instead.\n"

    return error_message


def validate_feature_string(name: str, value: str) -> str:

    if not isinstance(value, str):
        return f"The feature '{name}' is expected to be of type 'str', but type '{type(value)}' provided instead.\n"
    return ""


def validate_episode_buffer(episode_buffer: dict, total_episodes: int, features: dict) -> None:

    if "size" not in episode_buffer:
        raise ValueError("size key not found in episode_buffer")

    if "task" not in episode_buffer:
        raise ValueError("task key not found in episode_buffer")

    if episode_buffer["episode_index"] != total_episodes:

        raise NotImplementedError(
            "You might have manually provided the episode_buffer with an episode_index that doesn't "
            "match the total number of episodes already in the dataset. This is not supported for now."
        )

    if episode_buffer["size"] == 0:
        raise ValueError("You must add one or several frames with `add_frame` before calling `add_episode`.")

    buffer_keys = set(episode_buffer.keys()) - {"task", "size"}
    if not buffer_keys == set(features):
        raise ValueError(
            f"Features from `episode_buffer` don't match the ones in `features`."
            f"In episode_buffer not in features: {buffer_keys - set(features)}"
            f"In features not in episode_buffer: {set(features) - buffer_keys}"
        )
