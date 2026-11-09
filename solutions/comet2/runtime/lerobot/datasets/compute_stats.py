from __future__ import annotations

import logging

import numpy as np

from lerobot.processor import RelativeActionsProcessorStep
from lerobot.utils.constants import ACTION, OBS_STATE

from .io_utils import load_image_as_numpy

DEFAULT_QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


class RunningQuantileStats:


    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = num_quantile_bins

        self._quantile_list = quantile_list
        if self._quantile_list is None:
            self._quantile_list = DEFAULT_QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:

        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")

            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)


        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (
            num_elements / self._count
        )

        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:

        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2

        stddev = np.sqrt(np.maximum(0, variance))

        stats = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": stddev,
            "count": np.array([self._count]),
        }

        quantile_results = self._compute_quantiles()
        for i, q in enumerate(self._quantile_keys):
            stats[q] = quantile_results[i]

        return stats

    def _adjust_histograms(self):

        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]


            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1
            )


            old_centers = (old_edges[:-1] + old_edges[1:]) / 2
            new_hist = np.zeros(self._num_quantile_bins)

            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:

                    bin_idx = np.searchsorted(new_edges, old_center) - 1
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:

        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:

        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = []

            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                q_value = self._compute_single_quantile(hist, edges, target_count)
                q_values.append(q_value)

            results.append(np.array(q_values))
        return results

    def _compute_single_quantile(self, hist: np.ndarray, edges: np.ndarray, target_count: float) -> float:

        cumsum = np.cumsum(hist)
        idx = np.searchsorted(cumsum, target_count)

        if idx == 0:
            return edges[0]
        if idx >= len(cumsum):
            return edges[-1]


        count_before = cumsum[idx - 1]
        count_in_bin = cumsum[idx] - count_before


        if count_in_bin == 0:
            return edges[idx]


        fraction = (target_count - count_before) / count_in_bin
        return edges[idx] + fraction * (edges[idx + 1] - edges[idx])


def estimate_num_samples(
    dataset_len: int, min_num_samples: int = 100, max_num_samples: int = 10_000, power: float = 0.75
) -> int:

    if dataset_len < min_num_samples:
        min_num_samples = dataset_len
    return max(min_num_samples, min(int(dataset_len**power), max_num_samples))


def sample_indices(data_len: int) -> list[int]:
    num_samples = estimate_num_samples(data_len)
    return np.round(np.linspace(0, data_len - 1, num_samples)).astype(int).tolist()


def auto_downsample_height_width(img: np.ndarray, target_size: int = 150, max_size_threshold: int = 300):
    _, height, width = img.shape

    if max(width, height) < max_size_threshold:

        return img

    downsample_factor = int(width / target_size) if width > height else int(height / target_size)
    return img[:, ::downsample_factor, ::downsample_factor]


def sample_images(image_paths: list[str]) -> np.ndarray:
    sampled_indices = sample_indices(len(image_paths))

    images = None
    for i, idx in enumerate(sampled_indices):
        path = image_paths[idx]

        img = load_image_as_numpy(path, dtype=np.uint8, channel_first=True)
        img = auto_downsample_height_width(img)

        if images is None:
            images = np.empty((len(sampled_indices), *img.shape), dtype=np.uint8)

        images[i] = img

    return images


def _reshape_stats_by_axis(
    stats: dict[str, np.ndarray],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    original_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:

    if axis == (1,) and not keepdims:
        return stats

    result = {}
    for key, value in stats.items():
        if key == "count":
            result[key] = value
        else:
            result[key] = _reshape_single_stat(value, axis, keepdims, original_shape)

    return result


def _reshape_for_image_stats(value: np.ndarray, keepdims: bool) -> np.ndarray:

    if keepdims and value.ndim == 1:
        return value.reshape(1, -1, 1, 1)
    return value


def _reshape_for_vector_stats(
    value: np.ndarray, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray:

    if not keepdims:
        return value

    if len(original_shape) == 1 and value.ndim > 0:
        return value.reshape(1)
    elif len(original_shape) >= 2 and value.ndim == 1:
        return value.reshape(1, -1)
    return value


def _reshape_for_feature_stats(value: np.ndarray, keepdims: bool) -> np.ndarray:

    if not keepdims:
        return value

    if value.ndim == 0:
        return value.reshape(1, 1)
    elif value.ndim == 1:
        return value.reshape(-1, 1)
    return value


def _reshape_for_global_stats(
    value: np.ndarray, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray | float:

    if keepdims:
        target_shape = tuple(1 for _ in original_shape)
        return value.reshape(target_shape)

    return np.atleast_1d(value)


def _reshape_single_stat(
    value: np.ndarray, axis: int | tuple[int, ...] | None, keepdims: bool, original_shape: tuple[int, ...]
) -> np.ndarray | float:

    if axis == (0, 2, 3):
        return _reshape_for_image_stats(value, keepdims)

    if axis in [0, (0,)]:
        return _reshape_for_vector_stats(value, keepdims, original_shape)

    if axis == (1,):
        return _reshape_for_feature_stats(value, keepdims)

    if axis is None:
        return _reshape_for_global_stats(value, keepdims, original_shape)

    return value


def _prepare_array_for_stats(array: np.ndarray, axis: int | tuple[int, ...] | None) -> tuple[np.ndarray, int]:

    if axis == (0, 2, 3):
        batch_size, channels, height, width = array.shape
        reshaped = array.transpose(0, 2, 3, 1).reshape(-1, channels)
        return reshaped, batch_size

    if axis == 0 or axis == (0,):
        reshaped = array
        if array.ndim == 1:
            reshaped = array.reshape(-1, 1)
        return reshaped, array.shape[0]

    if axis == (1,):
        return array.T, array.shape[1]

    if axis is None:
        reshaped = array.reshape(-1, 1)

        return reshaped, array.shape[0] if array.ndim > 0 else 1

    raise ValueError(f"Unsupported axis configuration: {axis}")


def _compute_basic_stats(
    array: np.ndarray, sample_count: int, quantile_list: list[float] | None = None
) -> dict[str, np.ndarray]:

    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES
    quantile_list_keys = [f"q{int(q * 100):02d}" for q in quantile_list]

    stats = {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "count": np.array([sample_count]),
    }

    for q in quantile_list_keys:
        stats[q] = stats["mean"].copy()

    return stats


def get_feature_stats(
    array: np.ndarray,
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
    quantile_list: list[float] | None = None,
) -> dict[str, np.ndarray]:

    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    original_shape = array.shape
    reshaped, sample_count = _prepare_array_for_stats(array, axis)

    if reshaped.shape[0] < 2:
        stats = _compute_basic_stats(reshaped, sample_count, quantile_list)
    else:
        running_stats = RunningQuantileStats()
        running_stats.update(reshaped)
        stats = running_stats.get_statistics()
        stats["count"] = np.array([sample_count])

    stats = _reshape_stats_by_axis(stats, axis, keepdims, original_shape)
    return stats


def compute_episode_stats(
    episode_data: dict[str, list[str] | np.ndarray],
    features: dict,
    quantile_list: list[float] | None = None,
) -> dict:

    if quantile_list is None:
        quantile_list = DEFAULT_QUANTILES

    ep_stats = {}
    for key, data in episode_data.items():
        if features[key]["dtype"] == "string":
            continue

        if features[key]["dtype"] in ["image", "video"]:
            ep_ft_array = sample_images(data)
            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            ep_ft_array = data
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(
            ep_ft_array, axis=axes_to_reduce, keepdims=keepdims, quantile_list=quantile_list
        )

        if features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats


def _validate_stat_value(value: np.ndarray, key: str, feature_key: str) -> None:

    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"Stats must be composed of numpy array, but key '{key}' of feature '{feature_key}' "
            f"is of type '{type(value)}' instead."
        )

    if value.ndim == 0:
        raise ValueError("Number of dimensions must be at least 1, and is 0 instead.")

    if key == "count" and value.shape != (1,):
        raise ValueError(f"Shape of 'count' must be (1), but is {value.shape} instead.")

    if "image" in feature_key and key != "count" and value.shape not in ((3, 1, 1), (1, 1, 1)):
        raise ValueError(
            f"Shape of quantile '{key}' must be (3,1,1) or (1,1,1) but is {value.shape} instead."
        )


def _assert_type_and_shape(stats_list: list[dict[str, dict]]):

    for stats in stats_list:
        for feature_key, feature_stats in stats.items():
            for stat_key, stat_value in feature_stats.items():
                _validate_stat_value(stat_value, stat_key, feature_key)


def aggregate_feature_stats(stats_ft_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:

    means = np.stack([s["mean"] for s in stats_ft_list])
    variances = np.stack([s["std"] ** 2 for s in stats_ft_list])
    counts = np.stack([s["count"] for s in stats_ft_list])
    total_count = counts.sum(axis=0)


    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)


    weighted_means = means * counts
    total_mean = weighted_means.sum(axis=0) / total_count


    delta_means = means - total_mean
    weighted_variances = (variances + delta_means**2) * counts
    total_variance = weighted_variances.sum(axis=0) / total_count

    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }

    if stats_ft_list:
        quantile_keys = [k for k in stats_ft_list[0] if k.startswith("q") and k[1:].isdigit()]

        for q_key in quantile_keys:
            if all(q_key in s for s in stats_ft_list):
                quantile_values = np.stack([s[q_key] for s in stats_ft_list])
                weighted_quantiles = quantile_values * counts
                aggregated[q_key] = weighted_quantiles.sum(axis=0) / total_count

    return aggregated


def aggregate_stats(stats_list: list[dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:


    _assert_type_and_shape(stats_list)

    data_keys = {key for stats in stats_list for key in stats}
    aggregated_stats = {key: {} for key in data_keys}

    for key in data_keys:
        stats_with_key = [stats[key] for stats in stats_list if key in stats]
        aggregated_stats[key] = aggregate_feature_stats(stats_with_key)

    return aggregated_stats


def _get_valid_chunk_starts(episode_indices: np.ndarray, chunk_size: int) -> np.ndarray:

    total = len(episode_indices)
    if total < chunk_size:
        return np.array([], dtype=np.int64)
    max_start = total - chunk_size
    starts = np.arange(max_start + 1)
    valid = episode_indices[starts] == episode_indices[starts + chunk_size - 1]
    return starts[valid]


def _compute_relative_chunk_batch(
    start_indices: np.ndarray,
    all_actions: np.ndarray,
    all_states: np.ndarray,
    chunk_size: int,
    relative_mask: np.ndarray,
) -> np.ndarray:

    if len(start_indices) == 0:
        return np.empty((0, all_actions.shape[1]), dtype=np.float32)
    offsets = np.arange(chunk_size)
    frame_idx = start_indices[:, None] + offsets[None, :]
    chunks = all_actions[frame_idx].copy()
    states = all_states[start_indices]
    mask_dim = len(relative_mask)
    chunks[:, :, :mask_dim] -= states[:, None, :mask_dim] * relative_mask[None, None, :]
    return chunks.reshape(-1, all_actions.shape[1])


def compute_relative_action_stats(
    hf_dataset,
    features: dict,
    chunk_size: int,
    exclude_joints: list[str] | None = None,
    num_workers: int = 0,
) -> dict[str, np.ndarray]:

    if exclude_joints is None:
        exclude_joints = []

    action_dim = features[ACTION]["shape"][0]
    action_names = features.get(ACTION, {}).get("names")
    mask_step = RelativeActionsProcessorStep(
        enabled=True,
        exclude_joints=exclude_joints,
        action_names=action_names,
    )
    relative_mask = np.array(mask_step._build_mask(action_dim), dtype=np.float32)

    logging.info("Loading action/state data for relative action stats...")
    all_actions = np.array(hf_dataset[ACTION], dtype=np.float32)
    all_states = np.array(hf_dataset[OBS_STATE], dtype=np.float32)
    episode_indices = np.array(hf_dataset["episode_index"])

    valid_starts = _get_valid_chunk_starts(episode_indices, chunk_size)
    if len(valid_starts) == 0:
        raise RuntimeError(
            f"No valid chunks found (total_frames={len(episode_indices)}, chunk_size={chunk_size})"
        )

    effective_workers = max(num_workers, 1)
    logging.info(
        f"Computing relative action stats from {len(valid_starts)} chunks "
        f"(chunk_size={chunk_size}, workers={effective_workers})"
    )

    batch_size = 50_000
    batches = [valid_starts[i : i + batch_size] for i in range(0, len(valid_starts), batch_size)]

    running_stats = RunningQuantileStats()

    if num_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [
                pool.submit(
                    _compute_relative_chunk_batch,
                    batch,
                    all_actions,
                    all_states,
                    chunk_size,
                    relative_mask,
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                running_stats.update(future.result())
    else:
        for batch in batches:
            running_stats.update(
                _compute_relative_chunk_batch(batch, all_actions, all_states, chunk_size, relative_mask)
            )

    stats = running_stats.get_statistics()

    excluded_dims = int(len(relative_mask) - relative_mask.sum())
    total_frames = len(valid_starts) * chunk_size
    logging.info(
        f"Relative action stats ({len(valid_starts)} chunks, {total_frames} frames): "
        f"relative_dims={int(relative_mask.sum())}/{len(relative_mask)} (excluded={excluded_dims}), "
        f"mean={np.abs(stats['mean']).mean():.4f}, std={stats['std'].mean():.4f}, "
        f"q01={stats['q01'].mean():.4f}, q99={stats['q99'].mean():.4f}"
    )

    return stats
