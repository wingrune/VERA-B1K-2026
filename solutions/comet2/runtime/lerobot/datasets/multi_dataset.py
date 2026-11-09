import logging
from collections.abc import Callable
from pathlib import Path

import datasets
import torch
import torch.utils

from lerobot.utils.constants import HF_LEROBOT_HOME

from .compute_stats import aggregate_stats
from .feature_utils import get_hf_features_from_features
from .lerobot_dataset import LeRobotDataset
from .video_utils import VideoFrame

logger = logging.getLogger(__name__)


class MultiLeRobotDataset(torch.utils.data.Dataset):


    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else dict.fromkeys(repo_ids, 0.0001)


        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]


        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            if extra_keys:
                logger.warning(
                    f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                    "other datasets."
                )
                self.disabled_features.update(extra_keys)

        self.delta_timestamps = delta_timestamps


        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
        self.set_image_transforms(image_transforms)

    def set_image_transforms(self, image_transforms: Callable | None) -> None:

        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        for dataset in getattr(self, "_datasets", []):
            dataset.set_image_transforms(self.image_transforms)

    def clear_image_transforms(self) -> None:

        self.set_image_transforms(None)

    @property
    def repo_id_to_index(self):

        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def fps(self) -> int:

        return self._datasets[0].meta.info.fps

    @property
    def video(self) -> bool:

        return len(self._datasets[0].meta.video_keys) > 0

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update(
                {
                    k: v
                    for k, v in get_hf_features_from_features(dataset.features).items()
                    if k not in self.disabled_features
                }
            )
        return features

    @property
    def camera_keys(self) -> list[str]:

        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image | VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:

        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:

        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:

        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:


        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")

        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )
