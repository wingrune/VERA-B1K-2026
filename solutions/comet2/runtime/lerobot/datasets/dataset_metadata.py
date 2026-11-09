import contextlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import packaging.version
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from lerobot.configs.video import VideoEncoderConfig
from lerobot.utils.constants import DEFAULT_FEATURES, HF_LEROBOT_HOME, HF_LEROBOT_HUB_CACHE
from lerobot.utils.feature_utils import _validate_feature_names
from lerobot.utils.utils import flatten_dict

from .feature_utils import create_empty_dataset_info
from .io_utils import (
    get_file_size_in_mb,
    load_episodes,
    load_info,
    load_stats,
    load_subtasks,
    load_tasks,
    write_info,
    write_stats,
    write_tasks,
)
from .utils import (
    DEFAULT_EPISODES_PATH,
    check_version_compatibility,
    get_safe_version,
    has_legacy_hub_download_metadata,
    is_valid_version,
    update_chunk_file_indices,
)
from .video_utils import get_video_info

CODEBASE_VERSION = "v3.0"


class LeRobotDatasetMetadata:


    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        metadata_buffer_size: int = 10,
    ):

        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self._requested_root = Path(root) if root is not None else None
        self.root = self._requested_root if self._requested_root is not None else HF_LEROBOT_HOME / repo_id
        self._pq_writer = None
        self.latest_episode = None
        self._metadata_buffer: list[dict] = []
        self._metadata_buffer_size = metadata_buffer_size
        self._finalized = False

        try:
            if force_cache_sync or (
                self._requested_root is None and has_legacy_hub_download_metadata(self.root)
            ):
                raise FileNotFoundError
            self._load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            self._pull_from_repo(allow_patterns="meta/")
            self._load_metadata()

    def _flush_metadata_buffer(self) -> None:

        if not hasattr(self, "_metadata_buffer") or len(self._metadata_buffer) == 0:
            return

        combined_dict = {}
        for episode_dict in self._metadata_buffer:
            for key, value in episode_dict.items():
                if key not in combined_dict:
                    combined_dict[key] = []


                val = value[0] if isinstance(value, list) else value
                combined_dict[key].append(val.tolist() if isinstance(val, np.ndarray) else val)

        first_ep = self._metadata_buffer[0]
        chunk_idx = first_ep["meta/episodes/chunk_index"][0]
        file_idx = first_ep["meta/episodes/file_index"][0]

        table = pa.Table.from_pydict(combined_dict)

        if not self._pq_writer:
            path = Path(self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx))
            path.parent.mkdir(parents=True, exist_ok=True)
            self._pq_writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )

        self._pq_writer.write_table(table)

        self.latest_episode = self._metadata_buffer[-1]
        self._metadata_buffer.clear()

    def _close_writer(self) -> None:

        self._flush_metadata_buffer()

        writer = getattr(self, "_pq_writer", None)
        if writer is not None:
            writer.close()
            self._pq_writer = None

    def finalize(self) -> None:

        if getattr(self, "_finalized", False):
            return
        self._close_writer()
        self._finalized = True

    def __del__(self):


        with contextlib.suppress(Exception):
            self.finalize()

    def _load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks = load_tasks(self.root)
        self.subtasks = load_subtasks(self.root)
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)

    def ensure_readable(self) -> None:

        if self.episodes is None:
            self._load_metadata()

    def filter_episodes(
        self,
        predicate: Callable[[dict], bool],
        candidates: list[int] | None = None,
    ) -> list[int]:

        self.ensure_readable()
        if candidates is not None:
            candidate_set = set(candidates)
            combined = lambda ep: ep["episode_index"] in candidate_set and predicate(ep)
        else:
            combined = predicate
        filtered = self.episodes.filter(combined, keep_in_memory=True, load_from_cache_file=False)
        return sorted(int(idx) for idx in filtered["episode_index"])

    def _pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        if self._requested_root is None:
            self.root = Path(
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                )
            )
            return

        self._requested_root.mkdir(exist_ok=True, parents=True)
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self._requested_root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )
        self.root = self._requested_root

    @property
    def url_root(self) -> str:

        return f"hf://datasets/{self.repo_id}"

    @property
    def _version(self) -> packaging.version.Version:

        return packaging.version.parse(self.info.codebase_version)

    def get_data_file_path(self, ep_index: int) -> Path:

        if self.episodes is None:
            self.episodes = load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        chunk_idx = ep["data/chunk_index"]
        file_idx = ep["data/file_index"]
        fpath = self.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:

        if self.episodes is None:
            self.episodes = load_episodes(self.root)
        if ep_index >= len(self.episodes):
            raise IndexError(
                f"Episode index {ep_index} out of range. Episodes: {len(self.episodes) if self.episodes else 0}"
            )
        ep = self.episodes[ep_index]
        chunk_idx = ep[f"videos/{vid_key}/chunk_index"]
        file_idx = ep[f"videos/{vid_key}/file_index"]
        fpath = self.video_path.format(video_key=vid_key, chunk_index=chunk_idx, file_index=file_idx)
        return Path(fpath)

    @property
    def data_path(self) -> str:

        return self.info.data_path

    @property
    def video_path(self) -> str | None:

        return self.info.video_path

    @property
    def robot_type(self) -> str | None:

        return self.info.robot_type

    @property
    def fps(self) -> int:

        return self.info.fps

    @property
    def features(self) -> dict[str, dict]:

        return self.info.features

    @property
    def image_keys(self) -> list[str]:

        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:

        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def depth_keys(self) -> list[str]:


        def _is_depth(ft: dict) -> bool:
            info = ft.get("info") or {}
            video_info = ft.get("video_info") or {}
            return (
                info.get("is_depth_map", False)
                or info.get("video.is_depth_map", False)
                or video_info.get("video.is_depth_map", False)
            )

        return [key for key, ft in self.features.items() if _is_depth(ft)]

    @property
    def camera_keys(self) -> list[str]:

        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:

        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:

        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:

        return self.info.total_episodes

    @property
    def total_frames(self) -> int:

        return self.info.total_frames

    @property
    def total_tasks(self) -> int:

        return self.info.total_tasks

    @property
    def chunks_size(self) -> int:

        return self.info.chunks_size

    @property
    def data_files_size_in_mb(self) -> int:

        return self.info.data_files_size_in_mb

    @property
    def video_files_size_in_mb(self) -> int:

        return self.info.video_files_size_in_mb

    def get_task_index(self, task: str) -> int | None:

        if task in self.tasks.index:
            return int(self.tasks.loc[task].task_index)
        else:
            return None

    def save_episode_tasks(self, tasks: list[str]):

        if len(set(tasks)) != len(tasks):
            raise ValueError(f"Tasks are not unique: {tasks}")

        if self.tasks is None:
            new_tasks = tasks
            task_indices = range(len(tasks))
            self.tasks = pd.DataFrame({"task_index": task_indices}, index=pd.Index(tasks, name="task"))
        else:
            new_tasks = [task for task in tasks if task not in self.tasks.index]
            new_task_indices = range(len(self.tasks), len(self.tasks) + len(new_tasks))
            for task_idx, task in zip(new_task_indices, new_tasks, strict=False):
                self.tasks.loc[task] = task_idx

        if len(new_tasks) > 0:

            write_tasks(self.tasks, self.root)

    def _save_episode_metadata(self, episode_dict: dict) -> None:


        episode_dict = {key: [value] for key, value in episode_dict.items()}
        num_frames = episode_dict["length"][0]

        if self.latest_episode is None:

            chunk_idx, file_idx = 0, 0
            if self.episodes is not None and len(self.episodes) > 0:


                chunk_idx = self.episodes[-1]["meta/episodes/chunk_index"]
                file_idx = self.episodes[-1]["meta/episodes/file_index"]
                latest_num_frames = self.episodes[-1]["dataset_to_index"]
                episode_dict["dataset_from_index"] = [latest_num_frames]
                episode_dict["dataset_to_index"] = [latest_num_frames + num_frames]


                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
            else:
                episode_dict["dataset_from_index"] = [0]
                episode_dict["dataset_to_index"] = [num_frames]

            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
        else:
            chunk_idx = self.latest_episode["meta/episodes/chunk_index"][0]
            file_idx = self.latest_episode["meta/episodes/file_index"][0]

            latest_path = (
                self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
                if self._pq_writer is None
                else self._pq_writer.where
            )

            if Path(latest_path).exists():
                latest_size_in_mb = get_file_size_in_mb(Path(latest_path))
                latest_num_frames = self.latest_episode["episode_index"][0]

                av_size_per_frame = latest_size_in_mb / latest_num_frames if latest_num_frames > 0 else 0.0

                if latest_size_in_mb + av_size_per_frame * num_frames >= self.data_files_size_in_mb:

                    self._flush_metadata_buffer()
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.chunks_size)
                    self._close_writer()


            episode_dict["meta/episodes/chunk_index"] = [chunk_idx]
            episode_dict["meta/episodes/file_index"] = [file_idx]
            episode_dict["dataset_from_index"] = [self.latest_episode["dataset_to_index"][0]]
            episode_dict["dataset_to_index"] = [self.latest_episode["dataset_to_index"][0] + num_frames]


        self._metadata_buffer.append(episode_dict)
        self.latest_episode = episode_dict

        if len(self._metadata_buffer) >= self._metadata_buffer_size:
            self._flush_metadata_buffer()

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
        episode_metadata: dict,
    ) -> None:

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        episode_dict.update(episode_metadata)
        episode_dict.update(flatten_dict({"stats": episode_stats}))
        self._save_episode_metadata(episode_dict)


        self.info.total_episodes += 1
        self.info.total_frames += episode_length
        self.info.total_tasks = len(self.tasks)
        self.info.splits = {"train": f"0:{self.info.total_episodes}"}

        write_info(self.info, self.root)

        if self.stats is not None:
            from .compute_stats import aggregate_stats

            self.stats = aggregate_stats([self.stats, episode_stats])
        else:
            self.stats = episode_stats
        write_stats(self.stats, self.root)

    def update_video_info(
        self,
        video_key: str | None = None,
        video_encoder: VideoEncoderConfig | None = None,
    ) -> None:

        if video_key is not None and video_key not in self.video_keys:
            raise ValueError(f"Video key {video_key} not found in dataset")

        video_keys = [video_key] if video_key is not None else self.video_keys
        for key in video_keys:
            existing = self.features[key].get("info") or {}

            if set(existing.keys()) - {"is_depth_map"}:
                continue
            video_path = self.root / self.video_path.format(video_key=key, chunk_index=0, file_index=0)
            new_info = get_video_info(video_path, video_encoder=video_encoder)
            self.info.features[key]["info"] = {**existing, **new_info}

    def update_chunk_settings(
        self,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> None:

        if chunks_size is not None:
            if chunks_size <= 0:
                raise ValueError(f"chunks_size must be positive, got {chunks_size}")
            self.info.chunks_size = chunks_size

        if data_files_size_in_mb is not None:
            if data_files_size_in_mb <= 0:
                raise ValueError(f"data_files_size_in_mb must be positive, got {data_files_size_in_mb}")
            self.info.data_files_size_in_mb = data_files_size_in_mb

        if video_files_size_in_mb is not None:
            if video_files_size_in_mb <= 0:
                raise ValueError(f"video_files_size_in_mb must be positive, got {video_files_size_in_mb}")
            self.info.video_files_size_in_mb = video_files_size_in_mb


        write_info(self.info, self.root)

    def get_chunk_settings(self) -> dict[str, int]:

        return {
            "chunks_size": self.chunks_size,
            "data_files_size_in_mb": self.data_files_size_in_mb,
            "video_files_size_in_mb": self.video_files_size_in_mb,
        }

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        robot_type: str | None = None,
        root: str | Path | None = None,
        use_videos: bool = True,
        metadata_buffer_size: int = 10,
        chunks_size: int | None = None,
        data_files_size_in_mb: int | None = None,
        video_files_size_in_mb: int | None = None,
    ) -> "LeRobotDatasetMetadata":

        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj._requested_root = Path(root) if root is not None else None
        obj.root = obj._requested_root if obj._requested_root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        features = {**features, **DEFAULT_FEATURES}
        _validate_feature_names(features)

        obj.tasks = None
        obj.subtasks = None
        obj.episodes = None
        obj.stats = None
        obj.info = create_empty_dataset_info(
            CODEBASE_VERSION,
            fps,
            features,
            use_videos,
            robot_type,
            chunks_size,
            data_files_size_in_mb,
            video_files_size_in_mb,
        )
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError(
                f"Features contain video keys {obj.video_keys}, but 'use_videos' is set to False. "
                "Either remove video features from the features dict, or set 'use_videos=True'."
            )
        write_info(obj.info, obj.root)
        obj.revision = None
        obj._pq_writer = None
        obj.latest_episode = None
        obj._metadata_buffer = []
        obj._metadata_buffer_size = metadata_buffer_size
        obj._finalized = False
        return obj
