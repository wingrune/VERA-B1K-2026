import contextlib
import logging
from collections.abc import Callable
from pathlib import Path

import datasets
import torch
import torch.utils
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.configs.video import DepthEncoderConfig, VideoEncoderConfig
from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

from .dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from .dataset_reader import DatasetReader
from .utils import (
    create_lerobot_dataset_card,
    get_safe_version,
    is_valid_version,
)
from .video_utils import (
    StreamingVideoEncoder,
    get_safe_default_video_backend,
)

logger = logging.getLogger(__name__)


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        episode_filter: Callable[[dict], bool] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        return_uint8: bool = False,
        batch_encoding_size: int = 1,
        camera_encoder: VideoEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        encoder_threads: int | None = None,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
    ):

        super().__init__()
        self.repo_id = repo_id
        self._requested_root = Path(root) if root else None
        self.reader = None
        self.set_image_transforms(image_transforms)
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self._video_backend = video_backend if video_backend else get_safe_default_video_backend()
        self._return_uint8 = return_uint8
        self._batch_encoding_size = batch_encoding_size
        self._encoder_threads = encoder_threads

        if self._requested_root is not None:
            self._requested_root.mkdir(exist_ok=True, parents=True)


        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self._requested_root, self.revision, force_cache_sync=force_cache_sync
        )
        self.root = self.meta.root
        self.revision = self.meta.revision

        if episodes is not None and any(
            episode >= self.meta.total_episodes or episode < 0 for episode in episodes
        ):
            logger.warning(
                f"Some episodes in the provided episodes list are out of range for this dataset ({self.meta.total_episodes})."
            )

        if episode_filter is not None:
            resolved = self.meta.filter_episodes(episode_filter, candidates=episodes)
            if not resolved:
                raise ValueError(
                    "The episode filter did not match any episode. Make sure the filter and episodes list are valid and compatible."
                )
            logger.info(f"The episode filter matched {len(resolved)} episode(s).")
            episodes = resolved
        self.episodes = episodes


        self.reader = DatasetReader(
            meta=self.meta,
            root=self.root,
            episodes=episodes,
            tolerance_s=tolerance_s,
            video_backend=self._video_backend,
            delta_timestamps=delta_timestamps,
            image_transforms=image_transforms,
            return_uint8=self._return_uint8,
        )


        if force_cache_sync or not self.reader.try_load():
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)
            self._download(download_videos)
            self.reader.load_and_activate()


        _has_write_params = streaming_encoding or batch_encoding_size != 1
        if _has_write_params:
            import warnings

            warnings.warn(
                "Passing write-mode parameters (streaming_encoding, batch_encoding_size) to "
                "LeRobotDataset.__init__() is deprecated. Use LeRobotDataset.resume() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            streaming_enc = None
            if streaming_encoding and len(self.meta.video_keys) > 0:
                streaming_enc = self._build_streaming_encoder(
                    self.meta.fps,
                    camera_encoder,
                    depth_encoder,
                    encoder_queue_maxsize,
                    encoder_threads,
                )
            from .dataset_writer import DatasetWriter

            self.writer = DatasetWriter(
                meta=self.meta,
                root=self.root,
                camera_encoder=camera_encoder,
                depth_encoder=depth_encoder,
                encoder_threads=encoder_threads,
                batch_encoding_size=batch_encoding_size,
                streaming_encoder=streaming_enc,
                initial_frames=self.meta.total_frames,
            )
        else:
            self.writer = None

        self._is_finalized = False


    def _require_writer(self, method_name: str) -> None:
        if self.writer is None:
            raise RuntimeError(
                f"Cannot call '{method_name}()' on a read-only dataset. "
                f"Use LeRobotDataset.create() for new recording or "
                f"LeRobotDataset.resume() for resume recording."
            )


    def _ensure_reader(self) -> DatasetReader:

        if self.reader is None:
            self.meta.ensure_readable()
            self.reader = DatasetReader(
                meta=self.meta,
                root=self.root,
                episodes=self.episodes,
                tolerance_s=self.tolerance_s,
                video_backend=self._video_backend,
                delta_timestamps=self.delta_timestamps,
                image_transforms=self.image_transforms,
                return_uint8=self._return_uint8,
            )
        return self.reader

    @staticmethod
    def _build_streaming_encoder(
        fps: int,
        camera_encoder: VideoEncoderConfig | None,
        depth_encoder: DepthEncoderConfig | None,
        encoder_queue_maxsize: int,
        encoder_threads: int | None,
    ) -> StreamingVideoEncoder:
        return StreamingVideoEncoder(
            fps=fps,
            camera_encoder=camera_encoder,
            depth_encoder=depth_encoder,
            queue_maxsize=encoder_queue_maxsize,
            encoder_threads=encoder_threads,
        )


    @property
    def fps(self) -> int:

        return self.meta.fps

    @property
    def num_frames(self) -> int:


        if self.reader is None:
            return self.meta.total_frames
        return self.reader.num_frames

    @property
    def num_episodes(self) -> int:


        if self.reader is None:
            return self.meta.total_episodes
        return self.reader.num_episodes

    @property
    def features(self) -> dict[str, dict]:

        return self.meta.features

    @property
    def hf_dataset(self) -> datasets.Dataset:

        self.reader = self._ensure_reader()
        if self.reader.hf_dataset is None:
            self.reader.load_and_activate()
        return self.reader.hf_dataset


    def add_frame(self, frame: dict) -> None:

        self._require_writer("add_frame")
        self.writer.add_frame(frame)

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:

        self._require_writer("save_episode")
        self.writer.save_episode(episode_data, parallel_encoding)

    def clear_episode_buffer(self, delete_images: bool = True) -> None:

        self._require_writer("clear_episode_buffer")
        self.writer.clear_episode_buffer(delete_images)

    def has_pending_frames(self) -> bool:

        if self.writer is None:
            return False
        return self.writer.episode_buffer is not None and self.writer.episode_buffer["size"] > 0

    def finalize(self):

        if self._is_finalized:
            return
        if self.writer is not None:
            self.writer.finalize()
        self._is_finalized = True


    def __len__(self):

        return self.num_frames

    def __getitem__(self, idx) -> dict:

        if self.writer is not None and not self._is_finalized:
            raise RuntimeError(
                "Cannot read from a dataset that is being recorded. Call finalize() first, then access items."
            )
        reader = self._ensure_reader()
        if reader.hf_dataset is None:

            reader.load_and_activate()
        return reader.get_item(idx)

    def select_columns(self, column_names: str | list[str]):

        return self.hf_dataset.select_columns(column_names)

    def get_raw_item(self, idx) -> dict:

        return self.hf_dataset[idx]

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            f"}})"
        )

    def set_image_transforms(self, image_transforms: Callable | None) -> None:

        if image_transforms is not None and not callable(image_transforms):
            raise TypeError("image_transforms must be callable or None.")
        self.image_transforms = image_transforms
        if self.reader is not None:
            self.reader._image_transforms = image_transforms

    def clear_image_transforms(self) -> None:

        self.set_image_transforms(None)


    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:

        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, repo_id=self.repo_id, **card_kwargs
        )
        card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def _download(self, download_videos: bool = True) -> None:

        ignore_patterns = None if download_videos else "videos/"
        files = None
        if self.episodes is not None:

            files = self.reader.get_episodes_file_paths()

        if self._requested_root is None:
            self.meta.root = Path(
                snapshot_download(
                    self.repo_id,
                    repo_type="dataset",
                    revision=self.revision,
                    cache_dir=HF_LEROBOT_HUB_CACHE,
                    allow_patterns=files,
                    ignore_patterns=ignore_patterns,
                )
            )
        else:
            self._requested_root.mkdir(exist_ok=True, parents=True)
            snapshot_download(
                self.repo_id,
                repo_type="dataset",
                revision=self.revision,
                local_dir=self._requested_root,
                allow_patterns=files,
                ignore_patterns=ignore_patterns,
            )
            self.meta.root = self._requested_root


        self.root = self.meta.root
        self.reader.root = self.meta.root


    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        camera_encoder: VideoEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        metadata_buffer_size: int = 10,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
        video_files_size_in_mb: int | None = None,
        data_files_size_in_mb: int | None = None,
    ) -> "LeRobotDataset":

        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
            metadata_buffer_size=metadata_buffer_size,
            video_files_size_in_mb=video_files_size_in_mb,
            data_files_size_in_mb=data_files_size_in_mb,
        )
        obj.repo_id = obj.meta.repo_id
        obj._requested_root = obj.meta.root
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend is not None else get_safe_default_video_backend()
        obj._return_uint8 = False
        obj._batch_encoding_size = batch_encoding_size
        obj._encoder_threads = encoder_threads


        obj.reader = None

        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(
                fps, camera_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
            )
        from .dataset_writer import DatasetWriter

        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            camera_encoder=camera_encoder,
            depth_encoder=depth_encoder,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
        )

        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._is_finalized = False

        return obj

    @classmethod
    def resume(
        cls,
        repo_id: str,
        root: str | Path | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        camera_encoder: VideoEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        encoder_threads: int | None = None,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
    ) -> "LeRobotDataset":

        if not root:
            raise ValueError(
                "resume() requires an explicit 'root' directory because it creates a DatasetWriter. "
                "Writing into the revision-safe Hub snapshot cache (used when root=None) would corrupt "
                "the shared cache. Please provide a local directory path."
            )
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj._requested_root = Path(root)
        obj.revision = revision if revision else CODEBASE_VERSION
        obj.tolerance_s = tolerance_s
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.episodes = None
        obj._video_backend = video_backend if video_backend else get_safe_default_video_backend()
        obj._return_uint8 = False
        obj._batch_encoding_size = batch_encoding_size

        if obj._requested_root is not None:
            obj._requested_root.mkdir(exist_ok=True, parents=True)


        obj.meta = LeRobotDatasetMetadata(
            obj.repo_id, obj._requested_root, obj.revision, force_cache_sync=force_cache_sync
        )

        obj._encoder_threads = encoder_threads
        obj.root = obj.meta.root


        obj.reader = None

        streaming_enc = None
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            streaming_enc = cls._build_streaming_encoder(
                obj.meta.fps, camera_encoder, depth_encoder, encoder_queue_maxsize, encoder_threads
            )
        from .dataset_writer import DatasetWriter

        obj.writer = DatasetWriter(
            meta=obj.meta,
            root=obj.root,
            camera_encoder=camera_encoder,
            depth_encoder=depth_encoder,
            encoder_threads=encoder_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoder=streaming_enc,
            initial_frames=obj.meta.total_frames,
        )

        if image_writer_processes or image_writer_threads:
            obj.writer.start_image_writer(image_writer_processes, image_writer_threads)

        obj._is_finalized = False

        return obj
