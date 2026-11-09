import contextlib
import dataclasses
import importlib.resources
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import numpy as np
import packaging.version
import torch
from huggingface_hub import DatasetCard, DatasetCardData, HfApi
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.utils.utils import flatten_dict, unflatten_dict

V30_MESSAGE = """
The dataset you requested ({repo_id}) is in {version} format.

We introduced a new format since v3.0 which is not backward compatible with v2.1.
Please, update your dataset to the new format using this command:
```
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id={repo_id}
```

If you already have a converted version uploaded to the hub, then this error might be because of
an older version in your local cache. Consider deleting the cached version and retrying.

If you encounter a problem, contact LeRobot maintainers on [Discord](https://discord.com/invite/s3KuuzsPFb)
or open an [issue on GitHub](https://github.com/huggingface/lerobot/issues/new/choose).
"""

FUTURE_MESSAGE = """
The dataset you requested ({repo_id}) is only available in {version} format.
As we cannot ensure forward compatibility with it, please update your current version of lerobot.
"""


class CompatibilityError(Exception): ...


class BackwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        if version.major == 2 and version.minor == 1:
            message = V30_MESSAGE.format(repo_id=repo_id, version=version)
        else:
            raise NotImplementedError(
                "Contact the maintainer on [Discord](https://discord.com/invite/s3KuuzsPFb)."
            )
        super().__init__(message)


class ForwardCompatibilityError(CompatibilityError):
    def __init__(self, repo_id: str, version: packaging.version.Version):
        message = FUTURE_MESSAGE.format(repo_id=repo_id, version=version)
        super().__init__(message)


logger = logging.getLogger(__name__)


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"

EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"

CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DEFAULT_TASKS_PATH = "meta/tasks.parquet"
DEFAULT_SUBTASKS_PATH = "meta/subtasks.parquet"
DEFAULT_EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
DEFAULT_VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
DEFAULT_IMAGE_PATH = "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.png"
DEFAULT_DEPTH_PATH = "images/{image_key}/episode-{episode_index:06d}/frame-{frame_index:06d}.tiff"

LEGACY_EPISODES_PATH = "meta/episodes.jsonl"
LEGACY_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
LEGACY_TASKS_PATH = "meta/tasks.jsonl"


@dataclass
class DatasetInfo:


    codebase_version: str
    fps: int
    features: dict[str, dict]


    total_episodes: int = 0
    total_frames: int = 0
    total_tasks: int = 0


    chunks_size: int = field(default=DEFAULT_CHUNK_SIZE)
    data_files_size_in_mb: int = field(default=DEFAULT_DATA_FILE_SIZE_IN_MB)
    video_files_size_in_mb: int = field(default=DEFAULT_VIDEO_FILE_SIZE_IN_MB)


    data_path: str = field(default=DEFAULT_DATA_PATH)
    video_path: str | None = field(default=DEFAULT_VIDEO_PATH)


    robot_type: str | None = None
    splits: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:


        for ft in self.features.values():
            if isinstance(ft.get("shape"), list):
                ft["shape"] = tuple(ft["shape"])

        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.chunks_size <= 0:
            raise ValueError(f"chunks_size must be positive, got {self.chunks_size}")
        if self.data_files_size_in_mb <= 0:
            raise ValueError(f"data_files_size_in_mb must be positive, got {self.data_files_size_in_mb}")
        if self.video_files_size_in_mb <= 0:
            raise ValueError(f"video_files_size_in_mb must be positive, got {self.video_files_size_in_mb}")

    def to_dict(self) -> dict:

        d = dataclasses.asdict(self)
        for ft in d["features"].values():
            if isinstance(ft.get("shape"), tuple):
                ft["shape"] = list(ft["shape"])
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetInfo":

        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            logger.warning(f"Unknown fields in DatasetInfo: {unknown}. These will be ignored.")
        return cls(**{k: v for k, v in data.items() if k in known})


    def __getitem__(self, key: str):
        import warnings

        warnings.warn(
            f"Accessing DatasetInfo with dict-style syntax info['{key}'] is deprecated. "
            f"Use attribute access info.{key} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(key) from err

    def __setitem__(self, key: str, value) -> None:
        import warnings

        warnings.warn(
            f"Setting DatasetInfo with dict-style syntax info['{key}'] = ... is deprecated. "
            f"Use attribute assignment info.{key} = ... instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, key):
            raise KeyError(f"DatasetInfo has no field '{key}'")
        setattr(self, key, value)

    def __contains__(self, key: str) -> bool:

        return hasattr(self, key)

    def get(self, key: str, default=None):

        try:
            return getattr(self, key)
        except AttributeError:
            return default


def has_legacy_hub_download_metadata(root: Path) -> bool:

    return (root / ".cache" / "huggingface" / "download").exists()


def update_chunk_file_indices(chunk_idx: int, file_idx: int, chunks_size: int) -> tuple[int, int]:
    if file_idx == chunks_size - 1:
        file_idx = 0
        chunk_idx += 1
    else:
        file_idx += 1
    return chunk_idx, file_idx


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:

    serialized_dict = {}
    for key, value in flatten_dict(stats).items():
        if isinstance(value, (torch.Tensor | np.ndarray)):
            serialized_dict[key] = value.tolist()
        elif isinstance(value, list) and isinstance(value[0], (int | float | list)):
            serialized_dict[key] = value
        elif isinstance(value, np.generic):
            serialized_dict[key] = value.item()
        elif isinstance(value, (int | float)):
            serialized_dict[key] = value
        else:
            raise NotImplementedError(f"The value '{value}' of type '{type(value)}' is not supported.")
    return unflatten_dict(serialized_dict)


def is_valid_version(version: str) -> bool:

    try:
        packaging.version.parse(version)
        return True
    except packaging.version.InvalidVersion:
        return False


def check_version_compatibility(
    repo_id: str,
    version_to_check: str | packaging.version.Version,
    current_version: str | packaging.version.Version,
    enforce_breaking_major: bool = True,
) -> None:

    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logging.warning(FUTURE_MESSAGE.format(repo_id=repo_id, version=v_check))


def get_repo_versions(repo_id: str) -> list[packaging.version.Version]:

    api = HfApi()
    repo_refs = api.list_repo_refs(repo_id, repo_type="dataset")
    repo_refs = [b.name for b in repo_refs.branches + repo_refs.tags]
    repo_versions = []
    for ref in repo_refs:
        with contextlib.suppress(packaging.version.InvalidVersion):
            repo_versions.append(packaging.version.parse(ref))

    return repo_versions


def get_safe_version(repo_id: str, version: str | packaging.version.Version) -> str:

    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id)

    if not hub_versions:
        raise RevisionNotFoundError(
            f"""Your dataset must be tagged with a codebase version.
            Assuming _version_ is the codebase_version value in the info.json, you can run this:
            ```python
            from huggingface_hub import HfApi

            hub_api = HfApi()
            hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
            ```
            """
        )

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v for v in hub_versions if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(f"Revision {version} for {repo_id} not found, using version v{return_version}")
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))


def create_branch(repo_id: str, *, branch: str, repo_type: str | None = None) -> None:

    api = HfApi()

    branches = api.list_repo_refs(repo_id, repo_type=repo_type).branches
    refs = [branch.ref for branch in branches]
    ref = f"refs/heads/{branch}"
    if ref in refs:
        api.delete_branch(repo_id, repo_type=repo_type, branch=branch)

    api.create_branch(repo_id, repo_type=repo_type, branch=branch)


def create_lerobot_dataset_card(
    tags: list | None = None,
    dataset_info: DatasetInfo | None = None,
    **kwargs,
) -> DatasetCard:

    card_tags = ["LeRobot"]

    if tags:
        card_tags += tags
    if dataset_info:
        dataset_structure = "[meta/info.json](meta/info.json):\n"
        dataset_structure += f"```json\n{json.dumps(dataset_info.to_dict(), indent=4)}\n```\n"
        kwargs = {**kwargs, "dataset_structure": dataset_structure}
    card_data = DatasetCardData(
        license=kwargs.get("license"),
        tags=card_tags,
        task_categories=["robotics"],
        configs=[
            {
                "config_name": "default",
                "data_files": "data/*/*.parquet",
            }
        ],
    )

    card_template = (importlib.resources.files("lerobot.datasets") / "card_template.md").read_text()

    return DatasetCard.from_template(
        card_data=card_data,
        template_str=card_template,
        **kwargs,
    )


def is_float_in_list(target, float_list, threshold=1e-6):
    return any(abs(target - x) <= threshold for x in float_list)


def find_float_index(target, float_list, threshold=1e-6):
    for i, x in enumerate(float_list):
        if abs(target - x) <= threshold:
            return i
    return -1


def safe_shard(dataset: datasets.IterableDataset, index: int, num_shards: int) -> datasets.Dataset:

    shard_idx = min(dataset.num_shards, index + 1) - 1

    return dataset.shard(num_shards, index=shard_idx)
