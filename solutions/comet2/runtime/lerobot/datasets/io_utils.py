from pathlib import Path
from typing import Any

import datasets
import numpy as np
import pandas
import pandas as pd
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from datasets.table import embed_table_storage
from PIL import Image as PILImage

from lerobot.utils.io_utils import load_json, write_json
from lerobot.utils.utils import SuppressProgressBars, flatten_dict, unflatten_dict

from .utils import (
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_EPISODES_PATH,
    DEFAULT_SUBTASKS_PATH,
    DEFAULT_TASKS_PATH,
    EPISODES_DIR,
    INFO_PATH,
    STATS_PATH,
    DatasetInfo,
    serialize_dict,
)


def get_parquet_file_size_in_mb(parquet_path: str | Path) -> float:
    metadata = pq.read_metadata(parquet_path)
    total_uncompressed_size = 0
    for row_group in range(metadata.num_row_groups):
        rg_metadata = metadata.row_group(row_group)
        for column in range(rg_metadata.num_columns):
            col_metadata = rg_metadata.column(column)
            total_uncompressed_size += col_metadata.total_uncompressed_size
    return total_uncompressed_size / (1024**2)


def get_hf_dataset_size_in_mb(hf_ds: Dataset) -> int:
    return hf_ds.data.nbytes // (1024**2)


def load_nested_dataset(
    pq_dir: Path, features: datasets.Features | None = None, episodes: list[int] | None = None
) -> Dataset:

    paths = sorted(pq_dir.glob("*/*.parquet"))
    if len(paths) == 0:
        raise FileNotFoundError(f"Provided directory does not contain any parquet file: {pq_dir}")

    with SuppressProgressBars():

        filters = pa_ds.field("episode_index").isin(episodes) if episodes is not None else None
        return Dataset.from_parquet([str(path) for path in paths], filters=filters, features=features)


def get_parquet_num_frames(parquet_path: str | Path) -> int:
    metadata = pq.read_metadata(parquet_path)
    return metadata.num_rows


def get_file_size_in_mb(file_path: Path) -> float:

    file_size_bytes = file_path.stat().st_size
    return file_size_bytes / (1024**2)


def embed_images(dataset: datasets.Dataset) -> datasets.Dataset:


    format = dataset.format
    dataset = dataset.with_format("arrow")
    dataset = dataset.map(embed_table_storage, batched=False)
    dataset = dataset.with_format(**format)
    return dataset


def write_info(info: DatasetInfo, local_dir: Path) -> None:
    write_json(info.to_dict(), local_dir / INFO_PATH)


def load_info(local_dir: Path) -> DatasetInfo:

    raw = load_json(local_dir / INFO_PATH)
    return DatasetInfo.from_dict(raw)


def write_stats(stats: dict, local_dir: Path) -> None:

    serialized_stats = serialize_dict(stats)
    write_json(serialized_stats, local_dir / STATS_PATH)


def cast_stats_to_numpy(stats: dict) -> dict[str, dict[str, np.ndarray]]:

    stats = {key: np.array(value) for key, value in flatten_dict(stats).items()}
    return unflatten_dict(stats)


def load_stats(local_dir: Path) -> dict[str, dict[str, np.ndarray]] | None:

    if not (local_dir / STATS_PATH).exists():
        return None
    stats = load_json(local_dir / STATS_PATH)
    return cast_stats_to_numpy(stats)


def write_tasks(tasks: pandas.DataFrame, local_dir: Path) -> None:
    path = local_dir / DEFAULT_TASKS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tasks.to_parquet(path)


def load_tasks(local_dir: Path) -> pandas.DataFrame:
    tasks = pd.read_parquet(local_dir / DEFAULT_TASKS_PATH)
    tasks.index.name = "task"
    return tasks


def load_subtasks(local_dir: Path) -> pandas.DataFrame | None:

    subtasks_path = local_dir / DEFAULT_SUBTASKS_PATH
    if subtasks_path.exists():
        return pd.read_parquet(subtasks_path)
    return None


def write_episodes(episodes: Dataset, local_dir: Path) -> None:

    episode_size_mb = get_hf_dataset_size_in_mb(episodes)
    if episode_size_mb > DEFAULT_DATA_FILE_SIZE_IN_MB:
        raise NotImplementedError(
            f"Episodes dataset is too large ({episode_size_mb} MB) to write to a single file. "
            f"The current limit is {DEFAULT_DATA_FILE_SIZE_IN_MB} MB. "
            "This function only supports single-file episode metadata. "
        )

    fpath = local_dir / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(fpath)


def load_episodes(local_dir: Path) -> datasets.Dataset:
    episodes = load_nested_dataset(local_dir / EPISODES_DIR)


    episodes = episodes.select_columns([key for key in episodes.features if not key.startswith("stats/")])
    return episodes


def load_image_as_numpy(
    fpath: str | Path, dtype: np.dtype = np.float32, channel_first: bool = True
) -> np.ndarray:

    img = PILImage.open(fpath).convert("RGB")
    img_array = np.array(img, dtype=dtype)
    if channel_first:
        img_array = np.transpose(img_array, (2, 0, 1))
    if np.issubdtype(dtype, np.floating):
        img_array /= 255.0
    return img_array


def hf_transform_to_torch(items_dict: dict[str, list[Any]]) -> dict[str, list[torch.Tensor | str]]:

    for key in items_dict:
        first_item = items_dict[key][0]
        if isinstance(first_item, PILImage.Image):
            from torchvision import transforms

            to_tensor = transforms.ToTensor()
            items_dict[key] = [to_tensor(img) for img in items_dict[key]]
        elif first_item is None:
            pass
        else:
            items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]
    return items_dict


def to_parquet_with_hf_images(
    df: pandas.DataFrame, path: Path, features: datasets.Features | None = None
) -> None:


    ds = datasets.Dataset.from_dict(df.to_dict(orient="list"), features=features)
    ds.to_parquet(path)


def item_to_torch(item: dict) -> dict:

    for key, val in item.items():
        if isinstance(val, (np.ndarray | list)) and key not in ["task"]:

            item[key] = torch.tensor(val)
    return item
