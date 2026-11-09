from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.training.lerobot_compat as _lerobot_compat
import openpi.transforms as _transforms


def _patch_lerobot_seek_mode() -> None:

    try:
        import lerobot.datasets.video_utils as _vu
    except ImportError:
        return
    if getattr(_vu.VideoDecoderCache, "_b1k_patched", False):
        return
    import collections as _collections
    import threading as _threading

    import fsspec
    from torchcodec.decoders import VideoDecoder

    def _mode_for(video_path: str) -> str:
        forced = os.environ.get("B1K_VIDEO_SEEK_MODE")
        if forced:
            return forced
        try:
            import subprocess
            out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=has_b_frames", "-of", "csv=p=0", str(video_path)],
                                 capture_output=True, text=True, timeout=60).stdout.strip()
            mode = "approximate" if out and int(out) == 0 else "exact"
        except Exception:
            mode = "exact"
        logging.info(f"b1k video seek mode: {mode} (probed {video_path})")
        return mode


    _cap = int(os.environ.get("B1K_DECODER_CACHE", "64"))

    def get_decoder(self, video_path):
        video_path = str(video_path)
        with self._lock:
            if not isinstance(self._cache, _collections.OrderedDict):


                self._cache = _collections.OrderedDict(self._cache)
            if video_path in self._cache:
                self._cache.move_to_end(video_path)
            else:
                mode = getattr(type(self), "_b1k_seek_mode", None) or _mode_for(video_path)
                type(self)._b1k_seek_mode = mode
                file_handle = fsspec.open(video_path).__enter__()
                try:
                    decoder = VideoDecoder(file_handle, seek_mode=mode)
                except Exception:
                    file_handle.close()
                    raise
                self._cache[video_path] = (decoder, file_handle)
                while len(self._cache) > _cap:
                    _, (_, old_handle) = self._cache.popitem(last=False)
                    try:
                        old_handle.close()
                    except Exception:
                        pass
            return self._cache[video_path][0]

    def _b1k_init(self):
        self._cache = _collections.OrderedDict()
        self._lock = _threading.Lock()

    _vu.VideoDecoderCache.get_decoder = get_decoder
    _vu.VideoDecoderCache.__init__ = _b1k_init
    _vu.VideoDecoderCache._b1k_patched = True
    logging.info(f"b1k decoder cache: at most {_cap} open videos per worker (upstream: unbounded)")


_patch_lerobot_seek_mode()

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):


    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):


    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):


    def data_config(self) -> _config.DataConfig:

        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:


        retries = int(os.environ.get("B1K_GETITEM_RETRIES", "8"))
        for attempt in range(retries + 1):
            try:
                return self._transform(self._dataset[index])
            except Exception as e:
                if attempt >= retries:
                    raise
                logging.warning("sample %s failed (%s: %s); resampling (attempt %d/%d)",
                                index, type(e).__name__, str(e)[:160], attempt + 1, retries)
                index = int(np.random.randint(len(self._dataset)))
        raise RuntimeError("unreachable")

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:


                batch_size = next(v.shape[0] for v in sample.values())


                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]


                transformed = [self._transform(s) for s in individual_samples]


                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)

            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_b1k_dataset(data_config: _config.DataConfig, action_horizon: int) -> Dataset:

    if isinstance(data_config.repo_id, list):
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(
            repo_id=data_config.repo_id[0], root=os.path.join(data_config.dataset_root, data_config.repo_id[0])
        )
        dataset_kwargs = {"repo_ids": data_config.repo_id, **data_config.dataset_kwargs}
    else:
        dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(
            repo_id=data_config.repo_id, root=data_config.dataset_root
        )
        dataset_kwargs = {"repo_id": data_config.repo_id, **data_config.dataset_kwargs}
    dataset = data_config.data_cls(
        root=data_config.dataset_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        **dataset_kwargs,
    )


    if os.environ.get("B1K_SKIP_DEPTH", "0") == "1":
        _meta = getattr(dataset, "meta", None)
        _depth = set(getattr(_meta, "depth_keys", []) or [])
        _skipped = []
        for _holder in (getattr(_meta, "features", None), (getattr(_meta, "info", None) or {}).get("features")):
            if isinstance(_holder, dict):
                for _k, _ft in _holder.items():
                    if _k in _depth and isinstance(_ft, dict) and _ft.get("dtype") == "video":
                        _ft["dtype"] = "_skipped_depth_video"
                        _skipped.append(_k)
        _left = [k for k in _meta.video_keys if k in _depth]
        if _left:
            raise RuntimeError(f"B1K_SKIP_DEPTH: failed to disable depth decoding for {_left}")
        logging.info(f"B1K_SKIP_DEPTH=1: depth decoding disabled for {sorted(set(_skipped))}; "
                     f"still decoding {_meta.video_keys}")

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(_lerobot_compat.tasks_from_metadata(dataset_meta))]
        )

    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = _lerobot_compat.LeRobotDatasetMetadata(repo_id)
    dataset = _lerobot_compat.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(_lerobot_compat.tasks_from_metadata(dataset_meta))]
        )

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:

    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:

    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:

    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


class TaskTemperedSampler(torch.utils.data.Sampler):


    def __init__(self, ranges, *, alpha: float, num_samples: int, seed: int, rank: int = 0, resume_step: int = 0):
        self._starts = np.asarray([a for a, _ in ranges], np.int64)
        self._sizes = np.asarray([b - a for a, b in ranges], np.int64)
        w = self._sizes.astype(np.float64) ** alpha
        self._p = w / w.sum()
        self._n, self._key, self._pass = int(num_samples), (int(seed), int(rank), int(resume_step)), 0

    def shares(self) -> np.ndarray:
        return self._p

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([*self._key, self._pass]))
        self._pass += 1
        left = self._n
        while left > 0:
            k = min(left, 1_000_000)
            t = rng.choice(len(self._p), size=k, p=self._p)
            idx = self._starts[t] + (rng.random(k) * self._sizes[t]).astype(np.int64)
            yield from idx.tolist()
            left -= k


def _b1k_task_ranges(dataset_len: int) -> list[tuple[int, int]]:

    import glob

    import pyarrow.parquet as pq

    root = os.environ["B1K_DATASET_ROOT"]
    rows = []
    for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/*.parquet")):
        rows += pq.read_table(f, columns=["task_index", "dataset_from_index", "dataset_to_index"]).to_pylist()
    by = {}
    for r in rows:
        a, b = by.get(r["task_index"], (None, None))
        by[r["task_index"]] = (r["dataset_from_index"] if a is None else min(a, r["dataset_from_index"]),
                               r["dataset_to_index"] if b is None else max(b, r["dataset_to_index"]))
    ranges = sorted(by.values())
    total = sum(r["dataset_to_index"] - r["dataset_from_index"] for r in rows)
    if total != dataset_len or sum(b - a for a, b in ranges) != total or ranges[0][0] != 0 or ranges[-1][1] != dataset_len:
        raise ValueError(f"B1K_TASK_ALPHA: task blocks do not tile the dataset ({len(ranges)} tasks, {total} frames in "
                         f"metadata vs {dataset_len} in the dataset); refusing to sample")
    return ranges


def create_b1k_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = create_b1k_dataset(data_config=data_config, action_horizon=config.model.action_horizon)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)


    sampler = None


    _resume = int(os.environ.get("B1K_RESUME_STEP", "0"))
    _alpha = os.environ.get("B1K_TASK_ALPHA")
    if _alpha:
        sampler = TaskTemperedSampler(
            _b1k_task_ranges(len(dataset)), alpha=float(_alpha), num_samples=len(dataset) // jax.process_count(),
            seed=config.seed, rank=jax.process_index(), resume_step=_resume)
        _sh = np.sort(sampler.shares())
        logging.info(f"task-tempered sampler: alpha={_alpha} process {jax.process_index()}/{jax.process_count()} "
                     f"resume_step={_resume}; task shares min {100*_sh[0]:.2f}% median {100*np.median(_sh):.2f}% "
                     f"max {100*_sh[-1]:.2f}% over {len(_sh)} tasks")
    elif jax.process_count() > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=jax.process_count(), rank=jax.process_index(),
            shuffle=shuffle, seed=config.seed, drop_last=True,
        )
        sampler.set_epoch(_resume)
        _first = [i for _, i in zip(range(3), iter(sampler))]
        logging.info(f"multi-host sampler (b1k): process {jax.process_index()}/{jax.process_count()} gets "
                     f"{len(sampler)} of {len(dataset)} samples; resume_step={_resume}; first indices {_first}")

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        sampler=sampler,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:

    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)


    sampler = None
    if framework == "jax" and jax.process_count() > 1:


        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=jax.process_count(), rank=jax.process_index(),
            shuffle=shuffle, seed=seed, drop_last=True,
        )
        _first = next(iter(sampler), None)
        logging.info(f"multi-host sampler: process {jax.process_index()}/{jax.process_count()} gets "
                     f"{len(sampler)} of {len(dataset)} samples (first index {_first}); shards are disjoint by construction")
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:

    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:


    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):


        if jax.process_count() > 1 and os.environ.get("B1K_MULTIHOST") != "1":
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")


        self._sharding = sharding
        if sharding is None and framework == "jax":

            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,


            in_order=os.environ.get("B1K_LOADER_IN_ORDER", "1") != "0",
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1

                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):


    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:


    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    _patch_lerobot_seek_mode()


class RLDSDataLoader:


    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:

            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
