from collections.abc import Sequence
import dataclasses
from enum import Enum
from enum import auto
import json
import logging
from pathlib import Path

import tqdm

import openpi.shared.download as download


class DroidActionSpace(Enum):


    JOINT_POSITION = auto()
    JOINT_VELOCITY = auto()


@dataclasses.dataclass
class RLDSDataset:
    name: str
    version: str
    weight: float
    filter_dict_path: str | None = None


class DroidRldsDataset:
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datasets: Sequence[RLDSDataset],
        *,
        shuffle: bool = True,
        action_chunk_size: int = 16,

        action_space: DroidActionSpace = DroidActionSpace.JOINT_POSITION,
        max_loaded_steps_per_episode: int = 100,

        shuffle_buffer_size: int = 250_000,
        num_parallel_reads: int = -1,
        num_parallel_calls: int = -1,
    ):

        import dlimp as dl
        import tensorflow as tf
        import tensorflow_datasets as tfds


        tf.config.set_visible_devices([], "GPU")


        assert sum(dataset.weight for dataset in datasets) == 1.0, "Dataset weights must sum to 1.0"

        def prepare_single_dataset(dataset_cfg: RLDSDataset):

            ds_name, version = dataset_cfg.name, dataset_cfg.version
            builder = tfds.builder(ds_name, data_dir=data_dir, version=version)
            dataset = dl.DLataset.from_rlds(
                builder, split="train", shuffle=shuffle, num_parallel_reads=num_parallel_reads
            )


            dataset = dataset.filter(
                lambda traj: tf.strings.regex_full_match(
                    traj["traj_metadata"]["episode_metadata"]["file_path"][0], ".*success.*"
                )
            )


            dataset = dataset.repeat()


            filter_dict_path = dataset_cfg.filter_dict_path
            if filter_dict_path is not None:
                cached_filter_dict_path = download.maybe_download(filter_dict_path)
                with Path(cached_filter_dict_path).open("r") as f:
                    filter_dict = json.load(f)
                logging.info(f"Using filter dictionary with {len(filter_dict)} episodes")

                keys_tensor = []
                values_tensor = []

                for episode_key, ranges in tqdm.tqdm(filter_dict.items(), desc="Creating idle filter hash table..."):
                    for start, end in ranges:
                        for t in range(start, end):
                            frame_key = f"{episode_key}--{t}"
                            keys_tensor.append(frame_key)
                            values_tensor.append(True)
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor), default_value=False
                )
                logging.info("Filter hash table initialized")
            else:
                self.filter_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer([""], [True]), default_value=True
                )

            def restructure(traj):


                actions = tf.concat(
                    (
                        (
                            traj["action_dict"]["joint_position"]
                            if action_space == DroidActionSpace.JOINT_POSITION
                            else traj["action_dict"]["joint_velocity"]
                        ),
                        traj["action_dict"]["gripper_position"],
                    ),
                    axis=-1,
                )


                exterior_img = tf.cond(
                    tf.random.uniform(shape=[]) > 0.5,
                    lambda: traj["observation"]["exterior_image_1_left"],
                    lambda: traj["observation"]["exterior_image_2_left"],
                )
                wrist_img = traj["observation"]["wrist_image_left"]

                instruction = tf.random.shuffle(
                    [traj["language_instruction"], traj["language_instruction_2"], traj["language_instruction_3"]]
                )[0]

                traj_len = tf.shape(traj["action"])[0]
                indices = tf.as_string(tf.range(traj_len))


                step_id = (
                    traj["traj_metadata"]["episode_metadata"]["recording_folderpath"]
                    + "--"
                    + traj["traj_metadata"]["episode_metadata"]["file_path"]
                    + "--"
                    + indices
                )
                passes_filter = self.filter_table.lookup(step_id)

                return {
                    "actions": actions,
                    "observation": {
                        "image": exterior_img,
                        "wrist_image": wrist_img,
                        "joint_position": traj["observation"]["joint_position"],
                        "gripper_position": traj["observation"]["gripper_position"],
                    },
                    "prompt": instruction,
                    "step_id": step_id,
                    "passes_filter": passes_filter,
                }

            dataset = dataset.traj_map(restructure, num_parallel_calls)

            def chunk_actions(traj):

                traj_len = tf.shape(traj["actions"])[0]


                action_chunk_indices = tf.broadcast_to(
                    tf.range(action_chunk_size)[None],
                    [traj_len, action_chunk_size],
                ) + tf.broadcast_to(
                    tf.range(traj_len)[:, None],
                    [traj_len, action_chunk_size],
                )


                action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)


                traj["actions"] = tf.gather(traj["actions"], action_chunk_indices)
                return traj

            dataset = dataset.traj_map(chunk_actions, num_parallel_calls)


            dataset = dataset.flatten(num_parallel_calls=num_parallel_calls)


            def filter_from_dict(frame):
                return frame["passes_filter"]

            dataset = dataset.filter(filter_from_dict)


            def remove_passes_filter(frame):
                frame.pop("passes_filter")
                return frame

            dataset = dataset.map(remove_passes_filter)


            def decode_images(traj):
                traj["observation"]["image"] = tf.io.decode_image(
                    traj["observation"]["image"], expand_animations=False, dtype=tf.uint8
                )
                traj["observation"]["wrist_image"] = tf.io.decode_image(
                    traj["observation"]["wrist_image"], expand_animations=False, dtype=tf.uint8
                )
                return traj

            return dataset.frame_map(decode_images, num_parallel_calls)

        logging.info(f"Preparing {len(datasets)} datasets...")
        logging.info("-" * 50)
        for dataset in datasets:
            logging.info(f"    {dataset.name}:{dataset.version} with weight {dataset.weight:.2f}")
        logging.info("-" * 50)
        all_datasets = [prepare_single_dataset(dataset) for dataset in datasets]
        weights = [dataset.weight for dataset in datasets]

        final_dataset = dl.DLataset.sample_from_datasets(all_datasets, weights=weights)
        final_dataset = final_dataset.shuffle(shuffle_buffer_size)
        final_dataset = final_dataset.batch(batch_size)

        final_dataset = final_dataset.with_ram_budget(1)

        self.dataset = final_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        yield from self.dataset.as_numpy_iterator()

    def __len__(self):


        return 20_000_000
