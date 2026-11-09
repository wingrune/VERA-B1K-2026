import contextlib
import glob
import importlib
import logging
import queue
import shutil
import tempfile
import threading
import warnings
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import av
import fsspec
import numpy as np
import pyarrow as pa
import torch
from datasets.features.features import register_feature
from PIL import Image

from lerobot.configs.video import (
    DepthEncoderConfig,
    VideoEncoderConfig,
    camera_encoder_defaults,
    depth_encoder_defaults,
)
from lerobot.utils.import_utils import get_safe_default_video_backend

from .depth_utils import quantize_depth
from .pyav_utils import get_pix_fmt_channels

logger = logging.getLogger(__name__)


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:

    if backend != "pyav" and is_depth:
        logger.info("Decoding depth maps is only supported with the 'pyav' backend.")

        return decode_video_frames_pyav(video_path, timestamps, tolerance_s, return_uint8=True, is_depth=True)

    if backend is None:
        backend = get_safe_default_video_backend()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s, return_uint8=return_uint8)
    elif backend == "pyav":
        return decode_video_frames_pyav(
            video_path, timestamps, tolerance_s, return_uint8=return_uint8, is_depth=is_depth
        )
    elif backend == "video_reader":
        logger.warning("backend='video_reader' is deprecated and now aliases to 'pyav'.")
        return decode_video_frames_pyav(
            video_path, timestamps, tolerance_s, return_uint8=return_uint8, is_depth=is_depth
        )
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    return_uint8: bool = False,
    is_depth: bool = False,
) -> torch.Tensor:


    video_path = str(video_path)


    first_ts = min(timestamps)
    last_ts = max(timestamps)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []


    with av.open(video_path) as container:
        stream = container.streams.video[0]
        container.seek(int(first_ts * av.time_base), backward=True)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            if log_loaded_timestamps:
                logger.info(f"frame loaded at timestamp={current_ts:.4f}")
            if is_depth:
                arr = frame.to_ndarray(format="gray12le")
                loaded_frames.append(torch.from_numpy(arr).unsqueeze(0).contiguous())
            else:
                arr = frame.to_ndarray(format="rgb24")

                loaded_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

    if not loaded_frames:
        raise FrameTimestampError(
            f"No frames could be decoded from {video_path} in the timestamp range [{first_ts}, {last_ts}]."
        )

    query_ts = torch.tensor(timestamps)
    loaded_ts_t = torch.tensor(loaded_ts)


    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts_t}"
            f"\nvideo: {video_path}"
            f"\nbackend: pyav"
        )


    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts_t[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )

    if return_uint8 or is_depth:
        return closest_frames


    closest_frames = closest_frames.type(torch.float32) / 255
    return closest_frames


class VideoDecoderCache:


    def __init__(self):
        self._cache: dict[str, tuple[Any, Any]] = {}
        self._lock = Lock()

    def get_decoder(self, video_path: str):

        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError(
                "'torchcodec' is required but not installed. "
                "Install it with: pip install 'lerobot[dataset]' (or uv pip install 'lerobot[dataset]')"
            )

        video_path = str(video_path)

        with self._lock:
            if video_path not in self._cache:
                file_handle = fsspec.open(video_path).__enter__()
                try:
                    decoder = VideoDecoder(file_handle, seek_mode="approximate")
                except Exception:
                    file_handle.close()
                    raise
                self._cache[video_path] = (decoder, file_handle)

            return self._cache[video_path][0]

    def clear(self):

        with self._lock:
            for _, file_handle in self._cache.values():
                file_handle.close()
            self._cache.clear()

    def size(self) -> int:

        with self._lock:
            return len(self._cache)


class FrameTimestampError(ValueError):


    pass


_default_decoder_cache = VideoDecoderCache()


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache: VideoDecoderCache | None = None,
    return_uint8: bool = False,
) -> torch.Tensor:

    if decoder_cache is None:
        decoder_cache = _default_decoder_cache


    decoder = decoder_cache.get_decoder(str(video_path))

    loaded_ts = []
    loaded_frames = []


    metadata = decoder.metadata
    average_fps = metadata.average_fps

    frame_indices = [round(ts * average_fps) for ts in timestamps]

    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logger.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)


    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
        )


    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
        )

    if return_uint8:
        return closest_frames


    closest_frames = (closest_frames / 255.0).type(torch.float32)
    return closest_frames


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    video_encoder: VideoEncoderConfig | None = None,
    encoder_threads: int | None = None,
    *,
    log_level: int | None = av.logging.WARNING,
    overwrite: bool = False,
) -> None:

    if video_encoder is None:
        video_encoder = camera_encoder_defaults()
    vcodec = video_encoder.vcodec
    pix_fmt = video_encoder.pix_fmt

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    if video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {video_path}. Skipping encoding.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)


    suffix = ".png" if not isinstance(video_encoder, DepthEncoderConfig) else ".tiff"
    template = "frame-" + ("[0-9]" * 6) + suffix
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("-")[-1].split(".")[0])
    )

    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    video_options = video_encoder.get_codec_options(encoder_threads, as_strings=True)


    if log_level is not None:

        logging.getLogger("libav").setLevel(log_level)


    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height


        for input_data in input_list:
            with Image.open(input_data) as input_image:
                input_image = input_image.convert("RGB")
                input_frame = av.VideoFrame.from_image(input_image)
                packet = output_stream.encode(input_frame)
                if packet:
                    output.mux(packet)


        packet = output_stream.encode()
        if packet:
            output.mux(packet)


    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def concatenate_video_files(
    input_video_paths: list[Path | str],
    output_video_path: Path,
    overwrite: bool = True,
    compatibility_check: bool = False,
):


    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {output_video_path}. Skipping concatenation.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")


    if compatibility_check:
        reference_video_info = get_video_info(input_video_paths[0])
        for input_path in input_video_paths[1:]:
            video_info = get_video_info(input_path)
            if (
                video_info["video.height"] != reference_video_info["video.height"]
                or video_info["video.width"] != reference_video_info["video.width"]
                or video_info["video.fps"] != reference_video_info["video.fps"]
                or video_info["video.codec"] != reference_video_info["video.codec"]
                or video_info["video.pix_fmt"] != reference_video_info["video.pix_fmt"]
            ):
                raise ValueError(
                    f"Input video {input_path} is not compatible with the reference video {input_video_paths[0]}."
                )


    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as tmp_concatenate_file:
        tmp_concatenate_file.write("ffconcat version 1.0\n")
        for input_path in input_video_paths:
            tmp_concatenate_file.write(f"file '{str(input_path.resolve())}'\n")
        tmp_concatenate_file.flush()
        tmp_concatenate_path = tmp_concatenate_file.name


    input_container = av.open(
        tmp_concatenate_path, mode="r", format="concat", options={"safe": "0"}
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    output_container = av.open(
        tmp_output_video_path, mode="w", options={"movflags": "faststart"}
    )


    stream_map = {}
    for input_stream in input_container.streams:
        if input_stream.type in ("video", "audio", "subtitle"):
            stream_map[input_stream.index] = output_container.add_stream_from_template(
                template=input_stream, opaque=True
            )


            stream_map[input_stream.index].time_base = input_stream.time_base


    for packet in input_container.demux():

        if packet.stream.index not in stream_map:
            continue


        if packet.dts is None:
            continue

        output_stream = stream_map[packet.stream.index]
        packet.stream = output_stream
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    shutil.move(tmp_output_video_path, output_video_path)
    Path(tmp_concatenate_path).unlink()


class _CameraEncoderThread(threading.Thread):


    def __init__(
        self,
        video_path: Path,
        fps: int,
        video_encoder: VideoEncoderConfig,
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
        encoder_threads: int | None = None,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.fps = fps
        self.video_encoder = video_encoder
        self.is_depth = isinstance(video_encoder, DepthEncoderConfig)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.encoder_threads = encoder_threads

    def run(self) -> None:
        from .compute_stats import RunningQuantileStats, auto_downsample_height_width

        container = None
        output_stream = None
        stats_tracker = RunningQuantileStats()
        frame_count = 0

        try:
            logging.getLogger("libav").setLevel(av.logging.WARNING)

            while True:
                try:
                    frame_data = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

                if frame_data is None:

                    break


                if isinstance(frame_data, np.ndarray):
                    if frame_data.ndim == 3 and frame_data.shape[0] in (1, 3):

                        frame_data = frame_data.transpose(1, 2, 0)
                    if not self.is_depth and frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8)


                if container is None:
                    height, width = frame_data.shape[:2]
                    Path(self.video_path).parent.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.video_path), "w")
                    output_stream = container.add_stream(
                        self.video_encoder.vcodec,
                        self.fps,
                        options=self.video_encoder.get_codec_options(self.encoder_threads, as_strings=True),
                    )
                    output_stream.pix_fmt = self.video_encoder.pix_fmt
                    output_stream.width = width
                    output_stream.height = height
                    output_stream.time_base = Fraction(1, self.fps)


                if not self.is_depth:
                    pil_img = Image.fromarray(frame_data)
                    video_frame = av.VideoFrame.from_image(pil_img)
                else:
                    video_frame = quantize_depth(
                        frame_data,
                        depth_min=self.video_encoder.depth_min,
                        depth_max=self.video_encoder.depth_max,
                        shift=self.video_encoder.shift,
                        use_log=self.video_encoder.use_log,
                        video_backend=self.video_encoder.video_backend,
                    )
                video_frame.pts = frame_count
                video_frame.time_base = Fraction(1, self.fps)
                packet = output_stream.encode(video_frame)
                if packet:
                    container.mux(packet)


                img_chw = frame_data.transpose(2, 0, 1)
                img_downsampled = auto_downsample_height_width(img_chw)

                channels = img_downsampled.shape[0]
                img_for_stats = img_downsampled.transpose(1, 2, 0).reshape(-1, channels)
                stats_tracker.update(img_for_stats)

                frame_count += 1


            if output_stream is not None:
                packet = output_stream.encode()
                if packet:
                    container.mux(packet)

            if container is not None:
                container.close()

            av.logging.restore_default_callback()


            if frame_count >= 2:
                stats = stats_tracker.get_statistics()
                self.result_queue.put(("ok", stats))
            else:
                self.result_queue.put(("ok", None))

        except Exception as e:
            logger.error(f"Encoder thread error: {e}")
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()
            self.result_queue.put(("error", str(e)))


class StreamingVideoEncoder:


    def __init__(
        self,
        fps: int,
        camera_encoder: VideoEncoderConfig | None = None,
        depth_encoder: DepthEncoderConfig | None = None,
        queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):

        self.fps = fps
        self._camera_encoder = camera_encoder or camera_encoder_defaults()
        self._depth_encoder = depth_encoder or depth_encoder_defaults()
        self._encoder_threads = encoder_threads
        self.queue_maxsize = queue_maxsize

        self._frame_queues: dict[str, queue.Queue] = {}
        self._result_queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, _CameraEncoderThread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._video_paths: dict[str, Path] = {}
        self._dropped_frames: dict[str, int] = {}
        self._episode_active = False
        self._closed = False

    def start_episode(
        self, video_keys: list[str], temp_dir: Path, depth_video_keys: list[str] | None = None
    ) -> None:

        if self._episode_active:
            self.cancel_episode()

        self._dropped_frames.clear()

        if depth_video_keys is None:
            depth_video_keys = []

        for video_key in video_keys:
            frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_maxsize)
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()

            temp_video_dir = Path(tempfile.mkdtemp(dir=temp_dir))
            video_path = temp_video_dir / f"{video_key.replace('/', '_')}_streaming.mp4"

            encoder = self._depth_encoder if video_key in depth_video_keys else self._camera_encoder
            encoder_thread = _CameraEncoderThread(
                video_path=video_path,
                fps=self.fps,
                video_encoder=encoder,
                frame_queue=frame_queue,
                result_queue=result_queue,
                stop_event=stop_event,
                encoder_threads=self._encoder_threads,
            )
            encoder_thread.start()

            self._frame_queues[video_key] = frame_queue
            self._result_queues[video_key] = result_queue
            self._threads[video_key] = encoder_thread
            self._stop_events[video_key] = stop_event
            self._video_paths[video_key] = video_path

        self._episode_active = True

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:

        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")

        thread = self._threads[video_key]
        if not thread.is_alive():

            try:
                status, msg = self._result_queues[video_key].get_nowait()
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} crashed: {msg}")
            except queue.Empty:
                pass
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")

        try:
            self._frame_queues[video_key].put(image.copy(), timeout=0.1)
        except queue.Full:
            self._dropped_frames[video_key] = self._dropped_frames.get(video_key, 0) + 1
            count = self._dropped_frames[video_key]

            if count == 1 or count % 10 == 0:
                logger.warning(
                    f"Encoder queue full for {video_key}, dropped {count} frame(s). "
                    f"Consider using vcodec='auto' for hardware encoding or increasing encoder_queue_maxsize."
                )

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:

        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")

        results = {}


        for video_key, count in self._dropped_frames.items():
            if count > 0:
                logger.warning(f"Episode finished with {count} dropped frame(s) for {video_key}.")


        for video_key in self._frame_queues:
            self._frame_queues[video_key].put(None)


        for video_key in self._threads:
            self._threads[video_key].join(timeout=120)
            if self._threads[video_key].is_alive():
                logger.error(f"Encoder thread for {video_key} did not finish in time")
                self._stop_events[video_key].set()
                self._threads[video_key].join(timeout=5)
                results[video_key] = (self._video_paths[video_key], None)
                continue

            try:
                status, data = self._result_queues[video_key].get(timeout=5)
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} failed: {data}")
                results[video_key] = (self._video_paths[video_key], data)
            except queue.Empty:
                logger.error(f"No result from encoder thread for {video_key}")
                results[video_key] = (self._video_paths[video_key], None)

        self._cleanup()
        self._episode_active = False
        return results

    def cancel_episode(self) -> None:

        if not self._episode_active:
            return


        for video_key in self._stop_events:
            self._stop_events[video_key].set()


        for video_key in self._threads:
            self._threads[video_key].join(timeout=5)


            video_path = self._video_paths.get(video_key)
            if video_path is not None and video_path.exists():
                shutil.rmtree(str(video_path.parent), ignore_errors=True)

        self._cleanup()
        self._episode_active = False

    def close(self) -> None:

        if self._closed:
            return
        if self._episode_active:
            self.cancel_episode()
        self._closed = True

    def _cleanup(self) -> None:

        for q in self._frame_queues.values():
            with contextlib.suppress(Exception):
                while not q.empty():
                    q.get_nowait()
        self._frame_queues.clear()
        self._result_queues.clear()
        self._threads.clear()
        self._stop_events.clear()
        self._video_paths.clear()


@dataclass
class VideoFrame:


    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )

    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:

    logging.getLogger("libav").setLevel(av.logging.WARNING)


    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:

            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name


        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate


        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True


    av.logging.restore_default_callback()

    return audio_info


def get_video_info(
    video_path: Path | str,
    video_encoder: VideoEncoderConfig | None = None,
) -> dict:

    logging.getLogger("libav").setLevel(av.logging.WARNING)


    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:

            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt


        video_info["video.fps"] = int(video_stream.base_rate)
        video_info["video.channels"] = get_pix_fmt_channels(video_stream.pix_fmt)


    av.logging.restore_default_callback()


    video_info.update(**get_audio_info(video_path))


    if video_encoder is not None:
        for field_name, field_value in asdict(video_encoder).items():

            if field_name == "vcodec":
                continue
            video_info.setdefault(f"video.{field_name}", field_value)

    video_info["is_depth_map"] = isinstance(video_encoder, DepthEncoderConfig)

    return video_info


def get_video_duration_in_s(video_path: Path | str) -> float:

    with av.open(str(video_path)) as container:

        video_stream = container.streams.video[0]

        if video_stream.duration is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:

            duration = float(container.duration / av.time_base)
    return duration


class VideoEncodingManager:


    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        writer = self.dataset.writer
        if writer is not None:
            if exc_type is not None and writer._streaming_encoder is not None:
                writer.cancel_pending_videos()


            self.dataset.finalize()


            if exc_type is not None and writer._streaming_encoder is None:
                writer.cleanup_interrupted_episode(self.dataset.num_episodes)
        else:
            self.dataset.finalize()


        img_dir = self.dataset.root / "images"
        if img_dir.exists():
            png_files = list(img_dir.rglob("*.png"))
            if len(png_files) == 0:
                shutil.rmtree(img_dir)
                logger.debug("Cleaned up empty images directory")
            else:
                logger.debug(f"Images directory is not empty, containing {len(png_files)} PNG files")

        return False
