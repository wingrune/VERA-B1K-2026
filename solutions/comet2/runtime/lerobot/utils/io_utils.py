import json
import logging
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]
T = TypeVar("T", bound=JsonLike)


def load_json(fpath: Path) -> Any:

    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:

    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_video(video_path: str | Path, stacked_frames: list, fps: int) -> None:

    from .import_utils import require_package

    require_package("av", extra="av-dep")
    import av

    with av.open(str(video_path), mode="w") as container:
        orig_height, orig_width = stacked_frames[0].shape[:2]

        height = orig_height if orig_height % 2 == 0 else orig_height - 1
        width = orig_width if orig_width % 2 == 0 else orig_width - 1
        if height != orig_height or width != orig_width:
            logger.warning(
                "Frame dimensions %dx%d are not even; cropping to %dx%d for yuv420p compatibility.",
                orig_width,
                orig_height,
                width,
                height,
            )
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for frame_array in stacked_frames:
            if height != orig_height or width != orig_width:
                frame_array = frame_array[:height, :width]
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def deserialize_json_into_object(fpath: Path, obj: T) -> T:

    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)

    def _deserialize(target, source):


        if isinstance(target, dict):
            if not isinstance(source, dict):
                raise TypeError(f"Type mismatch: expected dict, got {type(source)}")


            if target.keys() != source.keys():
                raise ValueError(
                    f"Dictionary keys do not match.\nExpected: {target.keys()}, got: {source.keys()}"
                )


            for k in target:
                target[k] = _deserialize(target[k], source[k])

            return target


        elif isinstance(target, list):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list, got {type(source)}")


            if len(target) != len(source):
                raise ValueError(f"List length mismatch: expected {len(target)}, got {len(source)}")


            for i in range(len(target)):
                target[i] = _deserialize(target[i], source[i])

            return target


        elif isinstance(target, tuple):
            if not isinstance(source, list):
                raise TypeError(f"Type mismatch: expected list (for tuple), got {type(source)}")

            if len(target) != len(source):
                raise ValueError(f"Tuple length mismatch: expected {len(target)}, got {len(source)}")


            converted_items = []
            for t_item, s_item in zip(target, source, strict=False):
                converted_items.append(_deserialize(t_item, s_item))


            return tuple(converted_items)


        else:

            if type(target) is not type(source):
                raise TypeError(f"Type mismatch: expected {type(target)}, got {type(source)}")
            return source


    updated_obj = _deserialize(obj, data)
    return updated_obj
