import math
from typing import Literal

import av
import numpy as np
import torch
from numpy.typing import NDArray

from lerobot.configs.video import (
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_DEPTH_PIX_FMT,
    DEFAULT_DEPTH_SHIFT,
    DEFAULT_DEPTH_USE_LOG,
    DEPTH_QMAX,
)

from .pyav_utils import write_u16_plane

_MM_PER_METRE = 1000.0
_UINT16_MAX = 65535


def _validate_log_quant_params(depth_min: float, shift: float) -> None:

    if depth_min + shift <= 0:
        raise ValueError(
            f"depth_min + shift must be positive for logarithmic quantization, "
            f"got depth_min={depth_min} + shift={shift} = {depth_min + shift}"
        )


def _depth_input_to_float32_and_unit(
    depth: NDArray[np.integer] | NDArray[np.floating],
    input_unit: Literal["auto", "m", "mm"],
) -> tuple[NDArray[np.float32], Literal["m", "mm"]]:

    resolved_unit = (
        ("m" if np.issubdtype(depth.dtype, np.floating) else "mm") if input_unit == "auto" else input_unit
    )
    return depth.astype(np.float32, order="K"), resolved_unit


def quantize_depth(
    depth: NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    pix_fmt: str = DEFAULT_DEPTH_PIX_FMT,
    video_backend: str | None = "pyav",
    input_unit: Literal["auto", "m", "mm"] = "auto",
) -> NDArray[np.uint16] | av.VideoFrame:

    if input_unit not in ("auto", "m", "mm"):
        raise ValueError(f"input_unit must be 'auto', 'm', or 'mm', got {input_unit!r}")

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()


    if depth.ndim == 3 and (depth.shape[-1] == 1 or depth.shape[0] == 1):
        depth = depth.squeeze()

    depth_f, resolved_unit = _depth_input_to_float32_and_unit(depth, input_unit=input_unit)


    depth_min_u = np.float32(depth_min) if resolved_unit == "m" else np.float32(depth_min * _MM_PER_METRE)
    depth_max_u = np.float32(depth_max) if resolved_unit == "m" else np.float32(depth_max * _MM_PER_METRE)
    shift_u = np.float32(shift) if resolved_unit == "m" else np.float32(shift * _MM_PER_METRE)


    if use_log:
        _validate_log_quant_params(depth_min, shift)
        log_min = math.log(float(depth_min_u + shift_u))
        log_max = math.log(float(depth_max_u + shift_u))
        norm = (np.log(depth_f + shift_u) - log_min) / (log_max - log_min)
    else:
        norm = (depth_f - depth_min_u) / (depth_max_u - depth_min_u)

    quantized = np.rint(norm * DEPTH_QMAX).clip(0, DEPTH_QMAX).astype(np.uint16, copy=False)

    if video_backend == "pyav":
        frame = av.VideoFrame.from_ndarray(quantized, format=pix_fmt)
        write_u16_plane(frame.planes[0], quantized)
        return frame
    else:
        return quantized


def dequantize_depth(
    quantized: NDArray[np.uint16] | av.VideoFrame,
    depth_min: float = DEFAULT_DEPTH_MIN,
    depth_max: float = DEFAULT_DEPTH_MAX,
    shift: float = DEFAULT_DEPTH_SHIFT,
    use_log: bool = DEFAULT_DEPTH_USE_LOG,
    pix_fmt: str = DEFAULT_DEPTH_PIX_FMT,
    output_unit: Literal["m", "mm"] = "mm",
    output_tensor: bool = False,
) -> NDArray[np.uint16] | NDArray[np.float32] | torch.Tensor:

    if output_unit not in ("m", "mm"):
        raise ValueError(f"output_unit must be 'm' or 'mm', got {output_unit!r}")

    if isinstance(quantized, av.VideoFrame):
        quantized = quantized.to_ndarray(format=pix_fmt)

    norm = np.asarray(quantized, dtype=np.float32, order="K") / DEPTH_QMAX

    depth_min_m = np.float32(depth_min)
    depth_max_m = np.float32(depth_max)
    shift_m = np.float32(shift)


    if use_log:
        _validate_log_quant_params(depth_min, shift)
        log_min = math.log(float(depth_min_m + shift_m))
        log_max = math.log(float(depth_max_m + shift_m))
        depth_m = np.exp(norm * (log_max - log_min) + log_min) - shift_m
    else:
        depth_m = norm * (depth_max_m - depth_min_m) + depth_min_m
    depth_m = np.clip(depth_m, depth_min_m, depth_max_m).astype(np.float32, copy=False)


    if depth_m.ndim == 2:
        depth_m = depth_m[..., np.newaxis]


    if output_unit == "m":
        return torch.from_numpy(depth_m) if output_tensor else depth_m


    mm = np.rint(depth_m * _MM_PER_METRE).clip(0, _UINT16_MAX).astype(np.uint16, copy=False)
    if output_tensor:

        return torch.from_numpy(mm.astype(np.float32))
    else:
        return mm
