import asyncio
import functools
import http
import logging
import os
import msgpack
import numpy as np
import time
import traceback
import torch
import websockets.asyncio.server as _server
import websockets
from copy import deepcopy
from typing import Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["WebsocketPolicyServer", "task_from_request"]

_TASK_IDS = None


def task_from_request(obs) -> str:

    global _TASK_IDS
    if _TASK_IDS is None:
        import json, pathlib
        _TASK_IDS = json.loads((pathlib.Path(__file__).parents[1] / "configs/tasks/b1k_task_ids.json").read_text())
    return _TASK_IDS[str(int(np.asarray(obs["task_id"]).ravel()[0]))]


class WebsocketPolicyServer:


    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
        auto_task: bool = False,
    ) -> None:
        self._policy = policy
        self._auto_task = auto_task
        self._host = host
        self._port = port
        self._metadata = metadata or {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}...")
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,


            ping_interval=None,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if "reset" in result:
                    self._policy.reset()
                    continue

                obs = deepcopy(result)


                chunk_size = int(obs.pop("__action_chunk_size__", 0))
                if self._auto_task:
                    self._policy.set_task(task_from_request(obs))

                infer_time = time.monotonic()
                if chunk_size > 1:
                    chunk = self._policy.act_chunk(obs, chunk_size)
                    action = chunk[..., 0, :]
                else:
                    action = self._policy.act(obs)
                infer_time = time.monotonic() - infer_time


                if os.environ.get("B1K_LOG_INFER") == "1":
                    _stats = self.__dict__.setdefault("_b1k_stats", [])
                    _stats.append(infer_time * 1000)
                    if len(_stats) >= 100:
                        _model = [t for t in _stats if t > 20]; _cached = [t for t in _stats if t <= 20]
                        logger.info(f"INFER last {len(_stats)} requests: model calls {len(_model)} "
                                    f"mean {sum(_model) / max(len(_model), 1):.1f} ms max {max(_model, default=0):.1f} ms; "
                                    f"cached {len(_cached)} mean {sum(_cached) / max(len(_cached), 1):.2f} ms; "
                                    f"overall mean {sum(_stats) / len(_stats):.1f} ms per action; "
                                    f"compressed chunks {getattr(self._policy, 'n_compressed', 0)}/"
                                    f"{getattr(self._policy, 'n_chunks', 0)}")
                        _stats.clear()

                action = {
                    "action": action.cpu().numpy(),
                }
                if chunk_size > 1:
                    action["action_chunk"] = chunk.cpu().numpy()
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:

                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                try:

                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except AttributeError:

                    await websocket.close(code=1011, reason="Internal server error")
                raise


def _health_check(connection, request) -> Optional[Any]:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        else:

            return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"

    return None


"""
Adds NumPy array and PyTorch tensor support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

This module supports serializing both NumPy arrays and PyTorch tensors. PyTorch tensors are converted to
NumPy arrays (zero-copy when possible) before serialization. On deserialization, arrays are returned as NumPy arrays.

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""


def pack_data(obj):
    if isinstance(obj, torch.Tensor):
        data = obj.detach().cpu().numpy()
        return {
            b"__ndarray__": True,
            b"data": data.tobytes(),
            b"dtype": data.dtype.str,
            b"shape": data.shape,
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
packb = functools.partial(msgpack.packb, default=pack_data)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)
