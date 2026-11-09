"""Wire protocol of the BEHAVIOR 2026 evaluator (omnigibson.eval.utils.network_utils), dependency-light: numpy, msgpack, websockets.

The evaluator (client) connects to ws://host:port after GET /healthz answers 200, receives one metadata message, then sends
msgpack messages: {"reset": True} (no reply expected) or a flattened observation dict (reply {"action": float32 (..., action_dim)}).
A str reply is an error: the client raises with that text.

serve() differs from the stock WebsocketPolicyServer only in running policy calls in a worker thread, so websocket pings are
answered while a policy loads weights or plans (the client drops the connection after 300 s without a pong).
"""
from __future__ import annotations

import asyncio
import functools
import http
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import msgpack
import numpy as np
import websockets
from websockets.asyncio.server import serve as ws_serve

log = logging.getLogger('b1k_submit')


def _pack(obj: Any) -> Any:
    if hasattr(obj, 'detach'):                       # torch tensor, without importing torch
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ('V', 'O', 'c'):
            raise ValueError(f'unsupported dtype {obj.dtype}')
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
    return obj


def _unpack(obj: dict) -> Any:
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    return obj


packb = functools.partial(msgpack.packb, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack, strict_map_key=False)


def health(connection, request) -> Optional[Any]:
    if request.path == '/healthz':
        return connection.respond(http.HTTPStatus.OK, 'OK\n')
    return None


def serve(policy: Any, port: int, host: str = '0.0.0.0', metadata: Optional[dict] = None) -> None:
    """Serve policy.reset() and policy.act(obs) -> array over the evaluator protocol until killed."""
    pool = ThreadPoolExecutor(max_workers=1)

    async def handler(ws) -> None:
        loop = asyncio.get_running_loop()
        await ws.send(packb(metadata or {}))
        while True:
            try:
                msg = unpackb(await ws.recv())
                if 'reset' in msg:
                    await loop.run_in_executor(pool, policy.reset)
                    continue
                action = await loop.run_in_executor(pool, policy.act, msg)
                await ws.send(packb({'action': np.asarray(action, dtype=np.float32)}))
            except websockets.ConnectionClosed:
                return
            except Exception:  # noqa: BLE001 - the client gets the traceback as the error text
                text = traceback.format_exc()
                log.error(text)
                await ws.send(text)
                await ws.close(code=1011, reason='policy error')
                return

    async def main() -> None:
        async with ws_serve(handler, host, port, compression=None, max_size=None, process_request=health) as srv:
            log.info('serving on %s:%d', host, port)
            await srv.serve_forever()

    asyncio.run(main())
