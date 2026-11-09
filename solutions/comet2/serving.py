import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import numpy as np
import websockets
from websockets.asyncio.server import serve as ws_serve
from server.protocol import health, packb, unpackb

log = logging.getLogger(__name__)

class WebsocketPolicyServer:
    def __init__(self, policy, port, metadata=None):
        self.policy, self.port = policy, port
        self.metadata = metadata or {}

    def serve_forever(self):
        asyncio.run(self.run())

    async def run(self):
        lock = asyncio.Lock()
        with ThreadPoolExecutor(max_workers=1) as pool:
            async def handler(ws):
                if lock.locked():
                    await ws.send('backend serves one evaluator connection at a time')
                    return
                async with lock:
                    await ws.send(packb(self.metadata))
                    loop = asyncio.get_running_loop()
                    previous = None
                    latencies = []
                    async for raw in ws:
                        start = time.monotonic()
                        try:
                            obs = unpackb(raw)
                            if 'reset' in obs:
                                await loop.run_in_executor(pool, self.policy.reset)
                                continue
                            count = int(obs.pop('__action_chunk_size__', 0))
                            infer_start = time.monotonic()
                            if count > 1:
                                chunk = await loop.run_in_executor(pool, self.policy.act_chunk, obs, count)
                                reply = {'action': chunk[..., 0, :], 'action_chunk': chunk}
                            else:
                                reply = {'action': await loop.run_in_executor(pool, self.policy.act, obs)}
                            elapsed = (time.monotonic() - infer_start) * 1000
                            for value in reply.values():
                                if value.dtype != np.float32 or not np.isfinite(value).all() or value.shape[-1] != 23:
                                    raise ValueError('Invalid action output contract')
                            reply['server_timing'] = {'infer_ms': elapsed}
                            if previous is not None:
                                reply['server_timing']['prev_total_ms'] = previous
                            await ws.send(packb(reply))
                            previous = (time.monotonic() - start) * 1000
                            latencies.append(elapsed)
                            if len(latencies) == 100:
                                log.info('INFER requests=100 mean_ms=%.3f max_ms=%.3f chunks=%d compressed=%d recovery=%s',
                                         np.mean(latencies), max(latencies), self.policy.scheduler.n_chunks,
                                         self.policy.scheduler.n_compressed, self.policy.recovery.stats)
                                latencies.clear()
                        except websockets.ConnectionClosed:
                            return
                        except Exception:
                            log.exception('Policy request failed')
                            await ws.close(code=1011, reason='policy error')
                            return
            async with ws_serve(handler, '0.0.0.0', self.port, compression=None, max_size=None,
                                process_request=health, ping_interval=None) as server:
                log.info('Policy ready on port %d', self.port)
                await server.serve_forever()
