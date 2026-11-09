"""Batched model-fitting kernels of the fair planner of task 69 (torch). Frozen like fair_skills: the mutable perception modules call them with
their parameters and never copy them. One batch holds every yaw hypothesis of a fit.

backend 'gpu' runs on CUDA in float64 with TF32 off; 'cpu' runs the same code on the CPU (tests, reference). 'gpu' without CUDA raises
FitGpuRequired: there is no silent fallback. stats() gives calls, seconds and elements per kernel and backend (written to result.json).

  planar_icp(cloud, model, z0, yaws, xy0, ...)  trimmed planar ICP of a model standing on a support (origin height z0 fixed): x, y, yaw of
                                                 every hypothesis and the distances of the cloud points to the fitted model
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

_STATS: Dict[str, Dict[str, float]] = {}
CHUNK_ELEMS = 3e7


class FitGpuRequired(RuntimeError):
    """backend 'gpu' without CUDA."""


def _torch():
    import torch
    return torch


def _device(backend: str):
    torch = _torch()
    if backend == 'gpu':
        if not torch.cuda.is_available():
            raise FitGpuRequired('fitkern backend gpu: no CUDA device')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return torch.device('cuda')
    if backend == 'cpu':
        return torch.device('cpu')
    raise ValueError(f'backend must be gpu or cpu, not {backend!r}')


def _note(name: str, backend: str, seconds: float, elems: float) -> None:
    s = _STATS.setdefault(f'{name}:{backend}', dict(calls=0, seconds=0., elems=0.))
    s['calls'] += 1
    s['seconds'] += seconds
    s['elems'] += float(elems)


def stats() -> Dict[str, Dict[str, float]]:
    return {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in _STATS.items()}


def reset_stats() -> None:
    _STATS.clear()


def planar_icp(cloud, model, z0, yaws, xy0, iters=10, trim=0.03, backend='gpu', rotate=True) -> Tuple[np.ndarray, np.ndarray]:
    """Batched trimmed planar ICP of a model on a support.
    cloud (M, 3) observed points; model (N, 3) in its own frame, placed as T = (Rz(yaw), (x, y, z0)). yaws (Y,) and xy0 (Y, 2) start the
    hypotheses. Every iteration matches each cloud point to its nearest model point (3D) and solves the trimmed (distance < trim) planar
    rigid update (rotate=False: translation only, the yaw stays). Returns poses (Y, 3) = (x, y, yaw) and d (Y, M) cloud-to-model distances at
    the final poses."""
    torch = _torch()
    dev = _device(backend)
    t0 = time.time()
    cl = torch.as_tensor(np.asarray(cloud, float), dtype=torch.float64, device=dev)
    md = torch.as_tensor(np.asarray(model, float), dtype=torch.float64, device=dev)
    pose = torch.as_tensor(np.c_[np.asarray(xy0, float).reshape(-1, 2), np.asarray(yaws, float)], dtype=torch.float64, device=dev)
    M, N, Y = cl.shape[0], md.shape[0], pose.shape[0]
    chunk = max(1, int(CHUNK_ELEMS // max(1, M * N)))
    poses_out, d_out = [], []
    for a in range(0, Y, chunk):
        p = pose[a:a + chunk].clone()
        for it in range(iters + 1):
            c, s = torch.cos(p[:, 2])[:, None], torch.sin(p[:, 2])[:, None]
            mx = c * md[None, :, 0] - s * md[None, :, 1] + p[:, 0, None]
            my = s * md[None, :, 0] + c * md[None, :, 1] + p[:, 1, None]
            mz = md[None, :, 2] + z0
            d2 = ((cl[None, :, None, 0] - mx[:, None, :]) ** 2 + (cl[None, :, None, 1] - my[:, None, :]) ** 2
                  + (cl[None, :, None, 2] - mz[:, None, :]) ** 2)
            d2min, nn = torch.min(d2, dim=2)
            d = torch.sqrt(d2min)
            del d2
            if it == iters:
                break
            w = (d < trim).double()
            ax, ay = torch.gather(mx, 1, nn), torch.gather(my, 1, nn)
            bx, by = cl[None, :, 0].expand_as(ax), cl[None, :, 1].expand_as(ay)
            ws = torch.clamp(w.sum(1), min=1e-9)
            abx, aby = (w * ax).sum(1) / ws, (w * ay).sum(1) / ws
            bbx, bby = (w * bx).sum(1) / ws, (w * by).sum(1) / ws
            ux, uy, vx, vy = ax - abx[:, None], ay - aby[:, None], bx - bbx[:, None], by - bby[:, None]
            th = torch.atan2((w * ux * vy).sum(1) - (w * uy * vx).sum(1), (w * ux * vx).sum(1) + (w * uy * vy).sum(1))
            th = torch.where(w.sum(1) >= 3, th, torch.zeros_like(th)) if rotate else torch.zeros_like(th)
            ct, st = torch.cos(th), torch.sin(th)
            tx, ty = bbx - (ct * abx - st * aby), bby - (st * abx + ct * aby)
            tx = torch.where(w.sum(1) >= 3, tx, torch.zeros_like(tx))
            ty = torch.where(w.sum(1) >= 3, ty, torch.zeros_like(ty))
            p = torch.stack([ct * p[:, 0] - st * p[:, 1] + tx, st * p[:, 0] + ct * p[:, 1] + ty, p[:, 2] + th], 1)
        poses_out.append(p)
        d_out.append(d)
    out_p = torch.cat(poses_out).cpu().numpy()
    out_d = torch.cat(d_out).cpu().numpy()
    out_p[:, 2] = (out_p[:, 2] + np.pi) % (2 * np.pi) - np.pi
    _note('planar_icp', backend, time.time() - t0, Y * M * N * (iters + 1))
    return out_p, out_d
