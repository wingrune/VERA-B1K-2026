"""Port state -> native observation dict (contract tables 2.1-2.4).

Pure numpy: the simulator side only has to fill a ``PortFrame``. Quaternions are xyzw everywhere
(OmniGibson tensors, warp ``quatf`` and the port's Isaac Lab 3.0 data are all xyzw).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .spec import (BASE_JOINTS, BASE_YAW_JOINT, CAM_POSES_KEY, CAMERAS, JOINT_FIELDS, PROPRIO_DIM, PROPRIO_KEY,
                   PROPRIO_SLICES, TASK_ID_KEY)


@dataclass(frozen=True)
class PortFrame:
    """One synchronous port sample, world frame, xyzw quaternions, poses as [x, y, z, qx, qy, qz, qw].

    Attributes:
        joint_names: port joint order (any order, mapped by name).
        qpos, qvel: joint positions and velocities in ``joint_names`` order.
        base_pose: world pose of the robot frame (base_link).
        eef_pose: ``{'left', 'right'}`` world poses of left_eef_link / right_eef_link.
        cam_pose: camera role name -> world pose of the USD camera prim (-Z forward, +Y up).
        rgb: role -> uint8 image, HxWx3 or HxWx4.
        depth: role -> planar depth (distance_to_image_plane) in metres; misses may be 0 or non-finite.
    """

    joint_names: Sequence[str]
    qpos: np.ndarray
    qvel: np.ndarray
    base_pose: np.ndarray
    eef_pose: Mapping[str, np.ndarray]
    cam_pose: Mapping[str, np.ndarray]
    rgb: Mapping[str, np.ndarray]
    depth: Mapping[str, np.ndarray]


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """xyzw unit quaternion -> 3x3 rotation, no renormalisation (keeps identity bit-exact)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b, both xyzw."""
    ax, ay, az, aw = (float(v) for v in a)
    bx, by, bz, bw = (float(v) for v in b)
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def relative_pose(pose: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Pose expressed in the base frame, as OmniGibson T.relative_pose_transform(pose, base)."""
    pose, base = np.asarray(pose, np.float64), np.asarray(base, np.float64)
    pos = quat_to_matrix(base[3:]).T @ (pose[:3] - base[:3])
    conj = np.array([-base[3], -base[4], -base[5], base[6]])
    return np.concatenate((pos, quat_multiply(conj, pose[3:])))


def build_proprio(frame: PortFrame) -> np.ndarray:
    """61D proprio in the native order (contract 2.2), float32."""
    index = {n: i for i, n in enumerate(frame.joint_names)}
    qpos, qvel = np.asarray(frame.qpos, np.float64), np.asarray(frame.qvel, np.float64)
    out = np.empty(PROPRIO_DIM, np.float64)

    def put(field: str, value: np.ndarray) -> None:
        start, end = PROPRIO_SLICES[field]
        out[start:end] = value

    # Base joint velocities live in the articulation root frame; native reports them in the robot frame.
    vx, vy, wz = qvel[[index[n] for n in BASE_JOINTS]]
    yaw = qpos[index[BASE_YAW_JOINT]]
    c, s = np.cos(yaw), np.sin(yaw)
    put('base_qvel', [c * vx + s * vy, -s * vx + c * vy, wz])
    for group, names in JOINT_FIELDS.items():
        ids = [index[n] for n in names]
        put(group + '_qpos', qpos[ids])
        put(group + '_qvel', qvel[ids])
    for side in ('left', 'right'):
        rel = relative_pose(frame.eef_pose[side], frame.base_pose)
        put(f'eef_{side}_pos', rel[:3])
        put(f'eef_{side}_quat', rel[3:])
    return out.astype(np.float32)


def cam_rel_poses(frame: PortFrame) -> np.ndarray:
    """21D camera poses in the robot frame, order left wrist, right wrist, head (contract 2.3), float32."""
    return np.concatenate([relative_pose(frame.cam_pose[c.name], frame.base_pose) for c in CAMERAS]).astype(np.float32)


def native_depth(depth: np.ndarray) -> np.ndarray:
    """Planar depth with native misses: Newton writes 0 for no hit or beyond the far plane, native writes +inf."""
    depth = np.asarray(depth, np.float32)
    return np.where(np.isfinite(depth) & (depth > 0), depth, np.float32(np.inf)).astype(np.float32)


def native_rgba(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGBA. A 3-channel render gets alpha 255 (the policies read only rgb[..., :3])."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f'expected uint8 HxWx3|4, got {rgb.dtype} {rgb.shape}')
    if rgb.shape[-1] == 4:
        return rgb
    return np.concatenate((rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)), -1)


def pixel_rays(size: int, fx: float) -> np.ndarray:
    """|ray| / z per pixel for a square pinhole with fx == fy, principal point size / 2, pixel centres +0.5."""
    u = (np.arange(size) + .5 - size / 2) / fx
    return np.sqrt(1 + u[None, :] ** 2 + u[:, None] ** 2)


def ray_to_planar(ray: np.ndarray, fx: float) -> np.ndarray:
    """distance_to_camera -> distance_to_image_plane for a square camera (misses stay as they are)."""
    return (np.asarray(ray, np.float64) / pixel_rays(ray.shape[0], fx)).astype(np.float32)


def native_obs_numpy(frame: PortFrame, task_id: int) -> Dict[str, np.ndarray]:
    """Unbatched native obs dict (numpy). Image sizes are checked against the native roles."""
    obs = {PROPRIO_KEY: build_proprio(frame), CAM_POSES_KEY: cam_rel_poses(frame),
           TASK_ID_KEY: np.array([int(task_id)], np.int64)}
    for cam in CAMERAS:
        obs[cam.rgb_key] = native_rgba(frame.rgb[cam.name])
        obs[cam.depth_key] = native_depth(frame.depth[cam.name])
    check_obs(obs, batched=False)
    return obs


def native_obs(frame: PortFrame, task_id: int) -> Dict[str, Any]:
    """Native obs dict as the evaluator hands it to ``policy.forward``: torch tensors with batch dim 1."""
    import torch
    return {k: torch.from_numpy(np.ascontiguousarray(v)[None]) for k, v in native_obs_numpy(frame, task_id).items()}


def check_obs(obs: Mapping[str, Any], batched: bool = True) -> None:
    """Raise ValueError unless ``obs`` has exactly the native keys, shapes, dtypes and value domains."""
    lead = (1,) if batched else ()
    want = {PROPRIO_KEY: (lead + (PROPRIO_DIM,), np.float32), CAM_POSES_KEY: (lead + (21,), np.float32),
            TASK_ID_KEY: (lead + (1,), np.int64)}
    for cam in CAMERAS:
        want[cam.rgb_key] = (lead + (cam.size, cam.size, 4), np.uint8)
        want[cam.depth_key] = (lead + (cam.size, cam.size), np.float32)
    if set(obs) != set(want):
        raise ValueError(f'keys differ: missing {sorted(set(want) - set(obs))} extra {sorted(set(obs) - set(want))}')
    for key, (shape, dtype) in want.items():
        v = obs[key]
        v = v.detach().cpu().numpy() if hasattr(v, 'detach') else np.asarray(v)
        if v.shape != shape or v.dtype != dtype:
            raise ValueError(f'{key}: {v.dtype}{v.shape}, expected {np.dtype(dtype)}{shape}')
        if key.endswith('::depth_linear'):
            if np.isnan(v).any() or (v <= 0).any() or np.isneginf(v).any():
                raise ValueError(f'{key}: depth must be > 0 or +inf (no zeros, NaN, -inf)')
        elif v.dtype.kind == 'f' and not np.isfinite(v).all():
            raise ValueError(f'{key}: non-finite values')
    quats = np.asarray(obs[CAM_POSES_KEY]).reshape(-1, 3, 7)[..., 3:]
    for start, _ in (PROPRIO_SLICES['eef_left_quat'], PROPRIO_SLICES['eef_right_quat']):
        quats = np.concatenate((quats, np.asarray(obs[PROPRIO_KEY]).reshape(-1, PROPRIO_DIM)[:, None, start:start + 4]), 1)
    if np.abs(np.linalg.norm(quats, axis=-1) - 1).max() > 1e-3:
        raise ValueError('pose quaternions are not unit length')
