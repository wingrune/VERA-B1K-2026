"""Arm-only GPU IK worker of the task 69 oracle (cuRobo, the trunk frozen): runs in the cuRobo venv (BIMBO_CUROBO_PY), one per episode
process, started by oracle/ik.py. The free hand's press on the held vacuum must not move the shared trunk (the trunk carries the holding
hand and the vacuum with it), and the trunk + arm solvers of skills/gpu-ik cannot hold the trunk. This worker builds, per arm, a cuRobo
model of the 7 arm joints only, rooted at the arm's mount link (`<side>_arm_base_link`, fixed on torso_link4), with the joint limits of
core.skills_lib.r1pro_kinematics; the client gives the goal of the gripper link in the mount frame (the trunk pose from the forward
kinematics of the current trunk joints). Prints READY, then one JSON request per stdin line -> one JSON line on stdout:

  {"op": "near", "side", "T": 16 (gripper link goal in the mount frame, row major), "q_seed": 7}
      -> {"mask": bool, "q": 7}   among the successful seeds, the solution nearest q_seed (limit-normalised L2); the client re-checks it in
                                  float64 with the CPU forward kinematics

Errors answer {"error": text}. Any other stdout line (library banners) is not JSON and is skipped by the client.
"""
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from core.skills_lib import r1pro_kinematics as K  # noqa: E402

TOL = (0.005, 0.05)
NUM_SEEDS = 32
_IK = {}


def write_model(side: str, out: Path) -> Path:
    by_child = {j[2]: j for j in K.JOINTS}
    mount = f'{side}_arm_base_link'
    chain, link = [], K.GRIPPER_LINK[side]
    while link != mount:
        chain.append(by_child[link])
        link = by_child[link][1]
    chain.reverse()
    dofs = list(K.ARM_DOFS[side])
    lim = K.arm_limits(side)[:, 4:]
    names, x = [], [f'<robot name="r1pro_{side}_arm">']
    x += [f'<link name="{name}"/>' for name in sorted({mount} | {j[2] for j in chain})]
    j_ = lambda v: ' '.join(map(repr, map(float, v)))  # noqa: E731
    for dof, parent, child, kind, xyz, rpy, axis in chain:
        if dof < 0:
            x.append(f'<joint name="{child}_fixed" type="fixed"><parent link="{parent}"/><child link="{child}"/><origin xyz="{j_(xyz)}" rpy="{j_(rpy)}"/></joint>')
            continue
        k = dofs.index(dof)
        names.append(f'j{dof}')
        x.append(f'<joint name="j{dof}" type="revolute"><parent link="{parent}"/><child link="{child}"/><origin xyz="{j_(xyz)}" rpy="{j_(rpy)}"/>'
                 f'<axis xyz="{j_(axis)}"/><limit lower="{lim[0, k]}" upper="{lim[1, k]}" effort="100" velocity="3"/></joint>')
    x.append('</robot>')
    assert len(names) == 7, names
    (out / f'arm_{side}.urdf').write_text('\n'.join(x))
    (out / f'arm_{side}.yml').write_text(f"""robot_cfg:
  kinematics:
    format_version: 2.0
    asset_root_path: {out}
    urdf_path: {out}/arm_{side}.urdf
    base_link: {mount}
    tool_frames: [{K.GRIPPER_LINK[side]}]
    collision_link_names: []
    cspace:
      joint_names: {names}
      default_joint_position: {list(map(float, K.rest(side)[4:]))}
      null_space_weight: {[1.0] * 7}
      cspace_distance_weight: {[1.0] * 7}
      max_acceleration: 15.0
      max_jerk: 500.0
""")
    return out / f'arm_{side}.yml'


def solver(side: str):
    if side not in _IK:
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        out = Path(tempfile.mkdtemp(prefix='ik69_arm_'))
        cfg = InverseKinematicsCfg.create(robot=str(write_model(side, out)), num_seeds=NUM_SEEDS, max_batch_size=1,
                                          self_collision_check=False, load_collision_spheres=False,
                                          position_tolerance=float(TOL[0]), orientation_tolerance=float(TOL[1]))
        _IK[side] = InverseKinematics(cfg)
    return _IK[side]


def near(side: str, T: np.ndarray, q_seed: np.ndarray):
    import torch
    from curobo.types import GoalToolPose, Pose
    from scipy.spatial.transform import Rotation as R
    ik = solver(side)
    xyzw = R.from_matrix(T[:3, :3]).as_quat()
    pos = torch.tensor(T[None, :3, 3], dtype=torch.float32, device='cuda')
    quat = torch.tensor(np.r_[xyzw[3], xyzw[:3]][None], dtype=torch.float32, device='cuda')
    goal = GoalToolPose.from_poses({ik.tool_frames[0]: Pose(position=pos, quaternion=quat)}, num_goalset=1)
    seed = torch.tensor(np.asarray(q_seed, float)[None, None], dtype=torch.float32, device='cuda')
    r = ik.solve_pose(goal, seed_config=seed, return_seeds=NUM_SEEDS)
    torch.cuda.synchronize()
    ok = r.success.reshape(-1).cpu().numpy()
    sol = r.solution.reshape(-1, 7).cpu().numpy().astype(float)
    lo, hi = K.arm_limits(side)[:, 4:]
    d = np.linalg.norm((sol - np.asarray(q_seed, float)) / (hi - lo), axis=1)
    d[~ok] = np.inf
    j = int(d.argmin())
    return bool(np.isfinite(d[j])), sol[j]


def main() -> None:
    t0 = time.time()
    T = np.eye(4)
    T[:3, 3] = (0.3, 0.0, -0.3)
    for side in ('left', 'right'):       # build both solvers (CUDA graph capture) before READY
        near(side, T, K.rest(side)[4:])
    print(json.dumps(dict(ready=True, seconds=round(time.time() - t0, 1))), flush=True)
    for line in sys.stdin:
        try:
            req = json.loads(line)
            t = time.perf_counter()
            if req.get('op') == 'quit':
                break
            if req['op'] != 'near':
                raise ValueError(f"unknown op {req['op']!r}")
            mask, q = near(req['side'], np.asarray(req['T'], float).reshape(4, 4), np.asarray(req['q_seed'], float))
            print(json.dumps(dict(mask=mask, q=q.tolist(), seconds=round(time.perf_counter() - t, 4))), flush=True)
        except Exception:  # noqa: BLE001 - the worker keeps serving, the client decides
            print(json.dumps(dict(error=traceback.format_exc()[-600:])), flush=True)


if __name__ == '__main__':
    main()
