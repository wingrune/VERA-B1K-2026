"""Native 23D action -> port command (contract table 3).

The port env decodes native actions in ``PortEnv._apply_native`` (base velocity scaled to 0.75/0.75/1.0 and rotated from
the robot frame into the articulation root frame, grippers linear onto the finger range, trunk and arms absolute and
clamped). This module only checks that the export contract the decoder is driven by equals the native semantics, and
hands the policy output over.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .spec import ACTION_DIM, ACTION_SLICES, BASE_OUTPUT_LIMITS, GRIPPER_OUTPUT_LIMITS

_TYPES = {'base': 'HolonomicBaseJointController', 'gripper': 'MultiFingerGripperController'}


def check_contract(contract: Mapping[str, Any]) -> None:
    """Raise ValueError unless the export ACTION_CONTRACT.json rows equal the native 23D semantics."""
    rows = contract['per_robot_controllers'][0]
    got = [(r['name'], r['start'], r['end'], tuple(r['dof_names'])) for r in rows]
    if got != [tuple(s) for s in ACTION_SLICES]:
        raise ValueError(f'action layout differs from native: {got}')
    for r in rows:
        kind = r['name'].split('_')[0]
        want_type = _TYPES.get(kind, 'JointController')
        if r['controller_type'] != want_type:
            raise ValueError(f"{r['name']}: controller {r['controller_type']}, native {want_type}")
        if kind == 'base':
            ok = (r['motor_type'] == 'velocity' and np.allclose(r['command_input_limits'], [[-1] * 3, [1] * 3])
                  and np.allclose(r['command_output_limits'], BASE_OUTPUT_LIMITS))
        elif kind == 'gripper':
            ok = (np.allclose(r['command_input_limits'], [[-1], [1]])
                  and np.allclose(np.ravel(r['command_output_limits']), GRIPPER_OUTPUT_LIMITS, atol=1e-6))
        else:  # absolute position targets: no input/output normalisation, no deltas
            generated = contract['generated_controller_config'][r['name']]
            ok = (r['motor_type'] == 'position' and r['command_input_limits'] is None
                  and r['command_output_limits'] is None and generated['use_delta_commands'] is False)
        if not ok:
            raise ValueError(f"{r['name']}: limits or command mode differ from native")


def to_port(action: Any, device: str = 'cpu') -> Any:
    """Policy output (1, 23) -> float32 torch [1, 23] on ``device`` for ``PortEnv.step``."""
    import torch
    a = action.detach().cpu().numpy() if hasattr(action, 'detach') else np.asarray(action)
    if a.shape != (1, ACTION_DIM) or not np.isfinite(a).all():
        raise ValueError(f'expected finite (1, {ACTION_DIM}) action, got {a.shape}')
    return torch.as_tensor(a.astype(np.float32), device=device)
