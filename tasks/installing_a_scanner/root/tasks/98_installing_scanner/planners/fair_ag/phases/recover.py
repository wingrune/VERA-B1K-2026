"""Phase unstick: the head camera sees the floor below the robot frame, i.e. the base stands on something it drove onto (a chair). Back off along
the way it came until the floor is level again, mark the place ahead as a hard obstacle and plan anew.
Success: lift below LIFT_OK_M. A back-off that does not free the base within UNSTICK_STEPS fails the navigation (the recovery of the `failed`
phase then looks around and plans again)."""
from __future__ import annotations

import math

import numpy as np

from .. import params as P

LIFT_OK_M = 0.02
UNSTICK_STEPS = 300
BACK_SPEED = 0.16      # fraction of the full base speed
NAV_PHASES = {'go_button': 'plan_button', 'go_push': 'plan_push', 'go_grasp': 'plan_grasp'}     # phase -> where to resume after the back-off


def start_unstick(c) -> None:
    """Enter the phase from a navigation phase: remember the heading of the last drive command and mark the obstacle ahead."""
    drive = c.last_drive[:2]
    n = float(np.linalg.norm(drive))
    c.unstick_dir = -drive / n if n > 1e-3 else np.array([-1., 0.])
    pose = c.perc.pose()
    d_odom = np.array([[math.cos(pose[2]), -math.sin(pose[2])], [math.sin(pose[2]), math.cos(pose[2])]]) @ (drive / n if n > 1e-3 else np.array([1., 0.]))
    c.perc.add_bump(pose[:2] + d_odom * 0.35, P.BUMP_RADIUS_M)
    c.counts['climbs'] = c.counts.get('climbs', 0) + 1
    c.resume_phase = NAV_PHASES[c.phase]
    c.change('unstick', f'the base climbed onto something (lift {c.perc.lift:.2f} m); marking it and backing off')


def step_unstick(c, action, q, w) -> None:
    c.to_rest(action, q)
    action[:2] = c.unstick_dir * BACK_SPEED
    c.reason = f'backing off a climb, lift {c.perc.lift:.3f} m'
    if c.perc.lift < LIFT_OK_M and c.age > 20:
        c.change(c.resume_phase, f'base level again after {c.age} controls')
    elif c.age > UNSTICK_STEPS:
        c.change('failed', 'could not get off the obstacle')


PHASES = {'unstick': step_unstick}
