"""Observation-only alternative pickup approaches; independent of object yaw.

Development candidate: a symmetric object's fitted yaw must not determine which
half of the table the robot is allowed to approach. Prefer the current viewing
direction, but search the full circle through the observed occupancy map.
"""
import numpy as np
from navigation import rot2, wrap


def plan_pickup_approach(depth_map, odom, object_position):
    center = np.asarray(object_position)[:2]
    view = np.asarray(odom)[:2] - center
    view /= max(np.linalg.norm(view), 1e-8)
    candidates = []
    for radius in [.65, .73, .81, .90]:
        for turn in np.linspace(-np.pi, np.pi, 33)[:-1]:
            direction = rot2(turn) @ view
            heading = np.arctan2(-direction[1], -direction[0])
            left = np.array([-np.sin(heading), np.cos(heading)])
            goal = center + radius * direction - .12 * left
            path = depth_map.path(odom[:2], goal)
            if path:
                length = sum(np.linalg.norm(b-a) for a,b in zip(path,path[1:]))
                score = length + .10*abs(turn) + .10*abs(wrap(heading-odom[2]))
                candidates.append((score, goal, heading, path))
    if not candidates:
        return None
    _,goal,heading,path = min(candidates, key=lambda value:value[0])
    return goal,heading,path
