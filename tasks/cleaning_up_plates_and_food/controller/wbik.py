"""Whole-body (trunk + one arm) inverse kinematics on the static R1Pro URDF.

Faster replacement for the per-call tree walk in kinematics.py: joints are sorted
once, and pose goals are solved from many seeds with analytic Jacobians. Frames:
everything is in the robot base_link frame, z up, x forward.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def _origin(joint):
    t = np.eye(4); origin = joint.find('origin')
    if origin is not None:
        t[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = R.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def _rodrigues(axis, angle):
    x, y, z = axis; c, s = np.cos(angle), np.sin(angle); k = 1 - c
    return np.array([[c + x*x*k, x*y*k - z*s, x*z*k + y*s],
                     [y*x*k + z*s, c + y*y*k, y*z*k - x*s],
                     [z*x*k - y*s, z*y*k + x*s, c + z*z*k]])


class Body:
    """Forward kinematics of the whole robot for one configuration of the 28 named dofs."""

    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.loads((folder / 'robot_model.json').read_text())
        self.names = self.spec['dof_names']
        self.lower, self.upper = np.asarray(self.spec['limits'], float)
        joints = {}; root = ET.parse(folder / 'r1pro.urdf').getroot()
        # Link masses and centres of mass, and joint effort limits, for static load checks.
        self.inertial = {}
        for link in root.findall('link'):
            inertial = link.find('inertial')
            if inertial is not None:
                origin = inertial.find('origin')
                self.inertial[link.get('name')] = (float(inertial.find('mass').get('value')),
                                                   np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3))
        self.effort = {j.get('name'): float(j.find('limit').get('effort')) for j in root.findall('joint') if j.find('limit') is not None and j.find('limit').get('effort')}
        for j in root.findall('joint'):
            axis = j.find('axis')
            joints[j.find('child').get('link')] = (
                j.get('name'), j.find('parent').get('link'), j.get('type'), _origin(j),
                np.fromstring(axis.get('xyz', '1 0 0'), sep=' ') if axis is not None else np.array([1., 0, 0]))
        order, placed = [], {'base_link'}
        pending = dict(joints)
        while pending:
            ready = [child for child, value in pending.items() if value[1] in placed]
            if not ready:
                raise ValueError('Disconnected URDF')
            for child in ready:
                order.append((child, *pending.pop(child))); placed.add(child)
        self.order = [(child, name, parent, kind, origin, axis, self.names.index(name) if name in self.names else -1)
                      for child, name, parent, kind, origin, axis in order]
        self.parent = {child: parent for child, _, parent, *_ in self.order}
        self.joint_of = {child: (kind, axis, index) for child, _, _, kind, _, axis, index in self.order}

    def index(self, group):
        """Indices into the 28-dof vector of 'trunk', 'arm_left', 'arm_right'."""
        return [self.names.index(n) for c in self.spec['controllers'] if c['name'] == group for n in c['dof_names']]

    def fk(self, q):
        """Returns ({link: 4x4 pose}, {link: 4x4 pose of the joint frame before its motion})."""
        poses = {'base_link': np.eye(4)}; frames = {}
        for child, _, parent, kind, origin, axis, index in self.order:
            t = poses[parent] @ origin; frames[child] = t
            if index >= 0 and kind in ('revolute', 'continuous'):
                motion = np.eye(4); motion[:3, :3] = _rodrigues(axis, q[index]); t = t @ motion
            elif index >= 0 and kind == 'prismatic':
                motion = np.eye(4); motion[:3, 3] = axis * q[index]; t = t @ motion
            poses[child] = t
        return poses, frames

    def chain(self, link):
        """Actuated ancestors of @link, root first, as (child_link, kind, local_axis, dof_index)."""
        out = []
        while link != 'base_link':
            kind, axis, index = self.joint_of[link]
            if index >= 0:
                out.append((link, kind, axis, index))
            link = self.parent[link]
        return out[::-1]

    def gravity_torques(self, poses, frames, columns, payload=None):
        """Static torque (N m) each joint in @columns must hold against gravity.

        payload: optional (mass, link, point in that link's frame) for a carried object.
        Returned with the sign convention of the joint axis; compare magnitudes with self.effort.
        """
        where = {dof: i for i, dof in enumerate(columns)}; torque = np.zeros(len(columns))
        loads = [(mass, link, com) for link, (mass, com) in self.inertial.items()]
        if payload is not None:
            loads.append(payload)
        for mass, link, com in loads:
            if link not in poses:
                continue
            t = poses[link]; point = t[:3, 3] + t[:3, :3] @ com; force = np.array([0., 0., -9.81 * mass])
            for child, kind, axis, index in self.chain(link):
                if index in where and kind != 'prismatic':
                    frame = frames[child]
                    torque[where[index]] -= np.dot(frame[:3, :3] @ axis, np.cross(point - frame[:3, 3], force))
        return torque

    def jacobian(self, link, point_world, frames, columns):
        """6 x len(columns) geometric Jacobian of a world point rigidly attached to @link."""
        jac = np.zeros((6, len(columns))); where = {dof: i for i, dof in enumerate(columns)}
        for child, kind, axis, index in self.chain(link):
            if index not in where:
                continue
            frame = frames[child]; a = frame[:3, :3] @ axis
            if kind == 'prismatic':
                jac[:3, where[index]] = a
            else:
                jac[:3, where[index]] = np.cross(a, point_world - frame[:3, 3]); jac[3:, where[index]] = a
        return jac


def _left_jacobian_inverse(phi):
    """d(rotvec of Rgoal R^T)/d(omega) for an incremental world-frame rotation omega of R."""
    angle = np.linalg.norm(phi); x, y, z = phi
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    coefficient = 1/12 if angle < 1e-5 else (1 - .5*angle/np.tan(.5*angle))/(angle*angle)
    return np.eye(3) + .5*skew + coefficient*(skew @ skew)


class WholeBodyIK:
    """Pose goals for the tool point of one arm, solved over trunk(4) + arm(7)."""

    def __init__(self, folder, arm='left', tool=(0., 0., -.060)):
        self.body = Body(folder); self.arm = arm; self.link = f'{arm}_gripper_link'; self.tool = np.asarray(tool, float)
        self.columns = self.body.index('trunk') + self.body.index(f'arm_{arm}')
        self.lower = self.body.lower[self.columns].copy(); self.upper = self.body.upper[self.columns].copy()
        # A natural squat/reach posture; used as a weak attractor and as a seed.
        self.rest = np.array([.5, -.65, -.5, 0., -1.4, .35, .2, -.7, 0., .3, 0.])
        if arm == 'right':
            self.rest[[5, 6, 8, 10]] *= -1

    def tool_pose(self, q):
        t = self.body.fk(q)[0][self.link].copy(); t[:3, 3] += t[:3, :3] @ self.tool; return t

    def seeds(self, current, goal, count, rng):
        low = goal[2] < .35
        out = [current, self.rest]
        for trunk in ([[1.6, -2.5, -.94], [1.2, -1.7, -.5], [.8, -1.2, -.5], [1.8, -2.4, -.3], [1.0, -2.0, -1.0]] if low else [[.3, -.5, -.3], [.7, -1.2, -.5]]):
            seed = self.rest.copy(); seed[:3] = trunk; out.append(seed)
        span = self.upper - self.lower
        while len(out) < count:
            base = out[rng.integers(0, min(len(out), 7))]
            out.append(np.clip(base + rng.normal(0, .12, len(base)) * np.minimum(span, 3.), self.lower + 1e-6, self.upper - 1e-6))
        return out[:count]

    def solve(self, q, position, rotation, count=24, validator=None, extra=None, rest_weight=.008,
              position_tolerance=.006, rotation_tolerance=.05, free_axis=None, seed=0, locked=None, all_solutions=False, stay=0.):
        """
        q: current 28-dof configuration (non-chain dofs are kept as they are).
        position, rotation: tool goal in base_link. free_axis: optional unit vector in the TOOL frame;
        rotation about it is left free (e.g. z for a yaw-free top-down approach). rotation None: the tool's
        attitude is left free altogether (a hand that holds something by a ball joint).
        stay: weight of a term that keeps the solution near the current configuration (for following a path
        step by step: with the attitude free the weak pull to the rest posture alone would wander off).
        extra(q_full, poses) -> (residual vector, None): additional soft terms (finite-difference Jacobian).
        validator(q_full, poses) -> bool. Returns the 11-vector closest to the current one, or None.
        """
        rng = np.random.default_rng(seed); current = np.asarray(q, float)[self.columns]
        lower, upper = self.lower.copy(), self.upper.copy()
        for index, value in (locked or {}).items():
            lower[index] = value - 1e-7; upper[index] = value + 1e-7
        projector = np.eye(3) if free_axis is None else np.eye(3) - np.outer(free_axis, free_axis)
        if rotation is None:
            projector = np.zeros((3, 3))

        def pieces(x):
            full = np.asarray(q, float).copy(); full[self.columns] = x
            poses, frames = self.body.fk(full); t = poses[self.link]; point = t[:3, 3] + t[:3, :3] @ self.tool
            phi = np.zeros(3) if rotation is None else R.from_matrix(rotation @ t[:3, :3].T).as_rotvec()
            return full, poses, frames, t, point, phi

        def residual(x):
            full, poses, frames, t, point, phi = pieces(x)
            # Orientation error expressed in the tool frame so that one tool axis can be released.
            local = projector @ (t[:3, :3].T @ phi)
            parts = [5*(point - position), local, rest_weight*(x - self.rest)]
            if stay:
                parts.append(stay*(x - current))
            if extra is not None:
                parts.append(extra(full, poses))
            return np.concatenate(parts)

        def jacobian(x):
            full, poses, frames, t, point, phi = pieces(x)
            jac = self.body.jacobian(self.link, point, frames, self.columns)
            rot = -projector @ t[:3, :3].T @ _left_jacobian_inverse(phi) @ jac[3:]
            rows = [5*jac[:3], rot, rest_weight*np.eye(len(x))]
            if stay:
                rows.append(stay*np.eye(len(x)))
            if extra is not None:
                base = extra(full, poses); numeric = np.zeros((len(base), len(x)))
                for i in range(len(x)):
                    trial = full.copy(); trial[self.columns[i]] += 1e-5
                    numeric[:, i] = (extra(trial, self.body.fk(trial)[0]) - base) / 1e-5
                rows.append(numeric)
            return np.vstack(rows)

        found = []
        for start in self.seeds(current, position, count, rng):
            result = least_squares(residual, np.clip(start, lower + 1e-9, upper - 1e-9), jac=jacobian, bounds=(lower, upper), max_nfev=60, xtol=1e-9, ftol=1e-10)
            full, poses, frames, t, point, phi = pieces(result.x)
            if np.linalg.norm(point - position) > position_tolerance or np.linalg.norm(projector @ (t[:3, :3].T @ phi)) > rotation_tolerance:
                continue
            if validator is not None and not validator(full, poses):
                continue
            found.append((float(np.linalg.norm(result.x - current)), result.x))
        if not found:
            return [] if all_solutions else None
        found.sort(key=lambda item: item[0])
        return [x for _, x in found] if all_solutions else found[0][1]
