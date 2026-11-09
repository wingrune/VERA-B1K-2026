"""Keep category tracks exclusive when current RGB-D confirms another object."""
import numpy as np
from pose_confirmation import confirm_tool


def distinct_tool_fits(observer, category, base, fits, tracks):
    if category == 'tote' or not fits:
        return fits
    confirmed = []
    for other, track in tracks.items():
        if other in (category, 'tote'):
            continue
        pose = track['world']
        bounds = observer.models[other]['bounds']
        extent = np.linalg.norm(bounds[1] - bounds[0])
        if not any(np.linalg.norm(candidate[:3, 3] - pose[:3, 3]) <= extent for candidate, _ in fits):
            continue
        evidence = confirm_tool(observer, other, base, pose)
        if evidence is None:
            # A stale map track can be centimetres off after navigation. Reobserve
            # the competing category before claiming that it occupies this place.
            support = float(observer.scene.meta['scope']['desk.n.01_1']['aabb_hi'][2])
            alternatives = observer.fits(other, base, support, prior=pose)
            for alternative, _ in alternatives:
                if np.linalg.norm(alternative[:3, 3] - pose[:3, 3]) > extent:
                    continue
                evidence = confirm_tool(observer, other, base, alternative)
                if evidence is not None:
                    pose = alternative
                    break
        if evidence is None:
            continue
        tolerance = max(view['tolerance'] for view in evidence['views'])
        confirmed.append((other, pose, bounds, tolerance))
    accepted = []
    center = observer.models[category]['bounds'].mean(0)
    for pose, info in fits:
        world_center = pose[:3, :3] @ center + pose[:3, 3]
        conflicts = []
        for other, other_pose, bounds, tolerance in confirmed:
            local = (world_center - other_pose[:3, 3]) @ other_pose[:3, :3]
            if np.all(local >= bounds[0] - tolerance) and np.all(local <= bounds[1] + tolerance):
                conflicts.append(other)
        if conflicts:
            observer.debug.setdefault(category, []).append(dict(
                rejected=dict(reason='current RGB-D confirms a different category at this location',
                              categories=conflicts, pose=pose.tolist())))
        else:
            accepted.append((pose, info))
    return accepted
