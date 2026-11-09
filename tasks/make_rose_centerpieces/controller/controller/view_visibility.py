"""Conservative sight-line checks using public hulls and observed surfaces."""
import numpy as np

def sight_line(robot, poses, scene, origin, target, payload=None):
    distance=float(np.linalg.norm(target-origin))
    if distance<=.14:
        return dict(clear=False,reason='target_too_close')
    # Ignore the sensor housing and the final target surface. Sampling the
    # segment, rather than just its endpoint, detects intervening robot links.
    along=np.linspace(.06,distance-.08,max(2,int(np.ceil(distance/.01))))
    points=origin+along[:,None]*(target-origin)/distance
    blocked=robot.point_self_mask(points,poses,padding=.005)
    ids=np.flatnonzero(blocked)
    if len(ids):
        return dict(clear=False,reason='self_occlusion',distance_m=float(along[ids[0]]))
    if payload is not None:
        local=(points-payload[:3,3])@payload[:3,:3]
        ids=np.flatnonzero(((local>=scene.lower-.005)&(local<=scene.upper+.005)).all(axis=1))
        if len(ids):
            return dict(clear=False,reason='payload_occlusion',distance_m=float(along[ids[0]]))
    if len(scene.points):
        blocked=scene.tree.query(points,k=1)[0]<.025
        ids=np.flatnonzero(blocked)
        if len(ids):
            return dict(clear=False,reason='observed_surface_occlusion',distance_m=float(along[ids[0]]))
    return dict(clear=True,reason='observed_ray_clear')
