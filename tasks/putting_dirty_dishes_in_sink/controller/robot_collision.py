"""Self-collision checks from the public R1Pro collision-sphere configuration."""
import json
from pathlib import Path
import numpy as np


class RobotCollision:
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        self.groups = []
        names, centers, radii = [], [], []
        buffers = data.get('self_collision_buffer', {})
        for name, spheres in data['collision_spheres'].items():
            if name.startswith('attached_object'):
                continue
            start = len(names)
            for sphere in spheres:
                if sphere['radius'] <= 0:
                    continue
                names.append(name); centers.append(sphere['center'])
                radii.append(sphere['radius']+buffers.get(name, 0))
            self.groups.append((name, slice(start,len(names))))
        self.names=np.array(names);self.local=np.asarray(centers);self.radii=np.asarray(radii)
        ignore={frozenset((name,other)) for name,others in data['self_collision_ignore'].items() for other in others}
        pairs=[(i,j) for i in range(len(names)) for j in range(i+1,len(names))
               if names[i]!=names[j] and frozenset((names[i],names[j])) not in ignore]
        self.a,self.b=np.array(pairs).T
        self.required=self.radii[self.a]+self.radii[self.b]

    def centers(self, poses):
        points=np.empty_like(self.local)
        for name,sl in self.groups:
            t=poses[name];points[sl]=self.local[sl]@t[:3,:3].T+t[:3,3]
        return points

    def margins(self, poses):
        points=self.centers(poses)
        return np.linalg.norm(points[self.a]-points[self.b],axis=1)-self.required

    def valid(self, poses, baseline=None):
        margin=self.margins(poses)
        threshold=np.full(len(margin),-.003) if baseline is None else np.minimum(-.003,baseline-.003)
        return bool(np.all(margin>=threshold))

    def worst(self, poses, limit=5):
        margin=self.margins(poses)
        return [(str(self.names[self.a[i]]),str(self.names[self.b[i]]),float(margin[i])) for i in np.argsort(margin)[:limit]]

    def payload_clear(self, poses, held, lower, upper):
        points=(self.centers(poses)-held[:3,3])@held[:3,:3]
        distance=np.linalg.norm(points-np.clip(points,lower,upper),axis=1)-self.radii
        allowed=np.array([name.startswith('left_gripper') or name in ['left_arm_link7','left_realsense_link'] for name in self.names])
        return bool(np.all(distance[~allowed]>.005))
