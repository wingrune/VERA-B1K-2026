"""Self-collision checks from the public R1Pro collision-sphere configuration."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree


class RobotCollision:
    def __init__(self, path):
        data = json.loads(Path(path).read_text())
        geometry=Path(path).with_name('robot_geometry.json')
        self.hulls=json.loads(geometry.read_text())['colliders'] if geometry.exists() else []
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

    def point_self_mask(self,points,poses,padding=.012,tree=None):
        mask=np.zeros(len(points),bool);tree=cKDTree(points) if tree is None else tree
        for hull in self.hulls:
            if hull['link'] not in poses:continue
            pose=poses[hull['link']];lo,hi=np.asarray(hull['bounds']);center=(lo+hi)/2
            radius=np.linalg.norm((hi-lo)/2+padding)+1e-8
            ids=np.asarray(tree.query_ball_point(center@pose[:3,:3].T+pose[:3,3],radius),dtype=int)
            if not len(ids):continue
            ids=ids[~mask[ids]];local=(points[ids]-pose[:3,3])@pose[:3,:3]
            keep=((local>=lo-padding)&(local<=hi+padding)).all(axis=1);ids=ids[keep];local=local[keep]
            if not len(ids):continue
            eq=np.asarray(hull['equations'])
            for chunk in np.array_split(np.arange(len(ids)),max(1,int(np.ceil(len(ids)/128)))):
                mask[ids[chunk]]=np.all(local[chunk]@eq[:,:3].T+eq[:,3]<=padding,axis=1)
        for name,c,r in zip(self.names,self.centers(poses),self.radii):
            if name.startswith(('wheel_motor','steer_motor')):
                ids=np.asarray(tree.query_ball_point(c,r+.025+1e-8),dtype=int)
                mask[ids]|=np.linalg.norm(points[ids]-c,axis=1)<r+.025
        return mask
