"""Collision checks against observed RGB-D surfaces and public robot geometry."""
import numpy as np
from scipy.spatial import cKDTree
from perception import pointcloud
from fast_geometry import voxel_first_indices


class ObservedSceneCollision:
    def __init__(self, cameras, robot, poses, held, lower, upper):
        self.robot=robot;self.lower=np.asarray(lower);self.upper=np.asarray(upper)
        self._dual=None
        extra=getattr(robot,'extra_payloads',[])
        if extra:
            from dual_scene import DualSceneCollision
            target=None if held is None else dict(pose=held,bounds=[lower,upper])
            self._dual=DualSceneCollision(cameras,robot,poses,extra,target=target,active_bounds=[lower,upper],active_arm='left')
            for name in ('points','tree','self_baseline','scene_baseline','samples'):
                setattr(self,name,getattr(self._dual,name))
            return
        cloud=np.concatenate([pointcloud(c,stride=4)[0] for c in cameras.values()])
        ground=cameras.get('head',next(iter(cameras.values()))).get('ground_from_body',np.eye(4))
        heights=cloud@ground[2,:3]+ground[2,3]
        cloud=cloud[(np.linalg.norm(cloud[:,:2],axis=1)<2.5)&(heights>.015)&(heights<2.6)]
        centers=robot.centers(poses)
        tree=cKDTree(cloud);mask=~robot.point_self_mask(cloud,poses,tree=tree) if robot.hulls else np.ones(len(cloud),bool)
        # Also exclude the public collision envelope: visual hulls can omit
        # small housing details that otherwise appear as self obstacles.
        for ids in tree.query_ball_point(centers,robot.radii+.020):mask[ids]=False
        if held is not None:
            local=(cloud-held[:3,3])@held[:3,:3]
            mask &= ~np.all((local>self.lower-.055)&(local<self.upper+.055),axis=1)
        cloud=cloud[mask]
        if len(cloud):
            ids=voxel_first_indices(cloud);cloud=cloud[ids]
        self.points=cloud;self.tree=cKDTree(cloud)
        self.self_baseline=robot.margins(poses)
        self.scene_baseline=self.sphere_margins(poses)
        axes=[np.linspace(a,b,max(2,int(np.ceil((b-a)/.03))+1)) for a,b in zip(self.lower,self.upper)]
        grid=np.meshgrid(*axes,indexing='ij');self.samples=np.stack(grid,axis=-1).reshape(-1,3)

    def sphere_margins(self, poses):
        return self.tree.query(self.robot.centers(poses),k=1)[0]-self.robot.radii

    def valid(self, poses, held):
        if self._dual is not None:return self._dual.valid(poses,held)
        if not self.robot.valid(poses,self.self_baseline):return False
        margins=self.sphere_margins(poses)
        if np.any(margins<np.minimum(.008,self.scene_baseline-.006)):return False
        if held is None:return True
        if not self.robot.payload_clear(poses,held,self.lower,self.upper):return False
        points=self.samples@held[:3,:3].T+held[:3,3]
        return bool(self.tree.query(points,k=1)[0].min()>.035)

    def diagnose(self, poses, held):
        if self._dual is not None:return self._dual.diagnose(poses,held)
        margins=self.sphere_margins(poses);ids=np.argsort(margins)[:5]
        return dict(points=len(self.points), robot_scene=[(str(self.robot.names[i]),float(margins[i])) for i in ids],
                    payload_scene=None if held is None else float(self.tree.query(self.samples@held[:3,:3].T+held[:3,3],k=1)[0].min()),
                    self_clear=self.robot.valid(poses,self.self_baseline),
                    payload_self_clear=True if held is None else self.robot.payload_clear(poses,held,self.lower,self.upper))
