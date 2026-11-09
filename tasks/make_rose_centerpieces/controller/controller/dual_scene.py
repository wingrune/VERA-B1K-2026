"""RGB-D dual-held collision checks. No simulator state or attach operations."""
import numpy as np
from scipy.spatial import cKDTree
from perception import pointcloud
from fast_geometry import voxel_first_indices

PAD_POINT=np.array([0.,0.,-.072])

def pad_pose(poses,arm):
    hand=np.asarray(poses[arm+'_gripper_link']).copy()
    hand[:3,3]+=hand[:3,:3]@PAD_POINT
    return hand

def obb_separated(a,ab,b,bb,margin=.005):
    al,ah=np.asarray(ab);bl,bh=np.asarray(bb)
    ac=a[:3,3]+a[:3,:3]@((al+ah)/2);bc=b[:3,3]+b[:3,:3]@((bl+bh)/2)
    ae=(ah-al)/2;be=(bh-bl)/2
    axes=[*a[:3,:3].T,*b[:3,:3].T]
    axes.extend(np.cross(x,y) for x in a[:3,:3].T for y in b[:3,:3].T)
    for axis in axes:
        norm=np.linalg.norm(axis)
        if norm<1e-8:continue
        axis=axis/norm
        gap=abs((bc-ac)@axis)-abs(a[:3,:3].T@axis)@ae-abs(b[:3,:3].T@axis)@be
        if gap>margin:return True
    return False

class DualSceneCollision:
    def __init__(self,cameras,robot,poses,payload_descriptors=None,target=None,active_bounds=None,active_arm="left"):
        self.robot=robot;self.descriptors=[];self.active_arm=active_arm;self.active_bounds=None if active_bounds is None else np.asarray(active_bounds,dtype=float)
        for d in payload_descriptors or []:
            arm=d['arm']
            if arm not in ('left','right'):raise ValueError('Unknown payload arm')
            local=np.asarray(d['local'],dtype=float);bounds=np.asarray(d['bounds'],dtype=float)
            if local.shape!=(4,4) or bounds.shape!=(2,3) or not np.isfinite(local).all() or not np.isfinite(bounds).all():raise ValueError('Invalid payload geometry')
            if np.any(bounds[1]<bounds[0]):raise ValueError('Reversed payload bounds')
            self.descriptors.append(dict(arm=arm,local=local.copy(),bounds=bounds.copy()))
        if len({d['arm'] for d in self.descriptors})!=len(self.descriptors):raise ValueError('Duplicate hand payload')
        clouds=[pointcloud(c,stride=4)[0] for c in cameras.values()]
        cloud=np.concatenate(clouds) if clouds else np.empty((0,3));self.observation_available=len(cloud)>0
        ground=next(iter(cameras.values())).get('ground_from_body',np.eye(4)) if cameras else np.eye(4)
        heights=cloud@ground[2,:3]+ground[2,3]
        cloud=cloud[(np.linalg.norm(cloud[:,:2],axis=1)<2.5)&(heights>.015)&(heights<2.6)]
        tree=cKDTree(cloud)
        mask=~robot.point_self_mask(cloud,poses,tree=tree) if robot.hulls else np.ones(len(cloud),bool)
        for ids in tree.query_ball_point(robot.centers(poses),robot.radii+.020):mask[ids]=False
        objects=[(pad_pose(poses,d['arm'])@d['local'],d['bounds']) for d in self.descriptors]
        if target is not None:
            if isinstance(target,dict):objects.append((np.asarray(target['pose']),np.asarray(target['bounds'])))
            else:objects.append((np.asarray(target[0]),np.asarray([target[1],target[2]])))
        # Match existing current-object exclusion; keep this limitation explicit in reports.
        for obj,(lo,hi) in objects:
            local=(cloud-obj[:3,3])@obj[:3,:3]
            mask&=~np.all((local>lo-.055)&(local<hi+.055),axis=1)
        cloud=cloud[mask]
        if len(cloud):
            ids=voxel_first_indices(cloud);cloud=cloud[ids]
        self.points=cloud;self.tree=cKDTree(cloud);self.self_baseline=robot.margins(poses);self.scene_baseline=self.sphere_margins(poses)
        bounds=self.active_bounds if self.active_bounds is not None else self.descriptors[0]['bounds'] if self.descriptors else (np.asarray(target['bounds']) if isinstance(target,dict) else np.zeros((2,3)))
        self.lower,self.upper=bounds;self.samples=self._samples(bounds)
        for d in self.descriptors:d['samples']=self._samples(d['bounds'])
    @staticmethod
    def _samples(bounds):
        lo,hi=bounds;axes=[np.linspace(a,b,max(2,int(np.ceil((b-a)/.03))+1)) for a,b in zip(lo,hi)]
        return np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    def sphere_margins(self,poses):return self.tree.query(self.robot.centers(poses))[0]-self.robot.radii
    def payload_robot_clear(self,poses,payload,bounds,arm):
        lo,hi=bounds;local=(self.robot.centers(poses)-payload[:3,3])@payload[:3,:3]
        distance=np.linalg.norm(local-np.clip(local,lo,hi),axis=1)-self.robot.radii
        allowed=np.array([str(n).startswith(arm+'_gripper') or n in [arm+'_arm_link7',arm+'_realsense_link'] for n in self.robot.names])
        return bool(np.all(distance[~allowed]>.005))
    def payload_poses(self,poses,active_payload=None):
        payloads=[(d,pad_pose(poses,d['arm'])@d['local']) for d in self.descriptors]
        if active_payload is not None and self.active_bounds is not None:
            if any(d['arm']==self.active_arm for d,_ in payloads):raise ValueError('Active payload duplicates descriptor hand')
            payloads.append((dict(arm=self.active_arm,bounds=self.active_bounds,samples=self._samples(self.active_bounds)),np.asarray(active_payload)))
        return payloads
    def valid(self,poses,held=None):
        if not self.observation_available:return False
        if not self.robot.valid(poses,self.self_baseline):return False
        if np.any(self.sphere_margins(poses)<np.minimum(.008,self.scene_baseline-.006)):return False
        payloads=self.payload_poses(poses,held)
        for d,obj in payloads:
            if not self.payload_robot_clear(poses,obj,d['bounds'],d['arm']):return False
            samples=d['samples']@obj[:3,:3].T+obj[:3,3]
            if self.tree.query(samples)[0].min()<=.035:return False
        for i,(a,ap) in enumerate(payloads):
            for b,bp in payloads[i+1:]:
                if not obb_separated(ap,a['bounds'],bp,b['bounds'],margin=.005):return False
        return True
    def diagnose(self,poses,held=None):
        rows=[];payloads=self.payload_poses(poses,held)
        for d,obj in payloads:rows.append({'arm':d['arm'],'robot_clear':self.payload_robot_clear(poses,obj,d['bounds'],d['arm']),'scene_min_m':float(self.tree.query(d['samples']@obj[:3,:3].T+obj[:3,3])[0].min())})
        margins=self.sphere_margins(poses);ids=np.argsort(margins)[:5];self_clear=self.robot.valid(poses,self.self_baseline)
        return {'observation_available':self.observation_available,'points':len(self.points),'robot_self_clear':self_clear,'self_clear':self_clear,'robot_scene':[(str(self.robot.names[i]),float(margins[i])) for i in ids],'payload_scene':min((r['scene_min_m'] for r in rows),default=None),'payload_self_clear':all(r['robot_clear'] for r in rows),'payloads':rows,'pair_clear':all(obb_separated(a,da['bounds'],b,db['bounds']) for i,(da,a) in enumerate(payloads) for db,b in payloads[i+1:])}
