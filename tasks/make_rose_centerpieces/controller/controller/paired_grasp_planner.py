"""Finite RGB-D paired-grasp planning with shared trunk and full-payload gates."""
import copy,time
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from dual_scene import DualSceneCollision,obb_separated
from pair_geometry import pair_condition
from grasp_regions import grasp_regions

class PlanningTimeout(Exception):pass

class TargetProtectedScene(DualSceneCollision):
    """Excluded grasp targets still block every non-designated robot link."""
    def __init__(self,*args,protected_targets=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.protected_targets=list(protected_targets or [])
    def valid(self,poses,held=None):
        return super().valid(poses,held) and all(self.payload_robot_clear(poses,pose,bounds,hand) for hand,pose,bounds in self.protected_targets)

def coupled_ik(left,right,robot,q,left_goal,right_goal,baseline=None,deadline=None,seed=None):
    """Solve trunk+both arms against pad poses; final collision validation is caller-owned.

    Goals are4x4 robot-body transforms; no simulator geometry/state is queried.
    A self-collision penalty selects useful redundant postures without weakening
    the caller's exact endpoint and swept-path admission requirements.
    """
    ids=left.indices+right.indices[4:]
    lower=np.r_[left.limits[0],right.limits[0,4:]]
    upper=np.r_[left.limits[1],right.limits[1,4:]]
    baseline=robot.margins(left.kin.forward(q)[0]) if baseline is None else baseline
    threshold=np.minimum(-.003,baseline-.003)+.002
    def residual(x):
        if deadline is not None and time.monotonic()>=deadline:raise PlanningTimeout()
        trial=q.copy();trial[ids]=x;l=left.pose(trial);r=right.pose(trial)
        return np.r_[5*(l[:3,3]-left_goal[:3,3]),R.from_matrix(left_goal[:3,:3]@l[:3,:3].T).as_rotvec(),
                     5*(r[:3,3]-right_goal[:3,3]),R.from_matrix(right_goal[:3,:3]@r[:3,:3].T).as_rotvec(),
                     .008*(x-q[ids]),20*np.maximum(0,threshold-robot.margins(left.kin.forward(trial)[0]))]
    x=q[ids] if seed is None else np.asarray(seed)[ids]
    fit=least_squares(residual,np.clip(x,lower+1e-5,upper-1e-5),bounds=(lower,upper),max_nfev=120)
    error=residual(fit.x)
    if not(np.linalg.norm(error[:3])/5<.015 and np.linalg.norm(error[3:6])<.12 and np.linalg.norm(error[6:9])/5<.015 and np.linalg.norm(error[9:12])<.12):return None
    trial=q.copy();trial[ids]=fit.x;return trial

def plan_pair(robot,left,right,cameras,q,model,level,candidates,vase_body,step,seconds=35.):
    """Return complete serial dual-pickup joint paths, or fail closed in finite time.

    Candidates must come from independently supported onboard-image matching.
    The caller must still confirm both physical pickups before vase transfer.
    Scene points are unprojected once at actual q; no hypothetical image poses.
    """
    start=time.monotonic();deadline=start+float(seconds);debug={'reason':'no_complete_pair_path','trials':[]}
    eligibility=pair_condition(candidates,vase_body,level,step)
    if not eligibility['eligible']:return None,dict(debug,reason=eligibility['reason'])
    tracks={c['track_id']:c for c in candidates};bounds=model['bounds'];up=level[:3,:3].T@np.array([0.,0.,1.])
    def check_time():
        if time.monotonic()>=deadline:raise PlanningTimeout()
    def sweep(a,b,valid):
        for f in np.linspace(0.,1.,101):
            check_time()
            if not valid(a+(b-a)*f):return False
        return True
    def goal(obj,sign):
        regions=[r for r in grasp_regions('rose',model) if r[3].startswith('blossom') and abs((level[:3,:3]@obj[:3,r[1]])[2])<.30]
        if not regions:return None
        center,axis,width,name=regions[0];closing=obj[:3,axis]*sign;approach=up-closing*(closing@up)
        if np.linalg.norm(approach)<.35:return None
        approach/=np.linalg.norm(approach);g=np.eye(4);g[:3,:3]=np.column_stack([np.cross(closing,approach),closing,approach]);g[:3,3]=obj[:3,:3]@center+obj[:3,3]+up*.007
        return g,width,name
    def inject(full,servo,x):
        trial=full.copy();trial[servo.indices]=x;return trial
    pairs=sorted(eligibility['pairs'],key=lambda pair:-np.asarray(pair['left_pose'])[1,3]+np.asarray(pair['right_pose'])[1,3])
    try:
        for pair in pairs:
            a=np.asarray(pair['left_pose']);b=np.asarray(pair['right_pose'])
            scene=TargetProtectedScene(cameras,robot,left.kin.forward(q)[0],target={'pose':a,'bounds':bounds},protected_targets=[('left',a,bounds),('right',b,bounds)])
            # Both supported intended targets are excluded only for fingertip
            # contact; carried target separation is checked explicitly below.
            local=(scene.points-b[:3,3])@b[:3,:3]
            keep=~np.all((local>bounds[0]-.055)&(local<bounds[1]+.055),axis=1)
            scene.points=scene.points[keep];scene.tree=cKDTree(scene.points);scene.scene_baseline=scene.sphere_margins(left.kin.forward(q)[0])
            def with_payloads(ds):
                s=copy.copy(scene);s.descriptors=[dict(d,samples=s._samples(d['bounds'])) for d in ds];held_hands={d['arm'] for d in ds};s.protected_targets=[t for t in scene.protected_targets if t[0] not in held_hands];return s
            for sl in (-1.,1.):
                for sr in (1.,-1.):
                    check_time();lg=goal(a,sl);rg=goal(b,sr)
                    if lg is None or rg is None:continue
                    gl,lwidth,lregion=lg;gr,rwidth,rregion=rg
                    down=coupled_ik(left,right,robot,q,gl,gr,scene.self_baseline,deadline)
                    record={'left_id':pair['left_id'],'right_id':pair['right_id'],'signs':[sl,sr],'coupled_descent_ik':down is not None};debug['trials'].append(record)
                    if down is None or not scene.valid(left.kin.forward(down)[0]):continue
                    locks={i:float(down[left.indices[i]]) for i in range(4)};staged=None
                    valid_empty=lambda v:scene.valid(left.kin.forward(v)[0])
                    for height in (.025,.05,.08,.10,.14):
                        check_time();lp=gl[:3,3]+up*height;rp=gr[:3,3]+up*height
                        lx=left.solve(down,lp,gl[:3,:3],locked=locks)
                        if lx is None:continue
                        trial=inject(down,left,lx);rx=right.solve(trial,rp,gr[:3,:3],locked=locks)
                        if rx is None:continue
                        trial=inject(trial,right,rx)
                        if sweep(q,trial,valid_empty):staged=trial;record['pre_height']=height;break
                    if staged is None:continue
                    paths={'stage':[staged]};current=staged.copy()
                    def segment(name,servo,g,ss,left_payload=None,cartesian=False):
                        nonlocal current
                        def valid(v):
                            if not ss.valid(left.kin.forward(v)[0]):return False
                            return left_payload is None or obb_separated(left.pose(v)@left_payload,bounds,b,bounds)
                        origin=servo.pose(current);nodes=[]
                        for fraction in (np.linspace(.05,1.,20) if cartesian else [1.]):
                            check_time();point=origin[:3,3]+(g[:3,3]-origin[:3,3])*fraction
                            x=servo.solve(current,point,g[:3,:3],locked=locks,validator=valid)
                            if x is None:return False
                            target=inject(current,servo,x)
                            if not sweep(current,target,valid):return False
                            nodes.append(target);current=target
                        paths[name]=nodes;return True
                    if not segment('left_descend',left,gl,scene):continue
                    llocal=np.linalg.inv(left.pose(current))@a
                    leftscene=with_payloads([{'arm':'left','local':llocal,'bounds':bounds}]);lift=left.pose(current);lift[:3,3]+=up*.18
                    if not segment('left_lift',left,lift,leftscene,llocal,cartesian=True):continue
                    if not segment('right_descend',right,gr,leftscene,llocal):continue
                    rlocal=np.linalg.inv(right.pose(current))@b
                    bothscene=with_payloads([{'arm':'left','local':llocal,'bounds':bounds},{'arm':'right','local':rlocal,'bounds':bounds}]);lift=right.pose(current);lift[:3,3]+=up*.18
                    if not segment('right_lift',right,lift,bothscene):continue
                    result=dict(pair=pair,paths=paths,left_grasp=gl,right_grasp=gr,left_width=lwidth,right_width=rwidth,left_region=lregion,right_region=rregion,initial_left=a,initial_right=b,initial_q=q.copy(),initial_base=None,planned_left_local=llocal,planned_right_local=rlocal)
                    return result,dict(debug,reason='complete_checked_pair_path',seconds=time.monotonic()-start)
    except PlanningTimeout:debug['reason']='finite_planning_timeout'
    debug['seconds']=time.monotonic()-start;return None,debug
