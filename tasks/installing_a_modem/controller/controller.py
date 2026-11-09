"""Static-URDF arm servo with bounded native joint actions."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from kinematics import Kinematics,from_proprio
class ArmServo:
    def __init__(self,assets,arm='left'):
        self.spec=json.loads((Path(assets)/'robot_model.json').read_text());self.names=self.spec['dof_names'];self.kin=Kinematics(Path(assets)/'r1pro.urdf',self.names);self.arm=arm
        self.indices=[self.names.index(n) for c in self.spec['controllers'] if c['name'] in ['trunk','arm_'+arm] for n in c['dof_names']]
        self.limits=np.asarray(self.spec['limits'])[:,self.indices];self.limits[:,3]=[-.75,.75];self.link=arm+'_gripper_link';self.point=np.array([0.,0.,-.060]);self.target=None;self.synchronized=False
    def q(self,proprio):return from_proprio(proprio,self.names)
    def pose(self,q):
        t=self.kin.forward(q)[0][self.link].copy();t[:3,3]+=t[:3,:3]@self.point;return t
    def capsules(self,q):
        poses=self.kin.forward(q)[0];result=[]
        for _,a,b,_,_,_ in self.kin.joints:
            if a.startswith(('left_arm','right_arm','left_gripper','right_gripper')) or b.startswith(('left_arm','right_arm','left_gripper','right_gripper')):result.append((poses[a][:3,3],poses[b][:3,3],.10))
        return result
    def solve(self,q,goal,rotation,validator=None):
        rest=np.r_[.5,-.65,-.5,0.,-1.4,.35,.2,-.7,0.,.3,0.];seeds=[q[self.indices],rest,rest+np.r_[0,0,0,0,0,.5,-.8,0,0,0,0]];choices=[]
        def residual(x):
            trial=q.copy();trial[self.indices]=x;t=self.pose(trial)
            return np.r_[5*(t[:3,3]-goal),R.from_matrix(rotation@t[:3,:3].T).as_rotvec(),.008*(x-rest)]
        for seed in seeds:
            result=least_squares(residual,np.clip(seed,self.limits[0]+1e-5,self.limits[1]-1e-5),bounds=(self.limits[0],self.limits[1]),max_nfev=90)
            v=residual(result.x)
            if np.linalg.norm(v[:3])/5<.015 and np.linalg.norm(v[3:6])<.12:
                trial=q.copy();trial[self.indices]=result.x
                if validator is None or validator(trial):choices.append((np.linalg.norm(result.x-q[self.indices]),result.x))
        return min(choices,key=lambda x:x[0])[1] if choices else None
    def step(self,action,q,goal,rotation,plan=False,gaze_pitch=None):
        cache=self.kin.forward(q);t=cache[0][self.link];point=t[:3,3]+t[:3,:3]@self.point;err=goal-point;ori=R.from_matrix(rotation@t[:3,:3].T).as_rotvec()
        if plan and self.target is None:self.target=self.solve(q,goal,rotation)
        if self.target is not None:
            delta=self.target-q[self.indices]
            dq=delta*min(1.,.016/max(np.max(np.abs(delta)),1e-9)) if self.synchronized else np.clip(delta,-.016,.016)
            if np.max(np.abs(self.target-q[self.indices]))<.035:self.target=None
        else:
            J=self.kin.jacobian(self.link,q,self.point,cache)[:,self.indices];delta=np.r_[np.clip(err*.25,-.012,.012),np.clip(ori*.2,-.03,.03)]
            if gaze_pitch is not None:
                gaze=np.zeros(len(self.indices));gaze[:3]=[1.,1.,-1.]
                current=q[self.indices];error_pitch=gaze_pitch-(current[0]+current[1]-current[2])
                J=np.vstack([J,.5*gaze]);delta=np.r_[delta,np.clip(error_pitch*.10,-.012,.012)*.5]
            dq=J.T@np.linalg.solve(J@J.T+.002*np.eye(len(delta)),delta);dq=np.clip(dq,-.016,.016)
        command=np.clip(q[self.indices]+dq,self.limits[0]+1e-5,self.limits[1]-1e-5);action[3:7]=command[:4];action[7:14] = command[4:] if self.arm=='left' else action[7:14]
        if self.arm=='right':action[15:22]=command[4:]
        return float(np.linalg.norm(err)),float(np.linalg.norm(ori))
    def step_position(self,action,q,goal):
        """Move a calibrated fingertip without imposing an unnecessary wrist orientation."""
        cache=self.kin.forward(q);t=cache[0][self.link];point=t[:3,3]+t[:3,:3]@self.point;err=goal-point
        J=self.kin.jacobian(self.link,q,self.point,cache)[:3,self.indices]
        delta=np.clip(err*.20,-.008,.008);dq=J.T@np.linalg.solve(J@J.T+.002*np.eye(3),delta)
        command=np.clip(q[self.indices]+np.clip(dq,-.012,.012),self.limits[0]+1e-5,self.limits[1]-1e-5)
        action[3:7]=command[:4]
        if self.arm=='left':action[7:14]=command[4:]
        else:action[15:22]=command[4:]
        return float(np.linalg.norm(err)),0.
