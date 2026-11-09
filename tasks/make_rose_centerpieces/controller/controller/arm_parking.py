"""Plan a bounded, observation-checked parking motion for the unused arm."""
import numpy as np
from scipy.optimize import least_squares
from controller import ArmServo
from robot_collision import RobotCollision
from scene_collision import ObservedSceneCollision


def plan_right_parking(assets,q,cameras):
    servo=ArmServo(assets,arm='right')
    robot=RobotCollision(assets/'robot_collision_spheres.json')
    ids=servo.indices[4:]
    limits=np.asarray(servo.spec['limits'])[:,ids]
    start=servo.kin.forward(q)[0]
    baseline=robot.margins(start)
    scene=ObservedSceneCollision(cameras,robot,start,None,[0,0,0],[0,0,0])
    choices=[]
    for goal in [[.20,-.45,.85],[.10,-.50,.85],[.20,-.40,1.0],[.10,-.45,.70]]:
        def residual(x):
            trial=q.copy();trial[ids]=x
            pose=servo.pose(trial);margin=robot.margins(servo.kin.forward(trial)[0])
            return np.r_[5*(pose[:3,3]-goal),.01*(x-q[ids]),
                         20*np.minimum(0,margin-np.minimum(-.003,baseline-.003))]
        fit=least_squares(residual,q[ids],bounds=(limits[0]+1e-5,limits[1]-1e-5),max_nfev=120)
        trial=q.copy();trial[ids]=fit.x
        if np.linalg.norm(servo.pose(trial)[:3,3]-goal)>.035:continue
        if not all(scene.valid(servo.kin.forward(q+a*(trial-q))[0],None) for a in np.linspace(0,1,35)):continue
        choices.append((np.linalg.norm(fit.x-q[ids]),fit.x.copy()))
    return min(choices,key=lambda x:x[0])[1] if choices else None
