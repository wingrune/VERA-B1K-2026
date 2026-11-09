"""Joint RGB-D refinement of the observed bin body and its articulated lid."""
import numpy as np
from scipy.optimize import differential_evolution
from recipient_geometry import lid_transform
from compost_skills import upright

def refine(observer,base,seed,log=lambda *args:None):
    records=[]
    for camera in ('head','left','right'):
        def pose(x):
            out=seed.copy();out[:2,3]+=x[:2];out[:3,:3]=upright(x[2])@seed[:3,:3];return out
        def evaluate(x):
            body=pose(x)
            a=observer.depth_agreement('ashcan',camera,base,body,tolerance=.020,occlusion=.014)
            b=observer.depth_agreement('ashcan_lid',camera,base,body@lid_transform(float(x[3])),tolerance=.015,occlusion=.012)
            if a['points']<25 or a['visible_fraction']<.30 or b['points']<12 or b['visible_fraction']<.25:return 1.,a,b
            lid_score=b['cost']+.035*(1.-b['agreement'])+.012*(1.-b['visible_fraction'])
            return lid_score+.4*a['cost']+.012*(1.-a['agreement']),a,b
        result=differential_evolution(lambda x:evaluate(x)[0],[(-.035,.035),(-.035,.035),(-.20,.20),(0.,3.124139)],seed=0,popsize=6,maxiter=24,tol=.003,polish=False,workers=1)
        cost,a,b=evaluate(result.x);lid_score=b['cost']+.035*(1.-b['agreement'])+.012*(1.-b['visible_fraction'])
        record=dict(camera=camera,x=result.x.tolist(),objective=float(cost),body=a,lid=b,lid_score=float(lid_score));log('JOINT_FIT',record)
        if a['points']<25 or a['visible_fraction']<.30 or a['agreement']<.60:continue
        if b['points']<12 or b['visible_fraction']<.25 or b['agreement']<.70 or lid_score>.025:continue
        records.append((cost,pose(result.x),dict(angle=float(result.x[3]),camera=camera,score=lid_score,depth=b,body_depth=a,joint_refinement=True)))
    if not records:return None
    records.sort(key=lambda row:row[0]);return records[0][1:]
