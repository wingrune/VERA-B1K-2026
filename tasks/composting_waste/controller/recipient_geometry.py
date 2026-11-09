"""Canonical ashcan geometry, RGB-D articulated lid fitting, and release candidates."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar
from compost_skills import OrientedBoxes,transform,upright

ROOT=Path(__file__).parent

def lid_transform(angle):
    data=json.loads((ROOT/'priors/ashcan_lid/joint_geometry.json').read_text())['j_link_0']
    parent=transform(Rotation.from_quat(data['physics:localRot0']).as_matrix(),data['physics:localPos0'])
    child=transform(Rotation.from_quat(data['physics:localRot1']).as_matrix(),data['physics:localPos1'])
    return parent@transform(Rotation.from_euler('y',float(angle)).as_matrix(),[0.,0.,0.])@np.linalg.inv(child)

def lid_from_rgbd(observer,bin_pose,base):
    """Estimate a joint angle from projected visible depth; no default instance angle."""
    records=[]
    for camera in ('head','left','right'):
        def evaluate(angle):
            report=observer.depth_agreement('ashcan_lid',camera,base,bin_pose@lid_transform(angle),tolerance=.015,occlusion=.012)
            if report['points']<12 or report['visible_fraction']<.25:return 1.,report
            score=report['cost']+.035*(1.-report['agreement'])+.012*(1.-report['visible_fraction'])
            return score,report
        coarse=[(evaluate(a)[0],float(a)) for a in np.linspace(0.,3.124139,40)]
        score,angle=min(coarse)
        if score>=1.:continue
        fit=minimize_scalar(lambda a:evaluate(a)[0],bounds=(max(0.,angle-.10),min(3.124139,angle+.10)),method='bounded',options={'xatol':.004})
        score,report=evaluate(float(fit.x))
        if report['agreement']<.70 or score>.025:continue
        records.append(dict(angle=float(fit.x),camera=camera,score=float(score),depth=report))
    if not records:return None
    records.sort(key=lambda x:x['score'])
    return records[0]

def recipient_guards(bin_pose,lid_angle):
    boxes=[]
    for category,frame in [('ashcan',np.eye(4)),('ashcan_lid',lid_transform(lid_angle))]:
        geometry=np.load(ROOT/f'priors/{category}/collision.npz')
        for key in geometry.files:
            if not key.endswith('|v'):continue
            vertices=geometry[key]@frame[:3,:3].T+frame[:3,3]
            boxes.append(np.r_[vertices.min(0),vertices.max(0)])
    return OrientedBoxes(boxes,bin_pose)

def release_goals(ctx,category,bin_pose,lid_angle,occupied=()):
    """Narrow-axis orientation and a rim-clear release above the measured aperture.

    Region comes from the back wall and hinge-bar collision meshes, not object world coordinates.
    Object support is kept above the highest current recipient collision point intersecting its
    projected footprint, with existing gripper/payload validators authoritative before release.
    """
    guard=recipient_guards(bin_pose,lid_angle);model=ctx.models[category]['xyz']
    candidates=[]
    for yaw in np.linspace(-np.pi,np.pi,72,endpoint=False):
        rotation=upright(yaw);shaped=model@rotation.T;lo=shaped.min(0);hi=shaped.max(0);span=hi-lo
        # Canonical back opening; native collision mesh leaves x[-.101,.005], y[-.13,.13].
        if span[0]>.095 or span[1]>.235:continue
        for y in (-.035,.035,0.):
            centre=np.array([-.052,y]);point=np.r_[centre-(lo[:2]+hi[:2])/2,0.]
            low=point[:2]+lo[:2];high=point[:2]+hi[:2]
            if low[0]<-.101 or high[0]>.005 or low[1]<-.13 or high[1]>.13:continue
            relevant=np.all(guard.boxes[:,3:5]>=low-.022,axis=1)&np.all(guard.boxes[:,:2]<=high+.022,axis=1)
            rim=float(guard.boxes[relevant,5].max()) if np.any(relevant) else float(guard.boxes[:,5].max())
            point[2]=rim+.075-lo[2]
            # Validate release height using exactly the same conservative payload lattice
            # as Context, plus the open-finger sweep. The narrow actual fruit silhouette
            # selects XY/yaw, while its collision envelope determines required rim clearance.
            bounds=ctx.models[category]['bounds'];spacing=.012
            axes=[np.linspace(a,b,max(2,int(np.ceil((b-a)/spacing))+1)) for a,b in zip(*bounds)]
            payload=np.stack(np.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
            payload=payload@rotation.T;payload_radius=spacing*.6
            arm,_,relation=ctx.held
            tip=transform(rotation,[0.,0.,0.])@np.linalg.inv(relation)
            finger=ctx.gripper[arm+'_collision'][::max(1,len(ctx.gripper[arm+'_collision'])//400)]-ctx.iks[arm].tool
            finger=finger@tip[:3,:3].T+tip[:3,3]
            def clear(z):
                origin=np.r_[point[:2],z]
                for samples,radius,margin in ((payload,payload_radius,.020),(finger,.0008,.020)):
                    pts=samples+origin;near=np.clip(pts[None],guard.boxes[:,None,:3],guard.boxes[:,None,3:])
                    if np.min(np.linalg.norm(pts[None]-near,axis=2)-radius)<margin:return False
                return True
            maximum=float(guard.boxes[:,5].max())+.18-lo[2]
            while not clear(point[2]) and point[2]<maximum:point[2]+=.005
            if not clear(point[2]):continue
            clearance=min([np.linalg.norm(centre-np.asarray(other)[:2]) for other in occupied] or [.20])
            score=float(span[0])-.03*clearance+.02*abs(y)
            candidates.append((score,bin_pose@transform(rotation,point)))
    candidates.sort(key=lambda x:x[0]);selected=[]
    for _,pose in candidates:
        if any(np.linalg.norm(pose[:2,3]-old[:2,3])<.02 and np.linalg.norm(pose[:3,:3]-old[:3,:3])<.4 for old in selected):continue
        selected.append(pose)
        if len(selected)>=12:break
    return selected
