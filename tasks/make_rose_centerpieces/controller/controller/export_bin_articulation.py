"""OFFLINE training-template mesh/joint prior. Never imported by online control."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from export_priors import array

def export_bin_articulation(evaluator,output):
 import omnigibson.lazy as lazy
 out=Path(output);out.mkdir(parents=True,exist_ok=True)
 scope=evaluator.instance_eval_states[0].env_accessor.object_scope
 obj=next(obj for key,obj in scope.items() if 'ashcan' in key)
 rp,rq=obj.root_link.get_position_orientation();rp=array(rp);rr=R.from_quat(array(rq)).as_matrix();parts={};rng=np.random.default_rng(90)
 for name,link in obj.links.items():
  lp,lq=link.get_position_orientation();lp=array(lp);lr=R.from_quat(array(lq)).as_matrix();cloud=[]
  # Use one USD frame for mesh-to-link geometry. USD world transforms and
  # live physics link poses can differ before the first synchronization.
  link_matrix=np.array(lazy.pxr.UsdGeom.Xformable(link.prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default()))
  for prim in lazy.pxr.Usd.PrimRange(link.prim):
   if not prim.IsA(lazy.pxr.UsdGeom.Mesh) or 'visual' not in str(prim.GetPath()).lower():continue
   mesh=lazy.pxr.UsdGeom.Mesh(prim);points=np.array(mesh.GetPointsAttr().Get(),float);matrix=np.array(lazy.pxr.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default()))
   vertices=(np.c_[points,np.ones(len(points))]@matrix@np.linalg.inv(link_matrix))[:,:3]
   indices=np.array(mesh.GetFaceVertexIndicesAttr().Get(),int);counts=np.array(mesh.GetFaceVertexCountsAttr().Get(),int);faces=[];offset=0
   for count in counts:
    polygon=indices[offset:offset+count];offset+=count
    for k in range(1,count-1):faces.append([polygon[0],polygon[k],polygon[k+1]])
   triangles=vertices[np.array(faces)];areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
   chosen=rng.choice(len(triangles),12000,p=areas/areas.sum());uv=rng.random((len(chosen),2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
   sample=triangles[chosen,0]+uv[:,0,None]*(triangles[chosen,1]-triangles[chosen,0])+uv[:,1,None]*(triangles[chosen,2]-triangles[chosen,0]);cloud.append(sample)
  xyz=np.concatenate(cloud);canonical=str(name);np.savez_compressed(out/(canonical+'.npz'),xyz=xyz,bounds=[xyz.min(0),xyz.max(0)])
  root_from_link=np.eye(4);root_from_link[:3,:3]=rr.T@lr;root_from_link[:3,3]=rr.T@(lp-rp)
  parts[canonical]=dict(root_from_link_at_training_q=root_from_link.tolist(),points=len(xyz))
 def value(v):
  if hasattr(v,'GetReal'):return list(v.GetImaginary())+[float(v.GetReal())]
  if isinstance(v,(str,int,float,bool)) or v is None:return v
  return list(v)
 joints={}
 for name,joint in obj.joints.items():
  attrs={}
  for field in ['physics:localPos0','physics:localPos1','physics:localRot0','physics:localRot1','physics:axis','physics:lowerLimit','physics:upperLimit']:
   attr=joint.prim.GetAttribute(field)
   if attr and attr.HasAuthoredValue():attrs[field]=value(attr.Get())
  joints[str(name)]=dict(attributes=attrs,training_qpos=array(joint.get_state()[0]).tolist())
 (out/'articulation.json').write_text(json.dumps(dict(source='canonical training template before public reset; no world poses',category=obj.category,model=obj.model,parts=parts,joints=joints),indent=2))
 print('ARTICULATION_PRIOR_EXPORTED',list(parts),list(joints),flush=True)
