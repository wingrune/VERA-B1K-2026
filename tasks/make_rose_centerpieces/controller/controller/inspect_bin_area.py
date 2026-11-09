"""PRIVILEGED CAMERA DIAGNOSTIC ONLY. Never imported by online policies."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def array(value):
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def inspect_bin_area(evaluator,output):
    import torch
    import omnigibson as og
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    accessor=evaluator.instance_eval_states[0].env_accessor
    scope=accessor.object_scope
    bin_obj=next(obj for key,obj in scope.items() if 'ashcan' in key)
    fridges=[obj for obj in accessor.scene.objects if any(s in obj.category.lower() for s in ('fridge','refrigerator'))]
    if not fridges:raise RuntimeError('No loaded fridge object for the requested camera diagnostic')
    fridge=fridges[0]
    fridge_center=np.mean([array(x) for x in fridge.aabb],axis=0)
    food=[np.mean([array(x) for x in obj.aabb],axis=0) for key,obj in scope.items()
          if obj is not None and any(s in key for s in ('half__banana','half__pomegranate'))]
    counter=np.mean(food,axis=0)
    direction=counter[:2]-fridge_center[:2];direction/=max(np.linalg.norm(direction),1e-6)
    tangent=np.array([-direction[1],direction[0]])
    bin_lo,bin_hi=[array(x) for x in bin_obj.aabb];bin_center=(bin_lo+bin_hi)/2
    camera=accessor.robot.sensors[evaluator.robot_camera_names['head'].split('::')[1]]
    original=tuple(x.clone() for x in camera.get_position_orientation())
    camera.add_modality('seg_instance')
    records=[];images=[]
    try:
        for i,(distance,lateral) in enumerate([(0.8,0.),(1.1,0.),(.8,-.35),(.8,.35)]):
            eye=np.r_[fridge_center[:2]+direction*distance+tangent*lateral,bin_lo[2]+1.25]
            # The primary view surveys the fridge-side floor; a second view
            # aims at the native bin solely to distinguish occlusion from aim.
            area=np.r_[(fridge_center[:2]+counter[:2])/2,bin_lo[2]+.25]
            for label,target in [('area',area),('bin_audit',bin_center)]:
                back=eye-target;back/=np.linalg.norm(back)
                right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right)
                up=np.cross(back,right);rotation=np.column_stack([right,up,back])
                camera.set_position_orientation(position=torch.tensor(eye,dtype=torch.float32),
                    orientation=torch.tensor(R.from_matrix(rotation).as_quat(),dtype=torch.float32))
                for _ in range(6):og.sim.render()
                obs,info=camera.get_obs();rgb=array(obs['rgb'])[...,:3]
                segmentation=array(obs['seg_instance'])
                mapping=info.get('seg_instance',{})
                ids=[int(k) for k,v in mapping.items() if bin_obj.name in str(v)]
                mask=np.isin(segmentation,ids);pixels=int(mask.sum())
                stem=f'position_{i}_{label}'
                cv2.imwrite(str(output/(stem+'.png')),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                np.savez_compressed(output/(stem+'_depth.npz'),depth=array(obs['depth_linear']))
                annotated=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                if pixels:
                    yy,xx=np.where(mask);cv2.rectangle(annotated,(int(xx.min()),int(yy.min())),(int(xx.max()),int(yy.max())),(0,0,255),3)
                cv2.putText(annotated,f'{stem}: bin pixels={pixels}',(8,28),0,.7,(0,0,255),2)
                cv2.imwrite(str(output/(stem+'_annotated.jpg')),annotated)
                images.append(cv2.resize(annotated,(480,480)))
                row=dict(position_index=i,view=label,eye_world=eye.tolist(),target_world=target.tolist(),
                         image=str(output/(stem+'.png')),bin_visible_pixels=pixels,
                         segmentation_ids=ids,segmentation_mapping={str(k):str(v) for k,v in mapping.items()})
                records.append(row);print('BIN_CAMERA_DIAGNOSTIC',stem,pixels,flush=True)
    finally:
        camera.set_position_orientation(position=original[0],orientation=original[1])
    cv2.imwrite(str(output/'views.jpg'),np.concatenate([np.concatenate(images[:4],axis=1),np.concatenate(images[4:],axis=1)],axis=0))
    result=dict(diagnostic_only=True,policy_access=False,scored=False,robot_not_teleported=True,
                camera_target_basis='native fridge/counter geometry for explicit offline visibility inspection',
                bin_name=bin_obj.name,fridge_name=fridge.name,views=records,
                visible_in_primary_area_view=any(r['bin_visible_pixels']>0 for r in records if r['view']=='area'))
    (output/'result.json').write_text(json.dumps(result,indent=2))
