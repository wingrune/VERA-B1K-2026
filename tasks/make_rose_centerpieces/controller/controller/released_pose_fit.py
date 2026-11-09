"""Research released-only full-SO3 search with RGB-D parent/depth constraints."""
import time,itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from perception import pointcloud
from rose_geometry import _prepare_guided,_guided_rose_fit_single,FitTimeLimit,visible_model_support

def parent_supported(model,vase_model,pose,vase):
 lo,hi=model['bounds'];corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])@pose[:3,:3].T+pose[:3,3];center=(corners.min(0)+corners.max(0))/2;center=(center-vase[:3,3])@vase[:3,:3];vl,vh=vase_model['bounds'];radius=.32*min((vh-vl)[:2]);return bool(np.linalg.norm(center[:2]-(vl[:2]+vh[:2])/2)<radius and (vl[2]+vh[2])/2-.025<center[2]<vh[2]-.015)

def rotation_seeds():
 rotations=[]
 for permutation in itertools.permutations(range(3)):
  for signs in itertools.product((-1,1),repeat=3):
   matrix=np.eye(3)[:,permutation]*np.array(signs)[None,:]
   if np.linalg.det(matrix)>.5:
    rotations.append(matrix)
    for axis in np.eye(3):
     for angle in (-.35,.35):rotations.append(R.from_rotvec(axis*angle).as_matrix()@matrix)
 return rotations

def global_release_fit(model,cameras,predicted,vase_model,vase,seconds=3,max_candidates=12):
 started=time.monotonic()
 # Releasedflowers may settle undergravity afteropening. Static vaseheight
 # bounds total downward motion; observedred points proposeeachseedheight.
 # Keep the original75mm XY/upwards limits relative tolastacceptedRGB-Dpose.
 down_bound=float((vase_model['bounds'][1]-vase_model['bounds'][0])[2])*.5
 if not np.isfinite(down_bound) or down_bound<=0:return None,{'reason':'invalid_parent_height'}
 if seconds<=0:return None,{'reason':'deadline_before_search','wall_seconds':0.}
 seconds=min(float(seconds),3.);max_candidates=min(int(max_candidates),12);deadline=started+seconds;masked={k:dict(c,depth=c['depth'].copy()) for k,c in cameras.items()};vl,vh=vase_model['bounds']
 for c in masked.values():
  points,_,uv=pointcloud(c);local=(points-vase[:3,3])@vase[:3,:3];keep=((local>=vl-.15)&(local<=vh+.15)).all(axis=1);c['depth'][uv[~keep,1],uv[~keep,0]]=np.nan
 prepared=_prepare_guided(model,masked);mr,mg,clouds=prepared
 # Spatialassociation fromlastallowedflowerroot, notNativeobjectidentity.
 clouds['red']=[p[np.linalg.norm(p-predicted[:3,3],axis=1)<.13] for p in clouds['red']]
 green_filtered=[]
 for points in clouds['green']:
  local=(points-vase[:3,3])@vase[:3,:3];center=(vl[:2]+vh[:2])/2
  keep=(np.linalg.norm(local[:,:2]-center,axis=1)<.22)&(local[:,2]>vl[2]-.05)&(local[:,2]<vh[2]+.15)
  green_filtered.append(points[keep])
 clouds['green']=green_filtered
 ro=np.concatenate([p[::max(1,int(np.ceil(len(p)/400)))] for p in clouds['red'] if len(p)]) if any(len(p) for p in clouds['red']) else np.empty((0,3))
 go=np.concatenate([p[::max(1,int(np.ceil(len(p)/400)))] for p in clouds['green'] if len(p)]) if any(len(p) for p in clouds['green']) else np.empty((0,3))
 if min(len(ro),len(go))<10:return None,{'reason':'missing_color_points','red':len(ro),'green':len(go)}
 rt=cKDTree(mr,copy_data=True);gt=cKDTree(mg,copy_data=True);ranked=[]
 for rotation in rotation_seeds():
  if time.monotonic()>deadline:break
  pose=predicted.copy();pose[:3,:3]=rotation@pose[:3,:3]
  proposal=np.median(ro,axis=0)-pose[:3,:3]@np.median(mr,axis=0)-pose[:3,3]
  pose[2,3]+=float(np.clip(proposal[2],-down_bound,0.))
  delta=np.clip(np.median(ro,axis=0)-pose[:3,:3]@np.median(mr,axis=0)-pose[:3,3],-.071,.071);dr=rt.query((ro-pose[:3,3]-delta)@pose[:3,:3])[0];dg=gt.query((go-pose[:3,3]-delta)@pose[:3,:3])[0];score=float(np.quantile(np.r_[dr,dg],.65));ranked.append((score,pose,delta))
 candidates=[];rejected=[];attempts=0
 for _,seed,delta in sorted(ranked,key=lambda x:x[0])[:max_candidates]:
  if time.monotonic()>deadline:break
  attempts+=1
  try:fit=_guided_rose_fit_single(model,masked,seed,translation_limit=.075,red_min=.4,green_min=.5,visible_only=True,prepared=prepared,deadline=deadline,initial_delta=delta,rotation_limit=.60)
  except FitTimeLimit:break
  if fit is None:continue
  pose,stats=fit;shift_vector=pose[:3,3]-predicted[:3,3];shift=float(np.linalg.norm(shift_vector));shift_max=float(np.max(abs(shift_vector)));support=visible_model_support(model,cameras,pose);contained=parent_supported(model,vase_model,pose,vase)
  if np.max(abs(shift_vector[:2]))>.075 or shift_vector[2]<-down_bound or shift_vector[2]>.075 or not contained or support['matched_fraction']<.65 or support['unoccluded_fraction']<=.50:
   rejected.append({'root_shift':shift,'root_shift_max_component_m':shift_max,'contained':contained,'depth_support':support,'residual':stats['residual']});continue
  stats=dict(stats,method='released_parent_supported_global_so3',parent_supported=True,root_shift=shift,root_shift_l2_m=shift,root_shift_max_component_m=shift_max,root_shift_xyz_m=shift_vector.tolist(),downward_root_bound_m=down_bound,local_rotation_limit_rad=.60,observed_seed_downward_shift_m=float(predicted[2,3]-seed[2,3]),depth_support=support,total_rotation_from_previous_rad=float(R.from_matrix(predicted[:3,:3].T@pose[:3,:3]).magnitude()),released_depth_certified=True,rotation_seeds=168,optimized_candidates=attempts,deadline_seconds=seconds);score=stats['residual']-.005*(stats['red_coverage']+stats['green_coverage'])-.01*support['matched_fraction'];candidates.append((score,pose,stats))
 diagnostics={'downward_root_bound_m':down_bound,'local_rotation_limit_rad':.60,'sampling_rotation_seeds':168,'red_association_radius_m':.13,'green_parent_radius_m':.22,'deadline_seconds':seconds,'ranked':len(ranked),'optimized':attempts,'accepted':len(candidates),'rejections':rejected,'wall_seconds':time.monotonic()-started}
 if not candidates:return None,diagnostics
 _,pose,stats=min(candidates,key=lambda x:x[0]);stats.update(fit_seconds=time.monotonic()-started,optimized_candidates=attempts,ranked_candidates=len(ranked));return (pose,stats),diagnostics


def certify_released_fit(model,cameras,predicted,vase_model,vase,fit):
 """Certify legacy/global candidates using only RGB-D/static canonical geometry."""
 if fit is None:return None,{'reason':'no_candidate'}
 pose,stats=fit;shift_vector=pose[:3,3]-predicted[:3,3];root_shift=float(np.linalg.norm(shift_vector));shift_max=float(np.max(abs(shift_vector)));support=visible_model_support(model,cameras,pose);contained=parent_supported(model,vase_model,pose,vase)
 green_min=.50 if stats.get('correspondence_direction')=='observed_to_model' else .35
 color_ok=stats.get('red_coverage',0)>=.4 and stats.get('green_coverage',0)>=green_min and stats.get('residual',1)<=.018
 if shift_max>.075 or not contained or support['matched_fraction']<.65 or support['unoccluded_fraction']<=.50 or not color_ok:
  return None,{'reason':'released_geometry_or_depth_rejected','root_shift_l2_m':root_shift,'root_shift_max_component_m':shift_max,'parent_supported':contained,'depth_support':support,'color_ok':color_ok}
 return (pose,dict(stats,released_depth_certified=True,parent_supported=True,root_shift=root_shift,root_shift_l2_m=root_shift,root_shift_max_component_m=shift_max,depth_support=support)),{'reason':'released_certified'}
