import numpy as np
from scipy.spatial.transform import Rotation as R
from .skills import transform,rot2
from .grasp_regions import supported_aperture_minimum
def update_pickup(policy,s):
 debug=dict(getattr(policy,'release_debug',None) or {})
 debug['insertion_previous_issued']=dict(control_step=int(policy.step)+1,
  action=None if policy.action is None else policy.action.copy().tolist())
 debug['pending_vase_capture']=None if policy.child!='vase' or policy.phase!='lift' or policy.held is None else dict(control_step=int(policy.step)+1,held=policy.held.copy().tolist())
 policy.release_debug=debug
 joints=policy.current_q[policy.servo.indices].copy();policy.motion_history.append(joints);policy.motion_history=policy.motion_history[-12:]
 hand=policy.servo.pose(policy.current_q)
 if policy.child=='vase' and policy.held is not None:
  policy.perception.vase_predicted=hand@policy.held
 policy.hand_history.append(hand.copy());policy.hand_history=policy.hand_history[-12:]
 stationary=len(policy.hand_history)==12 and np.max(np.ptp(np.array([h[:3,3] for h in policy.hand_history]),axis=0))<.012 and max(R.from_matrix(policy.hand_history[0][:3,:3].T@h[:3,:3]).magnitude() for h in policy.hand_history)<.12 and np.linalg.norm(s[:3])<.015
 policy.stationary_taskspace=stationary
 if policy.phase=='close':
  gap=float(s[24:26].sum());policy.close_history.append(gap);policy.close_history=policy.close_history[-8:]
  width=policy.grasp.get('nominal_width',.03)
  policy.close_ready=len(policy.close_history)==8 and np.ptp(policy.close_history)<.001 and gap<max(.015,width+.01)
 else:policy.close_ready=False;policy.close_history=[]
 policy.pickup_ready=False
 if policy.phase in ('lift','pickup_reacquire') and stationary:
  fresh=policy.step-policy.seen.get(policy.child,-1000)<25
  pose=policy.poses.get(policy.child)
  rose_risen=fresh and pose is not None and policy.center(policy.child)[2]-policy.grasp['initial_height']>.080 and float(s[24:26].sum())>supported_aperture_minimum(policy.grasp.get('nominal_width',.03))
  if rose_risen:
   relative=np.linalg.inv(policy.base()@hand)@pose
   policy.relative_history.append(relative.copy());policy.relative_history=policy.relative_history[-12:]
   consistent=len(policy.relative_history)==12 and np.max(np.ptp(np.array([h[:3,3] for h in policy.relative_history]),axis=0))<.025 and max(R.from_matrix(policy.relative_history[0][:3,:3].T@h[:3,:3]).magnitude() for h in policy.relative_history)<.18
   seen_step=policy.seen[policy.child]
   if seen_step not in policy.pickup_observation_steps:policy.pickup_observation_steps.append(seen_step)
   policy.pickup_confirmations=policy.pickup_confirmations+1 if consistent else 0
  else:policy.pickup_confirmations=0;policy.relative_history=[];policy.pickup_observation_steps=[]
  policy.pickup_ready=policy.pickup_confirmations>=18 and len(policy.pickup_observation_steps)>=3
 else:policy.pickup_confirmations=0;policy.relative_history=[];policy.pickup_observation_steps=[]
 previous=policy.action
 zero=previous is not None and bool(np.all(np.asarray(previous[:3])==0.))
 removed=np.zeros(2)
 if zero:
  removed=rot2(float(policy.odom[2]))@s[:2]/30
  policy.odom[:2]-=removed
 policy.release_debug=dict(policy.release_debug,zero_command_xy=dict(
  mode='previous_zero_base_xy_precompensation' if zero else 'original_xy',
  control_step=int(policy.step)+1,
  previous_base_action=None if previous is None else previous[:3].tolist(),
  removed_xy=removed.tolist()))
 return stationary

def prepare_reacquire(policy):
 if (policy.phase=='lift' and policy.held is not None and not policy.pickup_ready
   and ((policy.child=='rose' and policy.age>=170) or (policy.child=='vase' and policy.age>=140
  ))):
  policy.pickup_reacquire_command=policy.action.copy()
  policy.pickup_reacquire_started=policy.step
  policy.pickup_reacquire_failed=False
  policy.pickup_view_state=None
  policy.change('pickup_reacquire')

def finish_reacquire(p,action,s):
 debug=dict(getattr(p,'release_debug',None) or {})
 history=[] if p.step<=1 else debug.get('allowed_chain_history',[])
 history.append(dict(step=int(p.step),phase=p.phase,
  q=p.current_q.tolist(),estimated_base=p.base().tolist(),base_velocity=s[:3].tolist(),
  held=None if p.held is None else p.held.tolist(),
  estimated_poses={name:p.poses[name].tolist() for name in ('rose','vase','coffee_table') if name in p.poses},
  seen={name:int(p.seen[name]) for name in ('rose','vase','coffee_table') if name in p.seen},
  cameras={name:{key:p.current_cameras[name][key].tolist() for key in ('R','t','K')} for name in ('head','left_wrist','right_wrist') if name in p.current_cameras}))
 p.release_debug=dict(debug,allowed_chain_history=history[-60:])
 pending=debug.get('pending_vase_capture')
 if p.child=='vase' and p.phase=='lift' and pending is not None and pending['control_step']==p.step:
  p.held=np.asarray(pending['held']).copy()
 action=guard_vase_camera_approach(p,action,s)
 action=measured_torso_insertion(p,action,s)
 from .insertion import recover_blocked_pregrasp
 action=recover_blocked_pregrasp(p,action,s)
 if (p.phase=='lift' and p.child=='rose'
   and p.held is not None and not p.left_pickup_confirmed
   and p.seen.get('rose',-1000)==p.step and float(s[24:26].sum())>supported_aperture_minimum(p.grasp.get('nominal_width',.03))):
  observed=p.poses['rose'];world_hand=p.base()@p.servo.pose(p.current_q)
  risen=p.center('rose')[2]-p.grasp['initial_height']>.055
  if risen and np.linalg.norm(p.center('rose')-world_hand[:3,3])<.20:
   previous=p.held.copy();p.held=np.linalg.inv(world_hand)@observed
   p.unconfirmed_refit_debug=dict(step=p.step,source='newly_accepted_rgbd_fit',position_delta_m=float(np.linalg.norm(p.held[:3,3]-previous[:3,3])),rotation_delta_rad=float(R.from_matrix(previous[:3,:3].T@p.held[:3,:3]).magnitude()),confirmation_not_changed=True)
 if p.pickup_reacquire_command is not None:
  if p.phase=='pickup_reacquire' and p.pickup_ready and (p.child!='vase' or np.linalg.norm(p.center('vase')-(p.base()@p.servo.pose(p.current_q))[:3,3])<.20):
   p.held=np.linalg.inv(p.base()@p.servo.pose(p.current_q))@p.poses[p.child]
   p.left_pickup_confirmed=True;p.pickup_reacquire_command=None;p.pickup_view_state=None
   p.change('search_target' if p.child=='vase' else 'carry')
   if p.child=='vase':
    action[:3]=0.;action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35];action[14]=-1.
  else:
   action[:3]=0.;action[3:14]=p.pickup_reacquire_command[3:14]
   action[15:22]=p.pickup_reacquire_command[15:22];action[14]=-1.;action[22]=1.
   if p.phase=='pickup_reacquire':
    state=getattr(p,'pickup_view_state',None)
    if state is None:
     observed_rise='rose' in p.poses and p.center('rose')[2]-p.grasp['initial_height']>.080
     support=float(s[24:26].sum())>supported_aperture_minimum(p.grasp.get('nominal_width',.03))
     # A stale fit cannot show the lift. Prediction requests a view only;
     # pickup_ready still requires fresh fits and original confirmations.
     predicted_rise=False
     if p.child=='rose' and p.held is not None:
      predicted=p.base()@p.servo.pose(p.current_q)@p.held
      predicted_rise=float(predicted[2,3])-p.grasp['initial_height']>.080
     state=plan_vase_camera_view(p) if p.child=='vase' and support else plan_pickup_color_view(p) if (observed_rise or predicted_rise) and support else dict(started=p.step,settled=None)
     p.pickup_view_state=state
    if 'right_target' in state:
     ids=(11,13,15,17,19,21,23);measured=p.current_q[list(ids)]
     previous=np.clip(p.pickup_reacquire_command[15:22],measured-.12,measured+.12)
     delta=state['right_target']-previous
     proposed=previous+delta*min(1.,.024/max(float(np.max(abs(delta))),1e-9))
     trial=p.current_q.copy();trial[list(ids)]=proposed
     safe=all(state['scene'].valid(p.servo.kin.forward(p.current_q+a*(trial-p.current_q))[0],p.servo.pose(p.current_q+a*(trial-p.current_q))@p.held) for a in (.33,.67,1.))
     action[15:22]=proposed if safe else measured
     if np.max(abs(state['right_target']-measured))<.035 and state['settled'] is None:state['settled']=p.step
     if p.step-state['started']<=300 and (state['settled'] is None or p.step-state['settled']<=90):return action
    if 'target' in state:
     measured=p.current_q[p.servo.indices]
     previous=np.clip(p.pickup_reacquire_command[3:14],measured-.12,measured+.12)
     delta=state['target']-previous
     proposed=previous+delta*min(1.,.024/max(float(np.max(abs(delta))),1e-9))
     trial=p.current_q.copy();trial[p.servo.indices]=proposed
     safe=all(state['scene'].valid(p.servo.kin.forward(p.current_q+a*(trial-p.current_q))[0],p.servo.pose(p.current_q+a*(trial-p.current_q))@p.held) for a in (.33,.67,1.))
     action[3:14]=proposed if safe else measured
     action[15:22]=s[28:35]
     if np.max(abs(state['target']-measured))<.035 and state['settled'] is None:state['settled']=p.step
     expired=p.step-state['started']>300 or (state['settled'] is not None and p.step-state['settled']>90)
     if not expired:return action
    elif p.step-p.pickup_reacquire_started<=90:return action
    action[:3]=0.;action[3:7]=s[53:57];action[7:14]=s[3:10];action[15:22]=s[28:35]
    p.pickup_reacquire_command=action.copy();p.pickup_view_state=None
    p.pickup_reacquire_failed=True;p.change('done_hold')
 return action

def measured_torso_insertion(p,action,s):
 """Observed insertion fallback preserving stalled joints' motor support."""
 debug=p.release_debug
 state=debug.get('measured_torso_insertion')
 eligible=(p.phase=='execute' and p.child=='rose' and p.parent=='vase'
      and p.held is not None and 0<=p.waypoint<len(p.waypoints)
      and not p.dual.active and not p.paired.active
      and p.dual.right_held is None and p.pickup_reacquire_command is None)
 if not eligible:
  if state is not None:
   state['active']=False;state['samples']=[];state['status']='outside_execute'
  return action
 prior=debug.get('insertion_previous_issued')
 if (prior is None or prior.get('control_step')!=p.step
   or prior.get('action') is None):return action
 previous=np.asarray(prior['action'])
 q=p.current_q;ids=p.servo.indices
 world=p.waypoints[p.waypoint][0];held=p.held
 if (previous.shape!=(23,) or not np.isfinite(q).all()
   or not np.isfinite(world).all() or not np.isfinite(held).all()
   or not np.isfinite(previous).all() or not np.isfinite(action).all()):
  return action
 same=(state is not None and state.get('waypoint')==p.waypoint
    and np.array_equal(np.asarray(state['world_goal']),world)
    and np.array_equal(np.asarray(state['held']),held))
 if not same:
  state=dict(waypoint=int(p.waypoint),world_goal=world.tolist(),held=held.tolist(),
       active=False,samples=[],attempted=0,last_solve_step=None,
       target=None,status='collecting_stationary_inputs')
  debug['measured_torso_insertion']=state
 if state.get('last_step') is not None and p.step!=state['last_step']+1:
  state['samples']=[]
 state['last_step']=int(p.step)
 goal=np.linalg.inv(p.base())@world@np.linalg.inv(held)
 hand=p.servo.pose(q)
 error=float(np.linalg.norm(hand[:3,3]-goal[:3,3]))
 angle=float(R.from_matrix(goal[:3,:3]@hand[:3,:3].T).magnitude())
 support=previous[3:7].copy()
 samples=state['samples']
 samples.append(dict(step=int(p.step),hand=hand.tolist(),error_m=error,
           torso=q[ids[:4]].tolist(),torso_command=support.tolist(),
           base_velocity=s[:3].tolist()))
 samples=samples[-60:];state['samples']=samples
 residual=float(np.max(abs(support-q[ids[:4]])))
 state['error_m']=error;state['angle_rad']=angle
 state['torso_residual_rad']=residual
 if not state['active']:
  if len(samples)<60:return action
  hands=[np.asarray(row['hand']) for row in samples]
  position_spread=float(np.max(np.ptp(np.array([h[:3,3] for h in hands]),axis=0)))
  rotation_spread=max(float(R.from_matrix(hands[0][:3,:3].T@h[:3,:3]).magnitude()) for h in hands)
  torsos=np.array([row['torso'] for row in samples])
  commands=np.array([row['torso_command'] for row in samples])
  torso_spread=np.ptp(torsos,axis=0);command_spread=np.ptp(commands,axis=0)
  residuals=abs(commands-torsos)
  locked_indices=[i for i in range(4) if torso_spread[i]<.006 and command_spread[i]<.001
          and residuals[:,i].min()>.010 and residuals[:,i].max()<=.120]
  base_speed=max(float(np.linalg.norm(row['base_velocity'])) for row in samples)
  improvement=max(0.,float(samples[0]['error_m']-min(row['error_m'] for row in samples)))
  stationary=position_spread<.005 and rotation_spread<.01 and base_speed<.015
  state['window']=dict(controls=60,position_spread_m=position_spread,
            rotation_spread_rad=rotation_spread,torso_spread_rad=torso_spread.tolist(),
            torso_command_spread_rad=command_spread.tolist(),base_speed=base_speed,
            error_improvement_m=improvement)
  if not (stationary and improvement<.003 and error>=.026 and locked_indices):
   state['status']='stationary_trigger_not_met';return action
  state['support_torso']=support.tolist();state['locked_indices']=locked_indices
 else:
  support=np.asarray(state['support_torso'])
  locked_indices=state['locked_indices']
  if max(abs(support[i]-q[ids[i]]) for i in locked_indices)>.120:
   state['status']='support_residual_outside_bound';return action
 locked_ids=[ids[i] for i in locked_indices]

 from .skills import ObservedSceneCollision
 poses=p.servo.kin.forward(q)[0];baseline=p.robot.margins(poses)
 scene=ObservedSceneCollision(p.current_cameras,p.robot,poses,hand@held,*p.bounds('rose'))
 right_ids=(11,13,15,17,19,21,23)
 def geometry_valid(trial):
  tf=p.servo.kin.forward(trial)[0]
  return p.robot.valid(tf,baseline) and scene.valid(tf,p.servo.pose(trial)@held)
 def both_envelopes_valid(trial):
  if not geometry_valid(trial):return False
  issued=trial.copy();issued[locked_ids]=support[locked_indices];issued[list(right_ids)]=action[15:22]
  return geometry_valid(issued)
 def endpoint_error(trial):
  pose=p.servo.pose(trial)
  return (float(np.linalg.norm(pose[:3,3]-goal[:3,3])),
      float(R.from_matrix(goal[:3,:3]@pose[:3,:3].T).magnitude()))
 target=None if state['target'] is None else np.asarray(state['target'])
 physical=q.copy()
 if target is not None:physical[ids]=target;physical[locked_ids]=q[locked_ids]
 predicted=endpoint_error(physical)
 needs_solve=(target is None or predicted[0]>.010 or predicted[1]>.10
       or np.max(abs(q[locked_ids]-np.asarray(state['actual_torso'])[locked_indices]))>.005
       or not both_envelopes_valid(physical))
 admission=None
 if needs_solve:
  if state['attempted']>=3:
   state['status']='same_goal_attempt_limit';return action
  if state['last_solve_step'] is not None and p.step-state['last_solve_step']<60:
   state['status']='same_goal_solve_cooldown';return action
  state['attempted']+=1;state['last_solve_step']=int(p.step)
  receipts=dict(candidate_endpoints=0,fk_rejected=0,envelope_rejected=0)
  def valid(trial):
   receipts['candidate_endpoints']+=1
   e,a=endpoint_error(trial)
   if e>.010 or a>.10:
    receipts['fk_rejected']+=1;return False
   if not both_envelopes_valid(trial):
    receipts['envelope_rejected']+=1;return False
   return True
  locked={i:float(q[ids[i]]) for i in locked_indices}
  solution=p.servo.solve(q,goal[:3,3],goal[:3,:3],locked=locked,validator=valid)
  state['solver_receipt']=receipts
  if solution is None:
   state['status']='locked_ik_or_endpoint_rejected';return action
  physical=q.copy();physical[ids]=solution;physical[locked_ids]=q[locked_ids]
  e,a=endpoint_error(physical)
  if e>.010 or a>.10:
   state['status']='strict_endpoint_rejected';return action
  if not all(both_envelopes_valid(q+alpha*(physical-q)) for alpha in np.linspace(0.,1.,101)):
   state['status']='full_sweep_rejected';return action
  target=solution.copy();target[locked_indices]=q[locked_ids]
  admission=(e,a)

 motor_target=target.copy();motor_target[locked_indices]=support[locked_indices]
 working=action.copy();working[3:14]=previous[3:14]
 original=(p.servo.target,p.servo.synchronized,p.servo.retain_target)
 p.servo.target=motor_target.copy();p.servo.synchronized=True
 p.servo.retain_target=True
 try:p.servo.step(working,q,goal[:3,3],goal[:3,:3])
 finally:
  p.servo.target,p.servo.synchronized,p.servo.retain_target=original
 for i in locked_indices:working[3+i]=support[i]
 if np.max(abs(working[3:14]-previous[3:14]))>.0240001:
  state['status']='previous_issued_slew_rejected';return action
 command=q.copy();command[ids]=working[3:14];command[locked_ids]=q[locked_ids]
 if not all(both_envelopes_valid(q+alpha*(command-q)) for alpha in (.33,.67,1.)):
  state['status']='live_command_envelope_rejected';return action
 issued=command.copy();issued[locked_ids]=support[locked_indices];issued[list(right_ids)]=working[15:22]
 if not all(geometry_valid(q+alpha*(issued-q)) for alpha in (.33,.67,1.)):
  state['status']='issued_command_sweep_rejected';return action
 p.servo.target=motor_target.copy();p.servo.synchronized=True
 state['target']=target.tolist();state['active']=True
 if admission is not None:
  state['actual_torso']=q[ids[:4]].tolist();state['admitted_step']=int(p.step)
  state['admitted_error_m']=admission[0];state['admitted_angle_rad']=admission[1]
  state['full_sweep_samples']=101
 state['status']='moving_same_goal';state['command_step']=int(p.step)
 state['command_arm']=working[7:14].tolist()
 return working

def pickup_color_support(p,q,camera):
 model=p.perception.models[p.child];n=8 if p.child=='rose' else 32;xyz=model['xyz'][::n];rgb=model['rgb'][::n].astype(float)
 red=(rgb[:,0]>70)&(rgb[:,0]>1.5*rgb[:,1])&(rgb[:,0]>1.5*rgb[:,2])
 green=(rgb[:,1]>1.15*rgb[:,0])&(rgb[:,1]>1.15*rgb[:,2])
 if p.child=='rose' and min(int(red.sum()),int(green.sum()))<5:return 0.
 payload=p.servo.pose(q)@p.held;points=xyz@payload[:3,:3].T+payload[:3,3]
 local=(points-camera['t'])@camera['R'];image=local@camera['K'].T;uv=image[:,:2]/np.maximum(local[:,2,None],1e-8)
 height,width=camera['rgb'].shape[:2]
 inside=(local[:,2]>.06)&(uv[:,0]>.05*width)&(uv[:,0]<.95*width)&(uv[:,1]>.05*height)&(uv[:,1]<.95*height)
 rays=points-camera['t'];length=np.linalg.norm(rays,axis=1);direction=rays/np.maximum(length[:,None],1e-8)
 ids=list(range(len(p.robot.names))) if p.child=='vase' else [i for i,name in enumerate(p.robot.names) if name.startswith(('left_arm','left_gripper','left_realsense','right_arm','right_gripper','right_realsense'))]
 delta=p.robot.centers(p.servo.kin.forward(q)[0])[ids]-camera['t'];along=direction@delta.T
 distance=np.linalg.norm(delta[None]-along[:,:,None]*direction[:,None],axis=2)
 blocked=((distance<p.robot.radii[ids][None]+.008)&(along>0)&(along<length[:,None])).any(axis=1)
 visible=inside&~blocked
 return float(visible.mean()) if p.child=='vase' else float(min((visible&red).sum()/red.sum(),(visible&green).sum()/green.sum()))

def plan_pickup_color_view(p):
 from .skills import ObservedSceneCollision
 from scipy.spatial import cKDTree
 state=dict(started=p.step,settled=None,kind='pickup_color_view')
 if p.held is None or 'head' not in p.current_cameras or p.dual.active or p.paired.active or p.dual.right_held is not None:return state
 q=p.current_q;hand=p.servo.pose(q);camera=p.current_cameras['head']
 scene=ObservedSceneCollision(p.current_cameras,p.robot,p.servo.kin.forward(q)[0],hand@p.held,*p.bounds('rose'))
 if 'vase' in p.poses:
  vase=np.linalg.inv(p.base())@p.poses['vase'];xyz=p.perception.models['vase']['xyz'][::12]
  memory=xyz@vase[:3,:3].T+vase[:3,3]
  scene.points=np.concatenate((scene.points,memory));scene.tree=cKDTree(scene.points)
  scene.scene_baseline=scene.sphere_margins(p.servo.kin.forward(q)[0])
 state['scene']=scene;current=pickup_color_support(p,q,camera);choices=[]
 for offset in ((.12,-.10,0.),(.15,0.,0.),(0.,-.12,0.),(0.,.12,0.),(-.08,.12,0.),(.08,-.12,.10),(0.,.12,.10)):
  solution=p.servo.solve(q,hand[:3,3]+offset,hand[:3,:3],locked={i:q[p.servo.indices[i]] for i in range(4)},validator=lambda trial:scene.valid(p.servo.kin.forward(trial)[0],p.servo.pose(trial)@p.held))
  if solution is None:continue
  target=q.copy();target[p.servo.indices]=solution
  path=[(p.servo.kin.forward(q+a*(target-q))[0],p.servo.pose(q+a*(target-q))) for a in np.linspace(0.,1.,101)]
  if not all(scene.valid(poses,hand@p.held) for poses,hand in path):continue
  robust=True
  for shift in ((.005,0.,0.),(-.005,0.,0.),(0.,.005,0.),(0.,-.005,0.),(0.,0.,.005),(0.,0.,-.005)):
   payload=p.held.copy();payload[:3,3]+=shift
   if not all(scene.valid(poses,hand@payload) for poses,hand in path):robust=False;break
  if not robust:continue
  gain=pickup_color_support(p,target,camera)-current
  if gain>.10:choices.append((gain,solution.copy(),offset))
 if choices:
  gain,target,offset=max(choices,key=lambda row:row[0]);state['target']=target;state['predicted_color_gain']=float(gain)
  p.release_debug=dict(getattr(p,'release_debug',None) or {},pickup_color_view=dict(started=p.step,offset=offset,predicted_color_gain=float(gain),safe_paths=len(choices),proof_unchanged=True))
 return state

def plan_vase_camera_view(p):
 from .skills import ObservedSceneCollision
 state=dict(started=p.step,settled=None)
 if p.held is None or 'right_wrist' not in p.current_cameras or p.dual.active or p.paired.active or p.dual.right_held is not None:return state
 q=p.current_q;ids=[11,13,15,17,19,21,23];camera=p.current_cameras['right_wrist']
 scene=ObservedSceneCollision(p.current_cameras,p.robot,p.servo.kin.forward(q)[0],p.servo.pose(q)@p.held,*p.bounds('vase'))
 mount=np.linalg.inv(p.servo.kin.forward(q)[0]['right_gripper_link'])@transform(camera['R'],camera['t']);choices=[]
 for shoulder in (.3,.5,.7):
  for w5 in (2.15,2.28):
   for w6 in (.85,.95):
    for w7 in (-1.4,):
     joints=np.array([shoulder,-.6,-.15,-.7,w5,w6,w7]);trial=q.copy();trial[ids]=joints
     cam=p.servo.kin.forward(trial)[0]['right_gripper_link']@mount
     score=pickup_color_support(p,trial,dict(camera,R=cam[:3,:3],t=cam[:3,3]))
     if score>.60:choices.append((score,joints))
 choices=sorted(choices,key=lambda x:-x[0]);admitted=[]
 for score,joints in choices[:4]:
  target=q.copy();target[ids]=joints
  path=[(p.servo.kin.forward(q+a*(target-q))[0],p.servo.pose(q+a*(target-q))) for a in np.linspace(0.,1.,101)]
  if not all(scene.valid(poses,hand@p.held) for poses,hand in path):continue
  robust=True
  for shift in ((.005,0.,0.),(-.005,0.,0.),(0.,.005,0.),(0.,-.005,0.),(0.,0.,.005),(0.,0.,-.005)):
   held=p.held.copy();held[:3,3]+=shift
   if not all(scene.valid(poses,hand@held) for poses,hand in path):robust=False;break
  if robust:admitted.append((score,joints))
  if len(admitted)>=2:break
 if admitted:
  state['scene']=scene;state['right_target']=admitted[0][1]
 p.release_debug=dict(getattr(p,'release_debug',None) or {},vase_camera_view=dict(started=p.step,admitted_paths=len(admitted),predicted_robot_clear=0. if not admitted else admitted[0][0],proof_unchanged=True))
 return state


def guard_vase_camera_approach(p,action,s):
 """Protect camera housings from the visually localized vase."""
 if (p.child!='vase' or p.held is not None or p.phase not in ('pregrasp','descend')
   or p.grasp is None or 'vase' not in p.poses):return action
 from scipy.spatial import cKDTree
 q=p.current_q;obj=np.linalg.inv(p.base())@p.poses['vase']
 xyz=p.perception.models['vase']['xyz']
 tree=cKDTree(xyz@obj[:3,:3].T+obj[:3,3])
 ids=[i for i,name in enumerate(p.robot.names) if name.startswith(('left_realsense','right_realsense'))]
 if not ids:return action
 def margins(trial):
  centers=p.robot.centers(p.servo.kin.forward(trial)[0])[ids]
  return tree.query(centers,k=1)[0]-p.robot.radii[ids]
 baseline=margins(q);required=np.minimum(.008,baseline)
 trial=q.copy();trial[p.servo.indices]=action[3:14]
 trial[[11,13,15,17,19,21,23]]=action[15:22]
 sweeps=[margins(q+a*(trial-q)) for a in (.33,.67,1.)]
 if p.phase=='pregrasp':
  endpoint=q.copy();endpoint[p.servo.indices]=p.grasp['solution']
  sweeps.extend(margins(q+a*(endpoint-q)) for a in np.linspace(0.,1.,101))
 safe=all(np.all(row>=required-1e-6) for row in sweeps)
 p.release_debug=dict(p.release_debug,vase_camera_approach=dict(step=p.step,
  baseline_margin_m=baseline.tolist(),issued_sweep_min_m=np.min(sweeps,axis=0).tolist(),
  required_margin_m=required.tolist(),safe=bool(safe),region=p.grasp.get('region')))
 if safe:return action
 region=p.grasp.get('region','bounding_box')
 p.grasp_trials[region]=p.grasp_trials.get(region,0)+1
 action[:3]=0.;action[3:14]=q[p.servo.indices];action[15:22]=s[28:35]
 action[14]=1.;p.change('grasp_plan')
 return action
