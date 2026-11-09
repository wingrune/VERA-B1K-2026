"""Task56 observed-vase memory and dense motion validation.

Inputs are controller RGB-D fits, proprioception and canonical geometry only.
This task-owned overlay preserves the archived servo and original live guard.
It makes no placement claim and never reads evaluator or simulator state.
"""
import json
import math
import numpy as np
from scipy.spatial import cKDTree


def motion_samples(a, b, resolution=.01):
    return np.linspace(0., 1., max(2, int(np.ceil(np.max(abs(b-a))/resolution))+1))


def base_increment(action, alpha):
    angle=float(action[2])*alpha/30.;c,s=math.cos(angle),math.sin(angle)
    result=np.eye(4);result[:2,:2]=[[c,-s],[s,c]]
    result[:2,3]=np.asarray(action[:2])*alpha/30.
    return result


class ObservedVaseMemory:
    def __init__(self):
        self.seen=None;self.pose=None;self.tree=None;self.last_step=None
        self.last_issued=None;self.calls=0;self.rejected=0
        self.payload_samples=None
        self.history_unknown=False;self.in_act=False;self.successful_controls=0

    def update(self, p):
        seen=p.seen.get('vase')
        if seen is None or 'vase' not in p.poses:return
        pose=np.asarray(p.poses['vase'])
        if seen==self.seen and self.pose is not None and np.array_equal(pose,self.pose):return
        if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('Nonfinite observed vase pose')
        self.pose=pose.copy();self.seen=seen
        model=p.perception.models['vase']['xyz']
        self.tree=cKDTree(model@pose[:3,:3].T+pose[:3,3])

    def active(self,p):
        # Intended contact/insertion retains its original checks. Egress has
        # its own payload corridor; this memory additionally protects robot links.
        return self.tree is not None and p.child=='rose' and p.phase in (
            'search','navigate','grasp_plan','pregrasp','descend','close','lift',
            'carry','search_target','target_navigate','vertical_retreat','release_retract')

    def margins(self,p,q,base):
        centers=p.robot.centers(p.servo.kin.forward(q)[0])
        world=centers@base[:3,:3].T+base[:3,3]
        return self.tree.query(world)[0]-p.robot.radii

    def validate(self,p,action,previous):
        if not self.active(p):return True,dict(reason='outside_memory_motion_scope')
        q=p.current_q;base=p.base();ids=p.servo.indices
        right=[p.servo.names.index(f'right_arm_joint{i}') for i in range(1,8)]
        target=q.copy();target[ids]=action[3:14];target[right]=action[15:22]
        prior=q.copy()
        if previous is not None:prior[ids]=previous[3:14];prior[right]=previous[15:22]
        # 8mm original clearance plus5mm observed-pose uncertainty. Existing
        # proximity permits non-worsening escape, never a6mm inward allowance.
        protect_payload=p.held is not None and p.phase in ('carry','search_target','target_navigate')
        payload_required=None
        if protect_payload:
            if self.payload_samples is None:
                lo,hi=p.perception.models['rose']['bounds']
                axes=[np.linspace(a,b,max(2,int(np.ceil((b-a)/.025))+1)) for a,b in zip(lo,hi)]
                self.payload_samples=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        worst=float('inf');checked=0
        for start in (q,prior):
            # Each measured/issued envelope must preserve its own initial clearance.
            required=np.minimum(.013,self.margins(p,start,base))
            if protect_payload:
                pose=base@p.servo.pose(start)@p.held
                distance=float(self.tree.query(self.payload_samples@pose[:3,:3].T+pose[:3,3])[0].min())
                payload_required=min(.045,distance)
            count=max(len(motion_samples(start,target)),
                      int(np.ceil(np.linalg.norm(action[:2])/30./.002))+1,
                      int(np.ceil(abs(float(action[2]))/30./.005))+1,3)
            for alpha in np.linspace(0.,1.,count):
                tq=start+alpha*(target-start)
                margins=self.margins(p,tq,base@base_increment(action,alpha))
                worst=min(worst,float(margins.min()));checked+=1
                if np.any(margins<required-1e-6):
                    link=int(np.argmin(margins-required))
                    return False,dict(reason='remembered_vase_sweep_rejected',link=str(p.robot.names[link]),
                        margin_m=float(margins[link]),required_m=float(required[link]),alpha=float(alpha),samples=checked,
                        previous_envelope=bool(start is prior),seen=int(self.seen))
                if protect_payload:
                    pose=base@base_increment(action,alpha)@p.servo.pose(tq)@p.held
                    distance=float(self.tree.query(self.payload_samples@pose[:3,:3].T+pose[:3,3])[0].min())
                    if distance<payload_required-1e-6:
                        return False,dict(reason='remembered_vase_payload_sweep_rejected',distance_m=distance,
                            required_m=payload_required,alpha=float(alpha),samples=checked,seen=int(self.seen))
        return True,dict(reason='remembered_vase_sweep_accepted',minimum_margin_m=worst,samples=checked,seen=int(self.seen))


def dense_joint_path(servo,robot,q,target,cams,obj,bounds,seconds=4.,locked=(),payload_local=None,scene_override=None):
    """Original bidirectional search strategy with10mrad validated edges.

    Fixed iteration budget follows the existing no-search-wall-time policy.
    The original scene predicate and servo are retained. A path is a planning
    result; original servo/live guard still validate its physical execution.
    """
    from scene_collision import ObservedSceneCollision
    scene=scene_override if scene_override is not None else ObservedSceneCollision(cams,robot,servo.kin.forward(q)[0],obj,*bounds)
    ids=servo.indices;start=q[ids].copy();target=np.asarray(target).copy();rng=np.random.default_rng(481)
    def valid(x):
        trial=q.copy();trial[ids]=x
        payload=None if payload_local is None else servo.pose(trial)@payload_local
        return scene.valid(servo.kin.forward(trial)[0],payload)
    def edge(a,b):
        return all(valid(a+alpha*(b-a)) for alpha in motion_samples(a,b))
    if not valid(target):return None
    if edge(start,target):return [target.copy()]
    trees=[([start],[-1]),([target.copy()],[-1])]
    def extend(tree,sample):
        nodes,parents=tree;j=int(np.argmin([np.linalg.norm(x-sample) for x in nodes]))
        delta=sample-nodes[j];distance=np.linalg.norm(delta)
        new=nodes[j]+delta*min(1.,.3/max(distance,1e-8))
        if not edge(nodes[j],new):return None,False
        nodes.append(new);parents.append(j);return len(nodes)-1,distance<=.3
    def chain(tree,index):
        out=[]
        while index>=0:out.append(tree[0][index]);index=tree[1][index]
        return out[::-1]
    for iteration in range(1000):
        side=iteration%2
        sample=trees[1-side][0][-1] if rng.random()<.4 else np.clip(
            start*rng.random()+target*(1-rng.random())+rng.normal(0,.5,len(ids)),servo.limits[0]+1e-4,servo.limits[1]-1e-4)
        sample=sample.copy();sample[list(locked)]=start[list(locked)]
        a,_=extend(trees[side],sample)
        if a is None:continue
        for _ in range(25):
            b,joined=extend(trees[1-side],trees[side][0][a])
            if b is None:break
            if joined:
                left,right=chain(trees[side],a),chain(trees[1-side],b)
                path=left+right[::-1] if side==0 else right+left[::-1]
                short=[path[0]];i=0
                while i<len(path)-1:
                    j=len(path)-1
                    while j>i+1 and not edge(path[i],path[j]):j-=1
                    if not edge(path[i],path[j]):return None
                    short.append(path[j]);i=j
                return short[1:]
    return None


def bounded_escape_candidates(p, hold):
    """Short original-IK proposals; callers must validate every issued action."""
    if p.child!='rose' or p.paired.active or p.dual.active or 'vase' not in p.poses:
        return
    if p.held is None:
        if p.phase not in ('pregrasp','release_retract') or p.dual.right_held is not None:return
    elif p.phase not in ('carry','search_target','target_navigate'):
        return
    q=p.current_q;hand=p.servo.pose(q);base=p.base()
    away=(base@hand)[:3,3]-p.poses['vase'][:3,3];away[2]=0.
    norm=np.linalg.norm(away)
    if norm<1e-8:return
    radial=base[:3,:3].T@(away/norm)
    up=base[:3,:3].T@np.array([0.,0.,1.])
    saved_target=p.servo.target
    try:
        for distance in (.002,.005,.01,.02,.04,.05):
            for name,vector in (('radial',radial),('up',up),('up_radial',(radial+up)/np.sqrt(2.))):
                action=hold.copy();p.servo.target=None
                p.servo.step(action,q,hand[:3,3]+distance*vector,hand[:3,:3],plan=False)
                yield name,distance,action
    finally:
        p.servo.target=saved_target


def install(policy_class,egress_action_check=None):
    """Explicit task-owned binding; both original final guard call sites use it."""
    original_guard=policy_class.check_live_action;original_reset=policy_class.reset
    def reset(p):
        original_reset(p);p.observed_egress_memory=ObservedVaseMemory()
        if hasattr(p,'release_debug'):p.release_debug.pop('blocked_pregrasp_retry',None)
    def checked(p,action,proprio):
        if not hasattr(p,'observed_egress_memory'):p.observed_egress_memory=ObservedVaseMemory()
        memory=p.observed_egress_memory;memory.update(p)
        # History is committed only by the outer Planner.act boundary.
        # Shared forward has already overwritten p.action with a current proposal.
        if memory.history_unknown or not memory.in_act:
            raise RuntimeError('Observed-vase guard: issued-command history unavailable')
        original=original_guard(p,action.copy(),proprio)
        safe,receipt=memory.validate(p,original,memory.last_issued)
        receipt=dict(receipt,previous_source="validated_Planner_act_return",
                     previous_successful_control=memory.successful_controls or None,
                     previous_action23=None if memory.last_issued is None else memory.last_issued.tolist())
        egress=(p.phase=='vertical_retreat' and p.retreat_state is not None
                and p.retreat_state.get('exit') is not None and p.held is not None)
        if safe and egress and egress_action_check is not None:
            safe,egress_receipt=egress_action_check(p,p.retreat_state,original,memory.last_issued)
            receipt=dict(receipt,egress_post_guard=egress_receipt)
        result=original;memory.calls+=1
        retreat=p.release_debug.get('blocked_pregrasp_retry',{}).get('active',False) and p.phase=='pregrasp'
        if not safe or retreat:
            memory.rejected+=1
            hold=original.copy();hold[:3]=0.;hold[3:7]=proprio[53:57]
            hold[7:14]=proprio[3:10];hold[15:22]=proprio[28:35]
            result=original_guard(p,hold,proprio)
            hold_safe,hold_receipt=memory.validate(p,result,memory.last_issued)
            if hold_safe and egress and egress_action_check is not None:
                hold_safe,egress_receipt=egress_action_check(p,p.retreat_state,result,memory.last_issued)
                hold_receipt=dict(hold_receipt,egress_post_guard=egress_receipt)
            receipt=dict(receipt,fallback=hold_receipt,fallback_safe=bool(hold_safe))
            if not hold_safe or retreat:
                attempts=[];candidates=bounded_escape_candidates(p,hold)
                try:
                    for name,distance,candidate in candidates:
                        candidate=candidate.astype(np.float32)
                        guarded=original_guard(p,candidate.copy(),proprio)
                        unchanged=np.array_equal(candidate,guarded)
                        candidate_safe,candidate_receipt=memory.validate(p,guarded,memory.last_issued)
                        if candidate_safe and egress and egress_action_check is not None:
                            candidate_safe,egress_receipt=egress_action_check(p,p.retreat_state,guarded,memory.last_issued)
                            candidate_receipt=dict(candidate_receipt,egress_post_guard=egress_receipt)
                        attempts.append(dict(direction=name,distance_m=distance,
                            original_guard_unchanged=bool(unchanged),safe=bool(candidate_safe and unchanged),
                            receipt=candidate_receipt))
                        if candidate_safe and unchanged:
                            result=guarded;hold_safe=True;break
                finally:
                    candidates.close()
                receipt=dict(receipt,bounded_escape_attempts=attempts,bounded_escape_safe=bool(hold_safe))
            if not hold_safe:
                # Do not pretend an unchecked fallback is a safe recovery.
                with (p.output/'observed_egress_guard.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=int(p.step),phase=p.phase,error='no_verified_fallback',receipt=receipt))+'\n')
                raise RuntimeError('Observed-vase guard: no verified fallback command')
        if not safe or retreat or (memory.active(p) and p.step%30==0):
            with (p.output/'observed_egress_guard.jsonl').open('a') as f:
                f.write(json.dumps(dict(step=int(p.step),phase=p.phase,accepted=bool(safe),receipt=receipt,
                    original_guard_action=original.tolist(),final_action=result.tolist(),rejections=memory.rejected))+'\n')
        return result
    policy_class.reset=reset;policy_class.check_live_action=checked
    import reach_planner,observation_policy
    reach_planner.joint_path=dense_joint_path;observation_policy.joint_path=dense_joint_path
    assert reach_planner.joint_path is observation_policy.joint_path is dense_joint_path


def run_with_command_history(p, act):
    """Commit only the final validated action; poison unknown harness fallbacks."""
    if not hasattr(p,'observed_egress_memory'):
        p.observed_egress_memory=ObservedVaseMemory()
    memory=p.observed_egress_memory
    if memory.history_unknown or memory.in_act:
        memory.history_unknown=True
        raise RuntimeError('Observed-vase guard: issued-command history unavailable until reset')
    memory.in_act=True
    try:
        result=act()
        # Same public output contract as FrozenPlanner, repeated at commit boundary.
        if result.shape!=(1,23) or result.dtype!=np.float32 or not np.isfinite(result).all():
            raise ValueError('Cannot commit invalid issued command')
        with (p.output/'issued_command_history.jsonl').open('a') as f:
            f.write(json.dumps(dict(successful_control=memory.successful_controls+1,
                policy_step=int(p.step),action23=result[0].tolist(),dtype=str(result.dtype)))+'\n')
        memory.last_issued=result[0].copy()
        memory.successful_controls+=1
        return result
    except BaseException:
        memory.last_issued=None;memory.history_unknown=True
        raise
    finally:
        memory.in_act=False
