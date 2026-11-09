"""Observation-compatible manipulation skills. No simulator imports or state queries.

Map and object poses are estimates supplied by the caller. Development jobs may substitute an
oracle, but that substitution is outside this module and outside the policy import closure.
"""
import json,time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from manip import Context as BaseContext, pose2, hold, follow, move_to, drive, TALL_TRUNK, TUCK
from skills import interpolate_cartesian,JointMove

ITEMS = ('pen', 'calculator', 'pencil', 'notebook', 'folder')


class OrientedBoxes:
    """Axis-aligned boxes in a rigid recipient frame, avoiding inflated world AABBs."""
    def __init__(self,boxes,frame):self.boxes=np.asarray(boxes,float);self.frame=np.asarray(frame,float).copy()
    def __len__(self):return len(self.boxes)
    def __getitem__(self,key):return self.boxes[key]


def transform(rotation, point):
    t = np.eye(4); t[:3, :3] = rotation; t[:3, 3] = point
    return t


def upright(yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class Context(BaseContext):
    """Existing whole-body context plus a carried-object collision model."""
    def __init__(self, folder, scene, estimator, log=print):
        super().__init__(folder, scene, estimator, log)
        self.models = {k: dict(np.load(Path(folder) / 'priors' / k / 'model.npz'))
                       for k in ('tote',) + ITEMS}
        self.held = None
        self.payload_clearance = .012
        self.gripper = dict(np.load(Path(folder) / 'priors' / 'robot_gripper.npz'))
        self.failed_pinches = set()
        self.failed_tote_geometry=set()
        self.eraser_broad = False
        self.requires_regrasp = False

    def validator(self, base=None, **kwargs):
        check = super().validator(base=base, **kwargs)
        world = self.base_matrix() if base is None else base
        held = self.held
        if held is None:
            return check
        arm, category, tip_to_object = held
        ik = self.iks[arm]
        bounds = self.models[category]['bounds']
        # A lattice covers the carried object's volume, including the tote handles. The tote's
        # interior is deliberately conservative during transport. Placement releases it above
        # the support, then lets native gravity establish contact.
        spacing = .035 if category == 'tote' else .012
        axes = [np.linspace(a, b, max(2, int(np.ceil((b-a)/spacing))+1)) for a,b in zip(*bounds)]
        pts = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        radius = spacing * .6
        clearance = self.payload_clearance
        def object_samples(poses,frame):
            link=poses[ik.link];tip=link.copy();tip[:3,3]+=link[:3,:3]@ik.tool
            obj=frame@tip@tip_to_object
            return pts@obj[:3,:3].T+obj[:3,3]
        if kwargs.get('relax_from') is not None:
            baseline=kwargs.get('baseline_base')
            if baseline is None:baseline=world
            initial=object_samples(self.body.fk(kwargs['relax_from'])[0],baseline)
            clearance=np.minimum(clearance,self.scene.contact_clearance(initial,np.full(len(pts),radius),
                                  static_only=kwargs.get('static_only',False))-.004)
        def payload_check(full, poses):
            if not check(full, poses):
                return False
            samples=object_samples(poses,world)
            # The recipient is an observed movable object, excluded from the static map.
            # Check the held object's swept volume against its walls/handles as well as the robot.
            if len(self.guard):
                if category=='calculator':
                    from payload_bounds import recipient_bounds_clear
                    link=poses[ik.link];tip=link.copy();tip[:3,3]+=link[:3,:3]@ik.tool
                    if not recipient_bounds_clear(bounds,world@tip@tip_to_object,self.guard,clearance=.015):return False
                else:
                    guard_frame=getattr(self.guard,'frame',None)
                    guarded=samples if guard_frame is None else (samples-guard_frame[:3,3])@guard_frame[:3,:3]
                    nearest=np.clip(guarded[None],self.guard[:,None,:3],self.guard[:,None,3:])
                    if np.min(np.linalg.norm(guarded[None]-nearest,axis=2)-radius)<.015:return False
            return bool(np.all(self.scene.contact_clearance(samples, np.full(len(pts), radius),
                            static_only=kwargs.get('static_only', False)) > clearance))
        return payload_check


def tucked(ctx):
    full = ctx.q.copy(); full[ctx.iks['left'].columns[:4]] = TALL_TRUNK
    for arm in ('left', 'right'):
        full[ctx.iks[arm].columns[4:]] = TUCK[arm]
    return full


def tuck(ctx, arms=('left', 'right')):
    previous = ctx.arm; result = 'done'
    for arm in arms:
        ctx.arm = arm
        status = yield from move_to(ctx, np.r_[TALL_TRUNK, TUCK[arm]], rate=.03, label='tuck')
        if status != 'done': result = status
    ctx.arm = previous
    return result


def grasp_candidates(ctx, category, obj, base=None,coarse=False):
    """Several object-relative pinches; elongated tools are pinched across their long axis."""
    if category=='calculator':
        from raised_calculator import raised_section_candidates
        yield from raised_section_candidates(ctx,obj)
        return
    if category!='tote' and min(ctx.models[category]['bounds'][1,:2]-ctx.models[category]['bounds'][0,:2])>.088:
        from school_push import edge_grasp_candidates
        if category in ('notebook','folder'):yield from edge_grasp_candidates(ctx,category,obj)
        if getattr(ctx,'edge_only',False):return
        from school_grasps import corner_candidates
        yield from corner_candidates(ctx,category,obj,base=base,coarse=coarse)
        return
    if category == 'tote':
        # Target a native AG ray through the arch, rather than the empty space below it.
        model=ctx.models['tote']['xyz'];ray=ctx.gripper[ctx.arm+'_ag']
        low=ray[np.argsort(ray[:,2])[:2]].mean(0)-ctx.ik.tool
        for side in (1., -1.):
            for x in (0.,-.025,.025,-.055,.055):
                points=model[(abs(model[:,0]-x)<.01)&(side*model[:,1]>.07)&(model[:,2]>.23)]
                if len(points)<4:continue
                centre=np.median(points,axis=0)
                for pitch in (-.65,0.,.65):
                    c, s = np.cos(pitch), np.sin(pitch)
                    rot = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
                    for dz in (0.,.003,-.003):
                        point=centre+np.array([0.,0.,dz])-rot@low
                        yield obj@transform(rot,point)
    else:
        bounds = ctx.models[category]['bounds']; centre = bounds.mean(0)
        # Paintbrush handle is narrower than its brush head; the object's native long axis is X.
        xs = (-.055, -.075, -.035) if category == 'paintbrush' else (float(centre[0]),0.,-.02) if category=='glue_stick' else (0.,float(centre[0]),-.02)
        if category=='rubber_eraser':xs=(float(centre[0]),float(centre[0])-.008,float(centre[0])+.008)
        world=ctx.base_matrix() if base is None else base
        radial=float(np.arctan2(obj[1,3]-world[1,3],obj[0,3]-world[0,3]))
        object_axis=float(np.arctan2(obj[1,0],obj[0,0]))
        azimuths=(object_axis,object_axis+np.pi,object_axis-.4,object_axis+.4,
                  object_axis+np.pi-.4,object_axis+np.pi+.4,radial,radial+np.pi)
        if category=='glue_stick':azimuths+= (object_axis+np.pi/2,object_axis-np.pi/2)
        for x in xs:
            point=(obj@np.array([x,float(centre[1]),float(centre[2]),1.]))[:3]
            pitches=(-1.2,-1.5,-.75,0.,.65) if category=='paintbrush' else (0.,-.75,-1.2,.65) if category=='rubber_eraser' else (0.,-1.2,-.75,.65)
            for pitch in pitches:
                for azimuth in azimuths:
                    c,s=np.cos(pitch),np.sin(pitch)
                    rotation=upright(azimuth)@np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])
                    closing=obj[:3,:3].T@rotation[:,1]
                    width=float(abs(closing)@(bounds[1]-bounds[0]))
                    if width>(.099 if category=='glue_stick' else .088):continue
                    if category=='rubber_eraser' and not ctx.eraser_broad and width>.032:continue
                    # TIP is the fingers' bottom, not the pad centre. The lowest native AG ray
                    # is 3.9 mm above it. Project the actual robot collision geometry for tilted
                    # pinches so fingers clear the support while rays intersect thin tools.
                    tool=ctx.ik.tool
                    floor_delta=float(np.min((ctx.gripper[ctx.arm+'_collision']-tool)@rotation[2]))
                    ray_low=float(np.min((ctx.gripper[ctx.arm+'_ag']-tool)@rotation[2]))-floor_delta
                    # OnTop(desk) can mean a lower shelf: train 2's eraser rests at 0.723 m,
                    # while the main desktop is 0.773 m. Use the observed object's underside.
                    support=float(np.quantile(ctx.models[category]['xyz']@obj[2,:3]+obj[2,3],.002))
                    object_top=float(obj[2,3]+bounds[1,2])
                    if support+ray_low>object_top+.002:continue
                    for clearance in ((.0008,) if coarse else (.0008,.003,-.0005)):
                        target=point.copy();target[2]=support+clearance-floor_delta
                        if category in ('glue_stick','rubber_eraser'):
                            rays=ctx.gripper[ctx.arm+'_ag'];low=rays[np.argsort(rays[:,2])[:2]].mean(0)-tool
                            target[:2]-=(rotation@low)[:2]
                        yield transform(rotation,target)


def plan_pinch(ctx, category, obj, base=None, q=None, count=12, candidate_limit=None, preferred_candidate=None, seconds=25.):
    deadline=time.monotonic()+seconds
    world = ctx.base_matrix() if base is None else base
    full = ctx.q if q is None else q
    check = ctx.validator(base=base, max_load=1.05)
    goals=list(grasp_candidates(ctx,category,obj,base=world,coarse=False))
    indices=range(len(goals))
    if candidate_limit is not None and len(goals)>candidate_limit:
        # A prefix budget only tried the first contact location / pitch, and
        # discarded the higher-clearance variants before any collision test.
        # Spread the same finite budget over the complete candidate family.
        spread=np.r_[np.linspace(0,len(goals)-1,min(12,candidate_limit)).astype(int),
                     np.linspace(0,len(goals)-1,candidate_limit).astype(int)]
        indices=list(dict.fromkeys(map(int,spread)))[:candidate_limit]
    if preferred_candidate is not None and 0<=preferred_candidate<len(goals):
        indices=[preferred_candidate]+[i for i in indices if i!=preferred_candidate]
    for searched,n in enumerate(indices):
        if time.monotonic()>deadline:
            ctx.event('pinch_planning_budget',category=category,searched=searched,total=len(goals),seconds=seconds);break
        n=int(n);goal=goals[n]
        if category!='tote':
            # Reject a finger envelope already intersecting the wall before expensive IK.
            # The full robot/self/load/edge validator remains mandatory after this precheck.
            mesh=ctx.gripper[ctx.arm+'_collision']
            edge_mode=category in ('notebook','folder') and getattr(ctx,'edge_only',False)
            if edge_mode:
                from school_push import EDGE_GRIPPER_CLEARANCE
                clearance=EDGE_GRIPPER_CLEARANCE
            else:
                mesh=mesh[::max(1,len(mesh)//300)];clearance=-.0002
            mesh=mesh-ctx.ik.tool
            fingers=mesh@goal[:3,:3].T+goal[:3,3]
            if np.min(ctx.scene.contact_clearance(fingers,np.full(len(fingers),.0004)))<clearance:continue
            # Sphere hand margins deliberately allow approximate surface contact. They cannot
            # permit the real open fingers to cut through a higher desktop above a lower shelf.
            lower=fingers.min(0)-.0003;upper=fingers.max(0)+.0003
            boxes=ctx.scene.boxes
            if not ctx.scene.precise_support_hulls and len(boxes) and np.any(np.all(upper>boxes[:,:3],axis=1)&np.all(lower<boxes[:,3:],axis=1)):continue
        relative=np.linalg.inv(obj)@goal
        geometry_key=(category,tuple(np.round(relative[:3,:3].ravel(),2)),tuple(np.round(relative[:3,3],3)))
        if category=='tote' and geometry_key in ctx.failed_tote_geometry:continue
        key=(category,ctx.arm,round(float(world[0,3]),1),round(float(world[1,3]),1),
             tuple(np.round(relative[:3,:3].ravel(),2)),tuple(np.round(relative[:3,3],3)))
        if key in ctx.failed_pinches:continue
        tip = np.linalg.inv(world) @ goal
        point, rotation = tip[:3, 3], tip[:3, :3]
        precision=dict(position_tolerance=.0008,rotation_tolerance=.02,rest_weight=.0005)
        found = ctx.ik.solve(full, point, rotation, count=count, validator=check, all_solutions=True,**precision)
        for low in found[:4]:
            start = full.copy(); start[ctx.ik.columns] = low
            # Solve at contact, then grow a collision-checked approach upwards, as in m05.
            up = interpolate_cartesian(ctx.ik, start, point, point+[0.,0.,.12], rotation,
                                       validator=check, step=.02)
            if up is not None:
                approach=up
                if category in ('notebook','folder') and (getattr(ctx,'edge_only',False) or abs(goal[2,1])>.9):
                    # The jaws straddle the thickness. Enter horizontally from
                    # outside the exposed edge; a vertical approach would sweep
                    # the lower finger through the object before closing.
                    away=world[:3,:3].T@goal[:3,2]*.10
                    approach=interpolate_cartesian(ctx.ik,start,point,point+away,rotation,
                                                   validator=check,step=.01)
                    if approach is None:continue
                return dict(low=low, up=up, approach=approach, pre=approach[-1], tip=tip, world_tip=goal,
                            solutions=len(found), candidate=n,key=key,geometry_key=geometry_key)
    return None


def contact_servo(ctx, plan, limit=65):
    """Bounded FK feedback compensates position-controller sag using proprioception alone."""
    command=ctx.chain.copy();nominal=plan['low'];check=ctx.validator(relax_from=ctx.q,max_load=1.10)
    for _ in range(limit):
        full=ctx.q;poses,frames=ctx.body.fk(full);actual=ctx.tip()
        delta=plan['tip'][:3,3]-actual[:3,3]
        phi=Rotation.from_matrix(plan['tip'][:3,:3]@actual[:3,:3].T).as_rotvec()
        if np.linalg.norm(delta)<.0015 and np.linalg.norm(phi)<.025:
            yield from hold(ctx,4);return True
        jac=ctx.body.jacobian(ctx.ik.link,actual[:3,3],frames,ctx.ik.columns)
        weights=np.array([5.,5.,5.,1.,1.,1.])
        matrix=jac*weights[:,None];error=np.r_[delta,phi]*weights
        change=matrix.T@np.linalg.solve(matrix@matrix.T+.0004*np.eye(6),error)
        wanted=np.clip(command+np.clip(.35*change,-.003,.003),nominal-.07,nominal+.07)
        wanted=np.clip(wanted,ctx.ik.lower+1e-5,ctx.ik.upper-1e-5)
        trial=full.copy();trial[ctx.ik.columns]=wanted
        if not check(trial,ctx.body.fk(trial)[0]):
            ctx.event('contact_servo',ok=False,reason='command collision or load',error=float(np.linalg.norm(delta)))
            return False
        command=wanted;yield ctx.action(chain=command)
    return False


def settle_contact_view(ctx,max_steps=90):
    """Wait for measured articulation/camera stability before close-range RGB-D."""
    observer=getattr(ctx.est,'ob',None)
    previous_q=ctx.q.copy()
    previous_rel=None if observer is None else np.array(observer.data['robot_r1::cam_rel_poses']).reshape(3,7).copy()
    stable=0
    for n in range(max_steps):
        yield ctx.action()
        q=ctx.q.copy();joint_delta=float(np.max(abs(q-previous_q)))
        rel=None if observer is None else np.array(observer.data['robot_r1::cam_rel_poses']).reshape(3,7).copy()
        camera_delta=0.;rotation_delta=0.
        if rel is not None and previous_rel is not None:
            camera_delta=float(np.max(np.linalg.norm(rel[:,:3]-previous_rel[:,:3],axis=1)))
            dot=np.sum(rel[:,3:]*previous_rel[:,3:],axis=1)
            rotation_delta=float(np.max(2*np.arccos(np.clip(abs(dot),0.,1.))))
        stable=stable+1 if joint_delta<.001 and camera_delta<.00075 and rotation_delta<.004 else 0
        previous_q=q;previous_rel=rel
        if stable>=6:
            ctx.event('contact_view_settled',ok=True,actions=n+1,joint_delta=joint_delta,camera_delta=camera_delta)
            return True
    ctx.event('contact_view_settled',ok=False,actions=max_steps,joint_delta=joint_delta,camera_delta=camera_delta)
    return False


def pinch(ctx, category, obj, refresh=None, preferred_candidate=None):
    plan = plan_pinch(ctx, category, obj, preferred_candidate=preferred_candidate)
    if plan is None:
        ctx.event('pinch', category=category, ok=False, reason='no approach'); return False
    ctx.set_grip(1.)
    status = yield from move_to(ctx, plan['pre'], label='approach', max_steps=400)
    if status != 'done': return False
    yield from hold(ctx, 12)
    if category in ('notebook','folder') and getattr(ctx,'edge_only',False):
        if not (yield from settle_contact_view(ctx)):return False
    if refresh is not None:
        fresh=None
        if category in ('notebook','folder') and getattr(ctx,'edge_only',False):
            from flat_rectification import confirm_contact_pose
            observer=getattr(ctx.est,'ob',None)
            confirmation=None if observer is None else confirm_contact_pose(observer,category,ctx.base_matrix(),obj)
            ctx.event('flat_contact_pose_check',stage='pregrasp',category=category,ok=confirmation is not None,pose=obj.tolist())
            if confirmation is not None:
                fresh=obj
                ctx.est.remember(category,fresh,source='current_RGBD_contact_confirmation')
        if fresh is None:fresh=refresh(category)
        if fresh is None and category in ('notebook','folder') and getattr(ctx,'edge_only',False):
            observer=getattr(ctx.est,'ob',None)
            if observer is not None:
                from pose_confirmation import confirm_tool
                confirmation=confirm_tool(observer,category,ctx.base_matrix(),obj,allow_flat_edge=True)
                ctx.event('edge_pose_confirmation',category=category,ok=confirmation is not None,info=confirmation)
                if confirmation is not None:fresh=obj
        if fresh is None:
            ctx.event('pinch', category=category, ok=False, reason='not observed at approach'); return False
        obj = fresh
        updated = plan_pinch(ctx, category, obj, count=20, preferred_candidate=plan['candidate'])
        if updated is None: return False
        plan = updated
        status = yield from move_to(ctx, plan['pre'], label='refined_approach', max_steps=200)
        if status != 'done': return False
    contact_object=np.linalg.inv(ctx.base_matrix())@obj
    down = plan.get('approach',plan['up'])[::-1][1:] + [plan['low']]
    status = yield from follow(ctx, down, rate=.012, max_steps=280)
    move=JointMove([plan['low']],rate=.006,lead=.025,tolerance=.0015,stall_steps=40)
    for _ in range(80):
        command,fine_status=move.step(ctx.chain)
        yield ctx.action(chain=command)
        if fine_status!='moving':break
    yield from hold(ctx,5)
    yield from contact_servo(ctx,plan)
    if category in ('notebook','folder') and getattr(ctx,'edge_only',False):
        if not (yield from settle_contact_view(ctx)):return False
    error = float(np.linalg.norm(ctx.tip()[:3,3] - plan['tip'][:3,3]))
    delta=ctx.tip()[:3,3]-plan['tip'][:3,3]
    reached=np.linalg.norm(delta[:2])<.0035 and abs(delta[2])<.0035
    if status != 'done' or not reached:
        ctx.failed_pinches.add(plan['key'])
        ctx.event('pinch', category=category, ok=False, reason='contact pose not reached',
                  status=status, error=error,delta=(ctx.tip()[:3,3]-plan['tip'][:3,3]).tolist()); return False
    if category!='tote':
        rays=ctx.gripper[ctx.arm+'_ag'];actual=ctx.base_matrix()@ctx.tip()
        heights=(rays-ctx.ik.tool)@actual[:3,:3].T+actual[:3,3]
        bounds=ctx.models[category]['bounds'];top=float(obj[2,3]+bounds[1,2])
        if heights[:,2].min()>top+.001:
            ctx.failed_pinches.add(plan['key'])
            ctx.event('pinch',category=category,ok=False,reason='rays above item',ray_z=float(heights[:,2].min()),item_top=top)
            return False
    if refresh is not None and category!='tote':
        fresh=None
        if category in ('notebook','folder') and getattr(ctx,'edge_only',False):
            from flat_rectification import confirm_contact_pose
            observer=getattr(ctx.est,'ob',None)
            predicted=ctx.base_matrix()@contact_object
            confirmation=None if observer is None else confirm_contact_pose(observer,category,ctx.base_matrix(),predicted)
            ctx.event('flat_contact_pose_check',stage='before_close',category=category,ok=confirmation is not None,pose=predicted.tolist())
            if confirmation is not None:
                fresh=predicted
                ctx.est.remember(category,fresh,source='current_RGBD_contact_confirmation')
                ctx.event('flat_contact_pose_confirmed',category=category,pose=fresh.tolist(),evidence=confirmation)
        if fresh is None:fresh=refresh(category)
        if fresh is not None:
            relative=np.linalg.inv(ctx.base_matrix())@fresh
            shift=float(np.linalg.norm(relative[:3,3]-contact_object[:3,3]))
            if shift>({'paintbrush':.045,'marker':.025}.get(category,.012)):
                ctx.failed_pinches.add(plan['key'])
                ctx.event('pinch',category=category,ok=False,reason='object moved during approach',shift=shift)
                yield from follow(ctx,plan['up'],rate=.018,max_steps=260)
                return False
            contact_object=relative;obj=fresh
    if category in ('notebook','folder') and getattr(ctx,'edge_only',False):
        observer=getattr(ctx.est,'ob',None)
        if observer is not None:
            from flat_rectification import depth_contradictions
            # A failed final refresh used to silently permit closing around a
            # stale pose, even when current depth showed free space there.
            current_object=ctx.base_matrix()@contact_object
            bad,evidence=depth_contradictions(observer,category,ctx.base_matrix(),current_object)
            if bad:
                ctx.failed_pinches.add(plan['key'])
                ctx.event('pinch',category=category,ok=False,reason='current depth contradicts contact pose',cameras=bad,depth=evidence)
                yield from follow(ctx,plan['up'],rate=.018,max_steps=260)
                return False
    ctx.event('contact',category=category,arm=ctx.arm,error=error,delta=delta.tolist(),candidate=plan['candidate'])
    ctx.set_grip(-1.); yield from hold(ctx, 35)
    before = ctx.fingers().copy()
    limit={'glue_stick':.033,'marker':.028}.get(category)
    if category=='glue_stick':
        closing=contact_object[:3,:3].T@ctx.tip()[:3,1]
        if abs(float(closing[0]))>.80:limit=.104
    if limit is not None and float(np.sum(before))>limit:
        ctx.failed_pinches.add(plan['key']);ctx.set_grip(1.)
        yield from hold(ctx,12)
        ctx.event('pinch',category=category,ok=False,reason='grip width inconsistent with tool',gap=float(np.sum(before)),maximum=limit)
        return False
    # Remember the object-to-tip relation from the latest pose estimate, never a simulator flag.
    # Keep contact geometry in the robot frame; map corrections during descent must not
    # turn a stationary, previously observed object into a shifted grasp transform.
    relation = np.linalg.inv(ctx.tip()) @ contact_object
    ctx.held = (ctx.arm, category, relation)
    status = yield from follow(ctx, plan['up'], rate=.018, max_steps=260)
    yield from hold(ctx, 10)
    gap = float(np.sum(ctx.fingers()))
    held = status == 'done' and gap > (.012 if category == 'tote' else .003)
    ctx.event('pinch', category=category, arm=ctx.arm, ok=held, gap=gap,
              before=before.tolist(), lift=status, candidate=plan['candidate'], solutions=plan['solutions'])
    if not held:
        ctx.failed_pinches.add(plan['key'])
        if category=='tote':ctx.failed_tote_geometry.add(plan['geometry_key'])
        ctx.held = None; ctx.set_grip(1.)
    return held


def floor_stance(ctx, obj, exclude=()):
    nominal = tucked(ctx); previous = ctx.arm; best = None; candidates = []
    centre = obj[:2,3]
    for radius in (.62, .75, .50):
        for angle in np.linspace(-np.pi, np.pi, 16, endpoint=False):
            xy = centre + radius*np.array([np.cos(angle),np.sin(angle)])
            yaw = float(np.arctan2(*(centre-xy)[::-1])); pose = (float(xy[0]),float(xy[1]),yaw)
            if any(np.linalg.norm(xy-np.asarray(e[:2]))<.10 and abs((yaw-e[2]+np.pi)%(2*np.pi)-np.pi)<.3 for e in exclude): continue
            if not ctx.scene.base_free(xy): continue
            if ctx.scene.path(ctx.est.base()[:2], xy, keepout=((centre,.48),)) is None: continue
            candidates.append((np.linalg.norm(xy-np.asarray(ctx.est.base()[:2])),pose))
    try:
        for _, pose in sorted(candidates, key=lambda x:x[0])[:24]:
            world = pose2(*pose)
            for arm in ('left','right'):
                ctx.arm=arm
                if not ctx.validator(base=world)(nominal,ctx.body.fk(nominal)[0]): continue
                plan=plan_pinch(ctx,'tote',obj,base=world,q=nominal,count=8,candidate_limit=18)
                if plan is not None:
                    found=dict(base=pose,arm=arm,solutions=plan['solutions'])
                    if best is None or found['solutions']>best['solutions']: best=found
                    if best['solutions']>=3: return best
        return best
    finally:
        ctx.arm=previous


def release_ik(ctx,tip,base=None):
    """Search both the carried configuration and the proven nominal torso/arm branch."""
    check=ctx.validator(base=base,max_load=1.10)
    found=ctx.ik.solve(ctx.q,tip[:3,3],tip[:3,:3],count=24,validator=check,all_solutions=True)
    if len(found)<3:
        current_count=len(found)
        nominal=ctx.q.copy();nominal[ctx.ik.columns]=np.r_[TALL_TRUNK,TUCK[ctx.arm]]
        alternate=ctx.ik.solve(nominal,tip[:3,3],tip[:3,:3],count=24,validator=check,all_solutions=True)
        for low in alternate:
            if not any(np.max(abs(low-other))<.02 for other in found):found.append(low)
        if alternate:ctx.event('release_ik_alternative',current_solutions=current_count,nominal_solutions=len(alternate),added=len(found)-current_count)
    if ctx.held is not None and ctx.held[1]=='calculator':
        # Check both fingers throughout opening, retaining full body, payload,
        # support, recipient and load checks before committing to the approach.
        fingers=[i for i,name in enumerate(ctx.body.names) if name.startswith(ctx.arm+'_gripper_finger')]
        safe=[]
        for low in found:
            full=ctx.q.copy();full[ctx.ik.columns]=low;initial=full[fingers].copy()
            valid=True
            for fraction in np.linspace(0.,1.,7):
                full[fingers]=initial*(1-fraction)+.05*fraction
                if not check(full,ctx.body.fk(full)[0]):valid=False;break
            if valid:safe.append(low)
        found=safe
    return sorted(found,key=lambda low:np.linalg.norm(low-ctx.chain))


def place_object(ctx, goal, label='place'):
    """Move the held object to a geometric release pose, open, retreat vertically, and settle."""
    if ctx.held is None: return False
    arm, category, relation = ctx.held; ctx.arm=arm
    tip = np.linalg.inv(ctx.base_matrix()) @ goal @ np.linalg.inv(relation)
    if category=='tote':
        reached=yield from move_tote_upright(ctx,goal)
        found=[None] if reached else []
    else:
        found = release_ik(ctx,tip)
    for low in found[:4]:
        if low is None:status='done'
        else:
            table_insert=label=='insert' and getattr(ctx.guard,'frame',np.eye(4))[2,3]>=.55
            full=ctx.q.copy();full[ctx.ik.columns]=low
            up=interpolate_cartesian(ctx.ik,full,tip[:3,3],tip[:3,3]+[0.,0.,.06 if table_insert else .15],tip[:3,:3],
                                     validator=ctx.validator(max_load=1.10),step=.02)
            status='no_above_path'
            if up is not None:
                status=yield from move_to(ctx,up[-1],label=label+'_above',rate=.022,max_steps=450,max_load=1.10)
                if status=='done':status=yield from follow(ctx,up[::-1][1:]+[low],rate=.015,max_steps=300)
            if status!='done' and table_insert:
                # Release is already above the handle envelope. A checked joint path may
                # reach it even when the additional overhead Cartesian pose is unreachable.
                status=yield from move_to(ctx,low,label=label+'_direct',rate=.015,max_steps=450,max_load=1.10)
        if status!='done': continue
        yield from hold(ctx,14)
        error=float(np.linalg.norm(ctx.tip()[:3,3]-tip[:3,3]))
        if error>.014: continue
        ctx.set_grip(1.); yield from hold(ctx,25); ctx.held=None
        up=interpolate_cartesian(ctx.ik,ctx.q,ctx.tip()[:3,3],ctx.tip()[:3,3]+[0.,0.,.15],
                                 ctx.tip()[:3,:3],validator=ctx.validator(),step=.025)
        if up is not None: yield from follow(ctx,up,rate=.022,max_steps=220)
        yield from hold(ctx,35)
        ctx.event(label,ok=True,category=category,error=error,retreat=up is not None)
        return True
    ctx.event(label,ok=False,category=category,reason='no reachable release',solutions=len(found))
    return False


def move_tote_upright(ctx, goal):
    """Carry a filled tote with an upright object path, solving the existing IK at each waypoint."""
    if ctx.held is None or ctx.held[1]!='tote':return False
    relation=ctx.held[2];base=ctx.base_matrix();start=base@ctx.tip()@relation
    yaw0=float(np.arctan2(start[1,0],start[0,0]));yaw1=float(np.arctan2(goal[1,0],goal[0,0]))
    delta=(yaw1-yaw0+np.pi)%(2*np.pi)-np.pi
    count=max(2,int(np.ceil(np.linalg.norm(goal[:3,3]-start[:3,3])/.035)),int(np.ceil(abs(delta)/.10)))
    full=ctx.q.copy();path=[];check=ctx.validator(relax_from=ctx.q,max_load=1.10)
    for fraction in np.linspace(0.,1.,count+1)[1:]:
        obj=transform(upright(yaw0+fraction*delta),start[:3,3]*(1-fraction)+goal[:3,3]*fraction)
        tip=np.linalg.inv(base)@obj@np.linalg.inv(relation)
        low=ctx.ik.solve(full,tip[:3,3],tip[:3,:3],count=6,validator=check,rest_pose=full[ctx.ik.columns])
        if low is None:
            low=ctx.ik.solve(full,tip[:3,3],tip[:3,:3],count=32,validator=check,rest_pose=full[ctx.ik.columns])
        if low is None:
            nominal=full.copy();nominal[ctx.ik.columns]=np.r_[TALL_TRUNK,TUCK[ctx.arm]]
            low=ctx.ik.solve(nominal,tip[:3,3],tip[:3,:3],count=24,validator=check,rest_pose=full[ctx.ik.columns])
        if low is None:
            ctx.event('upright_transfer',ok=False,reason='IK waypoint',fraction=float(fraction));return False
        previous=full[ctx.ik.columns].copy()
        # Validate the joint interpolation too: Cartesian endpoints alone do not prove a safe edge.
        for blend in np.linspace(0.,1.,max(3,int(np.max(abs(low-previous))/.04)+1)):
            trial=full.copy();trial[ctx.ik.columns]=previous*(1-blend)+low*blend
            if not check(trial,ctx.body.fk(trial)[0]):return False
        full[ctx.ik.columns]=low;path.append(low)
    status=yield from follow(ctx,path,rate=.006,max_steps=950)
    ctx.event('upright_transfer',ok=status=='done',status=status,waypoints=len(path))
    return status=='done'


def surface_stances(ctx, point):
    """Stances around a known work surface, including offsets to pass chairs."""
    desk=ctx.scene.meta['target_surface'];lo=np.array(desk['aabb_lo']);hi=np.array(desk['aabb_hi'])
    candidates=[]
    for offset in (.28,.36,.46,.24,.20):
        # A long bed can have furniture blocking the stance directly in front of
        # the object. Cover diagonal reaches from the neighbouring free aisle.
        # Every resulting stance still passes the footprint and whole-body checks.
        for lateral in (0.,.15,-.15,.3,-.3,.45,-.45,.60,-.60,.75,-.75,.90,-.90):
            for side in ('north','east','west','south'):
                if side in ('north','south'):
                    x=float(np.clip(point[0]+lateral,lo[0]-.15,hi[0]+.15))
                    y=float(hi[1]+offset if side=='north' else lo[1]-offset)
                    yaw=-np.pi/2 if side=='north' else np.pi/2
                else:
                    x=float(hi[0]+offset if side=='east' else lo[0]-offset)
                    y=float(np.clip(point[1]+lateral,lo[1]-.15,hi[1]+.15))
                    yaw=np.pi if side=='east' else 0.
                if ctx.scene.base_free((x,y)) or ctx.scene.footprint_free(x,y,yaw,(-.40,.24,.345),margin=.015,movable_margin=.02):
                    candidates.append((x,y,yaw))
    return sorted(candidates,key=lambda p:np.linalg.norm(np.asarray(p[:2])-ctx.est.base()[:2]))


def chassis_surface_drive(ctx,stance):
    """Local fallback: oriented chassis, full body/payload and observed recipient swept checks."""
    start=np.asarray(ctx.est.base(),float);yaw=stance[2]
    from carried_navigation import floor_tool_keepouts
    floor_tools=floor_tool_keepouts(ctx,start,())
    if any(np.linalg.norm(np.asarray(stance[:2])-point)<radius for point,radius in floor_tools):return 'no_route'
    full=ctx.q.copy();poses=ctx.body.fk(full)[0];baseline=ctx.base_matrix()
    delta=(yaw-start[2]+np.pi)%(2*np.pi)-np.pi
    def safe(point,angle):
        if any(np.linalg.norm(np.asarray(point)-centre)<radius-.01 for centre,radius in floor_tools):return False
        # The rectangle includes empty chassis corners. The existing detailed robot
        # spheres, full body, payload and observed recipient remain the final validator.
        return ctx.validator(base=pose2(*point,angle),relax_from=full,baseline_base=baseline)(full,poses)
    path=ctx.scene.chassis_path(start[:2],stance[:2],yaw,valid=lambda point:safe(point,yaw))
    if path is None:return 'no_route'
    for angle in np.linspace(start[2],start[2]+delta,max(2,int(np.ceil(abs(delta)/.05))+1)):
        if not safe(start[:2],angle):return 'no_route'
    for a,b in zip(path,path[1:]):
        for point in np.linspace(a,b,max(2,int(np.ceil(np.linalg.norm(b-a)/.025))+1)):
            if not safe(point,yaw):return 'no_route'
    ctx.event('chassis_route',ok=True,stance=list(stance),waypoints=[p.tolist() for p in path],
              recipient_checked=bool(len(ctx.guard)))
    # Drive individual straight legs, so the generic lookahead cannot cut a checked corner.
    legs=[(start[:2],yaw)]+[(point,yaw) for point in path[1:]]
    for point,angle in legs:
        motion=drive(ctx,point,angle,direct=True,face_travel=False,speed=.18,tolerance=.012,
                     yaw_tolerance=.015,min_turn=.06,max_steps=450)
        while True:
            try:action=next(motion)
            except StopIteration as done:
                status=done.value;break
            # Reject a proposed next base step if observation updates invalidate the sweep.
            x,y,current_yaw=ctx.est.base();c,s=np.cos(current_yaw),np.sin(current_yaw)
            local=np.asarray(action[:2])*.75/30.
            predicted=np.array([x+c*local[0]-s*local[1],y+s*local[0]+c*local[1]])
            predicted_yaw=current_yaw+float(action[2])/30.
            current_full=ctx.q;current_poses=ctx.body.fk(current_full)[0]
            if any(np.linalg.norm(predicted-centre)<radius-.01 for centre,radius in floor_tools):
                yield ctx.action(base=(0.,0.,0.));return 'no_route'
            if not ctx.validator(base=pose2(*predicted,predicted_yaw),relax_from=full,baseline_base=baseline)(current_full,current_poses):
                yield ctx.action(base=(0.,0.,0.))
                ctx.event('chassis_route',ok=False,reason='updated observation invalidates next step')
                return 'no_route'
            yield action
        if status!='arrived':return status
    return 'arrived'


def surface_drive(ctx, stance):
    """Approach a surface using chassis geometry for the final short stretch under its top."""
    recipient=getattr(ctx.guard,'frame',None)
    keepout=()
    if recipient is not None and recipient[2,3]<.55:
        bounds=ctx.models['tote']['bounds']
        # Route simplification accepts a 3 cm disc slack; include it in the planning radius.
        radius=ctx.scene.base_radius+float(np.max(np.linalg.norm(bounds[:,:2],axis=1)))+.055
        keepout=((recipient[:2,3],radius),)
        if np.linalg.norm(np.asarray(stance[:2])-recipient[:2,3])<radius:
            return (yield from chassis_surface_drive(ctx,stance))
    path=ctx.scene.path(ctx.est.base()[:2],stance[:2],keepout=keepout)
    if path is None:
        back=np.array(stance[:2])-.24*np.array([np.cos(stance[2]),np.sin(stance[2])])
        path=ctx.scene.path(ctx.est.base()[:2],back,keepout=keepout)
        if path is None:return (yield from chassis_surface_drive(ctx,stance))
        full=ctx.q;poses=ctx.body.fk(full)[0]
        for xy in np.linspace(back,stance[:2],12):
            if any(np.linalg.norm(xy-centre)<radius for centre,radius in keepout):return (yield from chassis_surface_drive(ctx,stance))
            if not ctx.scene.footprint_free(*xy,stance[2],(-.40,.24,.345),margin=.015,movable_margin=.02):return (yield from chassis_surface_drive(ctx,stance))
            if not ctx.validator(base=pose2(*xy,stance[2]))(full,poses):return (yield from chassis_surface_drive(ctx,stance))
        path=path+[np.asarray(stance[:2])]
    # The XY route alone does not validate rotations of the whole robot near furniture.
    from carried_navigation import checked_carried_drive
    return (yield from checked_carried_drive(ctx,stance,max_steps=1000,keepout=keepout,route=path))


def carry(ctx,height=None,stable_view=True):
    if ctx.held is None:return False
    arm,category,relation=ctx.held;ctx.arm=arm
    height=(1.03 if category=='tote' else 1.00) if height is None else float(height)
    tote_yaw=None
    if category=='tote':
        current=ctx.base_matrix()@ctx.tip()@relation;tote_yaw=float(np.arctan2(current[1,0],current[0,0]))
        if current[2,3]<.70:
            lift=transform(upright(tote_yaw),current[:3,3]+[0.,0.,.72-current[2,3]])
            if not (yield from move_tote_upright(ctx,lift)):
                ctx.event('carry',ok=False,category=category,reason='vertical lift before translation');return False
    points=((.35,.20 if arm=='left' else -.20),(.40,0.),(.25,.25 if arm=='left' else -.25))
    # Keep normal transports unchanged. Recipient reacquisition can explicitly ask
    # for extra tool-yaw choices with the ordinary torso before a free-torso branch.
    view_yaws=(0.,np.pi/2,-np.pi/2,np.pi) if stable_view else (0.,)
    options=([(xy,None,0.) for xy in points] if category=='tote' else
             [(xy,True,angle) for angle in view_yaws for xy in points]+
             [(xy,False,0.) for xy in points])
    for xy,fixed_trunk,object_yaw in options:
        # A failed search must return an action before trying another pose.
        # This keeps the public WebSocket evaluator responsive during recovery.
        yield ctx.action()
        obj=ctx.base_matrix()@transform(upright(object_yaw),[xy[0],xy[1],height-.012])
        if category=='tote':
            obj[:3,:3]=upright(tote_yaw)
            if (yield from move_tote_upright(ctx,obj)):return True
            continue
        tip=np.linalg.inv(ctx.base_matrix())@obj@np.linalg.inv(relation)
        locked={i:float(v) for i,v in enumerate(TALL_TRUNK)} if fixed_trunk else None
        found=ctx.ik.solve(ctx.q,tip[:3,3],tip[:3,:3],count=16,validator=ctx.validator(),
                           all_solutions=True,locked=locked)
        for low in found[:3]:
            yield ctx.action()
            status=yield from move_to(ctx,low,label='carry',rate=.025,max_steps=400)
            if status=='done':
                yield from hold(ctx,6)
                retained=float(np.sum(ctx.fingers()))>.003
                ctx.event('carry',ok=retained,category=category,fixed_trunk=fixed_trunk,stable_view=stable_view,object_yaw=object_yaw,height=height,gap=float(np.sum(ctx.fingers())))
                if not retained:ctx.held=None
                return retained
    ctx.event('carry',ok=False,category=category);return False


def carried_drive(ctx, stance, max_steps=1200, keepout=()):
    from carried_navigation import checked_carried_drive
    return (yield from checked_carried_drive(ctx,stance,max_steps=max_steps,keepout=keepout))


def leave_surface(ctx):
    """Back away without turning while the chassis is close to or under the desk top."""
    desk=ctx.scene.meta['target_surface'];current=ctx.est.base();xy=np.array(current[:2])
    nearest=np.clip(xy,desk['aabb_lo'][:2],desk['aabb_hi'][:2])
    if np.linalg.norm(xy-nearest)>.65:return True
    direction=-np.array([np.cos(current[2]),np.sin(current[2])])
    full=ctx.q.copy();poses=ctx.body.fk(full)[0]
    start_world=ctx.base_matrix()
    for distance in (.35,.45,.25):
        goal=xy+distance*direction
        if not all(ctx.validator(base=pose2(*point,current[2]),relax_from=full,baseline_base=start_world)(full,poses)
                   for point in np.linspace(xy,goal,12)):continue
        status=yield from drive(ctx,goal,current[2],direct=True,face_travel=False,speed=.18,
                               tolerance=.015,max_steps=320)
        ctx.event('leave_surface',ok=status=='arrived',status=status,distance=distance)
        if status=='arrived':return True
    # A floor recipient behind the chassis can block every straight retreat.
    # Search a bounded oblique egress without turning the carried flat object.
    # Preserve the established straight choices; accept only full swept-body,
    # payload, recipient and floor-tool checks, including online revalidation.
    from carried_navigation import checked_carried_drive, floor_tool_keepouts, discs_clear
    tools=floor_tool_keepouts(ctx,np.asarray(current),())
    for angle in (np.pi/6,-np.pi/6,np.pi/3,-np.pi/3):
        direction=-np.array([np.cos(current[2]+angle),np.sin(current[2]+angle)])
        for distance in (.25,.35,.45,.55,.70):
            goal=xy+distance*direction
            if np.linalg.norm(goal-np.clip(goal,desk['aabb_lo'][:2],desk['aabb_hi'][:2]))<=.65:continue
            if not ctx.scene.base_free(goal) or not discs_clear([xy,goal],tools):continue
            points=np.linspace(xy,goal,max(2,int(np.ceil(distance/.01))+1))
            if not all(ctx.validator(base=pose2(*point,current[2]),relax_from=full,baseline_base=start_world)(full,poses)
                       for point in points):continue
            ctx.event('leave_surface_oblique_plan',angle=float(angle),distance=distance,goal=goal.tolist())
            status=yield from checked_carried_drive(ctx,(*goal,current[2]),max_steps=320,route=[xy,goal])
            ctx.event('leave_surface',ok=status=='arrived',status=status,distance=distance,oblique=True)
            # Recompute from a fresh invocation after any partial motion.
            return status=='arrived'
    ctx.event('leave_surface',ok=False,reason='no checked retreat');return False


def bag_slots(ctx, objects=()):
    """Desk placements with the full tote footprint separated from known tools."""
    desk=ctx.scene.meta['target_surface'];lo=np.array(desk['aabb_lo']);hi=np.array(desk['aabb_hi'])
    bounds=ctx.models['tote']['bounds'];half=(bounds[1,:2]-bounds[0,:2])/2+.035
    candidates=[]
    for x in np.linspace(lo[0]+half[0],hi[0]-half[0],9):
        for y in np.linspace(lo[1]+half[1],hi[1]-half[1],3):
            clear=min([np.linalg.norm(np.maximum(np.abs(np.asarray(p)[:2]-[x,y])-half-.075,0.)) for p in objects] or [1.])
            if clear<.035:continue
            candidates.append((clear-.03*np.linalg.norm(np.array([x,y])-ctx.est.base()[:2]),
                               transform(upright(),[x,y,hi[2]-bounds[0,2]+.055])))
    return [p for _,p in sorted(candidates,key=lambda x:-x[0])]


def hybrid_bag_slots(ctx,remaining,preferred_xy,occupied_points=()):
    """Reserve a real table footprint clear of the remaining observed canonical objects."""
    desk=ctx.scene.meta['target_surface'];lo=np.array(desk['aabb_lo']);hi=np.array(desk['aabb_hi'])
    bounds=ctx.models['tote']['bounds'];centre=bounds.mean(0);candidates=[];occupied_points=np.asarray(occupied_points)
    surfaces=[b for b in ctx.scene.boxes if b[5]>hi[2]-.004 and b[5]<=hi[2]+.004
              and b[3]-b[0]>.4 and b[4]-b[1]>.3 and b[5]-b[2]<.10]
    if surfaces:
        plate=max(surfaces,key=lambda b:(b[3]-b[0])*(b[4]-b[1]));lo[:2]=plate[:2];hi[:2]=plate[3:5]
    for yaw in (0.,np.pi/2):
        rotation=upright(yaw);half=abs(rotation[:2])@(bounds[1]-bounds[0])/2+.025
        if np.any(hi[:2]-lo[:2]<2*half):continue
        for x in np.linspace(lo[0]+half[0],hi[0]-half[0],11):
            for y in np.linspace(lo[1]+half[1],hi[1]-half[1],5):
                position=np.array([x,y]);clear=1.
                if occupied_points.size and np.any(np.all(abs(occupied_points[:,:2]-position)<half+.015,axis=1)):continue
                for category,obj in remaining.items():
                    other=ctx.models[category]['bounds'];other_centre=(obj@np.r_[other.mean(0),1.])[:2]
                    other_half=abs(obj[:2,:3])@(other[1]-other[0])/2
                    gap=float(np.linalg.norm(np.maximum(abs(position-other_centre)-half-other_half-.035,0.)))
                    clear=min(clear,gap)
                if clear<.025:continue
                # Root pose follows the canonical centre, while the whole footprint stays supported.
                point=np.r_[position,hi[2]-bounds[0,2]+.045];point[:2]-=(rotation@centre)[:2]
                score=.30*clear-.75*np.linalg.norm(position-np.asarray(preferred_xy))-.03*np.linalg.norm(position-ctx.est.base()[:2])
                candidates.append((score,transform(rotation,point)))
    return [p for _,p in sorted(candidates,key=lambda item:-item[0])]


def diverse_bag_goals(goals, primary=12, fallback=24):
    """Keep proven preferred slots, then cover distinct bed regions and orientations.

    Adjacent samples can share the same inaccessible aisle. Greedy farthest
    sampling bounds the fallback search without repeatedly trying that region.
    Every selected goal still requires the unchanged IK and carried-route checks.
    """
    goals=list(goals);selected=list(range(min(primary,len(goals))))
    remaining=list(range(len(selected),len(goals)))
    def distance(i,j):
        a,b=goals[i],goals[j]
        yaw_a=np.arctan2(a[1,0],a[0,0]);yaw_b=np.arctan2(b[1,0],b[0,0])
        yaw_delta=abs((yaw_a-yaw_b+np.pi)%(2*np.pi)-np.pi)
        return float(np.linalg.norm(a[:2,3]-b[:2,3])+.15*yaw_delta)
    for _ in range(min(fallback,len(remaining))):
        index=max(remaining,key=lambda i:min(distance(i,j) for j in selected))
        selected.append(index);remaining.remove(index)
    return [goals[i] for i in selected]


def place_bag(ctx, objects=(),goals=None):
    for goal in diverse_bag_goals(bag_slots(ctx,objects) if goals is None else goals):
        # Clamping lateral samples at mattress corners produces repeated poses.
        # Spend the finite stance budget on distinct physical approaches.
        unique_stances={}
        for stance in surface_stances(ctx,goal[:3,3]):
            unique_stances.setdefault(tuple(np.round(stance,5)),stance)
        for stance in list(unique_stances.values())[:16]:
            yield ctx.action()
            world=pose2(*stance);relation=ctx.held[2]
            tip=np.linalg.inv(world)@goal@np.linalg.inv(relation)
            # The release endpoint must obey the same load gate as every
            # waypoint in move_tote_upright, before spending navigation actions.
            found=ctx.ik.solve(ctx.q,tip[:3,3],tip[:3,:3],count=8,
                               validator=ctx.validator(base=world,max_load=1.10),all_solutions=True)
            if not found:continue
            ctx.event('bag_placement_plan',goal=goal.tolist(),stance=list(stance),
                      solutions=len(found),max_load=1.10)
            if ctx.scene.path(ctx.est.base()[:2],stance[:2]) is None:continue
            status=yield from carried_drive(ctx,stance)
            if status!='arrived':continue
            if (yield from place_object(ctx,goal,label='place_bag')):return goal
    return None


def pick_surface(ctx, category, obj, refresh=None, excluded=(),allow_fallback=True):
    nominal=tucked(ctx);previous=ctx.arm;physical_attempts=0
    stances=[p for p in surface_stances(ctx,obj[:3,3]) if np.linalg.norm(np.asarray(p[:2])-obj[:2,3])<1.10]
    for stance in stances[:60]:
        if any(np.linalg.norm(np.asarray(stance[:2])-e[:2])<.08 for e in excluded):continue
        world=pose2(*stance)
        for arm in ('left','right'):
            yield ctx.action()
            ctx.arm=arm
            if not ctx.validator(base=world)(nominal,ctx.body.fk(nominal)[0]):continue
            plan=plan_pinch(ctx,category,obj,base=world,q=nominal,count=8,candidate_limit=40)
            if plan is None:continue
            status=yield from surface_drive(ctx,stance)
            if status!='arrived':continue
            yield from tuck(ctx)
            refine=getattr(ctx.est,'refine_desk',None)
            if refine is not None:
                info=refine()
                if info is not None:ctx.event('desk_alignment',**info)
            target=refresh(category) if refresh else obj
            if target is None:continue
            if physical_attempts>=4:
                ctx.event('surface_reach',ok=False,category=category,reason='physical attempt limit');ctx.arm=previous
                if category=='rubber_eraser' and allow_fallback and not ctx.eraser_broad:
                    ctx.eraser_broad=True
                    ctx.event('grasp_fallback',category=category,reason='narrow attempt budget exhausted; broad pickup then reposition')
                    return (yield from pick_surface(ctx,category,obj,refresh=refresh,excluded=excluded,allow_fallback=False))
                return False
            physical_attempts+=1
            if (yield from pinch(ctx,category,target,refresh=refresh,preferred_candidate=plan['candidate'])):
                ctx.requires_regrasp=category=='rubber_eraser' and float(np.sum(ctx.fingers()))>.032
                return True
            ctx.event('retry',category=category,stance=list(stance),arm=arm)
            yield from tuck(ctx)
    ctx.arm=previous
    if category=='rubber_eraser' and allow_fallback and not ctx.eraser_broad:
        ctx.eraser_broad=True
        ctx.event('grasp_fallback',category=category,reason='narrow jaw approach blocked; reposition on clear tabletop')
        return (yield from pick_surface(ctx,category,obj,refresh=refresh,excluded=excluded,allow_fallback=False))
    ctx.event('surface_reach',ok=False,category=category,reason='stances exhausted',target=obj[:3,3].tolist())
    return False


def reorient_on_surface(ctx,objects=()):
    """An intermediate tabletop placement frees room for a narrow grasp away from a wall."""
    if ctx.held is None:return None
    category=ctx.held[1];desk=ctx.scene.meta['target_surface']
    lo=np.asarray(desk['aabb_lo']);hi=np.asarray(desk['aabb_hi']);bounds=ctx.models[category]['bounds']
    base=ctx.base_matrix();yaw=float(np.arctan2(base[1,0],base[0,0]))
    for forward in (.55,.65,.45):
        for lateral in (.15,-.15,0.,.28,-.28):
            centre=(base@np.array([forward,lateral,0.,1.]))[:3];centre[2]=hi[2]+(bounds[1,2]-bounds[0,2])/2+.04
            if np.any(centre[:2]<lo[:2]+.08) or np.any(centre[:2]>hi[:2]-.08):continue
            if any(np.linalg.norm(centre[:2]-np.asarray(p)[:2])<.14 for p in objects):continue
            probe=centre.copy();probe[2]=hi[2]+.08
            if ctx.scene.sphere_clearance(probe[None],np.array([.04]))[0]<.015:continue
            goal=transform(upright(yaw),centre);goal[:3,3]-=goal[:3,:3]@bounds.mean(0)
            if (yield from place_object(ctx,goal,label='reorient_table')):
                ctx.requires_regrasp=False;ctx.eraser_broad=False
                ctx.event('reorient_table',category=category,ok=True,target=goal[:3,3].tolist())
                return goal
    ctx.event('reorient_table',category=category,ok=False,reason='no clear reachable temporary support')
    return None


def insertion_goal(ctx, category, bag, slot=0, height_offset=0.,world_yaw=None):
    # Body rim lies below the handles. Let the object lie horizontally and release just above
    # that rim; reaching into the bottom of the bag was a recorded historical failure.
    offsets=((0.,0.),(.018,0.),(-.018,0.),(0.,0.),(0.,0.))
    x,y=offsets[slot%len(offsets)];bounds=ctx.models[category]['bounds']
    centre=bounds.mean(0)
    # Keep the opened fingers above the handle envelope (.295 m). The lower eraser
    # release used in source 14 had no collision-free IK after a successful regrasp.
    # The calculator fits between the handles. Release its bottom 45 mm
    # above the .17 m body rim; higher checked alternatives remain available.
    # This reduces the free fall and the lateral wall impact after opening.
    base_height=.215 if category=='calculator' else .335
    z=base_height+height_offset+float((bounds[1,2]-bounds[0,2])/2)
    rotation=bag[:3,:3]@upright((slot%2)*np.pi/2 if category=='rubber_eraser' else 0.) if world_yaw is None else upright(world_yaw)
    half=abs((bag[:3,:3].T@rotation)[:2])@(bounds[1]-bounds[0])/2
    room=np.maximum(np.array([.14,.09])-half-.012,0.)
    x,y=np.clip([x,y],-room,room)
    point=(bag@np.array([x,y,z,1.]))[:3]-rotation@centre
    return transform(rotation,point)


def bag_guards(bag):
    """Four walls and two handle regions, leaving the central opening free."""
    local=[[-.175,-.13,-.19,-.14,.13,.17],[.14,-.13,-.19,.175,.13,.17],
           [-.14,-.13,-.19,.14,-.09,.17],[-.14,.09,-.19,.14,.13,.17],
           [-.11,-.135,.17,.11,-.08,.295],[-.11,.08,.17,.11,.135,.295]]
    return OrientedBoxes(local,bag)


def radial_stances(ctx,centre,radii,minimum=8):
    """Refine angular sampling only when the free approach corridor is narrow."""
    centre=np.asarray(centre)[:2];result=[];seen=set()
    for count in (16,64):
        for radius in radii:
            for angle in np.linspace(-np.pi,np.pi,count,endpoint=False):
                xy=centre+radius*np.array([np.cos(angle),np.sin(angle)])
                key=tuple(np.round(xy,4))
                if key in seen:continue
                seen.add(key)
                if ctx.scene.base_free(xy):
                    result.append((*map(float,xy),float(np.arctan2(*(centre-xy)[::-1]))))
        if len(result)>=minimum:break
    return result


def release_yaw(ctx,category,bag,preferred):
    """Keep the projected object footprint inside the observed recipient mouth."""
    extent=ctx.models[category]['bounds'][1]-ctx.models[category]['bounds'][0]
    bag_yaw=float(np.arctan2(bag[1,0],bag[0,0]))
    choices=[preferred,bag_yaw,bag_yaw+np.pi,bag_yaw+np.pi/2,bag_yaw-np.pi/2]
    choices.sort(key=lambda yaw:abs((yaw-preferred+np.pi)%(2*np.pi)-np.pi))
    for yaw in choices:
        half=abs((bag[:3,:3].T@upright(yaw))[:2])@extent/2
        if np.all(half<=np.array([.14,.09])-.025):return float(yaw)
    return None


def insert(ctx, category, bag, slot=0, refresh=None, observe=None):
    if bag[2,3]<.55 and not (yield from leave_surface(ctx)):return False
    if bag[2,3]>=.55 and not (yield from carry(ctx,height=bag[2,3]+.41)):return False
    view_attempts=0
    for n in range(3):
        goal=insertion_goal(ctx,category,bag,slot+n,height_offset=.06*n)
        ctx.event('insertion_target',category=category,attempt=n,bag=bag.tolist(),goal=goal.tolist(),height_offset=.06*n)
        on_floor=bag[2,3]<.55
        candidates=[]
        if on_floor:
            candidates=radial_stances(ctx,bag[:2,3],(.62,.75,.52))
            candidates.sort(key=lambda p:np.linalg.norm(np.array(p[:2])-ctx.est.base()[:2]))
        else:
            candidates=surface_stances(ctx,bag[:3,3])
            candidates.sort(key=lambda p:np.linalg.norm(np.asarray(p[:2])-bag[:2,3])+.10*np.linalg.norm(np.asarray(p[:2])-ctx.est.base()[:2]))
        for stance in [ctx.est.base()]+candidates[:(24 if on_floor else 64)]:
            yield ctx.action()
            inv_relation=np.linalg.inv(ctx.held[2])
            grip_yaw=float(np.arctan2(inv_relation[1,0],inv_relation[0,0]))
            yaw=release_yaw(ctx,category,bag,stance[2]-grip_yaw+(0.,.65,-.65)[n])
            if yaw is None:continue
            goal=insertion_goal(ctx,category,bag,slot+n,height_offset=.06*n,world_yaw=yaw)
            tip=np.linalg.inv(pose2(*stance))@goal@np.linalg.inv(ctx.held[2])
            found=release_ik(ctx,tip,base=pose2(*stance))
            if not found:
                ctx.event('insertion_plan',category=category,slot=slot+n,ok=False,
                          stance=list(stance),reason='no collision-free release IK')
                continue
            status=yield from carried_drive(ctx,stance,max_steps=900,keepout=((bag[:2,3],.50),) if on_floor else ())
            if status!='arrived':continue
            if refresh is not None:
                fresh=refresh('tote')
                if fresh is None and observe is not None and view_attempts<2:
                    view_attempts+=1
                    fresh=yield from observe(bag)
                if fresh is None:
                    ctx.event('recipient_refresh',ok=False,reason='not observed before release');continue
                ctx.event('recipient_refresh',ok=True,shift=float(np.linalg.norm(fresh[:3,3]-bag[:3,3])))
                bag=fresh;ctx.guard=bag_guards(bag)
                yaw=release_yaw(ctx,category,bag,ctx.est.base()[2]-grip_yaw+(0.,.65,-.65)[n])
                if yaw is None:continue
                goal=insertion_goal(ctx,category,bag,slot+n,height_offset=.06*n,world_yaw=yaw)
            ctx.event('insertion_release_target',category=category,bag=bag.tolist(),goal=goal.tolist(),yaw=float(yaw))
            if (yield from place_object(ctx,goal,label='insert')):return True
    return False
