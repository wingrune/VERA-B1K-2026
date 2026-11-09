"""Partial-Q-first book packing with observation-driven choices and bounded retries."""
import time
import numpy as np
from art_skills import tuck,tucked,pinch,carry,place_object,OrientedBoxes,transform,upright,leave_surface
from art_pipeline import localize
from carried_navigation import checked_carried_drive
from manip import pose2,hold
from book_grasps import plan

# After the adjacent first pair, prefer the smaller lower-shelf book before large flat volumes.
BOOKS=('book1','book2','book6','book4','book5','book3')

def bounded(ctx,script,actions,label):
    for _ in range(actions):
        try:yield next(script)
        except StopIteration as done:return done.value
    script.close();ctx.event('stage_budget',stage=label,actions=actions);yield from hold(ctx,6);return None

def box_guards(ctx,box):
    # Exact object-local collision meshes are retained; their AABBs enclose each wall separately.
    from pathlib import Path
    data=np.load(Path(__file__).parent/'priors/box/collision.npz');boxes=[]
    for name in data.files:
        if name.endswith('|v'):
            pts=data[name];boxes.append(np.r_[pts.min(0),pts.max(0)])
    return OrientedBoxes(boxes,box)

def update_box(ctx,est):
    box=est.object('box',max_age=20000)
    if box is not None:ctx.guard=box_guards(ctx,box)
    return box

def scan(ctx,est,categories,limit=180):
    found={};baseline=ctx.base_matrix();q=ctx.q.copy()
    for n in range(limit):
        if n%35==0:
            for category in categories:
                if category in found:continue
                obj=est.refresh(category)
                if obj is not None:
                    found[category]=obj;ctx.event('observed_object',category=category,pose=obj.tolist())
            update_box(ctx,est)
            if len(found)==len(categories):break
        base=np.asarray(est.base());base[2]+=.4/30.
        if not ctx.validator(base=pose2(*base),relax_from=q,baseline_base=baseline)(ctx.q,ctx.body.fk(ctx.q)[0]):break
        yield ctx.action(base=(0.,0.,.4))
    yield from hold(ctx,8);return found

def cabinet_views(ctx,cabinet,point=None):
    lo=np.asarray(cabinet['aabb_lo']);hi=np.asarray(cabinet['aabb_hi']);centre=(lo+hi)/2
    target=centre if point is None else np.asarray(point);out=[]
    for distance in (.65,.80,.50,.95):
        for offset in (0.,.18,-.18,.35,-.35):
            for axis in (int(np.argmin((hi-lo)[:2])),):
                for sign in (-1.,1.):
                    xy=target[:2].copy();xy[axis]=(lo[axis] if sign<0 else hi[axis])+sign*distance
                    xy[1-axis]=np.clip(xy[1-axis]+offset,lo[1-axis]-.10,hi[1-axis]+.10)
                    yaw=float(np.arctan2(*(target[:2]-xy)[::-1]));stance=(*xy,yaw)
                    if not ctx.scene.base_free(xy):continue
                    out.append(stance)
    return sorted(out,key=lambda s:np.linalg.norm(np.asarray(s[:2])-ctx.est.base()[:2])+.4*np.linalg.norm(np.asarray(s[:2])-target[:2]))

def find_box(ctx,est):
    yield from scan(ctx,est,('box',),360)
    box=update_box(ctx,est)
    if box is not None:return box
    floor=ctx.scene.meta['scope']['floor.n.01_1'];lo=np.asarray(floor['aabb_lo']);hi=np.asarray(floor['aabb_hi'])
    views=[]
    for x in np.linspace(lo[0]+.7,hi[0]-.7,7):
        for y in np.linspace(lo[1]+.7,hi[1]-.7,5):
            if ctx.scene.base_free((x,y)):views.append((x,y))
    views.sort(key=lambda v:np.linalg.norm(np.asarray(v)-est.base()[:2]));visited=[]
    for xy in views:
        if len(visited)>=5:break
        if any(np.linalg.norm(np.asarray(xy)-v)<1.3 for v in visited):continue
        visited.append(np.array(xy));status=yield from checked_carried_drive(ctx,(*xy,est.base()[2]),max_steps=800)
        if status!='arrived':continue
        yield from scan(ctx,est,('box',),360);box=update_box(ctx,est)
        if box is not None:return box
    return None

def pick_book(ctx,est,category):
    cabinet=ctx.scene.meta['bookcases'][ctx.scene.meta['bookcase_assignment'][category]]
    ctx.scene.meta['target_surface']=cabinet
    target=est.object(category,max_age=20000);attempts=0
    views=cabinet_views(ctx,cabinet,None if target is None else target[:3,3])
    for stance in views[:14]:
        yield ctx.action();update_box(ctx,est)
        if (yield from checked_carried_drive(ctx,stance,max_steps=900))!='arrived':continue
        yield from tuck(ctx);yield from hold(ctx,12)
        target=est.refresh(category)
        if target is None:
            ctx.event('book_unseen',category=category,stance=list(stance),detail=est.ob.debug.get(category));continue
        # Search the observed book's reachable approach before committing the chassis.
        lo=np.asarray(cabinet['aabb_lo']);hi=np.asarray(cabinet['aabb_hi']);axis=int(np.argmin((hi-lo)[:2]));front=np.zeros(2);front[axis]=np.sign(est.base()[axis]-(lo[axis]+hi[axis])/2)
        heading=float(np.arctan2(-front[1],-front[0]));tangent=np.array([-front[1],front[0]])
        nominal=tucked(ctx);selected=None;search_start=time.monotonic()
        for distance,offset in [(d,o) for d in (.55,.60,.70,.50,.80) for o in (0.,-.12,.12)]:
            if time.monotonic()-search_start>30.:break
            xy=target[:2,3]+front*distance+tangent*offset
            if not ctx.scene.base_free(xy):continue
            world=pose2(*xy,heading)
            for arm in ('left','right'):
                yield ctx.action();ctx.arm=arm
                if not ctx.validator(base=world)(nominal,ctx.body.fk(nominal)[0]):continue
                candidate=plan(ctx,category,target,base=world,q=nominal,count=6,candidate_limit=32,seconds=3.)
                if candidate is not None:selected=(xy,arm);break
            if selected is not None:break
        if selected is None:
            ctx.event('book_no_reachable_stance',category=category);continue
        xy,preferred_arm=selected
        ctx.event('book_reachable_stance',category=category,stance=[*xy,heading],arm=preferred_arm)
        if (yield from checked_carried_drive(ctx,(*xy,heading),max_steps=450))!='arrived':continue
        yield from hold(ctx,12);fresh=est.refresh(category)
        if fresh is not None:target=fresh
        est.remember(category,target,source='current_RGBD');ctx.event('book_target',category=category,pose=target.tolist())
        for arm in (preferred_arm,'right' if preferred_arm=='left' else 'left'):
            yield ctx.action();ctx.arm=arm
            planned=plan(ctx,category,target,count=10,candidate_limit=40,seconds=15.)
            if planned is None:continue
            attempts+=1
            if (yield from pinch(ctx,category,target,refresh=est.refresh,preferred_candidate=planned['candidate'])):return True
            yield from tuck(ctx)
            if attempts>=3:return False
    return False

def deliver(ctx,est,category,delivered):
    began=time.monotonic();box=update_box(ctx,est)
    if box is None or ctx.held is None:return False
    if not (yield from leave_surface(ctx)):return False
    if not (yield from carry(ctx)):return False
    bounds=ctx.models['box']['bounds'];radius=.29+np.linalg.norm(np.maximum(abs(bounds[0,:2]),abs(bounds[1,:2])))+.035
    stances=[]
    for r in (radius+.025,radius+.10,radius+.18):
        for angle in np.linspace(-np.pi,np.pi,24,endpoint=False):
            xy=box[:2,3]+r*np.array([np.cos(angle),np.sin(angle)])
            if not ctx.scene.base_free(xy):continue
            stances.append((*xy,float(angle+np.pi)))
    stances.sort(key=lambda s:np.linalg.norm(np.asarray(s[:2])-est.base()[:2])+3.*abs(np.linalg.norm(np.asarray(s[:2])-box[:2,3])-(radius+.025)))
    for stance in stances[:12]:
        status=yield from checked_carried_drive(ctx,stance,max_steps=1800,keepout=((box[:2,3],radius),))
        if status!='arrived':continue
        yield from hold(ctx,10);fresh=est.refresh('box')
        if fresh is not None:box=fresh
        ctx.guard=box_guards(ctx,box)
        item=ctx.models[category]['bounds'];centre=item.mean(0);size=item[1]-item[0]
        # Spread books among four locations to keep their centres below the shallow rim.
        offsets=[(0.,0.),(-.10,-.10),(.10,.10),(-.10,.10),(.10,-.10)]
        offsets=offsets[len(delivered)%4:4]+offsets[:len(delivered)%4]+offsets[4:]
        for xy in offsets[:1]:
            for yaw in (0.,np.pi/2,np.pi,-np.pi/2):
                rotation=box[:3,:3]@upright(yaw);half=abs(upright(yaw))@size/2
                local=np.clip(np.array(xy),bounds[0,:2]+half[:2]+.025,bounds[1,:2]-half[:2]-.025)
                for clearance in (.16,.26):
                    if time.monotonic()-began>240.:return False
                    point=(box@np.r_[local,bounds[1,2]+clearance,1.])[:3]
                    goal=transform(rotation,point-rotation@centre)
                    yield ctx.action()
                    if (yield from place_object(ctx,goal,label='book_release')):
                        yield from hold(ctx,45);est.clear(category);delivered.append(category)
                        ctx.event('delivery_belief',category=category,delivered=list(delivered),slot=local.tolist())
                        yield from tuck(ctx);return True
    return False

def task(ctx,est):
    if not (yield from localize(ctx,est)):return 'localization_failed'
    yield from tuck(ctx)
    box=yield from bounded(ctx,find_box(ctx,est),4500,'find_box')
    if box is None:return 'box_not_observed'
    delivered=[]
    for category in BOOKS:
        if ctx.steps>32500:break
        ctx.event('phase',phase='pick_book',category=category,delivered=delivered)
        ok=yield from bounded(ctx,pick_book(ctx,est,category),4500,'pick_'+category)
        if not ok:
            if ctx.held is not None:return 'retained_payload_after_pick_failure'
            continue
        if not (yield from bounded(ctx,deliver(ctx,est,category,delivered),5000,'deliver_'+category)):
            if ctx.held is not None:return 'retained_payload_after_delivery_failure'
    yield from hold(ctx,120);ctx.event('books_complete',delivered=delivered)
    return 'bounded_task_complete'
