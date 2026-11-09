"""Task87 a023: t01 motion reuse, observed floor cans, stationary bucket.

Runtime inputs are RGB-D, proprioception, relative camera poses only.
Static map is exported from the training template, excluding all movable task objects.
"""
import json, os
from pathlib import Path
import numpy as np
from scene_map import SceneMap
from manip import Context, hold
from observer import Observer
from trash_observed import ObservedEstimator, Runtime, localize, survey
import trash_observed
from trash_skills import tuck
from trash_pipeline import fetch, deliver

KEYS = ['robot_r1::proprio', 'robot_r1::cam_rel_poses'] + [f'robot_r1::robot_r1:{link}:Camera:0::{kind}' for link in ('zed_link','left_realsense_link','right_realsense_link') for kind in ('rgb','depth_linear')]

def pipeline(rt):
    ctx, est = rt.ctx, rt.est
    if not (yield from localize(rt)): return 'not_localized'
    yield from tuck(ctx)
    yield from survey(rt,'start')
    if est.bin() is None: yield from survey(rt,'bin')
    if est.bin() is None: return 'no_bucket'
    used=[]; attempts={}; delivered=0; refreshed=set()
    for cycle in range(14):
        options=[k for k in est.known() if attempts.get(k,0)<2]
        if not options:
            before=np.asarray(est.base()[:2]).copy()
            rt.survey_ignore={j for j in est.known() if attempts.get(j,0)>=2}
            yield from survey(rt,'cans')
            rt.survey_ignore=set()
            # A route may be blocked only by the current turning clearance.
            # A new lookout is a new reachable starting pose, not a failed grasp.
            if np.linalg.norm(np.asarray(est.base()[:2])-before)>.5:
                for j in est.known():
                    if j not in refreshed and attempts.get(j,0)>=2:
                        attempts[j]=0;refreshed.add(j)
                        ctx.event('retry_from_new_view',track=j)
            options=[k for k in est.known() if attempts.get(k,0)<2]
        if not options: return f'no_visible_cans_after_{delivered}_releases'
        k=min(options,key=lambda k:np.linalg.norm(est.soda(k)-np.asarray(est.base()[:2])))
        attempts[k]=attempts.get(k,0)+1
        arm='left' if attempts[k]==1 else 'right'
        how=yield from fetch(ctx,est,k,arm)
        ctx.event('fetch_complete',track=k,arm=arm,how=how)
        if not how: continue
        gone=yield from deliver(ctx,est,[(arm,how)],used)
        delivered+=len(gone)
        ctx.event('release_complete',count=delivered,confirmed_inside=False)
        if not gone:return 'holding_can_no_delivery'
        yield from hold(ctx,30)
    return 'bounded_cycles_complete'

class Policy:
    def __init__(self):
        self.root=Path(__file__).parent
        self.output=Path(os.environ.get('POLICY_OUTPUT','.')); self.output.mkdir(parents=True,exist_ok=True)
        self.scene=SceneMap(self.root/'priors/hotel_map.npz')
        # Generic map-coverage lookouts, no instance coordinates.
        floor=self.scene.meta['start_floor'];lo=np.array(floor['lo']);hi=np.array(floor['hi'])
        gx,gy=np.meshgrid(np.arange(lo[0]+.5,hi[0]-.4,.6),np.arange(lo[1]+.5,hi[1]-.4,.6),indexing='ij')
        spots=[xy for xy in np.c_[gx.ravel(),gy.ravel()] if self.scene.base_free(xy,margin=.08)]
        chosen=[]
        if spots:
            chosen=[min(spots,key=lambda xy:np.linalg.norm(xy-(lo+hi)/2))]
            for _ in range(min(6,len(spots))-1):
                chosen.append(max(spots,key=lambda xy:min(np.linalg.norm(xy-c) for c in chosen)))
        trash_observed.LOOKOUTS={k:tuple(tuple(float(x) for x in xy) for xy in chosen) for k in ('cans','bin')}
        self.reset()
    def reset(self):
        self.observer=Observer(self.root,self.scene)
        self.estimator=ObservedEstimator(self.observer,self.scene)
        self.context=Context(self.root,self.scene,self.estimator,log=self._log)
        self.runtime=Runtime(self.context,self.observer,self.estimator,pipeline)
    def set_action_dim(self,n):assert n==23
    def _log(self,*args):
        if args and args[0]=='EVENT':
            with (self.output/'trace.jsonl').open('a') as f:f.write(json.dumps(args[1],default=str)+'\n')
        print(*args,flush=True)
    def action(self,data):
        data={k:np.asarray(data[k]) for k in KEYS}
        action=np.asarray(self.runtime.step(data),dtype=np.float32)
        if action.shape!=(23,) or not np.isfinite(action).all():raise ValueError('Invalid action')
        return action
