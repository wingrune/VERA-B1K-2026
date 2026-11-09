"""Multi-instance RGB-D localization with canonical, object-local training priors."""
import numpy as np
import time
from object_perception import ObjectPerception
from perception import pointcloud

class RosePerception(ObjectPerception):
    def __init__(self, folder):
        super().__init__(folder,['rose','vase','coffee_table'])
        self.world_transform=lambda:np.eye(4)
        self.active=None
        self.predicted=None
        self.carried_blossom=False
        self.released=False
        self.released_vase_body=None
        self.unheld_predicted=None
        self.unheld_navigation=False
        self.release_diagnostics={}
        self.vase_predicted=None
        self.vase_robot=None
        self.vase_robot_poses=None
        self.excluded=[]
        self.protected_parent=lambda:None
        self.candidates=[]
        self.diagnostics={}
        self.suppress_rose_fit=False
        self.identity_world=None

    def all_roses(self,cameras):
        masked={k:dict(c,depth=c['depth'].copy()) for k,c in cameras.items()}
        found=[]
        for _ in range(3):
            result=super().fit('rose',masked)
            if result is None:break
            pose,stats=result
            found.append((pose,stats))
            lo,hi=self.models['rose']['bounds']
            for c in masked.values():
                points,_,uv=pointcloud(c)
                local=(points-pose[:3,3])@pose[:3,:3]
                inside=((local>=lo-.015)&(local<=hi+.015)).all(axis=1)
                c['depth'][uv[inside,1],uv[inside,0]]=np.nan
        if not found:
            from rose_geometry import rose_candidates
            found=rose_candidates(self.models['rose'],cameras,self.level)
        self.candidates=found
        return found

    def fit(self,category,cameras,features=None):
        if category=='rose' and self.suppress_rose_fit:return None
        if category!='rose':
            found=super().fit(category,cameras,features)
            if found is not None and category=='vase':
                from rose_geometry import visible_model_support
                support=visible_model_support(self.models['vase'],cameras,found[0])
                if support['matched_fraction']<.65:found=None
                else:found[1]['visible_support']=support
            if found is None and category=='vase':
                from rose_geometry import vase_fit
                found=vase_fit(self.models['vase'],cameras,self.level)
            if found is None and category=='vase' and self.vase_predicted is not None:
                from vase_tracking import guided_vase_fit
                found=guided_vase_fit(self.models['vase'],cameras,self.vase_predicted,self.vase_robot,self.vase_robot_poses)
            if found is None and category=='coffee_table':
                from rose_geometry import table_fit
                found=table_fit(self.models['coffee_table'],cameras,self.level)
            return found
        if self.released:
            from rose_geometry import guided_rose_fit
            from released_pose_fit import global_release_fit,certify_released_fit
            if self.predicted is None or self.released_vase_body is None:
                self.candidates=[];self.release_diagnostics={'reason':'missing_released_parent_or_prediction'}
                return None
            started=time.monotonic();fit=None;rejections=[]
            for partial in (False,True):
                if time.monotonic()-started>=3.:break
                legacy=guided_rose_fit(self.models['rose'],cameras,self.predicted,translation_limit=.075,red_min=.4,green_min=.50 if partial else .35,multistart=partial,visible_only=partial)
                fit,debug=certify_released_fit(self.models['rose'],cameras,self.predicted,self.models['vase'],self.released_vase_body,legacy)
                if fit is not None:break
                rejections.append(debug)
            if fit is None:
                fit,debug=global_release_fit(self.models['rose'],cameras,self.predicted,self.models['vase'],self.released_vase_body,seconds=3.-(time.monotonic()-started))
                self.release_diagnostics=dict(debug,legacy_rejections=rejections)
            else:self.release_diagnostics={'reason':'certified_legacy','legacy_rejections':rejections}
            self.candidates=[] if fit is None else [fit]
            if fit is not None:
                fit=(fit[0],dict(fit[1],released_search_total_seconds=time.monotonic()-started,legacy_rejection_count=len(rejections)))
                self.active=(self.world_transform()@fit[0])[:3,3]
            return fit
        if self.predicted is not None:
            from rose_geometry import guided_rose_fit
            fit=guided_rose_fit(self.models['rose'],cameras,self.predicted,translation_limit=.14 if self.carried_blossom else .075,red_min=.4,green_min=.50 if self.carried_blossom else .35)
            if fit is None and self.carried_blossom:
                fit=guided_rose_fit(self.models['rose'],cameras,self.predicted,translation_limit=.14,red_min=.4,green_min=.50,multistart=True,visible_only=False)
            if fit is None and (self.carried_blossom or self.released):
                fit=guided_rose_fit(self.models['rose'],cameras,self.predicted,translation_limit=.075 if self.released else .14,red_min=.4,green_min=.50,multistart=True,visible_only=True)
            self.candidates=[] if fit is None else [fit]
            if fit is not None:self.active=(self.world_transform()@fit[0])[:3,3]
            return fit
        base=self.world_transform();available=[]
        for pose,stats in self.all_roses(cameras):
            world=base@pose
            if self.identity_world is not None and np.linalg.norm(world[:3,3]-self.identity_world[:3,3])>.060:continue
            if self.excluded_candidate(world):continue
            distance=0. if self.active is None else np.linalg.norm(world[:3,3]-self.active)
            if self.active is not None and distance>.30:continue
            available.append((distance-.0001*stats.get('inliers',0),pose,stats,world))
        self.diagnostics=dict(instances_visible=len(self.candidates),available=len(available))
        if not available:
            if self.unheld_predicted is None:return None
            from rose_geometry import guided_rose_fit
            tracked=guided_rose_fit(self.models['rose'],cameras,self.unheld_predicted,translation_limit=.075,red_min=.4,green_min=.35)
            if tracked is None:return None
            pose,stats=tracked;world=base@pose
            if self.unheld_navigation:
                from rose_geometry import visible_model_support
                support=visible_model_support(self.models['rose'],cameras,pose)
                if stats.get('observed_red_points',0)<20 or support['matched_fraction']<.65 or support['unoccluded_fraction']<=.50:return None
                stats=dict(stats,visible_support=support)
            if self.excluded_candidate(world):return None
            self.active=world[:3,3].copy()
            return pose,dict(stats,method='bounded_unheld_visual_rose_tracking',**self.diagnostics)
        _,pose,stats,world=min(available,key=lambda x:x[0])
        self.active=world[:3,3].copy()
        return pose,dict(stats,**self.diagnostics)

    def excluded_candidate(self, world):
        """Protect the observed vase neighborhood after an issued release.

        This filter only limits acquisition of unheld roses. It does not assert
        placement success, update observation timestamps, or modify held fits.
        """
        point = np.asarray(world)[:3, 3]
        if any(np.linalg.norm(point - p) < .14 for p in self.excluded):
            return True
        parent = self.protected_parent()
        if not self.excluded or parent is None:
            return False
        parent = np.asarray(parent)
        if parent.shape != (4, 4) or not np.isfinite(parent).all():
            return False
        lo, hi = np.asarray(self.models['vase']['bounds'])
        # Use the whole prior envelope with a bounded localization allowance.
        # The vertical allowance covers protruding blossoms, while table-level
        # roses outside the vessel neighborhood remain available.
        local = (point - parent[:3, 3]) @ parent[:3, :3]
        radius = max(np.linalg.norm(c) for c in
                     ([lo[0], lo[1]], [lo[0], hi[1]],
                      [hi[0], lo[1]], [hi[0], hi[1]])) + .075
        return bool(np.linalg.norm(local[:2]) <= radius
                    and lo[2] - .025 <= local[2] <= hi[2] + .30)
