"""Bin-first sequencing and counter hints from onboard RGB-D only."""
import cv2
import numpy as np
from perception import pointcloud


class BinSearch:
    def __init__(self):
        self.confirmed=False
        self.samples=0
        self.last_seen=-1000
        self.last_center=None
        self.counter=None
        self.counter_origin=None
        self.counter_from_food=False
        self.bin_pose=None
        self.counter_revision=0
        self.fridge=None;self.fridge_samples=0;self.fridge_last=None;self.fridge_seen=-1000

    def observe_bin(self,step,seen,pose,center):
        if self.confirmed:return True
        if step-seen>54:
            self.samples=0
            return False
        if seen==self.last_seen:return False
        consistent=(self.last_center is not None and seen-self.last_seen<=54
                    and np.linalg.norm(center-self.last_center)<.06)
        self.samples=self.samples+1 if consistent else 1
        self.last_seen=seen;self.last_center=center.copy()
        if self.samples>=3:
            self.confirmed=True;self.bin_pose=pose.copy()
        return self.confirmed

    def observe_counter(self,camera,level,base,odom,food_centers):
        # Use a verified fruit's supporting area when available. Otherwise
        # find a broad horizontal worktop, rather than inferring a world pose.
        if food_centers:
            center=np.mean(food_centers,axis=0)
        elif self.counter is None:
            points,_,pixels=pointcloud(camera)
            leveled=points@level[:3,:3].T+level[:3,3]
            grid=np.full(camera['depth'].shape+(3,),np.nan)
            grid[pixels[:,1],pixels[:,0]]=leveled
            normals=np.cross(grid[1:,:-1]-grid[:-1,:-1],grid[:-1,1:]-grid[:-1,:-1])
            horizontal=np.zeros(camera['depth'].shape,np.uint8)
            horizontal[:-1,:-1]=(abs(normals[...,2])>.95*np.linalg.norm(normals,axis=-1))
            horizontal[pixels[:,1],pixels[:,0]] &= ((leveled[:,2]>.65)&(leveled[:,2]<1.25)).astype(np.uint8)
            n,labels,stats,_=cv2.connectedComponentsWithStats(horizontal)
            proposals=[]
            for i in range(1,n):
                if stats[i,cv2.CC_STAT_AREA]<150:continue
                patch,_,_=pointcloud(camera,labels==i)
                patch_level=patch@level[:3,:3].T+level[:3,3]
                if np.min(np.ptp(patch_level[:,:2],axis=0))<.25:continue
                if np.ptp(patch_level[:,2])>.04:continue
                world=patch@base[:3,:3].T+base[:3,3]
                proposals.append((len(patch),np.median(world,axis=0)))
            if not proposals:return
            center=max(proposals,key=lambda p:p[0])[1]
        else:return
        if self.counter is None or (food_centers and (not self.counter_from_food or np.linalg.norm(center[:2]-self.counter[:2])<1.5)):
            changed=self.counter is None or (food_centers and not self.counter_from_food) or np.linalg.norm(center[:2]-self.counter[:2])>.30
            if changed:self.counter_revision+=1
            if food_centers and not self.counter_from_food:self.counter_origin=odom[:2].copy()
            self.counter=center.copy()
            self.counter_from_food=bool(food_centers)
            if self.counter_origin is None:self.counter_origin=odom[:2].copy()

    def observe_fridge(self,camera,level,base,step):
        if not self.counter_from_food:return
        from appliance_landmark import fridge_candidates
        candidates=fridge_candidates(camera,level,base,self.counter,self.counter_origin)
        if not candidates:
            if step-self.fridge_seen>54:self.fridge_samples=0
            return
        position=candidates[0]['position']
        consistent=self.fridge_last is not None and step-self.fridge_seen<=54 and np.linalg.norm(position-self.fridge_last)<.15
        self.fridge_samples=self.fridge_samples+1 if consistent else 1
        self.fridge_last=position.copy();self.fridge_seen=step
        if self.fridge_samples>=3:
            if self.fridge is None or np.linalg.norm(position-self.fridge)>.30:self.counter_revision+=1
            self.fridge=position.copy()

    def viewpoint_bonus(self,goal):
        if self.counter is None:return 0.
        normal=self.counter[:2]-self.counter_origin
        normal/=max(np.linalg.norm(normal),1e-6)
        offset=goal-self.counter[:2]
        # Prefer the opposite side of the observed counter, with exploration
        # still available when collision checking rejects those viewpoints.
        return .8*np.clip(offset@normal,-1.,1.)-.35*np.linalg.norm(offset)
