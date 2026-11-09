"""Canonical RGB-D feature identity or verified supported fruit geometry."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from object_perception import ObjectPerception
from perception import pointcloud

class FoodPerception(ObjectPerception):
    def __init__(self,*args):
        super().__init__(*args)
        self.confirmations={};self.world_transform=None;self.carried_prediction=None;self.provisional={}

    def fit(self,category,cameras,features=None):
        if getattr(self,'held',None)==category and self.carried_prediction is not None:
            predicted=self.carried_prediction(category)
            if predicted is not None:
                from held_tool_tracking import refine_held
                found=refine_held(self.models[category],cameras,predicted,getattr(self,'robot_mask',None))
                if found is not None:return found
        found=super().fit(category,cameras,features)
        if found is not None:return found
        if category=='trash_can':
            from bin_perception import fit_bin
            prediction=getattr(self,'bin_prediction',None)
            proposal,diagnostics=fit_bin(self.models[category],cameras,self.level,None if prediction is None else prediction())
            self.bin_diagnostics=diagnostics
        else:
            proposal=self.supported_proposal(category,cameras)
        if proposal is None:
            self.confirmations.pop(category,None);self.provisional.pop(category,None)
            return None
        pose,stats=proposal
        center=pose[:3,3]
        if self.world_transform is not None:center=(self.world_transform()@pose)[:3,3]
        old=self.confirmations.get(category)
        count=old[1]+1 if old is not None and np.linalg.norm(center-old[0])<.055 else 1
        self.confirmations[category]=(center.copy(),count)
        self.provisional[category]=(pose.copy(),stats.copy(),count)
        if count<3:return None
        stats['consistent_views']=count
        return pose,stats

    def supported_proposal(self,category,cameras):
        self.diagnostics=getattr(self,'diagnostics',{})
        diag={'candidates':[]}
        self.diagnostics[category]=diag
        if category not in ('half_banana','half_pomegranate'):return None
        model=self.models[category];lo,hi=model['bounds'];ext=hi-lo
        ref=cv2.cvtColor(model['rgb'][None].astype(np.uint8),cv2.COLOR_RGB2HSV)[0]
        ref=ref[(ref[:,1]>100)&(ref[:,2]>70)]
        if len(ref)<30:return None
        hist=np.bincount((ref[:,0]//6).astype(int),minlength=30)
        diag['reference_colored_pixels']=len(ref)
        hues=np.where(hist>hist.max()*.45)[0]*6+3
        diag['hues']=hues.tolist()
        model_tree=cKDTree(model['xyz'][::2]);choices=[]
        for name,camera in cameras.items():
            hsv=cv2.cvtColor(camera['rgb'],cv2.COLOR_RGB2HSV)
            diff=abs(hsv[...,0,None].astype(float)-hues)
            mask=(np.minimum(diff,180-diff).min(-1)<7)&(hsv[...,1]>100)&(hsv[...,2]>70)
            all_full,_,pixels=pointcloud(camera)
            leveled_full=all_full@self.level[:3,:3].T+self.level[:3,3]
            all_level=leveled_full[::2]
            height_image=np.full(mask.shape,np.nan)
            height_image[pixels[:,1],pixels[:,0]]=leveled_full[:,2]
            grid=np.full(mask.shape+(3,),np.nan);grid[pixels[:,1],pixels[:,0]]=leveled_full
            normals=np.cross(grid[1:,:-1]-grid[:-1,:-1],grid[:-1,1:]-grid[:-1,:-1])
            lengths=np.linalg.norm(normals,axis=-1)
            horizontal=np.zeros(mask.shape,bool);horizontal[:-1,:-1]=abs(normals[...,2])>lengths*.95
            valid_normal=horizontal[pixels[:,1],pixels[:,0]]
            candidates=leveled_full[(leveled_full[:,2]>.60)&(leveled_full[:,2]<1.25)&valid_normal]
            plane_heights=[]
            if len(candidates):
                height_bins=np.round(candidates[:,2]/.004).astype(int)
                values,counts=np.unique(height_bins,return_counts=True)
                for index in np.argsort(counts)[::-1][:12]:
                    plane=values[index]*.004
                    inliers=candidates[abs(candidates[:,2]-plane)<.004]
                    if len(inliers)>150 and np.min(np.ptp(inliers[:,:2],axis=0))>.25 and not any(abs(plane-h)<.03 for h in plane_heights):
                        plane_heights.append(plane)
            diag.setdefault('planes',{})[name]=plane_heights
            component_sets=[]
            for plane in plane_heights:
                plane_mask=mask&(height_image>plane+.008)&(height_image<plane+ext[2]*1.5+.02)
                n,labels,stats,_=cv2.connectedComponentsWithStats(plane_mask.astype(np.uint8))
                component_sets.extend((labels,i,stats,plane) for i in range(1,n))
            for labels,i,stats,plane_hint in component_sets:
                if stats[i,cv2.CC_STAT_AREA]<15:continue
                colored,_,_=pointcloud(camera,labels==i)
                colored=colored@self.level[:3,:3].T+self.level[:3,3]
                if len(colored)<15:continue
                center=np.median(colored,axis=0)
                candidate={'camera':name,'colored_points':len(colored),'center':center.tolist(),'span':np.ptp(colored,axis=0).tolist(),'reason':'height'}
                diag['candidates'].append(candidate)
                if not .60<center[2]<1.25:continue
                candidate['reason']='span'
                span=np.ptp(colored,axis=0)
                if np.linalg.norm(span)>np.linalg.norm(ext)*1.25:continue
                candidate['reason']='support_annulus'
                distance=np.linalg.norm(all_level[:,:2]-center[:2],axis=1)
                annulus=all_level[(distance>max(ext[:2])*.65)&(distance<max(ext[:2])*1.8)&(abs(all_level[:,2]-center[2])<.10)]
                if len(annulus)<30:continue
                z=float(plane_hint)
                support=annulus[abs(annulus[:,2]-z)<.007]
                candidate.update(support_height=z,support_points=len(support),protrusion=float(center[2]-z),reason='support_span')
                if len(support)<25 or np.min(np.ptp(support[:,:2],axis=0))<min(ext[:2])*.8:continue
                candidate['reason']='protrusion'
                if not .008<center[2]-z<ext[2]+.04:continue
                surface=all_level[(distance<max(ext[:2])*.8)&(all_level[:,2]>z+.007)&(all_level[:,2]<z+ext[2]+.02)]
                candidate['reason']='surface_points';candidate['surface_points']=len(surface)
                if len(surface)<20:continue
                observed_span=np.sort(cv2.minAreaRect(surface[:,:2].astype(np.float32))[1])
                expected_span=np.sort(ext[:2])
                candidate['reason']='observed_span';candidate['observed_span']=observed_span.tolist()
                if np.any(observed_span<expected_span*.40) or np.any(observed_span>expected_span*1.3):continue
                surface=surface[::max(1,len(surface)//300)]
                fits=[]
                for yaw in np.linspace(0,2*np.pi,12,endpoint=False):
                    rot=R.from_euler('z',yaw).as_matrix();position=surface.mean(0)-rot@model['xyz'].mean(0);position[2]=z-lo[2]
                    def residual(x):
                        rot=R.from_euler('z',x[3]).as_matrix()
                        return np.minimum(model_tree.query((surface-x[:3])@rot)[0],.04)
                    initial=np.r_[position,yaw]
                    lower=initial-np.array([.035,.035,.005,.40]);upper=initial+np.array([.035,.035,.005,.40])
                    fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=25,diff_step=.001)
                    error=float(np.quantile(residual(fit.x),.75));fits.append((error,fit.x))
                error,x=min(fits,key=lambda a:a[0])
                candidate['reason']='residual';candidate['residual']=error
                if error>.010:continue
                candidate['reason']='accepted'
                pose=np.eye(4);pose[:3,:3]=R.from_euler('z',x[3]).as_matrix();pose[:3,3]=x[:3]
                pose=np.linalg.inv(self.level)@pose
                choices.append((error,pose,dict(method='supported_training_color_shape',camera=name,residual=error,
                    points=len(surface),support_points=len(support),support_height=z,protrusion=float(center[2]-z),orientation_ambiguous=True)))
        if choices:
            _,pose,stats=min(choices,key=lambda x:x[0]);return pose,stats
        return None
