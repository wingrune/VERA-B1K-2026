"""Recover a locally inconsistent recipient track only with two-camera agreement."""
import numpy as np
from scipy.spatial import cKDTree

def corroborated_recipient(observer,base,support,prior):
    from art_perception import CAMERAS
    fits=observer.fits('tote',base,support,prior=prior,reacquire=True)
    # One camera proposes a local pose; other cameras independently register and
    # check visible depth around that hypothesis, avoiding a bad old ICP basin.
    for pose,info in list(fits[:3]):
        cameras=tuple(camera for camera in CAMERAS if camera!=info['camera'])
        cross=observer.fits('tote',base,support,cameras=cameras,prior=pose)
        fits.extend((other,evidence) for other,evidence in cross
                    if np.linalg.norm(other[:2,3]-prior[:2,3])<=.20)
    bounds=observer.models['tote']['bounds'];x,y=np.meshgrid(bounds[:,0],bounds[:,1]);corners=np.c_[x.ravel(),y.ravel()]
    def footprint(pose):return corners@pose[:2,:2].T+pose[:2,3]
    def pixel(info):
        link,slot,focal=CAMERAS[info['camera']]
        width=observer.data[f'robot_r1::robot_r1:{link}:Camera:0::depth_linear'].shape[1]
        return info['range']/(focal*width)
    retained=[]
    for pose,info in fits:
        agree=[];own=footprint(pose)
        for alternative,evidence in fits:
            if evidence['camera']==info['camera']:continue
            other=footprint(alternative)
            mismatch=max(cKDTree(own).query(other)[0].max(),cKDTree(other).query(own)[0].max())
            tolerance=info['residual']+evidence['residual']+pixel(info)+pixel(evidence)
            if mismatch<=tolerance:agree.append(evidence['camera'])
        if agree:
            retained.append((pose,dict(info,source='two_camera_local_recipient_reacquisition',corroborating_cameras=sorted(set(agree)))))
    observer.debug['tote_reacquisition']=dict(candidates=len(fits),accepted=len(retained),reason='requires independent camera footprint agreement')
    return retained
