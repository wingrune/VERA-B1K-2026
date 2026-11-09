"""Reachable observed/unknown boundaries for room-scale RGB-D exploration."""
import numpy as np
from scipy.ndimage import label,binary_dilation,uniform_filter


def frontier_proposals(depth,odom,visited):
    known=(depth.log_odds[:,:,:8]<0).any(axis=2)
    occupied=depth.occupancy();free=known&~occupied
    cell=depth.ij(odom[:2]);xx,yy=np.ogrid[:depth.n,:depth.n]
    footprint=(xx-cell[0])**2+(yy-cell[1])**2<(depth.radius/depth.res)**2
    free|=footprint
    labels,_=label(free)
    component=labels[tuple(cell)]
    if component==0:return []
    reachable=labels==component
    frontier=reachable&binary_dilation(~known,iterations=2)&~footprint
    groups,count=label(frontier)
    information=uniform_filter((~known).astype(float),size=35)
    proposals=[]
    for group in range(1,count+1):
        cells=np.argwhere(groups==group)
        if len(cells)<6:continue
        points=depth.xy(cells);distance=np.linalg.norm(points-odom[:2],axis=1)
        eligible=(distance>.75)&(distance<10.)
        for old in visited:
            eligible&=np.linalg.norm(points-old,axis=1)>.55
        candidates=np.where(eligible)[0]
        if not len(candidates):continue
        gains=information[cells[candidates,0],cells[candidates,1]]
        # Prefer useful boundaries without restricting goals to a local radius.
        utility=3*gains-.08*distance[candidates]+.08*np.log1p(len(cells))
        for index in np.argsort(utility)[-2:]:
            k=candidates[index];proposals.append((float(utility[index]),points[k]))
    return proposals
