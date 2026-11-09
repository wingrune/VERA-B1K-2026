"""Conservative collision guards for currently visible, unrecognized kitchen floor objects."""
import numpy as np
from scipy.ndimage import binary_dilation,label
from localizer import camera_points
from compost_perception import CAMERAS
from kinematics import from_proprio

def observe(observer,base):
    points=[]
    for link,slot,focal in CAMERAS.values():
        depth=observer.data[f'robot_r1::robot_r1:{link}:Camera:0::depth_linear']
        rel=observer.data['robot_r1::cam_rel_poses'].reshape(3,7)[slot]
        points.append(camera_points(depth,rel,focal,stride=9,max_range=3.))
    body=np.concatenate(points);world=body@base[:3,:3].T+base[:3,3]
    room=observer.scene.meta['scope']['floor.n.01_1'];lo=np.array(room['aabb_lo']);hi=np.array(room['aabb_hi'])
    keep=(world[:,2]>.10)&(world[:,2]<.82)&np.all(world[:,:2]>lo[:2]-.12,axis=1)&np.all(world[:,:2]<hi[:2]+.12,axis=1)
    body=body[keep];world=world[keep]
    if len(world)<15:return []
    full=from_proprio(observer.data['robot_r1::proprio'],observer.body.names);poses=observer.body.fk(full)[0]
    centres=observer.collision.centers(poses);radii=observer.collision.radii+.02
    if observer.payload is not None:
        link,pts,radius=observer.payload;t=poses[link]
        centres=np.vstack([centres,pts@t[:3,:3].T+t[:3,3]]);radii=np.r_[radii,np.full(len(pts),radius+.025)]
    keep=np.ones(len(world),bool)
    for centre,radius in zip(centres,radii):keep&=np.sum((body-centre)**2,axis=1)>radius**2
    world=world[keep]
    if len(world)<15:return []
    gap=observer.scene.contact_clearance(world,np.zeros(len(world)),ignore_below=np.full(len(world),.06))
    world=world[gap>.055]
    if len(world)<15:return []
    origin=np.r_[lo[:2]-.2,0.];resolution=.05;shape=np.ceil((np.r_[hi[:2]+.2,.9]-origin)/resolution).astype(int)+1
    cell=np.floor((world-origin)/resolution).astype(int);valid=np.all(cell>=0,axis=1)&np.all(cell<shape,axis=1);cell=cell[valid];world=world[valid]
    occupied=np.zeros(shape,bool);occupied[tuple(cell.T)]=True
    labels,_=label(binary_dilation(occupied));ids=labels[tuple(cell.T)];boxes=[]
    for item in np.unique(ids):
        pts=world[ids==item]
        if len(pts)<15:continue
        low=pts.min(0);high=pts.max(0);span=high-low
        if max(span[:2])>.65 or span[2]<.07:continue
        # Enclose measured surfaces plus pose/depth uncertainty; identity is not assumed.
        boxes.append(np.r_[low-[.045,.045,.035],high+[.045,.045,.035]])
    return boxes
