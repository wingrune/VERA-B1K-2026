"""Static URDF forward kinematics and analytical geometric Jacobians (no simulator)."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

class Kinematics:
 def __init__(self,path,names):
  root=ET.parse(path).getroot();self.names=names;self.joints=[]
  for j in root.findall('joint'):
   origin=j.find('origin');t=np.eye(4)
   if origin is not None:
    t[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ');t[:3,:3]=R.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
   axis=j.find('axis');v=np.fromstring(axis.get('xyz','1 0 0'),sep=' ') if axis is not None else np.array([1.,0,0])
   self.joints.append((j.get('name'),j.find('parent').get('link'),j.find('child').get('link'),j.get('type'),t,v))
 def forward(self,q):
  values=dict(zip(self.names,q));poses={'base_link':np.eye(4)};axes={};anc={'base_link':[]};pending=self.joints.copy()
  while pending:
   nxt=[]
   for name,parent,child,kind,origin,axis in pending:
    if parent not in poses:nxt.append((name,parent,child,kind,origin,axis));continue
    t=poses[parent]@origin;motion=np.eye(4);v=values.get(name,0.)
    if kind in ('revolute','continuous'):motion[:3,:3]=R.from_rotvec(axis*v).as_matrix()
    elif kind=='prismatic':motion[:3,3]=axis*v
    axes[name]=(t[:3,3],t[:3,:3]@axis,kind);poses[child]=t@motion;anc[child]=anc[parent]+([name] if name in self.names else [])
   if len(nxt)==len(pending):raise ValueError('Disconnected URDF')
   pending=nxt
  return poses,axes,anc
 def jacobian(self,link,q,point=None,cache=None):
  poses,axes,anc=cache or self.forward(q);t=poses[link];xyz=t[:3,3] if point is None else t[:3,3]+t[:3,:3]@point;jac=np.zeros((6,len(self.names)))
  for name in anc[link]:
   pos,axis,kind=axes[name];idx=self.names.index(name)
   if kind=='prismatic':jac[:3,idx]=axis
   elif kind in ('revolute','continuous'):jac[:3,idx]=np.cross(axis,xyz-pos);jac[3:,idx]=axis
  return jac

def from_proprio(s,names):
 d={}
 for arm,sl in [('left',slice(3,10)),('right',slice(28,35))]:d.update(zip([f'{arm}_arm_joint{i}' for i in range(1,8)],s[sl]))
 d.update(zip([f'torso_joint{i}' for i in range(1,5)],s[53:57]))
 for arm,sl in [('left',slice(24,26)),('right',slice(49,51))]:d.update(zip([f'{arm}_gripper_finger_joint{i}' for i in range(1,3)],s[sl]))
 return np.array([d.get(n,0.) for n in names])
