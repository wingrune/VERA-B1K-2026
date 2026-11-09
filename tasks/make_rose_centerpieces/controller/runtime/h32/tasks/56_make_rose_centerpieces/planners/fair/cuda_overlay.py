"""Import the pinned controller with task-owned CUDA perception substitutions."""
import importlib.abc
import importlib.util
from pathlib import Path
from .search_time import transform
from .cuda_backend import CudaBackend
from . import cuda_registration as registration
from . import cuda_collision as collision

MODULES = {'perception','object_perception','ground_frame','visual_odometry','rose_geometry','kinematics','robot_collision','scene_collision'}


def rewrite(name, source):
    source = transform(name,source)
    if name=='object_perception':
        start = source.index('   rng=np.random.default_rng(839);best=None')
        end = source.index('   ii=best[1];',start)
        source = source[:start]+'''   mask=cuda_backend.ransac(a,b,839,min(700,len(a)*30),.018,6)
   if mask is None:continue
   best=(int(mask.sum()),mask)
'''+source[end:]
        source = source.replace('cv2.BFMatcher().knnMatch','cuda_backend.matcher().knnMatch')
    elif name=='scene_collision':
        source=source.replace("for ids in tree.query_ball_point(centers,robot.radii+.020):mask[ids]=False","mask &= ~cuda_sphere_mask(cloud,centers,robot.radii+.020,tree)")
    elif name=='kinematics':
        source=source.replace("R.from_rotvec(axis*v).as_matrix()","self._rotation_matrix(name,axis,v)")
    elif name=='visual_odometry':
        start = source.index('    rng = np.random.default_rng(91)')
        end = source.index('    if best.sum() < 24',start)
        source = source[:start]+'''    best = cuda_backend.ransac(source,target,91,180,.025,24,planar=True)
    if best is None:
        return None, {'reason': 'insufficient_consensus', 'matches': len(source), 'inliers': 0}
'''+source[end:]
        source = source.replace('cv2.BFMatcher(cv2.NORM_HAMMING)','cuda_backend.matcher(hamming=True)')
    return source


class Loader(importlib.abc.Loader):
    def __init__(self,name,path,backend):
        self.name,self.path,self.backend = name,path,backend
    def create_module(self,spec):
        return None
    def exec_module(self,module):
        module.__dict__['cuda_backend'] = self.backend
        exec(compile(rewrite(self.name,self.path.read_text()),str(self.path),'exec'),module.__dict__)
        gpu = self.backend
        if self.name=='scene_collision':
            module.cuda_sphere_mask=lambda points,centers,radii,tree: collision.sphere_mask(gpu,points,centers,radii,tree)
        elif self.name=='robot_collision':
            original_mask=module.RobotCollision.point_self_mask
            module.RobotCollision.point_self_mask=lambda robot,points,poses,padding=.012,tree=None: collision.point_mask(gpu,robot,points,poses,padding,tree,original_mask)
        elif self.name=='kinematics':
            original_init=module.Kinematics.__init__
            def initialize(kin,*args,**kwargs):
                original_init(kin,*args,**kwargs);kin._rotation_cache={}
            def matrix(kin,name,axis,value):
                import numpy as np
                key=(axis.dtype.str,axis.tobytes(),np.asarray(value).dtype.str,np.asarray(value).tobytes())
                cached=kin._rotation_cache.get(name)
                if cached is None or cached[0]!=key:
                    cached=(key,module.R.from_rotvec(axis*value).as_matrix())
                    kin._rotation_cache[name]=cached
                return cached[1]
            module.Kinematics.__init__=initialize
            module.Kinematics._rotation_matrix=matrix
        elif self.name=='perception':
            module.pointcloud = gpu.pointcloud
        elif self.name=='ground_frame':
            module.GroundFrameEstimator.update = lambda estimator,camera: gpu.ground(estimator,camera)
        elif self.name=='rose_geometry':
            module.rose_candidates = lambda *a,**k: registration.rose_candidates(gpu,module,*a,**k)
            module.vase_fit = lambda *a,**k: registration.vase_fit(gpu,module,*a,**k)
            module._guided_rose_fit_single = lambda *a,**k: registration.guided_single(gpu,module,*a,**k)
            module.visible_model_support = gpu.support


class CudaFinder(importlib.abc.MetaPathFinder):
    def __init__(self,root):
        self.root = Path(root);self.backend = CudaBackend()
    def find_spec(self,fullname,path=None,target=None):
        if fullname not in MODULES:
            return None
        source = self.root/(fullname+'.py')
        return importlib.util.spec_from_file_location(fullname,source,loader=Loader(fullname,source,self.backend))
