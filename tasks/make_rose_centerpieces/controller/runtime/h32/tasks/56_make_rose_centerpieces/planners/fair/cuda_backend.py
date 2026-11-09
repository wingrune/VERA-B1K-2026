"""CUDA numerical kernels for onboard RGB-D registration; no simulator access.

Odometry and gravity use FP64; bounded object registration uses checked FP32
distances. Hypotheses are evaluated together; pixels and canonical object-local
priors are the only data.
OpenCV feature detection and sparse component labels stay on the host.
"""
import numpy as np
import torch


class CudaBackend:
    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA perception explicitly selected but CUDA is unavailable')
        self.device = torch.device('cuda')
        self.dtype = torch.float64
        from .cuda_kdtree import Nearest
        self.indexed_nearest = Nearest()

    def tensor(self, value):
        return torch.as_tensor(np.ascontiguousarray(value), device=self.device, dtype=self.dtype)

    def numpy(self, value):
        return value.detach().cpu().numpy()

    def pose_tensor(self, value):
        return torch.as_tensor(np.ascontiguousarray(value),device=self.device,dtype=torch.float32)

    def nearest(self, queries, targets):
        return self.indexed_nearest(queries,targets)

    def brute_nearest(self, queries, targets):
        # Direct differences avoid cancellation in squared-distance GEMM.
        parts = []
        for start in range(0, queries.shape[-2], 256):
            q = queries[..., start:start+256, :]
            best = torch.full(q.shape[:-1], torch.inf, device=self.device, dtype=queries.dtype)
            for offset in range(0, targets.shape[-2], 2048):
                d = torch.cdist(q, targets[..., offset:offset+2048, :],
                                compute_mode='donot_use_mm_for_euclid_dist').amin(-1)
                best = torch.minimum(best, d)
            parts.append(best)
        return torch.cat(parts, -1)

    def rigid(self, source, target):
        a, b = source.mean(-2), target.mean(-2)
        u, _, vh = torch.linalg.svd((source-a[...,None,:]).transpose(-1,-2) @ (target-b[...,None,:]))
        v = vh.transpose(-1,-2)
        diagonal = torch.ones(source.shape[:-2]+(source.shape[-1],),device=self.device,dtype=self.dtype)
        diagonal[..., -1] = torch.linalg.det(v @ u.transpose(-1,-2))
        rotation = v @ torch.diag_embed(diagonal) @ u.transpose(-1,-2)
        return rotation, b-(rotation @ a[...,None]).squeeze(-1)

    def ransac(self, source, target, seed, trials, threshold, minimum, planar=False):
        a, b = self.tensor(source), self.tensor(target)
        # Identical draws and first-winner tie order to the CPU controller.
        rng = np.random.default_rng(seed)
        ids = torch.as_tensor(np.stack([rng.choice(len(a),3,replace=False) for _ in range(trials)]),device=self.device)
        aa, bb = a[ids], b[ids]
        if planar:
            valid = torch.linalg.vector_norm(aa.amax(1)-aa.amin(1),dim=-1) >= .20
        else:
            valid = torch.linalg.vector_norm(torch.linalg.cross(aa[:,1]-aa[:,0],aa[:,2]-aa[:,0]),dim=-1) >= .00015
            da = torch.linalg.vector_norm(aa[:,:,None]-aa[:,None,:],dim=-1)
            db = torch.linalg.vector_norm(bb[:,:,None]-bb[:,None,:],dim=-1)
            valid &= (da-db).abs().amax((1,2)) <= .035
        rotations, translations = self.rigid(aa,bb)
        errors = torch.linalg.vector_norm(a[None] @ rotations.transpose(-1,-2)+translations[:,None]-b[None],dim=-1)
        masks = errors < threshold
        counts = masks.sum(-1)
        counts = torch.where(valid,counts,torch.full_like(counts,-1))
        winner = int(counts.argmax())
        if int(counts[winner]) < minimum:
            return None
        return self.numpy(masks[winner])

    def pointcloud(self, cam, mask=None, stride=1):
        d = self.tensor(cam['depth'])
        valid = torch.isfinite(d)&(d>.06)&(d<5)
        if mask is not None:
            valid &= torch.as_tensor(np.ascontiguousarray(mask),device=self.device,dtype=torch.bool)
        y,x = torch.where(valid)
        y,x = y[::stride],x[::stride]
        z = d[y,x]; k = self.tensor(cam['K'])
        p = torch.stack(((x+.5-k[0,2])*z/k[0,0],(y+.5-k[1,2])*z/k[1,1],z),-1)
        p = p @ self.tensor(cam['R']).T+self.tensor(cam['t'])
        uv = self.numpy(torch.stack((x,y),-1))
        return self.numpy(p),cam['rgb'][uv[:,1],uv[:,0]],uv

    def ground(self, estimator, camera):
        from scipy.spatial.transform import Rotation
        points = self.tensor(self.pointcloud(camera,stride=5)[0])
        radius = torch.linalg.vector_norm(points[:,:2],dim=1)
        points = points[(radius>.5)&(radius<4)&(points[:,2]>-.9)&(points[:,2]<.55)]
        points = points[::max(1,len(points)//1800)]
        if len(points)<60:
            return estimator.transform.copy()
        rng = np.random.default_rng(905)
        ids = torch.as_tensor(np.stack([rng.choice(len(points),3,replace=False) for _ in range(240)]),device=self.device)
        a,b,c = points[ids].unbind(1)
        normal = torch.linalg.cross(b-a,c-a)
        norm = torch.linalg.vector_norm(normal,dim=1)
        normal = normal/norm.clamp_min(1e-20)[:,None]
        normal = torch.where(normal[:,2:3]<0,-normal,normal)
        offset = -(normal*a).sum(1)
        intercept = -offset/normal[:,2].clamp_min(1e-20)
        masks = (points @ normal.T+offset).abs().T < .012
        counts = masks.sum(1)
        xy = points[:,:2].expand(240,-1,-1)
        lo = torch.where(masks[:,:,None],xy,torch.inf).amin(1)
        hi = torch.where(masks[:,:,None],xy,-torch.inf).amax(1)
        spread = torch.linalg.vector_norm(hi-lo,dim=1)
        valid = (norm>=.02)&(normal[:,2]>=.80)&(intercept>-.45)&(intercept<.15)&(counts>=100)&(spread>=.9)
        scores = counts-200*intercept.clamp_min(0)
        scores = torch.where(valid,scores,-torch.inf)
        if not bool(valid.any()):
            return estimator.transform.copy()
        selected = points[masks[scores.argmax()]]
        center = selected.mean(0)
        _,_,axes = torch.linalg.svd(selected-center,full_matrices=False)
        normal = axes[-1]
        if float(normal[2])<0:
            normal = -normal
        residual = float(torch.quantile(((selected-center)@normal).abs(),.5))
        if residual>.003:
            return estimator.transform.copy()
        n = self.numpy(normal);cross = np.cross(n,[0.,0.,1.]);sine = np.linalg.norm(cross)
        rot = np.eye(3) if sine<1e-8 else Rotation.from_rotvec(cross/sine*np.arctan2(sine,n[2])).as_matrix()
        height = -float(torch.quantile(selected @ self.tensor(rot[2]),.5))
        transform = np.eye(4);transform[:3,:3] = rot;transform[2,3] = height
        estimator.transform = transform
        estimator.stats = dict(valid=True,normal_body=n.tolist(),height_m=height,inliers=len(selected),residual_m=residual)
        return transform.copy()

    def matcher(self, hamming=False):
        backend = self
        class Matcher:
            def knnMatch(self, source, target, k=2):
                import cv2
                if k!=2:
                    raise ValueError('Registration matcher requires k=2')
                if len(target)<2:
                    return []
                rows = []
                if hamming:
                    a = torch.as_tensor(source,device=backend.device,dtype=torch.uint8)
                    b = torch.as_tensor(target,device=backend.device,dtype=torch.uint8)
                    table = torch.as_tensor([i.bit_count() for i in range(256)],device=backend.device,dtype=torch.int16)
                    for start in range(0,len(a),64):
                        xor = torch.bitwise_xor(a[start:start+64,None,:],b[None,:,:])
                        distance = table[xor.long()].sum(-1).to(backend.dtype)
                        values,ids = torch.sort(distance,dim=-1,stable=True)
                        rows.extend(zip(backend.numpy(values[:,:2]),backend.numpy(ids[:,:2])))
                else:
                    a,b = backend.tensor(source),backend.tensor(target)
                    for start in range(0,len(a),128):
                        distance = torch.cdist(a[start:start+128],b,compute_mode='donot_use_mm_for_euclid_dist')
                        values,ids = torch.sort(distance,dim=-1,stable=True)
                        rows.extend(zip(backend.numpy(values[:,:2]),backend.numpy(ids[:,:2])))
                return [[cv2.DMatch(i,int(ids[j]),float(values[j])) for j in range(2)] for i,(values,ids) in enumerate(rows)]
        return Matcher()

    def rotation(self, vec):
        # Rodrigues with sinc form is finite and differentiable at zero.
        angle = torch.linalg.vector_norm(vec,dim=-1)
        x,y,z = vec.unbind(-1);o = torch.zeros_like(x)
        skew = torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(vec.shape[:-1]+(3,3))
        eye = torch.eye(3,device=self.device,dtype=vec.dtype)
        return eye+torch.sinc(angle/torch.pi)[...,None,None]*skew + \
            (.5*torch.sinc(angle/(2*torch.pi)).square())[...,None,None]*(skew@skew)

    def yaw(self, angle):
        c,s = torch.cos(angle),torch.sin(angle);o = torch.zeros_like(c);i = torch.ones_like(c)
        return torch.stack((c,-s,o,s,c,o,o,o,i),-1).reshape(angle.shape+(3,3))

    def optimize(self, residual, initial, lower, upper, iterations, robust=False):
        """Bounded independent LM hypotheses, finite Jacobians computed on GPU."""
        x = self.pose_tensor(initial);lo = self.pose_tensor(lower);hi = self.pose_tensor(upper)
        damping = torch.full((len(x),),1e-3,device=self.device,dtype=x.dtype)
        eye = torch.eye(x.shape[1],device=self.device,dtype=x.dtype)
        def cost(r):
            return (2*.012**2*(torch.sqrt(1+(r/.012).square())-1)).sum(-1) if robust else r.square().sum(-1)
        for _ in range(iterations):
            h = torch.maximum(x.abs()*.001,torch.full_like(x,1e-5))
            trials = [x]
            for j in range(x.shape[1]):
                trial = x.clone();trial[:,j] += h[:,j]
                trials.append(trial)
            evaluated = residual(torch.cat(trials,0)).split(len(x),0)
            r = evaluated[0]
            jac = torch.stack([(evaluated[j+1]-r)/h[:,j,None] for j in range(x.shape[1])],-1)
            weight = (1+(r/.012).square()).pow(-.25) if robust else torch.ones_like(r)
            jw = jac*weight[:,:,None];rw = r*weight
            normal = jw.transpose(-1,-2)@jw
            rhs = -(jw.transpose(-1,-2)@rw[:,:,None]).squeeze(-1)
            regularized = normal+damping[:,None,None]*torch.diag_embed(normal.diagonal(dim1=-2,dim2=-1).clamp_min(1e-8))+1e-10*eye
            delta = torch.linalg.solve(regularized,rhs[:,:,None]).squeeze(-1)
            candidate = torch.maximum(lo,torch.minimum(hi,x+delta))
            better = cost(residual(candidate)) < cost(r)
            change = torch.where(better[:,None],candidate-x,torch.zeros_like(x))
            x = torch.where(better[:,None],candidate,x)
            damping = torch.where(better,damping*.3,damping*10).clamp(1e-8,1e8)
            if bool((torch.linalg.vector_norm(change,dim=-1)<1e-7).all() & better.all()):
                break
        return x

    def support(self, model, cameras, pose):
        xyz = self.tensor(model['xyz'][::3]);t = self.tensor(pose)
        best = dict(matched_fraction=0.,unoccluded_fraction=0.,pixels=0)
        for role,c in cameras.items():
            p = (xyz @ t[:3,:3].T+t[:3,3]-self.tensor(c['t'])) @ self.tensor(c['R'])
            p = p[p[:,2]>.06]
            uv = p @ self.tensor(c['K']).T
            uv = torch.round(uv[:,:2]/uv[:,2:3]-.5).long()
            h,w = c['depth'].shape
            valid = (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            uv,z = uv[valid],p[valid,2]
            front = torch.full((h*w,),torch.inf,device=self.device,dtype=self.dtype)
            front.scatter_reduce_(0,uv[:,1]*w+uv[:,0],z,reduce='amin',include_self=True)
            ids = torch.where(torch.isfinite(front))[0]
            if len(ids)<60:
                continue
            observed = self.tensor(c['depth']).flatten()[ids];expected = front[ids]
            visible = torch.isfinite(observed)&(observed>.06)&(observed>=expected-.025)
            matches = visible&((observed-expected).abs()<.025)
            unoccluded = float(visible.to(self.dtype).mean());fraction = float(matches.sum()/visible.sum().clamp_min(1))
            if fraction>best['matched_fraction'] and unoccluded>.50:
                best = dict(camera=role,matched_fraction=fraction,unoccluded_fraction=unoccluded,pixels=len(ids))
        return best
