"""Exact nearest-neighbour traversal on CUDA; CPU builds an immutable index.

The index contains only the caller's observed/canonical points. CUDA computes
all query distances and branch decisions. No simulator or scene labels.
"""
from collections import OrderedDict
import numpy as np
from scipy.spatial import cKDTree
import torch
import warp as wp

Stack64=wp.types.vector(length=64,dtype=wp.int32)


@wp.kernel(enable_backward=False)
def _query(queries: wp.array2d(dtype=wp.float32), points: wp.array2d(dtype=wp.float32),
           indices: wp.array(dtype=wp.int32), low: wp.array2d(dtype=wp.float32),
           high: wp.array2d(dtype=wp.float32), children: wp.array2d(dtype=wp.int32),
           begin: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
           split_dim: wp.array(dtype=wp.int32), split: wp.array(dtype=wp.float32),
           output: wp.array(dtype=wp.float32), error: wp.array(dtype=wp.int32), nodes: int):
    q=wp.tid()
    position=wp.vec3(queries[q,0],queries[q,1],queries[q,2])
    stack=Stack64(0)
    top=int(1)
    best=float(1.0e30)
    error[q]=0
    while top>0:
        top=top-1
        node=stack[top]
        if node<0 or node>=nodes:
            error[q]=1
            break
        dx=wp.max(wp.max(low[node,0]-position[0],position[0]-high[node,0]),0.0)
        dy=wp.max(wp.max(low[node,1]-position[1],position[1]-high[node,1]),0.0)
        dz=wp.max(wp.max(low[node,2]-position[2],position[2]-high[node,2]),0.0)
        if dx*dx+dy*dy+dz*dz<=best:
            size=count[node]
            if size>0:
                for k in range(size):
                    point_id=indices[begin[node]+k]
                    difference=wp.vec3(points[point_id,0],points[point_id,1],points[point_id,2])-position
                    best=wp.min(best,wp.dot(difference,difference))
            else:
                if top>=63:
                    error[q]=2
                    break
                left=children[node,0]
                right=children[node,1]
                near=left
                far=right
                if position[split_dim[node]]>split[node]:
                    near=right
                    far=left
                stack[top]=far
                stack[top+1]=near
                top=top+2
    output[q]=wp.sqrt(best)


class Index:
    def __init__(self, points):
        wp.init()
        if points.ndim!=2 or points.shape[1]!=3 or len(points)==0:
            raise ValueError('Nonempty Nx3 points required')
        self.owner = points
        array = points.detach().cpu().double().numpy()
        if not np.isfinite(array).all():
            raise ValueError('KD index requires finite points')
        tree = cKDTree(array,leafsize=16,compact_nodes=True,balanced_tree=True)
        lo=[];hi=[];children=[];starts=[];counts=[];dims=[];splits=[];indices=[]
        def visit(node,depth):
            if depth>=62:
                raise ValueError('KD tree exceeds CUDA stack capacity')
            number=len(lo);subset=array[node.indices]
            lo.append(subset.min(0));hi.append(subset.max(0));children.append([-1,-1])
            starts.append(0);counts.append(0);dims.append(node.split_dim);splits.append(node.split)
            if node.split_dim<0:
                if len(node.indices)>16 and not np.all(subset==subset[0]):
                    raise ValueError('Oversized nondegenerate KD leaf')
                starts[number]=len(indices);counts[number]=min(len(node.indices),16);indices.extend(node.indices.tolist())
            else:
                children[number]=[visit(node.lesser,depth+1),visit(node.greater,depth+1)]
            return number
        visit(tree.tree,0)
        convert=lambda a,dtype: torch.as_tensor(np.asarray(a),device=points.device,dtype=dtype).contiguous()
        self.points=points.float().contiguous();self.lo=convert(lo,torch.float32);self.hi=convert(hi,torch.float32)
        # Expand float bounds outward by one ULP to prevent false branch pruning.
        self.lo=torch.nextafter(self.lo,torch.full_like(self.lo,-torch.inf))
        self.hi=torch.nextafter(self.hi,torch.full_like(self.hi,torch.inf))
        self.children=convert(children,torch.int32);self.starts=convert(starts,torch.int32)
        self.counts=convert(counts,torch.int32);self.dims=convert(dims,torch.int32)
        self.splits=convert(splits,torch.float32);self.indices=convert(indices,torch.int32)
        self.warp_arrays=[wp.from_torch(t) for t in (self.points,self.indices,self.lo,self.hi,
                         self.children,self.starts,self.counts,self.dims,self.splits)]

    def query(self, queries):
        shape=queries.shape[:-1];q=queries.float().contiguous().reshape(-1,3)
        output=torch.empty(len(q),device=q.device,dtype=torch.float32)
        if len(q)==0:
            return output.reshape(shape)
        error=torch.zeros(len(q),device=q.device,dtype=torch.int32)
        arrays=[wp.from_torch(q)]+self.warp_arrays+[wp.from_torch(output),wp.from_torch(error)]
        stream=wp.stream_from_torch(torch.cuda.current_stream(q.device))
        wp.launch(_query,dim=len(q),inputs=arrays+[len(self.counts)],device=str(q.device),stream=stream)
        if bool(error.any()):
            raise RuntimeError('CUDA KD traversal invariant failed: '+str(torch.unique(error).tolist()))
        return output.reshape(shape).to(queries.dtype)


class Nearest:
    def __init__(self):
        self.cache=OrderedDict()
    def __call__(self, queries, targets):
        key=(targets.data_ptr(),tuple(targets.shape),targets._version)
        if key not in self.cache:
            self.cache[key]=Index(targets)
            if len(self.cache)>32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key].query(queries)
