"""Exact lexicographic voxel deduplication without structured-array sorting."""
import numpy as np

def voxel_first_indices(cloud, size=.02):
    if not len(cloud):return np.empty(0,dtype=np.int64)
    cells=np.floor(cloud/size).astype(np.int64)
    low=cells.min(0);span=cells.max(0)-low+1
    if np.any(span<=0) or int(span[0])*int(span[1])*int(span[2])>=np.iinfo(np.int64).max:
        return np.unique(cells,axis=0,return_index=True)[1]
    shifted=cells-low
    code=(shifted[:,0]*span[1]+shifted[:,1])*span[2]+shifted[:,2]
    return np.unique(code,return_index=True)[1]
