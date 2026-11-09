"""FP64 collision membership without materializing point-by-plane tensors."""
import torch
import warp as wp


@wp.kernel(enable_backward=False)
def sphere_union(points: wp.array2d(dtype=wp.float64),
                 centers: wp.array2d(dtype=wp.float64),
                 radii: wp.array(dtype=wp.float64),
                 result: wp.array(dtype=wp.int32),
                 uncertain: wp.array(dtype=wp.int32), count: int):
    i = wp.tid()
    inside = int(0)
    for j in range(count):
        x = points[i, 0] - centers[j, 0]
        y = points[i, 1] - centers[j, 1]
        z = points[i, 2] - centers[j, 2]
        margin = x*x + y*y + z*z - radii[j]*radii[j]
        if wp.abs(margin) < wp.float64(1.0e-10):
            wp.atomic_or(uncertain, 0, 1)
        if margin <= wp.float64(0.0):
            inside = 1
    result[i] = inside


@wp.kernel(enable_backward=False)
def hull_union(points: wp.array2d(dtype=wp.float64),
               transforms: wp.array3d(dtype=wp.float64),
               bounds: wp.array3d(dtype=wp.float64),
               planes: wp.array3d(dtype=wp.float64),
               ids: wp.array(dtype=wp.int32), owners: wp.array(dtype=wp.int32),
               result: wp.array(dtype=wp.int32), uncertain: wp.array(dtype=wp.int32),
               padding: wp.float64, width: int):
    pair = wp.tid()
    i = ids[pair]
    h = owners[pair]
    x = points[i, 0] - transforms[h, 0, 3]
    y = points[i, 1] - transforms[h, 1, 3]
    z = points[i, 2] - transforms[h, 2, 3]
    local = wp.vec3d(x*transforms[h, 0, 0]+y*transforms[h, 1, 0]+z*transforms[h, 2, 0],
                     x*transforms[h, 0, 1]+y*transforms[h, 1, 1]+z*transforms[h, 2, 1],
                     x*transforms[h, 0, 2]+y*transforms[h, 1, 2]+z*transforms[h, 2, 2])
    inside = int(1)
    for k in range(3):
        lo = local[k] - (bounds[h, 0, k]-padding)
        hi = bounds[h, 1, k]+padding-local[k]
        if wp.abs(lo) < wp.float64(1.0e-10) or wp.abs(hi) < wp.float64(1.0e-10):
            wp.atomic_or(uncertain, 0, 1)
        if lo < wp.float64(0.0) or hi < wp.float64(0.0):
            inside = 0
    for k in range(width):
        value = local[0]*planes[h, k, 0]+local[1]*planes[h, k, 1]+local[2]*planes[h, k, 2]+planes[h, k, 3]
        if wp.abs(value-padding) < wp.float64(1.0e-10):
            wp.atomic_or(uncertain, 0, 1)
        if value > padding:
            inside = 0
    if inside == 1:
        wp.atomic_or(result, i, 1)


def launch(kernel, tensors, count, constants=()):
    """Hold Torch owners and use their current stream through completion."""
    wp.init()
    wp.launch(kernel, dim=count,
              inputs=[wp.from_torch(t) for t in tensors]+list(constants),
              device=str(tensors[0].device),
              stream=wp.stream_from_torch(torch.cuda.current_stream(tensors[0].device)))
