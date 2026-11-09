"""Bounded joint-space planning with carried-object and observed-wall clearance."""
import time
import numpy as np


def wall_clearance_path(servo, q, target, grasp, wall_point, normal, seconds=12., seed=0, bounds=None, collision=None, wall_margin=.022):
    indices = servo.indices
    start = q[indices].copy()
    bounds=np.array([[-.005,-.155,-.2225],[.008,.155,.2225]]) if bounds is None else np.asarray(bounds)
    corners = np.array([[x,y,z,1.] for x in bounds[:,0] for y in bounds[:,1] for z in bounds[:,2]])
    deadline = time.monotonic()+seconds
    rng = np.random.default_rng(seed)

    def valid(x):
        trial = q.copy(); trial[indices] = x
        held = servo.pose(trial)@grasp
        points = (corners@held.T)[:, :3]
        if ((points-wall_point)@normal).min() < wall_margin or points[:,2].min() < .40:
            return False
        if held[0,3] < .12 or abs(held[1,3]) > .80:
            return False
        poses = servo.kin.forward(trial)[0]
        if collision is not None and not collision.valid(poses,held):return False
        for name, pose in poses.items():
            if name.startswith(('left_arm_link','left_gripper')) and (pose[:3,3]-wall_point)@normal < .055:
                return False
        return True

    def edge(a,b):
        return all(valid(a+t*(b-a)) for t in np.linspace(0,1,max(2,int(np.max(np.abs(b-a))/.065)+1)))

    if not valid(start) or not valid(target):
        return None
    if edge(start,target):
        return [target.copy()]
    trees = [([start],[-1]),([target.copy()],[-1])]
    def extend(tree, sample):
        nodes, parents = tree
        nearest = int(np.argmin([np.linalg.norm(x-sample) for x in nodes]))
        delta = sample-nodes[nearest]; distance = np.linalg.norm(delta)
        new = nodes[nearest]+delta*min(1.,.28/max(distance,1e-9))
        if not edge(nodes[nearest],new):
            return None,False
        nodes.append(new);parents.append(nearest)
        return len(nodes)-1,distance<=.28

    for iteration in range(1600):
        if time.monotonic()>deadline:
            break
        side = iteration%2
        if rng.random()<.25:
            sample = trees[1-side][0][-1]
        else:
            alpha = rng.random()
            sample = np.clip(start*(1-alpha)+target*alpha+rng.normal(0,.7,len(start)),servo.limits[0]+1e-4,servo.limits[1]-1e-4)
        ia,_ = extend(trees[side],sample)
        if ia is None:
            continue
        for _ in range(40):
            ib,reached = extend(trees[1-side],trees[side][0][ia])
            if ib is None:
                break
            if reached:
                def ancestry(tree,index):
                    values=[]
                    while index>=0:
                        values.append(tree[0][index]);index=tree[1][index]
                    return values[::-1]
                left=ancestry(trees[side],ia);right=ancestry(trees[1-side],ib)
                path=left+right[::-1] if side==0 else right+left[::-1]
                # Greedy shortcuts retain the same swept-volume checks.
                result=[path[0]];i=0
                while i<len(path)-1:
                    j=len(path)-1
                    while j>i+1 and not edge(path[i],path[j]):j-=1
                    result.append(path[j]);i=j
                return result[1:]
    return None
