"""High-contrast floor marks from onboard RGB-D; no particle-state access."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from perception import pointcloud


def floor_marks(cameras, level, capsules=()):
    marks = []
    for role, camera in cameras.items():
        points, rgb, pixels = pointcloud(camera)
        leveled = points @ level[:3, :3].T + level[:3, 3]
        floor = (abs(leveled[:, 2]) < .025) & (np.linalg.norm(leveled[:, :2], axis=1) > .45)
        for a, b, radius in capsules:
            axis = b-a
            alpha = np.clip((points-a) @ axis / max(float(axis @ axis), 1e-10), 0, 1)
            floor &= np.linalg.norm(points-a-alpha[:, None]*axis, axis=1) > radius+.015
        gray = cv2.cvtColor(camera['rgb'], cv2.COLOR_RGB2GRAY)
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        dark = (gray < 115) & (background.astype(float)-gray > 35)
        mask = np.zeros(gray.shape, np.uint8)
        selected = pixels[floor]
        mask[selected[:, 1], selected[:, 0]] = 1
        floor_mask = mask.copy()
        mask &= dark.astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
        label_at_point = labels[pixels[:, 1], pixels[:, 0]]
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 5:
                continue
            component = labels == label
            if np.mean(floor_mask[component]) < .85:
                continue
            x,y,w,h,_ = stats[label]
            x0,y0=max(0,x-4),max(0,y-4);x1,y1=min(gray.shape[1],x+w+4),min(gray.shape[0],y+h+4)
            if np.mean(floor_mask[y0:y1,x0:x1]) < .80:
                continue
            p = leveled[(label_at_point == label) & floor]
            if len(p)<5:
                continue
            extent = np.ptp(p[:, :2], axis=0)
            if np.max(extent) > .35 or np.linalg.norm(extent) < .012 or min(extent)<.12*max(extent):
                continue
            marks.append(dict(position=np.median(p, axis=0), camera=role,
                              pixels=int(len(p)), extent=extent.tolist()))
    if not marks:
        return []
    centers = np.array([m['position'] for m in marks])
    tree = cKDTree(centers[:, :2]); unseen = set(range(len(marks))); result = []
    while unseen:
        seed = unseen.pop(); group = [seed]; todo = [seed]
        while todo:
            for other in tree.query_ball_point(centers[todo.pop(), :2], .085):
                if other in unseen:
                    unseen.remove(other); group.append(other); todo.append(other)
        if np.max(np.ptp(centers[group, :2], axis=0)) > .35:
            continue
        weights = np.array([marks[i]['pixels'] for i in group])
        position = np.average(centers[group], axis=0, weights=weights); position[2] = 0.
        result.append(dict(position=position.tolist(), pixels=int(weights.sum()),
                           cameras=sorted(set(marks[i]['camera'] for i in group))))
    return result
