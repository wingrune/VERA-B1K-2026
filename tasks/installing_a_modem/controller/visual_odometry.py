"""Short-baseline planar RGB-D odometry from allowed onboard observations.

No simulator access, world coordinates, task identities, or success labels.
The caller supplies robot-relative calibrated camera images. Scene correspondences
correct proprioceptive drift; ambiguous or physically implausible fits are rejected.
"""
import cv2
import numpy as np


def extract(camera):
    image = camera['rgb']
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.ORB_create(nfeatures=1800, fastThreshold=10).detectAndCompute(gray, None)
    if descriptors is None:
        return np.empty((0, 3)), None
    uv = np.asarray([point.pt for point in keypoints])
    pixels = np.rint(uv).astype(int)
    pixels[:, 0] = pixels[:, 0].clip(0, gray.shape[1] - 1)
    pixels[:, 1] = pixels[:, 1].clip(0, gray.shape[0] - 1)
    z = camera['depth'][pixels[:, 1], pixels[:, 0]]
    K = camera['K']
    local = np.c_[(uv[:, 0] + .5 - K[0, 2]) / K[0, 0] * z,
                  (uv[:, 1] + .5 - K[1, 2]) / K[1, 1] * z, z]
    points = local @ camera['R'].T + camera['t']
    valid = np.isfinite(points).all(1) & (z > .2) & (z < 4.5)
    # Exclude the near robot body and the red target; neither is a stationary landmark.
    valid &= np.linalg.norm(points[:, :2], axis=1) > .55
    excluded = camera.get('moving_object_mask')
    if excluded is not None:
        valid &= ~excluded[pixels[:, 1], pixels[:, 0]]
    rgb = image[pixels[:, 1], pixels[:, 0]].astype(float)
    valid &= ~((rgb[:, 0] > 85) & (rgb[:, 0] > 2.5 * rgb[:, 1] + 20)
               & (rgb[:, 0] > 2.5 * rgb[:, 2] + 20))
    return points[valid], descriptors[valid]


def rigid_xy(source, target):
    a, b = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - a).T @ (target - b))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, b - rotation @ a


def fit_correspondences(source, target, dt):
    """Fit current-frame points to previous-frame points; return None on ambiguity."""
    source, target = np.asarray(source), np.asarray(target)
    valid = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    valid &= np.abs(source[:, 2] - target[:, 2]) < .06
    source, target = source[valid, :2], target[valid, :2]
    if len(source) < 24:
        return None, {'reason': 'few_height_consistent_matches', 'matches': len(source)}
    rng = np.random.default_rng(91)
    best = np.zeros(len(source), bool)
    for _ in range(180):
        indices = rng.choice(len(source), 3, replace=False)
        if np.linalg.norm(np.ptp(source[indices], axis=0)) < .20:
            continue
        rotation, translation = rigid_xy(source[indices], target[indices])
        error = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        inliers = error < .025
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < 24 or best.mean() < .4:
        return None, {'reason': 'insufficient_consensus', 'matches': len(source), 'inliers': int(best.sum())}
    for _ in range(3):
        rotation, translation = rigid_xy(source[best], target[best])
        errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
        best = errors < .025
        if best.sum() < 24:
            return None, {'reason': 'unstable_refinement'}
    # Require landmarks distributed over space rather than one moving object.
    spread = np.linalg.norm(np.ptp(source[best], axis=0))
    cells = len(np.unique(np.floor(source[best] / .20).astype(int), axis=0))
    median_error = float(np.median(errors[best]))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    stats = {'matches': len(source), 'inliers': int(best.sum()), 'median_residual_m': median_error,
             'spread_m': float(spread), 'cells': cells, 'yaw_delta_rad': yaw,
             'translation_m': translation.tolist()}
    if best.mean() < .4 or spread < .45 or cells < 6 or median_error > .015:
        return None, dict(stats, reason='weak_spatial_support')
    # Deliberately loose bounds from native robot speed limits, not measured GT motion.
    if np.linalg.norm(translation) > 1.2 * dt + .03 or abs(yaw) > 1.2 * dt + .03:
        return None, dict(stats, reason='implausible_motion')
    return np.r_[translation, yaw], dict(stats, reason='accepted')


def match(previous, current, dt):
    old_points, old_descriptors = previous
    new_points, new_descriptors = current
    if old_descriptors is None or new_descriptors is None:
        return None, {'reason': 'no_descriptors'}
    raw = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(old_descriptors, new_descriptors, k=2)
    candidates = [pair[0] for pair in raw if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
    unique, used = [], set()
    for item in sorted(candidates, key=lambda item: item.distance):
        if item.trainIdx not in used:
            unique.append(item)
            used.add(item.trainIdx)
    if len(unique) < 24:
        return None, {'reason': 'few_unique_matches', 'matches': len(unique)}
    return fit_correspondences(np.asarray([new_points[m.trainIdx] for m in unique]),
                               np.asarray([old_points[m.queryIdx] for m in unique]), dt)


class VisualOdometry:
    def __init__(self, interval=5):
        self.interval = interval
        self.anchor = None
        self.stats = {'reason': 'uninitialized'}

    def update(self, camera, step, predicted_odom):
        if self.anchor is not None and step - self.anchor[0] < self.interval:
            return predicted_odom
        current = extract(camera)
        corrected = np.asarray(predicted_odom).copy()
        if self.anchor is not None:
            old_step, old_features, old_odom = self.anchor
            delta, self.stats = match(old_features, current, (step - old_step) / 30.)
            if delta is not None:
                c, s = np.cos(old_odom[2]), np.sin(old_odom[2])
                corrected[:2] = old_odom[:2] + np.array([[c, -s], [s, c]]) @ delta[:2]
                corrected[2] = old_odom[2] + delta[2]
                self.stats['correction_from_proprio'] = (corrected - predicted_odom).tolist()
        else:
            self.stats = {'reason': 'initialized'}
        self.stats['step'] = int(step)
        self.anchor = (step, current, corrected.copy())
        return corrected
