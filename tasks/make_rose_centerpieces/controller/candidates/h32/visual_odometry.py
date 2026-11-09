"""Retain the original visual yaw correction and integrated proprio XY.

The trusted factory preserves the original CPU/CUDA image matcher and gates.
The reference contains only onboard image features and planar odometry.
"""
from .skills import create_visual_odometry


class VisualOdometry:
    def __init__(self, interval=5):
        self.reference = create_visual_odometry(interval)
        self.stats = dict(self.reference.stats)

    def update(self, camera, step, predicted_odom):
        corrected = self.reference.update(camera, step, predicted_odom).copy()
        discarded_xy = (corrected[:2] - predicted_odom[:2]).copy()
        corrected[:2] = predicted_odom[:2]
        anchor = self.reference.anchor
        if anchor is not None and anchor[0] == step:
            anchor[2][:2] = predicted_odom[:2]
        self.stats = dict(self.reference.stats)
        self.stats.update(
            mode="proprio_xy_visual_yaw",
            visual_translation_weight=0.0,
            reference_xy_correction=discarded_xy.tolist(),
            correction_from_proprio=(corrected - predicted_odom).tolist(),
        )
        return corrected
