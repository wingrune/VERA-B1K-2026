"""The button of the fair planner of task 98: the template point of the scanner, replaced by the red blob when it is fresh and agrees.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .. import params as P

class ButtonMixin:
    def marker_template(self) -> Optional[np.ndarray]:
        s = self.scanner
        if s is None:
            return None
        ex = np.array([math.cos(s.yaw), math.sin(s.yaw)])
        xy = s.centre + ex * P.SCANNER_MARKER_FRONT_M
        return np.array([xy[0], xy[1], s.z_hi - P.SCANNER_MARKER_BELOW_TOP_M])

    def marker_estimate(self) -> Optional[np.ndarray]:
        """The button: the red blob seen in RGB-D when it is fresh and agrees with the template of the scanner (a few millimetres against about a
        centimetre of the template), else the template point of the scanner. The blob is only a position, never a sign that the button was pressed."""
        template = self.marker_template()
        if template is None or self.marker_blob is None or self.step - self.marker_blob_step > P.BLOB_FRESH_STEPS:
            return template
        if np.linalg.norm(self.marker_blob - template) > P.BLOB_AGREE_M:
            return template
        return self.marker_blob.copy()
