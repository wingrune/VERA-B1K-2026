"""Bounded RGB-D recipient reacquisition using existing checked carry poses."""
import numpy as np
from art_skills import carry, bag_guards
from manip import hold


def observe_recipient(ctx, est, bag):
    """Observe from a normal torso pose, then restore clearance before release."""
    if ctx.held is None or ctx.held[1] == 'tote':
        return None
    if not (yield from carry(ctx, height=1.0, stable_view=True)):
        ctx.event('recipient_view', ok=False, reason='no checked observation carry')
        return None
    yield from hold(ctx, 12)
    fresh = est.refresh('tote')
    ctx.event('recipient_view', ok=fresh is not None,
              reason='RGB-D observation from normal carry',
              shift=None if fresh is None else float(np.linalg.norm(fresh[:3,3]-bag[:3,3])))
    if fresh is not None:
        ctx.guard = bag_guards(fresh)
    recipient = bag if fresh is None else fresh
    if not (yield from carry(ctx, height=recipient[2,3]+.41, stable_view=True)):
        ctx.event('recipient_view_restore', ok=False, reason='no checked release carry')
        return None
    return fresh
