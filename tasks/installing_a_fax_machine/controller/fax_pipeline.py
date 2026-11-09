"""Task script for installing the fax: press its button on the floor, pick it up, put it on a desk.

The order is deliberate: the toggle state persists once set, and on the floor the button is
reached with the same arm posture as the grasp, so the hard-to-reach press inside the cubicle
after placement is avoided.
"""
import numpy as np
from fax_skills import (plan_floor_stance, floor_prestance, choose_rotation, press_button, grasp_from_floor, to_carry,
                        choose_site, place_on_desk, drive, hold, follow, NAV_TRUNK, FAX_KEEPOUT)


def pipeline(ctx, do_press=True):
    fax = ctx.est.fax()
    if fax is None:
        ctx.event('pipeline', ok=False, reason='fax pose unknown'); return 'no_fax'
    stance = plan_floor_stance(ctx, fax)
    if stance is None:
        return 'no_stance'
    ctx.arm = stance[2]
    fax_xy = fax[:2, 3]; pre = floor_prestance(ctx, stance[0], fax_xy)
    status = yield from drive(ctx, pre, stance[1], keepout=[(fax_xy, FAX_KEEPOUT)])
    if status == 'arrived':
        status = yield from drive(ctx, stance[0], stance[1], speed=.25, direct=True)
    ctx.event('at_stance', status=status, base=[round(float(v), 3) for v in ctx.est.base()])
    if status != 'arrived':
        return 'stance_' + status
    rotation = choose_rotation(ctx, ctx.fax_in_base())
    if rotation is None:
        ctx.event('pipeline', ok=False, reason='grasp not reachable from stance'); return 'no_grasp_pose'
    if do_press:
        yield from press_button(ctx, rotation)
    grasped = yield from grasp_from_floor(ctx, rotation)
    if not grasped:
        return 'grasp_failed'
    carried = yield from to_carry(ctx, rotation)
    if not carried:
        return 'carry_failed'
    chosen = choose_site(ctx)
    if chosen is None:
        ctx.event('pipeline', ok=False, reason='no reachable desk'); return 'no_site'
    key, site = chosen
    ctx.event('site', carrel=key, entry=list(site['entry']), stance=list(site['stance']))
    status = yield from drive(ctx, site['entry'], site['yaw'], speed=.45)
    if status != 'arrived':
        return 'entry_' + status
    # The chair in the nook is on castors; the base pushes it aside while moving in slowly.
    status = yield from drive(ctx, site['stance'], site['yaw'], speed=.2, direct=True, tolerance=.05, yaw_tolerance=.08)
    ctx.event('in_nook', status=status, base=[round(float(v), 3) for v in ctx.est.base()])
    placed = yield from place_on_desk(ctx, site['target'])
    if not placed:
        return 'place_failed'
    yield from hold(ctx, 20)
    return 'done'
