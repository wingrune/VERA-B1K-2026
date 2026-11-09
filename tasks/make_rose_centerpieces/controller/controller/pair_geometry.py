"""Conditional pairing from supported RGB-D tracks, never Native object initialization."""
import numpy as np

def pair_condition(candidates,vase_body,level,step,base=None):
    supported=[]
    for c in candidates:
        state=c.get('state','available');held=state in ('held','assigned')
        if state not in ('available','held','assigned') or step-int(c.get('seen_step',-10000))>(25 if held else 45):continue
        if held and not c.get('pickup_confirmed',False):continue
        if held:
            if 'source_body_pose' in c:source=np.asarray(c['source_body_pose'])
            elif 'source_world_pose' in c and base is not None:source=np.linalg.inv(base)@np.asarray(c['source_world_pose'])
            else:continue
        else:source=np.asarray(c['body_pose'])
        stats=c.get('stats',{});depth=stats.get('depth_support',stats.get('visible_support',{}))
        if not held and (depth.get('matched_fraction',0)<.65 or depth.get('unoccluded_fraction',0)<=.50):continue
        pose=np.asarray(c['body_pose'])
        if pose.shape!=(4,4) or not np.isfinite(pose).all():continue
        supported.append((c,pose,source))
    if len(supported)<2:return {'eligible':False,'reason':'fewer_than_two_distinct_supported_tracks','pairs':[]}
    vp=(level@np.asarray(vase_body))[:2,3];pairs=[]
    for i,(a,ap,asp) in enumerate(supported):
        for b,bp,bsp in supported[i+1:]:
            if a['track_id']==b['track_id']:continue
            # Distinct physical roots and distinct pickup locations, not view aliases.
            if np.linalg.norm(asp[:3,3]-bsp[:3,3])<.065:continue
            ax=(level@asp)[:2,3];bx=(level@bsp)[:2,3];distance=float(np.linalg.norm(ax-bx))
            if distance>.35:continue
            av=ax-vp;bv=bx-vp;ar=np.linalg.norm(av);br=np.linalg.norm(bv)
            if min(ar,br)<.10 or max(ar,br)>1.20:continue
            cosine=float(av@bv/(ar*br))
            if cosine<.70:continue
            for left,right,lp,rp in [(a,b,ap,bp),(b,a,bp,ap)]:
                pairs.append({'left_id':left['track_id'],'right_id':right['track_id'],'left_pose':lp.tolist(),'right_pose':rp.tolist(),'distance_m':distance,'vase_side_cosine':cosine,'requires_fullq_ik_and_swept_dual_payload_validation':True})
    pairs.sort(key=lambda p:p['distance_m'])
    return {'eligible':bool(pairs),'reason':'geometric_pair_requires_fk_validation' if pairs else 'no_nearby_same_side_distinct_pair','pairs':pairs}
