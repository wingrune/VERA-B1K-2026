"""Grasp affordances computed from cached, object-local training geometry."""
import numpy as np


def grasp_regions(category, model):
    xyz=model['xyz'];lo,hi=model['bounds'];extent=hi-lo;regions=[]
    if category=='rose':
        rgb=model['rgb'].astype(float)
        flower=xyz[(rgb[:,0]>70)&(rgb[:,0]>rgb[:,1]*1.5)&(rgb[:,0]>rgb[:,2]*1.5)]
        if len(flower)>30:
            center=(flower.min(0)+flower.max(0))/2
            for axis in range(3):
                width=np.ptp(flower[:,axis])
                if .015<width<.085:
                    regions.append((center,axis,float(width),f'blossom_body_{axis}'))
    elif category=='vase':
        # Pinch a segment of the upper wall, rather than the complete diameter.
        upright=int(np.argmax(extent));other=[a for a in range(3) if a!=upright]
        rim=xyz[xyz[:,upright]>hi[upright]-.045]
        for axis in other:
            for sign in (-1,1):
                band=rim[rim[:,axis]<lo[axis]+.035] if sign<0 else rim[rim[:,axis]>hi[axis]-.035]
                if len(band)<15:continue
                center=np.median(band,axis=0);width=np.ptp(band[:,axis])
                if .003<width<.065:regions.append((center,axis,float(width),f'vase_wall_{axis}_{sign}'))
    elif category=='half_banana':
        long=int(np.argmax(extent[:2]));closing=1-long
        for fraction in (.3,.5,.7):
            section=xyz[abs(xyz[:,long]-(lo[long]+fraction*extent[long]))<.012]
            if len(section)<30:continue
            center=(section.min(0)+section.max(0))/2;width=np.ptp(section[:,closing])
            if .015<width<.09:regions.append((center,closing,float(width),f'banana_section_{fraction}'))
    elif category=='half_pomegranate':
        section=xyz[xyz[:,2]>lo[2]+.60*extent[2]]
        if len(section)>30:
            center=(section.min(0)+section.max(0))/2
            for axis in (0,1):
                width=np.ptp(section[:,axis])
                if .015<width<.095:regions.append((center,axis,float(width),'pomegranate_upper_body'))
    elif category=='cabinet_door':
        # The raised handle projects out of the otherwise thin door panel.
        points=xyz[xyz[:,0]>lo[0]+.65*extent[0]]
        if len(points)>30:
            center=np.median(points,axis=0);width=np.ptp(points[:,1])
            if .002<width<.085:
                regions.append((center,1,float(width),'raised_handle'))
    elif category=='digital_camera':
        # The forward lens barrel is narrower than the complete camera body.
        points=xyz[xyz[:,0]>lo[0]+.72*extent[0]]
        if len(points)>50:
            center=(points.min(0)+points.max(0))/2;width=np.ptp(points[:,1])
            if .02<width<.095:
                regions.append((center,1,float(width),'lens_barrel'))
    elif category=='facsimile':
        # Pinch the raised handset, leaving the wide base below the fingers.
        points=xyz[xyz[:,2]>lo[2]+.80*extent[2]]
        if len(points)>50:
            middle=points[abs(points[:,1]-(lo[1]+hi[1])/2)<.025]
            if len(middle)>25:
                center=(middle.min(0)+middle.max(0))/2;width=np.ptp(middle[:,0])
                if .015<width<.095:
                    regions.append((center,0,float(width),'raised_handset'))
    return regions

def supported_aperture_minimum(nominal_width):
    """Require object-supported separation, allowing narrow stem/wall contact.

    Blossom envelopes describe the whole grasp region; real fingers can pinch
    its narrower stem or petals. Empty full closure still fails this floor.
    Independent observed rise and stable relative motion remain mandatory.
    """
    return max(.006,min(.025,.15*float(nominal_width)))
