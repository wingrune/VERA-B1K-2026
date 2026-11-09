"""Object-local training geometry; lid angle is fitted from current RGB-D."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_articulation(folder):
    folder = Path(folder)
    path = folder / 'articulation.json'
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    joint = metadata['joints']['j_link_0']['attributes']
    axis = np.eye(3)['XYZ'.index(joint['physics:axis'])]
    parent = R.from_quat(joint['physics:localRot0']).as_matrix()
    child = R.from_quat(joint['physics:localRot1']).as_matrix()
    return dict(body=np.load(folder / 'base_link.npz')['xyz'],
                lid=np.load(folder / 'link_0.npz')['xyz'],
                parent=parent, child=child, axis=axis,
                pivot=np.array(joint['physics:localPos0']),
                child_pivot=np.array(joint['physics:localPos1']),
                limits=np.deg2rad([joint['physics:lowerLimit'], joint['physics:upperLimit']]))


def articulated_points(prior, angle):
    rotation = prior['parent'] @ R.from_rotvec(prior['axis'] * angle).as_matrix() @ prior['child'].T
    lid = (prior['lid'] - prior['child_pivot']) @ rotation.T + prior['pivot']
    return np.concatenate([prior['body'], lid])
