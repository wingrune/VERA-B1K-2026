"""OFFLINE ONLY: canonical priors from template zero before any public reset.

Template object-local bounds mask training renders. Saved geometry and features
are object-local; no instance/world poses, contacts, goals or scores are saved.
This module is never imported by the online policy.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def export_priors(evaluator, output):
    import torch
    import omnigibson as og
    import omnigibson.lazy as lazy
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from omnigibson.eval.utils.eval_utils import CAMERA_INTRINSICS
    robot = evaluator.instance_eval_states[0].env_accessor.robot
    camera = robot.sensors[evaluator.robot_camera_names['head'].split('::')[1]]
    original_pose = tuple(x.clone() for x in camera.get_position_orientation())
    K = CAMERA_INTRINSICS['R1Pro']['head']
    sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=.008)
    manifest = {}
    scope = evaluator.instance_eval_states[0].env_accessor.object_scope
    for key, obj in scope.items():
        if not any(s in key for s in ('rose.n.', 'vase.n.', 'coffee_table.n.')):
            continue
        assert obj is not None
        category = obj.category
        folder = output / category
        folder.mkdir(exist_ok=True)
        pos, quat = obj.root_link.get_position_orientation()
        pos, rotation = array(pos), R.from_quat(array(quat)).as_matrix()
        vertices = []
        for prim in lazy.pxr.Usd.PrimRange(obj.root_link.prim):
            if not prim.IsA(lazy.pxr.UsdGeom.Mesh) or 'visual' not in str(prim.GetPath()).lower():
                continue
            points = np.asarray(lazy.pxr.UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
            transform = np.asarray(lazy.pxr.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(lazy.pxr.Usd.TimeCode.Default()))
            vertices.append(((np.c_[points, np.ones(len(points))] @ transform)[:, :3] - pos) @ rotation)
        assert vertices, category
        vertices = np.concatenate(vertices)
        bounds = np.array([vertices.min(0), vertices.max(0)])
        local_center = bounds.mean(0)
        target = pos + rotation @ local_center
        distance = max(.30, np.max(np.ptp(bounds, axis=0)) * 2.3)
        clouds, colors, feature_xyz, descriptors = [], [], [], []
        view = 0
        for elevation in (.30, .75, 1.15):
            for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                eye = target + distance * np.array([np.cos(angle)*np.cos(elevation), np.sin(angle)*np.cos(elevation), np.sin(elevation)])
                back = eye-target
                back /= np.linalg.norm(back)
                right = np.cross([0, 0, 1.], back)
                right /= np.linalg.norm(right)
                up = np.cross(back, right)
                camera_rot = np.column_stack([right, up, back])
                camera.set_position_orientation(position=torch.tensor(eye, dtype=torch.float32), orientation=torch.tensor(R.from_matrix(camera_rot).as_quat(), dtype=torch.float32))
                for _ in range(4):
                    og.sim.render()
                obs, info = camera.get_obs()
                rgb, depth = array(obs['rgb'])[..., :3], array(obs['depth_linear']).squeeze()
                height, width = depth.shape
                y, x = np.indices(depth.shape)
                points = np.stack([(x+.5-K[0,2])*depth/K[0,0], (y+.5-K[1,2])*depth/K[1,1], depth], axis=-1)
                xyz = ((points.reshape(-1,3) @ (camera_rot @ np.diag([1,-1,-1])).T + eye - pos) @ rotation).reshape(height,width,3)
                mask = np.isfinite(depth) & (depth > .03) & np.all((xyz >= bounds[0]+.002) & (xyz <= bounds[1]+.004),axis=-1)
                if mask.sum() < 50:
                    print('PRIOR_VIEW_REJECTED', category, view, int(mask.sum()), flush=True)
                    view += 1
                    continue
                clouds.append(xyz[mask][::3])
                colors.append(rgb[mask][::3])
                masked = rgb.copy()
                masked[~mask] = 0
                cv2.imwrite(str(folder / f'view_{view:02d}.png'), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
                kp, des = sift.detectAndCompute(rgb, mask.astype(np.uint8)*255)
                if des is not None:
                    xy = np.rint([k.pt for k in kp]).astype(int)
                    xy[:,0]=xy[:,0].clip(0,width-1);xy[:,1]=xy[:,1].clip(0,height-1)
                    valid = mask[xy[:,1],xy[:,0]]
                    feature_xyz.append(xyz[xy[valid,1],xy[valid,0]])
                    descriptors.append(des[valid])
                view += 1
        assert clouds and descriptors, ('No usable training views', category)
        xyz, rgb = np.concatenate(clouds), np.concatenate(colors)
        finite = np.isfinite(xyz).all(1)
        xyz, rgb = xyz[finite], rgb[finite]
        _, indices = np.unique(np.floor(xyz/.002).astype(int), axis=0, return_index=True)
        np.savez_compressed(folder/'model.npz', xyz=xyz[indices], rgb=rgb[indices], bounds=bounds)
        np.savez_compressed(folder/'features.npz', xyz=np.concatenate(feature_xyz), descriptors=np.concatenate(descriptors))
        metadata = dict(category=category, model=obj.model, scale=array(obj.scale).tolist(), links={},
                        source='training template zero, before public reset', world_poses_exported=False,
                        point_count=len(indices), feature_count=sum(len(d) for d in descriptors), bounds=bounds.tolist())
        (folder/'metadata.json').write_text(json.dumps(metadata, indent=2))
        manifest[category] = metadata
        print('PRIOR_EXPORTED', category, metadata['point_count'], metadata['feature_count'], flush=True)
    camera.set_position_orientation(position=original_pose[0],orientation=original_pose[1])
    for _ in range(4):
        og.sim.render()
    assert len(manifest) == 3, manifest.keys()
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
