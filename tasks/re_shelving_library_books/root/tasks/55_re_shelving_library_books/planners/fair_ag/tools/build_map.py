"""Offline builder of the static scene map of the task 55 fair planner, from observation dumps of TRAINING episodes only.

usage: python build_map.py --out <fair_ag dir> <episode dir> [<episode dir> ...]
  every episode dir holds obs/obs_*.npz written by run_episode.py --dump-every (head depth, cam_rel, proprio, and the harness truth of the
  same control). The truth is used here, offline, as the pose of the mapping camera; the planner never reads it at test time.

Writes <out>/map_data.py (a Python literal module the planner imports) and <out>/data/scene_map_debug.npz:
  occupancy   5 cm cells hit by head points 0.10-1.40 m above the floor (walls, furniture, the desk, the bookcases), by count and ratio
  seen        cells seen as floor or as an obstacle
  table, cases, cubbies  desk and bookcase boxes from the training truth; cubbies from the bookcase asset (container boxes relative to the case)
  books       extents of the three book models (root-frame boxes), the 3D-model prior
  base_z      height of the robot frame origin above the map origin
  desk_top_visual_z  height of the desk top surface in the training depth
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4]))
sys.path.insert(0, str(HERE.parents[1]))

from core.skills_lib import r1pro_kinematics as K  # noqa: E402

RES = 0.05
BAND = (0.10, 1.40)
FLOOR_BAND = 0.03
MIN_HITS, MIN_RATIO = 4, 0.25
BOOKS = ('book1', 'book2', 'book3')
CASES = ('bookcase1', 'bookcase2', 'bookcase3', 'bookcase4')


def load_cloud_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('fair55_cloud', HERE.parent / 'cloud.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rle(flat: np.ndarray) -> list:
    """Run lengths of a boolean array, starting with a run of False."""
    out, cur, n = [], False, 0
    for v in flat.astype(bool):
        if v == cur:
            n += 1
        else:
            out.append(n)
            cur, n = v, 1
    out.append(n)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--asset-geometry', required=True, help='scene_geometry.json of the bookcase asset containers (the export of the scene)')
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('episodes', nargs='+')
    args = ap.parse_args(argv)
    C = load_cloud_module()
    files = sorted(f for e in args.episodes for f in glob.glob(str(Path(e) / 'obs' / 'obs_*.npz')))
    print('dumps', len(files))
    pts_all, floor_all = [], []
    truth = dict(table=[], cases={c: [] for c in CASES}, books={b: [] for b in BOOKS}, base_z=[])
    top_z, top_pts = [], []
    for f in files:
        d = np.load(f)
        q = K.q_from_proprio(np.asarray(d['proprio'], float))
        base, bz = np.asarray(d['truth_base'], float), float(d['truth_base_z'])
        if 'depth_head' in d.files:
            depth, stride = d['depth_head'], args.stride
        else:                                   # slim dump: every third pixel of the head depth, put back at its place in the image
            depth = np.full((720, 720), np.inf, np.float32)
            depth[::3, ::3] = d['depth_head_s3']
            stride = 3
        p = C.camera_points(depth, 'head', d['cam_rel'], q, stride, max_range=5.0)
        w = C.to_map(p, base, bz)
        pts_all.append(w[(w[:, 2] > BAND[0]) & (w[:, 2] < BAND[1])][:, :2])
        floor_all.append(w[np.abs(w[:, 2]) < FLOOR_BAND][:, :2])
        tlo, thi = np.asarray(d['truth_table_lo']), np.asarray(d['truth_table_hi'])
        truth['table'].append(np.r_[tlo, thi])
        on = w[(w[:, 0] > tlo[0] + 0.03) & (w[:, 0] < thi[0] - 0.03) & (w[:, 1] > tlo[1] + 0.03) & (w[:, 1] < thi[1] - 0.03)
               & (np.abs(w[:, 2] - thi[2]) < 0.04)]
        top_z.append(on[:, 2])
        top_pts.append(w[(w[:, 0] > tlo[0] - 0.08) & (w[:, 0] < thi[0] + 0.08) & (w[:, 1] > tlo[1] - 0.08) & (w[:, 1] < thi[1] + 0.08)
                         & (np.abs(w[:, 2] - thi[2]) < 0.04)])
        for c in CASES:
            truth['cases'][c].append(np.r_[d[f'truth_{c}_lo'], d[f'truth_{c}_hi']])
        for b in BOOKS:
            truth['books'][b].append(np.r_[d[f'truth_{b}_local_lo'], d[f'truth_{b}_local_hi']])
        truth['base_z'].append(bz)
    pts, flo = np.concatenate(pts_all), np.concatenate(floor_all)
    lo = np.floor(np.minimum(pts.min(0), flo.min(0)) / RES) * RES - 0.5
    hi = np.ceil(np.maximum(pts.max(0), flo.max(0)) / RES) * RES + 0.5
    shape = tuple(int(round(v)) for v in (hi - lo) / RES)
    idx = lambda xy: np.clip(((xy - lo) / RES).astype(int), 0, np.array(shape) - 1)  # noqa: E731
    hits = np.zeros(shape, np.int32)
    floor = np.zeros(shape, np.int32)
    np.add.at(hits, tuple(idx(pts).T), 1)
    np.add.at(floor, tuple(idx(flo).T), 1)
    occ = (hits >= MIN_HITS) & (hits >= MIN_RATIO * (hits + floor))
    seen = occ | (floor >= 2)
    spread = lambda a: float(np.max(np.ptp(np.array(a), axis=0)))  # noqa: E731
    table = np.median(np.array(truth['table']), 0)
    cases = {c: np.median(np.array(v), 0) for c, v in truth['cases'].items()}
    books = {b: np.median(np.array(v), 0) for b, v in truth['books'].items()}
    asset = json.loads(Path(args.asset_geometry).read_text())
    cubbies = {}
    for c in CASES:
        ref = np.asarray(asset[c]['collision_box'][0], float)
        cubbies[c] = [(np.asarray(b, float).reshape(2, 3) - ref + cases[c][:3]).round(4).tolist() for b in asset[c]['containers']]
    tz = np.concatenate(top_z)
    desk_z = float(np.median(tz))
    tp = np.concatenate(top_pts)
    tp = tp[np.abs(tp[:, 2] - desk_z) < 0.010]
    edges = []
    for axis in (0, 1):
        other = 1 - axis
        inner = tp[(tp[:, other] > table[other] + 0.05) & (tp[:, other] < table[3 + other] - 0.05)]
        edges.append([round(float(np.percentile(inner[:, axis], 0.5)), 4), round(float(np.percentile(inner[:, axis], 99.5)), 4)])
    data = dict(
        res=RES, origin=lo.round(3).tolist(), shape=list(shape), occupancy_rle=rle(occ.ravel()), seen_rle=rle(seen.ravel()),
        table=[table[:3].round(4).tolist(), table[3:].round(4).tolist()],
        cases={c: dict(box=[v[:3].round(4).tolist(), v[3:].round(4).tolist()], cubbies=cubbies[c]) for c, v in cases.items()},
        books={b: [v[:3].round(5).tolist(), v[3:].round(5).tolist()] for b, v in books.items()},
        base_z=round(float(np.median(truth['base_z'])), 4),
        desk_top_visual_z=round(desk_z, 4),
        desk_visual_edges=dict(x=edges[0], y=edges[1]),
        dumps=len(files), episodes=[Path(e).name for e in args.episodes],
        spread=dict(table=spread(truth['table']), cases=max(spread(v) for v in truth['cases'].values()), books=max(spread(v) for v in truth['books'].values())),
    )
    out = Path(args.out)
    text = ('"""Static scene map of the task 55 fair planner, generated offline by tools/build_map.py from TRAINING episodes (see data/PROVENANCE.json).\n'
            'Do not edit by hand."""\n\nMAP = ' + json.dumps(data) + '\n')
    (out / 'map_data.py').write_text(text)
    (out / 'data').mkdir(exist_ok=True)
    np.savez_compressed(out / 'data' / 'scene_map_debug.npz', hits=hits, floor=floor, occ=occ, seen=seen, origin=lo, res=RES)
    print(json.dumps({k: v for k, v in data.items() if not k.endswith('_rle')})[:3000])
    print('occupied cells', int(occ.sum()), 'seen', int(seen.sum()), 'shape', shape, 'desk top z', data['desk_top_visual_z'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
