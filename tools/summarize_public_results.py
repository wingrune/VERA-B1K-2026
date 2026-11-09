#!/usr/bin/env python3
"""Summarize an immutable local Hugging Face checkout without downloading videos."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
DATASET = 'VERA-VLA/b1k_final_submission'
FOLDER = 'standard.public.VERA_team.team.20261015'


def summarize(checkout):
    revision = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    # Read committed blobs, so a modified working copy cannot silently change the evidence.
    paths = subprocess.check_output(['git', '-C', str(checkout), 'ls-tree', '-r', '--name-only', revision], text=True).splitlines()
    videos = {p for p in paths if p.startswith(FOLDER + '/videos/') and p.endswith('.mp4')}
    ids = json.loads((REPO / 'server/task_ids_2026.json').read_text())
    routes = {r['task']: r['solution'] for r in csv.DictReader((REPO / 'config/routing.csv').open())}
    grouped = collections.defaultdict(list)
    manifest = []
    for path in sorted(p for p in paths if p.startswith(FOLDER + '/json/') and p.endswith('.json')):
        raw = subprocess.check_output(['git', '-C', str(checkout), 'show', f'{revision}:{path}'])
        row = json.loads(raw)
        task, instance = row['task'], row['instance_id']
        q = row['q_score']['final']
        assert task in ids and type(instance) is int and 301 <= instance <= 310, path
        assert row['rollout_id'] == 0 and type(row['success']) is bool, path
        assert type(q) in (float, int) and math.isfinite(q) and 0 <= q <= 1, path
        assert not row['success'] or q == 1, path
        assert Path(path).name == f'{task}_{instance}_0.json', path
        video = FOLDER + '/videos/' + Path(path).stem + '.mp4'
        assert video in videos, f'Missing video entry: {video}'
        grouped[task].append({'instance': instance, 'q': q, 'success': row['success'], 'steps': row['steps']})
        manifest.append({'path': path, 'sha256': hashlib.sha256(raw).hexdigest(), 'video_path': video})
    tasks = []
    for task in sorted(ids, key=ids.get):
        rows = sorted(grouped[task], key=lambda r: r['instance'])
        if rows:
            assert [r['instance'] for r in rows] == list(range(301, 311)), task
        tasks.append({'task': task, 'id': ids[task], 'release_route': routes[task], 'episodes': len(rows),
                      'mean_q': sum(r['q'] for r in rows) / len(rows) if rows else None,
                      'successes': sum(r['success'] for r in rows), 'instances': rows})
    published = [t for t in tasks if t['episodes']]
    summary = {'dataset': DATASET, 'revision': revision, 'folder': FOLDER,
               'source_commit': '309bb650dc1606319a4309d5b0ab64702f8ad6ab',
               'scope': 'Published public instances 301-310, rollout 0. Release routes are not per-episode policy provenance. No leaderboard score is asserted.',
               'total_tasks': len(ids), 'published_tasks': len(published), 'episodes': len(manifest),
               'video_entries': len(manifest), 'successes': sum(t['successes'] for t in published),
               'mean_q_published_tasks': sum(t['mean_q'] for t in published) / len(published),
               'routing_counts': dict(sorted(collections.Counter(routes[t] for t in ids).items())), 'tasks': tasks}
    return summary, {'dataset': DATASET, 'revision': revision, 'files': manifest,
                     'video_verification': 'File presence in the dataset Git tree only; video bytes were not downloaded or decoded.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    parser.add_argument('--output', type=Path, default=REPO / 'docs/data')
    args = parser.parse_args()
    summary, manifest = summarize(args.checkout)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in [('public-results.json', summary), ('evidence-manifest.json', manifest)]:
        (args.output / name).write_text(json.dumps(data, indent=2) + '\n')
    print(f"{summary['published_tasks']} tasks; {summary['episodes']} episodes; mean Q {summary['mean_q_published_tasks']:.6f} (published tasks only)")
