#!/usr/bin/env python3
"""Check local release links and consistency without downloading model weights."""
import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        row = dict(attrs)
        for field in ('href', 'src', 'poster'):
            if row.get(field):
                self.refs.append(row[field])
        if row.get('id'):
            assert row['id'] not in self.ids, f'Duplicate HTML ID: {row["id"]}'
            self.ids.add(row['id'])


def main():
    errors = []
    pages = [ROOT / 'docs/index.html']
    for path in pages:
        parser = Links()
        parser.feed(path.read_text())
        for ref in parser.refs:
            parsed = urlsplit(ref)
            if parsed.scheme or parsed.netloc:
                continue
            target = path.parent / unquote(parsed.path) if parsed.path else path
            if not target.exists():
                errors.append(f'{path.relative_to(ROOT)}: missing {ref}')
            if target == path and parsed.fragment and parsed.fragment not in parser.ids:
                errors.append(f'{path.relative_to(ROOT)}: missing anchor {ref}')
    for path in [ROOT / 'README.md', ROOT / 'THIRD_PARTY_NOTICES.md', ROOT / 'docs/reproduce.md', ROOT / 'docs/weights.md']:
        for ref in re.findall(r'\[[^\]]+\]\(([^)]+)\)', path.read_text()):
            parsed = urlsplit(ref)
            if not parsed.scheme and not (path.parent / unquote(parsed.path)).exists():
                errors.append(f'{path.relative_to(ROOT)}: missing {ref}')
    summary = json.loads((ROOT / 'docs/data/public-results.json').read_text())
    evidence = json.loads((ROOT / 'docs/data/evidence-manifest.json').read_text())
    routes = {r['task']: r['solution'] for r in csv.DictReader((ROOT / 'config/routing.csv').open())}
    assert summary['revision'] == evidence['revision']
    assert len(summary['tasks']) == summary['total_tasks'] == 100
    assert sum(bool(t['episodes']) for t in summary['tasks']) == summary['published_tasks']
    assert sum(t['episodes'] for t in summary['tasks']) == len(evidence['files']) == summary['episodes']
    assert sum(t['successes'] for t in summary['tasks']) == summary['successes']
    assert sum(summary['routing_counts'].values()) == 100
    for task in summary['tasks']:
        assert task['release_route'] == routes[task['task']]
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Release links, task counts, evidence revision and routing data are consistent.')


if __name__ == '__main__':
    main()
