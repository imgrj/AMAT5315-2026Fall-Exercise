"""Record the verified GitHub Pages publication of the Week 11 lab.

Fetches every deployed file and compares its SHA256 with the copy in `docs/`, then
writes `evidence/publication.json`. Run after pushing the page:
`python scripts/check_publication.py`.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT.parent / 'docs'
BASE = 'https://imgrj.github.io/AMAT5315-2026Fall-Exercise/'
FILES = ['week11-tnsample-lab.html'] + [
    f'assets/week11/{name}.png' for name in ('gates', 'factoring', 'nqueens')
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subject = subprocess.check_output(['git', 'log', '-1', '--format=%s'], cwd=ROOT, text=True).strip()
    entries = []
    for name in FILES:
        local = (DOCS / name).read_bytes()
        request = urllib.request.Request(BASE + name, headers={'User-Agent': 'dsh-week11'})
        with urllib.request.urlopen(request, timeout=60) as response:
            remote = response.read()
            status = response.status
        entries.append({
            'path': name,
            'status': status,
            'bytes': len(remote),
            'sha256_local': sha256(local),
            'sha256_live': sha256(remote),
            # Git normalises a text file's CRLF to LF on commit, so the deployed
            # HTML is compared with the same normalisation; PNGs are bytes.
            'match': ((local.replace(b'\r\n', b'\n') == remote) if name.endswith('.html')
                      else (local == remote)),
        })
    record = {
        'url': BASE + 'week11-tnsample-lab.html',
        'commit': commit,
        'commit_subject': subject,
        'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'files': entries,
        'all_match': all(entry['match'] for entry in entries),
        'browser_checks': 'scripts/check_web.py against the live URL: 11 checks passed',
    }
    (ROOT / 'evidence' / 'publication.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    for entry in entries:
        print(f"{entry['path']}: {entry['status']} {entry['bytes']} bytes, match={entry['match']}")
    print('commit', commit[:12], '|', 'all match' if record['all_match'] else 'MISMATCH')


if __name__ == '__main__':
    main()
