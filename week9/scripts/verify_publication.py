"""Verify the published Week 9 lab against the committed artifacts.

Run from week9/: python scripts/verify_publication.py
Checks the GitHub Pages build for the current commit and compares every
deployed file with docs/ (line endings normalised for text files).
Writes evidence/publication.json.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://imgrj.github.io/AMAT5315-2026Fall-Exercise/'
FILES = [
    'week9-inference-lab.html',
    'assets/week9/graphs.svg',
    'assets/week9/posteriors.png',
    'assets/week9/joint.png',
    'assets/week9/orders.png',
    'assets/week9/bp-convergence.png',
    'assets/week9/bp-comparison.png',
    'assets/week9/bp-accuracy.png',
    'assets/week9/promedus-orders.png',
    'assets/week9/promedus-bp.png',
    'index.html',
]


def main() -> None:
    root = ROOT.parent
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    build = json.loads(subprocess.check_output(
        ['gh', 'api', 'repos/imgrj/AMAT5315-2026Fall-Exercise/pages/builds/latest'], text=True))
    assert build['status'] == 'built', build['status']
    print(f"pages build {build['status']} for {build['commit'][:7]} (site root {BASE})")

    def check(path: str) -> dict:
        local = (root / 'docs' / path).read_bytes()
        last_error = None
        for attempt in range(3):
            request = Request(BASE + path + '?verify=' + build['commit'] + f'&attempt={attempt}',
                              headers={'Cache-Control': 'no-cache'})
            try:
                with urlopen(request, timeout=90) as response:
                    status, remote = response.status, response.read()
                break
            except Exception as error:  # transient CDN reads happen; retry the file
                last_error = error
                time.sleep(2)
        else:
            raise AssertionError(f'{path}: {last_error}')
        if path.endswith(('.html', '.svg')):
            local = local.replace(b'\r\n', b'\n')
            remote = remote.replace(b'\r\n', b'\n')
        assert status == 200 and remote == local, path
        return {'path': path, 'status': status, 'sha256': hashlib.sha256(remote).hexdigest()}

    with ThreadPoolExecutor(max_workers=5) as executor:
        files = list(executor.map(check, FILES))

    record = {
        'url': BASE + 'week9-inference-lab.html',
        'artifact_commit': build['commit'],
        'pages_build_status': build['status'],
        'pages_build_created_at': build['created_at'],
        'checked_from': commit,
        'files': files,
        'comparison': 'Exact bytes for PNG, CRLF/LF normalised for HTML and SVG',
    }
    path = ROOT / 'evidence' / 'publication.json'
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'artifact_commit': build['commit'][:7],
                      'status': build['status'],
                      'matching_live_files': len(files),
                      'page': record['url']}, indent=2))


if __name__ == '__main__':
    main()
