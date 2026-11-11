"""Verify the published Week 10 lab against the committed artifacts.

Run from week10/: python scripts/verify_publication.py

Waits for the GitHub Pages build of the current commit, downloads every deployed
file and compares it with docs/ (line endings normalised for text files).
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
    'week10-tensor-lab.html',
    'assets/week10/svd.png',
    'assets/week10/boundary-critical.png',
    'assets/week10/trg-critical.png',
    'assets/week10/g-curves.png',
    'assets/week10/phase-compare.png',
    'assets/week10/boundary-rows.png',
    'assets/week10/trg-steps.png',
    'index.html',
]


def latest_build(repo: str, attempts: int = 12) -> dict:
    """Ask GitHub for the Pages build of HEAD, retrying a flaky network."""
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip()
    last = ''
    for attempt in range(attempts):
        try:
            build = json.loads(subprocess.check_output(
                ['gh', 'api', f'repos/{repo}/pages/builds/latest'], text=True))
        except subprocess.CalledProcessError as error:
            last = error.stderr or 'gh api failed'
        else:
            if build['status'] == 'built' and build['commit'] == commit:
                return build
            last = f"{build['status']} for {build['commit'][:7]}"
        time.sleep(10)
    raise AssertionError(f'no built Pages deployment for HEAD: {last}')


def main() -> None:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT.parent, text=True).strip()
    build = latest_build('imgrj/AMAT5315-2026Fall-Exercise')
    print(f"pages build {build['status']} for {build['commit'][:7]} (site root {BASE})")

    def check(path: str) -> dict:
        local = (ROOT.parent / 'docs' / path).read_bytes()
        last_error = None
        status = None
        remote = b''
        for attempt in range(4):
            request = Request(BASE + path + '?verify=' + build['commit'] + f'&attempt={attempt}',
                              headers={'Cache-Control': 'no-cache'})
            try:
                with urlopen(request, timeout=90) as response:
                    status, remote = response.status, response.read()
                break
            except Exception as error:  # transient CDN reads happen; retry the file
                last_error = error
                time.sleep(3)
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
        'url': BASE + 'week10-tensor-lab.html',
        'artifact_commit': build['commit'],
        'pages_build_status': build['status'],
        'pages_build_created_at': build['created_at'],
        'checked_from': commit,
        'files': files,
        'comparison': 'Exact bytes for PNG, CRLF/LF normalised for HTML',
    }
    common_path = ROOT / 'evidence' / 'publication.json'
    common_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'artifact_commit': build['commit'][:7],
                      'status': build['status'],
                      'matching_live_files': len(files),
                      'page': record['url']}, indent=2))


if __name__ == '__main__':
    main()
