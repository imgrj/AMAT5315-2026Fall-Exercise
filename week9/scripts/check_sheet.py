"""Check the Week 9 learning sheet against the official copy and record provenance.

Run from week9/: python scripts/check_sheet.py [--offline]
Downloads the official PDF with a cache-busting query, compares it with the
committed `week9-learning-sheet.pdf` byte for byte, extracts its text, and
writes evidence/sheet.json.
"""
from __future__ import annotations

import argparse
import json
import time
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / 'week9-learning-sheet.pdf'
OFFICIAL = 'https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week9-learning-sheet.pdf'


def extract(path: Path) -> tuple[list[str], dict]:
    reader = PdfReader(str(path))
    pages = [page.extract_text() for page in reader.pages]
    return pages, dict(reader.metadata or {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='skip the download')
    arguments = parser.parse_args()

    local_bytes = LOCAL.read_bytes()
    local_pages, metadata = extract(LOCAL)
    text = '\n'.join(local_pages)
    record = {
        'path': 'week9-learning-sheet.pdf',
        'bytes': len(local_bytes),
        'sha256': sha256(local_bytes).hexdigest(),
        'pages': len(local_pages),
        'text_characters': len(text),
        'text_sha256': sha256(text.encode('utf-8')).hexdigest(),
        'creator': metadata.get('/Creator'),
        'mod_date': metadata.get('/ModDate'),
        'official_url': OFFICIAL,
        'checked': time.strftime('%Y-%m-%d'),
    }

    if arguments.offline:
        record['download'] = 'skipped (--offline)'
    else:
        request = Request(f'{OFFICIAL}?v={int(time.time())}',
                          headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})
        with urlopen(request, timeout=90) as response:
            official = response.read()
        record['official'] = {
            'bytes': len(official),
            'sha256': sha256(official).hexdigest(),
            'identical': official == local_bytes,
        }

    path = ROOT / 'evidence' / 'sheet.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
