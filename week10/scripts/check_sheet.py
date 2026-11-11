"""Check the Week 10 learning sheet against the official copy and record provenance.

Run from week10/: python scripts/check_sheet.py [--offline]
Downloads the official PDF with a cache-busting query, compares it byte for byte
with the committed `week10-learning-sheet.pdf`, extracts its text, and writes
`evidence/sheet.json`.
"""
from __future__ import annotations

import argparse
import time
from hashlib import sha256

from pypdf import PdfReader

import _common as common

LOCAL = common.ROOT / 'week10-learning-sheet.pdf'
OFFICIAL = 'https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/week10-learning-sheet.pdf'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='skip the download')
    arguments = parser.parse_args()
    local_bytes = LOCAL.read_bytes()
    reader = PdfReader(str(LOCAL))
    pages = [page.extract_text() or '' for page in reader.pages]
    text = '\n'.join(pages)
    record = {
        'path': 'week10-learning-sheet.pdf',
        'bytes': len(local_bytes),
        'sha256': sha256(local_bytes).hexdigest(),
        'pages': len(pages),
        'text_characters': len(text),
        'text_sha256': sha256(text.encode('utf-8')).hexdigest(),
        'official_url': OFFICIAL,
        'checked': time.strftime('%Y-%m-%d'),
    }
    if arguments.offline:
        record['download'] = 'skipped (--offline)'
    else:
        from urllib.request import Request, urlopen
        request = Request(f'{OFFICIAL}?v={int(time.time())}',
                          headers={'Cache-Control': 'no-cache', 'Pragma': 'no-cache'})
        with urlopen(request, timeout=90) as response:
            official = response.read()
        record['official'] = {
            'bytes': len(official),
            'sha256': sha256(official).hexdigest(),
            'identical': official == local_bytes,
        }
    common.write_json(common.EVIDENCE / 'sheet.json', record)
    print(f"{record['pages']} pages, {record['bytes']} bytes, sha256 {record['sha256'][:16]}...")
    if 'official' in record:
        print(f"official copy identical: {record['official']['identical']}")


if __name__ == '__main__':
    main()
