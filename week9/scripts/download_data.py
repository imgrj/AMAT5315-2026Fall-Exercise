"""Download the two pinned Week 9 data sets and verify their SHA256 digests.

Run from week9/: python scripts/download_data.py
Files land in week9/asia.uai and week9/promedus/ and stay out of Git.
"""
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

ASIA_BASE = 'https://raw.githubusercontent.com/TensorBFS/TensorInference.jl'
ASIA_REV = '935ed86f8e473f523cac9461c45a1f470e74f0f9'
PROMEDUS_BASE = 'https://raw.githubusercontent.com/mroavi/uai-2014-inference-competition'
PROMEDUS_REV = '22005844423f5bca62798548df808cc418fc0d7d'

# (url, relative destination, sha256, size in bytes)
FILES = [
    (f'{ASIA_BASE}/{ASIA_REV}/examples/asia-network/model.uai', 'asia.uai',
     '0d78314c96b1cbbfd94682f0e99f73461cb50a935a1da85cbc814c70e3fff32d', 265),
    (f'{PROMEDUS_BASE}/{PROMEDUS_REV}/uai2014/PR/Promedus_14.uai', 'promedus/Promedus_14.uai',
     'f213b76a7f145a06f39daa9061408981dbd525e4ce2e5156edb83a7073d71349', 48170),
    (f'{PROMEDUS_BASE}/{PROMEDUS_REV}/uai2014/PR/Promedus_14.uai.evid', 'promedus/Promedus_14.uai.evid',
     'e345a4ebecf02cdd9fdaf42e7456afa5f6ecafa989b6a3c15ef525abe736a403', 47),
    (f'{PROMEDUS_BASE}/{PROMEDUS_REV}/uai2014/PR/Promedus_14.uai.PR', 'promedus/Promedus_14.uai.PR',
     'd52ca345f79e79f0385f7bd91a3c2e3521eff165b3dcb7b29be1e56ca5f96dfc', 12),
    (f'{PROMEDUS_BASE}/{PROMEDUS_REV}/uai2014/MAR/Promedus_14.uai.MAR', 'promedus/Promedus_14.uai.MAR',
     '06e3145bb59fd0a22af235af43bc7c45bee71a764647e152314bdded4f0d256f', 8607),
]


def fetch(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def main() -> None:
    for url, name, digest, size in FILES:
        path = ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and sha256(path.read_bytes()).hexdigest() == digest:
            print(f'cached  {name}  {size} bytes')
            continue
        data = fetch(url)
        assert len(data) == size, f'{name}: expected {size} bytes, got {len(data)}'
        actual = sha256(data).hexdigest()
        assert actual == digest, f'{name}: sha256 {actual} != {digest}'
        path.write_bytes(data)
        print(f'fetched {name}  {size} bytes  {digest[:12]}...')


if __name__ == '__main__':
    main()
