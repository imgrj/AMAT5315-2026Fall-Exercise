"""Fetch only the sources pinned by the learning sheet; record SHA256 hashes."""
from _common import ROOT, save_json
import hashlib
import urllib.request

SOURCES={
 'model.safetensors':'https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/model.safetensors',
 'tokenizer.json':'https://huggingface.co/minishlab/potion-base-8M/resolve/bf8b056/tokenizer.json',
 'TinyStoriesV2-GPT4-valid.txt':'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/TinyStoriesV2-GPT4-valid.txt',
 'tok512.model':'https://huggingface.co/karpathy/tinyllamas/resolve/0bd21da/stories260K/tok512.model',
 'chinchilla-runs.csv':'https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/92258837425e1b5f2851d624287f0120583a3d0e/data/svg_extracted_data.csv',
}

if __name__=='__main__':
    manifest={}
    for name,url in SOURCES.items():
        path=ROOT/'data'/name
        if not path.exists():
            print('Downloading',name,flush=True)
            temporary=path.with_suffix(path.suffix+'.partial')
            urllib.request.urlretrieve(url,temporary)
            temporary.replace(path)
        manifest[name]={'url':url,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        print(name,manifest[name]['bytes'],flush=True)
    save_json('sources.json',manifest)
