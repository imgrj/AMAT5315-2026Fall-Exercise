"""Fixed story split and SentencePiece tokens, without validation leakage."""
from pathlib import Path
import numpy as np
import sentencepiece as spm


def split_stories(text, fraction=.9):
    stories=[s.strip() for s in text.split('<|endoftext|>') if s.strip()]
    cut=int(len(stories)*fraction)
    return stories[:cut],stories[cut:]


def load_corpus(root):
    root=Path(root)
    tokenizer=spm.SentencePieceProcessor(model_file=str(root/'data/tok512.model'))
    train,valid=split_stories((root/'data/TinyStoriesV2-GPT4-valid.txt').read_text(encoding='utf-8'))
    cache=root/'artifacts/tokens.npz'
    if cache.exists():
        with np.load(cache) as z:
            a,b=z['train'],z['valid']
    else:
        def encode(stories):
            output=[tokenizer.eos_id()]
            for ids in tokenizer.encode(stories):
                output.extend(ids);output.append(tokenizer.eos_id())
            return np.asarray(output,dtype=np.int32)
        a,b=encode(train),encode(valid)
        np.savez(cache,train=a,valid=b)
    roundtrip=[i for i,s in enumerate(valid) if tokenizer.decode(tokenizer.encode(s))!=s]
    info={'train_stories':len(train),'valid_stories':len(valid),'train_tokens':len(a),
          'valid_tokens':len(b),'train_characters':sum(map(len,train)),
          'valid_characters':sum(map(len,valid)), 'vocab':tokenizer.vocab_size(),
          'eos':tokenizer.eos_id(),'roundtrip_exception_indices':roundtrip,
          'characters_per_token':sum(map(len,valid))/(len(b)-1)}
    return a,b,tokenizer,info
