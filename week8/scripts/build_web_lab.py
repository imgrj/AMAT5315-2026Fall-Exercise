"""Build the existing GitHub Pages lab using measured runs and trained logits."""
from _common import ROOT,np
from src.data import load_corpus
from src.training import load_checkpoint
from src.models import transformer
import jax.numpy as jnp
import json
import shutil

GALLERY=[
 ('part1-directions.png','1 / Meaning as geometry','part1_embeddings.py'),
 ('part2-counts.png','2 / Counts run out of data','part2_counts.py'),
 ('part3-interference.png','3 / Interference grows with memory','part3_memory.py'),
 ('part3-capacity.png','3 / Capacity grows with dimension','part3_memory.py'),
 ('part3-mlp.png','3 / MLP learns the bigram table','part3_mlp.py'),
 ('part4-recall.png','4 / Attention recalls the picture','part4_attention.py'),
 ('part5-training.png','5 / Training the transformer','part5_transformer.py'),
 ('part5-comparison.png','5 / Same stories, better predictions','part5_evaluate.py'),
 ('part5-context.png','5 / Context lowers uncertainty','part5_evaluate.py'),
 ('part5-consistency.png','5 / Samples match the model','part5_evaluate.py'),
 ('part5-attention.png','5 / Eight attention heads','part5_evaluate.py'),
 ('part6-fit.png','6 / Refit the published runs','part6_scaling.py'),
 ('part6-residuals.png','6 / Printed parameters miss','part6_scaling.py'),
 ('part6-allocation.png','6 / Spend the compute budget','part6_scaling.py'),
]


def main():
    docs=ROOT.parent/'docs';assets=docs/'assets/week8';assets.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((ROOT/'evidence'/name).read_text())
    _,_,tok,_=load_corpus(ROOT);params=load_checkpoint(ROOT/'artifacts/transformer.pkl')
    prefixes=['Once upon a time, there was a','Lily went to the','The little dog was','They lived happily ever']
    contexts=[{'text':s,'logits':np.asarray(transformer(params,jnp.array([tok.encode(s)])))[0,-1].round(7).tolist()} for s in prefixes]
    with np.load(ROOT/'artifacts/picture-recall.npz') as z:gallery=z['gallery'].round(5).tolist()
    data={'contexts':contexts,'pieces':[tok.id_to_piece(i).replace('▁','·') for i in range(512)],
          'recalls':gallery,'memory':read('part4_metrics.json'),'scaling':read('part6_metrics.json'),
          'transformer':read('part5_metrics.json'),'counts':read('part2_metrics.json'),'gallery':GALLERY}
    for name,_,_ in GALLERY:shutil.copyfile(ROOT/'evidence'/name,assets/name)
    template=(ROOT/'web/lab_template.html').read_text(encoding='utf-8')
    rendered=template.replace('__DATA__',json.dumps(data,ensure_ascii=True).replace('</','<\\/'))
    (docs/'week8-language-lab.html').write_text(rendered,encoding='utf-8')
    print('Built docs/week8-language-lab.html with',len(GALLERY),'evidence figures')


if __name__=='__main__':main()
