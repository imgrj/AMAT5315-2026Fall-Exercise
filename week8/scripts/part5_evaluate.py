from _common import ROOT,save_json,figure,plt,np
from src.data import load_corpus
from src.models import transformer,mlp as mlp_forward
from src.training import load_checkpoint
from src.generation import probabilities,generate,draw_tokens
from src.sampling import distribution,chi_square
import jax
import jax.numpy as jnp
import json


def save_openings(metrics,counts,mlp):
    rows=[(r['bits_per_character'],r['model'],counts['samples_seed1848'][r['model']]) for r in counts['models'][:4]]
    rows += [(mlp['bits_per_character'],'MLP',metrics['mlp_opening_60_tokens']),
             (metrics['bits_per_character'],'Transformer',metrics['opening_60_tokens'])]
    lines=['60-token openings, T=1, seed 1848, EOS excluded for display.',
           'Ordered by held-out bits/character; control bytes shown as squares.',
           'The transformer uses JAX RNG; count models/MLP use NumPy RNG.', '']
    for bpc,name,sample in sorted(rows):
        visible=''.join(c if c.isprintable() else (' / ' if c=='\n' else '□') for c in sample)
        lines.extend([f'{name}: {bpc:.6f} bits/character',visible,''])
    (ROOT/'evidence/story-openings.txt').write_text('\n'.join(lines),encoding='utf-8')


def main():
    train,valid,tok,info=load_corpus(ROOT);params=load_checkpoint(ROOT/'artifacts/transformer.pkl')
    @jax.jit
    def score(p,x,y):
        logits=transformer(p,x)
        return -jnp.take_along_axis(jax.nn.log_softmax(logits),y[...,None],axis=-1)[...,0]
    count=(len(valid)-1)//64;windows=valid[:count*64].reshape(count,64);targets=valid[1:count*64+1].reshape(count,64)
    losses=[]
    for i in range(0,count,64):losses.append(np.asarray(score(params,jnp.asarray(windows[i:i+64]),jnp.asarray(targets[i:i+64]))))
    loss_array=np.concatenate(losses).astype(np.float64);position=loss_array.mean(0);se=loss_array.std(0,ddof=1)/np.sqrt(count)
    # Score the remaining suffix too, so the reported corpus loss includes all targets.
    remainder=len(valid)-1-count*64
    tail=np.asarray(score(params,jnp.asarray(valid[count*64:-1][None,:]),jnp.asarray(valid[count*64+1:][None,:]))) if remainder else np.array([])
    mean_loss=float((loss_array.sum()+tail.sum())/(len(valid)-1))
    bpc=mean_loss*(len(valid)-1)/info['valid_characters']/np.log(2)
    counts=json.loads((ROOT/'evidence/part2_metrics.json').read_text());mlp=json.loads((ROOT/'evidence/part3_mlp_metrics.json').read_text())
    settings=[('temperature',{'temperature':.8}),('top-k',{'temperature':.8,'top_k':20}),
              ('top-p',{'temperature':.8,'top_p':.9}),('min-p',{'temperature':.7,'min_p':.1})]
    # Fix a broad noun/adjective continuation before inspecting sampling p-values.
    sampler_context='Once upon a time, there was a'
    logits=np.asarray(transformer(params,jnp.array([tok.encode(sampler_context)])))[0,-1]
    sampler_tests={};rng=np.random.default_rng(1850)
    for sampler_index,(name,kw) in enumerate(settings):
        actual=np.asarray(probabilities(jnp.asarray(logits),**kw),dtype=np.float64);actual/=actual.sum()
        reference=distribution(logits,**kw)
        draws=np.asarray(draw_tokens(jax.random.key(1850+sampler_index),jnp.asarray(actual),shape=(100000,)))
        stat,df,p=chi_square(np.bincount(draws,minlength=512),reference)
        sampler_tests[name]={'statistic':stat,'df':df,'p':p,'max_probability_error':float(np.max(abs(actual-reference))),
                             'retained_tokens':int(np.count_nonzero(reference)),'settings':kw}
    greedy=np.asarray(probabilities(jnp.asarray(logits),temperature=.0001));greedy=greedy/greedy.sum()
    greedy_ok=bool(np.all(np.asarray(draw_tokens(jax.random.key(1859),jnp.asarray(greedy),shape=(100000,)))==np.argmax(logits)))
    reference=distribution(logits,temperature=.8,top_p=.9);order=np.argsort(-reference);last=order[np.count_nonzero(reference)-1]
    defective=reference.copy();defective[last]=0;defective/=defective.sum()
    negative=chi_square(np.bincount(rng.choice(512,100000,p=defective),minlength=512),reference)
    samples={};consistency={}
    for name,kw in [('T=1',{'temperature':1.})]+settings:
        ids,_,_=generate(params,seed=1848,batch=1,length=128,eos=tok.eos_id(),exclude_eos=True,**kw)
        samples[name]=tok.decode(np.asarray(ids[0]).tolist())
    opening,_,_=generate(params,seed=1848,batch=1,length=60,eos=tok.eos_id(),exclude_eos=True)
    mlp_params=load_checkpoint(ROOT/'artifacts/mlp.pkl')
    mlp_logits=np.asarray(mlp_forward(mlp_params,jnp.arange(512)))
    mlp_rng=np.random.default_rng(1848);previous=tok.eos_id();mlp_tokens=[]
    for _ in range(60):
        prob=distribution(mlp_logits[previous]);prob[tok.eos_id()]=0;prob/=prob.sum()
        previous=int(mlp_rng.choice(512,p=prob));mlp_tokens.append(previous)
    for temperature in [1.,.7]:
        ids,surprise,entropy=generate(params,seed=1851,batch=128,length=128,eos=tok.eos_id(),temperature=temperature)
        difference=np.asarray(surprise,dtype=float).mean(1)-np.asarray(entropy,dtype=float).mean(1)
        standard_error=difference.std(ddof=1)/np.sqrt(len(difference))
        consistency[str(temperature)]={'surprise':float(surprise.mean()),'entropy':float(entropy.mean()),
            'mean_difference':float(difference.mean()),'standard_error':float(standard_error),'z':float(difference.mean()/standard_error)}
        np.savez(ROOT/f'artifacts/consistency-T{temperature}.npz',tokens=ids,surprise=surprise,entropy=entropy)
        print('Consistency',temperature,consistency[str(temperature)],flush=True)
    _,maps=transformer(params,jnp.asarray(valid[1:33][None,:]),return_attention=True)
    fig,axes=plt.subplots(2,4,figsize=(15,8))
    labels=[tok.id_to_piece(int(t)).replace('▁','·') for t in valid[1:33]]
    for layer in range(2):
        for head in range(4):
            ax=axes[layer,head];ax.imshow(np.asarray(maps[layer])[0,head],vmin=0,vmax=1,cmap='magma')
            ax.set_title(f'Layer {layer+1}, head {head+1}');ax.set_xticks(range(0,32,4),labels[::4],rotation=90,fontsize=7);ax.set_yticks(range(0,32,4),labels[::4],fontsize=7)
    fig.suptitle('Where each head reads in a held-out story (all eight heads)');figure('part5-attention.png',fig)
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(1,65)
    ax.plot(x,position,label='Transformer');ax.fill_between(x,position-2*se,position+2*se,alpha=.2)
    for n,name in [(2,'Bigram'),(3,'Trigram')]:ax.axhline(counts['models'][n]['loss'],ls='--',label=name)
    ax.set(xlabel='Tokens read in this window',ylabel='Held-out nats / token',title='The transformer gains from the whole context');ax.legend();figure('part5-context.png',fig)
    records=counts['models']+[{'model':'MLP','bits_per_character':mlp['bits_per_character']},{'model':'Transformer','bits_per_character':bpc}]
    fig,ax=plt.subplots(figsize=(11,4));colors=['#c0c9d2']*7+['#6b98b1','#176bba']
    ax.bar([r['model'] for r in records],[r['bits_per_character'] for r in records],color=colors)
    for i,r in enumerate(records):ax.text(i,r['bits_per_character']+.05,f"{r['bits_per_character']:.3f}",ha='center')
    ax.set(ylabel='Held-out bits / character',title='Same tokenizer and held-out stories for every model',ylim=(0,4.9));figure('part5-comparison.png',fig)
    fig,ax=plt.subplots(figsize=(8,4));z=[consistency['1.0']['z'],consistency['0.7']['z']]
    ax.bar(['T = 1','T = 0.7'],z,color=['#176bba','#d67930']);ax.axhspan(-4,4,color='#176bba',alpha=.1)
    ax.set(ylabel='z-score of surprise minus entropy',title='128 independent sequences x 128 tokens; scored at T = 1');figure('part5-consistency.png',fig)
    metrics={'loss':mean_loss,'bits_per_character':bpc,'windows':count,'remaining_tokens':remainder,
        'position_loss':position,'position_standard_error':se,'last8_loss':float(position[-8:].mean()),
        'after32_loss':float(position[31:].mean()),'sampler_tests':sampler_tests,'greedy_all_argmax':greedy_ok,
        'top_p_negative_control':{'statistic':negative[0],'df':negative[1],'p':negative[2]},
        'sampler_context':sampler_context,'sampler_context_entropy':float(-np.sum(distribution(logits)*np.log(distribution(logits)))),
        'self_consistency':consistency,'samples':samples,'opening_60_tokens':tok.decode(np.asarray(opening[0]).tolist()),
        'mlp_opening_60_tokens':tok.decode(mlp_tokens),
        'sampling_note':'Displayed stories exclude EOS; self-consistency includes EOS and continues to exactly 128 tokens.'}
    save_json('part5_metrics.json',metrics)
    save_openings(metrics,counts,mlp)
    print('Transformer',mean_loss,'nats',bpc,'bits/char; samplers',sampler_tests,flush=True)


if __name__=='__main__':main()
