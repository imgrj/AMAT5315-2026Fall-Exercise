from _common import ROOT,save_json,figure,plt,np,environment
from src.data import load_corpus
from src.models import init_mlp,mlp
from src.training import learning_rate,save_checkpoint
import jax
import jax.numpy as jnp
import optax
import time
import json


def main():
    train,valid,tok,info=load_corpus(ROOT)
    config={'steps':4000,'batch':16384,'warmup':200,'peak_lr':.006,'final_lr_fraction':.1,
            'weight_decay':.01,'seed':1833,'embedding':64,'hidden':512,'environment':environment()}
    save_json('mlp-settings.json',config,raw=True)
    params=init_mlp(jax.random.key(config['seed']));initial=params
    schedule=lambda step:learning_rate(step,4000,200,.006)
    optimizer=optax.adamw(schedule,weight_decay=.01);state=optimizer.init(params)
    data=jnp.asarray(train);all_tokens=jnp.arange(512)
    # Duplicate predecessor tokens have identical logits. Aggregate the sampled
    # 16384 pairs before evaluating the loss: exactly the same minibatch gradient.
    @jax.jit
    def step(p,s,key):
        index=jax.random.randint(key,(16384,),0,len(train)-1)
        counts=jnp.bincount(data[index]*512+data[index+1],length=512**2).reshape(512,512)/16384
        def loss(weights):return -jnp.sum(counts*jax.nn.log_softmax(mlp(weights,all_tokens)))
        loss_value,grad=jax.value_and_grad(loss)(p)
        updates,s=optimizer.update(grad,s,p)
        return optax.apply_updates(p,updates),s,loss_value
    valid_counts=np.bincount(valid[:-1]*512+valid[1:],minlength=512**2).reshape(512,512)
    @jax.jit
    def evaluate(p):return -jnp.sum(jnp.asarray(valid_counts)*jax.nn.log_softmax(mlp(p,all_tokens)))/valid_counts.sum()
    history=[{'step':0,'valid_loss':float(evaluate(params))}];key=jax.random.key(1834);start=time.perf_counter()
    for i in range(4000):
        key,sub=jax.random.split(key);params,state,loss=step(params,state,sub)
        if (i+1)%200==0:
            row={'step':i+1,'train_loss':float(loss),'valid_loss':float(evaluate(params))}
            history.append(row);print(row,flush=True)
    elapsed=time.perf_counter()-start
    save_checkpoint(ROOT/'artifacts/mlp.pkl',params)
    bigram=json.loads((ROOT/'evidence/part2_metrics.json').read_text())['models'][2]['loss']
    def unit_summary(p):
        activ=np.asarray(jnp.tanh(p['embedding']@p['hidden']['w']+p['hidden']['b']))
        reports=[]
        for sign in [1,-1]:
            squared=np.maximum(sign*activ,0)**2
            order=np.argsort(squared,axis=0)[-5:][::-1]
            share=np.take_along_axis(squared,order,axis=0).sum(0)/squared.sum(0)
            for unit in range(512):
                next_ids=np.argsort(sign*np.asarray(p['readout']['w'])[unit])[-5:][::-1]
                reports.append({'unit':unit,'side':sign,'share':float(share[unit]),
                                'previous':[tok.id_to_piece(int(v)) for v in order[:,unit]],
                                'next':[tok.id_to_piece(int(v)) for v in next_ids]})
        return reports
    trained=unit_summary(params);untrained=unit_summary(initial)
    loss=float(evaluate(params));metrics={'loss':loss,'bigram_loss':bigram,'difference':loss-bigram,
        'bits_per_character':loss*(len(valid)-1)/info['valid_characters']/np.log(2),
        'seconds':elapsed,'settings':config,'history':history,
        'trained_median_top5_share':float(np.median([r['share'] for r in trained])),
        'initial_median_top5_share':float(np.median([r['share'] for r in untrained])),
        'most_concentrated_sides':sorted(trained,key=lambda r:-r['share'])[:8]}
    save_json('mlp-units.json',trained,raw=True);save_json('part3_mlp_metrics.json',metrics)
    fig,ax=plt.subplots(figsize=(8,4));ax.plot([h['step'] for h in history],[h['valid_loss'] for h in history],label='MLP held-out')
    ax.axhline(bigram,ls='--',color='#d67930',label=f'Bigram {bigram:.4f}')
    ax.set(xlabel='AdamW update',ylabel='Nats / token',title=f'One-token MLP stores the bigram table: difference {loss-bigram:+.4f}')
    ax.legend();figure('part3-mlp.png',fig);print('Final',loss,'difference',loss-bigram,'seconds',elapsed,flush=True)


if __name__=='__main__':main()
