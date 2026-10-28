from _common import ROOT,save_json,figure,plt,np,environment
from src.data import load_corpus
from src.models import init_transformer,transformer,cross_entropy
from src.training import learning_rate,save_checkpoint
import jax
import jax.numpy as jnp
import optax
import time


def main():
    train,valid,tok,info=load_corpus(ROOT)
    config={'steps':6000,'batch':32,'context':64,'dim':96,'heads':4,'layers':2,'peak_lr':.003,
            'warmup':300,'final_lr_fraction':.1,'weight_decay':.01,'seed':1848,
            'initialization':'normal std .02; residual outputs divided by sqrt(2*layers)',
            'environment':environment()}
    save_json('transformer-settings.json',config,raw=True)
    params=init_transformer(jax.random.key(1848));data=jnp.asarray(train)
    optimizer=optax.adamw(lambda s:learning_rate(s,6000,300,.003),weight_decay=.01)
    state=optimizer.init(params);key=jax.random.key(1849)
    @jax.jit
    def step(p,s,key):
        starts=jax.random.randint(key,(32,),0,len(train)-64)
        batch=data[starts[:,None]+jnp.arange(65)]
        loss,grad=jax.value_and_grad(lambda w:cross_entropy(transformer(w,batch[:,:64]),batch[:,1:]))(p)
        updates,s=optimizer.update(grad,s,p)
        return optax.apply_updates(p,updates),s,loss
    eval_starts=np.arange(0,min(len(valid)-65,65536),65)
    eval_batch=jnp.asarray(valid[eval_starts[:,None]+np.arange(65)])
    evaluate=jax.jit(lambda p,b:cross_entropy(transformer(p,b[:,:64]),b[:,1:]))
    history=[];start=time.perf_counter();losses=[]
    for i in range(6000):
        key,sub=jax.random.split(key);params,state,loss=step(params,state,sub);losses.append(loss)
        if (i+1)%200==0:
            val=np.mean([float(evaluate(params,eval_batch[j:j+32])) for j in range(0,len(eval_batch),32)])
            row={'step':i+1,'train_loss':float(jnp.mean(jnp.stack(losses))),'monitor_valid_loss':float(val),'seconds':time.perf_counter()-start}
            history.append(row);losses=[];print(row,flush=True)
            save_checkpoint(ROOT/'artifacts/transformer.pkl',params)
            save_json('transformer-training.json',{'settings':config,'history':history},raw=True)
    elapsed=time.perf_counter()-start
    save_json('part5_training.json',{'settings':config,'history':history,'seconds':elapsed,
                                   'parameters':sum(x.size for x in jax.tree.leaves(params))})
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot([h['step'] for h in history],[h['train_loss'] for h in history],label='Training batch mean')
    ax.plot([h['step'] for h in history],[h['monitor_valid_loss'] for h in history],label='Fixed validation monitor')
    ax.set(xlabel='AdamW update',ylabel='Nats / token',title='Two-layer transformer, 6000 CPU updates');ax.legend()
    figure('part5-training.png',fig)


if __name__=='__main__':main()
