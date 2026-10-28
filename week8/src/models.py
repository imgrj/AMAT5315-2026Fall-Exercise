"""Small explicit JAX models, all linear maps biased, no high-level framework."""
import jax
import jax.numpy as jnp


def linear(p,x):return x@p['w']+p['b']


def init_mlp(key,vocab=512,dim=64,hidden=512):
    a,b,c=jax.random.split(key,3)
    return {'embedding':jax.random.normal(a,(vocab,dim))*.1,
            'hidden':{'w':jax.random.normal(b,(dim,hidden))/dim**.5,'b':jnp.zeros(hidden)},
            'readout':{'w':jax.random.normal(c,(hidden,vocab))*.01,'b':jnp.zeros(vocab)}}


def mlp(p,tokens):return linear(p['readout'],jnp.tanh(linear(p['hidden'],p['embedding'][tokens])))


def cross_entropy(logits,targets):
    return -jnp.take_along_axis(jax.nn.log_softmax(logits),targets[...,None],axis=-1)[...,0].mean()


def attention(q,k,v):
    scores=jnp.einsum('bthd,bshd->bhts',q,k)/q.shape[-1]**.5
    causal=jnp.arange(q.shape[1])[:,None]>=jnp.arange(k.shape[1])[None,:]
    weights=jax.nn.softmax(jnp.where(causal,scores,-jnp.inf),axis=-1)
    return jnp.einsum('bhts,bshd->bthd',weights,v),weights


def init_transformer(key,vocab=512,dim=96,heads=4,layers=2,context=64):
    if dim%heads:raise ValueError('Width must be divisible by head count')
    keys=iter(jax.random.split(key,4+6*layers))
    def normal(shape,scale=.02):return jax.random.normal(next(keys),shape)*scale
    def dense(a,b,residual=False):return {'w':normal((a,b),.02/(2*layers)**.5 if residual else .02),'b':jnp.zeros(b)}
    def norm():return {'g':jnp.ones(dim),'b':jnp.zeros(dim)}
    blocks=[]
    embedding=normal((vocab,dim));position=normal((context,dim))
    for _ in range(layers):
        blocks.append({'ln1':norm(),'q':dense(dim,dim),'k':dense(dim,dim),'v':dense(dim,dim),
                       'o':dense(dim,dim,True),'ln2':norm(),'up':dense(dim,4*dim),'down':dense(4*dim,dim,True)})
    return {'embedding':embedding,'position':position,'blocks':blocks,'ln':norm(),'readout':dense(dim,vocab)}


def layer_norm(p,x):
    mean=x.mean(axis=-1,keepdims=True);var=jnp.mean((x-mean)**2,axis=-1,keepdims=True)
    return (x-mean)*jax.lax.rsqrt(var+1e-5)*p['g']+p['b']


def transformer(p,tokens,heads=4,return_attention=False):
    b,t=tokens.shape;d=p['embedding'].shape[1]
    x=p['embedding'][tokens]+p['position'][:t];maps=[]
    for block in p['blocks']:
        z=layer_norm(block['ln1'],x)
        q,k,v=[linear(block[name],z).reshape(b,t,heads,d//heads) for name in ('q','k','v')]
        a,w=attention(q,k,v);maps.append(w)
        x=x+linear(block['o'],a.reshape(b,t,d))
        x=x+linear(block['down'],jax.nn.gelu(linear(block['up'],layer_norm(block['ln2'],x))))
    logits=linear(p['readout'],layer_norm(p['ln'],x))
    return (logits,maps) if return_attention else logits
