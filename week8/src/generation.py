"""JAX ancestral sampling; score surprise with the original T=1 distribution."""
from functools import partial
import jax
import jax.numpy as jnp
from .models import transformer


def draw_tokens(key,probs,shape=None):
    return jax.random.categorical(key,jnp.log(probs),axis=-1,shape=shape).astype(jnp.int32)


def probabilities(logits,temperature=1.,top_k=None,top_p=None,min_p=None):
    probs=jax.nn.softmax(logits/temperature,axis=-1)
    keep=jnp.ones_like(probs,dtype=bool)
    if top_k is not None:
        _,ids=jax.lax.top_k(probs,top_k)
        keep=jnp.sum(jax.nn.one_hot(ids,probs.shape[-1]),axis=-2)>0
    if top_p is not None:
        order=jnp.argsort(-probs,axis=-1,stable=True)
        sorted_p=jnp.take_along_axis(probs,order,axis=-1)
        previous=jnp.cumsum(sorted_p,axis=-1)-sorted_p
        rank=jnp.argsort(order,axis=-1)
        keep &= jnp.take_along_axis(previous<top_p,rank,axis=-1)
    if min_p is not None:keep &= probs>=min_p*jnp.max(probs,axis=-1,keepdims=True)
    probs=jnp.where(keep,probs,0)
    return probs/jnp.sum(probs,axis=-1,keepdims=True)


@partial(jax.jit,static_argnames=('seed','batch','length','eos','heads','temperature','top_k','top_p','min_p','exclude_eos'))
def generate(params,seed,batch,length,eos=2,heads=4,temperature=1.,top_k=None,top_p=None,min_p=None,exclude_eos=False):
    context=params['position'].shape[0]
    buffer=jnp.zeros((batch,context),dtype=jnp.int32).at[:,0].set(eos)
    def body(carry,t):
        buffer,key=carry;key,sub=jax.random.split(key);position=jnp.minimum(t,context-1)
        logits=transformer(params,buffer,heads=heads)[:,position,:]
        logp=jax.nn.log_softmax(logits);base=jnp.exp(logp)
        sampling_logits=logits.at[:,eos].set(-jnp.inf) if exclude_eos else logits
        p=probabilities(sampling_logits,temperature,top_k,top_p,min_p)
        tokens=draw_tokens(sub,p)
        surprise=-jnp.take_along_axis(logp,tokens[:,None],axis=-1)[:,0]
        entropy=-jnp.sum(base*logp,axis=-1)
        buffer=jax.lax.cond(t<context-1,lambda b:b.at[:,position+1].set(tokens),
                            lambda b:jnp.concatenate([b[:,1:],tokens[:,None]],axis=1),buffer)
        return (buffer,key),(tokens,surprise,entropy)
    _,outputs=jax.lax.scan(body,(buffer,jax.random.key(seed)),jnp.arange(length))
    return tuple(a.T for a in outputs)
