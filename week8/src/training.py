"""AdamW schedule and portable NumPy-pickle checkpoints (local trusted files)."""
import pickle
import jax
import jax.numpy as jnp
import numpy as np


def learning_rate(step,total,warmup,peak):
    decay=.1+.9*.5*(1+jnp.cos(jnp.pi*jnp.clip((step-warmup)/(total-1-warmup),0,1)))
    return peak*jnp.where(step<warmup,step/warmup,decay)


def save_checkpoint(path,params):
    with open(path,'wb') as f:pickle.dump(jax.tree.map(np.asarray,params),f)


def load_checkpoint(path):
    with open(path,'rb') as f:return jax.tree.map(jnp.asarray,pickle.load(f))
