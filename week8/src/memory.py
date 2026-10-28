"""Linear, polynomial and softmax key-value stores in NumPy float64."""
import numpy as np
from scipy.special import softmax


def unit_rows(rng,n,d):
    a=rng.normal(size=(n,d))
    return a/np.linalg.norm(a,axis=1,keepdims=True)


def recall(keys,values,indices,kind,beta=8):
    queries=keys[indices]
    if kind=='outer':
        return queries@(keys.T@values),np.sum(queries**2,axis=1)
    if kind=='least_squares':
        inverse=np.linalg.pinv(keys)
        output=queries@(inverse@values)
        own=np.sum(queries*inverse[:,indices].T,axis=1)
        return output,own
    similarity=queries@keys.T
    if kind.startswith('relu'):
        weights=np.maximum(similarity,0)**int(kind[-1])
    elif kind=='softmax':
        weights=softmax(beta*similarity,axis=-1)
    else:raise ValueError(kind)
    return weights@values,weights[np.arange(len(indices)),indices]


def interference(output,values,own):
    signal=own[:,None]*values
    return np.linalg.norm(output-signal,axis=1)/np.linalg.norm(signal,axis=1)


def capacity(memories,ratios):
    cross=np.flatnonzero(np.asarray(ratios)>=1)
    if len(cross)==0:return float('nan')
    j=cross[0]
    if j==0:return float(memories[0])
    return float(np.exp(np.interp(0,np.log(ratios[j-1:j+1]),np.log(memories[j-1:j+1]))))
