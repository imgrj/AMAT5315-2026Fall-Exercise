"""Sparse exact counts. Base-V uint64 keys fit all six 512-token n-grams."""
from dataclasses import dataclass
import numpy as np


def pack_windows(tokens,n,vocab):
    tokens=np.asarray(tokens,dtype=np.uint64)
    result=np.zeros(max(0,len(tokens)-n+1),dtype=np.uint64)
    for i in range(n):
        result=result*np.uint64(vocab)+tokens[i:i+len(result)]
    return result


def lookup(keys,values,query):
    idx=np.searchsorted(keys,query)
    safe=np.minimum(idx,len(keys)-1)
    return np.where((idx<len(keys)) & (keys[safe]==query),values[safe],0)


@dataclass
class NGram:
    n:int
    vocab:int
    k:float
    keys:np.ndarray
    counts:np.ndarray
    contexts:np.ndarray
    totals:np.ndarray

    @classmethod
    def fit(cls,tokens,n,vocab=512,k=.001):
        keys,counts=np.unique(pack_windows(tokens,n,vocab),return_counts=True)
        contexts,starts=np.unique(keys//np.uint64(vocab),return_index=True)
        totals=np.add.reduceat(counts,starts)
        return cls(n,vocab,k,keys,counts,contexts,totals)

    def probabilities(self,context):
        context=list(context)[-(self.n-1):] if self.n>1 else []
        if len(context)!=self.n-1:
            raise ValueError('Incorrect context length')
        code=0
        for t in context:code=code*self.vocab+int(t)
        keys=np.uint64(code*self.vocab)+np.arange(self.vocab,dtype=np.uint64)
        counts=lookup(self.keys,self.counts,keys)
        return (counts+self.k)/(counts.sum()+self.k*self.vocab)

    def observed_counts(self,tokens):
        query=pack_windows(tokens,self.n,self.vocab)
        return lookup(self.keys,self.counts,query),lookup(self.contexts,self.totals,query//np.uint64(self.vocab))

    def score(self,tokens):
        count,total=self.observed_counts(tokens)
        return float(-np.log((count+self.k)/(total+self.k*self.vocab)).mean()),float(np.mean(count==0))

    def dense(self):
        if self.n>2:raise ValueError('Dense conversion only for unigram/bigram')
        return np.stack([self.probabilities([i]) for i in range(self.vocab)]) if self.n==2 else self.probabilities([])


def enumerate_mass(p,start):
    # Explicitly form every joint probability. Only chunk the first token at l=3.
    first=p[start]
    second=first[:,None]*p
    third=sum(np.sum(second[i,:,None]*p,dtype=np.float64) for i in range(len(p)))
    return [float(first.sum()),float(second.sum()),float(third)]


def sample_chain(p,count,start,seed,lag=1):
    rng=np.random.default_rng(seed); draws=rng.random(count); cdf=np.cumsum(p,axis=1)
    cdf[:,-1]=1
    output=np.empty(count,dtype=np.int32)
    for i in range(count):
        context=output[i-lag] if i>=lag else start
        output[i]=np.searchsorted(cdf[context],draws[i],side='right')
    return output
