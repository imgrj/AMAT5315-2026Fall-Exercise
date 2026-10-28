"""Auditable probability transforms and grouped chi-square checks."""
import numpy as np
from scipy.special import softmax
from scipy.stats import chi2


def distribution(logits,temperature=1.,top_k=None,top_p=None,min_p=None):
    if temperature<=0:raise ValueError('Temperature must be positive')
    probs=softmax(np.asarray(logits,dtype=np.float64)/temperature)
    keep=np.ones(len(probs),dtype=bool)
    order=np.argsort(-probs,kind='stable')
    if top_k is not None:
        keep[:]=False;keep[order[:top_k]]=True
    if top_p is not None:
        # Cumulative mass BEFORE this token: retain the token crossing top_p.
        previous=np.r_[0,np.cumsum(probs[order])[:-1]]
        keep[order[previous>=top_p]]=False
    if min_p is not None:keep &= probs>=min_p*probs.max()
    probs=np.where(keep,probs,0)
    return probs/probs.sum()


def chi_square(counts,probabilities):
    observed=np.asarray(counts,dtype=np.float64)
    expected=observed.sum()*np.asarray(probabilities,dtype=np.float64)
    if np.any((expected==0)&(observed>0)):return float('inf'),1,0.
    positive=expected>0;observed=observed[positive];expected=expected[positive]
    order=np.argsort(expected);o=[];e=[];so=se=0.
    for i in order:
        so+=observed[i];se+=expected[i]
        if se>=5:o.append(so);e.append(se);so=se=0.
    if se:
        if not e:return 0.,0,1.
        o[-1]+=so;e[-1]+=se
    stat=float(np.sum((np.array(o)-e)**2/e));df=len(e)-1
    return stat,df,float(chi2.sf(stat,df)) if df else 1.
