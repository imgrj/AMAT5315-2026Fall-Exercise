"""Stable log-space Chinchilla fitting with an analytic Huber gradient."""
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


def predict(params,n,d):
    e,a,b,alpha,beta=params
    return e+a/np.asarray(n)**alpha+b/np.asarray(d)**beta


def objective(theta,log_n,log_d,log_loss):
    e,a,b,alpha,beta=theta
    terms=np.array([np.full_like(log_n,e),a-alpha*log_n,b-beta*log_d])
    pred=logsumexp(terms,axis=0);r=pred-log_loss;delta=.001
    # Scale by delta to improve optimizer stopping, without changing minimizer.
    loss=np.where(abs(r)<=delta,.5*r*r,delta*(abs(r)-.5*delta)).mean()/delta
    dr=np.clip(r,-delta,delta)/delta/len(r);weights=np.exp(terms-pred)
    grad=np.array([np.sum(dr*weights[0]),np.sum(dr*weights[1]),np.sum(dr*weights[2]),
                   -np.sum(dr*weights[1]*log_n),-np.sum(dr*weights[2]*log_d)])
    return float(loss),grad


def fit(n,d,loss,start):
    return minimize(objective,start,args=(np.log(n),np.log(d),np.log(loss)),jac=True,method='L-BFGS-B',
                    bounds=[(-5,3),(-5,25),(-5,25),(.01,1.5),(.01,1.5)],
                    options={'maxiter':3000,'ftol':1e-14,'gtol':1e-9,'maxls':50})


def physical(theta):return np.r_[np.exp(theta[:3]),theta[3:]]


def optimal_allocation(params,compute):
    e,a,b,alpha,beta=params
    log_g=np.log(alpha*a/(beta*b))/(alpha+beta)
    log_c=np.log(np.asarray(compute)/6)
    return np.exp(log_g+beta/(alpha+beta)*log_c),np.exp(-log_g+alpha/(alpha+beta)*log_c)
