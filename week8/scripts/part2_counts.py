from _common import ROOT,save_json,figure,plt,np
from src.data import load_corpus
from src.ngrams import NGram,enumerate_mass,sample_chain
from src.sampling import chi_square
from scipy.stats import chi2
import pickle
import time


def pooled_chain_test(p,tokens,start,contexts):
    previous=np.r_[start,tokens[:-1]]
    table=np.bincount(previous*len(p)+tokens,minlength=len(p)**2).reshape(p.shape)
    checks=[chi_square(table[c],p[c]) for c in contexts]
    statistic=sum(c[0] for c in checks);df=sum(c[1] for c in checks)
    return {'statistic':statistic,'df':df,'p':float(chi2.sf(statistic,df))}


def main():
    start=time.perf_counter();train,valid,tok,info=load_corpus(ROOT)
    save_json('corpus.json',info);print(info,flush=True)
    ratio=(len(valid)-1)/info['valid_characters']/np.log(2)
    records=[{'model':'uniform','n':0,'loss':np.log(512),'perplexity':512,'bits_per_character':np.log(512)*ratio}]
    cut=int(.9*len(train));models={}
    for n in range(1,7):
        fit=NGram.fit(train[:cut],n);a,b=fit.observed_counts(train[cut-n+1:])
        choices=[.0001,.001,.01,.1,1.]
        tuning=[float(-np.log((a+k)/(b+k*512)).mean()) for k in choices]
        k=choices[int(np.argmin(tuning))]
        model=NGram.fit(train,n,k=k);loss,unseen=model.score(valid)
        record={'model':f'{n}-gram','n':n,'k':k,'tuning_losses':dict(zip(map(str,choices),tuning)),
                'loss':loss,'perplexity':np.exp(loss),'bits_per_character':loss*ratio,'unseen_fraction':unseen,
                'observed_ngrams':len(model.keys),'observed_contexts':len(model.contexts)}
        records.append(record);print(record,flush=True)
        with open(ROOT/f'artifacts/ngram{n}.pkl','wb') as f:pickle.dump(model,f)
        if n<=3:models[n]=model
    p=models[2].dense();mass=enumerate_mass(p,tok.eos_id())
    scores=np.exp(np.asarray([[0,1],[1,0]],float)) # independent unnormalized negative control
    bad_mass=enumerate_mass(scores,0)
    contexts=np.argsort(np.bincount(train,minlength=512))[-20:]
    good=sample_chain(p,250000,tok.eos_id(),1848)
    bad=sample_chain(p,250000,tok.eos_id(),1848,lag=2)
    chain=pooled_chain_test(p,good,tok.eos_id(),contexts)
    negative=pooled_chain_test(p,bad,tok.eos_id(),contexts)
    rng=np.random.default_rng(1848);samples={'uniform':tok.decode(rng.integers(3,512,60).tolist())}
    for n,m in models.items():
        rng=np.random.default_rng(1848);context=[tok.eos_id()]*(n-1);out=[]
        for _ in range(60):
            prob=m.probabilities(context);prob[tok.eos_id()]=0;prob/=prob.sum()
            token=int(rng.choice(512,p=prob));out.append(token);context=(context+[token])[-(n-1):] if n>1 else []
        samples[f'{n}-gram']=tok.decode(out)
    metrics={'models':records,'enumeration_masses':mass,'enumeration_max_error':max(abs(np.array(mass)-1)),
             'unnormalized_control_masses':bad_mass,'ancestral_test':chain,'lag2_negative_control':negative,
             'samples_seed1848':samples,'seconds':time.perf_counter()-start}
    save_json('part2_metrics.json',metrics);print('Checks',mass,chain,negative,flush=True)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].plot(range(1,7),[r['bits_per_character'] for r in records[1:]],'o-',color='#176bba')
    ax[0].set(xlabel='n-gram order',ylabel='Held-out bits / character',title='More context helps until counts run out',xticks=range(1,7))
    ax[1].plot(range(1,7),[100*r['unseen_fraction'] for r in records[1:]],'o-',color='#d67930')
    ax[1].set(xlabel='n-gram order',ylabel='Unseen held-out n-grams (%)',title='The curse of dimensionality',xticks=range(1,7))
    figure('part2-counts.png',fig)


if __name__=='__main__':main()
