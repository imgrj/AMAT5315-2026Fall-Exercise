from _common import ROOT,save_json,figure,plt,np
from src.memory import unit_rows,recall,interference,capacity
import time


def main():
    kinds=['outer','least_squares','relu1','relu2','relu3'];dims=[16,32,64]
    sizes=2**np.arange(3,17);raw=np.zeros((3,3,len(sizes),5));correlated=np.zeros((3,3,len(sizes)))
    start=time.perf_counter()
    for di,d in enumerate(dims):
        for mi,m in enumerate(sizes):
            for draw in range(3):
                rng=np.random.default_rng(1830+10000*di+100*mi+draw)
                keys=unit_rows(rng,m,d);values=unit_rows(rng,m,d)
                # Exhaustive for small stores; fixed 512 stored queries for large
                # stores estimates mean squared interference without an M^2 matrix.
                indices=np.arange(min(m,512))
                for si,kind in enumerate(kinds):
                    out,own=recall(keys,values,indices,kind)
                    ratios=interference(out,values[indices],own)
                    raw[di,draw,mi,si]=np.sqrt(np.mean(ratios**2))
                ck=np.column_stack((np.full(m,np.sqrt(.3)),np.sqrt(.7)*unit_rows(rng,m,d-1)))
                out,own=recall(ck,values,indices,'outer')
                correlated[di,draw,mi]=np.sqrt(np.mean(interference(out,values[indices],own)**2))
            print('d',d,'M',m,'R',np.median(raw[di,:,mi],axis=0),flush=True)
    curves=np.median(raw,axis=1);caps=np.array([[capacity(sizes,curves[di,:,si]) for si in range(5)] for di in range(3)])
    slopes=np.array([np.polyfit(np.log(dims),np.log(caps[:,si]),1)[0] for si in range(5)])
    growth={}
    for si,kind in enumerate(kinds):
        use=curves[2,:,si]>.5
        if kind=='least_squares':use= sizes>=512
        growth[kind]=float(np.polyfit(np.log(sizes[use]),np.log(curves[2,use,si]),1)[0])
    metrics={'dimensions':dims,'sizes':sizes,'kinds':kinds,'query_count':'min(M,512) stored keys',
        'aggregation':'RMS over stored queries; median of three independent draws','curves':curves,
        'capacities':caps,'dimension_slopes':dict(zip(kinds,slopes)),'growth_slopes_d64':growth,
        'least_squares_max_below_d':float(raw[2,:,sizes<=64,1].max()),
        'correlated_capacities':[capacity(sizes,np.median(correlated[di],axis=0)) for di in range(3)],
        'seconds':time.perf_counter()-start,'seed_formula':'1830 + 10000*dimension_index + 100*size_index + draw'}
    save_json('part3_memory_metrics.json',metrics)
    np.savez(ROOT/'artifacts/memory-runs.npz',raw=raw,correlated=correlated,sizes=sizes,dimensions=dims)
    fig,ax=plt.subplots(figsize=(9,5));colors=plt.cm.tab10.colors
    for si,kind in enumerate(kinds):ax.loglog(sizes,curves[2,:,si],'o-',label=kind,color=colors[si])
    predictions=[np.sqrt(sizes/64),np.sqrt(np.maximum(sizes/64-1,1e-10)),np.sqrt(sizes/128),np.sqrt(3*sizes/(2*64**2)),np.sqrt(15*sizes/(2*64**3))]
    for si,y in enumerate(predictions):ax.loglog(sizes,y,'--',alpha=.5,color=colors[si])
    ax.axhline(1,color='black',lw=1);ax.set(ylim=(1e-3,100),xlabel='Stored memories M',ylabel='Interference / own signal',title='d = 64: measured recall and Gaussian predictions');ax.legend(ncol=2)
    figure('part3-interference.png',fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for si,kind in enumerate(kinds):ax.loglog(dims,caps[:,si],'o-',label=f'{kind}: slope {slopes[si]:.2f}')
    ax.loglog(dims,metrics['correlated_capacities'],'s--',color='gray',label='outer, common direction 0.3')
    ax.set(xticks=dims,xticklabels=dims,xlabel='Dimension d',ylabel='Capacity at R = 1',title='Nonlinear memories store powers of the dimension');ax.legend()
    figure('part3-capacity.png',fig);print('Capacities',caps,'slopes',slopes,flush=True)


if __name__=='__main__':main()
