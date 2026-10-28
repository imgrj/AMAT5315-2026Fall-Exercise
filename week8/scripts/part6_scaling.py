from _common import ROOT,save_json,figure,plt,np
from src.scaling import fit,physical,predict,optimal_allocation
import csv
import itertools
import time


def main():
    rows=list(csv.DictReader((ROOT/'data/chinchilla-runs.csv').open()))
    n=np.array([float(r['Model Size']) for r in rows]);c=np.array([float(r['Training FLOP']) for r in rows]);loss=np.array([float(r['loss']) for r in rows])
    retained=np.argsort(loss)[:-5];d=c/(6*n);n,c,d,loss=[a[retained] for a in (n,c,d,loss)]
    start=time.perf_counter();best=None;fits=[]
    for e,a,b,alpha,beta in itertools.product([1.,1.5,2.],[100.,500.,2000.],[100.,1000.,10000.],[.25,.4],[.25,.4]):
        result=fit(n,d,loss,np.r_[np.log([e,a,b]),alpha,beta]);fits.append({'objective':result.fun,'success':bool(result.success)})
        if best is None or result.fun<best.fun:best=result
    p=physical(best.x);print('Best',p,'objective',best.fun,flush=True)
    rng=np.random.default_rng(1860);boot=[];success=[]
    for i in range(400):
        indices=rng.integers(0,len(n),len(n));result=fit(n[indices],d[indices],loss[indices],best.x)
        boot.append(physical(result.x));success.append(bool(result.success))
        if (i+1)%50==0:print('Bootstrap',i+1,'/',400,flush=True)
    boot=np.asarray(boot);printed=np.array([1.69,406.4,410.7,.34,.28]);pred=predict(p,n,d)
    residual=np.log(pred)-np.log(loss);oldres=np.log(predict(printed,n,d))-np.log(loss)
    budget=5.76e23;opt_n,opt_d=optimal_allocation(p,budget)
    ratios=np.array([float(dd/nn) for nn,dd in [optimal_allocation(bp,budget) for bp in boot]])
    budgets=np.logspace(18,25,100);ns,ds=optimal_allocation(p,budgets)
    metrics={'runs_downloaded':len(rows),'runs_fitted':len(n),'dropped_row_indices':np.argsort([float(r['loss']) for r in rows])[-5:],
        'parameters':dict(zip(['E','A','B','alpha','beta'],p)),
        'standard_errors':dict(zip(['E','A','B','alpha','beta'],boot.std(0,ddof=1))),
        'grid_starts':len(fits),'best_objective':best.fun,'fit_success':bool(best.success),
        'bootstrap_resamples':400,'bootstrap_successes':sum(success),'mean_log_residual':residual.mean(),
        'printed_mean_log_residual':oldres.mean(),'compute':budget,'optimal_parameters':float(opt_n),
        'optimal_tokens':float(opt_d),'tokens_per_parameter':float(opt_d/opt_n),
        'ratio_95_interval':np.quantile(ratios,[.025,.975]),
        'printed_tokens_per_parameter':float(np.divide(*optimal_allocation(printed,budget)[::-1])),
        'model_compute_exponent':p[4]/(p[3]+p[4]),'data_compute_exponent':p[3]/(p[3]+p[4]),
        'budget_grid':budgets,'optimal_N_grid':ns,'optimal_D_grid':ds,'seconds':time.perf_counter()-start,'seed':1860}
    save_json('part6_metrics.json',metrics);save_json('scaling-grid.json',fits,raw=True)
    np.savez(ROOT/'artifacts/scaling-fit.npz',n=n,d=d,c=c,loss=loss,bootstrap=boot,params=p)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].scatter(n,loss,c='#bdc6cf',s=12)
    for budget,color in zip([1e19,1e20,1e21],['#cf7238','#2287ac','#7d59ad']):
        use=abs(c/budget-1)<.15;ax[0].scatter(n[use],loss[use],c=color,s=22)
        grid=np.logspace(7.5,10.5,200);ax[0].plot(grid,predict(p,grid,budget/(6*grid)),color=color,label=f'{budget:.0e} FLOP')
    ax[0].set(xscale='log',ylim=(2,5.3),xlabel='Parameters N',ylabel='Final loss',title='Iso-compute curves have a minimum');ax[0].legend()
    points=ax[1].scatter(pred,loss,c=np.log10(c),s=15,cmap='viridis');ax[1].plot([2,4.5],[2,4.5],'k--');ax[1].set(xlabel='Fitted loss',ylabel='Measured loss',title='240 published runs');fig.colorbar(points,ax=ax[1],label='log10 training FLOP')
    figure('part6-fit.png',fig)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for r,label,color in [(residual,'Refit','#176bba'),(oldres,'Printed','#d67930')]:
        ax[0].scatter(n,r,s=15,label=f'{label}: mean {r.mean():+.4f}',alpha=.7,color=color);ax[1].hist(r,bins=25,alpha=.5,label=label,color=color)
    ax[0].axhline(0,color='black',lw=1);ax[0].set(xscale='log',xlabel='Parameters N',ylabel='Predicted - measured log loss');ax[0].legend()
    ax[1].set(xlabel='Log-loss residual',ylabel='Runs');ax[1].legend();figure('part6-residuals.png',fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for pp,ls,label in [(p,'-','Refit'),(printed,'--','Printed')]:
        nn,dd=optimal_allocation(pp,budgets);ax.loglog(budgets,nn,ls,color='#176bba',label=label+' N*');ax.loglog(budgets,dd,ls,color='#d67930',label=label+' D*')
    ax.scatter([5.88e23,5.88e23],[70e9,1.4e12],marker='*',s=130,color=['#176bba','#d67930'])
    ax.set(xlabel='Training compute C (FLOP)',ylabel='Optimal parameters / tokens',title=f'At 5.76e23 FLOP: D*/N* = {opt_d/opt_n:.1f}, 95% CI {np.quantile(ratios,.025):.1f}–{np.quantile(ratios,.975):.1f}')
    ax.legend(ncol=2);figure('part6-allocation.png',fig);print(metrics,flush=True)


if __name__=='__main__':main()
