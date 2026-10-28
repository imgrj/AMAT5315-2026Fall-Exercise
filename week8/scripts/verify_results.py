"""Independent acceptance audit of the saved full experiments against SPEC.md."""
from _common import ROOT,save_json,np
import json


def main():
    read=lambda name:json.loads((ROOT/'evidence'/name).read_text())
    checks={}
    def check(name,condition):
        checks[name]=bool(condition);print(('PASS ' if condition else 'FAIL ')+name,flush=True)
    corpus=read('corpus.json');p1=read('part1_metrics.json');p2=read('part2_metrics.json')
    mem=read('part3_memory_metrics.json');mlp=read('part3_mlp_metrics.json');pic=read('part4_metrics.json')
    tr=read('part5_metrics.json');training=read('part5_training.json');sc=read('part6_metrics.json')
    check('corpus split', (corpus['train_stories'],corpus['valid_stories'])==(24867,2763))
    check('three tokenizer roundtrip exceptions',len(corpus['roundtrip_exception_indices'])==3)
    check('BPE vocabulary and characters/token',corpus['vocab']==512 and abs(corpus['characters_per_token']-2.05)<.02)
    for name,rel in p1['relations'].items():
        check(name+' trained direction',rel['mean']>=.3);check(name+' shuffled control',abs(rel['control_mean'])<=.15)
    models=p2['models']
    uniform=models[0]
    check('uniform model exact normalization and metrics',
          abs(uniform['loss']-np.log(512))<1e-14 and uniform['perplexity']==512
          and abs(uniform['bits_per_character']-9*(corpus['valid_tokens']-1)/corpus['valid_characters'])<1e-14)
    for i,(loss,perplexity,bpc) in enumerate([(np.log(512),512,4.386),(4.839,126.4,3.403),(3.038,20.9,2.136),(2.094,8.12,1.473)]):
        r=models[i];check(r['model']+' loss/perplexity/bits',abs(r['loss']-loss)<.03 and abs(r['perplexity']/perplexity-1)<.03 and abs(r['bits_per_character']-bpc)<.03)
    check('4-gram is best count model',np.argmin([m['bits_per_character'] for m in models])==4)
    check('4/5/6-gram loss reversal',all(abs(models[i]['bits_per_character']-ref)<.03 for i,ref in [(4,1.310),(5,1.46),(6,1.79)]))
    check('unseen n-gram fractions',all(abs(models[i]['unseen_fraction']-ref)<.02 for i,ref in [(3,.006),(4,.042),(5,.122),(6,.235)]))
    check('exhaustive bigram normalization',p2['enumeration_max_error']<1e-12)
    check('bigram sampling chi-square',p2['ancestral_test']['p']>.001)
    check('wrong-lag negative control rejected',p2['lag2_negative_control']['p']<.001)
    check('unnormalized negative control rejected',max(abs(np.array(p2['unnormalized_control_masses'])-1))>1)
    caps=np.asarray(mem['capacities']);slopes=mem['dimension_slopes']
    check('outer capacity at d=64',50<=caps[2,0]<=85);check('least-squares capacity at d=64',100<=caps[2,1]<=170)
    check('least-squares exact through d memories',mem['least_squares_max_below_d']<1e-10)
    check('linear/ReLU capacity exponents',all(abs(slopes[k]-1)<=.1 for k in ['outer','least_squares','relu1']))
    check('quadratic/cubic capacity exponents',abs(slopes['relu2']-2)<=.15 and slopes['relu3']>2.5)
    check('sqrt(M) interference growth',all(abs(s-.5)<.06 for s in mem['growth_slopes_d64'].values()))
    check('MLP matches bigram',abs(mlp['difference'])<.01 and mlp['settings']['steps']==4000 and mlp['settings']['batch']==16384)
    r=np.asarray(pic['median_ratios']);check('picture outer/ReLU predictions at 1024',abs(r[2,0]/4-1)<.15 and abs(r[2,1]/np.sqrt(8)-1)<.15)
    check('quadratic memory loses picture at 4096',r[3,2]>1)
    check('softmax retains every picture',np.all(r[:,3]<.1) and np.all(np.asarray(pic['gallery_ratios'])[:,3]<.1))
    check('exponential normalization preserves recall',pic['softmax_exponential_max_difference']<1e-12)
    check('transformer training budget',training['settings']['steps']==6000 and training['settings']['batch']==32 and training['parameters']==328832)
    check('transformer beats all count models',tr['bits_per_character']<min(m['bits_per_character'] for m in models) and tr['bits_per_character']<1.310)
    check('transformer uses long context',tr['after32_loss']<2.09)
    for name,s in tr['sampler_tests'].items():check(name+' sampler chi-square',s['p']>.001 and s['max_probability_error']<1e-6)
    check('near-zero temperature is greedy',tr['greedy_all_argmax'])
    check('incorrect top-p negative control rejected',tr['top_p_negative_control']['p']<.001)
    check('T=1 self consistency',abs(tr['self_consistency']['1.0']['z'])<4)
    check('T=.7 negative control',abs(tr['self_consistency']['0.7']['z'])>40)
    p=sc['parameters'];check('scaling exponents and floor',.32<=p['alpha']<=.38 and .32<=p['beta']<=.42 and 1.75<=p['E']<=1.90)
    check('240 runs and 400 successful bootstrap refits',sc['runs_downloaded']==245 and sc['runs_fitted']==240 and sc['bootstrap_resamples']==sc['bootstrap_successes']==400)
    check('refit residual centered',abs(sc['mean_log_residual'])<.005)
    check('printed parameters miss runs',sc['printed_mean_log_residual']>.01)
    check('compute optimal ratio',10<=sc['tokens_per_parameter']<=30)
    check('bootstrap interval includes Chinchilla 20',sc['ratio_95_interval'][0]<20<sc['ratio_95_interval'][1])
    result={'checks':checks,'passed':sum(checks.values()),'total':len(checks)}
    save_json('acceptance.json',result)
    if not all(checks.values()):raise SystemExit('Acceptance failure; inspect the failed criterion without relaxing it')
    print(f"{result['passed']}/{result['total']} full-experiment acceptance checks passed")


if __name__=='__main__':main()
