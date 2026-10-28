from _common import ROOT,save_json,figure,plt,np
from safetensors.numpy import load_file
from tokenizers import Tokenizer
from src.embeddings import PAIRS,relation_cosines


def main():
    arrays=load_file(ROOT/'data/model.safetensors')
    table=next(a for a in arrays.values() if a.shape==(29528,256))
    tok=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    control=table[np.random.default_rng(1730).permutation(len(table))]
    fig,axes=plt.subplots(1,3,figsize=(15,5)); results={}
    for ax,(name,pairs) in zip(axes,PAIRS.items()):
        ids=[]
        for pair in pairs:
            row=[]
            for word in pair:
                encoded=tok.encode(word,add_special_tokens=False).ids
                assert len(encoded)==1,(word,encoded)
                row.append(encoded[0])
            ids.append(row)
        ids=np.array(ids)
        real=relation_cosines(table,ids); shuffled=relation_cosines(control,ids)
        shuffles=[float(relation_cosines(table[np.random.default_rng(1730+i).permutation(len(table))],ids).mean()) for i in range(20)]
        results[name]={'pairs':pairs,'ids':ids,'cosines':real,'control_cosines':shuffled,
                       'mean':float(real.mean()),'control_mean':float(shuffled.mean()),'twenty_shuffle_means':shuffles}
        for values,sign,color in [(real,1,'#176bba'),(shuffled,-1,'#8d969d')]:
            for c in values:
                ax.annotate('',xy=(c,sign*np.sqrt(max(0,1-c*c))),xytext=(0,0),arrowprops={'arrowstyle':'->','color':color,'alpha':.75})
        ax.axvline(0,color='#999',ls='--'); ax.axhline(0,color='black',lw=1)
        ax.set(xlim=(-.45,1.05),ylim=(-1.1,1.1),aspect='equal',xlabel='Cosine to the other eight offsets',
               title=f'{name}\ntrained {real.mean():.3f} | shuffled {shuffled.mean():.3f}')
    fig.suptitle('Meaning as a direction: all nine pairs, with row-shuffled control',fontsize=15)
    fig.tight_layout(rect=(0,0,1,.9))
    figure('part1-directions.png',fig)
    save_json('part1_metrics.json',{'shape':table.shape,'shuffle_seed':1730,'relations':results})
    print({k:(v['mean'],v['control_mean']) for k,v in results.items()},flush=True)


if __name__=='__main__':main()
