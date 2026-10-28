from _common import ROOT,save_json,figure,plt,np
from src.memory import unit_rows,recall,interference
from PIL import Image,ImageDraw,ImageFont
import os


def character():
    # Pin a raster alongside results so recall does not depend on installed fonts.
    path=ROOT/'evidence/character.png'
    if path.exists():return np.where(np.asarray(Image.open(path).convert('L'))>127,1.,-1.).reshape(-1)
    candidates=[os.environ.get('WEEK8_CJK_FONT',''),'C:/Windows/Fonts/simsun.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc']
    font_path=next((p for p in candidates if p and os.path.exists(p)),None)
    if font_path is None:raise FileNotFoundError('Set WEEK8_CJK_FONT to a CJK font or restore evidence/character.png')
    image=Image.new('L',(32,32),0);draw=ImageDraw.Draw(image);font=ImageFont.truetype(font_path,30)
    box=draw.textbbox((0,0),'算',font=font)
    draw.text(((32-(box[2]-box[0]))//2-box[0],(32-(box[3]-box[1]))//2-box[1]),'算',font=font,fill=255)
    image=image.point(lambda p:255 if p>127 else 0);image.save(path)
    return np.where(np.asarray(image)>127,1.,-1.).reshape(-1)


def main():
    picture=character();kinds=['outer','relu1','relu2','softmax'];sizes=[64,256,1024,4096]
    ratios=np.zeros((20,4,4));gallery=np.zeros((4,4,1024));invariance=[]
    for trial in range(20):
        rng=np.random.default_rng(1840+trial)
        for mi,m in enumerate(sizes):
            keys=unit_rows(rng,m,64);values=2*rng.integers(0,2,(m,1024),dtype=np.int8)-1;values[0]=picture
            for si,kind in enumerate(kinds):
                out,own=recall(keys,values,np.array([0]),kind)
                ratios[trial,mi,si]=interference(out,picture[None,:],own)[0]
                if trial==0:gallery[mi,si]=out[0]/own[0]
            if trial==0:
                weights=np.exp(8*(keys@keys[0]));unscaled=weights@values/weights[0]
                invariance.append(float(np.max(abs(unscaled-gallery[mi,3]))))
    med=np.median(ratios,axis=0)
    fig,axes=plt.subplots(4,4,figsize=(10,11));y,x=np.indices((32,32))
    for mi,m in enumerate(sizes):
        for si,kind in enumerate(kinds):
            ax=axes[mi,si];rec=gallery[mi,si];color='#ce4e4e' if ratios[0,mi,si]<1 else '#737e87'
            ax.scatter(x.ravel(),y.ravel(),s=np.clip((rec+1)/2,0,4)*3.3,c=color,linewidths=0)
            ax.set(ylim=(33,-2),xlim=(-2,33),aspect='equal',xticks=[],yticks=[])
            for spine in ax.spines.values():spine.set_visible(False)
            ax.set_xlabel(f'R = {ratios[0,mi,si]:.3f}')
            if mi==0:ax.set_title(kind if kind!='softmax' else 'Softmax, beta = 8')
            if si==0:ax.set_ylabel(f'M = {m}')
    fig.suptitle('Recall a 32 x 32 picture: dot area scales with recalled ink',fontsize=15)
    figure('part4-recall.png',fig)
    save_json('part4_metrics.json',{'sizes':sizes,'kinds':kinds,'median_ratios':med,'gallery_ratios':ratios[0],
        'all_ratios':ratios,'draws':20,'seeds':'1840..1859','beta':8,'dimension':64,
        'softmax_exponential_max_difference':max(invariance),'character':'算'})
    np.savez(ROOT/'artifacts/picture-recall.npz',gallery=gallery,ratios=ratios,picture=picture)
    print('Median ratios',med,flush=True)


if __name__=='__main__':main()
