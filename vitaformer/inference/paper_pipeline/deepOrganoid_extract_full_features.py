#!/usr/bin/env python3
"""
版本A 全量特征提取 — 修复归一化(16bit 1-99%拉伸) + 丰富可解释形态特征集,scale=1.0。
跑全部 3456 孔,输出 features/full_features.csv。GPU。
"""
from __future__ import annotations
import csv, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
VCO=Path('3D_OrganoidSeg/vco/organoid_seg')
TGT_CSV=Path('3D_OrganoidSeg/external_datasets/deepOrganoid/TrainingData/train_resolved.csv')
CKPT=Path('3D_OrganoidSeg/vco/runs/expFinal_fromscratch/best_miou.pth')
BCLIP=Path('pretrained/biomedclip/open_clip_pytorch_model.bin')
OUT=Path('3D_OrganoidSeg/deepOrganoid_v1')
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]; WINDOW=224; STRIDE=112; BATCH=128
sys.path.insert(0,str(VCO)); from model_v2 import OrganoidSegV2
def clean(v): return str(v).strip().strip('"')
def fnum(v,d=np.nan):
    try:
        x=float(v); return x if math.isfinite(x) else d
    except Exception: return d
def sid(r):
    try: col=str(int(float(r['Col'])))
    except Exception: col=clean(r['Col'])
    return f"{clean(r['CCL'])}_{clean(r['Row'])}{col}"
def load_new(p):
    raw=np.array(Image.open(p)).astype(np.float32); lo,hi=np.percentile(raw,1),np.percentile(raw,99)
    return np.stack([(np.clip((raw-lo)/max(hi-lo,1e-6),0,1)*255).astype(np.uint8)]*3,-1)
def coords(h,w):
    ph=max(h,WINDOW); pw=max(w,WINDOW); ys=list(range(0,ph-WINDOW+1,STRIDE)); xs=list(range(0,pw-WINDOW+1,STRIDE))
    if not ys or ys[-1]+WINDOW<ph: ys.append(ph-WINDOW)
    if not xs or xs[-1]+WINDOW<pw: xs.append(pw-WINDOW)
    return [(y,x) for y in ys for x in xs],ph-h,pw-w
@torch.inference_mode()
def infer(model,img,device):
    h,w=img.shape[:2]; cs,ph,pw=coords(h,w); imgp=np.pad(img,((0,ph),(0,pw),(0,0)),mode='reflect')
    a=np.asarray(imgp,np.float32)/255.0; chw=torch.from_numpy(a).permute(2,0,1)
    chw=(chw-torch.tensor(MEAN).view(3,1,1))/torch.tensor(STD).view(3,1,1)
    patches=torch.stack([chw[:,y:y+WINDOW,x:x+WINDOW] for y,x in cs])
    score=torch.zeros((4,h,w),device=device); cnt=torch.zeros((h,w),device=device)
    for st in range(0,len(cs),BATCH):
        ch=cs[st:st+BATCH]; b=patches[st:st+BATCH].to(device)
        with torch.amp.autocast('cuda',enabled=device.type=='cuda'):
            o=model(b); o=o[0] if isinstance(o,tuple) else o
        pr=F.softmax(o.float(),1)
        for i,(y,x) in enumerate(ch):
            y2=min(y+WINDOW,h); x2=min(x+WINDOW,w); score[:,y:y2,x:x2]+=pr[i,:,:y2-y,:x2-x]; cnt[y:y2,x:x2]+=1
    return (score/torch.clamp(cnt,min=1e-6).unsqueeze(0)).cpu().numpy()
def entropy_img(vals,bins=64):
    h,_=np.histogram(vals,bins=bins); p=h/max(h.sum(),1); p=p[p>0]
    return float(-(p*np.log(p)).sum())
def region_feats(mask,gray,prefix):
    out={}
    lab,n=ndi.label(mask); out[f'{prefix}_component_count']=int(n)
    areas=[]; circs=[]; solids=[]; eccs=[]; irregular=[]; holes=[]
    if n>0:
        objs=ndi.find_objects(lab)
        for i in range(1,n+1):
            sl=objs[i-1];
            if sl is None: continue
            obj=lab[sl]==i; area=int(obj.sum());
            if area<3: continue
            areas.append(area)
            er=ndi.binary_erosion(obj); bd=obj&(~er); per=max(float(bd.sum()),1.0)
            circs.append(float(4*math.pi*area/(per*per)))
            filled=ndi.binary_fill_holes(obj); solids.append(float(area/max(filled.sum(),1))); holes.append(float(max(filled.sum()-area,0)/max(area,1)))
            irregular.append(float(per/max(math.sqrt(area),1)))
            ys,xs=np.nonzero(obj)
            if len(xs)>4:
                cov=np.cov(np.stack([xs,ys])); ev=np.linalg.eigvalsh(cov); ev=np.clip(ev,1e-6,None)
                eccs.append(float(math.sqrt(1-ev[0]/ev[1])) if ev[1]>0 else 0.0)
    for q in (25,50,75): out[f'{prefix}_area_p{q}']=float(np.percentile(areas,q)) if areas else 0.0
    out[f'{prefix}_area_mean']=float(np.mean(areas)) if areas else 0.0
    out[f'{prefix}_area_max']=float(max(areas)) if areas else 0.0
    out[f'{prefix}_equiv_diam_median']=float(math.sqrt(4*np.median(areas)/math.pi)) if areas else 0.0
    out[f'{prefix}_small_frac']=float(np.mean([a<=80 for a in areas])) if areas else 0.0
    out[f'{prefix}_large_frac']=float(np.mean([a>=300 for a in areas])) if areas else 0.0
    out[f'{prefix}_circularity_mean']=float(np.mean(circs)) if circs else 0.0
    out[f'{prefix}_solidity_mean']=float(np.mean(solids)) if solids else 0.0
    out[f'{prefix}_eccentricity_mean']=float(np.mean(eccs)) if eccs else 0.0
    out[f'{prefix}_boundary_irreg_mean']=float(np.mean(irregular)) if irregular else 0.0
    out[f'{prefix}_hole_frac_mean']=float(np.mean(holes)) if holes else 0.0
    out[f'{prefix}_fragmentation']=float(n/max(mask.sum()/1000.0,1e-6)) if mask.sum()>0 else 0.0
    if mask.any():
        vals=gray[mask]; gy,gx=np.gradient(gray); grad=np.sqrt(gx*gx+gy*gy)
        out[f'{prefix}_mean_intensity']=float(vals.mean()); out[f'{prefix}_intensity_std']=float(vals.std())
        out[f'{prefix}_pellucidity']=float(np.percentile(vals,75)-np.percentile(vals,25))
        out[f'{prefix}_entropy']=entropy_img(vals); out[f'{prefix}_gradient']=float(grad[mask].mean())
        out[f'{prefix}_edge_density']=float((grad[mask]>np.percentile(grad,90)).mean())
    else:
        for k in ('mean_intensity','intensity_std','pellucidity','entropy','gradient','edge_density'): out[f'{prefix}_{k}']=0.0
    return out
def feats(prob,gray):
    pred=prob.argmax(0).astype(np.uint8); tot=pred.size
    fg=(pred==1)|(pred==2)
    d=dict(background_area_ratio=float((pred==0).mean()),foreground_area_ratio=float(fg.mean()),
        alive_hard_area_ratio=float((pred==1).mean()),dead_hard_area_ratio=float((pred==2).mean()),bubble_area_ratio=float((pred==3).mean()),
        alive_prob_sum_ratio=float(prob[1].sum()/tot),dead_prob_sum_ratio=float(prob[2].sum()/tot),foreground_prob_sum_ratio=float((prob[1]+prob[2]).sum()/tot),
        alive_dead_soft_ratio=float(prob[1].sum()/max((prob[1]+prob[2]).sum(),1e-8)),
        seg_uncertainty=float((-(np.clip(prob,1e-8,1)*np.log(np.clip(prob,1e-8,1))).sum(0)).mean()),
        pred_mean_confidence=float(prob.max(0).mean()))
    d.update(region_feats(fg,gray,'fg'))
    return d
def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log=open(OUT/'logs/extract_full.log','w')
    def P(*a):
        s=' '.join(str(x) for x in a); print(s); log.write(s+'\n'); log.flush()
    m=OrganoidSegV2(weight_path=str(BCLIP),num_classes=4,freeze_blocks=0)
    ck=torch.load(CKPT,map_location='cpu',weights_only=True); m.load_state_dict(ck.get('model_state',ck),strict=True); m.eval().to(device)
    rows=[r for r in csv.DictReader(open(TGT_CSV)) if Path(r['IXM_BestFocus_resolved']).exists()]
    P(f"device={device} 孔数={len(rows)}")
    recs=[]; t0=time.time()
    for i,r in enumerate(rows):
        meta=dict(sample_id=sid(r),Row=clean(r['Row']),Col=clean(r['Col']),Func=clean(r['Func']),Cmpd1=clean(r['Cmpd1']),
                  Conc1=fnum(r['Conc1']),Lum=fnum(r['Lum']),Lum_FA=fnum(r['Lum_FA']),CCL=clean(r['CCL']))
        img=load_new(r['IXM_BestFocus_resolved']); prob=infer(m,img,device)
        gray=np.array(Image.fromarray(img).convert('L')).astype(np.float32)
        rec=dict(meta); rec.update(feats(prob,gray)); recs.append(rec)
        if (i+1)%200==0: P(f"  {i+1}/{len(rows)} {time.time()-t0:.0f}s")
    fields=list(recs[0].keys())
    with open(OUT/'features/full_features.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(recs)
    P(f"写出 {len(recs)} 行 x {len(fields)} 列 -> features/full_features.csv"); P("DONE")
if __name__=='__main__': (OUT/'features').mkdir(exist_ok=True); main()
