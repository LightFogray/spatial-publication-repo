"""Chicago road audit, identity check and equal-risk calibration (Tables 1 and 2).

Part of the reproducibility package for
"Crime concentration after spatial publication: aligning random baselines with
observation processes".

Data are located through the SPATIAL_DATA environment variable and default to
<repo>/data/raw; outputs are written to <repo>/outputs.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("SPATIAL_DATA", REPO / "data" / "raw"))
OUTDIR = Path(os.environ.get("SPATIAL_OUTPUT", REPO / "outputs"))
OUTDIR.mkdir(parents=True, exist_ok=True)


from pathlib import Path
import json, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]; OUT=OUTDIR; OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(20261118)
STREET={'residential','primary','secondary','tertiary','unclassified','living_street','trunk'}
def gini(y):
    h=np.bincount(y); v=np.arange(len(h)); end=np.cumsum(h); start=end-h+1
    return (np.sum(v*h*(start+end))/(len(y)*y.sum())-(len(y)+1)/len(y)) if y.sum() else np.nan
def main():
    raw=json.load(open(str(RAW / "chicago_roads.geojson"),encoding='utf8'))['features']
    kept={}; excluded=0
    for f in raw:
        hw=f['properties'].get('highway'); hw=hw if isinstance(hw,list) else [hw]
        if not any(h in STREET for h in hw): excluded+=1; continue
        if f['geometry']['type']!='LineString': continue
        cs=tuple(tuple(x[:2]) for x in f['geometry']['coordinates']); key=min(cs,cs[::-1])
        kept.setdefault(key,f)
    # Split each line into <=25m straight pieces; midpoint quadrature with exact length weights.
    A=[]; Z=[]; owner=[]; lengths=[]
    scale=np.array([111320*np.cos(np.deg2rad(41.88)),110540.])
    for i,cs in enumerate(kept):
        pts=np.array(cs)*scale; total=0
        for a,z in zip(pts[:-1],pts[1:]):
            length=np.linalg.norm(z-a)
            if length<=0: continue
            n=int(np.ceil(length/25)); edges=a+(z-a)*np.linspace(0,1,n+1)[:,None]
            A.extend(edges[:-1]); Z.extend(edges[1:]); owner.extend([i]*n); total+=length
        lengths.append(total)
    A=np.array(A); Z=np.array(Z); owner=np.array(owner); L=np.array(lengths)
    assert (L>0).all()
    mid=(A+Z)/2; vec=Z-A; sq=np.sum(vec*vec,axis=1); w=np.sqrt(sq); tree=cKDTree(mid)
    def match(points):
        output=[]
        for off in range(0,len(points),5000):
            P=points[off:off+5000]; dist,ix=tree.query(P,k=16)
            delta=P[:,None,:]-A[ix]; t=np.clip(np.sum(delta*vec[ix],axis=2)/sq[ix],0,1)
            dd=np.linalg.norm(delta-t[:,:,None]*vec[ix],axis=2); best=dd.argmin(axis=1); hit=ix[np.arange(len(P)),best]
            uncertain=np.where(dist[:,-1]-12.5<=dd.min(axis=1))[0]
            for j in uncertain:
                cand=np.asarray(tree.query_ball_point(P[j],float(dd[j].min()+12.500001)))
                de=P[j]-A[cand]; tt=np.clip(np.sum(de*vec[cand],axis=1)/sq[cand],0,1)
                hit[j]=cand[np.argmin(np.linalg.norm(de-tt[:,None]*vec[cand],axis=1))]
            output.extend(owner[hit])
        return np.asarray(output)
    mappings={'identity':owner,'match':match(mid)}
    for grid in [60,160,250]: mappings[str(grid)]=match(np.round(mid/grid)*grid)
    audit=dict(raw_records=len(raw),excluded_road_class=excluded,canonical_roads=len(L),quadrature_points=len(mid),road_km=L.sum()/1000,
        matching_identity_disagreement=float(np.average(mappings['match']!=owner,weights=w)),seed=20261118,R=500,B=999,
        support='25m-or-shorter straight-piece midpoints weighted by piece length; not continuous point simulation')
    (OUT/'network_audit.json').write_text(json.dumps(audit,indent=2))
    print(audit,flush=True)
    # Empirical counts: same existing offense basket and road bounding box, with corrected geometry.
    c=pd.read_csv(RAW / "chicago_2024_violent.csv").dropna(subset=['latitude','longitude'])
    p=c[['longitude','latitude']].to_numpy()*scale
    mask=(p[:,0]>=A[:,0].min())&(p[:,0]<=np.maximum(A[:,0],Z[:,0]).max())&(p[:,1]>=np.minimum(A[:,1],Z[:,1]).min())&(p[:,1]<=np.maximum(A[:,1],Z[:,1]).max())
    p=p[mask]; N=len(p); y=np.bincount(match(p),minlength=len(L)); p0=L/L.sum()
    rows=[]; nulls={}
    for mode,mapping in mappings.items():
        q=np.bincount(mapping,weights=w,minlength=len(L)); q/=q.sum()
        nulls[mode]=np.array([gini(rng.multinomial(N,q)) for _ in range(999)])
    naive=nulls['identity']; summary=[]
    for sigma in [0,.5,1.]:
        rate=np.exp(rng.normal(-sigma*sigma/2,sigma,len(L))) if sigma else np.ones(len(L))
        for mode,mapping in mappings.items():
            q=np.bincount(mapping,weights=w*rate[owner],minlength=len(L)); q/=q.sum()
            stats=np.array([gini(rng.multinomial(N,q)) for _ in range(500)])
            for method,null in [('length',naive),('operator',nulls[mode])]:
                ps=(1+(null[:,None]>=stats).sum(axis=0))/1000; rej=(ps<=.05); n=500; ratehat=rej.mean(); z=norm.ppf(.975)
                center=(ratehat+z*z/(2*n))/(1+z*z/n); half=z*np.sqrt(ratehat*(1-ratehat)/n+z*z/(4*n*n))/(1+z*z/n)
                summary.append(dict(sigma=sigma,mode=mode,method=method,rejection=ratehat,low=center-half,high=center+half,mean_gini=stats.mean()))
                rows.extend(dict(sigma=sigma,mode=mode,method=method,rep=k,gini=float(stats[k]),p=float(ps[k])) for k in range(n))
            print('calibrated',sigma,mode,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'calibration_replicates.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'calibration_summary.csv',index=False)
    obs=dict(events=N,units=len(L),gini=gini(y),null_mean=naive.mean(),null_ratio=naive.mean()/gini(y),p=(1+sum(naive>=gini(y)))/1000)
    (OUT/'chicago_corrected.json').write_text(json.dumps(obs,indent=2)); print(obs,flush=True)
    pd.DataFrame({'length_m':L,'count':y}).to_csv(OUT/'chicago_canonical_counts.csv',index=False)
if __name__=='__main__': main()
