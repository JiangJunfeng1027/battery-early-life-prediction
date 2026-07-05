import numpy as np, os, json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon, spearmanr
rng = np.random.default_rng(0)
base='/sessions/exciting-admiring-allen/mnt/battery_prediction/tmp/natcomm2022_capacity_vs_eis/'
z=np.load(base+'variable_discharge_features.npz',allow_pickle=True)
cells,y=z['cell'],z['y']
freq=10**np.linspace(-1.66,3.9,100)
SPEC=z['data_2']  # (N,200): 100 Re + 100 -Im
blocks=[z['data_0'].reshape(-1,1),z['data_1'].reshape(-1,1),z['data_3'],z['data_4'].reshape(-1,1),z['data_5'].reshape(-1,1),z['data_6'],z['data_7'],z['data_8']]
CAP=np.hstack(blocks)
uc=sorted(set(cells.tolist()))
def fit_eval(Xtr,ytr,Xte,yte):
    sc=StandardScaler().fit(Xtr)
    m=Ridge(alpha=10.0).fit(sc.transform(Xtr),ytr)
    return float(np.mean(np.abs(yte-m.predict(sc.transform(Xte)))))
def loocv(X):
    per={}
    for c in uc:
        te=cells==c; per[c]=fit_eval(X[~te],y[~te],X[te],y[te])
    return per
# band compression: 3 physical bands x (Re,Im) means = 6D
def bands(spec):
    lo=freq<1.0; mid=(freq>=1.0)&(freq<=200.0); hi=freq>200.0
    cols=[]
    for m in (lo,mid,hi):
        cols += [spec[:,:100][:,m].mean(1,keepdims=True), spec[:,100:][:,m].mean(1,keepdims=True)]
    return np.hstack(cols)
# literature 3 freqs (2.16, 6.5, 17.8 Hz) Re+Im = 6D
li=[int(np.argmin(np.abs(freq-f))) for f in (2.16,6.5,17.8)]
LIT=np.hstack([SPEC[:,li],SPEC[:,[100+i for i in li]]])
ARMS={'cap':CAP,'eis_only':SPEC,'cap+eis':np.hstack([CAP,SPEC]),
      'cap+bands6':np.hstack([CAP,bands(SPEC)]),'cap+lit3':np.hstack([CAP,LIT])}
print('== FOUR(+)-ARM LOOCV (per-cell MAE mean / median, cycles) ==')
R={}
for k,X in ARMS.items():
    per=loocv(X); v=np.array([per[c] for c in uc]); R[k]=per
    print(f'{k:11s} dims={X.shape[1]:3d}  mean {v.mean():.2f}  median {np.median(v):.2f}')
a=np.array([R['cap'][c] for c in uc])
for k in ['eis_only','cap+eis','cap+bands6','cap+lit3']:
    b=np.array([R[k][c] for c in uc]); w=wilcoxon(a,b)
    print(f'  {k:11s} vs cap: improved {(b<a).sum()}/24, Wilcoxon p={w.pvalue:.1e}')
print()
print('== FREQUENCY ECONOMY (fold-internal selection, leakage-aware) ==')
KS=[1,2,3,5,8,12,20,50,100]
econ={k:[] for k in KS}
for c in uc:
    te=cells==c; tr=~te
    rho=np.zeros(100)
    for i in range(100):
        rho[i]=max(abs(spearmanr(SPEC[tr,i],y[tr]).statistic),abs(spearmanr(SPEC[tr,100+i],y[tr]).statistic))
    order=np.argsort(rho)[::-1]
    for k in KS:
        idx=order[:k]; cols=np.concatenate([idx,idx+100])
        X=np.hstack([CAP,SPEC[:,cols]])
        econ[k].append(fit_eval(X[tr],y[tr],X[te],y[te]))
cap_mean=np.mean([R['cap'][c] for c in uc]); full_mean=np.mean([R['cap+eis'][c] for c in uc])
print(f'reference: cap {cap_mean:.2f} | cap+full-spectrum {full_mean:.2f}')
for k in KS:
    m=np.mean(econ[k]); rec=(cap_mean-m)/(cap_mean-full_mean)*100
    print(f'  top-{k:3d} freqs: mean MAE {m:.2f}  ({rec:5.1f}% of full-spectrum gain)')
print()
print('== GROUP SHIFT (campaign holdout PJ097-112 <-> PJ145-152) ==')
gA=np.isin(cells,[c for c in uc if c<='PJ112']); gB=~gA
print(f'group A: {len(set(cells[gA]))} cells, RUL range {y[gA].min():.0f}-{y[gA].max():.0f} (mean {y[gA].mean():.0f})')
print(f'group B: {len(set(cells[gB]))} cells, RUL range {y[gB].min():.0f}-{y[gB].max():.0f} (mean {y[gB].mean():.0f})')
S={}
for nm,X in [('cap',CAP),('cap+eis',np.hstack([CAP,SPEC])),('cap+bands6',ARMS['cap+bands6'])]:
    ab=fit_eval(X[gA],y[gA],X[gB],y[gB]); ba=fit_eval(X[gB],y[gB],X[gA],y[gA])
    S[nm]=(ab,ba); print(f'{nm:11s}  A->B {ab:.2f}   B->A {ba:.2f}   (LOOCV ref {np.mean([R[nm][c] for c in uc]):.2f})')
np.savez(os.path.expanduser('~/work/eis_out/deep_suite.npz'),
         cells=np.array(uc),
         **{f'arm_{k}':np.array([R[k][c] for c in uc]) for k in ARMS},
         econ_ks=np.array(KS),econ=np.array([np.mean(econ[k]) for k in KS]),
         shift_json=json.dumps(S))
print('saved deep_suite.npz')
