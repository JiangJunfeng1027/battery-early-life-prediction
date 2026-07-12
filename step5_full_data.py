"""
Step 5: Full Dataset (124 cells) + Cross-Batch Transfer Learning
- Load all 3 batches
- Re-run best models with full data
- Cross-batch generalization experiment (train on B1+B2, test on B3)
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error
import shap
warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = './output/'
DATA_DIR = './data/MIT/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("Step 5: Full Dataset + Cross-Batch Transfer Learning")
print("="*60)

# ============================================================
# LOAD ALL 3 BATCHES
# ============================================================
import h5py

MAT_FILES = {
    'batch1': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
    'batch2': '2018-04-03_varcharge_batchdata_updated_struct_errorcorrect.mat',
    'batch3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat',
}

def load_batch(mat_path, prefix):
    print(f"  Loading {mat_path}...")
    f = h5py.File(mat_path, 'r')
    batch = f['batch']
    num = batch['summary'].shape[0]
    print(f"    Found {num} cells")
    
    bat = {}
    for i in range(num):
        print(f"    Cell {i+1}/{num}", end='\r')
        try:
            cl = f[batch['cycle_life'][i,0]][()].flatten()
        except:
            cl = np.array([np.nan])
        
        try:
            policy = f[batch['policy_readable'][i,0]][()].tobytes()[::2].decode()
        except:
            policy = 'unknown'
        
        sg = f[batch['summary'][i,0]]
        summary = {}
        for mf, of in {'IR':'IR','QCharge':'QC','QDischarge':'QD','Tavg':'Tavg',
                        'Tmin':'Tmin','Tmax':'Tmax','chargetime':'chargetime','cycle':'cycle'}.items():
            try:
                summary[of] = sg[mf][0,:]
            except:
                summary[of] = np.array([])
        
        cg = f[batch['cycles'][i,0]]
        cycles = {}
        nc = cg['I'].shape[0]
        for j in range(nc):
            cd = {}
            for mf2, of2 in {'I':'I','Qc':'Qc','Qd':'Qd','Qdlin':'Qdlin',
                              'T':'T','Tdlin':'Tdlin','V':'V',
                              'discharge_dQdV':'dQdV','t':'t'}.items():
                try:
                    val = cg[mf2][j,0]
                    if isinstance(val, h5py.h5r.Reference):
                        cd[of2] = f[val][()].flatten()
                    elif isinstance(val, (np.floating, float)):
                        cd[of2] = np.array([val])
                    else:
                        cd[of2] = np.atleast_1d(val)
                except:
                    pass
            cycles[str(j)] = cd
        
        bat[f'{prefix}{i}'] = {
            'cycle_life': cl, 'charge_policy': policy,
            'summary': summary, 'cycles': cycles
        }
    f.close()
    print(f"    Done: {num} cells")
    return bat

# Check which batches exist
available = {}
for bn, mf in MAT_FILES.items():
    mp = os.path.join(DATA_DIR, mf)
    pkl_path = os.path.join(OUTPUT_DIR, f'{bn}_full.pkl')
    
    if os.path.exists(pkl_path):
        print(f"Loading {bn} from cache...")
        with open(pkl_path, 'rb') as fp:
            available[bn] = pickle.load(fp)
        print(f"  {bn}: {len(available[bn])} cells")
    elif os.path.exists(mp):
        prefix = {'batch1':'b1c','batch2':'b2c','batch3':'b3c'}[bn]
        available[bn] = load_batch(mp, prefix)
        with open(pkl_path, 'wb') as fp:
            pickle.dump(available[bn], fp)
        print(f"  Saved {bn}_full.pkl")
    else:
        print(f"  WARNING: {mp} not found, skipping {bn}")

# Merge (same logic as step1)
def merge_all(batches):
    b1 = batches.get('batch1', {}).copy()
    # The public five-cell exclusion belongs to the 2017-05-12 batch, not this
    # separately loaded 2018-02-20 batch. Label/window validity is checked below.
    
    b2 = batches.get('batch2', {}).copy()
    if b2:
        # Merge continued cells
        b2k = ['b2c7','b2c8','b2c9','b2c15','b2c16']
        b1k = ['b1c0','b1c1','b1c2','b1c3','b1c4']
        add_len = [662, 981, 1060, 208, 482]
        for i, bk in enumerate(b1k):
            if bk in b1 and b2k[i] in b2:
                b1[bk]['cycle_life'] = b1[bk]['cycle_life'] + add_len[i]
                for j in b1[bk]['summary']:
                    if j == 'cycle':
                        b1[bk]['summary'][j] = np.hstack((b1[bk]['summary'][j], 
                            b2[b2k[i]]['summary'][j] + len(b1[bk]['summary'][j])))
                    else:
                        b1[bk]['summary'][j] = np.hstack((b1[bk]['summary'][j], b2[b2k[i]]['summary'][j]))
                last = len(b1[bk]['cycles'])
                for ji, jk in enumerate(b2[b2k[i]]['cycles']):
                    b1[bk]['cycles'][str(last + ji)] = b2[b2k[i]]['cycles'][jk]
        for bk in b2k:
            b2.pop(bk, None)
    
    b3 = batches.get('batch3', {}).copy()
    for bad in ['b3c37','b3c2','b3c23','b3c32','b3c42','b3c43']:
        b3.pop(bad, None)
    
    merged = {**b1, **b2, **b3}
    print(f"\nTotal cells after merge: {len(merged)}")
    return merged

all_bat = merge_all(available)

# Save full merged dataset
with open(os.path.join(OUTPUT_DIR, 'all_batteries_full.pkl'), 'wb') as f:
    pickle.dump(all_bat, f)
print("Saved all_batteries_full.pkl")

# Tag each cell with its batch
batch_tags = {}
for k in all_bat:
    if k.startswith('b1c'): batch_tags[k] = 'batch1'
    elif k.startswith('b2c'): batch_tags[k] = 'batch2'
    elif k.startswith('b3c'): batch_tags[k] = 'batch3'

# ============================================================
# FEATURE EXTRACTION (same as step2 but for full data)
# ============================================================
print("\nExtracting features for full dataset...")

def extract_features(bat_dict, keys):
    features_list, cycle_lives, cell_keys = [], [], []
    for k in keys:
        cell = bat_dict[k]
        cl = cell['cycle_life'].flatten()[0]
        if np.isnan(cl) or cl <= 0: continue
        cycles = cell['cycles']
        summary = cell['summary']
        if len(cycles) < 101: continue
        
        feats = {}
        try:
            ql100 = np.array(cycles['99']['Qdlin']).flatten()
            ql10 = np.array(cycles['9']['Qdlin']).flatten()
            if len(ql100) < 10 or len(ql10) < 10: continue
            ml = min(len(ql100), len(ql10))
            dq = ql100[:ml] - ql10[:ml]
            feats['dq_variance'] = np.log10(np.abs(np.var(dq)) + 1e-10)
            feats['dq_minimum'] = np.min(dq)
            feats['dq_mean'] = np.mean(dq)
            feats['dq_skewness'] = float(pd.Series(dq).skew())
            feats['dq_kurtosis'] = float(pd.Series(dq).kurtosis())
            feats['dq_max_minus_min'] = np.max(dq) - np.min(dq)
        except:
            continue
        
        qd = summary['QD']
        if len(qd) >= 100:
            feats['qd_cycle2'] = qd[1]
            feats['qd_cycle100'] = qd[99]
            feats['qd_ratio_100_2'] = qd[99]/qd[1] if qd[1]>0 else np.nan
            sl, ic = np.polyfit(np.arange(91,101), qd[90:100], 1)
            feats['qd_slope_91_100'] = sl
            feats['qd_intercept_91_100'] = ic
            sl2, _ = np.polyfit(np.arange(2,101), qd[1:100], 1)
            feats['qd_slope_2_100'] = sl2
        
        ct = summary['chargetime']
        if len(ct) > 1: feats['chargetime_cycle2'] = ct[1]
        if len(ct) >= 100: feats['chargetime_cycle100'] = ct[99]
        
        ir = summary['IR']
        if len(ir) > 1: feats['ir_cycle2'] = ir[1]
        if len(ir) >= 100:
            feats['ir_cycle100'] = ir[99]
            feats['ir_diff_100_2'] = ir[99] - ir[1]
        
        tavg, tmin, tmax = summary['Tavg'], summary['Tmin'], summary['Tmax']
        if len(tavg) >= 100:
            feats['tavg_first100'] = np.mean(tavg[:100])
            feats['tmin_first100'] = np.min(tmin[:100])
            feats['tmax_first100'] = np.max(tmax[:100])
            feats['tavg_slope'] = np.polyfit(np.arange(100), tavg[:100], 1)[0]
        
        try:
            ql5 = np.array(cycles['4']['Qdlin']).flatten()
            ql4 = np.array(cycles['3']['Qdlin']).flatten()
            ml2 = min(len(ql5), len(ql4))
            dq54 = ql5[:ml2] - ql4[:ml2]
            feats['dq_5_4_variance'] = np.log10(np.abs(np.var(dq54)) + 1e-10)
        except:
            pass
        
        features_list.append(feats)
        cycle_lives.append(cl)
        cell_keys.append(k)
    
    df = pd.DataFrame(features_list, index=cell_keys)
    df['cycle_life'] = cycle_lives
    return df

df_full = extract_features(all_bat, list(all_bat.keys()))
df_full.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix_full.csv'))
print(f"Full feature matrix: {df_full.shape[0]} cells x {df_full.shape[1]-1} features")

# Print stats by batch
for bn in ['batch1','batch2','batch3']:
    bk = [k for k in df_full.index if batch_tags.get(k) == bn]
    if bk:
        cls = df_full.loc[bk, 'cycle_life']
        print(f"  {bn}: {len(bk)} cells, life range {cls.min():.0f}-{cls.max():.0f}, mean {cls.mean():.0f}")

# ============================================================
# RE-RUN BEST MODELS ON FULL DATA
# ============================================================
print("\n" + "="*60)
print("FULL DATA MODEL COMPARISON")
print("="*60)

X_full = df_full.drop(columns=['cycle_life']).fillna(0).values
y_full = df_full['cycle_life'].values
y_full_log = np.log(y_full)
feat_names = list(df_full.drop(columns=['cycle_life']).columns)

def cv_eval(name, model_fn, X, yl, yr):
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    psum, pcnt = np.zeros(len(yr)), np.zeros(len(yr))
    rmses, mapes = [], []
    for tr, te in rkf.split(X):
        s = StandardScaler()
        m = model_fn()
        m.fit(s.fit_transform(X[tr]), yl[tr])
        p = np.exp(m.predict(s.transform(X[te])))
        rmses.append(np.sqrt(np.mean((yr[te]-p)**2)))
        mapes.append(np.mean(np.abs(yr[te]-p)/yr[te])*100)
        psum[te] += p; pcnt[te] += 1
    pred = psum / np.maximum(pcnt, 1)
    rm, mm = np.mean(rmses), np.mean(mapes)
    print(f"  {name:30s} RMSE={rm:.0f}±{np.std(rmses):.0f}  MAPE={mm:.1f}%±{np.std(mapes):.1f}%")
    return pred, rm, mm

results = {}
for name, fn in [
    ('ElasticNet', lambda: ElasticNetCV(l1_ratio=[.1,.5,.7,.9], cv=3, max_iter=10000)),
    ('Random Forest', lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)),
    ('Gradient Boosting', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3, subsample=0.8, random_state=42)),
]:
    p, r, m = cv_eval(name, fn, X_full, y_full_log, y_full)
    results[name] = (p, r, m)

# ============================================================
# CROSS-BATCH TRANSFER LEARNING
# ============================================================
print("\n" + "="*60)
print("CROSS-BATCH TRANSFER LEARNING")
print("="*60)
print("Train on Batch 1+2, Test on Batch 3 (unseen batch)")

b12_idx = [i for i, k in enumerate(df_full.index) if batch_tags.get(k) in ['batch1','batch2']]
b3_idx = [i for i, k in enumerate(df_full.index) if batch_tags.get(k) == 'batch3']

if len(b3_idx) > 0:
    X_b12, y_b12, yr_b12 = X_full[b12_idx], y_full_log[b12_idx], y_full[b12_idx]
    X_b3, y_b3, yr_b3 = X_full[b3_idx], y_full_log[b3_idx], y_full[b3_idx]
    
    print(f"  Train set (B1+B2): {len(b12_idx)} cells")
    print(f"  Test set (B3):     {len(b3_idx)} cells")
    
    transfer_results = {}
    for name, fn in [
        ('ElasticNet', lambda: ElasticNetCV(l1_ratio=[.1,.5,.7,.9], cv=3, max_iter=10000)),
        ('Random Forest', lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)),
        ('Gradient Boosting', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3, subsample=0.8, random_state=42)),
    ]:
        s = StandardScaler()
        m = fn()
        m.fit(s.fit_transform(X_b12), y_b12)
        pred = np.exp(m.predict(s.transform(X_b3)))
        rmse = np.sqrt(np.mean((yr_b3 - pred)**2))
        mape = np.mean(np.abs(yr_b3 - pred) / yr_b3) * 100
        transfer_results[name] = (pred, rmse, mape)
        print(f"  {name:30s} RMSE={rmse:.0f}  MAPE={mape:.1f}%")
    
    # Plot: Transfer learning results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, (pred, rmse, mape)) in zip(axes, transfer_results.items()):
        ax.scatter(yr_b3, pred, c='steelblue', edgecolors='k', s=60, alpha=0.7)
        lims = [min(yr_b3.min(), pred.min())*0.9, max(yr_b3.max(), pred.max())*1.1]
        ax.plot(lims, lims, 'r--', lw=2)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Observed (Batch 3)', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title(f'{name}\nMAPE: {mape:.1f}%', fontsize=13)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Cross-Batch Transfer: Train B1+B2 → Test B3', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig19_transfer_learning.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fig19_transfer_learning.png")
    
    # Reverse: Train B3, Test B1+B2
    print("\n  Reverse: Train B3 → Test B1+B2")
    for name, fn in [
        ('Random Forest', lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)),
    ]:
        s = StandardScaler()
        m = fn()
        m.fit(s.fit_transform(X_b3), y_b3)
        pred = np.exp(m.predict(s.transform(X_b12)))
        mape = np.mean(np.abs(yr_b12 - pred) / yr_b12) * 100
        print(f"    {name}: MAPE={mape:.1f}%")
else:
    print("  Batch 3 not available - skipping transfer learning")
    print("  (Download 2018-04-12_batchdata_updated_struct_errorcorrect.mat)")

# ============================================================
# FULL DATA PLOTS
# ============================================================
print("\nGenerating full data plots...")

# Cycle life distribution (full)
fig, ax = plt.subplots(figsize=(10, 5))
colors_batch = {'batch1': '#2196F3', 'batch2': '#4CAF50', 'batch3': '#FF9800'}
for bn, color in colors_batch.items():
    bk = [k for k in df_full.index if batch_tags.get(k) == bn]
    if bk:
        ax.hist(df_full.loc[bk, 'cycle_life'], bins=20, alpha=0.6, color=color, 
                label=f'{bn} ({len(bk)} cells)', edgecolor='black')
ax.set_xlabel('Cycle Life', fontsize=14)
ax.set_ylabel('Count', fontsize=14)
ax.set_title(f'Cycle Life Distribution - Full Dataset ({len(df_full)} cells)', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig20_full_distribution.png'), dpi=150)
plt.close()
print("Saved fig20_full_distribution.png")

# Model comparison bar chart (full data)
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
mapes = [results[n][2] for n in names]
colors = ['#2196F3', '#4CAF50', '#FF9800']
bars = ax.bar(names, mapes, color=colors, edgecolor='k', lw=0.5)
for b, m in zip(bars, mapes):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{m:.1f}%', 
            ha='center', fontsize=14, fontweight='bold')
ax.set_ylabel('MAPE (%)', fontsize=14)
ax.set_title(f'Model Performance on Full Dataset ({len(df_full)} cells)', fontsize=16)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig21_full_model_comparison.png'), dpi=150)
plt.close()
print("Saved fig21_full_model_comparison.png")

# SHAP on full data
print("Computing SHAP on full data...")
sc = StandardScaler()
X_s = sc.fit_transform(X_full)
rf_full = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)
rf_full.fit(X_s, y_full_log)
explainer = shap.TreeExplainer(rf_full)
sv = explainer.shap_values(X_s)

fig, ax = plt.subplots(figsize=(10, 8))
plt.sca(ax)
shap.summary_plot(sv, X_s, feature_names=feat_names, show=False, max_display=15)
ax.set_title(f'SHAP Summary - Full Dataset ({len(df_full)} cells)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig22_shap_full.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig22_shap_full.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("STEP 5 COMPLETE")
print("="*60)
print(f"""
Full dataset: {len(df_full)} cells (vs 35 before)
Results:""")
for name, (p, r, m) in results.items():
    print(f"  {name}: MAPE = {m:.1f}%")
if len(b3_idx) > 0:
    print(f"\nCross-batch transfer (B1+B2 → B3):")
    for name, (p, r, m) in transfer_results.items():
        print(f"  {name}: MAPE = {m:.1f}%")
print("""
New figures: fig19-fig22
""")
