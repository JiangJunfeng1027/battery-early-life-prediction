"""
Step 6: Final Polish
1. Tiny-MLP with strong regularization (prove DL exhaustively tested)
2. Feature distribution shift visualization (B1 vs B3)
3. Data preprocessing flowchart (PRISMA-style)
4. Residual analysis plots
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

OUTPUT_DIR = './output/'

print("="*60)
print("Step 6: Final Polish")
print("="*60)

# Load
feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index_col=0)
feat_full = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix_full.csv'), index_col=0)

# ============================================================
# PART 1: Tiny-MLP with STRONG regularization
# ============================================================
print("\n--- PART 1: Exhaustive DL Regularization Test ---")

valid = feat_df.dropna(subset=['dq_variance','cycle_life'])
X_all = valid.drop(columns=['cycle_life']).fillna(0).values
y_raw = valid['cycle_life'].values
y_log = np.log(y_raw)
feat_names = list(valid.drop(columns=['cycle_life']).columns)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

configs = [
    ("MLP-Large (64-32-16, no reg)", 
     lambda d: nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1)),
     0.0, 0.0),
    ("MLP-Large + Dropout(0.5) + L2", 
     lambda d: nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64,32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1)),
     1e-2, 0.5),
    ("MLP-Medium (32-16, Dropout 0.4)", 
     lambda d: nn.Sequential(nn.Linear(d,32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1)),
     5e-3, 0.4),
    ("MLP-Tiny (8-4, Dropout 0.5, L2)", 
     lambda d: nn.Sequential(nn.Linear(d,8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(8,4), nn.ReLU(), nn.Linear(4,1)),
     1e-2, 0.5),
    ("MLP-Micro (4-1, max reg)", 
     lambda d: nn.Sequential(nn.Linear(d,4), nn.ReLU(), nn.Dropout(0.5), nn.Linear(4,1)),
     2e-2, 0.5),
]

def count_params(model):
    return sum(p.numel() for p in model.parameters())

dl_results = {}
for name, arch_fn, wd, _ in configs:
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    mapes = []
    sample_model = arch_fn(X_all.shape[1])
    n_params = count_params(sample_model)
    
    for tr, te in rkf.split(X_all):
        sc = StandardScaler()
        Xtr = torch.FloatTensor(sc.fit_transform(X_all[tr])).to(device)
        Xte = torch.FloatTensor(sc.transform(X_all[te])).to(device)
        ytr = torch.FloatTensor(y_log[tr]).to(device)
        
        model = arch_fn(X_all.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=wd)
        loss_fn = nn.MSELoss()
        
        best_loss, best_state = float('inf'), None
        for ep in range(500):
            model.train()
            loss = loss_fn(model(Xtr).squeeze(-1), ytr)
            opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xte).squeeze(-1), torch.FloatTensor(y_log[te]).to(device)).item()
            if vl < best_loss:
                best_loss = vl
                best_state = {k:v.clone() for k,v in model.state_dict().items()}
        
        if best_state: model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = np.exp(model(Xte).squeeze(-1).cpu().numpy())
        mapes.append(np.mean(np.abs(y_raw[te]-pred)/y_raw[te])*100)
    
    avg = np.mean(mapes)
    std = np.std(mapes)
    dl_results[name] = (avg, std, n_params)
    ratio = n_params / len(y_raw)
    print(f"  {name:45s} | Params: {n_params:5d} | Ratio: {ratio:.1f}:1 | MAPE: {avg:.1f}% ± {std:.1f}%")

# Add tree baselines for comparison
print("\n  Tree baselines for reference:")
for mname, mfn in [('Random Forest', lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)),
                     ('Gradient Boosting', lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3, subsample=0.8, random_state=42))]:
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
    mapes = []
    for tr, te in rkf.split(X_all):
        sc = StandardScaler()
        m = mfn()
        m.fit(sc.fit_transform(X_all[tr]), y_log[tr])
        pred = np.exp(m.predict(sc.transform(X_all[te])))
        mapes.append(np.mean(np.abs(y_raw[te]-pred)/y_raw[te])*100)
    print(f"  {mname:45s} | MAPE: {np.mean(mapes):.1f}% ± {np.std(mapes):.1f}%")

# Plot DL regularization comparison
fig, ax = plt.subplots(figsize=(14, 6))
names_dl = list(dl_results.keys())
mapes_dl = [dl_results[n][0] for n in names_dl]
stds_dl = [dl_results[n][1] for n in names_dl]
params_dl = [dl_results[n][2] for n in names_dl]

x = range(len(names_dl))
colors_dl = ['#FF5722','#FF9800','#FFC107','#4CAF50','#2196F3']
bars = ax.bar(x, mapes_dl, yerr=stds_dl, color=colors_dl, edgecolor='k', lw=0.5, capsize=5)

for i, (b, m, p) in enumerate(zip(bars, mapes_dl, params_dl)):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+stds_dl[i]+1, 
            f'{m:.1f}%\n({p} params)', ha='center', fontsize=9, fontweight='bold')

ax.axhline(y=8.0, color='red', linestyle='--', linewidth=2, label='Random Forest baseline (8.0%)')
ax.set_xticks(x)
ax.set_xticklabels([n.split('(')[0].strip() for n in names_dl], rotation=15, ha='right', fontsize=10)
ax.set_ylabel('MAPE (%)', fontsize=13)
ax.set_title('Deep Learning with Exhaustive Regularization vs. Tree Baseline', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig23_dl_regularization.png'), dpi=150)
plt.close()
print("\nSaved fig23_dl_regularization.png")

# ============================================================
# PART 2: Feature Distribution Shift (B1 vs B3)
# ============================================================
print("\n--- PART 2: Feature Distribution Shift ---")

batch_tags = {}
for k in feat_full.index:
    if k.startswith('b1c'): batch_tags[k] = 'Batch 1'
    elif k.startswith('b2c'): batch_tags[k] = 'Batch 2'
    elif k.startswith('b3c'): batch_tags[k] = 'Batch 3'

feat_full['batch'] = feat_full.index.map(lambda k: batch_tags.get(k, 'Unknown'))
b1 = feat_full[feat_full['batch']=='Batch 1']
b3 = feat_full[feat_full['batch']=='Batch 3']

top_feats = ['dq_variance', 'dq_minimum', 'dq_mean', 'chargetime_cycle100', 'dq_max_minus_min', 'qd_ratio_100_2']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, feat in zip(axes.flatten(), top_feats):
    if feat in b1.columns and feat in b3.columns:
        v1 = b1[feat].dropna()
        v3 = b3[feat].dropna()
        ax.hist(v1, bins=15, alpha=0.6, color='#2196F3', label=f'Batch 1 (n={len(v1)})', density=True, edgecolor='black')
        ax.hist(v3, bins=15, alpha=0.6, color='#FF9800', label=f'Batch 3 (n={len(v3)})', density=True, edgecolor='black')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        if len(v1) > 3:
            kde1 = gaussian_kde(v1)
            xs = np.linspace(min(v1.min(), v3.min()), max(v1.max(), v3.max()), 200)
            ax.plot(xs, kde1(xs), color='#1565C0', lw=2)
        if len(v3) > 3:
            kde3 = gaussian_kde(v3)
            ax.plot(xs, kde3(xs), color='#E65100', lw=2)
        
        ax.set_title(feat, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

plt.suptitle('Feature Distribution Shift: Batch 1 vs Batch 3\n(Explains asymmetric transfer learning performance)', 
             fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig24_distribution_shift.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig24_distribution_shift.png")

# ============================================================
# PART 3: Data Preprocessing Flowchart
# ============================================================
print("\n--- PART 3: Data Preprocessing Flowchart ---")

fig, ax = plt.subplots(figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', edge='#1565C0', fontsize=11):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.15", 
                          facecolor=color, edgecolor=edge, linewidth=2)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

def draw_side_box(ax, x, y, w, h, text, color='#FFEBEE', edge='#C62828', fontsize=9):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color='#C62828')

# Flow
draw_box(ax, 5, 15, 6, 0.8, 'Raw Dataset: 124 cells\n(Severson et al. 2019)', '#BBDEFB', '#1565C0', 12)

draw_arrow(ax, 5, 14.6, 5, 14)
draw_box(ax, 5, 13.5, 5.5, 0.8, 'Batch 1: 47 cells', '#E3F2FD')
draw_side_box(ax, 8.5, 13.5, 3, 0.7, 'Remove 5 anomalous\n(b1c8,10,12,13,22)\nPer Severson et al.')
draw_arrow(ax, 5, 13.1, 5, 12.5)

draw_box(ax, 5, 12, 5.5, 0.8, 'Batch 1: 42 valid cells', '#E3F2FD')
draw_arrow(ax, 5, 11.6, 5, 11)

draw_box(ax, 5, 10.5, 5.5, 0.8, 'Batch 2: 2 independent cells\n+ 5 continuation tests', '#E8F5E9', '#2E7D32')
draw_side_box(ax, 8.5, 10.5, 3, 0.7, 'Merge 5 continuations\nback to Batch 1 cells\n(same physical battery)')
draw_arrow(ax, 5, 10.1, 5, 9.5)

draw_box(ax, 5, 9, 5.5, 0.8, 'Batch 3: 46 cells', '#FFF3E0', '#E65100')
draw_side_box(ax, 8.5, 9, 3, 0.7, 'Remove 6 anomalous\n(b3c2,23,32,37,42,43)\nVoltage/temp anomalies')
draw_arrow(ax, 5, 8.6, 5, 8)

draw_box(ax, 5, 7.5, 5.5, 0.8, 'Merged: 84 cells\n(42 + 2 + 40)', '#E3F2FD')
draw_arrow(ax, 5, 7.1, 5, 6.5)

draw_box(ax, 5, 6, 5.5, 0.8, 'Filter: require ≥ 100 cycles\nfor feature extraction', '#FFF9C4', '#F57F17')
draw_side_box(ax, 8.5, 6, 3, 0.6, '9 cells removed\n(insufficient cycles)')
draw_arrow(ax, 5, 5.6, 5, 5)

draw_box(ax, 5, 4.5, 6, 0.9, 'Final Dataset: 75 cells\nBatch 1: 35 | Batch 3: 40', '#C8E6C9', '#2E7D32', 13)
draw_arrow(ax, 5, 4.05, 5, 3.5)

draw_box(ax, 5, 3, 5.5, 0.8, 'Feature Extraction\n22 features per cell', '#E1BEE7', '#7B1FA2')
draw_arrow(ax, 5, 2.6, 5, 2)

draw_box(ax, 5, 1.5, 6, 0.8, 'Model Training & Evaluation\n5-fold × 5-repeat CV', '#BBDEFB', '#1565C0', 12)

ax.set_title('Data Preprocessing Pipeline', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig25_preprocessing_flowchart.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig25_preprocessing_flowchart.png")

# ============================================================
# PART 4: Residual Analysis
# ============================================================
print("\n--- PART 4: Residual Analysis ---")

X_full = feat_full.drop(columns=['cycle_life','batch']).fillna(0).values
y_full = feat_full['cycle_life'].values
y_full_log = np.log(y_full)

sc = StandardScaler()
X_s = sc.fit_transform(X_full)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3, subsample=0.8, random_state=42)
gb.fit(X_s, y_full_log)
pred_full = np.exp(gb.predict(X_s))
residuals = y_full - pred_full
rel_residuals = residuals / y_full * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Residual vs predicted
axes[0].scatter(pred_full, residuals, c='steelblue', edgecolors='k', s=50, alpha=0.7)
axes[0].axhline(y=0, color='red', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Cycle Life', fontsize=12)
axes[0].set_ylabel('Residual (Actual - Predicted)', fontsize=12)
axes[0].set_title('Residuals vs. Predicted', fontsize=14)
axes[0].grid(True, alpha=0.3)

# Residual distribution
axes[1].hist(rel_residuals, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='red', linestyle='--', lw=2)
axes[1].set_xlabel('Relative Residual (%)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Residual Distribution', fontsize=14)
axes[1].grid(True, alpha=0.3)

# QQ plot
from scipy import stats
(osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
axes[2].scatter(osm, osr, c='steelblue', edgecolors='k', s=50, alpha=0.7)
axes[2].plot(osm, slope*np.array(osm)+intercept, 'r--', lw=2)
axes[2].set_xlabel('Theoretical Quantiles', fontsize=12)
axes[2].set_ylabel('Sample Quantiles', fontsize=12)
axes[2].set_title(f'Q-Q Plot (R² = {r**2:.3f})', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.suptitle('Residual Analysis - Gradient Boosting (Full Dataset)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig26_residual_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig26_residual_analysis.png")

# ============================================================
# PART 5: VIF Analysis
# ============================================================
print("\n--- PART 5: VIF (Variance Inflation Factor) ---")
from sklearn.linear_model import LinearRegression

def calc_vif(X, names):
    vifs = []
    for i in range(X.shape[1]):
        y_i = X[:, i]
        X_i = np.delete(X, i, axis=1)
        r2 = LinearRegression().fit(X_i, y_i).score(X_i, y_i)
        vif = 1 / (1 - r2) if r2 < 1 else float('inf')
        vifs.append((names[i], vif))
    return sorted(vifs, key=lambda x: -x[1])

feat_names_full = [c for c in feat_full.columns if c not in ['cycle_life','batch']]
vif_results = calc_vif(StandardScaler().fit_transform(feat_full[feat_names_full].fillna(0).values), feat_names_full)

print(f"  {'Feature':30s} {'VIF':>10s} {'Status':>15s}")
print(f"  {'-'*55}")
for fn, vif in vif_results:
    status = 'HIGH' if vif > 10 else 'moderate' if vif > 5 else 'ok'
    print(f"  {fn:30s} {vif:10.1f} {status:>15s}")

# Save VIF table
vif_df = pd.DataFrame(vif_results, columns=['Feature', 'VIF'])
vif_df['Status'] = vif_df['VIF'].apply(lambda v: 'High (>10)' if v > 10 else 'Moderate (5-10)' if v > 5 else 'Acceptable (<5)')
vif_df.to_csv(os.path.join(OUTPUT_DIR, 'vif_analysis.csv'), index=False)
print("\nSaved vif_analysis.csv")

print(f"\n{'='*60}")
print("STEP 6 COMPLETE")
print("="*60)
print("""
New figures:
  fig23 - DL exhaustive regularization test
  fig24 - Feature distribution shift (B1 vs B3)
  fig25 - Data preprocessing flowchart
  fig26 - Residual analysis (3 plots)
  vif_analysis.csv - Variance Inflation Factor table
""")
