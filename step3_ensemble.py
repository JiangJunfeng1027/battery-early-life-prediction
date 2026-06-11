"""
Step 3 (revised): Ensemble & Tree Models + Simplified DL
Target: Beat ElasticNet baseline (8.3% MAPE) with methods suited for small data
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = './output/'

print("="*60)
print("Step 3: Ensemble Models + Model Comparison")
print("="*60)

# Load
with open(os.path.join(OUTPUT_DIR, 'all_batteries.pkl'), 'rb') as f:
    bat_dict = pickle.load(f)
feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index_col=0)

# Prepare
valid = feat_df.dropna(subset=['dq_variance','cycle_life'])
X_all = valid.drop(columns=['cycle_life']).fillna(0).values
y_raw = valid['cycle_life'].values
y_log = np.log(y_raw)
feat_names = [c for c in valid.columns if c != 'cycle_life']
print(f"Data: {len(y_raw)} cells, {len(feat_names)} features")

# Also prepare DeltaQ curve data for a lightweight CNN
def get_dq_curves(bat_dict, keys, vlen=200):
    X = []
    for k in keys:
        cell = bat_dict[k]
        cycles = cell['cycles']
        if len(cycles) < 101: 
            X.append(np.zeros(vlen))
            continue
        try:
            q100 = np.array(cycles['99']['Qdlin']).flatten()
            q10 = np.array(cycles['9']['Qdlin']).flatten()
            ml = min(len(q100), len(q10))
            dq = q100[:ml] - q10[:ml]
            X.append(np.interp(np.linspace(0,1,vlen), np.linspace(0,1,len(dq)), dq))
        except:
            X.append(np.zeros(vlen))
    return np.array(X)

X_dq = get_dq_curves(bat_dict, valid.index.tolist())

# ============================================================
# CROSS-VALIDATION FRAMEWORK
# ============================================================
def cv_evaluate(name, model_fn, X, y_log, y_raw, n_splits=5, n_repeats=5):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    preds_sum = np.zeros(len(y_raw))
    preds_count = np.zeros(len(y_raw))
    rmses, mapes = [], []
    
    for tr, te in rkf.split(X):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te])
        
        model = model_fn()
        model.fit(Xtr, y_log[tr])
        pred = np.exp(model.predict(Xte))
        
        rmses.append(np.sqrt(np.mean((y_raw[te]-pred)**2)))
        mapes.append(np.mean(np.abs(y_raw[te]-pred)/y_raw[te])*100)
        preds_sum[te] += pred
        preds_count[te] += 1
    
    avg_pred = preds_sum / np.maximum(preds_count, 1)
    r = np.mean(rmses)
    m = np.mean(mapes)
    print(f"  {name:30s} RMSE={r:.0f}±{np.std(rmses):.0f}  MAPE={m:.1f}%±{np.std(mapes):.1f}%")
    return avg_pred, r, m

results = {}

# ============================================================
# 1. BASELINE MODELS
# ============================================================
print("\n--- Baseline Models ---")
p, r, m = cv_evaluate("ElasticNet", lambda: ElasticNetCV(l1_ratio=[.1,.5,.7,.9], cv=3, max_iter=10000), X_all, y_log, y_raw)
results['ElasticNet'] = (p, r, m)

p, r, m = cv_evaluate("Ridge", lambda: RidgeCV(alphas=[0.01,0.1,1,10,100]), X_all, y_log, y_raw)
results['Ridge'] = (p, r, m)

p, r, m = cv_evaluate("Lasso", lambda: LassoCV(cv=3, max_iter=10000), X_all, y_log, y_raw)
results['Lasso'] = (p, r, m)

# ============================================================
# 2. TREE ENSEMBLE MODELS
# ============================================================
print("\n--- Tree Ensemble Models ---")
p, r, m = cv_evaluate("Random Forest", lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42), X_all, y_log, y_raw)
results['Random Forest'] = (p, r, m)

p, r, m = cv_evaluate("Gradient Boosting", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3, subsample=0.8, random_state=42), X_all, y_log, y_raw)
results['Gradient Boosting'] = (p, r, m)

# ============================================================
# 3. SVR
# ============================================================
print("\n--- Support Vector Regression ---")
p, r, m = cv_evaluate("SVR (RBF)", lambda: SVR(kernel='rbf', C=10, epsilon=0.1), X_all, y_log, y_raw)
results['SVR'] = (p, r, m)

# ============================================================
# 4. COMBINED FEATURES: Engineered + DeltaQ curve stats
# ============================================================
print("\n--- Enhanced Feature Set (+ DeltaQ curve PCA) ---")
from sklearn.decomposition import PCA

# Add PCA components of DeltaQ curve as extra features
pca = PCA(n_components=5)
dq_pca = pca.fit_transform(StandardScaler().fit_transform(X_dq))
X_enhanced = np.hstack([X_all, dq_pca])
print(f"  Enhanced features: {X_enhanced.shape[1]} (original {X_all.shape[1]} + 5 PCA)")

p, r, m = cv_evaluate("ElasticNet (enhanced)", lambda: ElasticNetCV(l1_ratio=[.1,.5,.7,.9], cv=3, max_iter=10000), X_enhanced, y_log, y_raw)
results['ElasticNet+'] = (p, r, m)

p, r, m = cv_evaluate("GradBoost (enhanced)", lambda: GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3, subsample=0.8, random_state=42), X_enhanced, y_log, y_raw)
results['GradBoost+'] = (p, r, m)

# ============================================================
# 5. STACKING ENSEMBLE (meta-learner)
# ============================================================
print("\n--- Stacking Ensemble ---")
def make_stacking():
    return StackingRegressor(
        estimators=[
            ('en', ElasticNetCV(l1_ratio=[.1,.5,.7,.9], cv=3, max_iter=10000)),
            ('rf', RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3, random_state=42)),
            ('svr', SVR(kernel='rbf', C=10, epsilon=0.1)),
        ],
        final_estimator=RidgeCV(alphas=[0.01,0.1,1,10]),
        cv=3
    )

p, r, m = cv_evaluate("Stacking Ensemble", make_stacking, X_all, y_log, y_raw)
results['Stacking'] = (p, r, m)

p, r, m = cv_evaluate("Stacking (enhanced)", make_stacking, X_enhanced, y_log, y_raw)
results['Stacking+'] = (p, r, m)

# ============================================================
# 6. LIGHTWEIGHT NEURAL NET (small architecture for small data)
# ============================================================
print("\n--- Lightweight Neural Net ---")
torch.manual_seed(42)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class TinyNet(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 16), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train_tinynet(X, y_log, y_raw):
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    preds_sum = np.zeros(len(y_raw))
    preds_count = np.zeros(len(y_raw))
    rmses, mapes = [], []
    
    for tr, te in rkf.split(X):
        sc = StandardScaler()
        Xtr = torch.FloatTensor(sc.fit_transform(X[tr])).to(device)
        Xte = torch.FloatTensor(sc.transform(X[te])).to(device)
        ytr = torch.FloatTensor(y_log[tr]).to(device)
        
        model = TinyNet(X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
        loss_fn = nn.MSELoss()
        
        best_loss = float('inf')
        best_state = None
        for ep in range(500):
            model.train()
            pred = model(Xtr)
            loss = loss_fn(pred, ytr)
            opt.zero_grad(); loss.backward(); opt.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k:v.clone() for k,v in model.state_dict().items()}
        
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pred = np.exp(model(Xte).cpu().numpy())
        
        rmses.append(np.sqrt(np.mean((y_raw[te]-pred)**2)))
        mapes.append(np.mean(np.abs(y_raw[te]-pred)/y_raw[te])*100)
        preds_sum[te] += pred
        preds_count[te] += 1
    
    avg_pred = preds_sum / np.maximum(preds_count, 1)
    r, m = np.mean(rmses), np.mean(mapes)
    print(f"  {'TinyNet':30s} RMSE={r:.0f}±{np.std(rmses):.0f}  MAPE={m:.1f}%±{np.std(mapes):.1f}%")
    return avg_pred, r, m

p, r, m = train_tinynet(X_all, y_log, y_raw)
results['TinyNet'] = (p, r, m)

p, r, m = train_tinynet(X_enhanced, y_log, y_raw)
results['TinyNet+'] = (p, r, m)

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n" + "="*60)
print("FULL RESULTS COMPARISON")
print("="*60)

summary = []
for name, (pred, rmse, mape) in sorted(results.items(), key=lambda x: x[1][2]):
    summary.append({'Model': name, 'MAPE (%)': f'{mape:.1f}', 'RMSE': f'{rmse:.0f}'})
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'full_comparison.csv'), index=False)

# Best model
best_name = summary[0]['Model']
best_mape = results[best_name][2]
print(f"\nBest model: {best_name} (MAPE: {best_mape:.1f}%)")

# ============================================================
# PLOTS
# ============================================================
# Plot 1: Bar chart of all models
fig, ax = plt.subplots(figsize=(14, 6))
names = [s['Model'] for s in summary]
mape_vals = [results[n][2] for n in names]
colors = ['#4CAF50' if m <= 8.3 else '#2196F3' if m <= 12 else '#FF9800' for m in mape_vals]
bars = ax.bar(range(len(names)), mape_vals, color=colors, edgecolor='k', lw=0.5)
for i, (b, m) in enumerate(zip(bars, mape_vals)):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, f'{m:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.axhline(y=8.3, color='red', linestyle='--', linewidth=2, label='Baseline (ElasticNet-Discharge: 8.3%)')
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
ax.set_ylabel('MAPE (%)', fontsize=13)
ax.set_title('Complete Model Comparison', fontsize=16)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(mape_vals)*1.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_full_comparison.png'), dpi=150)
plt.close()
print("\nSaved fig8_full_comparison.png")

# Plot 2: Top 4 models predicted vs observed
top4 = [s['Model'] for s in summary[:4]]
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, name in zip(axes, top4):
    pred, _, mape = results[name]
    ax.scatter(y_raw, pred, c='steelblue', edgecolors='k', s=60, alpha=0.7)
    lims = [min(y_raw.min(), pred.min())*0.9, max(y_raw.max(), pred.max())*1.1]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Observed', fontsize=11); ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'{name}\nMAPE: {mape:.1f}%', fontsize=13)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_top_models.png'), dpi=150)
plt.close()
print("Saved fig9_top_models.png")

# Plot 3: Feature importance from best tree model
print("\nGenerating feature importance plot...")
sc = StandardScaler()
X_s = sc.fit_transform(X_all)
gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, min_samples_leaf=3, random_state=42)
gb.fit(X_s, y_log)
imp = pd.DataFrame({'feature': feat_names, 'importance': gb.feature_importances_})
imp = imp.sort_values('importance', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(imp)), imp['importance'].values, color='steelblue')
ax.set_yticks(range(len(imp)))
ax.set_yticklabels(imp['feature'].values)
ax.set_xlabel('Importance', fontsize=13)
ax.set_title('Feature Importance (Gradient Boosting)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_gb_importance.png'), dpi=150)
plt.close()
print("Saved fig10_gb_importance.png")

print(f"\n{'='*60}")
print("DONE! All models compared.")
print(f"{'='*60}")
