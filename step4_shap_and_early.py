"""
Step 4: SHAP Interpretability + Early Prediction Curve + Classification
This is the KEY differentiator for Distinction grade.
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV, ElasticNetCV
from sklearn.metrics import (mean_squared_error, roc_curve, auc, 
                             confusion_matrix, classification_report, accuracy_score)
import shap
warnings.filterwarnings('ignore')
np.random.seed(42)

OUTPUT_DIR = './output/'

print("="*60)
print("Step 4: SHAP + Early Prediction + Classification")
print("="*60)

# ============================================================
# LOAD DATA
# ============================================================
with open(os.path.join(OUTPUT_DIR, 'all_batteries.pkl'), 'rb') as f:
    bat_dict = pickle.load(f)
feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index_col=0)

valid = feat_df.dropna(subset=['dq_variance','cycle_life'])
X_all = valid.drop(columns=['cycle_life']).fillna(0).values
y_raw = valid['cycle_life'].values
y_log = np.log(y_raw)
feat_names = list(valid.drop(columns=['cycle_life']).columns)
print(f"Data: {len(y_raw)} cells, {len(feat_names)} features")

# ============================================================
# PART 1: SHAP INTERPRETABILITY ANALYSIS
# ============================================================
print("\n" + "="*60)
print("PART 1: SHAP Interpretability")
print("="*60)

# Train final models on all data for SHAP
sc = StandardScaler()
X_scaled = sc.fit_transform(X_all)

# Random Forest (best model)
rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)
rf.fit(X_scaled, y_log)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, 
                                min_samples_leaf=3, subsample=0.8, random_state=42)
gb.fit(X_scaled, y_log)

# SHAP for Random Forest
print("Computing SHAP values for Random Forest...")
explainer_rf = shap.TreeExplainer(rf)
shap_values_rf = explainer_rf.shap_values(X_scaled)

# SHAP for Gradient Boosting
print("Computing SHAP values for Gradient Boosting...")
explainer_gb = shap.TreeExplainer(gb)
shap_values_gb = explainer_gb.shap_values(X_scaled)

# Plot 1: SHAP Summary (beeswarm)
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

plt.sca(axes[0])
shap.summary_plot(shap_values_rf, X_scaled, feature_names=feat_names, show=False, max_display=15)
axes[0].set_title('Random Forest', fontsize=14)

plt.sca(axes[1])
shap.summary_plot(shap_values_gb, X_scaled, feature_names=feat_names, show=False, max_display=15)
axes[1].set_title('Gradient Boosting', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig11_shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_shap_summary.png")

# Plot 2: SHAP Bar plot (mean |SHAP|)
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

plt.sca(axes[0])
shap.summary_plot(shap_values_rf, X_scaled, feature_names=feat_names, plot_type='bar', show=False, max_display=15)
axes[0].set_title('Random Forest - Mean |SHAP|', fontsize=14)

plt.sca(axes[1])
shap.summary_plot(shap_values_gb, X_scaled, feature_names=feat_names, plot_type='bar', show=False, max_display=15)
axes[1].set_title('Gradient Boosting - Mean |SHAP|', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig12_shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig12_shap_bar.png")

# Plot 3: SHAP Dependence plots for top 3 features
top3_rf = np.argsort(-np.abs(shap_values_rf).mean(0))[:3]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, idx in zip(axes, top3_rf):
    plt.sca(ax)
    shap.dependence_plot(idx, shap_values_rf, X_scaled, feature_names=feat_names, 
                         show=False, ax=ax)
    ax.set_title(f'SHAP Dependence: {feat_names[idx]}', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig13_shap_dependence.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig13_shap_dependence.png")

# Plot 4: SHAP for individual predictions (waterfall for short/mid/long life)
sorted_idx = np.argsort(y_raw)
examples = {'Short life': sorted_idx[0], 'Medium life': sorted_idx[len(sorted_idx)//2], 'Long life': sorted_idx[-1]}

fig, axes = plt.subplots(1, 3, figsize=(21, 6))
for ax, (label, eidx) in zip(axes, examples.items()):
    plt.sca(ax)
    shap_exp = shap.Explanation(values=shap_values_rf[eidx], 
                                 base_values=explainer_rf.expected_value,
                                 data=X_scaled[eidx],
                                 feature_names=feat_names)
    shap.waterfall_plot(shap_exp, max_display=10, show=False)
    ax.set_title(f'{label} (actual: {y_raw[eidx]:.0f} cycles)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig14_shap_waterfall.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig14_shap_waterfall.png")

# ============================================================
# PART 2: EARLY PREDICTION CURVE
# ============================================================
print("\n" + "="*60)
print("PART 2: Early Prediction (varying number of cycles)")
print("="*60)

def extract_features_n_cycles(bat_dict, keys, n_cycles):
    """Extract features using only first n_cycles of data."""
    features_list = []
    valid_keys = []
    cycle_lives = []
    
    for k in keys:
        cell = bat_dict[k]
        cl = cell['cycle_life'].flatten()[0]
        if np.isnan(cl) or cl <= 0: continue
        cycles = cell['cycles']
        summary = cell['summary']
        if len(cycles) < n_cycles + 1: continue
        
        feats = {}
        
        # DeltaQ features: use cycle n_cycles vs cycle 10 (or cycle 5 if n_cycles < 10)
        ref_cycle = min(9, n_cycles - 2)  # reference cycle
        target_cycle = n_cycles - 1
        
        try:
            qdlin_t = np.array(cycles[str(target_cycle)]['Qdlin']).flatten()
            qdlin_r = np.array(cycles[str(ref_cycle)]['Qdlin']).flatten()
            if len(qdlin_t) < 10 or len(qdlin_r) < 10: continue
            ml = min(len(qdlin_t), len(qdlin_r))
            dq = qdlin_t[:ml] - qdlin_r[:ml]
            feats['dq_variance'] = np.log10(np.abs(np.var(dq)) + 1e-10)
            feats['dq_minimum'] = np.min(dq)
            feats['dq_mean'] = np.mean(dq)
            feats['dq_skewness'] = float(pd.Series(dq).skew())
            feats['dq_kurtosis'] = float(pd.Series(dq).kurtosis())
        except:
            continue
        
        # QD features
        qd = summary['QD']
        if len(qd) >= n_cycles:
            feats['qd_cycle2'] = qd[1] if len(qd) > 1 else np.nan
            feats['qd_cycle_last'] = qd[n_cycles-1]
            if qd[1] > 0:
                feats['qd_ratio'] = qd[n_cycles-1] / qd[1]
            start = max(0, n_cycles - 10)
            x_fit = np.arange(start, n_cycles)
            y_fit = qd[start:n_cycles]
            if len(y_fit) >= 3:
                slope, intercept = np.polyfit(x_fit, y_fit, 1)
                feats['qd_slope'] = slope
        
        # IR features
        ir = summary['IR']
        if len(ir) >= n_cycles and len(ir) > 1:
            feats['ir_cycle2'] = ir[1]
            feats['ir_last'] = ir[n_cycles-1]
            feats['ir_diff'] = ir[n_cycles-1] - ir[1]
        
        # Temperature
        tavg = summary['Tavg']
        if len(tavg) >= n_cycles:
            feats['tavg_mean'] = np.mean(tavg[:n_cycles])
        
        features_list.append(feats)
        cycle_lives.append(cl)
        valid_keys.append(k)
    
    df = pd.DataFrame(features_list, index=valid_keys)
    df['cycle_life'] = cycle_lives
    return df

# Test different cycle windows
cycle_windows = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
early_results = {}

for nc in cycle_windows:
    print(f"\n  Testing with first {nc} cycles...")
    df_nc = extract_features_n_cycles(bat_dict, list(bat_dict.keys()), nc)
    if len(df_nc) < 15:
        print(f"    Only {len(df_nc)} cells, skipping")
        continue
    
    X_nc = df_nc.drop(columns=['cycle_life']).fillna(0).values
    y_nc = df_nc['cycle_life'].values
    y_nc_log = np.log(y_nc)
    
    # Test RF and ElasticNet
    for model_name, model_fn in [
        ('Random Forest', lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)),
        ('ElasticNet', lambda: ElasticNetCV(l1_ratio=[.1,.5,.7,.9], cv=3, max_iter=10000)),
    ]:
        rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)
        mapes = []
        for tr, te in rkf.split(X_nc):
            s = StandardScaler()
            Xtr = s.fit_transform(X_nc[tr])
            Xte = s.transform(X_nc[te])
            m = model_fn()
            m.fit(Xtr, y_nc_log[tr])
            pred = np.exp(m.predict(Xte))
            mapes.append(np.mean(np.abs(y_nc[te]-pred)/y_nc[te])*100)
        
        avg_mape = np.mean(mapes)
        std_mape = np.std(mapes)
        early_results[(nc, model_name)] = (avg_mape, std_mape)
        print(f"    {model_name:20s}: MAPE = {avg_mape:.1f}% ± {std_mape:.1f}%")

# Plot: Early prediction curve
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#374151',
    'axes.linewidth': 1.0,
})

fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=180)
ax.set_facecolor('#fbfbfa')

for model_name, color, marker in [('Random Forest', '#355C7D', 'o'), ('ElasticNet', '#C06C5B', 's')]:
    cycles_used = [nc for nc in cycle_windows if (nc, model_name) in early_results]
    means = [early_results[(nc, model_name)][0] for nc in cycles_used]
    stds = [early_results[(nc, model_name)][1] for nc in cycles_used]
    ax.errorbar(
        cycles_used, means, yerr=stds,
        marker=marker, markersize=8.5,
        linewidth=2.8, elinewidth=1.6, capsize=4,
        markerfacecolor='white', markeredgewidth=1.6,
        label=model_name, color=color, zorder=3
    )

ax.set_xlabel('Number of Early Cycles Used', fontsize=14, fontweight='semibold')
ax.set_ylabel('MAPE (%)', fontsize=14, fontweight='semibold')
ax.set_title('Prediction Accuracy vs. Number of Early Cycles', fontsize=17, fontweight='bold', pad=12)
ax.legend(
    fontsize=11.5, frameon=True, fancybox=False, edgecolor='#d1d5db',
    facecolor='white', framealpha=0.95, loc='upper right'
)
ax.grid(axis='y', color='#d1d5db', linestyle='--', linewidth=0.9, alpha=0.75)
ax.grid(axis='x', color='#e5e7eb', linestyle=':', linewidth=0.7, alpha=0.6)
ax.set_xticks(cycle_windows)
ax.tick_params(axis='both', labelsize=11, colors='#111827')
ax.margins(x=0.03)

best_rf = early_results[(100, 'Random Forest')][0]
best_en = early_results[(100, 'ElasticNet')][0]
ax.annotate(
    f'RF @100 = {best_rf:.1f}%',
    xy=(100, best_rf), xytext=(76, best_rf + 1.0),
    arrowprops=dict(arrowstyle='->', lw=1.1, color='#355C7D'),
    fontsize=10.5, color='#355C7D'
)
ax.annotate(
    f'EN @100 = {best_en:.1f}%',
    xy=(100, best_en), xytext=(76, best_en - 1.8),
    arrowprops=dict(arrowstyle='->', lw=1.1, color='#C06C5B'),
    fontsize=10.5, color='#C06C5B'
)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig15_early_prediction_curve.png'), dpi=220, facecolor=fig.get_facecolor())
plt.close()
print("\nSaved fig15_early_prediction_curve.png")

# ============================================================
# PART 3: CLASSIFICATION TASK
# ============================================================
print("\n" + "="*60)
print("PART 3: Classification (High/Low Lifetime)")
print("="*60)

# Binary classification: cycle_life > 550 = "high", else "low"
# Following Severson et al.
threshold = 550
y_class = (y_raw >= threshold).astype(int)
print(f"Threshold: {threshold} cycles")
print(f"Low lifetime (< {threshold}): {np.sum(y_class==0)} cells")
print(f"High lifetime (>= {threshold}): {np.sum(y_class==1)} cells")

# Test multiple classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

classifiers = {
    'Logistic Regression': lambda: LogisticRegressionCV(cv=3, max_iter=5000, random_state=42),
    'Random Forest': lambda: RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=2, random_state=42),
    'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
}

clf_results = {}
all_roc_data = {}

for clf_name, clf_fn in classifiers.items():
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    accs = []
    all_y_true, all_y_prob = [], []
    
    for tr, te in rkf.split(X_all):
        s = StandardScaler()
        Xtr = s.fit_transform(X_all[tr])
        Xte = s.transform(X_all[te])
        
        clf = clf_fn()
        clf.fit(Xtr, y_class[tr])
        pred = clf.predict(Xte)
        accs.append(accuracy_score(y_class[te], pred))
        
        if hasattr(clf, 'predict_proba'):
            prob = clf.predict_proba(Xte)[:, 1]
        else:
            prob = clf.decision_function(Xte)
        all_y_true.extend(y_class[te])
        all_y_prob.extend(prob)
    
    avg_acc = np.mean(accs)
    clf_results[clf_name] = avg_acc
    all_roc_data[clf_name] = (np.array(all_y_true), np.array(all_y_prob))
    print(f"  {clf_name:25s}: Accuracy = {avg_acc:.1%} ± {np.std(accs):.1%}")

# Also test with only first 5 cycles (as in Severson)
print(f"\n  Classification using only first 5 cycles:")
df_5 = extract_features_n_cycles(bat_dict, list(bat_dict.keys()), 5)
if len(df_5) > 15:
    X_5 = df_5.drop(columns=['cycle_life']).fillna(0).values
    y_5 = df_5['cycle_life'].values
    y_5_class = (y_5 >= threshold).astype(int)
    
    rkf5 = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)
    for clf_name, clf_fn in classifiers.items():
        accs5 = []
        for tr, te in rkf5.split(X_5):
            s = StandardScaler()
            Xtr = s.fit_transform(X_5[tr])
            Xte = s.transform(X_5[te])
            clf = clf_fn()
            clf.fit(Xtr, y_5_class[tr])
            accs5.append(accuracy_score(y_5_class[te], clf.predict(Xte)))
        print(f"    {clf_name:25s}: Accuracy = {np.mean(accs5):.1%} ± {np.std(accs5):.1%}")

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 8))
colors_roc = {'Logistic Regression': '#2196F3', 'Random Forest': '#4CAF50', 'Gradient Boosting': '#FF9800'}
for clf_name, (yt, yp) in all_roc_data.items():
    fpr, tpr, _ = roc_curve(yt, yp)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=colors_roc[clf_name], lw=2, label=f'{clf_name} (AUC = {roc_auc:.3f})')
ax.plot([0,1], [0,1], 'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_title(f'ROC Curves - Classification (threshold = {threshold} cycles)', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig16_roc_curves.png'), dpi=150)
plt.close()
print("\nSaved fig16_roc_curves.png")

# Confusion matrix for best classifier
best_clf_name = max(clf_results, key=clf_results.get)
print(f"\nBest classifier: {best_clf_name} ({clf_results[best_clf_name]:.1%})")

# Train on all data for confusion matrix visualization
sc_final = StandardScaler()
X_final = sc_final.fit_transform(X_all)
best_clf = classifiers[best_clf_name]()
best_clf.fit(X_final, y_class)
y_pred_final = best_clf.predict(X_final)
cm = confusion_matrix(y_class, y_pred_final)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels([f'Low (<{threshold})', f'High (≥{threshold})'], fontsize=12)
ax.set_yticklabels([f'Low (<{threshold})', f'High (≥{threshold})'], fontsize=12)
ax.set_xlabel('Predicted', fontsize=14); ax.set_ylabel('Actual', fontsize=14)
ax.set_title(f'Confusion Matrix - {best_clf_name}', fontsize=14)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=24, 
                color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig17_confusion_matrix.png'), dpi=150)
plt.close()
print("Saved fig17_confusion_matrix.png")

# ============================================================
# PART 4: ABLATION STUDY (feature group contribution)
# ============================================================
print("\n" + "="*60)
print("PART 4: Ablation Study")
print("="*60)

feature_groups = {
    'ΔQ(V) features only': [c for c in feat_names if c.startswith('dq_') and '5_4' not in c],
    'Capacity features only': [c for c in feat_names if c.startswith('qd_')],
    'IR features only': [c for c in feat_names if c.startswith('ir_')],
    'Temperature only': [c for c in feat_names if c.startswith('t')],
    'Charge time only': [c for c in feat_names if c.startswith('charge')],
    'ΔQ + Capacity': [c for c in feat_names if c.startswith('dq_') or c.startswith('qd_')],
    'All features': feat_names,
}

ablation_results = {}
for group_name, cols in feature_groups.items():
    cols_valid = [c for c in cols if c in feat_names]
    if len(cols_valid) == 0: continue
    col_idx = [feat_names.index(c) for c in cols_valid]
    X_sub = X_all[:, col_idx]
    
    rkf = RepeatedKFold(n_splits=3, n_repeats=5, random_state=42)
    mapes = []
    for tr, te in rkf.split(X_sub):
        s = StandardScaler()
        m = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42)
        m.fit(s.fit_transform(X_sub[tr]), y_log[tr])
        pred = np.exp(m.predict(s.transform(X_sub[te])))
        mapes.append(np.mean(np.abs(y_raw[te]-pred)/y_raw[te])*100)
    
    ablation_results[group_name] = (np.mean(mapes), np.std(mapes), len(cols_valid))
    print(f"  {group_name:30s} ({len(cols_valid):2d} feats): MAPE = {np.mean(mapes):.1f}% ± {np.std(mapes):.1f}%")

# Plot ablation
fig, ax = plt.subplots(figsize=(12, 6))
abl_names = list(ablation_results.keys())
abl_mapes = [ablation_results[n][0] for n in abl_names]
abl_stds = [ablation_results[n][1] for n in abl_names]
abl_nfeats = [ablation_results[n][2] for n in abl_names]

colors_abl = ['#4CAF50' if m == min(abl_mapes) else '#2196F3' for m in abl_mapes]
bars = ax.bar(range(len(abl_names)), abl_mapes, yerr=abl_stds, color=colors_abl, 
              edgecolor='k', lw=0.5, capsize=5)
for i, (b, m, n) in enumerate(zip(bars, abl_mapes, abl_nfeats)):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+abl_stds[i]+0.5, 
            f'{m:.1f}%\n({n}f)', ha='center', fontsize=9)
ax.set_xticks(range(len(abl_names)))
ax.set_xticklabels(abl_names, rotation=25, ha='right', fontsize=10)
ax.set_ylabel('MAPE (%)', fontsize=13)
ax.set_title('Ablation Study: Feature Group Contribution', fontsize=16)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig18_ablation.png'), dpi=150)
plt.close()
print("\nSaved fig18_ablation.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("STEP 4 COMPLETE - SUMMARY")
print("="*60)
print("""
Generated:
  fig11 - SHAP beeswarm (RF & GB)
  fig12 - SHAP bar (mean |SHAP|)
  fig13 - SHAP dependence (top 3 features)
  fig14 - SHAP waterfall (individual predictions)
  fig15 - Early prediction curve (cycles vs accuracy)
  fig16 - ROC curves (classification)
  fig17 - Confusion matrix
  fig18 - Ablation study

These figures cover:
  ✓ Model interpretability (SHAP) - KEY INNOVATION
  ✓ Practical applicability (early prediction)
  ✓ Classification task (replicates Severson)
  ✓ Ablation study (feature contribution)
""")
