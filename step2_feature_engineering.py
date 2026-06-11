"""
Step 2: Feature Engineering + Baseline Models
Based on Severson et al., Nature Energy 2019
"""
import os, pickle, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet, ElasticNetCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import pandas as pd

warnings.filterwarnings('ignore')

OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
with open(os.path.join(OUTPUT_DIR, 'all_batteries.pkl'), 'rb') as f:
    bat_dict = pickle.load(f)

# Filter valid cells
valid_keys = []
for k in bat_dict:
    cl = bat_dict[k]['cycle_life'].flatten()[0]
    if not np.isnan(cl) and cl > 0:
        valid_keys.append(k)
print(f"Valid cells: {len(valid_keys)}")

# ============================================================
# FEATURE EXTRACTION (Severson et al. 2019)
# ============================================================
def extract_features(bat_dict, keys):
    """
    Extract features from battery data following Severson et al.
    
    Key feature: DeltaQ_100-10(V) = Q100(V) - Q10(V)
    This is the difference in discharge capacity curves between cycle 100 and cycle 10,
    evaluated at uniformly spaced voltage points.
    
    The Qdlin field in the data is already the linearly interpolated discharge capacity.
    """
    features_list = []
    cycle_lives = []
    cell_keys = []
    
    for k in keys:
        cell = bat_dict[k]
        cl = cell['cycle_life'].flatten()[0]
        cycles = cell['cycles']
        summary = cell['summary']
        
        num_cycles = len(cycles)
        
        # Need at least 100 cycles of data
        if num_cycles < 101:
            print(f"  Skipping {k}: only {num_cycles} cycles")
            continue
        
        feats = {}
        
        # --------------------------------------------------
        # Feature Group 1: DeltaQ (V) features
        # DeltaQ_100-10(V) = Qdlin(cycle 100) - Qdlin(cycle 10)
        # --------------------------------------------------
        try:
            qdlin_100 = np.array(cycles['99']['Qdlin']).flatten()  # 0-indexed
            qdlin_10 = np.array(cycles['9']['Qdlin']).flatten()
            
            if len(qdlin_100) == 0 or len(qdlin_10) == 0:
                print(f"  Skipping {k}: empty Qdlin")
                continue
            
            # Make sure same length
            min_len = min(len(qdlin_100), len(qdlin_10))
            qdlin_100 = qdlin_100[:min_len]
            qdlin_10 = qdlin_10[:min_len]
            
            delta_q = qdlin_100 - qdlin_10
            
            feats['dq_variance'] = np.log10(np.abs(np.var(delta_q)) + 1e-10)
            feats['dq_minimum'] = np.min(delta_q)
            feats['dq_mean'] = np.mean(delta_q)
            feats['dq_skewness'] = float(pd.Series(delta_q).skew())
            feats['dq_kurtosis'] = float(pd.Series(delta_q).kurtosis())
            feats['dq_max_minus_min'] = np.max(delta_q) - np.min(delta_q)
        except Exception as e:
            print(f"  Skipping {k}: DeltaQ error: {e}")
            continue
        
        # --------------------------------------------------
        # Feature Group 2: Discharge capacity features
        # --------------------------------------------------
        try:
            qd = summary['QD']
            
            # Discharge capacity at cycle 2 (index 1)
            feats['qd_cycle2'] = qd[1] if len(qd) > 1 else np.nan
            
            # Discharge capacity at cycle 100 (index 99)
            feats['qd_cycle100'] = qd[99] if len(qd) > 99 else np.nan
            
            # Capacity ratio cycle 100 / cycle 2
            if feats['qd_cycle2'] > 0:
                feats['qd_ratio_100_2'] = feats['qd_cycle100'] / feats['qd_cycle2']
            else:
                feats['qd_ratio_100_2'] = np.nan
            
            # Slope & intercept of capacity fade (linear fit, cycles 91-100)
            if len(qd) >= 100:
                x_fit = np.arange(91, 101)
                y_fit = qd[90:100]
                if len(y_fit) == 10:
                    slope, intercept = np.polyfit(x_fit, y_fit, 1)
                    feats['qd_slope_91_100'] = slope
                    feats['qd_intercept_91_100'] = intercept
                else:
                    feats['qd_slope_91_100'] = np.nan
                    feats['qd_intercept_91_100'] = np.nan
            
            # Slope of cycles 2-100
            if len(qd) >= 100:
                x_fit2 = np.arange(2, 101)
                y_fit2 = qd[1:100]
                slope2, intercept2 = np.polyfit(x_fit2, y_fit2, 1)
                feats['qd_slope_2_100'] = slope2
        except Exception as e:
            print(f"  Warning {k}: QD features error: {e}")
        
        # --------------------------------------------------
        # Feature Group 3: Charge time, IR, Temperature
        # --------------------------------------------------
        try:
            ct = summary['chargetime']
            if len(ct) > 1:
                feats['chargetime_cycle2'] = ct[1]
            if len(ct) >= 100:
                feats['chargetime_cycle100'] = ct[99]
        except:
            pass
        
        try:
            ir = summary['IR']
            if len(ir) > 1:
                feats['ir_cycle2'] = ir[1]
            if len(ir) >= 100:
                feats['ir_cycle100'] = ir[99]
                feats['ir_diff_100_2'] = ir[99] - ir[1]
        except:
            pass
        
        try:
            tavg = summary['Tavg']
            tmin = summary['Tmin']
            tmax = summary['Tmax']
            if len(tavg) >= 100:
                feats['tavg_first100'] = np.mean(tavg[:100])
                feats['tmin_first100'] = np.min(tmin[:100])
                feats['tmax_first100'] = np.max(tmax[:100])
                feats['tavg_slope'] = np.polyfit(np.arange(100), tavg[:100], 1)[0]
        except:
            pass
        
        # --------------------------------------------------
        # Feature Group 4: Additional DeltaQ features (different cycle ranges)
        # --------------------------------------------------
        try:
            # DeltaQ between cycles 5 and 4 (for early classification)
            qdlin_5 = np.array(cycles['4']['Qdlin']).flatten()
            qdlin_4 = np.array(cycles['3']['Qdlin']).flatten()
            min_len2 = min(len(qdlin_5), len(qdlin_4))
            delta_q_5_4 = qdlin_5[:min_len2] - qdlin_4[:min_len2]
            feats['dq_5_4_variance'] = np.log10(np.abs(np.var(delta_q_5_4)) + 1e-10)
        except:
            pass
        
        features_list.append(feats)
        cycle_lives.append(cl)
        cell_keys.append(k)
    
    df = pd.DataFrame(features_list, index=cell_keys)
    df['cycle_life'] = cycle_lives
    return df

print("\nExtracting features...")
df = extract_features(bat_dict, valid_keys)
print(f"Feature matrix: {df.shape[0]} cells x {df.shape[1]-1} features")
print(f"\nFeature columns:\n{[c for c in df.columns if c != 'cycle_life']}")

# Save feature matrix
df.to_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'))
print("Saved feature_matrix.csv")

# ============================================================
# VISUALIZATION
# ============================================================
print("\nGenerating feature plots...")

# Plot 1: The key correlation - Var(DeltaQ) vs cycle life
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(df['dq_variance'], df['cycle_life'], c=df['cycle_life'], cmap='RdYlGn', edgecolors='k', linewidth=0.5, s=60)
ax.set_xlabel('log Var(ΔQ100-10(V))', fontsize=13)
ax.set_ylabel('Cycle Life', fontsize=13)
ax.set_title(f'Key Feature: ρ = {df["dq_variance"].corr(df["cycle_life"]):.3f}', fontsize=14)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(df['qd_slope_91_100'], df['cycle_life'], c=df['cycle_life'], cmap='RdYlGn', edgecolors='k', linewidth=0.5, s=60)
ax.set_xlabel('QD Slope (cycles 91-100)', fontsize=13)
ax.set_ylabel('Cycle Life', fontsize=13)
ax.set_title(f'Capacity Slope: ρ = {df["qd_slope_91_100"].corr(df["cycle_life"]):.3f}', fontsize=14)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.scatter(df['ir_diff_100_2'], df['cycle_life'], c=df['cycle_life'], cmap='RdYlGn', edgecolors='k', linewidth=0.5, s=60)
ax.set_xlabel('IR Change (cycle 100 - cycle 2)', fontsize=13)
ax.set_ylabel('Cycle Life', fontsize=13)
corr_ir = df['ir_diff_100_2'].corr(df['cycle_life']) if 'ir_diff_100_2' in df else 0
ax.set_title(f'IR Change: ρ = {corr_ir:.3f}', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_feature_correlations.png'), dpi=150)
plt.close()
print("Saved fig4_feature_correlations.png")

# Plot 2: Correlation heatmap
feat_cols = [c for c in df.columns if c != 'cycle_life']
fig, ax = plt.subplots(figsize=(14, 10))
corr_matrix = df[feat_cols + ['cycle_life']].corr()
im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix))); ax.set_yticks(range(len(corr_matrix)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(corr_matrix.columns, fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_title('Feature Correlation Matrix', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_correlation_heatmap.png'), dpi=150)
plt.close()
print("Saved fig5_correlation_heatmap.png")

# ============================================================
# BASELINE MODELS (Severson et al. 2019)
# ============================================================
print("\n" + "="*60)
print("BASELINE MODELS")
print("="*60)

y = np.log(df['cycle_life'].values)  # Predict log(cycle_life) as in paper
y_raw = df['cycle_life'].values

# --- Model 1: Variance-only model ---
print("\n--- Model 1: Variance Model (single feature) ---")
X1 = df[['dq_variance']].values

# Use repeated K-fold cross-validation since we don't have the exact train/test split
from sklearn.model_selection import RepeatedKFold
results = {'model': [], 'rmse_train': [], 'rmse_test': [], 'mape_train': [], 'mape_test': []}

def evaluate_model(X, y, y_raw, model_name, alpha_range=None):
    rkf = RepeatedKFold(n_splits=3, n_repeats=10, random_state=42)
    rmse_trains, rmse_tests, mape_trains, mape_tests = [], [], [], []
    all_test_pred = np.zeros_like(y_raw, dtype=float)
    all_test_count = np.zeros_like(y_raw, dtype=float)
    
    for train_idx, test_idx in rkf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_raw_train, y_raw_test = y_raw[train_idx], y_raw[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        if X.shape[1] == 1:
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000)
        else:
            model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=3, max_iter=10000, random_state=42)
        
        model.fit(X_train_s, y_train)
        
        pred_train = np.exp(model.predict(X_train_s))
        pred_test = np.exp(model.predict(X_test_s))
        
        rmse_trains.append(np.sqrt(mean_squared_error(y_raw_train, pred_train)))
        rmse_tests.append(np.sqrt(mean_squared_error(y_raw_test, pred_test)))
        mape_trains.append(np.mean(np.abs(y_raw_train - pred_train) / y_raw_train) * 100)
        mape_tests.append(np.mean(np.abs(y_raw_test - pred_test) / y_raw_test) * 100)
        
        all_test_pred[test_idx] += pred_test
        all_test_count[test_idx] += 1
    
    avg_pred = all_test_pred / np.maximum(all_test_count, 1)
    
    print(f"  RMSE  - Train: {np.mean(rmse_trains):.1f} ± {np.std(rmse_trains):.1f}, Test: {np.mean(rmse_tests):.1f} ± {np.std(rmse_tests):.1f}")
    print(f"  MAPE  - Train: {np.mean(mape_trains):.1f}% ± {np.std(mape_trains):.1f}%, Test: {np.mean(mape_tests):.1f}% ± {np.std(mape_tests):.1f}%")
    
    return avg_pred, np.mean(rmse_tests), np.mean(mape_tests)

# Model 1: Variance only
pred1, rmse1, mape1 = evaluate_model(X1, y, y_raw, "Variance")

# --- Model 2: Discharge model (6 features) ---
print("\n--- Model 2: Discharge Model (DeltaQ + capacity features) ---")
discharge_features = ['dq_variance', 'dq_minimum', 'dq_mean', 'dq_skewness', 'dq_kurtosis',
                       'qd_cycle2', 'qd_cycle100', 'qd_ratio_100_2', 'qd_slope_91_100', 
                       'qd_intercept_91_100', 'qd_slope_2_100']
discharge_features = [f for f in discharge_features if f in df.columns]
X2 = df[discharge_features].fillna(0).values
pred2, rmse2, mape2 = evaluate_model(X2, y, y_raw, "Discharge")

# --- Model 3: Full model (all features) ---
print("\n--- Model 3: Full Model (all features) ---")
all_features = [c for c in df.columns if c != 'cycle_life']
X3 = df[all_features].fillna(0).values
pred3, rmse3, mape3 = evaluate_model(X3, y, y_raw, "Full")

# --- Naive baseline ---
print("\n--- Naive Baseline (predict mean) ---")
naive_pred = np.full_like(y_raw, np.mean(y_raw))
naive_rmse = np.sqrt(mean_squared_error(y_raw, naive_pred))
naive_mape = np.mean(np.abs(y_raw - naive_pred) / y_raw) * 100
print(f"  RMSE: {naive_rmse:.1f}, MAPE: {naive_mape:.1f}%")

# ============================================================
# RESULTS PLOTS
# ============================================================
print("\nGenerating result plots...")

# Predicted vs Observed
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
titles = ['Variance Model', 'Discharge Model', 'Full Model']
preds = [pred1, pred2, pred3]
mapes = [mape1, mape2, mape3]

for ax, title, pred, mape in zip(axes, titles, preds, mapes):
    ax.scatter(y_raw, pred, c='steelblue', edgecolors='k', linewidth=0.5, s=60, alpha=0.7)
    lims = [min(y_raw.min(), pred.min()) * 0.9, max(y_raw.max(), pred.max()) * 1.1]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Observed Cycle Life', fontsize=13)
    ax.set_ylabel('Predicted Cycle Life', fontsize=13)
    ax.set_title(f'{title}\nMAPE: {mape:.1f}%', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_model_predictions.png'), dpi=150)
plt.close()
print("Saved fig6_model_predictions.png")

# Summary table
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
summary_df = pd.DataFrame({
    'Model': ['Naive (mean)', 'Variance', 'Discharge', 'Full'],
    'Test RMSE': [f'{naive_rmse:.0f}', f'{rmse1:.0f}', f'{rmse2:.0f}', f'{rmse3:.0f}'],
    'Test MAPE (%)': [f'{naive_mape:.1f}', f'{mape1:.1f}', f'{mape2:.1f}', f'{mape3:.1f}']
})
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(OUTPUT_DIR, 'model_results_summary.csv'), index=False)
print("\nSaved model_results_summary.csv")

# Feature importance (from full model)
print("\nTraining final full model for feature importance...")
scaler = StandardScaler()
X_full_s = scaler.fit_transform(X3)
final_model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9], cv=4, max_iter=10000, random_state=42)
final_model.fit(X_full_s, y)
importance = pd.DataFrame({'feature': all_features, 'coefficient': final_model.coef_})
importance['abs_coef'] = np.abs(importance['coefficient'])
importance = importance.sort_values('abs_coef', ascending=False)
print("\nFeature Importance (ElasticNet coefficients):")
print(importance[importance['abs_coef'] > 0].to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
top_feats = importance[importance['abs_coef'] > 0].head(15)
ax.barh(range(len(top_feats)), top_feats['coefficient'].values, color='steelblue')
ax.set_yticks(range(len(top_feats)))
ax.set_yticklabels(top_feats['feature'].values)
ax.set_xlabel('Coefficient', fontsize=13)
ax.set_title('Feature Importance (ElasticNet)', fontsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_feature_importance.png'), dpi=150)
plt.close()
print("Saved fig7_feature_importance.png")

print(f"\n{'='*60}")
print("DONE! Check output/ folder for all results.")
print("Next: use output folder results for downstream model comparison.")
print(f"{'='*60}")
