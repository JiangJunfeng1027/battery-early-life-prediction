"""
Step 3: Deep Learning Models for Battery Lifetime Prediction
- LSTM: Uses raw Qdlin sequences from early cycles
- 1D-CNN: Uses raw Qdlin sequences from early cycles  
- MLP: Uses engineered features (for comparison)
Goal: Beat baseline ElasticNet MAPE of 8.3%
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

OUTPUT_DIR = './output/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
print("="*60)
print("Step 3: Deep Learning Models")
print("="*60)

print("\nLoading data...")
with open(os.path.join(OUTPUT_DIR, 'all_batteries.pkl'), 'rb') as f:
    bat_dict = pickle.load(f)

# Also load feature matrix from step 2
feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index_col=0)

# ============================================================
# PREPARE SEQUENCE DATA FOR LSTM/CNN
# ============================================================
print("Preparing sequence data...")

def prepare_sequence_data(bat_dict, n_early_cycles=100, qdlin_len=1000):
    """
    Prepare raw Qdlin curve sequences for deep learning.
    For each cell, we stack Qdlin curves from cycle 1 to n_early_cycles.
    Also compute DeltaQ curves (difference from cycle 10).
    
    Returns:
        X_seq: (n_cells, n_cycles, qdlin_len) - raw Qdlin sequences
        X_dq: (n_cells, n_cycles, qdlin_len) - DeltaQ sequences  
        y: (n_cells,) - cycle life
        keys: list of cell keys
    """
    X_seq_list = []
    X_dq_list = []
    y_list = []
    key_list = []
    
    for k in bat_dict:
        cell = bat_dict[k]
        cl = cell['cycle_life'].flatten()[0]
        if np.isnan(cl) or cl <= 0:
            continue
        
        cycles = cell['cycles']
        if len(cycles) < n_early_cycles + 1:
            continue
        
        # Get Qdlin for first n_early_cycles
        qdlin_curves = []
        valid = True
        for c in range(n_early_cycles):
            cyc_key = str(c)
            if cyc_key not in cycles or 'Qdlin' not in cycles[cyc_key]:
                valid = False
                break
            ql = np.array(cycles[cyc_key]['Qdlin']).flatten()
            if len(ql) < 10:
                valid = False
                break
            # Interpolate to uniform length
            x_old = np.linspace(0, 1, len(ql))
            x_new = np.linspace(0, 1, qdlin_len)
            ql_interp = np.interp(x_new, x_old, ql)
            qdlin_curves.append(ql_interp)
        
        if not valid or len(qdlin_curves) != n_early_cycles:
            continue
        
        qdlin_arr = np.array(qdlin_curves)  # (n_cycles, qdlin_len)
        
        # DeltaQ: difference from cycle 10 (index 9)
        ref_cycle = qdlin_arr[9]  # cycle 10
        dq_arr = qdlin_arr - ref_cycle[np.newaxis, :]
        
        X_seq_list.append(qdlin_arr)
        X_dq_list.append(dq_arr)
        y_list.append(cl)
        key_list.append(k)
    
    return np.array(X_seq_list), np.array(X_dq_list), np.array(y_list), key_list

# Use fewer voltage points for memory efficiency
X_seq, X_dq, y_seq, seq_keys = prepare_sequence_data(bat_dict, n_early_cycles=100, qdlin_len=200)
print(f"Sequence data: {X_seq.shape[0]} cells, {X_seq.shape[1]} cycles, {X_seq.shape[2]} voltage points")
print(f"Cycle life range: {y_seq.min():.0f} - {y_seq.max():.0f}")

# ============================================================
# PREPARE FEATURE DATA (for MLP comparison)
# ============================================================
feat_keys = [k for k in seq_keys if k in feat_df.index]
feat_sub = feat_df.loc[feat_keys]
X_feat = feat_sub.drop(columns=['cycle_life']).fillna(0).values
y_feat = feat_sub['cycle_life'].values

# ============================================================
# BUILD MODELS
# ============================================================
print("\nImporting TensorFlow...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

tf.random.set_seed(42)
np.random.seed(42)

# Reduce Qdlin to key cycles only for efficiency
# Use cycles: 1,5,10,20,30,40,50,60,70,80,90,100 (12 cycles)
key_cycle_indices = [0, 4, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
X_key = X_seq[:, key_cycle_indices, :]  # (n_cells, 12, 200)
X_dq_key = X_dq[:, key_cycle_indices, :]

# Also create a flattened DeltaQ feature set
# Use DeltaQ at cycle 100 relative to cycle 10 (single curve per cell)
X_dq_100_10 = X_seq[:, 99, :] - X_seq[:, 9, :]  # (n_cells, 200)

# Log transform target (as in Severson)
y_log = np.log(y_seq)

def build_lstm_model(input_shape):
    """LSTM model for sequence prediction."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

def build_cnn_model(input_shape):
    """1D-CNN model for sequence prediction."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, kernel_size=3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

def build_cnn_2d_model(input_shape):
    """2D-CNN treating (cycles, voltage_points) as an image."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Reshape((*input_shape, 1)),  # Add channel dim
        layers.Conv2D(32, (3, 5), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

def build_mlp_model(input_dim):
    """MLP model using engineered features."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model

# ============================================================
# CROSS-VALIDATION EVALUATION
# ============================================================
def evaluate_dl_model(build_fn, X, y_log, y_raw, model_name, epochs=200, batch_size=8):
    """Evaluate a DL model with K-fold CV."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_preds = np.zeros_like(y_raw, dtype=float)
    all_counts = np.zeros_like(y_raw, dtype=float)
    rmse_list, mape_list = [], []
    history_list = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_log[train_idx], y_log[test_idx]
        y_raw_test = y_raw[test_idx]
        
        # Normalize
        if len(X.shape) == 2:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        else:
            # For sequence data, normalize per feature
            mean = X_train.mean(axis=0, keepdims=True)
            std = X_train.std(axis=0, keepdims=True) + 1e-8
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
        
        model = build_fn(X_train.shape[1:] if len(X.shape) > 2 else (X_train.shape[1],))
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss', patience=30, restore_best_weights=True, verbose=0)
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stop],
            verbose=0
        )
        history_list.append(history)
        
        pred_log = model.predict(X_test, verbose=0).flatten()
        pred = np.exp(pred_log)
        
        rmse = np.sqrt(np.mean((y_raw_test - pred)**2))
        mape = np.mean(np.abs(y_raw_test - pred) / y_raw_test) * 100
        
        rmse_list.append(rmse)
        mape_list.append(mape)
        
        all_preds[test_idx] += pred
        all_counts[test_idx] += 1
        
        print(f"  Fold {fold+1}: RMSE={rmse:.0f}, MAPE={mape:.1f}%")
    
    avg_pred = all_preds / np.maximum(all_counts, 1)
    
    print(f"  >> {model_name} Average: RMSE={np.mean(rmse_list):.0f}±{np.std(rmse_list):.0f}, MAPE={np.mean(mape_list):.1f}%±{np.std(mape_list):.1f}%")
    
    return avg_pred, np.mean(rmse_list), np.mean(mape_list), history_list

# --- Run all models ---
results = {}

print("\n" + "="*60)
print("MODEL 1: LSTM on DeltaQ sequences (12 key cycles x 200 points)")
print("="*60)
pred_lstm, rmse_lstm, mape_lstm, hist_lstm = evaluate_dl_model(
    lambda shape: build_lstm_model(shape),
    X_dq_key, y_log, y_seq, "LSTM", epochs=300, batch_size=8)
results['LSTM'] = {'pred': pred_lstm, 'rmse': rmse_lstm, 'mape': mape_lstm}

print("\n" + "="*60)
print("MODEL 2: 1D-CNN on DeltaQ_100-10 curve (single curve per cell)")
print("="*60)
# Reshape for 1D-CNN: (n_cells, 200, 1)
X_dq_cnn = X_dq_100_10[:, :, np.newaxis]
pred_cnn1d, rmse_cnn1d, mape_cnn1d, hist_cnn1d = evaluate_dl_model(
    lambda shape: build_cnn_model(shape),
    X_dq_cnn, y_log, y_seq, "1D-CNN", epochs=300, batch_size=8)
results['1D-CNN'] = {'pred': pred_cnn1d, 'rmse': rmse_cnn1d, 'mape': mape_cnn1d}

print("\n" + "="*60)
print("MODEL 3: 2D-CNN on DeltaQ sequence map (12 cycles x 200 points)")
print("="*60)
pred_cnn2d, rmse_cnn2d, mape_cnn2d, hist_cnn2d = evaluate_dl_model(
    lambda shape: build_cnn_2d_model(shape),
    X_dq_key, y_log, y_seq, "2D-CNN", epochs=300, batch_size=8)
results['2D-CNN'] = {'pred': pred_cnn2d, 'rmse': rmse_cnn2d, 'mape': mape_cnn2d}

print("\n" + "="*60)
print("MODEL 4: MLP on engineered features")
print("="*60)
pred_mlp, rmse_mlp, mape_mlp, hist_mlp = evaluate_dl_model(
    lambda shape: build_mlp_model(shape[0]),
    X_feat, np.log(y_feat), y_feat, "MLP", epochs=300, batch_size=8)
results['MLP'] = {'pred': pred_mlp, 'rmse': rmse_mlp, 'mape': mape_mlp}

# ============================================================
# RESULTS COMPARISON
# ============================================================
print("\n" + "="*60)
print("RESULTS COMPARISON: Baseline vs Deep Learning")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['ElasticNet-Variance (baseline)', 'ElasticNet-Discharge (baseline)', 
              'LSTM', '1D-CNN', '2D-CNN', 'MLP (features)'],
    'MAPE (%)': ['11.0', '8.3', f'{mape_lstm:.1f}', f'{mape_cnn1d:.1f}', 
                 f'{mape_cnn2d:.1f}', f'{mape_mlp:.1f}'],
    'RMSE': ['~130', '~90', f'{rmse_lstm:.0f}', f'{rmse_cnn1d:.0f}',
             f'{rmse_cnn2d:.0f}', f'{rmse_mlp:.0f}'],
    'Input': ['1 feature', '11 features', 'Raw DQ curves', 'Raw DQ curve', 
              'Raw DQ map', 'Engineered features']
})
print(comparison.to_string(index=False))
comparison.to_csv(os.path.join(OUTPUT_DIR, 'dl_comparison.csv'), index=False)

# ============================================================
# VISUALIZATION
# ============================================================
print("\nGenerating plots...")

# Plot 1: Predicted vs Observed for all DL models
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
models_to_plot = [
    ('LSTM', pred_lstm, mape_lstm, y_seq),
    ('1D-CNN', pred_cnn1d, mape_cnn1d, y_seq),
    ('2D-CNN', pred_cnn2d, mape_cnn2d, y_seq),
    ('MLP', pred_mlp, mape_mlp, y_feat),
]

for ax, (name, pred, mape, y_true) in zip(axes.flatten(), models_to_plot):
    ax.scatter(y_true, pred, c='steelblue', edgecolors='k', linewidth=0.5, s=60, alpha=0.7)
    lims = [min(y_true.min(), pred.min()) * 0.9, max(y_true.max(), pred.max()) * 1.1]
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Observed Cycle Life', fontsize=12)
    ax.set_ylabel('Predicted Cycle Life', fontsize=12)
    ax.set_title(f'{name} | MAPE: {mape:.1f}%', fontsize=14)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_dl_predictions.png'), dpi=150)
plt.close()
print("Saved fig8_dl_predictions.png")

# Plot 2: Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))
all_models = ['ElasticNet\n(Variance)', 'ElasticNet\n(Discharge)', 'LSTM', '1D-CNN', '2D-CNN', 'MLP']
all_mapes = [11.0, 8.3, mape_lstm, mape_cnn1d, mape_cnn2d, mape_mlp]
colors = ['#999999', '#999999', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
bars = ax.bar(all_models, all_mapes, color=colors, edgecolor='black', linewidth=0.5)

for bar, mape in zip(bars, all_mapes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{mape:.1f}%', ha='center', fontsize=12, fontweight='bold')

ax.set_ylabel('MAPE (%)', fontsize=14)
ax.set_title('Model Comparison: Mean Absolute Percentage Error', fontsize=16)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(all_mapes) * 1.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_model_comparison.png'), dpi=150)
plt.close()
print("Saved fig9_model_comparison.png")

# Plot 3: Training history for best model
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (name, hist_list) in enumerate([('LSTM', hist_lstm), ('1D-CNN', hist_cnn1d)]):
    ax = axes[i]
    for fold, h in enumerate(hist_list):
        ax.plot(h.history['loss'], alpha=0.5, label=f'Fold {fold+1} train')
        if 'val_loss' in h.history:
            ax.plot(h.history['val_loss'], '--', alpha=0.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(f'{name} Training History', fontsize=14)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_training_history.png'), dpi=150)
plt.close()
print("Saved fig10_training_history.png")

print(f"\n{'='*60}")
print("DONE! Deep learning models complete.")
print("Check output/ folder for all results.")
print(f"{'='*60}")
