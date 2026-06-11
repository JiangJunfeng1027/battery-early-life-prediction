"""
Step 3: Deep Learning Models (PyTorch version)
- LSTM, 1D-CNN, 2D-CNN, MLP
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = './output/'

print("="*60)
print("Step 3: Deep Learning Models (PyTorch)")
print("="*60)

# ============================================================
# LOAD DATA
# ============================================================
print("\nLoading data...")
with open(os.path.join(OUTPUT_DIR, 'all_batteries.pkl'), 'rb') as f:
    bat_dict = pickle.load(f)
feat_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'feature_matrix.csv'), index_col=0)

# ============================================================
# PREPARE SEQUENCE DATA
# ============================================================
print("Preparing sequence data...")

def prepare_data(bat_dict, n_cycles=100, vlen=200):
    X_list, y_list, keys = [], [], []
    for k in bat_dict:
        cl = bat_dict[k]['cycle_life'].flatten()[0]
        if np.isnan(cl) or cl <= 0: continue
        cycles = bat_dict[k]['cycles']
        if len(cycles) < n_cycles + 1: continue
        curves = []
        ok = True
        for c in range(n_cycles):
            if str(c) not in cycles or 'Qdlin' not in cycles[str(c)]:
                ok = False; break
            ql = np.array(cycles[str(c)]['Qdlin']).flatten()
            if len(ql) < 10:
                ok = False; break
            ql_interp = np.interp(np.linspace(0,1,vlen), np.linspace(0,1,len(ql)), ql)
            curves.append(ql_interp)
        if not ok or len(curves) != n_cycles: continue
        X_list.append(np.array(curves))
        y_list.append(cl)
        keys.append(k)
    return np.array(X_list), np.array(y_list), keys

X_seq, y_seq, seq_keys = prepare_data(bat_dict)
print(f"Data: {X_seq.shape[0]} cells, {X_seq.shape[1]} cycles, {X_seq.shape[2]} points")

# DeltaQ relative to cycle 10
X_dq = X_seq - X_seq[:, 9:10, :]

# Key cycles subset
kidx = [0,4,9,19,29,39,49,59,69,79,89,99]
X_dq_key = X_dq[:, kidx, :]  # (N, 12, 200)

# Single DeltaQ curve: cycle100 - cycle10
X_dq_single = X_seq[:,99,:] - X_seq[:,9,:]  # (N, 200)

# Feature data
fkeys = [k for k in seq_keys if k in feat_df.index]
feat_sub = feat_df.loc[fkeys]
X_feat = feat_sub.drop(columns=['cycle_life']).fillna(0).values
y_feat = feat_sub['cycle_life'].values

y_log = np.log(y_seq)

# ============================================================
# MODELS
# ============================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=200):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 64, batch_first=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(32,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU(), nn.Linear(16,1))
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])
        return self.fc(x).squeeze(-1)

class CNN1DModel(nn.Module):
    def __init__(self, seq_len=200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32,1))
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, L)
        x = self.conv(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

class CNN2DModel(nn.Module):
    def __init__(self, n_cycles=12, vlen=200):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, (3,5), padding=(1,2)), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 16, (3,3), padding=(1,1)), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32,1))
    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, V)
        x = self.conv(x).squeeze(-1).squeeze(-1)
        return self.fc(x).squeeze(-1)

class MLPModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

# ============================================================
# TRAINING
# ============================================================
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

def train_and_eval(model_cls, X, y_log, y_raw, name, epochs=300, lr=0.001, bs=8):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros(len(y_raw))
    all_counts = np.zeros(len(y_raw))
    rmses, mapes = [], []
    all_histories = []

    for fold, (tr, te) in enumerate(kf.split(X)):
        X_tr, X_te = X[tr].copy(), X[te].copy()
        y_tr, y_te = y_log[tr], y_log[te]
        y_raw_te = y_raw[te]

        # Normalize
        if X.ndim == 2:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)
        else:
            m = X_tr.mean(axis=0, keepdims=True)
            s = X_tr.std(axis=0, keepdims=True) + 1e-8
            X_tr = (X_tr - m) / s
            X_te = (X_te - m) / s

        Xt = torch.FloatTensor(X_tr).to(device)
        yt = torch.FloatTensor(y_tr).to(device)
        Xv = torch.FloatTensor(X_te).to(device)

        # Split train into train/val for early stopping
        n_val = max(2, len(Xt) // 5)
        Xt_train, Xt_val = Xt[n_val:], Xt[:n_val]
        yt_train, yt_val = yt[n_val:], yt[:n_val]

        if X.ndim == 2:
            model = model_cls(X.shape[1]).to(device)
        else:
            model = model_cls().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_val = float('inf')
        patience, wait = 30, 0
        best_state = None
        train_losses, val_losses = [], []

        for ep in range(epochs):
            model.train()
            idx = torch.randperm(len(Xt_train))
            ep_loss = 0
            for i in range(0, len(idx), bs):
                batch = idx[i:i+bs]
                pred = model(Xt_train[batch])
                loss = loss_fn(pred, yt_train[batch])
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            train_losses.append(ep_loss / max(1, len(idx)//bs))

            model.eval()
            with torch.no_grad():
                vl = loss_fn(model(Xt_val), yt_val).item()
            val_losses.append(vl)

            if vl < best_val:
                best_val = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        
        all_histories.append((train_losses, val_losses))

        model.eval()
        with torch.no_grad():
            pred = np.exp(model(Xv).cpu().numpy())

        rmse = np.sqrt(np.mean((y_raw_te - pred)**2))
        mape = np.mean(np.abs(y_raw_te - pred) / y_raw_te) * 100
        rmses.append(rmse); mapes.append(mape)
        all_preds[te] += pred
        all_counts[te] += 1
        print(f"  Fold {fold+1}: RMSE={rmse:.0f}, MAPE={mape:.1f}%")

    avg_pred = all_preds / np.maximum(all_counts, 1)
    print(f"  >> {name}: RMSE={np.mean(rmses):.0f}±{np.std(rmses):.0f}, MAPE={np.mean(mapes):.1f}%±{np.std(mapes):.1f}%")
    return avg_pred, np.mean(rmses), np.mean(mapes), all_histories

results = {}

print("\n" + "="*60)
print("MODEL 1: LSTM")
print("="*60)
p1, r1, m1, h1 = train_and_eval(lambda *a: LSTMModel(), X_dq_key, y_log, y_seq, "LSTM")
results['LSTM'] = (p1, r1, m1)

print("\n" + "="*60)
print("MODEL 2: 1D-CNN")
print("="*60)
p2, r2, m2, h2 = train_and_eval(lambda *a: CNN1DModel(), X_dq_single, y_log, y_seq, "1D-CNN")
results['1D-CNN'] = (p2, r2, m2)

print("\n" + "="*60)
print("MODEL 3: 2D-CNN")
print("="*60)
p3, r3, m3, h3 = train_and_eval(lambda *a: CNN2DModel(), X_dq_key, y_log, y_seq, "2D-CNN")
results['2D-CNN'] = (p3, r3, m3)

print("\n" + "="*60)
print("MODEL 4: MLP")
print("="*60)
p4, r4, m4, h4 = train_and_eval(MLPModel, X_feat, np.log(y_feat), y_feat, "MLP")
results['MLP'] = (p4, r4, m4)

# ============================================================
# RESULTS
# ============================================================
print("\n" + "="*60)
print("FULL COMPARISON")
print("="*60)
comp = pd.DataFrame({
    'Model': ['ElasticNet-Var', 'ElasticNet-Discharge', 'LSTM', '1D-CNN', '2D-CNN', 'MLP'],
    'MAPE%': ['11.0', '8.3', f'{m1:.1f}', f'{m2:.1f}', f'{m3:.1f}', f'{m4:.1f}'],
    'RMSE': ['~130', '~90', f'{r1:.0f}', f'{r2:.0f}', f'{r3:.0f}', f'{r4:.0f}']
})
print(comp.to_string(index=False))
comp.to_csv(os.path.join(OUTPUT_DIR, 'dl_comparison.csv'), index=False)

# ============================================================
# PLOTS
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, (name, (pred, _, mape)), yt in zip(axes.flatten(),
    [('LSTM', results['LSTM']), ('1D-CNN', results['1D-CNN']),
     ('2D-CNN', results['2D-CNN']), ('MLP', results['MLP'])],
    [y_seq, y_seq, y_seq, y_feat]):
    ax.scatter(yt, pred, c='steelblue', edgecolors='k', s=60, alpha=0.7)
    lims = [min(yt.min(), pred.min())*0.9, max(yt.max(), pred.max())*1.1]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Observed', fontsize=12); ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(f'{name} | MAPE: {mape:.1f}%', fontsize=14)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_dl_predictions.png'), dpi=150)
plt.close()
print("\nSaved fig8_dl_predictions.png")

fig, ax = plt.subplots(figsize=(10, 6))
models = ['ElasticNet\n(Var)', 'ElasticNet\n(Discharge)', 'LSTM', '1D-CNN', '2D-CNN', 'MLP']
mapes = [11.0, 8.3, m1, m2, m3, m4]
colors = ['#999','#999','#2196F3','#4CAF50','#FF9800','#9C27B0']
bars = ax.bar(models, mapes, color=colors, edgecolor='k', lw=0.5)
for b, m in zip(bars, mapes):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f'{m:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('MAPE (%)', fontsize=14)
ax.set_title('Model Comparison', fontsize=16)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(mapes)*1.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_model_comparison.png'), dpi=150)
plt.close()
print("Saved fig9_model_comparison.png")

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, hist) in zip(axes, [('LSTM', h1), ('1D-CNN', h2)]):
    for fold, (tl, vl) in enumerate(hist):
        ax.plot(tl, alpha=0.5, label=f'Fold {fold+1} train')
        ax.plot(vl, '--', alpha=0.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'{name} Training'); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_training_history.png'), dpi=150)
plt.close()
print("Saved fig10_training_history.png")

print(f"\n{'='*60}")
print("DONE!")
print(f"{'='*60}")
