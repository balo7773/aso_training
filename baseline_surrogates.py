#!/usr/bin/env python3
"""
baseline_surrogates.py
======================
Trains XGBoost and Random Forest baselines on the exact same train/test split
as your MLP. Evaluates all three models (XGB, RF, MLP) on the same test set.

Generates:
    baseline_results/baseline_error_analysis.png  <- comparable to MLP error_analysis.png
    baseline_results/baseline_results.txt         <- metrics table for paper Table 3

Run:
    python baseline_surrogates.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_CSV    = os.path.join(BASE_DIR, 'master_dataset_clean.csv')
SCALER_JSON = os.path.join(BASE_DIR, 'scaler_params.json')
MODEL_PATH  = os.path.join(BASE_DIR, 'model', 'best_model.pt')
OUT_DIR     = os.path.join(BASE_DIR, 'baseline_results')
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MLP ARCHITECTURE — exact copy from train_v2.py / b2.py
# ══════════════════════════════════════════════════════════════════════════════

class AirfoilMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, activation, dropout_rate):
        super().__init__()
        act_map = {
            'relu': nn.ReLU, 'gelu': nn.GELU,
            'tanh': nn.Tanh, 'silu': nn.SiLU,
        }
        act_cls = act_map.get(activation, nn.GELU)

        trunk_layers = []
        in_dim = input_dim
        for h_dim in hidden_layers[:-1]:
            trunk_layers += [
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                act_cls(),
                nn.Dropout(dropout_rate),
            ]
            in_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)

        head_dim = hidden_layers[-1]

        self.cl_head = nn.Sequential(
            nn.Linear(in_dim, head_dim), act_cls(), nn.Linear(head_dim, 1))
        self.cm_head = nn.Sequential(
            nn.Linear(in_dim, head_dim), act_cls(), nn.Linear(head_dim, 1))

        cd_head_input_dim = in_dim + 2
        self.cd_head = nn.Sequential(
            nn.Linear(cd_head_input_dim, head_dim), act_cls(), nn.Linear(head_dim, 1))

        self._re_idx    = input_dim - 2
        self._alpha_idx = input_dim - 1

    def forward(self, x):
        f = self.trunk(x)
        re_alpha = x[:, self._re_idx:self._alpha_idx + 1]
        cd_input = torch.cat([f, re_alpha], dim=1)
        return torch.cat([self.cl_head(f),
                          self.cd_head(cd_input),
                          self.cm_head(f)], dim=1)


# ══════════════════════════════════════════════════════════════════════════════
#  NORMALIZATION — exact copy from train_v2.py
# ══════════════════════════════════════════════════════════════════════════════

def load_scalers(path):
    with open(path, 'r') as f:
        return json.load(f)


def normalize_inputs(df, scalers, cst_cols):
    n_cst = len(cst_cols)
    X = np.zeros((len(df), n_cst + 2), dtype=np.float32)
    for i, col in enumerate(cst_cols):
        s = scalers[col]
        X[:, i] = (df[col].values - s['mean']) / (s['std'] + 1e-10)
    s = scalers['re']
    X[:, n_cst] = (df['re'].values - s['mean']) / (s['std'] + 1e-10)
    s = scalers['alpha']
    X[:, n_cst + 1] = (df['alpha'].values - s['mean']) / (s['std'] + 1e-10)
    return X


def normalize_outputs(df, scalers):
    Y = np.zeros((len(df), 3), dtype=np.float32)
    s = scalers['cl']
    Y[:, 0] = (df['cl'].values - s['mean']) / (s['std'] + 1e-10)
    s = scalers['cd']
    log_cd = np.log(df['cd'].clip(1e-6).values)
    Y[:, 1] = (log_cd - s['log_mean']) / (s['log_std'] + 1e-10)
    s = scalers['cm']
    Y[:, 2] = (df['cm'].values - s['mean']) / (s['std'] + 1e-10)
    return Y


def inverse_transform_outputs(Y_norm, scalers):
    Y_phys = np.zeros_like(Y_norm)
    s = scalers['cl']
    Y_phys[:, 0] = Y_norm[:, 0] * s['std'] + s['mean']
    s = scalers['cd']
    Y_phys[:, 1] = np.exp(Y_norm[:, 1] * s['log_std'] + s['log_mean'])
    s = scalers['cm']
    Y_phys[:, 2] = Y_norm[:, 2] * s['std'] + s['mean']
    return Y_phys


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def r2_score(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1 - ss_res / (ss_tot + 1e-10)


def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print('\n' + '#'*65)
    print('  BASELINE SURROGATES — XGBoost + Random Forest')
    print('  Same train/test split as MLP. Same metrics. Same plots.')
    print('#'*65)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f'\n  Loading {DATA_CSV}...')
    df = pd.read_csv(DATA_CSV)
    scalers = load_scalers(SCALER_JSON)
    cst_cols = [c for c in df.columns if c.startswith('cst_')]

    train_df = df[df['split'] == 'train'].copy()
    test_df  = df[df['split'] == 'test'].copy()
    print(f'  Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}')

    # ── Prepare features and targets ─────────────────────────────────────────
    X_train = normalize_inputs(train_df, scalers, cst_cols)
    Y_train = normalize_outputs(train_df, scalers)
    X_test  = normalize_inputs(test_df,  scalers, cst_cols)
    Y_test  = normalize_outputs(test_df,  scalers)

    # ── Train XGBoost ──────────────────────────────────────────────────────
    print('\n  Training XGBoost...')
    t0 = time.time()
    xgb_model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            tree_method='hist',   # change to 'gpu_hist' if xgboost has CUDA support
            n_jobs=4,
            random_state=42,
            objective='reg:squarederror'
        )
    )
    xgb_model.fit(X_train, Y_train)
    xgb_train_time = time.time() - t0
    print(f'  Done in {xgb_train_time:.1f} s')

    # ── Train Random Forest ────────────────────────────────────────────────
    print('\n  Training Random Forest...')
    t0 = time.time()
    rf_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            n_jobs=4,
            random_state=42
        )
    )
    rf_model.fit(X_train, Y_train)
    rf_train_time = time.time() - t0
    print(f'  Done in {rf_train_time:.1f} s')

    # ── Load MLP ─────────────────────────────────────────────────────────────
    print('\n  Loading MLP...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    cfg = ckpt['cfg']

    mlp = AirfoilMLP(
        input_dim=22,
        hidden_layers=cfg['hidden_layers'],
        activation=cfg['activation'],
        dropout_rate=0.0
    ).to(device)
    mlp.load_state_dict(ckpt['model_state'])
    mlp.eval()
    print(f'  Loaded epoch {ckpt["epoch"]} on {device}')

    # ── Predictions ──────────────────────────────────────────────────────────
    print('\n  Evaluating on test set...')

    # XGBoost
    t0 = time.time()
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_infer_total = time.time() - t0
    xgb_infer_per = xgb_infer_total / len(X_test) * 1e6

    # Random Forest
    t0 = time.time()
    y_pred_rf = rf_model.predict(X_test)
    rf_infer_total = time.time() - t0
    rf_infer_per = rf_infer_total / len(X_test) * 1e6

    # MLP
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    test_loader = DataLoader(TensorDataset(X_test_t), batch_size=512, shuffle=False)
    t0 = time.time()
    mlp_preds = []
    with torch.no_grad():
        for (xb,) in test_loader:
            mlp_preds.append(mlp(xb).cpu().numpy())
    mlp_infer_total = time.time() - t0
    mlp_infer_per = mlp_infer_total / len(X_test) * 1e6
    y_pred_mlp = np.concatenate(mlp_preds, axis=0)

    # ── Inverse transform to physical units ────────────────────────────────
    y_true_phys = inverse_transform_outputs(Y_test, scalers)
    xgb_phys    = inverse_transform_outputs(y_pred_xgb, scalers)
    rf_phys     = inverse_transform_outputs(y_pred_rf, scalers)
    mlp_phys    = inverse_transform_outputs(y_pred_mlp, scalers)

    # ── Compute metrics ──────────────────────────────────────────────────────
    names = ['CL', 'CD', 'CM']
    xgb_r2 = r2_score(Y_test, y_pred_xgb)
    rf_r2  = r2_score(Y_test, y_pred_rf)
    mlp_r2 = r2_score(Y_test, y_pred_mlp)

    xgb_mae = mae_score(y_true_phys, xgb_phys)
    rf_mae  = mae_score(y_true_phys, rf_phys)
    mlp_mae = mae_score(y_true_phys, mlp_phys)

    # ── Print results ──────────────────────────────────────────────────────
    print('\n' + '='*65)
    print('  RESULTS — Normalized R² (same space as MLP training logs)')
    print('='*65)
    print(f'  {"":>12} | {"CL":>8} | {"CD":>8} | {"CM":>8}')
    print(f'  {"-"*12}-+-{"-"*8}-+-{"-"*8}-+-{"-"*8}')
    print(f'  {"XGBoost":>12} | {xgb_r2[0]:>8.4f} | {xgb_r2[1]:>8.4f} | {xgb_r2[2]:>8.4f}')
    print(f'  {"RandomForest":>12} | {rf_r2[0]:>8.4f} | {rf_r2[1]:>8.4f} | {rf_r2[2]:>8.4f}')
    print(f'  {"MLP (Ours)":>12} | {mlp_r2[0]:>8.4f} | {mlp_r2[1]:>8.4f} | {mlp_r2[2]:>8.4f}')

    print('\n' + '='*65)
    print('  RESULTS — Physical MAE')
    print('='*65)
    print(f'  {"":>12} | {"CL":>10} | {"CD":>12} | {"CM":>10}')
    print(f'  {"-"*12}-+-{"-"*10}-+-{"-"*12}-+-{"-"*10}')
    print(f'  {"XGBoost":>12} | {xgb_mae[0]:>10.5f} | {xgb_mae[1]:>12.6f} | {xgb_mae[2]:>10.5f}')
    print(f'  {"RandomForest":>12} | {rf_mae[0]:>10.5f} | {rf_mae[1]:>12.6f} | {rf_mae[2]:>10.5f}')
    print(f'  {"MLP (Ours)":>12} | {mlp_mae[0]:>10.5f} | {mlp_mae[1]:>12.6f} | {mlp_mae[2]:>10.5f}')

    print('\n' + '='*65)
    print('  RESULTS — Inference Time')
    print('='*65)
    print(f'  XGBoost      : {xgb_infer_per:.1f} μs/sample  (total {xgb_infer_total:.3f} s)')
    print(f'  RandomForest : {rf_infer_per:.1f} μs/sample  (total {rf_infer_total:.3f} s)')
    print(f'  MLP (Ours)   : {mlp_infer_per:.1f} μs/sample  (total {mlp_infer_total:.3f} s)')

    # ── Save text report ─────────────────────────────────────────────────────
    report_path = os.path.join(OUT_DIR, 'baseline_results.txt')
    with open(report_path, 'w') as f:
        f.write('BASELINE SURROGATE COMPARISON\n')
        f.write('='*65 + '\n\n')
        f.write('Table 3: Surrogate Baseline Comparison (for paper)\n')
        f.write('-'*65 + '\n')
        f.write(f'{"Model":<18} | {"Test R² (CL/CD/CM)":<22} | {"MAE (CL/CD/CM)":<28} | {"Inference μs/sample":<18}\n')
        f.write('-'*65 + '\n')
        f.write(f'{"Random Forest":<18} | {rf_r2[0]:.4f}/{rf_r2[1]:.4f}/{rf_r2[2]:.4f} | {rf_mae[0]:.5f}/{rf_mae[1]:.6f}/{rf_mae[2]:.5f} | {rf_infer_per:.1f}\n')
        f.write(f'{"XGBoost":<18} | {xgb_r2[0]:.4f}/{xgb_r2[1]:.4f}/{xgb_r2[2]:.4f} | {xgb_mae[0]:.5f}/{xgb_mae[1]:.6f}/{xgb_mae[2]:.5f} | {xgb_infer_per:.1f}\n')
        f.write(f'{"MLP (Ours)":<18} | {mlp_r2[0]:.4f}/{mlp_r2[1]:.4f}/{mlp_r2[2]:.4f} | {mlp_mae[0]:.5f}/{mlp_mae[1]:.6f}/{mlp_mae[2]:.5f} | {mlp_infer_per:.1f}\n')
        f.write('-'*65 + '\n\n')
        f.write('Training Time:\n')
        f.write(f'  XGBoost      : {xgb_train_time:.1f} s\n')
        f.write(f'  RandomForest : {rf_train_time:.1f} s\n')
        f.write(f'  MLP (Ours)   : 6 hrs (from training logs)\n')
    print(f'\n  Report saved -> {report_path}')

    # ── Plot error analysis ────────────────────────────────────────────────
    print('\n  Generating error analysis plot...')
    fig, axes = plt.subplots(3, 3, figsize=(15, 14))
    fig.suptitle('Baseline Comparison — Prediction vs Actual (Test Set)',
                 fontsize=13, fontweight='bold')

    models = [('XGBoost', xgb_phys), ('Random Forest', rf_phys), ('MLP (Ours)', mlp_phys)]
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    for row, (model_name, y_pred) in enumerate(models):
        for col, name in enumerate(names):
            ax = axes[row, col]
            t = y_true_phys[:, col]
            p = y_pred[:, col]

            r2 = 1 - np.sum((t - p)**2) / (np.sum((t - t.mean())**2) + 1e-10)
            mae = np.mean(np.abs(t - p))

            ax.scatter(t, p, s=1.5, alpha=0.4, color=colors[row])
            lim = [min(t.min(), p.min()), max(t.max(), p.max())]
            ax.plot(lim, lim, 'k--', lw=1)
            ax.set_xlabel(f'Actual {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{model_name} — {name}\nR²={r2:.4f}  MAE={mae:.5f}', fontsize=10)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'baseline_error_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved -> {plot_path}')

    print('\n' + '#'*65)
    print('  DONE.')
    print('  Copy the table above into your paper (Table 3).')
    print('  Compare baseline_error_analysis.png to your MLP error_analysis.png.')
    print('#'*65 + '\n')


if __name__ == '__main__':
    main()