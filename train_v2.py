#!/usr/bin/python3
"""
Airfoil Surrogate Model — PyTorch MLP Training
===============================================
Trains a Multi-Layer Perceptron to predict CL, CD, CM
from CST geometry parameters + Re + alpha + transition locations.

Run from your conda environment in WSL:
    python train_mlp.py

Outputs saved to AIRFOIL_FILES/model/:
    best_model.pt          <- best weights (lowest val loss)
    final_model.pt         <- weights at end of training
    training_curves.png    <- loss curves + R2 per epoch (updated at every new best)
    error_analysis.png     <- prediction vs actual plots (updated at every new best)
    training_log.csv       <- full epoch-by-epoch metrics
    hyperparameter_guide.txt <- what to change based on what you see

CHANGES vs previous version:
    - AsymmetricLogCDLoss integrated (penalty_factor=4.0 for optimistic CD)
    - xtr_top, xtr_bot removed as input features (24 -> 22 inputs)
    - warm_start=False, epochs=300, lr=1e-3 for fresh training on rebuilt dataset
    - batch_size=256 for faster CPU training on 93k rows
    - CST gate raised from 1.5 -> 2.5 (FX/NLF airfoils have legitimate large CST)
    - Plots saved at every new best model AND at final epoch
    - Asymmetric loss applied ONLY during training; val/test use symmetric MSE

# ==============================================================================
# ADJUSTMENT 1: Re and alpha direct routing to CD head
# ------------------------------------------------------------------------------
# Problem diagnosed:
#   Re and alpha enter the shared trunk at the input layer and pass through
#   3x256 transformations before reaching the CD head. By that point their
#   signal is distributed across a 256-dim mixed representation alongside
#   all 20 CST geometry features. The CD head cannot sharply condition drag
#   predictions on flight regime from this blurred signal.
#
# Why CD specifically needs this fix:
#   CL is smooth and monotonic w.r.t Re/alpha -- trunk encoding is sufficient.
#   CD is highly nonlinear: drag bucket exists only in narrow alpha range,
#   transition location jumps discontinuously, laminar separation bubbles
#   appear and disappear. CD needs Re and alpha sharply and directly.
#
# What changed in the architecture:
#   - Trunk: UNCHANGED (22 inputs -> 256 -> 256 -> 256, output = 256-dim)
#   - CL head: UNCHANGED (256 -> 128 -> 1)
#   - CM head: UNCHANGED (256 -> 128 -> 1)
#   - CD head: receives [trunk_output(256) || re_norm(1) || alpha_norm(1)]
#              = 257-dim input -> 128 -> 1
#   - forward() extracts Re (index 20) and alpha (index 21) from input x
#     and concatenates directly to trunk output before the CD head only
#
# Baseline to beat (v1, best epoch 49):
#   CL  R2=0.9776  MAE=0.05077  residual_std=0.07982
#   CD  R2=0.9519  MAE=0.00257  residual_std=0.00597
#   CM  R2=0.9144  MAE=0.00971  residual_std=0.01344
#
# Success criteria:
#   CD R2  > 0.9519
#   CD MAE < 0.00257
#   CD residual std < 0.00597
#   No regression on CL or CM R2
# ==============================================================================
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
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
#  ASYMMETRIC CD LOSS
# ==============================================================================

class AsymmetricLogCDLoss(nn.Module):
    """
    Penalises optimistic CD predictions more than conservative ones.

    Why this matters:
    The Bayesian optimizer searches for minimum CD. If the MLP predicts
    CD = 0.003 when reality is 0.010 (optimistic), the optimizer exploits
    that blind spot and recommends an airfoil that fails in XFOIL.
    If the MLP predicts CD = 0.012 when reality is 0.010 (conservative),
    the optimizer skips a good airfoil -- a missed opportunity, not a failure.

    So we penalise under-prediction (pred < target) by penalty_factor,
    and leave over-prediction at normal weight.

    Applied in normalized log(CD) space -- same space the MLP trains in.
    Only active during training. Val/test use symmetric MSE for comparability.
    """
    def __init__(self, penalty_factor=4.0):
        super().__init__()
        self.penalty_factor = penalty_factor

    def forward(self, pred_cd, target_cd):
        """
        pred_cd, target_cd: (batch,) tensors in normalized log(CD) space.
        Lower normalized value = lower real CD.
        pred < target means pred is more negative in log space = predicting
        lower CD than reality = optimistic. That gets penalised.
        """
        errors_sq = (pred_cd - target_cd) ** 2
        optimistic_mask = (pred_cd < target_cd).float()
        weights = 1.0 + (self.penalty_factor - 1.0) * optimistic_mask
        return (errors_sq * weights).mean()


# ==============================================================================
#  HYPERPARAMETERS
# ==============================================================================

CFG = {
    # -- Architecture --
    'hidden_layers'   : [512,512,512,256],
    'activation'      : 'gelu',
    'dropout_rate'    : 0.05,

    # -- Training --
    # FRESH TRAINING on rebuilt 93k-row dataset with asymmetric CD loss.
    # warm_start=False -- old weights shaped by symmetric loss on sparse data.
    # lr=1e-3 -- standard Adam starting point for fresh training.
    # 300 epochs with early_stop=25 -- typically converges in 120-200 epochs.
    'epochs'          : 350,
    'batch_size'      : 256,
    'learning_rate'   : 5e-4,
    'weight_decay'    : 1e-4,

    # -- Loss weights [CL, CD, CM] --
    # CD raised to 2.5 -- combined with asymmetric loss, this strongly
    # prioritises accurate drag prediction.
    'loss_weights'    : [1.0, 3.0, 0.3],

    # -- Asymmetric CD loss --
    # 4.0 = optimistic CD predictions penalised 4x more than conservative.
    # Start here. If optimizer can't find anything good -> reduce to 3.0.
    # If CD still underpredicted -> increase to 5.0.
    'cd_penalty_factor' : 4.0,

    # -- Tier 2 sample weight --
    'tier2_weight'    : 0.7,

    # -- Learning rate scheduler --
    'lr_patience'     : 8,
    'lr_factor'       : 0.5,
    'lr_min'          : 1e-6,

    # -- Early stopping --
    'early_stop_patience' : 30,
    'early_stop_smooth'   : 5,

    # -- Warm start --
    # False = train from scratch (new data + new loss function)
    'warm_start'      : False,

    # -- Reproducibility --
    'seed'            : 42,
}

# -- Paths --
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_CSV    = os.path.join(BASE_DIR, 'master_dataset_clean.csv')
SCALER_JSON = os.path.join(BASE_DIR, 'scaler_params.json')
MODEL_DIR   = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================================================================
#  SCALER COMPUTATION
# ==============================================================================

def compute_and_save_scalers(df, cst_cols, save_path):
    """Compute normalization parameters from the TRAINING ROWS ONLY."""
    train_df = df[df['split'] == 'train'].copy()
    print(f'  Computing scalers from {len(train_df):,} training rows '
          f'({train_df["airfoil"].nunique()} airfoils)...')

    scalers = {}

    for col in cst_cols:
        scalers[col] = {
            'mean': float(train_df[col].mean()),
            'std' : float(train_df[col].std())
        }

    scalers['re'] = {
        'mean': float(train_df['re'].mean()),
        'std' : float(train_df['re'].std())
    }

    scalers['alpha'] = {
        'mean': float(train_df['alpha'].mean()),
        'std' : float(train_df['alpha'].std())
    }

    for col in ['xtr_top', 'xtr_bot']:
        scalers[col] = {
            'mean': float(train_df[col].mean()),
            'std' : float(train_df[col].std())
        }

    for col in ['cl', 'cm']:
        scalers[col] = {
            'mean': float(train_df[col].mean()),
            'std' : float(train_df[col].std())
        }

    log_cd = np.log(train_df['cd'].clip(1e-6))
    scalers['cd'] = {
        'mean'    : float(train_df['cd'].mean()),
        'std'     : float(train_df['cd'].std()),
        'log_mean': float(log_cd.mean()),
        'log_std' : float(log_cd.std())
    }

    with open(save_path, 'w') as f:
        json.dump(scalers, f, indent=2)

    print(f'  Scalers saved -> {save_path}')
    print(f'    Re      : mean={scalers["re"]["mean"]:.0f}  std={scalers["re"]["std"]:.0f}')
    print(f'    alpha   : mean={scalers["alpha"]["mean"]:.2f} deg  std={scalers["alpha"]["std"]:.2f} deg')
    print(f'    xtr_top : mean={scalers["xtr_top"]["mean"]:.4f}  std={scalers["xtr_top"]["std"]:.4f}')
    print(f'    xtr_bot : mean={scalers["xtr_bot"]["mean"]:.4f}  std={scalers["xtr_bot"]["std"]:.4f}')
    print(f'    CL      : mean={scalers["cl"]["mean"]:.4f}  std={scalers["cl"]["std"]:.4f}')
    print(f'    CD      : log_mean={scalers["cd"]["log_mean"]:.4f}  log_std={scalers["cd"]["log_std"]:.4f}')
    print(f'    CM      : mean={scalers["cm"]["mean"]:.4f}  std={scalers["cm"]["std"]:.4f}')

    return scalers


def load_scalers(path):
    with open(path, 'r') as f:
        return json.load(f)


# ==============================================================================
#  NORMALIZATION
# ==============================================================================

def normalize_inputs(df, scalers, cst_cols):
    """
    Input order: [cst_u0..u9, cst_l0..l9, re, alpha]
    = 20 CST + 2 operating = 22 inputs
    """
    n_cst = len(cst_cols)
    n_features = n_cst + 2  # <--- Changed from + 4 to + 2
    X = np.zeros((len(df), n_features), dtype=np.float32)

    for i, col in enumerate(cst_cols):
        s = scalers[col]
        X[:, i] = (df[col].values - s['mean']) / (s['std'] + 1e-10)

    s = scalers['re']
    X[:, n_cst] = (df['re'].values - s['mean']) / (s['std'] + 1e-10)
    
    s = scalers['alpha']
    X[:, n_cst + 1] = (df['alpha'].values - s['mean']) / (s['std'] + 1e-10)

    return X


def normalize_outputs(df, scalers):
    """CL, CM -> StandardScaler. CD -> log(CD) then StandardScaler."""
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
    """Reverse normalization to get physical CL, CD, CM values."""
    Y_phys = np.zeros_like(Y_norm)
    s = scalers['cl']
    Y_phys[:, 0] = Y_norm[:, 0] * s['std'] + s['mean']
    s = scalers['cd']
    Y_phys[:, 1] = np.exp(Y_norm[:, 1] * s['log_std'] + s['log_mean'])
    s = scalers['cm']
    Y_phys[:, 2] = Y_norm[:, 2] * s['std'] + s['mean']
    return Y_phys


# ==============================================================================
#  PYTORCH DATASET
# ==============================================================================

class AirfoilDataset(Dataset):
    def __init__(self, X, Y, sample_weights=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        if sample_weights is not None:
            self.W = torch.tensor(sample_weights, dtype=torch.float32)
        else:
            self.W = torch.ones(len(X), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.W[idx]


# ==============================================================================
#  MLP ARCHITECTURE
# ==============================================================================

class AirfoilMLP(nn.Module):
    """
    Shared trunk -> separate heads for CL, CD, CM.

    Input (22) -> Shared trunk -> [CL head | CM head]
                               -> [CD head]  <- ADJUSTMENT 1: also receives Re + alpha directly

    Input feature order (from normalize_inputs):
        indices 0-19  : CST coefficients (20 features)
        index   20    : re_normalized       <- routed directly to CD head
        index   21    : alpha_normalized    <- routed directly to CD head
    """
    def __init__(self, input_dim, hidden_layers, activation, dropout_rate):
        super().__init__()

        act_map = {
            'relu': nn.ReLU(), 'gelu': nn.GELU(),
            'tanh': nn.Tanh(), 'silu': nn.SiLU(),
        }
        act_fn = act_map.get(activation, nn.GELU())

        trunk_layers = []
        in_dim = input_dim
        for h_dim in hidden_layers[:-1]:
            trunk_layers.append(nn.Linear(in_dim, h_dim))
            trunk_layers.append(nn.LayerNorm(h_dim))
            trunk_layers.append(act_fn.__class__())
            if dropout_rate > 0:
                trunk_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        self.trunk = nn.Sequential(*trunk_layers)

        head_dim = hidden_layers[-1]

        # CL and CM heads: unchanged, receive trunk output (256-dim)
        self.cl_head = nn.Sequential(
            nn.Linear(in_dim, head_dim), act_fn.__class__(), nn.Linear(head_dim, 1))
        self.cm_head = nn.Sequential(
            nn.Linear(in_dim, head_dim), act_fn.__class__(), nn.Linear(head_dim, 1))

        # ADJUSTMENT 1: CD head receives trunk output (256) + Re (1) + alpha (1) = 258-dim
        # Re and alpha bypass the trunk and arrive fresh at the CD head,
        # giving it a sharp, unblurred flight-condition signal for drag prediction.
        cd_head_input_dim = in_dim + 2  # 256 + re + alpha = 258
        self.cd_head = nn.Sequential(
            nn.Linear(cd_head_input_dim, head_dim), act_fn.__class__(), nn.Linear(head_dim, 1))

        # Store indices for Re and alpha in the input vector
        self._re_idx    = input_dim - 2  # index 20
        self._alpha_idx = input_dim - 1  # index 21

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.trunk(x)

        # ADJUSTMENT 1: extract Re and alpha from raw input, concatenate to trunk output
        re_alpha = x[:, self._re_idx:self._alpha_idx + 1]  # shape (batch, 2)
        cd_input = torch.cat([features, re_alpha], dim=1)  # shape (batch, 258)

        return torch.cat([self.cl_head(features),
                          self.cd_head(cd_input),
                          self.cm_head(features)], dim=1)


# ==============================================================================
#  TRAINING UTILITIES
# ==============================================================================

def weighted_mse_loss(pred, target, sample_weights, loss_weights):
    """Standard symmetric MSE -- used for val/test evaluation."""
    output_w        = torch.tensor(loss_weights, dtype=torch.float32, device=pred.device)
    per_output_mse  = ((pred - target) ** 2) * output_w.unsqueeze(0)
    per_sample_loss = per_output_mse.mean(dim=1)
    weighted_loss   = per_sample_loss * sample_weights
    return weighted_loss.mean()


def asymmetric_training_loss(pred, target, sample_weights, loss_weights, cd_loss_fn):
    """
    Training loss: symmetric MSE for CL and CM, asymmetric for CD.
    """
    w_cl, w_cd, w_cm = loss_weights

    # CL -- standard MSE
    cl_mse = ((pred[:, 0] - target[:, 0]) ** 2) * w_cl
    # CD -- asymmetric loss (penalises underprediction)
    cd_loss = cd_loss_fn(pred[:, 1], target[:, 1]) * w_cd
    # CM -- standard MSE
    cm_mse = ((pred[:, 2] - target[:, 2]) ** 2) * w_cm

    per_sample = (cl_mse + cm_mse) / 2.0
    per_sample_weighted = per_sample * sample_weights
    return per_sample_weighted.mean() + cd_loss


def r_squared(pred, target):
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
    return (1 - ss_res / (ss_tot + 1e-10)).detach().cpu().numpy()


def run_epoch(model, loader, optimizer, loss_weights, device,
              training=True, cd_loss_fn=None):
    """
    Run one epoch. If training=True AND cd_loss_fn is provided,
    uses asymmetric CD loss. Otherwise uses symmetric MSE.
    """
    model.train(training)
    total_loss, all_pred, all_target = 0.0, [], []

    with torch.set_grad_enabled(training):
        for X_batch, Y_batch, W_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            W_batch = W_batch.to(device)

            pred = model(X_batch)

            if training and cd_loss_fn is not None:
                loss = asymmetric_training_loss(
                    pred, Y_batch, W_batch, loss_weights, cd_loss_fn)
            else:
                loss = weighted_mse_loss(pred, Y_batch, W_batch, loss_weights)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss  += loss.item() * len(X_batch)
            all_pred.append(pred.detach().cpu())
            all_target.append(Y_batch.detach().cpu())

    avg_loss   = total_loss / len(loader.dataset)
    all_pred   = torch.cat(all_pred,   dim=0)
    all_target = torch.cat(all_target, dim=0)
    return avg_loss, r_squared(all_pred, all_target)


# ==============================================================================
#  DIAGNOSTIC PRINTERS
# ==============================================================================

def print_epoch_header():
    print(f"\n{'---'*32}")
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>10}  "
          f"{'R2_CL':>7}  {'R2_CD':>7}  {'R2_CM':>7}  "
          f"{'LR':>9}  {'Time':>6}  {'Note'}")
    print(f"{'---'*32}")


def print_epoch(epoch, train_loss, val_loss, val_r2, lr, elapsed,
                best_val, patience_counter, early_patience):
    note = ''
    if val_loss <= best_val:
        note = 'BEST'
    elif patience_counter >= early_patience // 2:
        note = f'no improve {patience_counter}ep'
    print(f"  {epoch:>5}  {train_loss:>11.6f}  {val_loss:>10.6f}  "
          f"{val_r2[0]:>7.4f}  {val_r2[1]:>7.4f}  {val_r2[2]:>7.4f}  "
          f"{lr:>9.2e}  {elapsed:>5.1f}s  {note}")


def print_diagnostics(epoch, train_loss, val_loss, train_r2, val_r2,
                      lr, loss_history):
    gap = val_loss - train_loss
    r2_cl, r2_cd, r2_cm = val_r2
    if len(loss_history['val_loss']) >= 10:
        recent_trend = loss_history['val_loss'][-1] - loss_history['val_loss'][-10]
    else:
        recent_trend = 0

    print(f"\n{'='*65}")
    print(f"  DIAGNOSTIC REPORT -- Epoch {epoch}")
    print(f"{'='*65}")
    print(f"  Train loss    : {train_loss:.6f}")
    print(f"  Val loss      : {val_loss:.6f}")
    print(f"  Train/Val gap : {gap:.6f}  {'<- OVERFITTING if gap > 0.01' if gap > 0.01 else '<- OK'}")
    print(f"  Val R2  CL={r2_cl:.4f}  CD={r2_cd:.4f}  CM={r2_cm:.4f}")
    print(f"  Learning rate : {lr:.2e}")
    print(f"  10-epoch trend: {recent_trend:+.6f}  "
          f"{'<- still learning' if recent_trend < -1e-5 else '<- plateaued'}")
    print()

    advice = []
    if r2_cl < 0.85:
        advice.append("CL R2 < 0.85 -> underfitting")
        advice.append("  -> Widen hidden layers or reduce dropout_rate")
    if r2_cd < 0.80:
        advice.append("CD R2 < 0.80 -> CD underfitting")
        advice.append("  -> Increase CD loss weight or cd_penalty_factor")
    if r2_cm < 0.70:
        advice.append("CM R2 < 0.70 -> expected, CM from XFOIL is noisy near stall")
        advice.append("  -> Reduce CM loss weight to 0.3")
    if gap > 0.01:
        advice.append(f"Train/Val gap = {gap:.4f} -> OVERFITTING")
        advice.append("  -> Increase dropout_rate: 0.15 -> 0.25")
        advice.append("  -> Increase weight_decay: 5e-4 -> 1e-3")
    if recent_trend > -1e-6 and lr > 1e-5:
        advice.append("Loss plateaued -- scheduler will reduce LR automatically")
    if lr < 5e-6:
        advice.append("LR very low -- training nearly stopped")
    if np.isnan(train_loss) or train_loss > 100:
        advice.append("LOSS NaN OR EXPLODING -- STOP TRAINING")
        advice.append("  -> Reduce learning_rate 10x")
    if not advice:
        advice.append("Training looks healthy -- keep going.")
        if r2_cl > 0.97 and r2_cd > 0.93:
            advice.append("Excellent R2 -- model ready for optimizer integration.")

    print("  ADVICE:")
    for a in advice:
        print(f"    {a}")
    print(f"{'='*65}\n")


# ==============================================================================
#  PLOTTING
# ==============================================================================

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Training Curves', fontsize=13, fontweight='bold')
    epochs = range(1, len(history['train_loss']) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', lw=1.5, label='Train')
    ax.plot(epochs, history['val_loss'],   'r-', lw=1.5, label='Val')
    ax.set_title('Loss (weighted MSE)'); ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True, alpha=0.3)
    if history['best_epoch']:
        ax.axvline(history['best_epoch'], color='green', ls='--', lw=1,
                   label=f'Best epoch {history["best_epoch"]}')
        ax.legend()

    ax = axes[0, 1]
    ax.semilogy(epochs, history['lr'], 'g-', lw=1.5)
    ax.set_title('Learning Rate'); ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    gap = [v - t for t, v in zip(history['train_loss'], history['val_loss'])]
    ax.plot(epochs, gap, 'purple', lw=1.5)
    ax.axhline(0,    color='k',   lw=0.8, ls='--')
    ax.axhline(0.01, color='red', lw=0.8, ls=':', label='Overfit threshold 0.01')
    ax.set_title('Train/Val Gap'); ax.set_xlabel('Epoch')
    ax.legend(); ax.grid(True, alpha=0.3)

    for i, (name, col) in enumerate([('CL', 0), ('CD', 1), ('CM', 2)]):
        ax = axes[1, i]
        ax.plot(epochs, [r[col] for r in history['val_r2']],
                lw=1.5, label=f'Val R2 {name}')
        ax.plot(epochs, [r[col] for r in history['train_r2']],
                lw=1.0, alpha=0.6, label=f'Train R2 {name}')
        ax.axhline(1.0, color='green', lw=0.8, ls='--')
        ax.set_ylim(max(0, min([r[col] for r in history['val_r2']]) - 0.05), 1.05)
        ax.set_title(f'R2 -- {name}'); ax.set_xlabel('Epoch')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def plot_error_analysis(model, loader, scalers, device, save_path):
    model.eval()
    all_pred, all_target = [], []
    with torch.no_grad():
        for X_batch, Y_batch, _ in loader:
            pred = model(X_batch.to(device)).cpu().numpy()
            all_pred.append(pred)
            all_target.append(Y_batch.numpy())
    pred   = np.concatenate(all_pred,   axis=0)
    target = np.concatenate(all_target, axis=0)
    pred_phys   = inverse_transform_outputs(pred,   scalers)
    target_phys = inverse_transform_outputs(target, scalers)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Prediction vs Actual -- Validation Set', fontsize=13, fontweight='bold')
    names  = ['CL', 'CD', 'CM']
    colors = ['steelblue', 'coral', 'mediumseagreen']

    for i, (name, col) in enumerate(zip(names, colors)):
        t   = target_phys[:, i]
        p   = pred_phys[:, i]
        r2  = 1 - np.sum((t - p)**2) / (np.sum((t - t.mean())**2) + 1e-10)
        mae = np.mean(np.abs(t - p))

        ax = axes[0, i]
        ax.scatter(t, p, s=0.8, alpha=0.3, color=col)
        lim = [min(t.min(), p.min()), max(t.max(), p.max())]
        ax.plot(lim, lim, 'k--', lw=1, label='Perfect')
        ax.set_xlabel(f'Actual {name}'); ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}  R2={r2:.4f}  MAE={mae:.5f}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, i]
        residuals = p - t
        ax.hist(residuals, bins=60, color=col, edgecolor='white', alpha=0.8)
        ax.axvline(0, color='k', lw=1.5, ls='--')
        ax.axvline(residuals.mean(), color='red', lw=1, ls=':',
                   label=f'mean={residuals.mean():.4f}')
        ax.set_xlabel(f'{name} residual (pred - actual)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Residuals  std={residuals.std():.5f}')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


# ==============================================================================
#  HYPERPARAMETER GUIDE
# ==============================================================================

HYPERPARAMETER_GUIDE = """
HYPERPARAMETER TUNING GUIDE
============================
Read training_curves.png and error_analysis.png first.

SYMPTOM: Both train and val loss high, decreasing slowly
DIAGNOSIS: Underfitting
-> Widen: [256,256,256,128] -> [512,512,512,256]
-> Or reduce dropout_rate to 0.05
-> Or increase epochs

SYMPTOM: Train loss low, val loss high (gap > 0.01)
DIAGNOSIS: Overfitting
-> Increase dropout_rate: 0.15 -> 0.25
-> Increase weight_decay: 5e-4 -> 1e-3

SYMPTOM: Loss oscillating wildly
DIAGNOSIS: Learning rate too high
-> Reduce learning_rate: 1e-3 -> 5e-4

SYMPTOM: CL R2 > 0.97 but CD R2 < 0.88
DIAGNOSIS: CD needs more focus
-> loss_weights [1.0, 2.5, 0.5] -> [1.0, 3.5, 0.5]
-> Or increase cd_penalty_factor: 4.0 -> 5.0

SYMPTOM: CM R2 < 0.70
DIAGNOSIS: Expected -- XFOIL CM near stall is inherently noisy
-> Reduce CM weight: [1.0, 2.5, 0.5] -> [1.0, 2.5, 0.2]

SYMPTOM: CD residuals skewed positive (model overpredicts drag)
DIAGNOSIS: cd_penalty_factor too high -- model too conservative
-> Reduce cd_penalty_factor: 4.0 -> 3.0

SYMPTOM: CD residuals skewed negative (model still underpredicts)
DIAGNOSIS: cd_penalty_factor too low
-> Increase cd_penalty_factor: 4.0 -> 5.0 or 6.0

TARGET PERFORMANCE (before connecting Bayesian optimizer):
  CL  R2 > 0.97  MAE < 0.03
  CD  R2 > 0.93  MAE < 0.002
  CM  R2 > 0.80  MAE < 0.01
"""


# ==============================================================================
#  MAIN TRAINING LOOP
# ==============================================================================

def main():
    print('\n' + '#'*65)
    print('  AIRFOIL SURROGATE -- MLP TRAINING')
    print('  Asymmetric CD Loss | Fresh Train on 93k Rebuilt Dataset')
    print('#'*65)

    torch.manual_seed(CFG['seed'])
    np.random.seed(CFG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device : {device}')
    if device.type == 'cpu':
        print('  Note   : CPU training -- expect ~3-8 min per 100 epochs on i7-6700HQ.')

    # -- Load data --
    print(f'\n  Loading {DATA_CSV}...')
    if not os.path.exists(DATA_CSV):
        print(f'\n  X File not found: {DATA_CSV}')
        print(f'    Place master_dataset_clean.csv in the same folder as this script.')
        return
    df = pd.read_csv(DATA_CSV)
    print(f'  Rows     : {len(df):,}')
    print(f'  Airfoils : {df["airfoil"].nunique()}')
    print(f'  Split    : train={( df["split"]=="train").sum():,}  '
          f'val={(df["split"]=="val").sum():,}  '
          f'test={(df["split"]=="test").sum():,}')

    cst_cols = [c for c in df.columns if c.startswith('cst_')]
    n_inputs = len(cst_cols) + 4  # +xtr_top, +xtr_bot, +re, +alpha
    print(f'  CST cols : {len(cst_cols)}')
    print(f'  Inputs   : {n_inputs}  (CST + xtr_top + xtr_bot + Re + alpha)')
    print(f'  Outputs  : 3  (CL, CD, CM)')
    print(f'  CD loss  : AsymmetricLogCDLoss(penalty_factor={CFG["cd_penalty_factor"]})')

    # -- Verify data cleanliness --
    print(f'\n  Data quality check...')
    if 'fit_ok' in df.columns:
        bad_fit = (~df['fit_ok']).sum()
        if bad_fit > 0:
            print(f'  WARNING: {bad_fit} rows with fit_ok=False detected.')
            print(f'    Remove them before training.')
            return
        else:
            print(f'  OK: All rows have fit_ok=True')

    cst_data    = df[cst_cols]
    max_abs_cst = cst_data.abs().max().max()
    print(f'  Max |CST|: {max_abs_cst:.4f}  '
          f'{"OK (< 2.5)" if max_abs_cst < 2.5 else "WARNING: unusually large CST values"}')
    if max_abs_cst >= 2.5:
        return

    for col in ['xtr_top', 'xtr_bot']:
        if col not in df.columns:
            print(f'  X Column {col} not found in dataset.')
            print(f'    This version requires transition location data.')
            return
    print(f'  OK: xtr_top, xtr_bot present')

    # -- Compute and save scalers --
    print(f'\n  Computing scalers from training data...')
    scalers = compute_and_save_scalers(df, cst_cols, SCALER_JSON)

    # -- Normalize --
    print('\n  Normalizing features...')
    X_all = normalize_inputs(df, scalers, cst_cols)
    Y_all = normalize_outputs(df, scalers)

    # -- Sample weights --
    W_all = np.where(df['data_quality'] == 'tier1', 1.0,
                     CFG['tier2_weight']).astype(np.float32)

    # -- Sanity check --
    nan_count = (np.isnan(X_all).sum() + np.isinf(X_all).sum() +
                 np.isnan(Y_all).sum() + np.isinf(Y_all).sum())
    print(f'  NaN/Inf check : {nan_count} '
          f'{"<- OK CLEAN" if nan_count == 0 else "<- WARNING FIX BEFORE TRAINING"}')
    if nan_count > 0:
        return

    # -- Splits --
    train_mask = (df['split'] == 'train').values
    val_mask   = (df['split'] == 'val').values
    test_mask  = (df['split'] == 'test').values

    train_ds = AirfoilDataset(X_all[train_mask], Y_all[train_mask], W_all[train_mask])
    val_ds   = AirfoilDataset(X_all[val_mask],   Y_all[val_mask])
    test_ds  = AirfoilDataset(X_all[test_mask],  Y_all[test_mask])

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)

    # -- Model --
    input_dim = X_all.shape[1]
    model = AirfoilMLP(
        input_dim     = input_dim,
        hidden_layers = CFG['hidden_layers'],
        activation    = CFG['activation'],
        dropout_rate  = CFG['dropout_rate'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'\n  Model parameters : {total_params:,}')
    print(f'  Architecture     : {input_dim} -> '
          f'{" -> ".join(str(h) for h in CFG["hidden_layers"])} -> [CL|CD(+Re+alpha)|CM]')
    print(f'  CD head input    : 258 (256 trunk + Re + alpha) [ADJUSTMENT 1]')
    print(f'  Activation       : {CFG["activation"]}')
    print(f'  Dropout          : {CFG["dropout_rate"]}')

    # -- Warm start --
    best_model_path = os.path.join(MODEL_DIR, 'best_model.pt')
    warm_start_val  = float('inf')
    if CFG.get('warm_start', False):
        if os.path.exists(best_model_path):
            print(f'\n  Warm start: loading {best_model_path}...')
            ckpt = torch.load(best_model_path, map_location=device)
            saved_input_dim = list(ckpt['model_state'].values())[0].shape[1]
            if saved_input_dim != input_dim:
                print(f'  WARNING: Input dim mismatch: saved={saved_input_dim}, current={input_dim}')
                print(f'    Starting fresh instead.')
            else:
                model.load_state_dict(ckpt['model_state'])
                warm_start_val = ckpt.get('val_loss', float('inf'))
                loaded_r2      = ckpt.get('val_r2', [0, 0, 0])
                print(f'  Loaded epoch {ckpt.get("epoch","?")}  '
                      f'val_loss={warm_start_val:.6f}  '
                      f'R2 CL={loaded_r2[0]:.4f} CD={loaded_r2[1]:.4f} CM={loaded_r2[2]:.4f}')
        else:
            print(f'\n  WARNING: warm_start=True but {best_model_path} not found -- starting fresh.')
    else:
        print(f'\n  Training from scratch (warm_start=False)')

    # -- Optimizer and scheduler --
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CFG['learning_rate'],
                                 weight_decay=CFG['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  patience=CFG['lr_patience'],
                                  factor=CFG['lr_factor'],
                                  min_lr=CFG['lr_min'],
                                  verbose=False)

    # -- Asymmetric CD loss (training only) --
    cd_loss_fn = AsymmetricLogCDLoss(penalty_factor=CFG['cd_penalty_factor']).to(device)

    print(f'\n  Hyperparameters:')
    for k, v in CFG.items():
        print(f'    {k:<25} : {v}')

    print(f'\n  Starting training...')
    print(f'  Diagnostic report every 25 epochs.')
    print(f'  Plots saved at every new best model.\n')

    history = {'train_loss': [], 'val_loss': [],
               'train_r2': [],  'val_r2': [],
               'lr': [],        'best_epoch': None}

    best_val_loss      = warm_start_val
    best_smoothed_loss = float('inf')
    patience_count     = 0
    log_rows           = []

    print_epoch_header()
    t_start = time.time()

    for epoch in range(1, CFG['epochs'] + 1):
        t0 = time.time()

        # Training -- uses asymmetric CD loss
        train_loss, train_r2 = run_epoch(
            model, train_loader, optimizer, CFG['loss_weights'], device,
            training=True, cd_loss_fn=cd_loss_fn)

        # Validation -- uses symmetric MSE (for fair comparison)
        val_loss, val_r2 = run_epoch(
            model, val_loader, None, CFG['loss_weights'], device,
            training=False, cd_loss_fn=None)

        lr_now = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_r2'].append(train_r2)
        history['val_r2'].append(val_r2)
        history['lr'].append(lr_now)
        elapsed = time.time() - t0

        smooth_window = CFG['early_stop_smooth']
        if len(history['val_loss']) >= smooth_window:
            smoothed_val = np.mean(history['val_loss'][-smooth_window:])
        else:
            smoothed_val = val_loss

        # -- Save best model + plots --
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history['best_epoch'] = epoch
            torch.save({
                'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': val_loss, 'val_r2': val_r2.tolist(), 'cfg': CFG,
                'input_dim': input_dim,
            }, best_model_path)

            # Overwrite plots with best-so-far
            try:
                _tmp_hist = {k: history[k] for k in history}
                _tmp_hist['best_epoch'] = epoch
                plot_training_curves(_tmp_hist,
                    os.path.join(MODEL_DIR, 'training_curves.png'))
                plot_error_analysis(model, val_loader, scalers, device,
                    os.path.join(MODEL_DIR, 'error_analysis.png'))
            except Exception as _plot_err:
                print(f'  WARNING: Plot save failed: {_plot_err}')

        if smoothed_val < best_smoothed_loss:
            best_smoothed_loss = smoothed_val
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 5 == 0 or epoch == 1:
            print_epoch(epoch, train_loss, val_loss, val_r2, lr_now,
                        elapsed, best_val_loss, patience_count,
                        CFG['early_stop_patience'])
            if len(history['val_loss']) >= smooth_window:
                print(f"         {'':>5}  smoothed val={smoothed_val:.6f}  "
                      f"best_smoothed={best_smoothed_loss:.6f}")

        if epoch % 25 == 0:
            print_diagnostics(epoch, train_loss, val_loss, train_r2, val_r2,
                               lr_now, history)

        log_rows.append({
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'r2_cl': val_r2[0], 'r2_cd': val_r2[1], 'r2_cm': val_r2[2],
            'lr': lr_now, 'patience': patience_count,
        })

        if patience_count >= CFG['early_stop_patience']:
            print(f'\n  Early stopping at epoch {epoch}. '
                  f'No improvement for {patience_count} epochs.')
            break

    total_time = time.time() - t_start
    print(f'\n{"="*65}')
    print(f'  TRAINING COMPLETE')
    print(f'  Total time : {total_time/60:.1f} min')
    print(f'  Best val   : {best_val_loss:.6f}')

    if history['best_epoch'] is not None:
        best_r2 = history['val_r2'][history['best_epoch'] - 1]
        print(f'  Best epoch : {history["best_epoch"]} (found this session)')
    else:
        print(f'  Best epoch : warm start -- no new best found this session')
        _ckpt   = torch.load(best_model_path, map_location='cpu')
        best_r2 = _ckpt['val_r2']

    print(f'  Best R2    CL={best_r2[0]:.4f}  CD={best_r2[1]:.4f}  CM={best_r2[2]:.4f}')
    print(f'{"="*65}')

    # -- Save final model --
    final_path = os.path.join(MODEL_DIR, 'final_model.pt')
    torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                'val_loss': val_loss, 'val_r2': val_r2.tolist(), 'cfg': CFG,
                'input_dim': input_dim},
               final_path)
    print(f'\n  Best model  -> {best_model_path}')
    print(f'  Final model -> {final_path}')

    log_path = os.path.join(MODEL_DIR, 'training_log.csv')
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f'  Training log -> {log_path}')

    # -- Final plots (always generated) --
    print('\n  Generating final plots...')
    plot_training_curves(history, os.path.join(MODEL_DIR, 'training_curves.png'))

    print('  Loading best_model.pt for error analysis and test evaluation...')
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    plot_error_analysis(model, val_loader, scalers, device,
                        os.path.join(MODEL_DIR, 'error_analysis.png'))

    # -- Test set evaluation --
    print('\n  Evaluating on TEST SET (held-out benchmark)...')
    test_loss, test_r2 = run_epoch(
        model, test_loader, None, CFG['loss_weights'], device,
        training=False, cd_loss_fn=None)
    print(f'  Test loss : {test_loss:.6f}')
    print(f'  Test R2   CL={test_r2[0]:.4f}  CD={test_r2[1]:.4f}  CM={test_r2[2]:.4f}')
    print(f'  <- These test numbers are from best_model.pt (epoch {ckpt.get("epoch","?")})')
    print(f'    Use these as the official benchmark in your report.')

    guide_path = os.path.join(MODEL_DIR, 'hyperparameter_guide.txt')
    with open(guide_path, 'w') as f:
        f.write(HYPERPARAMETER_GUIDE)
    print(f'  Guide -> {guide_path}')

    print('\n' + '#'*65)
    print('  DONE.')
    print('  scaler_params.json has been regenerated automatically.')
    print('  best_model.pt and final_model.pt are in model/')
    print('  NOTE: bayesian_optimizer.py must be updated to pass')
    print('  xtr_top and xtr_bot removed as inputs (22 inputs, not 24).')
    print('#'*65 + '\n')


if __name__ == '__main__':
    main()
