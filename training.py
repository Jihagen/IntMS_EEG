# file: train_from_features.py
import os
import glob
import time
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import itertools
import matplotlib.pyplot as plt


# =========================
# Loading pre-extracted features
# =========================

REQ_KEYS = ("emg_rms_matrix_binned", "ref_binned", "bin_size")

def load_feature_file(path):
    """
    Load a .npy/.npz region feature file and return:
      R  : (B, C)  per-bin RMS per channel
      y  : (B,)    per-bin reference (force) mean
      bin_size : int
      meta : dict of any extra keys
    """
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        d = {k: obj[k] for k in obj.files}
    else:
        d = obj.item() if hasattr(obj, "item") else obj

    missing = [k for k in REQ_KEYS if k not in d]
    if missing:
        raise KeyError(f"{os.path.basename(path)} missing keys: {missing}")

    R = np.asarray(d["emg_rms_matrix_binned"], dtype=float)  # (B,C)
    y = np.asarray(d["ref_binned"], dtype=float).squeeze()   # (B,)
    bs = int(np.asarray(d["bin_size"]).squeeze())

    if R.ndim != 2:
        raise ValueError(f"{path}: 'emg_rms_matrix_binned' must be 2D, got {R.shape}")
    if y.ndim != 1:
        y = y.reshape(-1)
    if R.shape[0] != y.shape[0]:
        B = min(R.shape[0], y.shape[0])
        R = R[:B, :]
        y = y[:B]

    meta = {k: d[k] for k in d.keys() if k not in ("emg_rms_matrix_binned", "ref_binned", "bin_size")}
    return R, y, bs, meta


def collect_dataset(feature_dir):
    """
    Scan folder for .npy/.npz feature files and aggregate by bin_size.
    Returns dict: { bin_size : {"X": (N, C), "y": (N,), "files": [paths]} }
    """
    files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")) +
                   glob.glob(os.path.join(feature_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npy/.npz files in {feature_dir}")

    buckets = {}
    for p in files:
        try:
            R, y, bs, _ = load_feature_file(p)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(p)} -> {e}")
            continue
        if bs not in buckets:
            buckets[bs] = {"X": [], "y": [], "files": []}
        buckets[bs]["X"].append(R)
        buckets[bs]["y"].append(y)
        buckets[bs]["files"].append(p)

    for bs in list(buckets.keys()):
        X = np.vstack(buckets[bs]["X"])
        y = np.concatenate(buckets[bs]["y"])
        buckets[bs]["X"] = X
        buckets[bs]["y"] = y
        print(f"[LOAD] bin_size={bs}: X={X.shape}, y={y.shape}, files={len(buckets[bs]['files'])}")
    return buckets


# =========================
# Feature assembly (modes)
# =========================

def build_X(R, mode, channels=None):
    """
    R: (N, C)
    Returns 2D array suitable for sklearn: (N, k)
    """
    if mode == "rms_matrix":
        return R
    if mode == "all_channels":        # RMS across channels -> (N,1)
        return np.sqrt(np.mean(R**2, axis=1, keepdims=True))
    if mode == "average_channels":    # mean across channels -> (N,1)
        return R.mean(axis=1, keepdims=True)
    if mode == "single_channel":
        if not channels or len(channels) != 1:
            raise ValueError("Provide channels=[idx] for single_channel")
        return R[:, [channels[0]]]
    if mode == "iterative_addition":
        if channels is None:
            channels = list(range(R.shape[1]))
        # Returns a list of X_k with 1..K channels included
        X_list = [R[:, channels[:k]] for k in range(1, len(channels)+1)]
        return X_list
    raise ValueError(f"Unknown mode: {mode}")


# =========================
# Cross-val experiments
# =========================

def cv_scores(X, y, cv=5, use_scaler=True):
    """
    Get CV MSE/MAE for a single design matrix X and target y.
    """
    if use_scaler:
        model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                          ("ridge", Ridge())])
    else:
        model = Ridge()

    mse = -np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"))
    mae = -np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error"))
    return mse, mae


def run_modes(X, y, channel_modes, cv=5, use_scaler=True):
    """
    channel_modes: list of dicts like
      {'name':'all_channels', 'mode':'all_channels'}
      {'name':'single_ch_0', 'mode':'single_channel', 'channels':[0]}
      {'name':'iterative_add', 'mode':'iterative_addition', 'channels':[0,1,...]}
      {'name':'rms_matrix', 'mode':'rms_matrix'}
    """
    results = {}
    for cfg in channel_modes:
        name, mode = cfg["name"], cfg["mode"]
        chs = cfg.get("channels")
        print(f"[EXP] {name} ({mode}) channels={chs}")

        if mode == "iterative_addition":
            X_list = build_X(X, mode, chs)
            mses, maes = [], []
            for k, Xk in enumerate(X_list, start=1):
                mse, mae = cv_scores(Xk, y, cv=cv, use_scaler=use_scaler)
                mses.append(mse); maes.append(mae)
                print(f"      k={k} -> MSE={mse:.4f}, MAE={mae:.4f}")
            results[name] = {"mse": mses, "mae": maes}
        else:
            Xd = build_X(X, mode, chs)
            mse, mae = cv_scores(Xd, y, cv=cv, use_scaler=use_scaler)
            print(f"      MSE={mse:.4f}, MAE={mae:.4f}")
            results[name] = {"mse": mse, "mae": mae}
    return results


# =========================
# Train final models & save
# =========================

def train_and_save_models(X, y, channel_modes, out_dir, use_scaler=True):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for cfg in channel_modes:
        name, mode = cfg["name"], cfg["mode"]
        chs = cfg.get("channels")
        if mode == "iterative_addition":
            # Train separate models for each prefix length
            X_list = build_X(X, mode, chs)
            for k, Xk in enumerate(X_list, start=1):
                model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]) if use_scaler else Ridge()
                model.fit(Xk, y)
                path = os.path.join(out_dir, f"ridge_{name}_k{k}.joblib")
                joblib.dump(model, path)
                saved.append(path)
        else:
            Xd = build_X(X, mode, chs)
            model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]) if use_scaler else Ridge()
            model.fit(Xd, y)
            path = os.path.join(out_dir, f"ridge_{name}.joblib")
            joblib.dump(model, path)
            saved.append(path)
    print(f"[SAVE] Saved {len(saved)} models into {out_dir}")
    return saved


# =========================
# Save results CSV
# =========================

def save_results(results, out_dir, tag="results"):
    rows = []
    for name, res in results.items():
        if isinstance(res.get("mae"), list):
            for k, (mse, mae) in enumerate(zip(res["mse"], res["mae"]), start=1):
                rows.append({"mode": name, "k_channels": k, "mse": mse, "mae": mae})
        else:
            rows.append({"mode": name, "k_channels": None, "mse": res["mse"], "mae": res["mae"]})
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] Wrote {out_csv} ({df.shape[0]} rows)")
    return df


# =========================
# Optional: quick reconstruction plot (CV predict)
# =========================

def plot_cv_reconstruction(X, y, cfg, out_path, cv=5, use_scaler=True, bin_size=None):
    """
    One visual check: CV prediction vs true target for a given mode.
    """
    name, mode = cfg["name"], cfg["mode"]
    chs = cfg.get("channels")
    Xd = build_X(X, mode, chs)

    model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())]) if use_scaler else Ridge()
    y_hat = cross_val_predict(model, Xd, y, cv=cv)

    t = np.arange(len(y))
    xlabel = "Bin index" if bin_size is None else f"Time (bin={bin_size})"
    plt.figure(figsize=(10, 3.2))
    plt.plot(t, y,  label="true", lw=1.0)
    plt.plot(t, y_hat, label=f"CV {name}", lw=1.0, alpha=0.9)
    plt.xlabel(xlabel); plt.ylabel("ref_binned")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] {out_path}")


# =========================
# Main
# =========================

def main():
    FEATURE_DIR = "data/preprocessing/P3_prp_feat"  # <- your folder
    OUT_ROOT    = "./results_feat/P3"
    CV_FOLDS    = 5
    USE_SCALER  = True  # Standardize inputs inside Ridge (recommended)

    buckets = collect_dataset(FEATURE_DIR)
    for bin_size, bundle in buckets.items():
        X, y = bundle["X"], bundle["y"]
        out_dir = os.path.join(OUT_ROOT, f"bin_{bin_size}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[RUN] bin_size={bin_size} -> X={X.shape}, y={y.shape}")

        # Define modes (feel free to customize)
        n_ch = X.shape[1]
        channel_modes = [
            {"name": "rms_matrix",      "mode": "rms_matrix"},
            {"name": "all_channels",    "mode": "all_channels"},
            {"name": "average_channels","mode": "average_channels"},
            {"name": "single_channel_0","mode": "single_channel", "channels": [0]},
            {"name": "iterative_add",   "mode": "iterative_addition", "channels": list(range(n_ch))},
        ]

        # 1) Cross-val experiments
        results = run_modes(X, y, channel_modes, cv=CV_FOLDS, use_scaler=USE_SCALER)
        df = save_results(results, out_dir, tag="metrics")

        # 2) Train final models and save
        _saved = train_and_save_models(X, y, channel_modes, out_dir, use_scaler=USE_SCALER)

        # 3) Optional: one reconstruction plot for a chosen mode
        plot_cv_reconstruction(
            X, y,
            cfg={"name": "rms_matrix", "mode": "rms_matrix"},
            out_path=os.path.join(out_dir, "cv_reconstruction_rms_matrix.png"),
            cv=CV_FOLDS, use_scaler=USE_SCALER, bin_size=bin_size
        )

    print("\n[DONE] All bin sizes processed.")


if __name__ == "__main__":
    main()
