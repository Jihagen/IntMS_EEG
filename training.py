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
# Binning augment — tuned to your file schema
# Uses: data (C, T), ref_signal (T,)
# Produces: emg_rms_matrix_binned_{bs}, ref_binned_{bs}, bin_size_{bs}, T_use_for_bins_{bs}
# =========================
AUTO_ADD_MISSING_BINS = True
AUGMENT_BIN_SIZES = [500, 250]

import glob
from pathlib import Path

def _rms_bin_signal_matrix_from_CT(data_CT: np.ndarray, bin_size: int) -> tuple[np.ndarray, int]:
    """
    data_CT: (C, T) -> returns (B, C) RMS per bin, and T_used
    """
    data_CT = np.asarray(data_CT, dtype=float)
    if data_CT.ndim != 2:
        raise ValueError(f"'data' must be 2D (C,T), got {data_CT.shape}")
    C, T = data_CT.shape
    n_bins = T // bin_size
    if n_bins <= 0:
        return np.empty((0, C), dtype=float), 0
    T_used = n_bins * bin_size
    # reshape per channel: (C, n_bins, bin_size) -> RMS over last axis
    d = data_CT[:, :T_used].reshape(C, n_bins, bin_size)
    rms = np.sqrt((d * d).mean(axis=2))          # (C, n_bins)
    return rms.T, T_used                         # (B, C), T_used

def _mean_bin_1d(y: np.ndarray, bin_size: int) -> tuple[np.ndarray, int]:
    """
    y: (T,) -> returns (B,), and T_used
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n_bins = y.size // bin_size
    if n_bins <= 0:
        return np.array([], dtype=float), 0
    T_used = n_bins * bin_size
    yb = y[:T_used].reshape(n_bins, bin_size).mean(axis=1)
    return yb, T_used

def augment_folder_with_bins(feature_dir: Path, bin_sizes=(500, 250)):
    paths = sorted(glob.glob(str(feature_dir / "*.npy"))) + sorted(glob.glob(str(feature_dir / "*.npz")))
    if not paths:
        print(f"[AUGMENT] No files in {feature_dir}")
        return

    for p in paths:
        obj = np.load(p, allow_pickle=True)
        # support npz and npy(pickled-dict)
        if isinstance(obj, np.lib.npyio.NpzFile):
            d = {k: obj[k] for k in obj.files}
            container = "npz"
        else:
            try:
                d = obj.item()
                container = "npy"
            except Exception:
                print(f"[AUGMENT][SKIP] {Path(p).name}: not a pickled dict/npz")
                continue

        # your raw keys (from your dump)
        if "data" not in d or "ref_signal" not in d:
            print(f"[AUGMENT][SKIP] {Path(p).name}: missing 'data'/'ref_signal'")
            continue

        data_CT   = np.asarray(d["data"], dtype=float)          # (C, T)
        ref_T     = np.asarray(d["ref_signal"], dtype=float)    # (T,)
        changed = False

        for bs in bin_sizes:
            key_R   = f"emg_rms_matrix_binned_{bs}"
            key_ref = f"ref_binned_{bs}"
            key_bs  = f"bin_size_{bs}"
            key_T   = f"T_use_for_bins_{bs}"

            if key_R in d and key_ref in d:
                continue  # already present

            R_BC, T_used_sig = _rms_bin_signal_matrix_from_CT(data_CT, bs)
            ref_B, T_used_ref = _mean_bin_1d(ref_T, bs)
            # align just in case (should match)
            B = min(R_BC.shape[0], ref_B.shape[0])
            R_BC  = R_BC[:B, :]
            ref_B = ref_B[:B]
            T_used = min(T_used_sig, T_used_ref)

            d[key_R]  = R_BC
            d[key_ref]= ref_B
            d[key_bs] = int(bs)
            d[key_T]  = int(T_used)
            changed = True
            print(f"[AUGMENT][ADD] {Path(p).name}: +{bs} (bins={B}, C={R_BC.shape[1] if B>0 else 'n/a'})")

        if changed:
            if container == "npz":
                np.savez(p, **d)
            else:
                np.save(p, d)

    print("[AUGMENT] Completed.")


# =========================
# Loading pre-extracted features (now with bin-size selection)
# =========================

REQ_KEYS_BASE = ("emg_rms_matrix_binned", "ref_binned", "bin_size")

def load_feature_file(path, target_bin_size: int | None):
    """
    Load a .npy/.npz region feature file for a specific bin size.

    Returns:
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

    # Choose keys based on target_bin_size
    if target_bin_size is None or target_bin_size == 1000:
        if all(k in d for k in REQ_KEYS_BASE):
            R = np.asarray(d["emg_rms_matrix_binned"], dtype=float)
            y = np.asarray(d["ref_binned"], dtype=float).squeeze()
            bs = int(np.asarray(d["bin_size"]).squeeze())
        elif "emg_rms_matrix_binned_1000" in d and "ref_binned_1000" in d:
            R = np.asarray(d["emg_rms_matrix_binned_1000"], dtype=float)
            y = np.asarray(d["ref_binned_1000"], dtype=float).squeeze()
            bs = int(d.get("bin_size_1000", 1000))
        else:
            raise KeyError(f"{os.path.basename(path)} missing 1000-bin keys")
    else:
        key_R   = f"emg_rms_matrix_binned_{target_bin_size}"
        key_ref = f"ref_binned_{target_bin_size}"
        key_bs  = f"bin_size_{target_bin_size}"
        if key_R not in d or key_ref not in d:
            raise KeyError(f"{os.path.basename(path)} missing keys for bin {target_bin_size}: {key_R}, {key_ref}")
        R = np.asarray(d[key_R], dtype=float)
        y = np.asarray(d[key_ref], dtype=float).squeeze()
        bs = int(d.get(key_bs, target_bin_size))

    if R.ndim != 2:
        raise ValueError(f"{path}: 'emg_rms_matrix_binned*' must be 2D, got {R.shape}")
    if y.ndim != 1:
        y = y.reshape(-1)
    if R.shape[0] != y.shape[0]:
        B = min(R.shape[0], y.shape[0])
        R = R[:B, :]
        y = y[:B]

    if R.shape[0] == 0:
        raise ValueError(f"{os.path.basename(path)} -> zero bins for bin_size={bs}")

    # keep other keys as meta
    base_and_this = set(REQ_KEYS_BASE) | {f"emg_rms_matrix_binned_{target_bin_size}",
                                          f"ref_binned_{target_bin_size}",
                                          f"bin_size_{target_bin_size}"}
    meta = {k: d[k] for k in d.keys() if k not in base_and_this}
    return R, y, bs, meta


def collect_dataset(feature_dir, target_bin_size: int | None):
    """
    Scan folder and aggregate data for a specific target_bin_size.
    Returns dict: { target_bin_size : {"X": (N, C), "y": (N,), "files": [paths]} }
    """
    files = sorted(glob.glob(os.path.join(feature_dir, "*.npy")) +
                   glob.glob(os.path.join(feature_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npy/.npz files in {feature_dir}")

    X_list, y_list, used = [], [], []
    for p in files:
        try:
            R, y, bs, _ = load_feature_file(p, target_bin_size=target_bin_size)
            X_list.append(R)
            y_list.append(y)
            used.append(p)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(p)} -> {e}")
            continue

    if not X_list:
        raise RuntimeError(f"No usable files for bin_size={target_bin_size} in {feature_dir}")

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"[LOAD] bin_size={target_bin_size}: X={X.shape}, y={y.shape}, files={len(used)}")
    return {int(target_bin_size if target_bin_size is not None else 1000): {"X": X, "y": y, "files": used}}

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
        X_list = [R[:, channels[:k]] for k in range(1, len(channels)+1)]
        return X_list
    raise ValueError(f"Unknown mode: {mode}")

# =========================
# Cross-val experiments
# =========================

def cv_scores(X, y, cv=5, use_scaler=True):
    if use_scaler:
        model = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                          ("ridge", Ridge())])
    else:
        model = Ridge()

    mse = -np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"))
    mae = -np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error"))
    return mse, mae

def run_modes(X, y, channel_modes, cv=5, use_scaler=True):
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

def save_results(results, out_dir, tag="metrics"):
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
    # Adjust patient/folder here
    FEATURE_DIR = Path("data/preprocessing/P5_prp_feat")
    OUT_ROOT    = Path("./results_feat/P5")
    CV_FOLDS    = 5
    USE_SCALER  = True
    BIN_SIZES   = [1000, 500, 250]

    # 0) Augment files in-place with 500/250 if missing
    if AUTO_ADD_MISSING_BINS:
        augment_folder_with_bins(FEATURE_DIR, AUGMENT_BIN_SIZES)

    # 1) Loop experiments per bin size
    for target_bin in BIN_SIZES:
        print(f"\n[RUN] bin_size={target_bin}")
        buckets = collect_dataset(FEATURE_DIR, target_bin_size=target_bin)
        bundle = buckets[target_bin]
        X, y = bundle["X"], bundle["y"]

        out_dir = OUT_ROOT / f"bin_{target_bin}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Define modes
        n_ch = X.shape[1]
        channel_modes = [
            {"name": "rms_matrix",       "mode": "rms_matrix"},
            {"name": "all_channels",     "mode": "all_channels"},
            {"name": "average_channels", "mode": "average_channels"},
            {"name": "single_channel_0", "mode": "single_channel", "channels": [0]},
            {"name": "iterative_add",    "mode": "iterative_addition", "channels": list(range(n_ch))},
        ]

        # 1) Cross-val experiments
        results = run_modes(X, y, channel_modes, cv=CV_FOLDS, use_scaler=USE_SCALER)
        _ = save_results(results, out_dir, tag="metrics")

        # 2) Train final models and save
        _ = train_and_save_models(X, y, channel_modes, out_dir, use_scaler=USE_SCALER)

        # 3) Optional: quick CV reconstruction plot
        plot_cv_reconstruction(
            X, y,
            cfg={"name": "rms_matrix", "mode": "rms_matrix"},
            out_path=str(out_dir / "cv_reconstruction_rms_matrix.png"),
            cv=CV_FOLDS, use_scaler=USE_SCALER, bin_size=target_bin
        )

    print("\n[DONE] All bin sizes processed.")

if __name__ == "__main__":
    main()
