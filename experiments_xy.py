# angle_experiments/experiments_xy.py
# NumPy-only pipeline that reads pre-extracted region features
# Expected keys per file: emg_rms_matrix_binned (B,C), ref_binned (B,), bin_size (int)
# Angle is parsed from filename (e.g., "90_F1_1.npy" or "...90Deg...")

import os
import re
import math
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib


# -------------------- Path helpers --------------------

SCRIPT_DIR = Path(__file__).resolve().parent

def _resolve_path(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (SCRIPT_DIR / q).resolve()


# -------------------- Loading features --------------------

REQ_KEYS = ("emg_rms_matrix_binned", "ref_binned", "bin_size")

def angle_from_fname(fname: str) -> float:
    """
    Parse the angle as the leading number before the first '_' in the filename.
    Examples:
      90_F1_1.npy   -> 90
      135_F2_2.npy  -> 135
    Fallback: if that fails, look for '<num>deg' anywhere.
    """
    base = Path(fname).stem  # drop extension
    first_token = base.split('_', 1)[0]  # everything before first underscore
    m = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)', first_token)
    if m:
        return float(m.group(1))
    # fallback: ...Deg...
    m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*deg', base, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse angle from filename: {fname}")


def load_feature_file(path: str):
    """Return (R:(B,C), mag:(B,), bin_size:int, angle_deg:float)."""
    obj = np.load(path, allow_pickle=True)
    d = {k: obj[k] for k in obj.files} if isinstance(obj, np.lib.npyio.NpzFile) else (obj.item() if hasattr(obj, "item") else obj)

    missing = [k for k in REQ_KEYS if k not in d]
    if missing:
        raise KeyError(f"{Path(path).name} missing keys: {missing}")

    R  = np.asarray(d["emg_rms_matrix_binned"], dtype=float)      # (B,C)
    mag = np.asarray(d["ref_binned"], dtype=float).reshape(-1)     # (B,)
    bs = int(np.asarray(d["bin_size"]).squeeze())

    B = min(R.shape[0], mag.shape[0])
    R, mag = R[:B, :], mag[:B]
    ang = angle_from_fname(path)
    return R, mag, bs, ang

def _expand_feature_files(feature_dir_or_glob: str):
    p = _resolve_path(feature_dir_or_glob)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(glob.glob(str(p / "*.npy")) + glob.glob(str(p / "*.npz")))
    return sorted(glob.glob(str(p)))  # treat as glob pattern

def collect_dataset(feature_dir, include_angle_target=True):
    files = _expand_feature_files(feature_dir)
    print(f"[PATH] feature-dir: {feature_dir} -> resolved: {_resolve_path(feature_dir)}")
    print(f"[PATH] matched files: {len(files)}")
    if not files:
        raise FileNotFoundError(f"No feature files matched: {feature_dir}")

    buckets = {}
    for p in files:
        try:
            R, mag, bs, ang = load_feature_file(p)
        except Exception as e:
            print(f"[SKIP] {Path(p).name} -> {e}")
            continue
        theta = math.radians(ang)
        fx = mag * math.cos(theta)
        fy = mag * math.sin(theta)
        y = np.column_stack([fx, fy, np.full_like(fx, ang, dtype=float)]) if include_angle_target else np.column_stack([fx, fy])

        buckets.setdefault(bs, {"X": [], "y": [], "files": []})
        buckets[bs]["X"].append(R)
        buckets[bs]["y"].append(y)
        buckets[bs]["files"].append(p)

    for bs in list(buckets.keys()):
        X = np.vstack(buckets[bs]["X"])
        y = np.vstack(buckets[bs]["y"])
        buckets[bs]["X"], buckets[bs]["y"] = X, y
        print(f"[LOAD] bin_size={bs}: X={X.shape}, y={y.shape}, files={len(buckets[bs]['files'])}")
    return buckets


# -------------------- Feature modes --------------------

def build_X(R, mode, channels=None):
    """
    R: (N, C)
    modes:
      - rms_matrix: all channels (N,C)
      - all_channels: RMS across channels (N,1)
      - average_channels: mean across channels (N,1)
      - single_channel: channels=[i] -> (N,1)
      - iterative_addition: channels=list -> [ (N,1), (N,2), ... ]
    """
    if mode == "rms_matrix":
        return R
    if mode == "all_channels":
        return np.sqrt(np.mean(R**2, axis=1, keepdims=True))
    if mode == "average_channels":
        return R.mean(axis=1, keepdims=True)
    if mode == "single_channel":
        if not channels or len(channels) != 1:
            raise ValueError("Provide channels=[idx] for single_channel")
        return R[:, [channels[0]]]
    if mode == "iterative_addition":
        if channels is None:
            channels = list(range(R.shape[1]))
        return [R[:, channels[:k]] for k in range(1, len(channels)+1)]
    raise ValueError(f"Unknown mode: {mode}")


# -------------------- CV experiments (multi-output) --------------------

def cv_scores_multi(X, y, cv=5, use_scaler=True):
    model = MultiOutputRegressor(Ridge())
    if use_scaler:
        model = Pipeline([("xscale", StandardScaler()), ("ridge", model)])
    mse = -np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error"))
    mae = -np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error"))
    return mse, mae

def run_modes(X, y, channel_modes, cv=5, use_scaler=True, scale_targets=True):
    # Optionally scale y to [0,1] per output for training/score stability; report in that space.
    if scale_targets:
        y_scaler = MinMaxScaler()
        y_used = y_scaler.fit_transform(y)
    else:
        y_scaler = None
        y_used = y

    results = {}
    for cfg in channel_modes:
        name, mode = cfg["name"], cfg["mode"]
        chs = cfg.get("channels")
        print(f"[EXP] {name} ({mode}) channels={chs}")

        if mode == "iterative_addition":
            X_list = build_X(X, mode, chs)
            mses, maes = [], []
            for k, Xk in enumerate(X_list, start=1):
                mse, mae = cv_scores_multi(Xk, y_used, cv=cv, use_scaler=use_scaler)
                mses.append(mse); maes.append(mae)
                print(f"      k={k}: MSE={mse:.4f}, MAE={mae:.4f}")
            results[name] = {"mse": mses, "mae": maes}
        else:
            Xd = build_X(X, mode, chs)
            mse, mae = cv_scores_multi(Xd, y_used, cv=cv, use_scaler=use_scaler)
            print(f"      MSE={mse:.4f}, MAE={mae:.4f}")
            results[name] = {"mse": mse, "mae": mae}
    return results


# -------------------- Train & save --------------------

def train_and_save_models(X, y, channel_modes, out_dir, use_scaler=True, scale_targets=True):
    os.makedirs(out_dir, exist_ok=True)
    y_scaler = MinMaxScaler() if scale_targets else None
    y_train = y_scaler.fit_transform(y) if y_scaler else y

    saved = []
    for cfg in channel_modes:
        name, mode, chs = cfg["name"], cfg["mode"], cfg.get("channels")
        if mode == "iterative_addition":
            X_list = build_X(X, mode, chs)
            for k, Xk in enumerate(X_list, start=1):
                base = MultiOutputRegressor(Ridge())
                model = Pipeline([("xscale", StandardScaler()), ("ridge", base)]) if use_scaler else base
                model.fit(Xk, y_train)
                path = os.path.join(out_dir, f"ridge_{name}_k{k}.joblib")
                joblib.dump(model, path); saved.append(path)
        else:
            Xd = build_X(X, mode, chs)
            base = MultiOutputRegressor(Ridge())
            model = Pipeline([("xscale", StandardScaler()), ("ridge", base)]) if use_scaler else base
            model.fit(Xd, y_train)
            path = os.path.join(out_dir, f"ridge_{name}.joblib")
            joblib.dump(model, path); saved.append(path)

    if y_scaler:
        joblib.dump(y_scaler, os.path.join(out_dir, "y_scaler.joblib"))
    print(f"[SAVE] Saved {len(saved)} models into {out_dir}")
    return saved


# -------------------- Save results CSV --------------------

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


# -------------------- Diagnostic plot --------------------

def plot_cv_reconstruction(X, y, cfg, out_path, cv=5, use_scaler=True, scale_targets=True, bin_size=None):
    name, mode, chs = cfg["name"], cfg["mode"], cfg.get("channels")
    Xd = build_X(X, mode, chs)

    y_scaler = MinMaxScaler() if scale_targets else None
    y_used = y_scaler.fit_transform(y) if y_scaler else y

    model = MultiOutputRegressor(Ridge())
    model = Pipeline([("xscale", StandardScaler()), ("ridge", model)]) if use_scaler else model
    y_hat_norm = cross_val_predict(model, Xd, y_used, cv=cv)
    y_hat = y_scaler.inverse_transform(y_hat_norm) if y_scaler else y_hat_norm

    t = np.arange(len(y))
    fig, axes = plt.subplots(2,1, figsize=(10,5.2), sharex=True)
    axes[0].plot(t, y[:,0], label="Fx true", lw=1.0)
    axes[0].plot(t, y_hat[:,0], label="Fx pred", lw=1.0, alpha=0.9)
    axes[0].grid(True, alpha=0.3); axes[0].legend(); axes[0].set_ylabel("Fx")
    axes[1].plot(t, y[:,1], label="Fy true", lw=1.0)
    axes[1].plot(t, y_hat[:,1], label="Fy pred", lw=1.0, alpha=0.9)
    axes[1].grid(True, alpha=0.3); axes[1].legend(); axes[1].set_xlabel("Bin"); axes[1].set_ylabel("Fy")
    fig.suptitle(f"CV reconstruction • {name} ({mode}) • bin={bin_size}", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[PLOT] {out_path}")

    # Angle MAE derived from Fx/Fy
    ang_true = np.degrees(np.arctan2(y[:,1], y[:,0]))
    ang_pred = np.degrees(np.arctan2(y_hat[:,1], y_hat[:,0]))
    diff = (ang_pred - ang_true + 180) % 360 - 180
    print(f"[ANGLE] CV angle MAE (deg) from Fx/Fy: {np.mean(np.abs(diff)):.2f}")


# -------------------- main --------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", default="data/preprocessing/P3_prp_feat",
                    help="Folder, file, or glob for .npy/.npz feature files")
    ap.add_argument("--out-root", default="./results_feat/P3_angle")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--no-x-scaler", action="store_true", help="Disable StandardScaler on X")
    ap.add_argument("--no-y-scaler", action="store_true", help="Disable MinMaxScaler on y")
    ap.add_argument("--no-angle-target", action="store_true", help="Do not include angle as 3rd output")
    args = ap.parse_args()

    FEATURE_DIR = args.feature_dir
    OUT_ROOT    = args.out_root
    CV_FOLDS    = args.cv
    USE_SCALER  = not args.no_x_scaler
    SCALE_Y     = not args.no_y_scaler
    INCLUDE_ANGLE_TARGET = not args.no_angle_target

    buckets = collect_dataset(FEATURE_DIR, include_angle_target=INCLUDE_ANGLE_TARGET)

    for bin_size, bundle in buckets.items():
        X, y, files = bundle["X"], bundle["y"], bundle["files"]
        out_dir = _resolve_path(os.path.join(OUT_ROOT, f"bin_{bin_size}"))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[RUN] bin_size={bin_size} -> X={X.shape}, y={y.shape}")

        n_ch = X.shape[1]
        channel_modes = [
            {"name": "rms_matrix",       "mode": "rms_matrix"},
            {"name": "all_channels",     "mode": "all_channels"},
            {"name": "average_channels", "mode": "average_channels"},
            {"name": "single_ch_0",      "mode": "single_channel", "channels": [0]},
            {"name": "iterative_add",    "mode": "iterative_addition", "channels": list(range(n_ch))},
        ]

        results = run_modes(X, y, channel_modes, cv=CV_FOLDS, use_scaler=USE_SCALER, scale_targets=SCALE_Y)
        save_results(results, str(out_dir), tag="metrics")

        train_and_save_models(X, y, channel_modes, str(out_dir), use_scaler=USE_SCALER, scale_targets=SCALE_Y)

        plot_cv_reconstruction(
            X, y,
            cfg={"name": "rms_matrix", "mode": "rms_matrix"},
            out_path=str(out_dir / "cv_reconstruction_rms_matrix.png"),
            cv=CV_FOLDS, use_scaler=USE_SCALER, scale_targets=SCALE_Y, bin_size=bin_size
        )

        with open(out_dir / "files.txt", "w") as f:
            f.write("\n".join(files))

    print("\n[DONE] All bin sizes processed.")


if __name__ == "__main__":
    main()
