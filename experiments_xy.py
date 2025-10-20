# angle_experiments/experiments_xy.py
# New pipeline: read *_combined.npy, preprocess into plateau segments, bin, split by groups.

import os
import glob
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib

# -------------------- Path helpers --------------------

SCRIPT_DIR = Path(__file__).resolve().parent

def _resolve_path(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (SCRIPT_DIR / q).resolve()

# -------------------- Loader from *_combined.npy via preprocess_plateaus --------------------

from data.preprocessing.preprocess import preprocess_plateaus  # <-- your preprocessor

def _expand_combined_files(dir_or_glob: str):
    p = _resolve_path(dir_or_glob)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(glob.glob(str(p / "*_combined.npy")))
    return sorted(glob.glob(str(p)))  # treat as glob

def _bin_series(arr, bin_len, agg="mean"):
    """
    Bin a 1D or 2D array along axis=0 into chunks of length bin_len.
    - 1D: (N,)   -> (B,)
    - 2D: (N,C)  -> (B,C)
    Drops the tail if not divisible.
    """
    A = np.asarray(arr)
    if A.ndim == 1:
        N = A.shape[0]
        B = max(1, N // bin_len)
        if N < bin_len:
            return A.reshape(1, -1).mean(axis=1)
        A2 = A[:B*bin_len].reshape(B, bin_len)
        return A2.mean(axis=1) if agg == "mean" else np.median(A2, axis=1)
    elif A.ndim == 2:
        N, C = A.shape
        B = max(1, N // bin_len)
        if N < bin_len:
            return A.reshape(1, N, C).mean(axis=1)
        A2 = A[:B*bin_len].reshape(B, bin_len, C)
        return A2.mean(axis=1) if agg == "mean" else np.median(A2, axis=1)
    else:
        raise ValueError("arr must be 1D or 2D")

def _samples_per_bin_from_seconds(fs, bin_sec):
    return max(1, int(round(fs * float(bin_sec))))

def collect_dataset_from_combined(
    combined_dir_or_glob: str,
    bin_sec: float = 0.050,           # 50 ms bins
    include_angle_target: bool = True,
    rms_win_samples: int = 100,       # RMS window on the common grid
    modes=("rms_matrix",),            # we feed rms_matrix to dataset; other modes are tested later
    single_channel_idx=None,
    iterative_channels=None,
):
    """
    Returns buckets keyed by bin_len (samples):
      buckets[bin_len] = {"X": (sum_B,C), "y": (sum_B,2/3), "files": [...], "groups": (sum_B,), "angles": (sum_B,)}
      - groups: string ID per row -> 'path/to/file.npy::plateau01'
      - angles: angle label per row (deg), averaged per bin (for diagnostics)
    """
    files = _expand_combined_files(combined_dir_or_glob)
    print(f"[PATH] combined-dir: {combined_dir_or_glob} -> resolved: {_resolve_path(combined_dir_or_glob)}")
    print(f"[PATH] matched files: {len(files)}")
    if not files:
        raise FileNotFoundError(f"No *_combined.npy matched: {combined_dir_or_glob}")

    buckets = {}
    used_files = 0

    for f in files:
        try:
            payload = np.load(f, allow_pickle=True).item()
        except Exception as e:
            print(f"[SKIP] {Path(f).name} -> cannot load: {e}")
            continue

        pp = preprocess_plateaus(
            payload,
            rms_win_samples=rms_win_samples,
            modes=modes,
            single_channel_idx=single_channel_idx,
            iterative_channels=iterative_channels,
            keep_intended_angle=False
        )

        segs = pp.get("segments", [])
        if not segs:
            print(f"[SKIP] {Path(f).name} -> no plateaus found")
            continue

        # fs + bin
        fs = segs[0]["fs"] or (1.0 / np.median(np.diff(segs[0]["signals"]["t"])))
        bin_len = _samples_per_bin_from_seconds(fs, bin_sec)

        for seg in segs:  # plateau01 / plateau23
            R = seg["emg"]["rms_matrix"]             # (N,C)
            Fx = seg["signals"]["Fx"].reshape(-1)
            Fy = seg["signals"]["Fy"].reshape(-1)

            # Equalize length
            N = min(R.shape[0], Fx.size, Fy.size)
            R, Fx, Fy = R[:N, :], Fx[:N], Fy[:N]

            # Angle timeseries (achieved): from Fx,Fy (deg in [0,360))
            ang_ts = np.degrees(np.arctan2(Fy, Fx))
            ang_ts = np.where(ang_ts < 0, ang_ts + 360.0, ang_ts)

            # Bin
            Rb   = _bin_series(R,   bin_len, agg="mean")           # (B,C)
            Fxb  = _bin_series(Fx,  bin_len, agg="mean").reshape(-1,1)
            Fyb  = _bin_series(Fy,  bin_len, agg="mean").reshape(-1,1)
            Angb = _bin_series(ang_ts, bin_len, agg="mean").reshape(-1,1)

            if include_angle_target:
                yb = np.hstack([Fxb, Fyb, Angb])                   # (B,3)
            else:
                yb = np.hstack([Fxb, Fyb])                         # (B,2)

            # Defensive trim
            B = min(Rb.shape[0], yb.shape[0])
            Rb, yb, Angb = Rb[:B, :], yb[:B, :], Angb[:B, 0]

            # Group label for this segment (keep all its bins together)
            group_label = f"{f}::{seg['name']}"
            groups_seg = np.full((B,), group_label, dtype=object)

            key = int(bin_len)
            buckets.setdefault(key, {"X": [], "y": [], "files": [], "groups": [], "angles": []})
            buckets[key]["X"].append(Rb)
            buckets[key]["y"].append(yb)
            buckets[key]["files"].append(group_label)
            buckets[key]["groups"].append(groups_seg)
            buckets[key]["angles"].append(Angb)

        used_files += 1

    # Stack lists
    for key in list(buckets.keys()):
        X = np.vstack(buckets[key]["X"]) if buckets[key]["X"] else np.empty((0,0))
        y = np.vstack(buckets[key]["y"]) if buckets[key]["y"] else np.empty((0,0))
        groups = np.concatenate(buckets[key]["groups"]) if buckets[key]["groups"] else np.array([], dtype=object)
        angles = np.concatenate(buckets[key]["angles"]) if buckets[key]["angles"] else np.array([], dtype=float)
        buckets[key]["X"], buckets[key]["y"], buckets[key]["groups"], buckets[key]["angles"] = X, y, groups, angles
        approx_sec = (key / fs) if 'fs' in locals() and fs else float('nan')
        print(f"[LOAD] bin_len={key} (~{approx_sec:.3f}s): X={X.shape}, y={y.shape}, groups={len(np.unique(groups))}")
    print(f"[DONE] processed files: {used_files}/{len(files)}")
    return buckets

# -------------------- Feature modes (unchanged) --------------------

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

# -------------------- Group-aware CV helpers --------------------

def cv_scores_multi(X, y, groups=None, cv=5, use_scaler=True):
    model = MultiOutputRegressor(Ridge())
    if use_scaler:
        model = Pipeline([("xscale", StandardScaler()), ("ridge", model)])
    splitter = GroupKFold(n_splits=cv) if groups is not None else cv
    mse = -np.mean(cross_val_score(model, X, y, cv=splitter, groups=groups, scoring="neg_mean_squared_error"))
    mae = -np.mean(cross_val_score(model, X, y, cv=splitter, groups=groups, scoring="neg_mean_absolute_error"))
    return mse, mae

def run_modes(X, y, groups, channel_modes, cv=5, use_scaler=True, scale_targets=True):
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
                mse, mae = cv_scores_multi(Xk, y_used, groups=groups, cv=cv, use_scaler=use_scaler)
                mses.append(mse); maes.append(mae)
                print(f"      k={k}: MSE={mse:.4f}, MAE={mae:.4f}")
            results[name] = {"mse": mses, "mae": maes}
        else:
            Xd = build_X(X, mode, chs)
            mse, mae = cv_scores_multi(Xd, y_used, groups=groups, cv=cv, use_scaler=use_scaler)
            print(f"      MSE={mse:.4f}, MAE={mae:.4f}")
            results[name] = {"mse": mse, "mae": mae}
    return results

# -------------------- Train & save (unchanged) --------------------

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

# -------------------- Diagnostic plot (unchanged logic) --------------------

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

    ang_true = np.degrees(np.arctan2(y[:,1], y[:,0]))
    ang_pred = np.degrees(np.arctan2(y_hat[:,1], y_hat[:,0]))
    diff = (ang_pred - ang_true + 180) % 360 - 180
    print(f"[ANGLE] CV angle MAE (deg) from Fx/Fy: {np.mean(np.abs(diff)):.2f}")

# -------------------- Grouped holdout: one plateau per *nominal* angle --------------------

def nominal_angle_from_path(p: str) -> float:
    """
    Parse the intended/nominal angle from the combined file name.
    Accepts either a '<num>Deg' token or a leading number before '_'.
    Examples: '90Deg_F2_1_combined.npy' -> 90, '135_F1_2_combined.npy' -> 135
    """
    base = Path(p).stem
    m = re.search(r'([+-]?\d+(?:\.\d+)?)\s*deg', base, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    first = base.split('_', 1)[0]
    m = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)', first)
    if m:
        return float(m.group(1))
    raise ValueError(f"Cannot parse nominal angle from: {p}")

def split_one_plateau_per_angle_test(groups, rng=42):
    """
    groups are strings: '.../90Deg_F2_1_combined.npy::plateau01'
    We parse the nominal angle from the part before '::', bucket plateaus by that,
    and pick exactly one plateau per angle bucket as test.
    """
    rnd = np.random.RandomState(rng)

    # 1) rows per group
    group_to_rows = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_rows[g].append(i)
    group_to_rows = {g: np.asarray(ix, dtype=int) for g, ix in group_to_rows.items()}

    # 2) parse nominal angle per group
    angle_to_groups = defaultdict(list)
    for g in group_to_rows:
        src_path = g.split("::", 1)[0]
        ang_nom = nominal_angle_from_path(src_path)
        angle_to_groups[ang_nom].append(g)

    # 3) pick one group per nominal angle
    test_groups = [rnd.choice(glist) for glist in angle_to_groups.values()]

    # 4) rows -> indices
    test_mask = np.zeros(len(groups), dtype=bool)
    for g in test_groups:
        test_mask[group_to_rows[g]] = True

    test_idx  = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx, test_groups

# -------------------- main --------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", default="data/preprocessing/P5_combined",
                    help="Folder/file/glob for *_combined.npy files")
    ap.add_argument("--out-root", default="./results_feat/P5_angle")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--bin-sec", type=float, default=0.050, help="Bin size in seconds (e.g., 0.05)")
    ap.add_argument("--rms-win", type=int, default=100, help="RMS window (samples) on common time grid")
    ap.add_argument("--no-x-scaler", action="store_true", help="Disable StandardScaler on X")
    ap.add_argument("--no-y-scaler", action="store_true", help="Disable MinMaxScaler on y")
    ap.add_argument("--no-angle-target", action="store_true", help="Do not include angle as 3rd output")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    FEATURE_DIR = args.feature_dir
    OUT_ROOT    = args.out_root
    CV_FOLDS    = args.cv
    BIN_SEC     = args.bin_sec
    RMS_WIN     = args.rms_win
    USE_SCALER  = not args.no_x_scaler
    SCALE_Y     = not args.no_y_scaler
    INCLUDE_ANGLE_TARGET = not args.no_angle_target
    RNG         = args.random_state

    # Build dataset from combined files (two plateaus per file)
    buckets = collect_dataset_from_combined(
        FEATURE_DIR,
        bin_sec=BIN_SEC,
        include_angle_target=INCLUDE_ANGLE_TARGET,
        rms_win_samples=RMS_WIN,
        modes=("rms_matrix",),
    )

    for bin_len, bundle in buckets.items():
        X, y, groups = bundle["X"], bundle["y"], bundle["groups"]
        angles_per_row = bundle["angles"] if y.shape[1] < 3 else y[:, 2]

        out_dir = _resolve_path(os.path.join(OUT_ROOT, f"bin_{bin_len}"))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[RUN] bin_len={bin_len} -> X={X.shape}, y={y.shape} | unique groups={len(np.unique(groups))}")

        # --- Hold out exactly one plateau per *nominal* angle for test ---
        train_idx, test_idx, test_groups = split_one_plateau_per_angle_test(groups, rng=RNG)
        X_tr, y_tr, groups_tr = X[train_idx], y[train_idx], groups[train_idx]
        X_te, y_te, groups_te = X[test_idx],  y[test_idx],  groups[test_idx]

        # Cap CV folds to number of unique train groups
        n_train_groups = len(np.unique(groups_tr))
        cv_folds = min(CV_FOLDS, n_train_groups) if n_train_groups > 1 else 2

        # --- Channel modes to evaluate ---
        n_ch = X.shape[1]
        channel_modes = [
            {"name": "rms_matrix",       "mode": "rms_matrix"},
            {"name": "all_channels",     "mode": "all_channels"},
            {"name": "average_channels", "mode": "average_channels"},
            {"name": "single_ch_0",      "mode": "single_channel", "channels": [0]},
            {"name": "iterative_add",    "mode": "iterative_addition", "channels": list(range(n_ch))},
        ]

        # --- GroupKFold CV on train only (leakage-safe) ---
        results = run_modes(X_tr, y_tr, groups_tr, channel_modes, cv=cv_folds, use_scaler=USE_SCALER, scale_targets=SCALE_Y)
        save_results(results, str(out_dir), tag="cv_metrics_train")

        # --- Train on full train set and save models ---
        train_and_save_models(X_tr, y_tr, channel_modes, str(out_dir), use_scaler=USE_SCALER, scale_targets=SCALE_Y)

        # --- Optional: quick diagnostic CV plot (train set) ---
        plot_cv_reconstruction(
            X_tr, y_tr,
            cfg={"name": "rms_matrix", "mode": "rms_matrix"},
            out_path=str(out_dir / "cv_reconstruction_rms_matrix_train.png"),
            cv=cv_folds, use_scaler=USE_SCALER, scale_targets=SCALE_Y, bin_size=bin_len
        )

        # Save which groups were used
        with open(out_dir / "train_groups.txt", "w") as f:
            f.write("\n".join(map(str, sorted(np.unique(groups_tr)))))
        with open(out_dir / "test_groups.txt", "w") as f:
            f.write("\n".join(map(str, sorted(test_groups))))

        # Simple holdout description
        print(f"[HOLDOUT] test rows: {len(test_idx)} | groups: {len(np.unique(groups_te))}")

    print("\n[DONE] All bin sizes processed.")

if __name__ == "__main__":
    main()
