

# -------------------- Path helpers --------------------
# angle_experiments/experiments_xy.py
# Pipeline: read *_combined.npy, preprocess into segments, bin, then split by MCP angle groups.

import os
import glob
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

# -------------------- Preprocessor --------------------

from data.preprocessing.preprocess import preprocess_plateaus  # your preprocessor

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

# -------------------- Dataset collector --------------------

def collect_dataset_from_combined(
    combined_dir_or_glob: str,
    bin_sec: float = 0.050,            # 50 ms
    include_angle_target: bool = True,
    rms_win_samples: int = 100,        # RMS window on native grid
    modes=("rms_matrix", "all_channels", "average_channels", "iterative_addition"),
    single_channel_idx=None,
    iterative_channels=None,
    segment_kind: str = "plateau",
):
    """
    Returns dict keyed by bin_len (samples). For each bin_len:

      buckets[bin_len] = {
        "X_by_mode": {
           "rms_matrix":        (sum_B, C),
           "all_channels":      (sum_B, 1),
           "average_channels":  (sum_B, 1),
           "single_channel":    (sum_B, 1),              # if requested
           "iterative_addition": [ (sum_B,1), ... ]      # list over k
        },
        "y": (sum_B, 2/3),                 # Fx, Fy, [Angle]
        "groups": (sum_B,),                # 'path/file.npy::plateau01'
        "angles": (sum_B,),                # achieved angle (binned mean)
        "nominal_angles": (sum_B,),        # MCP_angle replicated per row
        "files": [...],
      }
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
            keep_intended_angle=False,
            segment_kind=segment_kind,
        )
        segs = pp.get("segments", [])
        if not segs:
            print(f"[SKIP] {Path(f).name} -> no segments found (kind={segment_kind})")
            continue

        # fs + bin
        fs = segs[0]["fs"] or (1.0 / np.median(np.diff(segs[0]["signals"]["t"])))
        bin_len = _samples_per_bin_from_seconds(fs, bin_sec)

        # nominal (intended) MCP angle for the trial
        try:
            mcp_angle = int(payload["MCP_Angle"])
        except Exception:
            mcp_angle = np.nan

        for seg in segs:
            # core signals from preprocessor
            Fx  = seg["signals"]["Fx"].reshape(-1)
            Fy  = seg["signals"]["Fy"].reshape(-1)
            ang = seg["signals"]["angle_achieved"].reshape(-1)  # [0,360)

            # defensive equalization on native grid (use rms_matrix as reference)
            R_native = seg["emg"]["rms_matrix"]
            N = min(R_native.shape[0], Fx.size, Fy.size, ang.size)
            Fx, Fy, ang = Fx[:N], Fy[:N], ang[:N]

            # --- bin y-targets
            Fxb  = _bin_series(Fx,  bin_len, agg="mean").reshape(-1, 1)
            Fyb  = _bin_series(Fy,  bin_len, agg="mean").reshape(-1, 1)
            Angb = _bin_series(ang, bin_len, agg="mean").reshape(-1, 1)
            yb   = np.hstack([Fxb, Fyb, Angb]) if include_angle_target else np.hstack([Fxb, Fyb])

            # --- bin each requested EMG mode from seg["emg"]
            X_modes_binned = {}

            # Helper: bin a mode array with the same logic and trim to B
            def _bin_and_trim(arr, B=None):
                out = _bin_series(arr, bin_len, agg="mean")
                if out.ndim == 1:
                    out = out.reshape(-1, 1)
                return out if B is None else out[:B, ...]

            # First, compute B by binning one reference (rms_matrix)
            Rb_ref = _bin_and_trim(R_native)
            B = min(Rb_ref.shape[0], yb.shape[0])

            # Now fill modes
            for mode in set(modes or []):
                if mode not in seg["emg"]:
                    continue
                val = seg["emg"][mode]
                if mode == "iterative_addition":
                    # list of (N,1) -> list of (B,1)
                    X_modes_binned[mode] = [ _bin_and_trim(v, B) for v in val ]
                else:
                    X_modes_binned[mode] = _bin_and_trim(val, B)

            # final trim of y + angles to B
            yb   = yb[:B, :]
            Angb = Angb[:B, 0]

            # group label + MCP vector
            group_label = f"{f}::{seg['name']}"
            groups_seg  = np.full((B,), group_label, dtype=object)
            nom_seg     = np.full((B,), mcp_angle, dtype=float)

            key = int(bin_len)
            if key not in buckets:
                buckets[key] = {
                    "X_by_mode": defaultdict(list),
                    "y": [], "groups": [], "angles": [], "nominal_angles": [], "files": []
                }

            # append per-mode
            for mode, Xb in X_modes_binned.items():
                buckets[key]["X_by_mode"][mode].append(Xb)
            buckets[key]["y"].append(yb)
            buckets[key]["groups"].append(groups_seg)
            buckets[key]["angles"].append(Angb)
            buckets[key]["nominal_angles"].append(nom_seg)
            buckets[key]["files"].append(group_label)

        used_files += 1

    # Stack lists
    for key, pack in buckets.items():
        # stack y, groups, angles, nominal
        y = np.vstack(pack["y"]) if pack["y"] else np.empty((0, 0))
        groups = np.concatenate(pack["groups"]) if pack["groups"] else np.array([], dtype=object)
        angles = np.concatenate(pack["angles"]) if pack["angles"] else np.array([], dtype=float)
        nomang = np.concatenate(pack["nominal_angles"]) if pack["nominal_angles"] else np.array([], dtype=float)

        # stack each mode
        X_by_mode = {}
        for mode, chunks in pack["X_by_mode"].items():
            if mode == "iterative_addition":
                # chunks is a list over segments; each item is a list over k -> we need to stack per k
                # Collect by k index first
                by_k = defaultdict(list)
                for seg_list in chunks:  # seg_list is [ (B,1), (B,1), ... ]
                    for k_idx, arr in enumerate(seg_list):
                        by_k[k_idx].append(arr)
                # stack each k
                X_by_mode[mode] = [ np.vstack(lst) if lst else np.empty((0,1)) for k_idx, lst in sorted(by_k.items()) ]
            else:
                X_by_mode[mode] = np.vstack(chunks) if chunks else np.empty((0, 0))

        buckets[key] = {
            "X_by_mode": X_by_mode,
            "y": y,
            "groups": groups,
            "angles": angles,
            "nominal_angles": nomang,
            "files": pack["files"],
        }

        approx_sec = (key / _samples_per_bin_from_seconds(1.0, 1.0))  # just placeholder for printing
        print(f"[LOAD] bin_len={key}: X_by_mode keys={list(X_by_mode.keys())}, y={y.shape}, groups={len(np.unique(groups))}")

    print(f"[DONE] processed files: {used_files}/{len(files)}")
    return buckets

# -------------------- Group-aware CV helpers --------------------

def cv_scores_multi(X, y, groups=None, cv=5, use_scaler=True):
    model = MultiOutputRegressor(Ridge())
    if use_scaler:
        model = Pipeline([("xscale", StandardScaler()), ("ridge", model)])
    splitter = GroupKFold(n_splits=cv) if groups is not None else cv
    mse = -np.mean(cross_val_score(model, X, y, cv=splitter, groups=groups, scoring="neg_mean_squared_error"))
    mae = -np.mean(cross_val_score(model, X, y, cv=splitter, groups=groups, scoring="neg_mean_absolute_error"))
    return mse, mae

def run_modes(X_by_mode, y, groups, modes_to_eval, cv=5, use_scaler=True, scale_targets=True):
    # optionally scale targets (shared for all modes)
    if scale_targets:
        y_scaler = MinMaxScaler()
        y_used = y_scaler.fit_transform(y)
    else:
        y_scaler = None
        y_used = y

    results = {}
    for entry in modes_to_eval:
        name = entry["name"]
        mode = entry["mode"]
        print(f"[EXP] {name} ({mode})")

        if mode == "iterative_addition":
            X_list = X_by_mode.get("iterative_addition", [])
            mses, maes = [], []
            for k, Xk in enumerate(X_list, start=1):
                mse, mae = cv_scores_multi(Xk, y_used, groups=groups, cv=cv, use_scaler=use_scaler)
                mses.append(mse); maes.append(mae)
                print(f"      k={k}: MSE={mse:.4f}, MAE={mae:.4f}")
            results[name] = {"mse": mses, "mae": maes}
        else:
            Xd = X_by_mode.get(mode)
            if Xd is None:
                print(f"      [WARN] Mode '{mode}' not available; skipping.")
                continue
            mse, mae = cv_scores_multi(Xd, y_used, groups=groups, cv=cv, use_scaler=use_scaler)
            print(f"      MSE={mse:.4f}, MAE={mae:.4f}")
            results[name] = {"mse": mse, "mae": mae}
    return results

# -------------------- Train & save --------------------

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


def train_and_save_models(X_by_mode, y, modes_to_train, out_dir, use_scaler=True, scale_targets=True):
    os.makedirs(out_dir, exist_ok=True)
    y_scaler = MinMaxScaler() if scale_targets else None
    y_train = y_scaler.fit_transform(y) if y_scaler else y

    saved = []
    for entry in modes_to_train:
        name, mode = entry["name"], entry["mode"]

        if mode == "iterative_addition":
            X_list = X_by_mode.get("iterative_addition", [])
            for k, Xk in enumerate(X_list, start=1):
                base = MultiOutputRegressor(Ridge())
                model = Pipeline([("xscale", StandardScaler()), ("ridge", base)]) if use_scaler else base
                model.fit(Xk, y_train)
                path = os.path.join(out_dir, f"ridge_{name}_k{k}.joblib")
                joblib.dump(model, path); saved.append(path)
        else:
            Xd = X_by_mode.get(mode)
            if Xd is None:
                print(f"[WARN] Mode '{mode}' not available; skipping train.")
                continue
            base = MultiOutputRegressor(Ridge())
            model = Pipeline([("xscale", StandardScaler()), ("ridge", base)]) if use_scaler else base
            model.fit(Xd, y_train)
            path = os.path.join(out_dir, f"ridge_{name}.joblib")
            joblib.dump(model, path); saved.append(path)

    if y_scaler:
        joblib.dump(y_scaler, os.path.join(out_dir, "y_scaler.joblib"))
    print(f"[SAVE] Saved {len(saved)} models into {out_dir}")
    return saved

# -------------------- Diagnostic plot (works for 2-dim targets only) --------------------

def plot_cv_reconstruction(Xd, y, cfg, out_path, cv=5, use_scaler=True, scale_targets=True, bin_size=None):
    name, mode = cfg["name"], cfg["mode"]

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

# -------------------- Grouped holdout via MCP_angle --------------------

def split_one_plateau_per_angle_test(groups, nominal_angles, rng=42):
    """
    Pick exactly one plateau group per nominal angle (from MCP_angle).
    Inputs are aligned 1:1 per-row arrays: groups, nominal_angles.
    """
    rnd = np.random.RandomState(rng)
    # rows per group
    group_to_rows = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_rows[g].append(i)

    # nominal angle per group (they should all be identical per group)
    group_nom = {}
    for g, idxs in group_to_rows.items():
        vals = np.asarray(nominal_angles)[idxs]
        group_nom[g] = float(np.nanmean(vals))

    # bucket groups by nominal angle
    angle_to_groups = defaultdict(list)
    for g, ang in group_nom.items():
        angle_to_groups[ang].append(g)

    # pick one group per nominal angle for test
    test_groups = [rnd.choice(glist) for glist in angle_to_groups.values()]

    # mask rows
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
    ap.add_argument("--kind", default="plateau", help="Segment kind to use from 'cuts' entries (e.g., 'plateau', 'ramp')")
    ap.add_argument("--out-root", default="./results_feat/P5_angle")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--bin-sec", type=float, default=0.050, help="Bin size in seconds (e.g., 0.05)")
    ap.add_argument("--rms-win", type=int, default=100, help="RMS window (samples) on native time grid")
    ap.add_argument("--no-x-scaler", action="store_true", help="Disable StandardScaler on X")
    ap.add_argument("--no-y-scaler", action="store_true", help="Disable MinMaxScaler on y")
    ap.add_argument("--no-angle-target", action="store_true", help="Do not include angle as 3rd output")
    ap.add_argument("--random-state", type=int, default=42)
    # optional: if you want single_channel or a custom channel order for iterative
    ap.add_argument("--single-channel-idx", type=int, default=None)
    ap.add_argument("--iterative-channels", type=str, default=None, help="comma-separated indices, e.g. 0,1,2,3")
    args = ap.parse_args()

    FEATURE_DIR = args.feature_dir
    KIND = args.kind
    OUT_ROOT    = args.out_root
    CV_FOLDS    = args.cv
    BIN_SEC     = args.bin_sec
    RMS_WIN     = args.rms_win
    USE_SCALER  = not args.no_x_scaler
    SCALE_Y     = not args.no_y_scaler
    INCLUDE_ANGLE_TARGET = not args.no_angle_target
    RNG         = args.random_state

    it_channels = None
    if args.iterative_channels:
        it_channels = [int(s) for s in args.iterative_channels.split(",") if s.strip()]

    MODES_REQUESTED = ("rms_matrix", "all_channels", "average_channels", "iterative_addition")

    # Build dataset from combined files (once; includes all requested modes)
    buckets = collect_dataset_from_combined(
        FEATURE_DIR,
        bin_sec=BIN_SEC,
        include_angle_target=INCLUDE_ANGLE_TARGET,
        rms_win_samples=RMS_WIN,
        modes=MODES_REQUESTED,
        single_channel_idx=args.single_channel_idx,
        iterative_channels=it_channels,
        segment_kind=KIND,
    )

    for bin_len, bundle in buckets.items():
        X_by_mode = bundle["X_by_mode"]
        y, groups = bundle["y"], bundle["groups"]
        nominal_angles = bundle["nominal_angles"]

        ################################################
        # TARGET CHOICES (select one)
        ################################################
        # Angle-only   (Fx=0, Fy=1, Angle=2)
        y = y[:, [2]]
        # y = y[:, [0]] # Fx-only
        # y = y[:, [1]] # Fy-only
        ################################################

        out_dir = _resolve_path(os.path.join(OUT_ROOT, f"bin_{bin_len}"))
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[RUN] bin_len={bin_len} -> y={y.shape} | unique groups={len(np.unique(groups))}")

        # --- Hold out exactly one plateau per nominal (MCP) angle for test ---
        train_idx, test_idx, test_groups = split_one_plateau_per_angle_test(groups, nominal_angles, rng=RNG)
        y_tr, y_te = y[train_idx], y[test_idx]
        groups_tr, groups_te = groups[train_idx], groups[test_idx]

        # Prepare X_by_mode splits
        X_by_mode_tr, X_by_mode_te = {}, {}
        for mode, X in X_by_mode.items():
            if mode == "iterative_addition":
                X_by_mode_tr[mode] = [ Xk[train_idx] for Xk in X ]
                X_by_mode_te[mode] = [ Xk[test_idx]  for Xk in X ]
            else:
                X_by_mode_tr[mode] = X[train_idx]
                X_by_mode_te[mode] = X[test_idx]

        # Cap CV folds to number of unique train groups
        n_train_groups = len(np.unique(groups_tr))
        cv_folds = min(CV_FOLDS, n_train_groups) if n_train_groups > 1 else 2

        # --- Modes to evaluate (match keys we built) ---
        modes_to_eval = [
            {"name": "rms_matrix",       "mode": "rms_matrix"},
            {"name": "all_channels",     "mode": "all_channels"},
            {"name": "average_channels", "mode": "average_channels"},
            {"name": "iterative_add",    "mode": "iterative_addition"},
        ]

        # --- GroupKFold CV on train only (leakage-safe) ---
        results = run_modes(X_by_mode_tr, y_tr, groups_tr, modes_to_eval, cv=cv_folds, use_scaler=USE_SCALER, scale_targets=SCALE_Y)
        save_results(results, str(out_dir), tag="cv_metrics_train")

        # --- Train on full train set and save models ---
        train_and_save_models(X_by_mode_tr, y_tr, modes_to_eval, str(out_dir), use_scaler=USE_SCALER, scale_targets=SCALE_Y)

        # --- Optional: quick diagnostic CV plot (only for non-iterative mode and 2-D Fx/Fy targets) ---
        # Example (uncomment to use with Fx/Fy targets):
        # plot_cv_reconstruction(
        #     X_by_mode_tr["rms_matrix"], y_tr,
        #     cfg={"name": "rms_matrix", "mode": "rms_matrix"},
        #     out_path=str(out_dir / "cv_reconstruction_rms_matrix_train.png"),
        #     cv=cv_folds, use_scaler=USE_SCALER, scale_targets=SCALE_Y, bin_size=bin_len
        # )

        # Save which groups were used
        with open(out_dir / "train_groups.txt", "w") as f:
            f.write("\n".join(map(str, sorted(np.unique(groups_tr)))))
        with open(out_dir / "test_groups.txt", "w") as f:
            f.write("\n".join(map(str, sorted(test_groups))))

        print(f"[HOLDOUT] test rows: {len(test_idx)} | groups: {len(np.unique(groups_te))}")

    print("\n[DONE] All bin sizes processed.")

if __name__ == "__main__":
    main()
