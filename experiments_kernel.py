#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# angle_experiments/experiments_kernel.py
#
# Pipeline: read *_combined.npy, preprocess into segments, bin, split by MCP-angle groups.
# Supports: Ridge, KernelRidge(RBF)
# Targets: fxfy, fxfy_achieved, achieved_angle, nominal_angle, achieved_sincos, nominal_sincos
#
# FIXES:
# - Safer y scaling per target (no MinMax on sin/cos; special handling for fxfy_achieved)
# - Optional projection of predicted sin/cos onto unit circle
# - Better debug prints to catch "collapse to zero-force" early
# - Saves run_meta.json per bin (target + scaler choices)

import os, glob, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from data.preprocessing.preprocess import preprocess_plateaus  # your preprocessor

# -------------------- Path helpers --------------------

SCRIPT_DIR = Path(__file__).resolve().parent

def _resolve_path(p: str) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (SCRIPT_DIR / q).resolve()

def _expand_combined_files(dir_or_glob: str):
    p = _resolve_path(dir_or_glob)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(glob.glob(str(p / "*.npy")))
    return sorted(glob.glob(str(p)))  # treat as glob

# -------------------- Binning --------------------

def _bin_series(arr, bin_len, agg="mean"):
    """Bin 1D/2D along axis=0 into chunks of length bin_len. Drops tail if not divisible."""
    A = np.asarray(arr)
    if A.ndim == 1:
        N = A.shape[0]
        B = max(1, N // bin_len)
        if N < bin_len:
            return A.reshape(1, -1).mean(axis=1)
        A2 = A[:B * bin_len].reshape(B, bin_len)
        return A2.mean(axis=1) if agg == "mean" else np.median(A2, axis=1)

    if A.ndim == 2:
        N, C = A.shape
        B = max(1, N // bin_len)
        if N < bin_len:
            return A.reshape(1, N, C).mean(axis=1)
        A2 = A[:B * bin_len].reshape(B, bin_len, C)
        return A2.mean(axis=1) if agg == "mean" else np.median(A2, axis=1)

    raise ValueError("arr must be 1D or 2D")

def _samples_per_bin_from_seconds(fs, bin_sec):
    return max(1, int(round(fs * float(bin_sec))))

# -------------------- Circular helpers --------------------

def _wrap_deg(a):
    return np.mod(a, 360.0)

def angle_to_sincos_deg(angle_deg):
    th = np.deg2rad(_wrap_deg(angle_deg))
    return np.stack([np.sin(th), np.cos(th)], axis=1)

def sincos_to_angle_deg(sin_cos_2d):
    s = sin_cos_2d[:, 0]
    c = sin_cos_2d[:, 1]
    ang = np.degrees(np.arctan2(s, c))
    return _wrap_deg(ang)

def circ_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0

def circ_mae_deg(a, b):
    return float(np.mean(np.abs(circ_diff_deg(a, b))))

def _project_to_unit_circle(sc2):
    sc2 = np.asarray(sc2, float)
    eps = 1e-9
    r = np.sqrt(sc2[:, 0] ** 2 + sc2[:, 1] ** 2) + eps
    return sc2 / r[:, None]

# -------------------- Dataset collector --------------------

def collect_dataset_from_combined(
    combined_dir_or_glob: str,
    bin_sec: float = 0.050,
    include_angle_target: bool = True,
    rms_win_samples: int = 100,
    modes=("rms_matrix", "all_channels", "average_channels", "iterative_addition"),
    single_channel_idx=None,
    iterative_channels=None,
    segment_kind: str = "plateau",
):
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

        fs = segs[0]["fs"] or (1.0 / np.median(np.diff(segs[0]["signals"]["t"])))
        bin_len = _samples_per_bin_from_seconds(fs, bin_sec)

        try:
            mcp_angle = int(payload.get("MCP_Angle", np.nan))
        except Exception:
            mcp_angle = np.nan

        for seg in segs:
            Fx = seg["signals"]["Fx"].reshape(-1)
            Fy = seg["signals"]["Fy"].reshape(-1)
            ang = seg["signals"]["angle_achieved"].reshape(-1)  # [0,360)

            R_native = seg["emg"]["rms_matrix"]
            N = min(R_native.shape[0], Fx.size, Fy.size, ang.size)
            Fx, Fy, ang = Fx[:N], Fy[:N], ang[:N]

            Fxb = _bin_series(Fx, bin_len, agg="mean").reshape(-1, 1)
            Fyb = _bin_series(Fy, bin_len, agg="mean").reshape(-1, 1)
            Angb = _bin_series(ang, bin_len, agg="mean").reshape(-1, 1)

            yb = np.hstack([Fxb, Fyb, Angb]) if include_angle_target else np.hstack([Fxb, Fyb])

            X_modes_binned = {}

            def _bin_and_trim(arr, B=None):
                out = _bin_series(arr, bin_len, agg="mean")
                if out.ndim == 1:
                    out = out.reshape(-1, 1)
                return out if B is None else out[:B, ...]

            Rb_ref = _bin_and_trim(R_native)
            B = min(Rb_ref.shape[0], yb.shape[0])

            for mode in set(modes or []):
                if mode not in seg["emg"]:
                    continue
                val = seg["emg"][mode]
                if mode == "iterative_addition":
                    X_modes_binned[mode] = [_bin_and_trim(v, B) for v in val]
                else:
                    X_modes_binned[mode] = _bin_and_trim(val, B)

            yb = yb[:B, :]
            Angb = Angb[:B, 0]

            group_label = f"{f}::{seg['name']}"
            groups_seg = np.full((B,), group_label, dtype=object)
            nom_seg = np.full((B,), mcp_angle, dtype=float)

            key = int(bin_len)
            if key not in buckets:
                buckets[key] = {
                    "X_by_mode": defaultdict(list),
                    "y": [],
                    "groups": [],
                    "angles": [],
                    "nominal_angles": [],
                    "files": [],
                }

            for mode, Xb in X_modes_binned.items():
                buckets[key]["X_by_mode"][mode].append(Xb)
            buckets[key]["y"].append(yb)
            buckets[key]["groups"].append(groups_seg)
            buckets[key]["angles"].append(Angb)
            buckets[key]["nominal_angles"].append(nom_seg)
            buckets[key]["files"].append(group_label)

        used_files += 1

    for key, pack in buckets.items():
        y = np.vstack(pack["y"]) if pack["y"] else np.empty((0, 0))
        groups = np.concatenate(pack["groups"]) if pack["groups"] else np.array([], dtype=object)
        angles = np.concatenate(pack["angles"]) if pack["angles"] else np.array([], dtype=float)
        nomang = np.concatenate(pack["nominal_angles"]) if pack["nominal_angles"] else np.array([], dtype=float)

        X_by_mode = {}
        for mode, chunks in pack["X_by_mode"].items():
            if mode == "iterative_addition":
                by_k = defaultdict(list)
                for seg_list in chunks:
                    for k_idx, arr in enumerate(seg_list):
                        by_k[k_idx].append(arr)
                X_by_mode[mode] = [np.vstack(lst) if lst else np.empty((0, 1)) for _, lst in sorted(by_k.items())]
            else:
                X_by_mode[mode] = np.vstack(chunks) if chunks else np.empty((0, 0))

        buckets[key] = {
            "X_by_mode": X_by_mode,
            "y": y,  # (N,3) Fx,Fy,angle_achieved
            "groups": groups,
            "angles": angles,          # binned achieved angle
            "nominal_angles": nomang,  # replicated MCP label
            "files": pack["files"],
        }

        print(f"[LOAD] bin_len={key}: X_by_mode keys={list(X_by_mode.keys())}, y={y.shape}, groups={len(np.unique(groups))}")

    print(f"[DONE] processed files: {used_files}/{len(files)}")
    return buckets

# -------------------- Grouped holdout --------------------

def split_one_plateau_per_angle_test(groups, nominal_angles, rng=42):
    """Pick exactly one plateau group per nominal angle for test."""
    rnd = np.random.RandomState(rng)
    group_to_rows = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_rows[g].append(i)

    group_nom = {}
    for g, idxs in group_to_rows.items():
        vals = np.asarray(nominal_angles)[idxs]
        group_nom[g] = float(np.nanmean(vals))

    angle_to_groups = defaultdict(list)
    for g, ang in group_nom.items():
        angle_to_groups[ang].append(g)

    test_groups = [rnd.choice(glist) for glist in angle_to_groups.values()]

    test_mask = np.zeros(len(groups), dtype=bool)
    for g in test_groups:
        test_mask[group_to_rows[g]] = True

    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx, test_groups

# -------------------- Target construction + y scaling --------------------

class _FxFyAngleScaler:
    """
    Columnwise scaling for y = [Fx, Fy, angle_deg]
    - Fx,Fy: StandardScaler
    - angle_deg: map to [-1,1] via (angle/180 - 1), then StandardScaler optional (we keep it identity)
    """
    def __init__(self):
        self.scaler_xy = StandardScaler()
        self.angle_mode = "deg_to_unit"  # internal marker

    def fit(self, y):
        y = np.asarray(y, float)
        self.scaler_xy.fit(y[:, :2])
        return self

    def transform(self, y):
        y = np.asarray(y, float)
        xy = self.scaler_xy.transform(y[:, :2])
        ang_unit = (np.mod(y[:, 2], 360.0) / 180.0) - 1.0  # [-1,1]
        return np.column_stack([xy, ang_unit])

    def inverse_transform(self, y_scaled):
        y_scaled = np.asarray(y_scaled, float)
        xy = self.scaler_xy.inverse_transform(y_scaled[:, :2])
        ang_unit = y_scaled[:, 2]
        ang_deg = np.mod((ang_unit + 1.0) * 180.0, 360.0)
        return np.column_stack([xy, ang_deg])

def build_target(TARGET, y_full, nominal_angles, achieved_angles):
    """
    Returns (y, target_kind) where target_kind describes how to evaluate.
    """
    TARGET = TARGET.lower()

    if TARGET == "fxfy":
        y = y_full[:, [0, 1]]
        return y, "fxfy"

    if TARGET == "fxfy_achieved":
        # NOTE: numeric mix → must use _FxFyAngleScaler (handled below)
        y = y_full[:, [0, 1, 2]]
        return y, "fxfy_achieved"

    if TARGET == "achieved_angle":
        y = achieved_angles.reshape(-1, 1).astype(float)
        return y, "angle_deg"

    if TARGET == "nominal_angle":
        y = nominal_angles.reshape(-1, 1).astype(float)
        return y, "angle_deg"

    if TARGET == "achieved_sincos":
        y = angle_to_sincos_deg(achieved_angles.astype(float))
        return y, "sincos"

    if TARGET == "nominal_sincos":
        y = angle_to_sincos_deg(nominal_angles.astype(float))
        return y, "sincos"

    raise ValueError(f"Unknown TARGET: {TARGET}")

def make_y_scaler(y_kind, y_scaler_choice):
    """
    Returns a fitted scaler object or None.
    y_scaler_choice in {"none","standard","fxfy_angle"}.
    """
    if y_kind == "sincos":
        # never scale sin/cos targets
        return None

    if y_scaler_choice == "none":
        return None

    if y_kind == "fxfy":
        if y_scaler_choice == "standard":
            return StandardScaler()
        raise ValueError("For y_kind=fxfy use y-scaler=standard or none.")

    if y_kind == "fxfy_achieved":
        # force the special scaler
        if y_scaler_choice in ("fxfy_angle", "standard"):
            return _FxFyAngleScaler()
        raise ValueError("For y_kind=fxfy_achieved use y-scaler=fxfy_angle (recommended).")

    if y_kind == "angle_deg":
        if y_scaler_choice == "standard":
            return StandardScaler()
        return None

    raise ValueError(f"Unhandled y_kind: {y_kind}")

# -------------------- Model builders --------------------

def make_estimator(model_name: str, krr_alpha: float, krr_gamma: float):
    model_name = model_name.lower()
    if model_name == "ridge":
        return Ridge()
    if model_name == "krr":
        return KernelRidge(alpha=float(krr_alpha), kernel="rbf", gamma=float(krr_gamma))
    raise ValueError(f"Unknown model: {model_name} (use 'ridge' or 'krr')")

def maybe_pipeline_x(est, x_scaler: str):
    if x_scaler == "standard":
        return Pipeline([("xscale", StandardScaler()), ("est", est)])
    if x_scaler == "none":
        return est
    raise ValueError("x-scaler must be 'standard' or 'none'")

# -------------------- Metrics --------------------

def eval_cv(X, y, groups, cv_folds, estimator):
    splitter = GroupKFold(n_splits=cv_folds)
    mse = -np.mean(cross_val_score(estimator, X, y, cv=splitter, groups=groups, scoring="neg_mean_squared_error"))
    mae = -np.mean(cross_val_score(estimator, X, y, cv=splitter, groups=groups, scoring="neg_mean_absolute_error"))
    return float(mse), float(mae)

# -------------------- main --------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--feature-dir", default="data/preprocessing/P5_combined")
    ap.add_argument("--kind", default="plateau")
    ap.add_argument("--out-root", default="./results_kernel/P5_kernel")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--bin-sec", type=float, default=0.050)
    ap.add_argument("--rms-win", type=int, default=100)
    ap.add_argument("--random-state", type=int, default=42)

    ap.add_argument("--x-scaler", choices=["standard", "none"], default="standard")
    # IMPORTANT: default changed away from minmax
    ap.add_argument("--y-scaler", choices=["none", "standard", "fxfy_angle"], default="standard",
                    help="Target scaling. For sin/cos it's forced to none. For fxfy_achieved use fxfy_angle.")

    ap.add_argument("--model", choices=["ridge", "krr"], default="krr")
    ap.add_argument("--krr-alpha", type=float, default=1.0)
    ap.add_argument("--krr-gamma", type=float, default=0.1)

    ap.add_argument("--target", default="fxfy", choices=[
        "fxfy",
        "fxfy_achieved",
        "achieved_angle",
        "nominal_angle",
        "achieved_sincos",
        "nominal_sincos",
    ])

    ap.add_argument("--modes", type=str, default="rms_matrix")
    ap.add_argument("--single-channel-idx", type=int, default=None)
    ap.add_argument("--iterative-channels", type=str, default=None)

    args = ap.parse_args()

    FEATURE_DIR = args.feature_dir
    KIND = args.kind
    OUT_ROOT = args.out_root
    CV_FOLDS = args.cv
    BIN_SEC = args.bin_sec
    RMS_WIN = args.rms_win
    RNG = args.random_state

    X_SCALER = args.x_scaler
    Y_SCALER = args.y_scaler
    MODEL = args.model
    KRR_ALPHA = args.krr_alpha
    KRR_GAMMA = args.krr_gamma
    TARGET = args.target

    it_channels = None
    if args.iterative_channels:
        it_channels = [int(s) for s in args.iterative_channels.split(",") if s.strip()]

    MODES_REQUESTED = tuple([m.strip() for m in args.modes.split(",") if m.strip()])

    buckets = collect_dataset_from_combined(
        FEATURE_DIR,
        bin_sec=BIN_SEC,
        include_angle_target=True,
        rms_win_samples=RMS_WIN,
        modes=MODES_REQUESTED,
        single_channel_idx=args.single_channel_idx,
        iterative_channels=it_channels,
        segment_kind=KIND,
    )

    for bin_len, bundle in buckets.items():
        X_by_mode = bundle["X_by_mode"]
        y_full = bundle["y"]  # (N,3)=Fx,Fy,achieved_angle
        groups = bundle["groups"]
        nominal_angles = bundle["nominal_angles"].astype(float)
        achieved_angles = bundle["angles"].astype(float)

        y, y_kind = build_target(TARGET, y_full, nominal_angles, achieved_angles)

        # enforce safe y-scaler rules
        if y_kind == "sincos":
            if Y_SCALER != "none":
                print("[WARN] forcing y-scaler=none for sin/cos target.")
            Y_SCALER_EFFECTIVE = "none"
        elif y_kind == "fxfy_achieved":
            # strongly recommend fxfy_angle
            if Y_SCALER not in ("fxfy_angle", "standard"):
                print("[WARN] forcing y-scaler=fxfy_angle for fxfy_achieved target.")
                Y_SCALER_EFFECTIVE = "fxfy_angle"
            else:
                Y_SCALER_EFFECTIVE = "fxfy_angle"
        else:
            Y_SCALER_EFFECTIVE = Y_SCALER

        out_dir = _resolve_path(os.path.join(OUT_ROOT, f"bin_{bin_len}"))
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[RUN] bin_len={bin_len} | target={TARGET} (kind={y_kind}) y={y.shape} | model={MODEL} | x={X_SCALER} y={Y_SCALER_EFFECTIVE}")
        print(f"      unique groups={len(np.unique(groups))}")

        # Holdout split
        train_idx, test_idx, test_groups = split_one_plateau_per_angle_test(groups, nominal_angles, rng=RNG)
        groups_tr = groups[train_idx]

        n_train_groups = len(np.unique(groups_tr))
        cv_folds = min(CV_FOLDS, n_train_groups) if n_train_groups > 1 else 2

        # Fit target scaler on train only
        y_tr, y_te = y[train_idx], y[test_idx]
        y_scaler = make_y_scaler(y_kind, Y_SCALER_EFFECTIVE)
        if y_scaler is not None:
            y_scaler.fit(y_tr)
            y_tr_s = y_scaler.transform(y_tr)
        else:
            y_tr_s = y_tr

        # Save meta
        meta = dict(
            bin_len=int(bin_len),
            target=TARGET,
            y_kind=y_kind,
            x_scaler=X_SCALER,
            y_scaler=Y_SCALER_EFFECTIVE,
            model=MODEL,
            krr_alpha=float(KRR_ALPHA),
            krr_gamma=float(KRR_GAMMA),
            n_train_groups=int(n_train_groups),
        )
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

        rows = []
        for mode_name, Xmat in X_by_mode.items():
            if mode_name == "iterative_addition":
                continue
            if mode_name not in MODES_REQUESTED:
                continue

            X_tr = Xmat[train_idx]
            X_te = Xmat[test_idx]

            est = make_estimator(MODEL, KRR_ALPHA, KRR_GAMMA)
            est = maybe_pipeline_x(est, X_SCALER)

            mse, mae = eval_cv(X_tr, y_tr_s, groups_tr, cv_folds, est)
            print(f"  [CV] mode={mode_name:>14s}  MSE={mse:.6f}  MAE={mae:.6f}")

            est.fit(X_tr, y_tr_s)
            yhat_te_s = np.asarray(est.predict(X_te))

            # invert y scaling if used
            if y_scaler is not None:
                try:
                    yhat_te = y_scaler.inverse_transform(yhat_te_s)
                except Exception:
                    yhat_te = yhat_te_s
            else:
                yhat_te = yhat_te_s

            # extra debug for "force collapse"
            if y_kind in ("fxfy", "fxfy_achieved"):
                fxhat, fyhat = yhat_te[:, 0], yhat_te[:, 1]
                print(f"       [DBG] pred Fx,Fy mean=({fxhat.mean():.4f},{fyhat.mean():.4f}) std=({fxhat.std():.4f},{fyhat.std():.4f}) "
                      f"| ||Fhat|| mean={np.mean(np.hypot(fxhat,fyhat)):.4f}")

            holdout_angle_mae = None
            if y_kind == "sincos":
                # project predictions to unit circle before converting
                yhat_sc = _project_to_unit_circle(yhat_te[:, :2])
                # y_te are already true sin/cos
                ang_pred = sincos_to_angle_deg(yhat_sc)
                ang_true = sincos_to_angle_deg(y_te[:, :2])
                holdout_angle_mae = circ_mae_deg(ang_pred, ang_true)
                print(f"       [HOLDOUT] sincos angle_MAE_deg={holdout_angle_mae:.3f} | mean_unit={np.mean(np.sqrt((yhat_sc**2).sum(1))):.3f}")

            rows.append({
                "bin_len": int(bin_len),
                "mode": mode_name,
                "cv_mse": mse,
                "cv_mae": mae,
                "holdout_angle_mae_deg": holdout_angle_mae,
            })

            joblib.dump(est, out_dir / f"{MODEL}_{mode_name}.joblib")

        # save y scaler
        if y_scaler is not None:
            joblib.dump(y_scaler, out_dir / "y_scaler.joblib")

        # save groups
        with open(out_dir / "train_groups.txt", "w") as f:
            f.write("\n".join(map(str, sorted(np.unique(groups_tr)))))
        with open(out_dir / "test_groups.txt", "w") as f:
            f.write("\n".join(map(str, sorted(test_groups))))

        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "metrics.csv", index=False)
        print(f"[SAVE] {out_dir / 'metrics.csv'}")

    print("\n[DONE] All bin sizes processed.")

if __name__ == "__main__":
    main()
