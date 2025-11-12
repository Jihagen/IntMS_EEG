import re, glob, math
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

from pathlib import Path
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from data.preprocessing.preprocess import preprocess_plateaus  # preprocessor
from experiments_xy import (
    collect_dataset_from_combined,   # builds X,y,groups from *_combined.npy 
) 


def _load_metrics_for_bin(bin_dir: Path, metric_col: str,
    fallback: str = "mae_norm_overall",) -> pd.DataFrame:
    f = bin_dir / "cv_metrics_test.csv"
    if not f.exists():
        raise FileNotFoundError(f"Missing cv_metrics_test.csv in {bin_dir}")
    df = pd.read_csv(f)
    if metric_col not in df.columns:
        # graceful fallback if someone saved only MAE
        fallback = "mae_norm_overall"
        if fallback in df.columns:
            print(f"[WARN] {bin_dir.name}: '{metric_col}' missing; using '{fallback}'")
            df[metric_col] = df[fallback]
        else:
            raise KeyError(f"{bin_dir}: neither '{metric_col}' nor '{fallback}' present.")
    return df

def _expand_combined_files(dir_or_glob):
    p = Path(dir_or_glob)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(glob.glob(str(p / "*_combined.npy")))
    return sorted(glob.glob(str(p)))

def _bin_series(arr, bin_len, agg="mean"):
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
    combined_dir_or_glob,
    bin_sec=0.050,
    include_angle_target=True,
    rms_win_samples=100,
    modes=("rms_matrix",),
):
    files = _expand_combined_files(combined_dir_or_glob)
    print(f"[PATH] matched files: {len(files)}")
    buckets = {}
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
            keep_intended_angle=False,
            segment_kind="ramp"
        )
        segs = pp.get("segments", [])
        if not segs:
            print(f"[SKIP] {Path(f).name} -> no plateaus found")
            continue

        fs = segs[0]["fs"] or (1.0 / np.median(np.diff(segs[0]["signals"]["t"])))
        bin_len = _samples_per_bin_from_seconds(fs, bin_sec)

        for seg in segs:
            R = seg["emg"]["rms_matrix"]             # (N,C)
            Fx = seg["signals"]["Fx"].reshape(-1)
            Fy = seg["signals"]["Fy"].reshape(-1)
            N = min(R.shape[0], Fx.size, Fy.size)
            R, Fx, Fy = R[:N, :], Fx[:N], Fy[:N]
            ang_ts = np.degrees(np.arctan2(Fy, Fx))
            ang_ts = np.where(ang_ts < 0, ang_ts + 360.0, ang_ts)

            Rb   = _bin_series(R,   bin_len, agg="mean")           # (B,C)
            Fxb  = _bin_series(Fx,  bin_len, agg="mean").reshape(-1,1)
            Fyb  = _bin_series(Fy,  bin_len, agg="mean").reshape(-1,1)
            Angb = _bin_series(ang_ts, bin_len, agg="mean").reshape(-1,1)
            yb   = np.hstack([Fxb, Fyb, Angb]) if include_angle_target else np.hstack([Fxb, Fyb])

            B = min(Rb.shape[0], yb.shape[0])
            Rb, yb, Angb = Rb[:B, :], yb[:B, :], Angb[:B, 0]

            group_label = f"{f}::{seg['name']}"
            key = int(bin_len)
            buckets.setdefault(key, {"X": [], "y": [], "groups": [], "angles": []})
            buckets[key]["X"].append(Rb)
            buckets[key]["y"].append(yb)
            buckets[key]["groups"].append(np.full((B,), group_label, dtype=object))
            buckets[key]["angles"].append(Angb)

    # stack
    for key in list(buckets.keys()):
        X = np.vstack(buckets[key]["X"]) if buckets[key]["X"] else np.empty((0,0))
        y = np.vstack(buckets[key]["y"]) if buckets[key]["y"] else np.empty((0,0))
        groups = np.concatenate(buckets[key]["groups"]) if buckets[key]["groups"] else np.array([], dtype=object)
        angles = np.concatenate(buckets[key]["angles"]) if buckets[key]["angles"] else np.array([], dtype=float)
        buckets[key]["X"], buckets[key]["y"], buckets[key]["groups"], buckets[key]["angles"] = X, y, groups, angles
        print(f"[LOAD] bin_len={key}: X={X.shape}, y={y.shape}, groups={len(np.unique(groups))}")
    return buckets

def nominal_from_group(g):
    src = g.split("::", 1)[0]
    return nominal_angle_from_path(src)

def nominal_angle_from_path(p: str) -> float:
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
    rnd = np.random.RandomState(rng)
   
    group_to_rows = defaultdict(list)
    for i, g in enumerate(groups):
        group_to_rows[g].append(i)
    group_to_rows = {g: np.asarray(ix, dtype=int) for g, ix in group_to_rows.items()}

    angle_to_groups = defaultdict(list)
    for g in group_to_rows:
        src_path = g.split("::", 1)[0]
        ang_nom = nominal_angle_from_path(src_path)
        angle_to_groups[ang_nom].append(g)

    test_groups = [rnd.choice(glist) for glist in angle_to_groups.values()]
    test_mask = np.zeros(len(groups), dtype=bool)
    for g in test_groups:
        test_mask[group_to_rows[g]] = True
    test_idx  = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx, test_groups


def _find_bin_dirs(base: Path):
    bins = {}
    for p in base.iterdir():
        if p.is_dir():
            m = re.match(r"bin_(\d+)$", p.name)
            if m:
                bins[int(m.group(1))] = p
    return dict(sorted(bins.items()))

def _plot_line(model_key: str, label: str):
    sub = df_plot[df_plot["model"] == model_key].copy()
    y = [float(sub.loc[sub["bin_size"] == bs, METRIC_COL].iloc[0]) if not sub.loc[sub["bin_size"] == bs, METRIC_COL].empty else np.nan
         for bs in bin_sizes_sorted]
    plt.plot(bin_sizes_sorted, y, marker="o", label=label)
    return y



# ===== helpers (angles, errors, mags) =====
def angle_deg(fx, fy):
    a = np.degrees(np.arctan2(fy, fx))
    return np.where(a < 0, a + 360.0, a)

def angle_err_deg(a_true, a_pred):
    return (a_pred - a_true + 180) % 360 - 180  # wrap to [-180,180]

def mag(fx, fy):
    return np.sqrt(fx**2 + fy**2)

# ===== plotting helpers =====
def plot_angle_error_hist(y_true, y_pred, bins=50, title="Angle error histogram"):
    a_t = angle_deg(y_true[:,0], y_true[:,1])
    a_p = angle_deg(y_pred[:,0], y_pred[:,1])
    err = angle_err_deg(a_t, a_p)
    mae = np.mean(np.abs(err))
    plt.figure(figsize=(7,3.5)); plt.hist(err, bins=bins)
    plt.xlabel("Angle error (deg)"); plt.ylabel("Count")
    plt.title(f"{title} — MAE={mae:.2f}°"); plt.grid(alpha=0.3); plt.tight_layout()

def plot_polar_all(y_true, y_pred, title="Polar: vectors true vs pred"):
    r_t, th_t = mag(y_true[:,0], y_true[:,1]), np.radians(angle_deg(y_true[:,0], y_true[:,1]))
    r_p, th_p = mag(y_pred[:,0], y_pred[:,1]), np.radians(angle_deg(y_pred[:,0], y_pred[:,1]))
    plt.figure(figsize=(5.8,5.8)); ax = plt.subplot(111, projection='polar')
    ax.scatter(th_t, r_t, s=10, label="true", alpha=0.85)
    ax.scatter(th_p, r_p, s=10, label="pred_from_vec", alpha=0.75)
    ax.set_title(title); ax.legend(loc="upper right"); plt.tight_layout()

def plot_true_vs_pred_angle_scatter(y_true, y_pred, title="All samples"):
    a_t = angle_deg(y_true[:,0], y_true[:,1]); a_p = angle_deg(y_pred[:,0], y_pred[:,1])
    plt.figure(figsize=(5.8,5.8)); lim_min, lim_max = 0, 360
    plt.plot([lim_min, lim_max], [lim_min, lim_max], lw=2, alpha=0.5)
    plt.scatter(a_t, a_p, s=10, alpha=0.7)
    plt.xlabel("True angle (deg)"); plt.ylabel("Pred angle (deg)")
    plt.title(title); plt.grid(alpha=0.3); plt.xlim(70, 225); plt.ylim(70, 225); plt.tight_layout()

def plot_mae_by_label(y_true, y_pred, labels_deg, title="Angle MAE by label"):
    a_t = angle_deg(y_true[:,0], y_true[:,1]); a_p = angle_deg(y_pred[:,0], y_pred[:,1])
    labs = np.unique(labels_deg); maes = []
    for L in labs:
        ix = np.where(labels_deg == L)[0]
        err = angle_err_deg(a_t[ix], a_p[ix]); maes.append(np.mean(np.abs(err)))
    plt.figure(figsize=(6.5,3.8)); plt.bar(labs.astype(float), maes, width=8)
    plt.xlabel("Angle (deg)"); plt.ylabel("MAE (deg)")
    plt.title(title); plt.grid(axis='y', alpha=0.3); plt.tight_layout()

def plot_polar_by_label(
    y_true, y_pred, labels_deg, title="Polar vectors by label",
    pred_direct_deg=None
):
    """
    If pred_direct_deg (deg) is provided, plot it as a third series using the
    same radius as |pred_from_vec| (so you can compare directions directly).
    """
    labs = np.unique(labels_deg); n = len(labs)
    fig, axes = plt.subplots(1, n, subplot_kw={'projection':'polar'}, figsize=(5.2*n, 4.8))
    if n == 1: axes = [axes]
    r_pred_vec = mag(y_pred[:,0], y_pred[:,1])
    th_pred_vec = np.radians(angle_deg(y_pred[:,0], y_pred[:,1]))
    th_true     = np.radians(angle_deg(y_true[:,0], y_true[:,1]))

    for ax, L in zip(axes, labs):
        ix = np.where(labels_deg == L)[0]
        ax.scatter(th_true[ix],     mag(y_true[ix,0], y_true[ix,1]), s=10, label="true", alpha=0.9)
        ax.scatter(th_pred_vec[ix], r_pred_vec[ix],                  s=10, label="pred_from_vec", alpha=0.75)
        if pred_direct_deg is not None:
            th_pred_dir = np.radians(pred_direct_deg[ix])
            ax.scatter(th_pred_dir, r_pred_vec[ix], marker="+", s=20, label="pred_direct", alpha=0.9)
        ax.set_title(f"{L}°"); ax.legend(loc="upper right")
    fig.suptitle(f"{title}", y=1.02); plt.tight_layout()

def plot_mean_vectors_by_label(
    y_true, y_pred, labels_deg, title="Mean vectors per label",
    pred_direct_deg=None
):
    """
    If pred_direct_deg is provided, also draw a mean ray for pred_direct,
    using the mean predicted magnitude for its length.
    """
    labs = np.unique(labels_deg); plt.figure(figsize=(6,6)); ax = plt.subplot(111, projection='polar')
    for L in labs:
        ix = (labels_deg == L)

        # true mean ray
        fx_t, fy_t = y_true[ix,0].mean(), y_true[ix,1].mean()
        rt, thetat = mag(fx_t, fy_t), np.radians(angle_deg(fx_t, fy_t))
        ax.plot([thetat, thetat], [0, rt], lw=3, label=f"{int(L)}° true")

        # pred_from_vec mean ray
        fx_p, fy_p = y_pred[ix,0].mean(), y_pred[ix,1].mean()
        rp, thetap = mag(fx_p, fy_p), np.radians(angle_deg(fx_p, fy_p))
        ax.plot([thetap, thetap], [0, rp], lw=3, ls='--', label=f"{int(L)}° pred_from_vec")

        # pred_direct mean ray (direction = mean of angles; radius = mean |pred_from_vec|)
        if pred_direct_deg is not None:
            # mean direction on a circle: use circular mean
            th = np.radians(pred_direct_deg[ix])
            th_mean = np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th)))
            if th_mean < 0: th_mean += 2*np.pi
            rp_dir = float(np.mean(mag(y_pred[ix,0], y_pred[ix,1])))
            ax.plot([th_mean, th_mean], [0, rp_dir], lw=3, ls='-.', label=f"{int(L)}° pred_direct")

    ax.set_title(f"{title}"); ax.legend(loc="upper right", ncol=2, fontsize=9); plt.tight_layout()

def plot_error_vs_magnitude(y_true, y_pred, title="Error vs magnitude (hexbin)"):
    a_t = angle_deg(y_true[:,0], y_true[:,1]); a_p = angle_deg(y_pred[:,0], y_pred[:,1])
    err = np.abs(angle_err_deg(a_t, a_p)); m = mag(y_true[:,0], y_true[:,1])
    plt.figure(figsize=(6.4,4.1))
    hb = plt.hexbin(m, err, gridsize=40, mincnt=1)
    plt.colorbar(hb, label="count"); plt.xlabel("|F| true"); plt.ylabel("|angle error| (deg)")
    plt.title(title); plt.grid(alpha=0.2); plt.tight_layout()

def full_plot_suite(
    y_true, y_pred, labels_deg, model_name="rms_matrix",
    pred_direct_deg=None
):
    plot_angle_error_hist(y_true, y_pred, title=f"Angle error histogram — {model_name}")
    plot_polar_all(y_true, y_pred, title=f"Polar: vectors true vs pred — {model_name}")
    plot_mae_by_label(y_true, y_pred, labels_deg, title=f"Angle MAE by label — model: {model_name}")
    plot_true_vs_pred_angle_scatter(y_true, y_pred, title=f"All samples • {model_name}")
    plot_polar_by_label(y_true, y_pred, labels_deg, title=f"Polar vectors by label • model: {model_name}",
                        pred_direct_deg=pred_direct_deg)
    plot_mean_vectors_by_label(y_true, y_pred, labels_deg,
                               title=f"Mean vectors per label • model: {model_name}",
                               pred_direct_deg=pred_direct_deg)
    plot_error_vs_magnitude(y_true, y_pred, title=f"Error vs magnitude • {model_name}")