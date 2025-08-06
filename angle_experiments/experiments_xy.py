import os
import re
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
import h5py
import joblib
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from functools import lru_cache

# -------------------- Loading & Utilities --------------------

def load_mat_file(file_path):
    """
    Load a MATLAB file, handling both v7.3 (HDF5) and earlier formats.

    Returns:
        data_dict (dict): mapping of variable names to arrays or h5py objects
        h5file (h5py.File or None): open HDF5 file when v7.3, else None
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        data = loadmat(file_path)
        return data, None
    except NotImplementedError:
        f = h5py.File(file_path, 'r')
        data = {key: f[key] for key in f.keys()}
        return data, f

@lru_cache(maxsize=None)
def load_mat_cached(path):
    return load_mat_file(path)


def angle_from_fname(fname):
    """Parse the pull angle (in degrees) from a filename like '135Deg_...'."""
    m = re.search(r"(\d+)Deg", fname)
    if not m:
        raise ValueError(f"Cannot parse angle from filename: {fname}")
    return float(m.group(1))

# -------------------- Signal Processing --------------------

def bin_signal(sig, bin_size):
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    n_samples, n_ch = sig.shape
    n_bins = n_samples // bin_size
    return sig[:n_bins * bin_size].reshape(n_bins, bin_size, n_ch)


def compute_rms(binned):
    # channel-wise RMS and overall RMS
    rms_per_ch = np.sqrt(np.mean(binned**2, axis=1))
    rms_all = np.sqrt(np.mean(rms_per_ch**2, axis=1))
    return rms_per_ch, rms_all

# -------------------- Feature & Label Extraction --------------------

def extract_features(data_dict, bin_size=1000, mode='all_channels', channels=None, angle_deg=None):
    """
    Extract EMG features (X) and 2D force labels (Fx, Fy) from a single trial.

    Returns:
        X (np.ndarray or list of np.ndarray): feature matrix or list for iterative modes
        y (np.ndarray): shape (n_bins, 2) force components
        fs (float): sampling rate
    """
    # --- reference signal (force magnitude) ---
    ref_raw = data_dict['ref_signal']
    if isinstance(ref_raw, h5py.Dataset):
        ref = ref_raw[()].squeeze()
    else:
        ref = np.asarray(ref_raw).squeeze()

    # reconstruct Fx, Fy from magnitude + known angle
    if angle_deg is None:
        raise ValueError("angle_deg must be provided to reconstruct 2D force")
    theta = np.deg2rad(angle_deg)
    fx_raw = ref * np.cos(theta)
    fy_raw = ref * np.sin(theta)

    # bin both axes
    bx = bin_signal(fx_raw, bin_size)
    by = bin_signal(fy_raw, bin_size)
    y_x = np.mean(np.abs(bx), axis=1).squeeze()
    y_y = np.mean(np.abs(by), axis=1).squeeze()
    y = np.stack([y_x, y_y], axis=1)

    # --- EMG feature extraction (RMS) ---
    emg_raw = data_dict['SIG']
    # handle HDF5 / object arrays
    if isinstance(emg_raw, h5py.Group):
        arrays = [np.asarray(emg_raw[k][()]).squeeze() for k in sorted(emg_raw.keys())]
        emg = np.stack(arrays, axis=1)
    elif isinstance(emg_raw, h5py.Dataset):
        data = emg_raw[()]
        if data.dtype == 'object':
            f = data_dict.get('__h5file') or emg_raw.file
            arrays = []
            for refobj in data.flat:
                if isinstance(refobj, h5py.Reference):
                    arrays.append(np.asarray(f[refobj][()]).squeeze())
                else:
                    arrays.append(np.asarray(refobj).squeeze())
            emg = np.stack(arrays, axis=1)
        else:
            emg = data
    elif isinstance(emg_raw, (list, tuple)) or (isinstance(emg_raw, np.ndarray) and emg_raw.dtype == object):
        cells = emg_raw if isinstance(emg_raw, (list, tuple)) else emg_raw.flatten()
        arrays = [np.asarray(c).squeeze() for c in cells]
        emg = np.stack(arrays, axis=1)
    else:
        emg = np.asarray(emg_raw, dtype=float)

    # align lengths (simple trim to min length)
    n = min(len(ref), emg.shape[0])
    fx_raw, fy_raw = fx_raw[:n], fy_raw[:n]
    emg = emg[:n]
    fs = float(np.asarray(data_dict.get('fsamp', data_dict.get('srate', 1))).squeeze())

    # bin EMG
    bemg = bin_signal(emg, bin_size)
    rms_per_ch, rms_all = compute_rms(bemg)

    # select feature matrix
    if mode == 'all_channels':
        X = rms_all[:, None]
    elif mode == 'single_channel':
        X = rms_per_ch[:, channels[0]][:, None]
    elif mode == 'rms_matrix':
        X = rms_per_ch
    elif mode == 'average_channels':
        X = np.mean(rms_per_ch, axis=1)[:, None]
    elif mode == 'iterative_addition':
        sel = channels or list(range(rms_per_ch.shape[1]))
        X = [rms_per_ch[:, sel[:k]] for k in range(1, len(sel) + 1)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return X, y, fs

# -------------------- Aggregation --------------------

def aggregate_rms_xy(data_dir, subjects, bin_size=1000):
    """
    Aggregate feature matrices (R) and 2D labels (y) for all trials.
    """
    R_list, y_list = [], []
    for subj in subjects:
        subj_path = os.path.join(data_dir, subj)
        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith('.mat'):
                continue
            path = os.path.join(subj_path, fname)
            (data, h5f) = load_mat_file(path)
            angle = angle_from_fname(fname)
            R, y, _ = extract_features(data, bin_size=bin_size, mode='rms_matrix', angle_deg=angle)
            R_list.append(R)
            y_list.append(y)
            if h5f is not None:
                h5f.close()
    R_all = np.vstack(R_list)
    y_all = np.vstack(y_list)
    return R_all, y_all

# -------------------- Model Building --------------------

def build_X(R, mode, channels=None):
    if mode == 'all_channels':
        return np.sqrt(np.mean(R**2, axis=1))[:, None]
    elif mode == 'average_channels':
        return np.mean(R, axis=1)[:, None]
    elif mode == 'single_channel':
        return R[:, [channels[0]]]
    elif mode == 'rms_matrix':
        return R
    else:
        raise ValueError(f"Unknown mode for build_X: {mode}")


def run_experiment_xy(R, y, channel_modes, cv=5):
    results = {}
    scoring = {'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
    for cfg in channel_modes:
        name = cfg['name']; mode = cfg['mode']; chs = cfg.get('channels')
        if mode == 'iterative_addition':
            hist = []
            sel = chs or list(range(R.shape[1]))
            current = []
            for ch in sel:
                current.append(ch)
                Xk = R[:, current]
                model = MultiOutputRegressor(Ridge())
                mse = -np.mean(cross_val_score(model, Xk, y, cv=cv, scoring=scoring['mse']))
                mae = -np.mean(cross_val_score(model, Xk, y, cv=cv, scoring=scoring['mae']))
                hist.append({'channels': list(current), 'mse': mse, 'mae': mae})
            results[name] = hist
        else:
            X = build_X(R, mode, chs)
            model = MultiOutputRegressor(Ridge())
            mse = -np.mean(cross_val_score(model, X, y, cv=cv, scoring=scoring['mse']))
            mae = -np.mean(cross_val_score(model, X, y, cv=cv, scoring=scoring['mae']))
            results[name] = {'mse': mse, 'mae': mae}
    return results

# -------------------- Saving Results --------------------

def save_results_xy(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for mode, res in results.items():
        if isinstance(res, list):
            for entry in res:
                records.append({
                    'mode': mode,
                    'channels': ','.join(map(str, entry['channels'])),
                    'mse': entry['mse'],
                    'mae': entry['mae']
                })
        else:
            records.append({
                'mode': mode,
                'channels': None,
                'mse': res['mse'],
                'mae': res['mae']
            })
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'results_xy.csv'), index=False)

# -------------------- Main --------------------

def main():
    data_dir = "data"
    subjects = ["P1"]
    bin_size = 1000

    print("[MAIN] Aggregating data...")
    R_all, y_all = aggregate_rms_xy(data_dir, subjects, bin_size)

    # Normalize labels to [0,1]
    y_max = np.max(np.abs(y_all), axis=0)
    y_norm = y_all / y_max

    # Define channel modes
    n_ch = R_all.shape[1]
    channel_modes = [
        {'name': 'all_channels',       'mode': 'all_channels'},
        {'name': 'avg_channels',       'mode': 'average_channels'},
        {'name': 'single_channel',     'mode': 'single_channel',    'channels': [0]},
        {'name': 'iterative_addition', 'mode': 'iterative_addition','channels': list(range(n_ch))}
    ]

    print("[MAIN] Running experiments...")
    results = run_experiment_xy(R_all, y_norm, channel_modes)

    out_dir = os.path.join('results_xy')
    print(f"[MAIN] Saving results to {out_dir}")
    save_results_xy(results, out_dir)

    # Save final models
    for cfg in channel_modes:
        name = cfg['name']; mode = cfg['mode']; chs = cfg.get('channels')
        if mode == 'iterative_addition':
            current = []
            for ch in chs:
                current.append(ch)
                Xk = R_all[:, current]
                model = MultiOutputRegressor(Ridge()).fit(Xk, y_norm)
                fname = f"ridge_{name}_{'_'.join(map(str,current))}.joblib"
                joblib.dump(model, os.path.join(out_dir, fname))
        else:
            X = build_X(R_all, mode, chs)
            model = MultiOutputRegressor(Ridge()).fit(X, y_norm)
            fname = f"ridge_{name}.joblib"
            joblib.dump(model, os.path.join(out_dir, fname))

    print("[MAIN] Done.")

if __name__ == "__main__":
    main()