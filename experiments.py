import os
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# -- Loading and Alignment -------------------------------------------------

def load_mat_file(file_path):
    """
    Load a MATLAB file, handling v7.3 (HDF5) and earlier formats.

    Returns a dict mapping variable names to numpy arrays or HDF5 objects,
    and stores the HDF5 File handle under '__h5file' for dereferencing.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        data = loadmat(file_path)
        print(f"Loaded non-v7.3 file: {file_path}")
        return data
    except NotImplementedError:
        f = h5py.File(file_path, 'r')
        print(f"Loaded v7.3 file: {file_path}")
        data = {key: f[key] for key in f.keys()}
        data['__h5file'] = f
        return data


def check_and_align_signals(ref, emg, data_dict):
    ref = np.asarray(ref).squeeze()
    emg = np.asarray(emg)
    # Transpose if needed
    if emg.ndim == 2 and ref.shape[0] == emg.shape[1] and ref.shape[0] != emg.shape[0]:
        print(f"Transposing EMG from {emg.shape} to {emg.T.shape}")
        emg = emg.T
    if ref.shape[0] != emg.shape[0]:
        raise ValueError(f"Length mismatch: ref has {ref.shape[0]} samples, EMG has {emg.shape[0]}")
    if np.all(ref == 0) or np.all(emg == 0):
        raise ValueError("Zero-variance signal detected.")
    fs = data_dict.get('force_fs') or data_dict.get('emg_fs') or data_dict.get('fs') or data_dict.get('srate')
    if isinstance(fs, (list, np.ndarray)):
        fs = float(fs[0])
    summed = np.sum(emg, axis=1)
    try:
        xc = np.correlate(summed - summed.mean(), ref - ref.mean(), mode='full')
        lag = xc.argmax() - (len(ref) - 1)
        if lag != 0:
            print(f"Aligning signals by shifting EMG {lag} samples")
            if lag > 0:
                ref, emg = ref[lag:], emg[:-lag]
            else:
                ref, emg = ref[:lag], emg[-lag:]
    except Exception:
        pass
    return ref, emg, fs

# -- Feature Extraction ----------------------------------------------------

def bin_signal(signal, bin_size):
    sig = np.asarray(signal)
    if sig.ndim == 1:
        sig = sig[:, None]
    n_samples, n_ch = sig.shape
    n_bins = n_samples // bin_size
    trimmed = sig[:n_bins * bin_size]
    return trimmed.reshape(n_bins, bin_size, n_ch)


def compute_rms(binned):
    rms_per_ch = np.sqrt(np.mean(binned**2, axis=1))
    rms_all = np.sqrt(np.mean(rms_per_ch**2, axis=1))
    return rms_per_ch, rms_all


def extract_features(data_dict, bin_size=1000, mode='all_channels', channels=None):
    """
    Extract RMS-based features and targets from data_dict, handling diverse EMG formats.
    """
    # Load reference
    ref = np.asarray(data_dict['ref_signal'], dtype=float).squeeze()

    # Load raw EMG
    emg_raw = data_dict['SIG']
    # Improved debug: actual dtype and shape
    try:
        raw_dtype = emg_raw.dtype
    except Exception:
        raw_dtype = type(emg_raw)
    try:
        raw_shape = emg_raw.shape
    except Exception:
        raw_shape = None
    print(f"DEBUG: emg_raw type={type(emg_raw)}, dtype={raw_dtype}, shape={raw_shape}")

    # Case A: HDF5 group of channels
    if isinstance(emg_raw, h5py.Group):
        arrays = []
        for key in sorted(emg_raw.keys()):
            ds = emg_raw[key]
            arr = np.asarray(ds[()]).squeeze()
            if arr.ndim != 1:
                raise ValueError(f"EMG dataset '{key}' has shape {arr.shape}; expected 1D array.")
            arrays.append(arr)
        emg = np.stack(arrays, axis=1)

    # Case B: HDF5 dataset
    elif isinstance(emg_raw, h5py.Dataset):
        data = emg_raw[()]
        if not np.issubdtype(getattr(data, 'dtype', object), np.number):
            f = data_dict.get('__h5file') or emg_raw.file
            arrays = []
            for idx, refobj in enumerate(data.flat):
                if isinstance(refobj, h5py.Reference):
                    arr = np.asarray(f[refobj][()]).squeeze()
                elif isinstance(refobj, np.ndarray):
                    arr = refobj.squeeze()
                else:
                    arr = np.asarray(refobj).squeeze()
                if arr.ndim != 1:
                    raise ValueError(f"Referenced EMG at idx {idx} has shape {arr.shape}")
                arrays.append(arr)
            lengths = [a.shape[0] for a in arrays]
            if len(set(lengths)) != 1:
                raise ValueError(f"Inconsistent lengths in dereferenced EMG: {lengths}")
            emg = np.stack(arrays, axis=1)
        else:
            emg = data

    # Case C: Python list/tuple of arrays
    elif isinstance(emg_raw, (list, tuple)):
        arrays = []
        for idx, el in enumerate(emg_raw):
            arr = np.asarray(el).squeeze()
            if arr.ndim != 1:
                raise ValueError(f"List element {idx} has shape {arr.shape}; expected 1D array.")
            arrays.append(arr)
        lengths = [a.shape[0] for a in arrays]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent lengths in list EMG: {lengths}")
        emg = np.stack(arrays, axis=1)

            # Case D: object ndarray (older MATLAB cell arrays)
    elif isinstance(emg_raw, np.ndarray) and emg_raw.dtype == object:
        raw_shape = emg_raw.shape
        # Flatten all cells into channel list (handles 16x1 or 8x2 as 16 channels)
        flat = emg_raw.flatten()
        arrays = []
        for idx, el in enumerate(flat):
            arr = np.asarray(el).squeeze()
            if arr.ndim != 1:
                raise ValueError(f"EMG cell element at index {idx} has shape {arr.shape}; expected 1D array.")
            arrays.append(arr)
        lengths = [a.shape[0] for a in arrays]
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent EMG lengths across cells: {lengths}")
        # Stack as channels (samples, n_channels)
        emg = np.stack(arrays, axis=1)

    # Case E: numeric ndarray or other
    else:
        emg = np.asarray(emg_raw, dtype=float)

    # Fix orientation
    n = ref.shape[0]
    if emg.ndim == 1:
        if emg.shape[0] != n:
            raise ValueError(f"Single-channel EMG length {emg.shape[0]} != ref length {n}")
        emg = emg[:, None]
    elif emg.ndim == 2:
        r, c = emg.shape
        if r != n and c == n:
            print(f"Auto-transposing EMG from {emg.shape} to {(emg.T.shape)}")
            emg = emg.T
        elif r != n and c != n:
            raise ValueError(f"EMG shape {emg.shape} does not match ref length {n}")
    else:
        raise ValueError(f"Unexpected EMG dims: {emg.ndim}")

    # Align signals
    ref, emg, fs = check_and_align_signals(ref, emg, data_dict)

    # Bin
    binned_ref = bin_signal(ref, bin_size)
    binned_emg = bin_signal(emg, bin_size)

    y = np.mean(np.abs(binned_ref), axis=1).squeeze()
    rms_per_ch, rms_all = compute_rms(binned_emg)

    # Build X
    if mode == 'all_channels':
        X = rms_all[:, None]
    elif mode == 'single_channel':
        idx = channels[0]
        X = rms_per_ch[:, idx][:, None]
    elif mode == 'iterative_addition':
        sel = channels or list(range(rms_per_ch.shape[1]))
        X = [rms_per_ch[:, sel[:k]] for k in range(1, len(sel)+1)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return X, y

# -- Experiment Runner (aggregate all files) -------------------------------- (aggregate all files) --------------------------------

def run_experiment(data_dir, subjects, bin_size=1000, channel_modes=None):
    scoring = {'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
    results = {}
    # Loop over each channel-mode configuration
    for cfg in channel_modes:
        mode = cfg['mode']; name = cfg['name']; channels = cfg.get('channels')
        # Prepare aggregators
        if mode == 'iterative_addition':
            # Will collect lists per channel count
            sample_y = None
            X_accumulators = []
            for subj in subjects:
                subj_path = os.path.join(data_dir, subj)
                for fname in sorted(os.listdir(subj_path)):
                    if not fname.endswith('.mat'): continue
                    data = load_mat_file(os.path.join(subj_path, fname))
                    X_list, y = extract_features(data, bin_size, mode, channels)
                    if sample_y is None:
                        sample_y = []
                        X_accumulators = [[] for _ in range(len(X_list))]
                    for k, Xk in enumerate(X_list):
                        X_accumulators[k].append(Xk)
                    sample_y.append(y)
            # Stack across files
            y_all = np.concatenate(sample_y)
            # Compute metrics for each k
            mse_list, mae_list = [], []
            for X_parts in X_accumulators:
                X_all = np.vstack(X_parts)
                mse_scores = cross_val_score(Ridge(), X_all, y_all, cv=5, scoring=scoring['mse'])
                mae_scores = cross_val_score(Ridge(), X_all, y_all, cv=5, scoring=scoring['mae'])
                mse_list.append(-np.mean(mse_scores))
                mae_list.append(-np.mean(mae_scores))
            results[name] = {'mse': mse_list, 'mae': mae_list}
        else:
            # Single feature modes: aggregate all files into one X,y
            X_parts, y_parts = [], []
            for subj in subjects:
                subj_path = os.path.join(data_dir, subj)
                for fname in sorted(os.listdir(subj_path)):
                    if not fname.endswith('.mat'): continue
                    data = load_mat_file(os.path.join(subj_path, fname))
                    X, y = extract_features(data, bin_size, mode, channels)
                    X_parts.append(X)
                    y_parts.append(y)
            X_all = np.vstack(X_parts)
            y_all = np.concatenate(y_parts)
            mse_scores = cross_val_score(Ridge(), X_all, y_all, cv=5, scoring=scoring['mse'])
            mae_scores = cross_val_score(Ridge(), X_all, y_all, cv=5, scoring=scoring['mae'])
            results[name] = {
                'mse': -np.mean(mse_scores),
                'mae': -np.mean(mae_scores)
            }
    return results


def save_and_plot(results):
    # Save to CSV
    records = []
    for name, metrics in results.items():
        if isinstance(metrics['mae'], list):
            for k, (mse, mae) in enumerate(zip(metrics['mse'], metrics['mae']), start=1):
                records.append({'mode': name, 'channels': k, 'mse': mse, 'mae': mae})
        else:
            records.append({'mode': name, 'channels': None, 'mse': metrics['mse'], 'mae': metrics['mae']})
    df = pd.DataFrame(records)
    df.to_csv('results_all_metrics.csv', index=False)
    print("Saved results_all_metrics.csv")
    # Plot MAE curves
    for name, grp in df[df['channels'].notna()].groupby('mode'):
        plt.figure()
        plt.plot(grp['channels'], grp['mae'], marker='o')
        plt.title(f"{name} (MAE) vs #Channels")
        plt.xlabel("#Channels")
        plt.ylabel("MAE")
        plt.grid(True)
        plt.savefig(f"iter_curve_mae_{name}.png")
        plt.show()

if __name__ == '__main__':
    data_dir = 'data'
    subjects = ['P1']
    channel_modes = [
        {'name': 'all_channels', 'mode': 'all_channels'},
        {'name': 'single_channel', 'mode': 'single_channel', 'channels': [0]},
        {'name': 'iterative_addition', 'mode': 'iterative_addition', 'channels': list(range(16))}
    ]
    results = run_experiment(data_dir, subjects, bin_size=1000, channel_modes=channel_modes)
    save_and_plot(results)
