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
    Load a MATLAB file (v7.3 HDF5 or earlier) and return a dict-like object.
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
        return {key: np.array(f[key]) for key in f.keys()}


def check_and_align_signals(ref, emg, data_dict):
    """
    Ensure ref and emg have correct shape, same length, non-zero variance,
    matching sampling rates, and zero lag (or correct via cross-correlation).
    Auto-transposes emg if needed. Returns aligned (ref, emg, fs).
    """
    ref = np.asarray(ref).squeeze()

    # Auto-transpose EMG if necessary
    emg = np.asarray(emg)
    if emg.ndim == 2 and ref.shape[0] == emg.shape[1] and ref.shape[0] != emg.shape[0]:
        print(f"Transposing EMG from {emg.shape} to {emg.T.shape}")
        emg = emg.T

    # Verify lengths match
    if ref.shape[0] != emg.shape[0]:
        raise ValueError(f"Length mismatch: ref has {ref.shape[0]} samples, EMG has {emg.shape[0]}.")

    # Check for non-zero variance
    if np.all(ref == 0):
        raise ValueError("Reference signal is all zeros—check your load or mapping.")
    if np.all(emg == 0):
        raise ValueError("EMG data are all zeros—check your load or mapping.")

    # Sampling-rate consistency (optional)
    fs = data_dict.get('force_fs') or data_dict.get('emg_fs') or data_dict.get('fs') or data_dict.get('srate')
    if isinstance(fs, (list, np.ndarray)):
        fs = float(fs[0])
    fs_ref = data_dict.get('force_fs', fs)
    fs_emg = data_dict.get('emg_fs', fs)
    if fs_ref and fs_emg and fs_ref != fs_emg:
        print(f"Warning: force fs = {fs_ref}, EMG fs = {fs_emg}. You may need to resample.")
    fs_use = fs_ref or fs_emg or fs

    # Auto-align by cross-correlation (optional)
    try:
        summed = np.sum(emg, axis=1)
        xc = np.correlate(summed - summed.mean(), ref - ref.mean(), mode='full')
        lag = xc.argmax() - (len(ref) - 1)
        if lag != 0:
            print(f"Shifting EMG by {lag} samples ({lag/fs_use:.3f}s) to align")
            if lag > 0:
                ref = ref[lag:]
                emg = emg[:-lag]
            else:
                shift = abs(lag)
                ref = ref[:-shift]
                emg = emg[shift:]
    except Exception:
        pass

    return ref, emg, fs_use

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
    rms_per_ch = np.sqrt(np.mean(binned**2, axis=1))  # (n_bins, n_ch)
    rms_all = np.sqrt(np.mean(rms_per_ch**2, axis=1))  # (n_bins,)
    return rms_per_ch, rms_all


def extract_features(data_dict, bin_size=1000, mode='all_channels', channels=None):
    """
    Extract RMS-based features and targets from data_dict.

    Parameters:
      data_dict: must contain 'ref_signal' and 'SIG'
      bin_size : samples per bin
      mode     : 'all_channels', 'single_channel', or 'iterative_addition'
      channels : list of channel indices (for single/iterative modes)

    Returns:
      X: feature array (n_bins, n_features) or list of such arrays for iterative
      y: target array (n_bins,)
    """
    # Get raw signals
    ref = data_dict['ref_signal']
    emg = data_dict['SIG']

    # Convert object arrays to numeric
    if isinstance(emg, np.ndarray) and emg.dtype == object:
        # Assume each element is a 1D array of length n_samples; stack as columns
        arrays = [np.asarray(el).squeeze() for el in emg.flat]
        emg = np.stack(arrays, axis=1)

    # Align and validate
    ref, emg, fs = check_and_align_signals(ref, emg, data_dict)

    # Bin into non-overlapping windows
    binned_ref = bin_signal(ref, bin_size)      # (n_bins, bin_size, 1)
    binned_emg = bin_signal(emg, bin_size)      # (n_bins, bin_size, n_ch)

    # Target: mean absolute force per bin
    y = np.mean(np.abs(binned_ref), axis=1).squeeze()

    # Compute RMS features
    rms_per_ch, rms_all = compute_rms(binned_emg)

    # Build feature(s)
    if mode == 'all_channels':
        # single combined RMS feature
        X = rms_all[:, None]

    elif mode == 'single_channel':
        if channels is None or len(channels) != 1:
            raise ValueError("Provide exactly one channel for single_channel mode.")
        # select one column
        X = rms_per_ch[:, channels[0]][:, None]

    elif mode == 'iterative_addition':
        n_ch = rms_per_ch.shape[1]
        sel = channels if channels is not None else list(range(n_ch))
        X_list = []
        for k in range(1, len(sel) + 1):
            # select first k channels
            X_current = rms_per_ch[:, sel[:k]]  # (n_bins, k)
            X_list.append(X_current)
        X = X_list

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return X, y


# -- Experiment Runner -----------------------------------------------------

def run_experiment(data_dir, subjects, bin_size=1000, channel_modes=None):
    """
    Run CV for each config; compute both MSE and MAE for each.
    Returns:
      results: dict mapping (subj,fname,mode,task) -> {'mse': scalar or list, 'mae': ...}
    """
    results = {}
    for subj in subjects:
        subj_path = os.path.join(data_dir, subj)
        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith('.mat'):
                continue

            # Load data
            data = load_mat_file(os.path.join(subj_path, fname))
            task = data.get('task_type', 'unknown')

            for cfg in channel_modes:
                mode = cfg['mode']
                channels = cfg.get('channels', None)

                # Extract features
                if mode == 'iterative_addition':
                    # Handle iterative addition mode
                    X_list, y = extract_features(data, bin_size, mode, channels)
                    mse_list = []
                    mae_list = []

                    for i, X in enumerate(X_list):
                        # Ensure X is 2D
                        assert X.ndim == 2, f"Expected 2D tensor for X, got {X.ndim}D tensor at step {i+1}"
                        assert y.ndim == 1, f"Expected 1D tensor for y, got {y.ndim}D tensor"

                        # Perform cross-validation
                        mse_scores = cross_val_score(Ridge(), X, y, cv=5, scoring='neg_mean_squared_error')
                        mae_scores = cross_val_score(Ridge(), X, y, cv=5, scoring='neg_mean_absolute_error')
                        mse_list.append(-np.mean(mse_scores))
                        mae_list.append(-np.mean(mae_scores))

                        # Store results
                        results[(subj, fname, cfg['name'], task, i + 1)] = {
                            'mse': mse_list[-1],
                            'mae': mae_list[-1]
                        }

                else:
                    # Handle other modes (e.g., 'all_channels', 'single_channel')
                    X, y = extract_features(data, bin_size, mode, channels)

                    # Ensure X is 2D
                    if X.ndim == 1:
                        X = X.reshape(-1, 1)

                    # Debug prints
                    print(f"X shape: {X.shape}")
                    print(f"y shape: {y.shape}")

                    # Perform cross-validation
                    mse_scores = cross_val_score(Ridge(), X, y, cv=5, scoring='neg_mean_squared_error')
                    mae_scores = cross_val_score(Ridge(), X, y, cv=5, scoring='neg_mean_absolute_error')

                    # Store results
                    results[(subj, fname, cfg['name'], task)] = {
                        'mse': -np.mean(mse_scores),
                        'mae': -np.mean(mae_scores)
                    }
    return results


def save_and_plot(results):
    """
    Save combined metrics to CSV and plot only MAE for iterative-addition.
    """
    rows = []
    for (subj, fname, mode, task), metrics in results.items():
        if isinstance(metrics['mae'], list):
            for i, (mse, mae) in enumerate(zip(metrics['mse'], metrics['mae']), start=1):
                rows.append({'subject': subj, 'file': fname, 'mode': mode,
                             'task': task, 'channels': i,
                             'mse': mse, 'mae': mae})
        else:
            rows.append({'subject': subj, 'file': fname, 'mode': mode,
                         'task': task, 'channels': None,
                         'mse': metrics['mse'], 'mae': metrics['mae']})
    df = pd.DataFrame(rows)
    df.to_csv('results_all_metrics.csv', index=False)
    print("Saved all metrics to results_all_metrics.csv")

    # Plot MAE saturation curves
    iter_df = df[df['mode'] == 'iter_add']
    for (subj, task), group in iter_df.groupby(['subject', 'task']):
        plt.figure()
        plt.plot(group['channels'], group['mae'], marker='o')
        plt.title(f"{subj} - {task} (MAE) vs #Channels")
        plt.xlabel("Number of channels")
        plt.ylabel("MAE")
        plt.grid(True)
        plt.savefig(f"iter_curve_mae_{subj}_{task}.png")
        plt.show()

if __name__ == '__main__':
    data_dir = 'data'
    subjects = ['P1', 
]
    channel_modes = [
        {'name': 'all_channels', 'mode': 'all_channels'},
        {'name': 'single_ch0', 'mode': 'single_channel', 'channels': [0]},
        {'name': 'iter_add', 'mode': 'iterative_addition', 'channels': list(range(16))}
    ]
    results = run_experiment(data_dir, subjects, bin_size=1000, channel_modes=channel_modes)
    save_and_plot(results)
