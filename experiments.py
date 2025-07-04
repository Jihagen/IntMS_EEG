import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_absolute_error

# -------------------- Loading & Alignment --------------------

def load_mat_file(file_path):
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
        emg = emg.T
    if ref.shape[0] != emg.shape[0]:
        raise ValueError(f"Length mismatch: ref {ref.shape[0]}, emg {emg.shape[0]}")
    if np.all(ref == 0) or np.all(emg == 0):
        raise ValueError("Zero-variance signal detected.")
    fs = data_dict.get('force_fs') or data_dict.get('emg_fs') or data_dict.get('fs') or data_dict.get('srate')
    if isinstance(fs, (list, np.ndarray)):
        fs = float(fs[0])
    # cross-correlation based alignment
    summed = np.sum(emg, axis=1)
    try:
        xc = np.correlate(summed - summed.mean(), ref - ref.mean(), mode='full')
        lag = xc.argmax() - (len(ref) - 1)
        if lag > 0:
            ref, emg = ref[lag:], emg[:-lag]
        elif lag < 0:
            ref, emg = ref[:lag], emg[-lag:]
    except Exception:
        pass
    return ref, emg, fs

# -------------------- Feature Extraction --------------------

def bin_signal(sig, bin_size):
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    n_samples, n_ch = sig.shape
    n_bins = n_samples // bin_size
    return sig[:n_bins * bin_size].reshape(n_bins, bin_size, n_ch)


def compute_rms(binned):
    rms_per_ch = np.sqrt(np.mean(binned**2, axis=1))
    rms_all = np.sqrt(np.mean(rms_per_ch**2, axis=1))
    return rms_per_ch, rms_all


def extract_features(data_dict, bin_size=1000, mode='all_channels', channels=None):
    # Load reference
    ref = np.asarray(data_dict['ref_signal'], dtype=float).squeeze()
    # Load raw EMG
    emg_raw = data_dict['SIG']
    # Cases A-E for HDF5 group/dataset, list, cell-array, numeric
    if isinstance(emg_raw, h5py.Group):
        arrays = []
        for key in sorted(emg_raw.keys()):
            arr = np.asarray(emg_raw[key][()]).squeeze()
            arrays.append(arr)
        emg = np.stack(arrays, axis=1)
    elif isinstance(emg_raw, h5py.Dataset):
        data = emg_raw[()]
        if not np.issubdtype(getattr(data, 'dtype', object), np.number):
            f = data_dict.get('__h5file') or emg_raw.file
            arrays = []
            for refobj in data.flat:
                if isinstance(refobj, h5py.Reference):
                    arr = np.asarray(f[refobj][()]).squeeze()
                else:
                    arr = np.asarray(refobj).squeeze()
                arrays.append(arr)
            emg = np.stack(arrays, axis=1)
        else:
            emg = data
    elif isinstance(emg_raw, (list, tuple)):
        arrays = [np.asarray(el).squeeze() for el in emg_raw]
        emg = np.stack(arrays, axis=1)
    elif isinstance(emg_raw, np.ndarray) and emg_raw.dtype == object:
        arrays = [np.asarray(el).squeeze() for el in emg_raw.flatten()]
        emg = np.stack(arrays, axis=1)
    else:
        emg = np.asarray(emg_raw, dtype=float)
    # Fix orientation
    n = ref.shape[0]
    if emg.ndim == 1:
        emg = emg[:, None]
    elif emg.ndim == 2:
        r, c = emg.shape
        if r != n and c == n:
            emg = emg.T
        elif r != n and c != n:
            raise ValueError(f"EMG shape mismatch: {emg.shape}")
    else:
        raise ValueError(f"Unexpected EMG dims: {emg.ndim}")
    # Align, bin, compute RMS
    ref, emg, fs = check_and_align_signals(ref, emg, data_dict)
    bref = bin_signal(ref, bin_size)
    bemg = bin_signal(emg, bin_size)
    y = np.mean(np.abs(bref), axis=1).squeeze()
    rms_per_ch, rms_all = compute_rms(bemg)
    # Build X
    if mode == 'all_channels':
        X = rms_all[:, None]
    elif mode == 'single_channel':
        X = rms_per_ch[:, channels[0]][:, None]
    elif mode == 'iterative_addition':
        sel = channels or list(range(rms_per_ch.shape[1]))
        X = [rms_per_ch[:, sel[:k]] for k in range(1, len(sel) + 1)]
    elif mode == 'rms_matrix':
        X = rms_per_ch
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return X, y, fs

# -------------------- Aggregation --------------------

def aggregate_rms(data_dir, subjects, bin_size=1000):
    R_list, y_list = [], []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith('.mat'): continue
            data = load_mat_file(os.path.join(subj_dir, fname))
            R, y, _ = extract_features(data, bin_size, mode='rms_matrix')
            R_list.append(R)
            y_list.append(y)
    return np.vstack(R_list), np.concatenate(y_list)

# -------------------- Subset Searches --------------------

def exhaustive_search(R, y, max_k=3, cv=5):
    best = {}
    for k in range(1, max_k+1):
        bscore, bcombo = np.inf, None
        for combo in itertools.combinations(range(R.shape[1]), k):
            Xk = R[:, combo]
            mae = -np.mean(cross_val_score(Ridge(), Xk, y, cv=cv,
                                           scoring='neg_mean_absolute_error'))
            if mae < bscore:
                bscore, bcombo = mae, combo
        best[k] = {'channels': bcombo, 'mae': bscore}
        print(f"Best {k}-combo: {bcombo}, MAE={bscore:.4f}")
    return best

def greedy_selection(R, y, cv=5):
    remaining = set(range(R.shape[1]))
    current, history = [], []
    while remaining:
        bscore, bch = np.inf, None
        for ch in remaining:
            trial = current + [ch]
            mae = -np.mean(cross_val_score(Ridge(), R[:, trial], y, cv=cv,
                                           scoring='neg_mean_absolute_error'))
            if mae < bscore:
                bscore, bch = mae, ch
        current.append(bch)
        remaining.remove(bch)
        history.append((tuple(current), bscore))
        print(f"Added ch {bch}: combo={current}, MAE={bscore:.4f}")
    return history

# -------------------- Averaged-Channel Modes --------------------

def exhaustive_average_search(R, y, max_k=3, cv=5):
    best = {}
    for k in range(1, max_k+1):
        bscore, bcombo, bpred = np.inf, None, None
        for combo in itertools.combinations(range(R.shape[1]), k):
            Xavg = R[:, combo].mean(axis=1).reshape(-1,1)
            pred = cross_val_predict(Ridge(), Xavg, y, cv=cv)
            mae = mean_absolute_error(y, pred)
            if mae < bscore:
                bscore, bcombo, bpred = mae, combo, pred
        best[k] = {'combo': bcombo, 'mae': bscore, 'pred': bpred}
        print(f"[avg-exh] k={k}: combo={bcombo}, MAE={bscore:.4f}")
    return best

def greedy_average_selection(R, y, cv=5):
    rem = set(range(R.shape[1]))
    curr, history = [], []
    while rem:
        bscore, bch, bpred = np.inf, None, None
        for ch in rem:
            trial = curr + [ch]
            Xavg = R[:, trial].mean(axis=1).reshape(-1,1)
            pred = cross_val_predict(Ridge(), Xavg, y, cv=cv)
            mae = mean_absolute_error(y, pred)
            if mae < bscore:
                bscore, bch, bpred = mae, ch, pred
        curr.append(bch)
        rem.remove(bch)
        history.append((tuple(curr), bscore, bpred))
        print(f"[avg-greedy] added {bch}, combo={curr}, MAE={bscore:.4f}")
    return history

# -------------------- Plotting Utilities --------------------

def plot_feature_target_correlation(R, y, best_combo, out_path):
    corrs = [np.corrcoef(R[:,ch], y)[0,1] for ch in range(R.shape[1])]
    agg = R[:, best_combo].mean(axis=1)
    corrs.append(np.corrcoef(agg, y)[0,1])
    labels = [f'ch{ch}' for ch in range(R.shape[1])] + ['combo']
    plt.figure(figsize=(10,4))
    bars = plt.bar(labels, corrs)
    bars[-1].set_color('tab:orange')
    plt.xticks(rotation=90)
    plt.ylabel('Pearson r')
    plt.title(f'Feature–Target Correlation (best combo={best_combo})')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_reconstruction(y_true, preds_dict, bin_size, fs, out_path):
    t = np.arange(len(y_true))
    if fs:
        t = t * bin_size / fs
        xlabel = 'Time (s)'
    else:
        xlabel = 'Bin index'
    plt.figure(figsize=(8,3))
    plt.plot(t, y_true, 'k-', label='True', alpha=0.7)
    for label, pred in preds_dict.items():
        plt.plot(t, pred, label=label, linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel('Binned |ref|')
    plt.legend(fontsize='small', ncol=len(preds_dict))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# -------------------- MAIN --------------------

if __name__=='__main__':
    data_dir = 'data'
    subjects = ['P1']          # change as needed
    bin_size = 1000

    for subj in subjects:
        out_dir = os.path.join('results', subj)
        os.makedirs(out_dir, exist_ok=True)

        # 1) aggregate
        R_all, y_all = aggregate_rms(data_dir, [subj], bin_size)

        # 2) subset searches
        best_exh = exhaustive_search(R_all, y_all, max_k=3)
        greedy = greedy_selection(R_all, y_all)

        # 3) save CSVs with normalized MAE
        df_exh = pd.DataFrame([
            {'k': k,
             'channels': best_exh[k]['channels'],
             'mae': best_exh[k]['mae'],
             'norm_mae': best_exh[k]['mae'] / (y_all.max() - y_all.min())}
            for k in sorted(best_exh)
        ])
        df_exh.to_csv(os.path.join(out_dir, 'best_combos_exhaustive.csv'), index=False)

        df_greedy = pd.DataFrame([
            {'step': i+1,
             'channels': combo,
             'mae': mae,
             'norm_mae': mae / (y_all.max() - y_all.min())}
            for i,(combo, mae) in enumerate(greedy)
        ])
        df_greedy.to_csv(os.path.join(out_dir, 'greedy_history.csv'), index=False)

        # 4) plot MAE vs k
        plt.figure()
        plt.plot(df_exh['k'], df_exh['mae'], 'o-', label='raw MAE')
        plt.plot(df_exh['k'], df_exh['norm_mae'], 's--', label='normalized MAE')
        plt.xlabel('# Channels')
        plt.ylabel('MAE')
        plt.title(f'Exhaustive MAE (subj={subj})')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(out_dir, 'exhaustive_mae_vs_k.png'))
        plt.close()

        # 5) pick overall best
        be = df_exh.loc[df_exh['mae'].idxmin()]
        bg = df_greedy.loc[df_greedy['mae'].idxmin()]
        if be['mae'] <= bg['mae']:
            best_combo, best_src = tuple(be['channels']), 'exhaustive'
            best_mae = be['mae']
        else:
            best_combo, best_src = tuple(bg['channels']), 'greedy'
            best_mae = bg['mae']
        norm_mae = best_mae / (y_all.max() - y_all.min())
        print(f"Subject {subj}: best MAE={best_mae:.4f}, norm MAE={norm_mae:.4f} ({best_src}) channels={best_combo}")

        # summary
        with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
            f.write(f"best_source: {best_src}\n")
            f.write(f"best_combo: {best_combo}\n")
            f.write(f"raw_mae: {best_mae:.4f}\n")
            f.write(f"normalized_mae: {norm_mae:.4f}\n")

        # 6) correlation plot
        plot_feature_target_correlation(
            R_all, y_all, best_combo,
            os.path.join(out_dir, 'feature_target_correlation.png')
        )

        # 7) reconstructions: single, all, combo
        single = best_combo[0]
        y_single = cross_val_predict(Ridge(), R_all[:, [single]], y_all, cv=5)
        y_allpred = cross_val_predict(Ridge(), R_all, y_all, cv=5)
        y_combo   = cross_val_predict(Ridge(), R_all[:, best_combo], y_all, cv=5)

        plot_reconstruction(
            y_all,
            {f'ch{single}': y_single, 'all': y_allpred, f'combo{best_combo}': y_combo},
            bin_size, fs=None,
            out_path=os.path.join(out_dir, 'reconstruction_single_all_combo.png')
        )

       # 8) Fit global models once
        model_all   = Ridge().fit(R_all,            y_all)
        model_single= Ridge().fit(R_all[:, [single]], y_all)
        model_combo = Ridge().fit(R_all[:, best_combo], y_all)

        subj_dir = os.path.join(data_dir, subj)
        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith('.mat'):
                continue

            # 9) extract this file's R_i, y_i
            data = load_mat_file(os.path.join(subj_dir, fname))
            R_i, y_i, _ = extract_features(data, bin_size, mode='rms_matrix')

            # 10) predict separately
            y_i_all    = model_all.predict(R_i)
            y_i_single = model_single.predict(R_i[:, [single]])
            y_i_combo  = model_combo.predict(R_i[:, best_combo])

            # 11) plot per-file reconstruction
            plt.figure(figsize=(6,2.5))
            plt.plot(y_i,    'k-', label='True', alpha=0.7)
            plt.plot(y_i_all,    'tab:orange', label='all')
            plt.plot(y_i_single, 'tab:blue',   label=f'ch{single}')
            plt.plot(y_i_combo,  'tab:green',  label=f'combo{best_combo}')
            plt.xlabel('Bin index')
            plt.ylabel('Binned |ref|')
            plt.title(f'{subj}/{fname}')
            plt.legend(fontsize='small', ncol=2)
            plt.tight_layout()

            out_png = os.path.join(out_dir, f'recon_{os.path.splitext(fname)[0]}.png')
            plt.savefig(out_png)
            plt.close()

    print("All done.")