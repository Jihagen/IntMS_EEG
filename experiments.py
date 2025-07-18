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
from functools import lru_cache

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

@lru_cache(maxsize=None)
def load_mat_cached(path):
    """Exactly like load_mat_file, but only actually hits disk once per path."""
    return load_mat_file(path)


def check_and_align_signals(ref, emg, data_dict):
    ref = np.asarray(ref).squeeze()
    emg = np.asarray(emg)
    if emg.ndim == 2 and ref.shape[0] == emg.shape[1] and ref.shape[0] != emg.shape[0]:
        emg = emg.T
    if ref.shape[0] != emg.shape[0]:
        raise ValueError(f"Length mismatch: ref {ref.shape[0]}, emg {emg.shape[0]}")
    if np.all(ref == 0) or np.all(emg == 0):
        raise ValueError("Zero-variance signal detected.")
    fs = data_dict.get('force_fs') or data_dict.get('emg_fs') or data_dict.get('fs') or data_dict.get('srate')
    if isinstance(fs, (list, np.ndarray)):
        fs = float(fs[0])
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
    ref = np.asarray(data_dict['ref_signal'], dtype=float).squeeze()
    emg_raw = data_dict['SIG']
    # Cases A-E for HDF5, list, ndarray
    if isinstance(emg_raw, h5py.Group):
        arrays = [np.asarray(emg_raw[k][()]).squeeze() for k in sorted(emg_raw.keys())]
        emg = np.stack(arrays, axis=1)
    elif isinstance(emg_raw, h5py.Dataset):
        data = emg_raw[()]
        if not np.issubdtype(getattr(data, 'dtype', object), np.number):
            f = data_dict.get('__h5file') or emg_raw.file
            arrays = [(np.asarray(f[refobj][()]).squeeze() if isinstance(refobj, h5py.Reference)
                       else np.asarray(refobj).squeeze()) for refobj in data.flat]
            emg = np.stack(arrays, axis=1)
        else:
            emg = data
    elif isinstance(emg_raw, (list, tuple)) or (isinstance(emg_raw, np.ndarray) and emg_raw.dtype == object):
        cells = emg_raw if isinstance(emg_raw, (list, tuple)) else emg_raw.flatten()
        arrays = [np.asarray(c).squeeze() for c in cells]
        emg = np.stack(arrays, axis=1)
    else:
        emg = np.asarray(emg_raw, dtype=float)
    # orientation
    n = ref.shape[0]
    if emg.ndim == 1:
        emg = emg[:, None]
    elif emg.ndim == 2:
        r, c = emg.shape
        if r != n and c == n:
            emg = emg.T
        elif r != n:
            raise ValueError(f"EMG shape mismatch: {emg.shape}")
    # align & bin
    ref, emg, fs = check_and_align_signals(ref, emg, data_dict)
    bref = bin_signal(ref, bin_size)
    bemg = bin_signal(emg, bin_size)
    y = np.mean(np.abs(bref), axis=1).squeeze()
    rms_per_ch, rms_all = compute_rms(bemg)
    # feature modes
    if mode == 'all_channels':
        X = rms_all[:, None]
    elif mode == 'single_channel':
        X = rms_per_ch[:, channels[0]][:, None]
    elif mode == 'iterative_addition':
        sel = channels or list(range(rms_per_ch.shape[1]))
        X = [rms_per_ch[:, sel[:k]] for k in range(1, len(sel)+1)]
    elif mode == 'rms_matrix':
        X = rms_per_ch
    elif mode == 'average_channels':
        X = np.mean(rms_per_ch, axis=1)[:, None]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return X, y, fs

# -------------------- Aggregation --------------------

def aggregate_rms(data_dir, subjects, bin_size=1000):
    R_list, y_list = [], []
    for subj in subjects:
        for fname in sorted(os.listdir(os.path.join(data_dir, subj))):
            if not fname.endswith('.mat'): continue
            data = load_mat_file(os.path.join(data_dir, subj, fname))
            R, y, _ = extract_features(data, bin_size, mode='rms_matrix')
            R_list.append(R); y_list.append(y)
    return np.vstack(R_list), np.concatenate(y_list)

# -------------------- Searches --------------------

def exhaustive_search(R, y, max_k=3, cv=5):
    best = {}
    for k in range(1, max_k+1):
        bscore, bcombo = np.inf, None
        for combo in itertools.combinations(range(R.shape[1]), k):
            mae = -np.mean(cross_val_score(Ridge(), R[:, combo], y, cv=cv,
                                           scoring='neg_mean_absolute_error'))
            if mae < bscore: bscore, bcombo = mae, combo
        best[k] = {'channels': bcombo, 'mae': bscore}
    return best

def greedy_selection(R, y, cv=5):
    remaining, current, hist = set(range(R.shape[1])), [], []
    while remaining:
        bscore, bch = np.inf, None
        for ch in remaining:
            mae = -np.mean(cross_val_score(Ridge(), R[:, current+[ch]], y, cv=cv,
                                           scoring='neg_mean_absolute_error'))
            if mae < bscore: bscore, bch = mae, ch
        current.append(bch); remaining.remove(bch)
        hist.append((tuple(current), bscore))
    return hist

# -------------------- Averaged-channel modes --------------------

def exhaustive_average_search(R, y, max_k=3, cv=5):
    best = {}
    for k in range(1, max_k+1):
        bscore, bcombo = np.inf, None
        for combo in itertools.combinations(range(R.shape[1]), k):
            pred = cross_val_predict(Ridge(), R[:, combo].mean(axis=1).reshape(-1,1), y, cv=cv)
            mae = mean_absolute_error(y, pred)
            if mae < bscore: bscore, bcombo = mae, combo
        best[k] = {'combo': bcombo, 'mae': bscore}
    return best

def greedy_average_selection(R, y, cv=5):
    rem, curr, hist = set(range(R.shape[1])), [], []
    while rem:
        bscore, bch = np.inf, None
        for ch in rem:
            pred = cross_val_predict(Ridge(), R[:, curr+[ch]].mean(axis=1).reshape(-1,1), y, cv=cv)
            mae = mean_absolute_error(y, pred)
            if mae < bscore: bscore, bch = mae, ch
        curr.append(bch); rem.remove(bch)
        hist.append((tuple(curr), bscore))
    return hist

# -------------------- Experiment Runner --------------------

def run_experiment(data_dir, subjects, bin_size, channel_modes):
    scoring = {'mse':'neg_mean_squared_error','mae':'neg_mean_absolute_error'}
    results = {}
    for cfg in channel_modes:
        mode, name = cfg['mode'], cfg['name']
        chs = cfg.get('channels')
        if mode=='iterative_addition':
            sample_y=None; Xacc=[]
            for subj in subjects:
                for fname in sorted(os.listdir(os.path.join(data_dir,subj))):
                    if not fname.endswith('.mat'): continue
                    data=load_mat_file(os.path.join(data_dir,subj,fname))
                    Xlist,y,_=extract_features(data,bin_size,mode,chs)
                    if sample_y is None:
                        sample_y=[]; Xacc=[[] for _ in Xlist]
                    for i,Xk in enumerate(Xlist): Xacc[i].append(Xk)
                    sample_y.append(y)
            yall=np.concatenate(sample_y)
            ms, mas = [],[]
            for parts in Xacc:
                Xall=np.vstack(parts)
                ms.append(-np.mean(cross_val_score(Ridge(),Xall,yall,scoring=scoring['mse'],cv=5)))
                mas.append(-np.mean(cross_val_score(Ridge(),Xall,yall,scoring=scoring['mae'],cv=5)))
            results[name]={'mse':ms,'mae':mas}
        else:
            Xparts, yparts = [],[]
            for subj in subjects:
                for fname in sorted(os.listdir(os.path.join(data_dir,subj))):
                    if not fname.endswith('.mat'): continue
                    data=load_mat_file(os.path.join(data_dir,subj,fname))
                    X,y,_=extract_features(data,bin_size,mode,chs)
                    Xparts.append(X); yparts.append(y)
            Xall=np.vstack(Xparts); yall=np.concatenate(yparts)
            mse = -np.mean(cross_val_score(Ridge(),Xall,yall,scoring=scoring['mse'],cv=5))
            mae = -np.mean(cross_val_score(Ridge(),Xall,yall,scoring=scoring['mae'],cv=5))
            results[name]={'mse':mse,'mae':mae}
    return results

# -------------------- Save & Plot --------------------

def save_and_plot(results, out_dir):
    records=[]
    for name,met in results.items():
        if isinstance(met['mae'], list):
            for k,(mse,mae) in enumerate(zip(met['mse'],met['mae']),start=1):
                records.append({'mode':name,'channels':k,'mse':mse,'mae':mae})
        else:
            records.append({'mode':name,'channels':None,'mse':met['mse'],'mae':met['mae']})
    df=pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir,'results_all_metrics.csv'),index=False)


def plot_reconstruction(y_true, preds_dict, bin_size, fs, out_path):
    import matplotlib.pyplot as plt
    t = np.arange(len(y_true))
    xlabel = 'Bin index'
    if fs:
        t = t * bin_size / fs
        xlabel = 'Time (s)'
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

# -------------------- Preload --------------------

def preload_data(subjects, data_dir):
    """Scan disk once, load every .mat into a dict keyed by (subject, filename)."""
    all_data = {}
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith('.mat'):
                continue
            key = (subj, fname)
            all_data[key] = load_mat_cached(os.path.join(subj_dir, fname))
    print(f"Preloaded {len(all_data)} files total.")
    return all_data

# -------------------- Aggregation (memory) --------------------

def aggregate_rms_from_data(all_data, subjects, bin_size=1000):
    R_list, y_list = [], []
    for subj in subjects:
        for (s, fname), data in all_data.items():
            if s != subj:
                continue
            # no more `if "_decomp_{bin_size}It_" not in fname: continue`
            R, y, _ = extract_features(data, bin_size, mode='rms_matrix')
            R_list.append(R)
            y_list.append(y)
    if not R_list:
        raise ValueError(f"No data found for subjects {subjects}")
    return np.vstack(R_list), np.concatenate(y_list)

# -------------------- Experiment Runner (memory) --------------------
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

def run_experiment_from_data(all_data, subjects, bin_size, channel_modes):
    """
    all_data: dict keyed by (subject, filename), values are loaded mat dicts
    subjects: list of subject IDs to include
    bin_size: bin size to pass into extract_features
    channel_modes: list of dicts, each with keys 'name', 'mode', and optional 'channels'
    """
    scoring = {'mse': 'neg_mean_squared_error',
               'mae': 'neg_mean_absolute_error'}
    results = {}

    for cfg in channel_modes:
        name = cfg['name']
        mode = cfg['mode']
        chs  = cfg.get('channels', None)

        if mode == 'iterative_addition':
            # Build Xacc and yall across all files in memory
            sample_y = []
            Xacc = None

            for subj in subjects:
                for (s, fname), data in all_data.items():
                    if s != subj:
                        continue
                    Xlist, y, _ = extract_features(data, bin_size, mode, chs)

                    if Xacc is None:
                        # initialize a bucket for each iteration-step
                        Xacc = [[] for _ in Xlist]

                    for i, Xk in enumerate(Xlist):
                        Xacc[i].append(Xk)

                    sample_y.append(y)

            yall = np.concatenate(sample_y)

            ms, mas = [], []
            for parts in Xacc:
                Xall = np.vstack(parts)
                ms.append(-np.mean(cross_val_score(
                    Ridge(), Xall,   yall,
                    cv=5, scoring=scoring['mse']
                )))
                mas.append(-np.mean(cross_val_score(
                    Ridge(), Xall,   yall,
                    cv=5, scoring=scoring['mae']
                )))
            results[name] = {'mse': ms, 'mae': mas}

        else:
            # Simple modes: aggregate X and y once per file
            Xparts, yparts = [], []

            for subj in subjects:
                for (s, fname), data in all_data.items():
                    if s != subj:
                        continue
                    X, y, _ = extract_features(data, bin_size, mode, chs)
                    Xparts.append(X)
                    yparts.append(y)

            Xall = np.vstack(Xparts)
            yall = np.concatenate(yparts)

            mse = -np.mean(cross_val_score(
                Ridge(), Xall, yall,
                cv=5, scoring=scoring['mse']
            ))
            mae = -np.mean(cross_val_score(
                Ridge(), Xall, yall,
                cv=5, scoring=scoring['mae']
            ))
            results[name] = {'mse': mse, 'mae': mae}

    return results

# -------------------- main() --------------------

def main():
    data_dir     = "data"
    subjects     = ["P1"]
    bin_sizes    = [1000, 500, 250]

    # 1) Preload all files into memory once
    all_data = preload_data(subjects, data_dir)

    for subj in subjects:
        for bin_size in bin_sizes:
            out_dir = os.path.join('results', subj, f'bin_{bin_size}')
            os.makedirs(out_dir, exist_ok=True)

            # 2) Aggregate RMS from memory
            R_all, y_all = aggregate_rms_from_data(all_data, [subj], bin_size)

            # 3) Build your channel_modes now that R_all is in scope
            channel_modes = [
                {'name':'all_channels',       'mode':'all_channels'},
                {'name':'avg_channels',       'mode':'average_channels'},
                {'name':'single_channel',     'mode':'single_channel',    'channels':[0]},
                {'name':'iterative_addition', 'mode':'iterative_addition','channels':list(range(R_all.shape[1]))}
            ]

            # 4) Run experiments from memory
            results = run_experiment_from_data(all_data, [subj], bin_size, channel_modes)
            save_and_plot(results, out_dir)

            # 5) Your original “plot reconstruction” block, unchanged:
            rms_all = np.sqrt(np.mean(R_all**2, axis=1))
            y_pred_all = cross_val_predict(Ridge(), rms_all[:, None], y_all, cv=5)
            y_pred_single = cross_val_predict(Ridge(), R_all[:, [0]], y_all, cv=5)
            y_pred_avgchan = cross_val_predict(Ridge(), np.mean(R_all, axis=1)[:, None], y_all, cv=5)
            iter_hist = greedy_selection(R_all, y_all)
            iter_best_combo = min(iter_hist, key=lambda x: x[1])[0]
            y_pred_iter = cross_val_predict(Ridge(), R_all[:, iter_best_combo], y_all, cv=5)
            avg_iter_hist = greedy_average_selection(R_all, y_all)
            avg_iter_best_combo = min(avg_iter_hist, key=lambda x: x[1])[0]
            y_pred_avgiter = cross_val_predict(
                Ridge(),
                np.mean(R_all[:, avg_iter_best_combo], axis=1)[:, None],
                y_all, cv=5
            )
            preds = {
                'all_channels': y_pred_all,
                'single_channel': y_pred_single,
                'avg_channels': y_pred_avgchan,
                f'iterative_{iter_best_combo}': y_pred_iter,
                f'avgiter_{avg_iter_best_combo}': y_pred_avgiter
            }
            plot_reconstruction(
                y_all,
                preds,
                bin_size,
                fs=None,
                out_path=os.path.join(out_dir, 'reconstruction_comparison.png')
            )

    print("All done.")

if __name__ == "__main__":
    main()