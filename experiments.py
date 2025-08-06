import os
import time
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
    print(f"[LOAD] Trying to load: {file_path}")
    t0 = time.time()
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        data = loadmat(file_path)
        print(f"[LOAD] Loaded non-v7.3 in {time.time()-t0:.2f}s")
        return data
    except NotImplementedError:
        f = h5py.File(file_path, 'r')
        print(f"[LOAD] Loaded v7.3 (h5py) in {time.time()-t0:.2f}s")
        data = {key: f[key] for key in f.keys()}
        data['__h5file'] = f
        return data

@lru_cache(maxsize=None)
def load_mat_cached(path):
    print(f"[CACHE] load_mat_cached hit for: {path}")
    return load_mat_file(path)

def check_and_align_signals(ref, emg, data_dict):
    print(f"[ALIGN] raw ref shape: {np.shape(ref)}, raw emg shape: {np.shape(emg)}")
    ref = np.asarray(ref).squeeze()
    emg = np.asarray(emg)
    if emg.ndim == 2 and ref.shape[0] == emg.shape[1] and ref.shape[0] != emg.shape[0]:
        print("[ALIGN] Transposing EMG")
        emg = emg.T
    if ref.shape[0] != emg.shape[0]:
        raise ValueError(f"[ALIGN] Length mismatch: ref {ref.shape[0]}, emg {emg.shape[0]}")
    if np.all(ref == 0) or np.all(emg == 0):
        raise ValueError("[ALIGN] Zero-variance signal detected.")
    fs = data_dict.get('force_fs') or data_dict.get('emg_fs') or data_dict.get('fs') or data_dict.get('srate')
    if isinstance(fs, (list, np.ndarray)):
        fs = float(fs[0])
    summed = np.sum(emg, axis=1)
    try:
        xc = np.correlate(summed - summed.mean(), ref - ref.mean(), mode='full')
        lag = xc.argmax() - (len(ref) - 1)
        print(f"[ALIGN] Computed lag = {lag}")
        if lag > 0:
            ref, emg = ref[lag:], emg[:-lag]
        elif lag < 0:
            ref, emg = ref[:lag], emg[-lag:]
    except Exception as e:
        print(f"[ALIGN] Correlation/lag failed: {e}")
    print(f"[ALIGN] aligned ref shape: {ref.shape}, emg shape: {emg.shape}, fs={fs}")
    return ref, emg, fs

# -------------------- Feature Extraction --------------------

def bin_signal(sig, bin_size):
    print(f"[BIN] Binning signal of shape {np.shape(sig)} with bin_size={bin_size}")
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    n_samples, n_ch = sig.shape
    n_bins = n_samples // bin_size
    binned = sig[:n_bins * bin_size].reshape(n_bins, bin_size, n_ch)
    print(f"[BIN] result bins={binned.shape[0]}, channels={binned.shape[2]}")
    return binned

def compute_rms(binned):
    print(f"[RMS] Computing RMS on binned shape {np.shape(binned)}")
    rms_per_ch = np.sqrt(np.mean(binned**2, axis=1))
    rms_all = np.sqrt(np.mean(rms_per_ch**2, axis=1))
    print(f"[RMS] rms_per_ch shape {rms_per_ch.shape}, rms_all shape {rms_all.shape}")
    return rms_per_ch, rms_all

def extract_features(data_dict, bin_size=1000, mode='all_channels', channels=None):
    print(f"\n[FEAT] extract_features mode={mode}, bin_size={bin_size}, channels={channels}")
    t0 = time.time()
    ref = np.asarray(data_dict['ref_signal'], dtype=float).squeeze()
    emg_raw = data_dict['SIG']
    print(f"[FEAT] raw ref len={ref.shape}, emg_raw type={type(emg_raw)}")

    if isinstance(emg_raw, h5py.Group):
        print("[FEAT] SIG is h5py.Group")
        arrays = [np.asarray(emg_raw[k][()]).squeeze() for k in sorted(emg_raw.keys())]
        emg = np.stack(arrays, axis=1)
    elif isinstance(emg_raw, h5py.Dataset):
        print("[FEAT] SIG is h5py.Dataset")
        data = emg_raw[()]
        if not np.issubdtype(getattr(data, 'dtype', object), np.number):
            print("[FEAT] Dataset of references, unpacking")
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
        print("[FEAT] SIG is cell array / object array")
        cells = emg_raw if isinstance(emg_raw, (list, tuple)) else emg_raw.flatten()
        arrays = [np.asarray(c).squeeze() for c in cells]
        emg = np.stack(arrays, axis=1)
    else:
        print("[FEAT] SIG is raw numeric array")
        emg = np.asarray(emg_raw, dtype=float)

    n = ref.shape[0]
    if emg.ndim == 1:
        emg = emg[:, None]
    elif emg.ndim == 2:
        r, c = emg.shape
        if r != n and c == n:
            print("[FEAT] Transposing after orientation check")
            emg = emg.T
        elif r != n:
            raise ValueError(f"[FEAT] EMG shape mismatch after orientation: {emg.shape}")

    ref2, emg2, fs = check_and_align_signals(ref, emg, data_dict)
    bref = bin_signal(ref2, bin_size)
    bemg = bin_signal(emg2, bin_size)
    y = np.mean(np.abs(bref), axis=1).squeeze()
    rms_per_ch, rms_all = compute_rms(bemg)

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
        raise ValueError(f"[FEAT] Unknown mode: {mode}")

    print(f"[FEAT] Done in {time.time()-t0:.2f}s, X type={type(X)}, y len={len(y)}, fs={fs}")
    return X, y, fs

# -------------------- Aggregation --------------------

def aggregate_rms(data_dir, subjects, bin_size=1000):
    print(f"\n[AGG] aggregate_rms for subjects={subjects}, bin_size={bin_size}")
    R_list, y_list = [], []
    for subj in subjects:
        print(f"[AGG] Subject: {subj}")
        subj_path = os.path.join(data_dir, subj)
        for fname in sorted(os.listdir(subj_path)):
            if not fname.endswith('.mat'):
                continue
            print(f"[AGG]  File: {fname}")
            full = os.path.join(subj_path, fname)
            data = load_mat_file(full)
            R, y, _ = extract_features(data, bin_size, mode='rms_matrix')
            R_list.append(R)
            y_list.append(y)
    R_all = np.vstack(R_list)
    y_all = np.concatenate(y_list)
    print(f"[AGG] aggregated R shape={R_all.shape}, y shape={y_all.shape}")
    return R_all, y_all

# -------------------- Searches --------------------

def exhaustive_search(R, y, max_k=3, cv=5):
    print(f"[SEARCH] exhaustive_search up to {max_k} channels, cv={cv}")
    best = {}
    for k in range(1, max_k+1):
        print(f"[SEARCH]  Testing combos of size {k}")
        bscore, bcombo = np.inf, None
        for combo in itertools.combinations(range(R.shape[1]), k):
            mae = -np.mean(cross_val_score(Ridge(), R[:, combo], y, cv=cv,
                                           scoring='neg_mean_absolute_error'))
            if mae < bscore:
                bscore, bcombo = mae, combo
        print(f"[SEARCH]   Best for k={k}: combo={bcombo}, mae={bscore:.4f}")
        best[k] = {'channels': bcombo, 'mae': bscore}
    return best

def greedy_selection(R, y, cv=5):
    print(f"[SEARCH] greedy_selection cv={cv}")
    remaining, current, hist = set(range(R.shape[1])), [], []
    while remaining:
        bscore, bch = np.inf, None
        for ch in remaining:
            mae = -np.mean(cross_val_score(Ridge(), R[:, current+[ch]], y, cv=cv,
                                           scoring='neg_mean_absolute_error'))
            if mae < bscore:
                bscore, bch = mae, ch
        current.append(bch); remaining.remove(bch)
        hist.append((tuple(current), bscore))
        print(f"[SEARCH]   Added ch={bch}, new mae={bscore:.4f}")
    return hist

def exhaustive_average_search(R, y, max_k=3, cv=5):
    print(f"[SEARCH] exhaustive_average_search up to {max_k} channels, cv={cv}")
    best = {}
    for k in range(1, max_k+1):
        bscore, bcombo = np.inf, None
        for combo in itertools.combinations(range(R.shape[1]), k):
            pred = cross_val_predict(Ridge(),
                                     R[:, combo].mean(axis=1).reshape(-1,1),
                                     y, cv=cv)
            mae = mean_absolute_error(y, pred)
            if mae < bscore:
                bscore, bcombo = mae, combo
        print(f"[SEARCH]   Avg best for k={k}: combo={bcombo}, mae={bscore:.4f}")
        best[k] = {'combo': bcombo, 'mae': bscore}
    return best

def greedy_average_selection(R, y, cv=5):
    print(f"[SEARCH] greedy_average_selection cv={cv}")
    rem, curr, hist = set(range(R.shape[1])), [], []
    while rem:
        bscore, bch = np.inf, None
        for ch in rem:
            pred = cross_val_predict(Ridge(),
                                     R[:, curr+[ch]].mean(axis=1).reshape(-1,1),
                                     y, cv=cv)
            mae = mean_absolute_error(y, pred)
            if mae < bscore:
                bscore, bch = mae, ch
        curr.append(bch); rem.remove(bch)
        hist.append((tuple(curr), bscore))
        print(f"[SEARCH]   Added ch={bch}, new mae={bscore:.4f}")
    return hist

# -------------------- Experiment Runner --------------------

def run_experiment(data_dir, subjects, bin_size, channel_modes):
    print(f"\n[EXP] run_experiment subjects={subjects}, bin_size={bin_size}")
    scoring = {'mse':'neg_mean_squared_error','mae':'neg_mean_absolute_error'}
    results = {}
    for cfg in channel_modes:
        mode, name = cfg['mode'], cfg['name']
        chs = cfg.get('channels')
        print(f"[EXP]  Mode={mode}, Name={name}, Chs={chs}")
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
                ms.append(-np.mean(cross_val_score(Ridge(),Xall,yall,cv=5,scoring=scoring['mse'])))
                mas.append(-np.mean(cross_val_score(Ridge(),Xall,yall,cv=5,scoring=scoring['mae'])))
            results[name]={'mse':ms,'mae':mas}
            print(f"[EXP]   Iterative results: mse={ms}, mae={mas}")
        else:
            Xparts, yparts = [],[]
            for subj in subjects:
                for fname in sorted(os.listdir(os.path.join(data_dir,subj))):
                    if not fname.endswith('.mat'): continue
                    data=load_mat_file(os.path.join(data_dir,subj,fname))
                    X,y,_=extract_features(data,bin_size,mode,chs)
                    Xparts.append(X); yparts.append(y)
            Xall=np.vstack(Xparts); yall=np.concatenate(yparts)
            mse = -np.mean(cross_val_score(Ridge(),Xall,yall,cv=5,scoring=scoring['mse']))
            mae = -np.mean(cross_val_score(Ridge(),Xall,yall,cv=5,scoring=scoring['mae']))
            results[name]={'mse':mse,'mae':mae}
            print(f"[EXP]   Simple results: mse={mse:.4f}, mae={mae:.4f}")
    return results

# -------------------- Save & Plot --------------------

def save_and_plot(results, out_dir):
    print(f"[SAVE] Saving results to {out_dir}")
    records=[]
    for name,met in results.items():
        if isinstance(met['mae'], list):
            for k,(mse,mae) in enumerate(zip(met['mse'],met['mae']),start=1):
                records.append({'mode':name,'channels':k,'mse':mse,'mae':mae})
        else:
            records.append({'mode':name,'channels':None,'mse':met['mse'],'mae':met['mae']})
    df=pd.DataFrame(records)
    out_csv = os.path.join(out_dir,'results_all_metrics.csv')
    df.to_csv(out_csv,index=False)
    print(f"[SAVE] Wrote CSV ({df.shape[0]} rows)")

def plot_reconstruction(y_true, preds_dict, bin_size, fs, out_path):
    print(f"[PLOT] plot_reconstruction to {out_path}")
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
    print(f"[PLOT] Saved figure")

# -------------------- Preload & Memory Aggregation --------------------

def preload_data(subjects, data_dir):
    print(f"[PRE] Preloading data for subjects={subjects} from {data_dir}")
    all_data = {}
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        for fname in sorted(os.listdir(subj_dir)):
            if not fname.endswith('.mat'):
                continue
            key = (subj, fname)
            path = os.path.join(subj_dir, fname)
            print(f"[PRE]  Loading {key}")
            all_data[key] = load_mat_cached(path)
    print(f"[PRE] Preloaded {len(all_data)} files total.")
    return all_data

def aggregate_rms_from_data(all_data, subjects, bin_size=1000):
    print(f"[MEM] aggregate_rms_from_data subjects={subjects}, bin_size={bin_size}")
    R_list, y_list = [], []
    for subj in subjects:
        for (s, fname), data in all_data.items():
            if s != subj: continue
            print(f"[MEM]  Using data {(s,fname)}")
            R, y, _ = extract_features(data, bin_size, mode='rms_matrix')
            R_list.append(R)
            y_list.append(y)
    if not R_list:
        raise ValueError(f"[MEM] No data found for subjects {subjects}")
    R_all = np.vstack(R_list)
    y_all = np.concatenate(y_list)
    print(f"[MEM] aggregated R shape={R_all.shape}, y shape={y_all.shape}")
    return R_all, y_all

def run_experiment_from_data(all_data, subjects, bin_size, channel_modes):
    print(f"\n[EXPD] run_experiment_from_data subjects={subjects}, bin_size={bin_size}")
    scoring = {'mse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error'}
    results = {}
    for cfg in channel_modes:
        name, mode, chs = cfg['name'], cfg['mode'], cfg.get('channels')
        print(f"[EXPD]  Mode={mode}, Name={name}, Chs={chs}")
        if mode == 'iterative_addition':
            sample_y = []
            Xacc = None
            for subj in subjects:
                for (s, fname), data in all_data.items():
                    if s != subj: continue
                    Xlist, y, _ = extract_features(data, bin_size, mode, chs)
                    if Xacc is None:
                        Xacc = [[] for _ in Xlist]
                    for i, Xk in enumerate(Xlist):
                        Xacc[i].append(Xk)
                    sample_y.append(y)
            yall = np.concatenate(sample_y)
            ms, mas = [], []
            for parts in Xacc:
                Xall = np.vstack(parts)
                ms.append(-np.mean(cross_val_score(Ridge(), Xall, yall,
                                                    cv=5, scoring=scoring['mse'])))
                mas.append(-np.mean(cross_val_score(Ridge(), Xall, yall,
                                                    cv=5, scoring=scoring['mae'])))
            results[name] = {'mse': ms, 'mae': mas}
            print(f"[EXPD]   Iterative results: mse={ms}, mae={mas}")
        else:
            Xparts, yparts = [], []
            for subj in subjects:
                for (s, fname), data in all_data.items():
                    if s != subj: continue
                    X, y, _ = extract_features(data, bin_size, mode, chs)
                    Xparts.append(X)
                    yparts.append(y)
            Xall = np.vstack(Xparts)
            yall = np.concatenate(yparts)
            mse = -np.mean(cross_val_score(Ridge(), Xall, yall,
                                            cv=5, scoring=scoring['mse']))
            mae = -np.mean(cross_val_score(Ridge(), Xall, yall,
                                            cv=5, scoring=scoring['mae']))
            results[name] = {'mse': mse, 'mae': mae}
            print(f"[EXPD]   Simple results: mse={mse:.4f}, mae={mae:.4f}")
    return results

# -------------------- main() --------------------

def main():
    import joblib  
    data_dir  = "data"
    subjects  = ["P3"]
    bin_sizes = [1000, 500, 250]

    print("[MAIN] Starting preload")
    all_data = preload_data(subjects, data_dir)

    for subj in subjects:
        for bin_size in bin_sizes:
            print(f"[MAIN] Running for subj={subj}, bin_size={bin_size}")
            out_dir = os.path.join('results', subj, f'bin_{bin_size}')
            os.makedirs(out_dir, exist_ok=True)

            # 1) aggregate in‐memory
            R_all, y_all = aggregate_rms_from_data(all_data, [subj], bin_size)

            # 2) define your channel modes
            channel_modes = [
                {'name':'all_channels',       'mode':'all_channels'},
                {'name':'avg_channels',       'mode':'average_channels'},
                {'name':'single_channel',     'mode':'single_channel',    'channels':[0]},
                {'name':'iterative_addition', 'mode':'iterative_addition','channels':list(range(R_all.shape[1]))}
            ]

            # 3) run cross‐val experiments (unchanged)
            results = run_experiment_from_data(all_data, [subj], bin_size, channel_modes)
            save_and_plot(results, out_dir)

            # 4) now train & save each final model
            for cfg in channel_modes:
                name = cfg['name']
                mode = cfg['mode']
                chs  = cfg.get('channels', None)

                # build X_full for this mode
                if mode == 'all_channels':
                    X_full = np.sqrt(np.mean(R_all**2, axis=1))[:, None]
                elif mode == 'average_channels':
                    X_full = np.mean(R_all, axis=1)[:, None]
                elif mode == 'single_channel':
                    X_full = R_all[:, [chs[0]]]
                elif mode == 'iterative_addition':
                    # pick the best greedy combo found earlier
                    best_combo = min(greedy_selection(R_all, y_all), key=lambda x: x[1])[0]
                    X_full = R_all[:, best_combo]
                else:
                    continue

                model = Ridge().fit(X_full, y_all)
                model_path = os.path.join(out_dir, f"ridge_{name}.joblib")
                joblib.dump(model, model_path)
                print(f"[MAIN] Saved trained model → {model_path}")

    print("[MAIN] All done.")

if __name__ == "__main__":
    main()