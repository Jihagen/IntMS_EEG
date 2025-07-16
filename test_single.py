import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import h5py
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

# import your existing functions:
from experiments import load_mat_file, extract_features, greedy_selection

# ---------- Minimal Test ----------

# 1) Specify the single file to test
data_dir = 'data'
subject = 'P1'
fname = '90Deg_F1_1_plateau_iAll_MUedit.mat_decomp_100It_FDI_DEMUSEEDIT_cut_C.mat'   # ← replace with a real filename under data/P1
file_path = os.path.join(data_dir, subject, fname)

# 2) Load and extract full-RMS matrix and target
data = load_mat_file(file_path)
bin_size = 1000
R, y, fs = extract_features(data, bin_size, mode='rms_matrix')

# 3) Compute predictions for each mode

# a) All channels together (global RMS)
X_all = np.sqrt(np.mean(R**2, axis=1))[:, None]
y_allpred = cross_val_predict(Ridge(), X_all, y, cv=5)

# b) Average‐channels mode
X_avg = np.mean(R, axis=1)[:, None]
y_avgpred = cross_val_predict(Ridge(), X_avg, y, cv=5)

# c) Iterative‐addition best combo
hist = greedy_selection(R, y)
best_combo = hist[ np.argmin([m for (_, m) in hist]) ][0]
X_iter = R[:, best_combo]
y_iterpred = cross_val_predict(Ridge(), X_iter, y, cv=5)

# 4) Plot reconstruction comparison
t = np.arange(len(y)) * (bin_size / fs if fs else 1)
plt.figure(figsize=(8,3))
plt.plot(t, y,     'k-',  label='True', alpha=0.7)
plt.plot(t, y_allpred, 'tab:orange', label='all_channels')
plt.plot(t, y_avgpred, 'tab:green',  label='avg_channels')
plt.plot(t, y_iterpred,'tab:blue',   label=f'iterative_{best_combo}')
plt.xlabel('Time (s)' if fs else 'Bin index')
plt.ylabel('Binned |ref|')
plt.legend(fontsize='small')
plt.tight_layout()
plt.show()