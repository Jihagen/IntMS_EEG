# preprocess_plateaus.py

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Iterable


# ---------- small helpers ----------

def _ensure_2xN(mat: np.ndarray) -> np.ndarray:
    """Force target-force matrix to shape (2, N). Accepts (N,2) or (2,N) or (N,)."""
    A = np.asarray(mat, float)
    if A.ndim == 1:
        A = A[np.newaxis, :]            # (1, N) -> caller should handle missing Y
    if A.shape[0] != 2 and A.shape[1] == 2:
        A = A.T
    return A

def _fs_from_payload(payload: Dict[str, Any]) -> Optional[float]:
    meta = payload.get("meta", {}) or {}
    fs_force = meta.get("fs_force")
    fs_emg   = meta.get("fs_emg")
    for v in (fs_force, fs_emg):
        if v is not None:
            try:
                vv = float(np.asarray(v).squeeze())
                if np.isfinite(vv):
                    return vv
            except Exception:
                pass
    # estimate from time axis
    t = np.asarray(payload.get("time_s", []), float)
    if t.size >= 2:
        dt = np.median(np.diff(t))
        if dt > 0:
            return float(1.0 / dt)
    return None

def _angle_ts_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Angle time series in [0, 360)."""
    ang = np.degrees(np.arctan2(y, x))  # (-180, 180]
    ang = np.where(ang < 0, ang + 360.0, ang)
    return ang.astype(float)

def _get_time_windows(payload: Dict[str, Any], desired_kind: Optional[str] = None) -> Tuple[List[Tuple[float,float]], Dict[str,bool]]:
    """
    Returns a list of (t0, t1) windows in seconds and a source flag.
    Priority:
      1) cuts: find the last entry matching desired_kind (if provided),
         else fall back to the last cuts entry.
      2) plateaus (legacy): [t0, t1, t2, t3]
    """
    src = {"cuts": False, "plateaus": False}
    wins: List[Tuple[float, float]] = []

    cuts = payload.get("cuts")
    # New structure: cuts is a list of dicts with keys like:
    # ['kind','t0','t1','t2','t3','idx01','idx23','indices_all','fs_est']
    if isinstance(cuts, list) and len(cuts) > 0:
        chosen = None
        if desired_kind is not None:
            # pick LAST matching kind
            for entry in reversed(cuts):
                try:
                    if str(entry.get("kind", "")).lower() == str(desired_kind).lower():
                        chosen = entry
                        break
                except Exception:
                    pass
        # fallback: last entry if no match or no desired_kind provided
        if chosen is None:
            chosen = cuts[-1]

        try:
            t0 = float(chosen["t0"]); t1 = float(chosen["t1"])
            t2 = float(chosen["t2"]); t3 = float(chosen["t3"])
            wins = [(min(t0,t1), max(t0,t1)), (min(t2,t3), max(t2,t3))]
            src["cuts"] = True
            # annotate which kind we actually used (best-effort)
            src["cuts_kind"] = str(chosen.get("kind")) if isinstance(chosen, dict) else None
            return wins, src
        except Exception:
            # if malformed, we fall back to plateaus
            pass

    P = payload.get("plateaus")
    if P is not None and len(P) == 4:
        t0, t1, t2, t3 = map(float, P)
        wins = [(min(t0,t1), max(t0,t1)), (min(t2,t3), max(t2,t3))]
        src["plateaus"] = True
        return wins, src

    return wins, src


def _times_to_idx(t: np.ndarray, t0: float, t1: float) -> np.ndarray:
    return np.where((t >= t0) & (t <= t1))[0]

def _pad_rms_same_len_1d(x: np.ndarray, win: int) -> np.ndarray:
    """
    Sliding RMS with edge padding to keep SAME length as x.
    y[n] = sqrt( mean( x^2 over centered window ) )
    """
    x = np.asarray(x, float)
    if win is None or win < 2:
        return np.sqrt(np.maximum(x**2, 0.0))
    padL = win // 2
    padR = win - 1 - padL
    x2 = np.pad(x**2, (padL, padR), mode='edge')
    y2 = np.convolve(x2, np.ones(win, dtype=float) / float(win), mode="valid")
    return np.sqrt(np.maximum(y2, 0.0))

def _rms_matrix_same_len(EMG: np.ndarray, win: int) -> np.ndarray:
    """
    Per-channel sliding RMS with SAME length.
    EMG: (C, N) -> returns (N, C)
    """
    EMG = np.asarray(EMG, float)
    assert EMG.ndim == 2, "EMG must be (channels, samples)"
    C, N = EMG.shape
    out = np.empty((N, C), dtype=float)
    for c in range(C):
        out[:, c] = _pad_rms_same_len_1d(EMG[c], win)
    return out


# ---------- main API ----------

def preprocess_plateaus(
    payload: Dict[str, Any],
    rms_win_samples: int = 100,
    modes: Iterable[str] = ("rms_matrix", "all_channels", "average_channels"),
    single_channel_idx: Optional[int] = None,
    iterative_channels: Optional[List[int]] = None,
    keep_intended_angle: bool = True,
    segment_kind: Optional[str] = "plateau",
) -> Dict[str, Any]:
    """
    Build two plateau samples (by time), compute angle time series and EMG RMS + aggregations.

    Parameters
    ----------
    payload : dict
        The combined trial dict loaded from *_combined.npy.
        Must contain: time_s, Fx, Fy, Fab, EMG. (force_target optional)
    rms_win_samples : int
        Sliding RMS window (samples on payload['time_s'] grid). Keep SAME length.
    modes : Iterable[str]
        Any of: 'rms_matrix', 'all_channels', 'average_channels', 'single_channel', 'iterative_addition'
    single_channel_idx : int | None
        Channel index to expose when 'single_channel' is requested.
    iterative_channels : list[int] | None
        Order of channel indices for cumulative RMS when 'iterative_addition' is requested.
    keep_intended_angle : bool
        If True and force_target exists, also compute 'angle_intended' time series; else set to None.

    Returns
    -------
    dict with keys:
      - 'segments': list of per-plateau dicts (see below)
      - 'payload_passthrough': the original payload
    Each segment dict contains:
      {
        'name': 'plateau01' | 'plateau23',
        't_range': (t0, t1),
        'idx': np.ndarray[int],
        'fs': float | None,
        'signals': {
            't': (N,),
            'Fx': (N,), 'Fy': (N,), 'Fab': (N,),
            'angle_achieved': (N,),
            'angle_intended': (N,) or None
        },
        'emg': {
            'rms_matrix': (N,C),
            'all_channels': (N,1),            # if requested
            'average_channels': (N,1),        # if requested
            'single_channel': (N,1),          # if requested
            'iterative_addition': [ (N,1), ... ]  # if requested
        },
        'meta': {
            'n_channels': C,
            'rms_win_samples': int,
            'source': {'cuts': bool, 'plateaus': bool}
        }
      }
    """
    # --- required arrays ---
    t   = np.asarray(payload["time_s"], float)
    Fx  = np.asarray(payload["Fx"], float)
    Fy  = np.asarray(payload["Fy"], float)
    Fab = np.asarray(payload["Fab"], float)
    EMG = np.asarray(payload["EMG"], float)     # (C, N)

    if EMG.ndim != 2:
        raise ValueError("payload['EMG'] must be 2D (channels, samples)")

    # optional intended vectors
    FT = payload.get("force_target", None)
    has_intended = False
    FTxy = None
    if keep_intended_angle and FT is not None:
        try:
            FTxy = _ensure_2xN(FT)
            if FTxy.shape[0] == 2 and FTxy.shape[1] == t.size:
                has_intended = True
        except Exception:
            has_intended = False

    fs = _fs_from_payload(payload)
    wins, source_flags = _get_time_windows(payload, desired_kind=segment_kind)

    kind_label = str(segment_kind) if segment_kind is not None else "plateau"
    segments: List[Dict[str, Any]] = []
    names = (f"{kind_label}01", f"{kind_label}23")

    for idx_seg, (nm, win_pair) in enumerate(zip(names, wins)):
        t0, t1 = win_pair
        idx = _times_to_idx(t, t0, t1)
        if idx.size == 0:
            # skip empty segments
            continue

        # slice signals (time series)
        t_seg   = t[idx]
        Fx_seg  = Fx[idx]
        Fy_seg  = Fy[idx]
        Fab_seg = Fab[idx]

        angle_ach = _angle_ts_from_xy(Fx_seg, Fy_seg)
        if has_intended:
            ftx, fty = FTxy[0], FTxy[1]
            angle_int = _angle_ts_from_xy(ftx[idx], fty[idx])
        else:
            angle_int = None

        # EMG RMS per channel (time-domain) -> (N_seg, C)
        # NOTE: EMG is (C, N); _rms_matrix_same_len returns (N, C)
        R = _rms_matrix_same_len(EMG[:, idx], rms_win_samples)

        # aggregations (all time series)
        emg_dict: Dict[str, Any] = {"rms_matrix": R}

        req = set(modes or [])
        if "all_channels" in req:
            emg_dict["all_channels"] = np.sqrt(np.nanmean(R**2, axis=1, keepdims=True))  # (N,1)
        if "average_channels" in req:
            emg_dict["average_channels"] = np.nanmean(R, axis=1, keepdims=True)          # (N,1)
        if "single_channel" in req:
            if single_channel_idx is None:
                raise ValueError("single_channel mode requires single_channel_idx")
            if not (0 <= single_channel_idx < R.shape[1]):
                raise IndexError(f"single_channel_idx {single_channel_idx} out of range 0..{R.shape[1]-1}")
            emg_dict["single_channel"] = R[:, [single_channel_idx]]                      # (N,1)
        if "iterative_addition" in req:
            if iterative_channels is None or len(iterative_channels) == 0:
                raise ValueError("iterative_addition mode requires iterative_channels (list of ints)")
            # cumulative sets: for k in 1..K, RMS across channels[0:k]
            steps: List[np.ndarray] = []
            for k in range(1, len(iterative_channels) + 1):
                subset = iterative_channels[:k]
                if not all(0 <= ch < R.shape[1] for ch in subset):
                    raise IndexError("iterative_channels contains invalid channel indices")
                # RMS across subset at each time step
                steps.append(np.sqrt(np.nanmean(R[:, subset]**2, axis=1, keepdims=True)))  # (N,1)
            emg_dict["iterative_addition"] = steps
      

        segments.append({
            "name": nm,
            "t_range": (float(t0), float(t1)),
            "idx": idx,
            "fs": fs,
            "signals": {
                "t": t_seg,
                "Fx": Fx_seg,
                "Fy": Fy_seg,
                "Fab": Fab_seg,
                "angle_achieved": angle_ach,
                "angle_intended": angle_int,
            },
            "emg": emg_dict,
            "meta": {
                "n_channels": int(EMG.shape[0]),
                "rms_win_samples": int(rms_win_samples),
                "source": dict(source_flags),
            }
        })

    return {
        "segments": segments,
        "payload_passthrough": payload
    }
