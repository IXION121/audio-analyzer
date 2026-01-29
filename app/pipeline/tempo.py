import numpy as np
import librosa

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))

def estimate_bpm_and_confidence(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Robust-ish tempo estimation:
    - HPSS -> percussive component
    - onset strength -> tempo estimation
    - confidence -> tempogram peak sharpness + top1-top2 separation
    Returns: (bpm, confidence in [0,1])
    """
    if y is None or len(y) < sr * 3:  # <3s ise tempo çok güvensiz olur
        return 0.0, 0.0

    # 1) Percussive ağırlıklı çalış
    try:
        _, y_perc = librosa.effects.hpss(y)
    except Exception:
        y_perc = y

    # 2) Onset envelope
    onset_env = librosa.onset.onset_strength(y=y_perc, sr=sr)

    if onset_env is None or len(onset_env) < 16:
        return 0.0, 0.0

    # 3) Frame-wise tempo tahminleri -> median daha stabil
    tempos = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)

    tempos = np.asarray(tempos, dtype=float)
    tempos = tempos[np.isfinite(tempos)]
    if tempos.size == 0:
        return 0.0, 0.0

    bpm = float(np.median(tempos))
    # Çok uç değerleri kırp (pratik)
    bpm = float(np.clip(bpm, 40.0, 220.0))

    # 4) Confidence: tempogram ortalaması üzerinden peak keskinliği
    try:
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        tg_mean = np.mean(tg, axis=1)  # (tempo_bins,)
        freqs = librosa.tempo_frequencies(tg.shape[0], sr=sr)

        # freqs: cycles per second -> bpm = freqs*60
        bpm_bins = freqs * 60.0

        # Geçerli aralık filtresi
        valid = (bpm_bins >= 40.0) & (bpm_bins <= 220.0) & np.isfinite(tg_mean) & np.isfinite(bpm_bins)
        if not np.any(valid):
            return bpm, 0.25  # en azından bir “zayıf” confidence

        tg_v = tg_mean[valid]
        bpm_v = bpm_bins[valid]

        # En yakın bin
        idx = int(np.argmin(np.abs(bpm_v - bpm)))
        peak1 = float(tg_v[idx])

        # İkinci en büyük peak (aynı çevredeki binleri “yakın” saymayalım)
        tg_masked = tg_v.copy()
        neighborhood = 2
        lo = max(0, idx - neighborhood)
        hi = min(len(tg_masked), idx + neighborhood + 1)
        tg_masked[lo:hi] = -np.inf
        peak2 = float(np.max(tg_masked)) if np.any(np.isfinite(tg_masked)) else 0.0

        # Peak keskinliği + ayrışma
        # - peak_ratio: peak1 / (peak1+peak2)
        # - peak_prom: (peak1 - peak2) / (peak1 + eps)
        eps = 1e-9
        peak_ratio = peak1 / (peak1 + max(peak2, 0.0) + eps)
        peak_prom = (peak1 - max(peak2, 0.0)) / (abs(peak1) + eps)

        # Onset enerjisi de etki etsin (çok düşükse confidence düşür)
        onset_power = float(np.mean(onset_env))
        onset_norm = _clamp(onset_power / (np.percentile(onset_env, 95) + eps))

        conf = 0.55 * _clamp(peak_ratio) + 0.35 * _clamp(peak_prom) + 0.10 * _clamp(onset_norm)
        return bpm, _clamp(conf)

    except Exception:
        # Tempogram hesaplanamazsa yine de bpm dön, confidence düşük
        return bpm, 0.35

