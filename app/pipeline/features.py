import numpy as np
import librosa

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))

def _safe_log(x: float) -> float:
    return float(np.log(max(x, 1e-12)))

def compute_audio_features(y: np.ndarray, sr: int, bpm: float | None = None, bpm_conf: float | None = None) -> dict:
    """
    Spotify/Sonoteller hissi veren "yaklaşık" features.
    (Hepsi 0..1 olacak şekilde normalize edilmeye çalışılır.)
    Not: Bunlar heuristics. Sonra gerekirse ML ile iyileştiririz.
    """

    # RMS energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_mean = float(np.mean(rms))
    rms_p95 = float(np.percentile(rms, 95))

    # Energy: RMS'i log ölçeğe alıp normalize et
    # tipik rms_mean aralığı kaba olarak 0.01-0.2
    energy = _clamp((_safe_log(rms_mean) - _safe_log(0.01)) / (_safe_log(0.20) - _safe_log(0.01)))

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    centroid_mean = float(np.mean(centroid))
    rolloff_mean = float(np.mean(rolloff))
    flatness_mean = float(np.mean(flatness))
    zcr_mean = float(np.mean(zcr))

    # Acousticness (heuristic):
    # - akustik parçalarda centroid/rolloff daha düşük, flatness daha düşük olabiliyor (çok kaba)
    # centroid: 1000-4500 Hz bandında normalize, ters çevir
    centroid_n = _clamp((centroid_mean - 1000.0) / (4500.0 - 1000.0))
    rolloff_n = _clamp((rolloff_mean - 2000.0) / (8000.0 - 2000.0))
    flatness_n = _clamp(flatness_mean)  # zaten 0..1 civarı

    acousticness = _clamp(1.0 - (0.55 * centroid_n + 0.30 * rolloff_n + 0.15 * flatness_n))

    # Speechiness (heuristic):
    # - konuşma/rap: ZCR + flatness + centroid orta-yüksek olur
    speechiness = _clamp(0.45 * _clamp(zcr_mean / 0.15) + 0.35 * flatness_n + 0.20 * centroid_n)

    # Danceability (heuristic):
    # - tempo 90-150 aralığında + bpm confidence yüksek + onset düzenli → daha dans edilebilir
    if bpm is None or bpm <= 0:
        danceability = None
    else:
        # tempo uygunluğu: 90-150 ideal (Gaussian benzeri)
        tempo_fit = float(np.exp(-((bpm - 120.0) ** 2) / (2 * (25.0 ** 2))))
        tempo_fit = _clamp(tempo_fit)

        # onset düzenliliği: onset strength tempogram peak keskinliğini zaten bpm_conf taşıyor
        conf = _clamp(float(bpm_conf or 0.0))

        # enerji çok düşükse dans edilebilirlik düşsün
        danceability = _clamp(0.55 * tempo_fit + 0.30 * conf + 0.15 * energy)

    # "Loudness" (LUFS değil, proxy):
    # RMS p95 -> dBFS proxy
    loudness_proxy = float(20.0 * np.log10(max(rms_p95, 1e-12)))  # genelde -35..-3 gibi
    # normalize: -35 -> 0, -5 -> 1
    loudness_norm = _clamp((loudness_proxy - (-35.0)) / ((-5.0) - (-35.0)))

    return {
        "loudness_lufs": None,               # şimdilik gerçek LUFS yok
        "loudness_proxy_db": loudness_proxy, # ek alan (istersen response şemasına da ekleriz)
        "loudness_norm": loudness_norm,
        "energy": energy,
        "danceability": float(danceability) if danceability is not None else None,
        "acousticness": acousticness,
        "speechiness": speechiness,
        "spectral_centroid_hz": centroid_mean,
        "spectral_rolloff_hz": rolloff_mean,
        "spectral_flatness": flatness_mean,
        "zcr": zcr_mean,
    }
