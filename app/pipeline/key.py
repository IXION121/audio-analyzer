import numpy as np
import librosa

# Krumhansl-Schmuckler key profiles
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.std(x) + 1e-9
    return (x - np.mean(x)) / s

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))

def estimate_key_and_confidence(y: np.ndarray, sr: int) -> tuple[str, str, float]:
    """
    Returns (key_name, scale, confidence).
    scale: "major" | "minor" | "unknown"
    confidence in [0,1] from correlation separation.
    """
    if y is None or len(y) < sr * 6:  # çok kısa parçada key sallanır
        return "unknown", "unknown", 0.0

    # Harmonik bileşenle çalışmak genelde daha stabil
    try:
        y_harm, _ = librosa.effects.hpss(y)
    except Exception:
        y_harm = y

    # Chroma (CQT genelde tonalite için iyi)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # sessiz/boş parça kontrolü
    if not np.isfinite(chroma_mean).all() or np.sum(chroma_mean) < 1e-6:
        return "unknown", "unknown", 0.0

    # Normalize
    chroma_n = _z(chroma_mean)
    maj = _z(KRUMHANSL_MAJOR)
    minr = _z(KRUMHANSL_MINOR)

    # 12 transpozisyon için major/minor skorları
    maj_scores = []
    min_scores = []
    for i in range(12):
        rolled = np.roll(chroma_n, -i)  # i kadar transpoze
        maj_scores.append(float(np.dot(rolled, maj)))
        min_scores.append(float(np.dot(rolled, minr)))

    maj_scores = np.array(maj_scores, dtype=float)
    min_scores = np.array(min_scores, dtype=float)

    # En iyi aday
    best_maj_i = int(np.argmax(maj_scores))
    best_min_i = int(np.argmax(min_scores))
    best_maj = float(maj_scores[best_maj_i])
    best_min = float(min_scores[best_min_i])

    # Hangisi daha iyi?
    if best_maj >= best_min:
        key = KEY_NAMES[best_maj_i]
        scale = "major"
        best = best_maj
        all_scores = np.concatenate([maj_scores, min_scores])
        best_idx_global = best_maj_i
    else:
        key = KEY_NAMES[best_min_i]
        scale = "minor"
        best = best_min
        all_scores = np.concatenate([maj_scores, min_scores])
        best_idx_global = 12 + best_min_i

    # Confidence: top1-top2 ayrımı + skorun “ne kadar iyi” olduğu
    # Top2'yi bul (aynı index hariç)
    all_scores2 = all_scores.copy()
    all_scores2[best_idx_global] = -np.inf
    second = float(np.max(all_scores2))

    # separation 0..1 gibi bir şeye map et
    # - sep: (best-second) / (abs(best)+eps)
    # - strength: sigmoid(best) benzeri
    eps = 1e-9
    sep = (best - second) / (abs(best) + eps)
    sep = _clamp((sep + 0.2) / 1.2)  # kaba ölçekleme (pratik)

    # best skor negatifse confidence düşsün
    strength = 1.0 / (1.0 + np.exp(-best / 2.5))  # 0..1
    conf = _clamp(0.65 * sep + 0.35 * float(strength))

    # aşırı düşükse unknown'a çek
    if conf < 0.25:
        return "unknown", "unknown", conf

    return key, scale, conf
