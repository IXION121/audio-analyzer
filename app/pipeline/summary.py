# app/pipeline/summary.py
from __future__ import annotations

from typing import Any, Dict, List


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _top_labels(dist: List[Dict[str, Any]], k: int = 3) -> List[str]:
    # dist: [{"label":..., "score":...}, ...]
    if not dist:
        return []
    # zaten skor sıralı geliyorsa bile garanti için sort
    dist_sorted = sorted(dist, key=lambda x: _safe_float(x.get("score", 0.0)), reverse=True)
    return [str(x.get("label", "")).strip() for x in dist_sorted[:k] if str(x.get("label", "")).strip()]


def build_ai_summary(
    *,
    bpm: float,
    bpm_conf: float,
    key_name: str,
    key_scale: str,
    genre: Dict[str, Any],
    mood: Dict[str, Any],
    audio_features: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Amaç: Mevcut model çıktılarından (rule-based) okunabilir bir 'AI summary' üretmek.
    LLM yok. Tamamen deterministik ve hızlı.
    """

    # ---- Extract basics
    energy = _clamp(_safe_float(audio_features.get("energy"), 0.0))
    danceability = _clamp(_safe_float(audio_features.get("danceability"), 0.0))
    acousticness = _clamp(_safe_float(audio_features.get("acousticness"), 0.0))
    speechiness = _clamp(_safe_float(audio_features.get("speechiness"), 0.0))

    valence = _clamp(_safe_float(mood.get("valence"), 0.5))
    arousal = _clamp(_safe_float(mood.get("arousal"), 0.5))

    genre_top = str(genre.get("top", "unknown"))
    genre_conf = _clamp(_safe_float(genre.get("confidence"), 0.0))
    genre_alt = _top_labels(genre.get("distribution", []), k=3)

    mood_tags = _top_labels(mood.get("tags", []), k=3)

    bpm = _safe_float(bpm, 0.0)
    bpm_conf = _clamp(_safe_float(bpm_conf, 0.0))

    # ---- Tempo label
    tempo_label = "unknown"
    if bpm > 0:
        if bpm < 70:
            tempo_label = "very_slow"
        elif bpm < 95:
            tempo_label = "slow"
        elif bpm < 125:
            tempo_label = "mid"
        elif bpm < 155:
            tempo_label = "fast"
        else:
            tempo_label = "very_fast"

    # ---- Vibe tags (basit ama işe yarar)
    vibe: List[str] = []

    if energy >= 0.70 or arousal >= 0.70:
        vibe.append("energetic")
    if valence >= 0.65:
        vibe.append("uplifting")
    if valence <= 0.35:
        vibe.append("melancholic")
    if acousticness >= 0.60:
        vibe.append("acoustic")
    if danceability >= 0.70 and bpm >= 110:
        vibe.append("danceable")
    if speechiness >= 0.55:
        vibe.append("talky_or_rap")

    # mood tag’lerini de vibe’a yedir (tekrar etmeyecek şekilde)
    for t in mood_tags:
        if t and t not in vibe:
            vibe.append(t)

    if not vibe:
        vibe = ["neutral"]

    # ---- “Commercial / playlist fit” benzeri skorlar (heuristic)
    # Not: bunlar “tahmin” değil; sadece feature tabanlı yardımcı skorlar.
    club_score = _clamp(0.45 * danceability + 0.35 * energy + 0.20 * (1.0 if bpm >= 115 else 0.0))
    chill_score = _clamp(0.50 * acousticness + 0.30 * (1.0 - arousal) + 0.20 * (1.0 - energy))
    focus_score = _clamp(0.45 * (1.0 - speechiness) + 0.35 * acousticness + 0.20 * (1.0 - arousal))

    # ---- Confidence overall (genre+bpm bazlı basit birleşim)
    overall_conf = _clamp(0.55 * genre_conf + 0.45 * bpm_conf)

    # ---- Text summary
    key_part = ""
    if key_name and key_name != "unknown":
        key_part = f"{key_name} {key_scale}".strip()

    top_genres_text = ", ".join([g for g in [genre_top] + genre_alt if g and g != "unknown"][:3]) or "unknown"

    summary_text = (
        f"Style leans toward {top_genres_text}. "
        f"Tempo is {tempo_label} ({bpm:.1f} BPM). "
        f"Energy {energy:.2f}, danceability {danceability:.2f}, valence {valence:.2f}, arousal {arousal:.2f}."
    )
    if key_part:
        summary_text += f" Key: {key_part}."

    return {
        "text": summary_text,
        "vibe": vibe,
        "tempo_label": tempo_label,
        "top_genres": [genre_top] + genre_alt,
        "top_moods": mood_tags,
        "scores": {
            "club": club_score,
            "chill": chill_score,
            "focus": focus_score,
        },
        "confidence": {
            "overall": overall_conf,
            "genre": genre_conf,
            "tempo": bpm_conf,
        },
    }
