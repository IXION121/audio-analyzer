# app/pipeline/genre_mood_from_wav.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np


# musicnn çıktı etiketleri içinde sık geçen genre/mood alt-kümeleri
GENRE_TAGS = [
    "rock", "pop", "alternative", "indie", "electronic", "hiphop", "rap", "jazz", "blues",
    "classical", "metal", "punk", "reggae", "latin", "country", "folk", "soul", "funk",
    "rnb", "dance"
]

MOOD_TAGS = [
    "happy", "sad", "relax", "aggressive", "party", "dark", "romantic", "epic"
]


def _top_and_dist(
    tags: List[str],
    scores: np.ndarray,
    topk: int = 8
) -> Tuple[str, float, List[Dict[str, float]]]:
    scores = scores.astype(np.float64)
    scores = np.clip(scores, 0.0, None)

    s = scores / (np.sum(scores) + 1e-12)  # normalize
    order = np.argsort(-s)

    top_idx = int(order[0])
    top_label = tags[top_idx]
    top_conf = float(s[top_idx])

    dist = [{"label": tags[int(i)], "score": float(s[int(i)])} for i in order[:topk]]
    return top_label, top_conf, dist


def predict_genre_and_mood_from_wav(wav_path: str) -> Dict[str, Any]:
    warnings: List[str] = []

    try:
        from musicnn.extractor import extractor
    except Exception as e:
        return {
            "genre": {"top": "unknown", "confidence": 0.0, "distribution": [{"label": "unknown", "score": 1.0}]},
            "mood": {"valence": 0.5, "arousal": 0.5, "tags": [{"label": "unknown", "score": 1.0}]},
            "warnings": [f"musicnn not available: {e}"],
        }

    try:
        out = extractor(wav_path, model="MSD_musicnn", input_length=3, input_overlap=False)

        # musicnn sürümüne göre dönüş: (taggram, tags) veya (taggram, tags, features) vb.
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            taggram, tags = out[0], out[1]
        else:
            raise RuntimeError(f"musicnn extractor returned unexpected value: {type(out)} len={getattr(out, '__len__', None)}")

        taggram = np.asarray(taggram, dtype=np.float64)
        if taggram.ndim != 2 or taggram.shape[0] == 0:
            raise RuntimeError(f"musicnn taggram invalid shape: {taggram.shape}")

        avg = taggram.mean(axis=0)  # (n_tags,)
        tags = [str(t).lower() for t in tags]

    except Exception as e:
        return {
            "genre": {"top": "unknown", "confidence": 0.0, "distribution": [{"label": "unknown", "score": 1.0}]},
            "mood": {"valence": 0.5, "arousal": 0.5, "tags": [{"label": "unknown", "score": 1.0}]},
            "warnings": [f"musicnn extractor failed: {e}"],
        }

    tag_to_score = {tags[i]: float(avg[i]) for i in range(min(len(tags), len(avg)))}

    # genre scores
    genre_scores = np.array([tag_to_score.get(t, 0.0) for t in GENRE_TAGS], dtype=np.float64)
    g_top, g_conf, g_dist = _top_and_dist(GENRE_TAGS, genre_scores, topk=10)

    # mood scores
    mood_scores = np.array([tag_to_score.get(t, 0.0) for t in MOOD_TAGS], dtype=np.float64)
    _, _, m_dist = _top_and_dist(MOOD_TAGS, mood_scores, topk=6)

    # basit valence/arousal heuristiği (0..1 aralığı)
    valence = 0.5 + 0.5 * (
        tag_to_score.get("happy", 0.0) +
        tag_to_score.get("party", 0.0) +
        tag_to_score.get("romantic", 0.0) -
        tag_to_score.get("sad", 0.0) -
        tag_to_score.get("dark", 0.0)
    )
    arousal = 0.5 + 0.5 * (
        tag_to_score.get("aggressive", 0.0) +
        tag_to_score.get("party", 0.0) +
        tag_to_score.get("epic", 0.0) -
        tag_to_score.get("relax", 0.0)
    )

    valence = float(np.clip(valence, 0.0, 1.0))
    arousal = float(np.clip(arousal, 0.0, 1.0))

    return {
        "genre": {
            "top": g_top,
            "confidence": g_conf,
            "distribution": g_dist,
        },
        "mood": {
            "valence": valence,
            "arousal": arousal,
            "tags": m_dist,
        },
        "warnings": warnings,
    }
