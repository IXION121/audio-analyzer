import os
import time
import uuid
import logging
from fastapi import UploadFile
from app.core.config import settings
from app.pipeline.decode import decode_to_wav
import librosa

from app.pipeline.tempo import estimate_bpm_and_confidence
from app.pipeline.key import estimate_key_and_confidence
from app.pipeline.features import compute_audio_features
from app.pipeline.genre_mood_from_wav import predict_genre_and_mood_from_wav
from app.pipeline.summary import build_ai_summary
from app.pipeline.loudness import compute_lufs

log = logging.getLogger("analyzer")


class AnalyzerService:
    async def analyze_upload(
        self,
        upload: UploadFile,
        preset: str,
        include_instruments: bool,
        include_segments: bool,
    ) -> dict:
        t0 = time.perf_counter()

        os.makedirs(settings.tmp_dir, exist_ok=True)
        job_id = uuid.uuid4().hex

        # 1) Save upload
        in_path = os.path.join(settings.tmp_dir, f"{job_id}_{upload.filename}")
        with open(in_path, "wb") as f:
            f.write(await upload.read())

        # 2) Decode to wav (mono)
        wav_path = os.path.join(settings.tmp_dir, f"{job_id}.wav")
        decoded = decode_to_wav(in_path, wav_path, sample_rate=settings.target_sr)

        # 3) Load audio
        y, sr = librosa.load(decoded.wav_path, sr=decoded.sample_rate, mono=True)
        duration = float(librosa.get_duration(y=y, sr=sr))

        # 4) Core analysis
        bpm, bpm_conf = estimate_bpm_and_confidence(y=y, sr=sr)
        key_name, key_scale, key_conf = estimate_key_and_confidence(y=y, sr=sr)
        features = compute_audio_features(y=y, sr=sr, bpm=bpm, bpm_conf=bpm_conf)

        # 5) Genre + mood (ML if available)
        warnings_list: list[str] = []
        gm = predict_genre_and_mood_from_wav(decoded.wav_path)
        warnings_list.extend(gm.get("warnings", []))

        # 6) Build result
        ai=summary = build_ai_summary(
            bpm=float(bpm),
            bpm_conf=float(bpm_conf),
            key_name=key_name,
            key_scale=key_scale,
            genre=gm["genre"],
            mood=gm["mood"],
            audio_features=features,
        )
        # 7) Compute LUFS

        lufs, lufs_warnings = compute_lufs(decoded.wav_path)
        warnings_list.extend(lufs_warnings)

        result = {
            "track": {"duration_sec": duration, "sample_rate": sr},
            "tempo": {"bpm": float(bpm), "confidence": float(bpm_conf)},
            "key": {"key": key_name, "scale": key_scale, "confidence": float(key_conf)},

            # IMPORTANT: gm kullan
            "genre": gm["genre"],
            "mood": gm["mood"],

            "instruments": None if not include_instruments else {
                "top": [{"label": "unknown", "score": 1.0}],
                "distribution": [{"label": "unknown", "score": 1.0}],
            },

            "audio_features": {
                "loudness_lufs": lufs,
                "loudness_proxy_db": features.get("loudness_proxy_db"),
                "loudness_norm": features.get("loudness_norm"),
                "energy": features.get("energy"),
                "danceability": features.get("danceability"),
                "acousticness": features.get("acousticness"),
                "speechiness": features.get("speechiness"),
                "spectral_centroid_hz": features.get("spectral_centroid_hz"),
                "spectral_rolloff_hz": features.get("spectral_rolloff_hz"),
                "spectral_flatness": features.get("spectral_flatness"),
                "zcr": features.get("zcr"),
            },

            "segments": None if not include_segments else {"beats_count": None, "sections": None},

            "meta": {
                "processing_ms": int((time.perf_counter() - t0) * 1000),
                "warnings": warnings_list,
            },
            "ai_summary": summary,
        }

        # cleanup
        for p in (in_path, decoded.wav_path):
            try:
                os.remove(p)
            except Exception:
                pass

        log.info("analyze complete job=%s ms=%s", job_id, result["meta"]["processing_ms"])
        return result
