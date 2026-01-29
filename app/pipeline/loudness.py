# app/pipeline/loudness.py
from __future__ import annotations

from typing import Tuple
import numpy as np
import soundfile as sf

def compute_lufs(wav_path: str) -> Tuple[float | None, list[str]]:
    """
    Integrated LUFS (EBU R128 style) hesaplar.
    Hata olursa None döner ve warnings listesi verir.
    """
    warnings: list[str] = []
    try:
        import pyloudnorm as pyln
    except Exception as e:
        return None, [f"pyloudnorm not available: {e}"]

    try:
        y, sr = sf.read(wav_path, always_2d=False)
        if y is None:
            return None, ["LUFS: could not read audio"]
        # stereo ise mono'ya indir (LUFS için stereo da yapılabilir ama basit/kararlı olsun)
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.mean(axis=1)
        y = np.asarray(y, dtype=np.float32)

        if y.size < sr * 1:  # 1 saniyeden kısa ise
            return None, ["LUFS: audio too short"]

        meter = pyln.Meter(sr)  # BS.1770 meter
        lufs = float(meter.integrated_loudness(y))
        return lufs, warnings
    except Exception as e:
        return None, [f"LUFS compute failed: {e}"]
