import os
import subprocess
from dataclasses import dataclass

@dataclass
class DecodedAudio:
    wav_path: str
    sample_rate: int

def decode_to_wav(input_path: str, output_wav_path: str, sample_rate: int = 44100) -> DecodedAudio:
    """
    mono wav PCM 16-bit.
    """
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        "-vn",
        "-f", "wav",
        "-acodec", "pcm_s16le",
        output_wav_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {p.stderr[-1000:]}")

    return DecodedAudio(wav_path=output_wav_path, sample_rate=sample_rate)
