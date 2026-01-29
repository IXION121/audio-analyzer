from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class LabelScore(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)

class TrackInfo(BaseModel):
    duration_sec: float
    sample_rate: int

class TempoInfo(BaseModel):
    bpm: float
    confidence: float = Field(ge=0.0, le=1.0)

class KeyInfo(BaseModel):
    key: str
    scale: Literal["major", "minor", "unknown"] = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)

class GenreInfo(BaseModel):
    top: str
    confidence: float = Field(ge=0.0, le=1.0)
    distribution: List[LabelScore]

class MoodInfo(BaseModel):
    valence: float = Field(ge=0.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    tags: List[LabelScore]

class InstrumentsInfo(BaseModel):
    top: List[LabelScore]
    distribution: List[LabelScore]

class AudioFeatures(BaseModel):
    loudness_lufs: Optional[float] = None
    energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speechiness: Optional[float] = Field(default=None, ge=0.0, le=1.0)

class SegmentsInfo(BaseModel):
    beats_count: Optional[int] = None
    sections: Optional[List[dict]] = None  # sonra şema netleştiririz

class MetaInfo(BaseModel):
    processing_ms: int
    warnings: List[str] = []

class AnalyzeResponse(BaseModel):
    track: TrackInfo
    tempo: TempoInfo
    key: KeyInfo
    genre: GenreInfo
    mood: MoodInfo
    instruments: Optional[InstrumentsInfo] = None
    audio_features: AudioFeatures
    segments: Optional[SegmentsInfo] = None
    meta: MetaInfo
class AudioFeatures(BaseModel):
    loudness_lufs: Optional[float] = None

    # ek (proxy) alanlar:
    loudness_proxy_db: Optional[float] = None
    loudness_norm: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    energy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    danceability: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    acousticness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    speechiness: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # ekstra debug/analiz
    spectral_centroid_hz: Optional[float] = None
    spectral_rolloff_hz: Optional[float] = None
    spectral_flatness: Optional[float] = None
    zcr: Optional[float] = None
