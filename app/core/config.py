from pydantic import BaseModel

class Settings(BaseModel):
    tmp_dir: str = "/tmp/audio-analyzer"
    target_sr: int = 44100

settings = Settings()
