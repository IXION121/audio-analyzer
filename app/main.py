from fastapi import FastAPI
from app.api.routes import router as api_router
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="Audio Analyzer API",
    version="0.1.0",
    description="MP3 upload -> track intelligence (genre/bpm/mood/instruments...)"
)

app.include_router(api_router)
