from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.api.schemas import AnalyzeResponse
from app.services.analyzer_service import AnalyzerService

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    preset: str = Query("full", pattern="^(fast|full)$"),
    include_instruments: bool = Query(True),
    include_segments: bool = Query(False),
):
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload mp3/wav/m4a/flac/ogg")

    service = AnalyzerService()
    result = await service.analyze_upload(
        upload=file,
        preset=preset,
        include_instruments=include_instruments,
        include_segments=include_segments,
    )
    return result
