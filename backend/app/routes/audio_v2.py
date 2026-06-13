"""
Audio Analysis Routes — Production API
=======================================
POST /api/analyze   — Upload any audio file → get full analysis
POST /api/image-to-signal — Convert image to signal waveform

Supports: WAV, MP3, FLAC, OGG, M4A, AIFF and more via librosa.
"""

import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.audio_processing import analyze_audio
from app.services.image_signal import image_to_signal

router = APIRouter(prefix="/api", tags=["Signal Analyzer"])
logger = logging.getLogger(__name__)

ALLOWED_AUDIO_EXTENSIONS = (
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac",
    ".wma", ".aiff", ".opus", ".webm",
)


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, OGG, etc.)"),
    method: str = Form(default="combined"),
):
    """
    Analyze an uploaded audio file.

    Pipeline:
        1. Load audio (any format via librosa)
        2. Normalize
        3. Apply noise reduction (method: combined/spectral/lowpass/wiener)
        4. Compute FFT + Spectrogram
        5. Return waveform data + base64 audio + spectrogram

    Returns JSON with original_signal, cleaned_signal, spectrogram, FFT,
    original_audio (base64), cleaned_audio (base64).
    """
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(ALLOWED_AUDIO_EXTENSIONS)}"
        )

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        if len(audio_bytes) > 50 * 1024 * 1024:  # 50 MB limit
            raise HTTPException(status_code=400, detail="File too large (max 50 MB)")

        logger.info(f"Analyzing: {file.filename} ({len(audio_bytes)} bytes), method={method}")

        result = analyze_audio(audio_bytes, filename=file.filename, method=method)

        logger.info(
            f"Analysis complete: {result['duration_seconds']}s, "
            f"SR={result['sample_rate']}, method={method}"
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio analysis error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {str(e)}")


@router.post("/image-to-signal")
async def convert_image_to_signal(
    file: UploadFile = File(..., description="Image file (PNG, JPG, etc.)"),
):
    """Convert an uploaded image to a 1D signal representation."""
    allowed_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff")
    if not file.filename.lower().endswith(allowed_ext):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format. Allowed: {', '.join(allowed_ext)}"
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        result = image_to_signal(image_bytes)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image-to-signal error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image conversion failed: {str(e)}")
