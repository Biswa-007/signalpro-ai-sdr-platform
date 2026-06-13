"""
Audio Processing Route
======================
POST /audio-process endpoint for audio noise addition and removal.

Accepts:
    - WAV file upload (multipart form)
    - noise_level (float, 0.0–1.0)
    - denoise_method ("lowpass" | "spectral" | "wiener" | "combined")

Returns JSON with:
    - original, noisy, cleaned waveform arrays (downsampled for display)
    - sample_rate
    - duration_seconds

Also provides endpoints to download the processed WAV files:
    - GET /audio/noisy.wav
    - GET /audio/cleaned.wav
"""

import logging
import base64
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from dsp.audio_processing import (
    load_wav_bytes,
    normalize_signal,
    add_noise_to_audio,
    denoise_audio,
    signal_to_wav_bytes,
    downsample_for_display,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# ─── In-memory cache for last processed audio (for download endpoints) ───────
_audio_cache = {
    "noisy_wav": None,
    "cleaned_wav": None,
}


@router.post("/audio-process")
async def process_audio(
    file: UploadFile = File(..., description="WAV audio file to process"),
    noise_level: float = Form(default=0.15, ge=0.0, le=1.0),
    denoise_method: str = Form(default="combined"),
):
    """
    Process an uploaded audio file: add noise, then denoise.

    Pipeline:
        1. Load WAV → numpy array
        2. Normalize to [-1, 1]
        3. Add Gaussian noise at specified level
        4. Apply denoising filter
        5. Return waveform data + base64-encoded WAV files

    Args:
        file: Uploaded WAV file.
        noise_level: Noise intensity (0.0=clean, 1.0=very noisy).
        denoise_method: "lowpass", "spectral", "wiener", or "combined".
    """
    # Validate file type
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    try:
        # Read the uploaded file
        wav_bytes = await file.read()
        if len(wav_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        logger.info(f"Processing audio: {file.filename} ({len(wav_bytes)} bytes)")

        # Step 1: Load WAV
        original, sample_rate = load_wav_bytes(wav_bytes)

        # Limit length to prevent memory issues (max 30 seconds)
        max_samples = sample_rate * 30
        if len(original) > max_samples:
            original = original[:max_samples]
            logger.info(f"Audio trimmed to 30 seconds ({max_samples} samples)")

        # Step 2: Normalize
        original = normalize_signal(original)

        # Step 3: Add noise
        noisy = add_noise_to_audio(original, noise_level=noise_level)

        # Step 4: Denoise
        cleaned = denoise_audio(noisy, sample_rate, method=denoise_method)

        # Step 5: Generate WAV files for playback
        noisy_wav = signal_to_wav_bytes(noisy, sample_rate)
        cleaned_wav = signal_to_wav_bytes(cleaned, sample_rate)

        # Cache for download endpoints
        _audio_cache["noisy_wav"] = noisy_wav
        _audio_cache["cleaned_wav"] = cleaned_wav

        # Step 6: Encode WAVs as base64 for inline playback
        noisy_b64 = base64.b64encode(noisy_wav).decode("ascii")
        cleaned_b64 = base64.b64encode(cleaned_wav).decode("ascii")

        # Step 7: Downsample for chart display
        duration = len(original) / sample_rate

        # Generate time arrays
        max_display_points = 2000
        original_display = downsample_for_display(original, max_display_points)
        noisy_display = downsample_for_display(noisy, max_display_points)
        cleaned_display = downsample_for_display(cleaned, max_display_points)

        # Time axis for display
        display_len = len(original_display)
        time_display = [round(i * duration / display_len, 6) for i in range(display_len)]

        logger.info(
            f"Audio processed: {duration:.2f}s, SR={sample_rate}, "
            f"method={denoise_method}, noise={noise_level}"
        )

        return {
            "time": time_display,
            "original": original_display,
            "noisy": noisy_display,
            "cleaned": cleaned_display,
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 3),
            "total_samples": len(original),
            "noise_level": noise_level,
            "denoise_method": denoise_method,
            "noisy_wav_b64": noisy_b64,
            "cleaned_wav_b64": cleaned_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


@router.get("/audio/noisy.wav")
async def download_noisy_wav():
    """Download the last processed noisy audio as a WAV file."""
    if _audio_cache["noisy_wav"] is None:
        raise HTTPException(status_code=404, detail="No processed audio available")
    return Response(
        content=_audio_cache["noisy_wav"],
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=noisy.wav"},
    )


@router.get("/audio/cleaned.wav")
async def download_cleaned_wav():
    """Download the last processed cleaned audio as a WAV file."""
    if _audio_cache["cleaned_wav"] is None:
        raise HTTPException(status_code=404, detail="No processed audio available")
    return Response(
        content=_audio_cache["cleaned_wav"],
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=cleaned.wav"},
    )
