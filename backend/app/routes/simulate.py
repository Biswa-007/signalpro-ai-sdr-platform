"""
Simulate Route
==============
POST /simulate endpoint for one-shot signal simulation.

Accepts simulation parameters as a JSON body, runs the full DSP pipeline,
and returns all signal data for frontend visualization.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.dsp_pipeline import run_simulation

router = APIRouter()


class SimulationRequest(BaseModel):
    """Request body for the /simulate endpoint."""

    carrier_freq: float = Field(
        default=200.0, ge=50.0, le=2000.0,
        description="Carrier frequency in Hz (50–2000)."
    )
    message_freq: float = Field(
        default=10.0, ge=1.0, le=100.0,
        description="Message frequency in Hz (1–100)."
    )
    noise_level: float = Field(
        default=0.1, ge=0.0, le=2.0,
        description="Noise standard deviation (0–2)."
    )
    interference_freq: float = Field(
        default=0.0, ge=0.0, le=2000.0,
        description="Interference frequency in Hz (0 = disabled)."
    )
    interference_amp: float = Field(
        default=0.0, ge=0.0, le=1.5,
        description="Interference amplitude (0 = disabled)."
    )
    filter_type: str = Field(
        default="none",
        description="Filter type: 'none', 'lowpass', or 'highpass'."
    )


@router.post("/simulate")
async def simulate(request: SimulationRequest):
    """
    Run the complete SDR simulation pipeline.

    Pipeline:
        1. Generate time array + message + carrier
        2. FM modulation
        3. Add interference (if configured)
        4. Add Gaussian noise
        5. Apply digital filter (if configured)
        6. Compute FFT + waterfall row
        7. Demodulate (FM)
        8. ML classification
        9. Calculate SNR

    Returns all signals, FFT data, waterfall row, classification, and metadata.
    """
    result = run_simulation(
        carrier_freq=request.carrier_freq,
        message_freq=request.message_freq,
        noise_level=request.noise_level,
        interference_freq=request.interference_freq,
        interference_amp=request.interference_amp,
        filter_type=request.filter_type,
    )
    return result
