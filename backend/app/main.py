"""
SDR Simulation Platform — FastAPI Application
==============================================
Main entry point for the Real-Time AI-Powered SDR Simulation backend.

Architecture:
    app/
        main.py          ← You are here
        routes/
            simulate.py  ← POST /simulate
            websocket.py ← WS /ws/stream
        services/
            dsp_pipeline.py ← DSP orchestration
        ml/
            classifier.py   ← Signal condition classifier
    dsp/
        signal_generation.py
        modulation.py
        noise.py
        fft_analysis.py
        demodulation.py
        filters.py
        interference.py

Run with:
    cd backend
    uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.simulate import router as simulate_router
from app.routes.websocket import router as websocket_router
from app.routes.audio import router as audio_router
from app.routes.audio_v2 import router as audio_v2_router
from app.ml.classifier import classifier

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan Handler ───────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle.
    Trains the ML classifier on startup so it's ready for inference.
    """
    logger.info("=" * 60)
    logger.info("  SDR Simulation Platform — Starting Up")
    logger.info("=" * 60)

    # Train the ML model on synthetic data at startup
    classifier.train(n_samples_per_class=200)
    logger.info("ML Classifier ready for inference")

    logger.info("=" * 60)
    logger.info("  Backend is LIVE — http://localhost:8000")
    logger.info("  API Docs       — http://localhost:8000/docs")
    logger.info("  WebSocket      — ws://localhost:8000/ws/stream")
    logger.info("  Audio Process  — POST /audio-process")
    logger.info("=" * 60)

    yield  # Application runs here

    logger.info("SDR Simulation Platform — Shutting Down")


# ─── App Initialization ─────────────────────────────────────────────────────
app = FastAPI(
    title="SDR Simulation Platform",
    description=(
        "Real-Time AI-Powered Software Defined Radio simulation backend. "
        "Provides signal generation, FM modulation, interference simulation, "
        "digital filtering, FFT analysis, waterfall spectrogram data, "
        "demodulation, and ML-based signal classification."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ─── CORS Middleware ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Register Routes ────────────────────────────────────────────────────────
app.include_router(simulate_router, tags=["Simulation"])
app.include_router(websocket_router, tags=["WebSocket"])
app.include_router(audio_router, tags=["Audio Processing"])
app.include_router(audio_v2_router, tags=["Audio Processing v2"])


# ─── Health Check ────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Health check / welcome endpoint."""
    return {
        "status": "online",
        "service": "SDR Simulation Platform",
        "version": "2.0.0",
        "endpoints": {
            "simulate": "POST /simulate",
            "stream": "WS /ws/stream",
            "audio_process": "POST /audio-process",
            "docs": "GET /docs",
        },
    }
