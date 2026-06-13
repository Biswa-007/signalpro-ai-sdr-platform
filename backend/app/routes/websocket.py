"""
WebSocket Stream Route
======================
WS /ws/stream endpoint for real-time signal streaming.

Protocol:
    1. Client connects to ws://host:port/ws/stream
    2. Server starts streaming signal frames at ~100ms intervals
    3. Client can send JSON messages to update parameters in real-time:
       {"carrier_freq": 300, "message_freq": 20, "noise_level": 0.5, ...}
    4. Server adjusts the DSP pipeline parameters on the fly
    5. Each frame includes: time-domain, FFT, waterfall row, demodulated, classification

The waterfall buffer is maintained server-side (rolling 50 rows).
Each frame sends the latest waterfall row; the frontend accumulates them.
"""

import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.dsp_pipeline import run_simulation

router = APIRouter()
logger = logging.getLogger(__name__)

# Default streaming parameters
DEFAULT_PARAMS = {
    "carrier_freq": 200.0,
    "message_freq": 10.0,
    "noise_level": 0.1,
    "interference_freq": 0.0,
    "interference_amp": 0.0,
    "filter_type": "none",
}

STREAM_INTERVAL = 0.1  # 100ms between frames → ~10 FPS


@router.websocket("/ws/stream")
async def stream_signals(websocket: WebSocket):
    """
    WebSocket endpoint for real-time signal streaming.

    Continuously generates and sends signal frames while the connection is alive.
    Accepts parameter updates from the client as JSON messages.
    """
    await websocket.accept()
    logger.info("WebSocket client connected")

    # Local copy of parameters (mutable per-connection)
    params = dict(DEFAULT_PARAMS)

    try:
        while True:
            # ── Check for incoming parameter updates (non-blocking) ──────
            try:
                # Wait a tiny bit for a message, then continue if none
                raw_message = await asyncio.wait_for(
                    websocket.receive_text(), timeout=STREAM_INTERVAL
                )
                try:
                    update = json.loads(raw_message)
                    # Merge valid fields into current params
                    for key in DEFAULT_PARAMS:
                        if key in update:
                            params[key] = float(update[key]) if key != "filter_type" else str(update[key])
                    logger.debug(f"Params updated: {params}")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client: {raw_message}")

            except asyncio.TimeoutError:
                # No message received — that's fine, just generate next frame
                pass

            # ── Generate signal frame ────────────────────────────────────
            result = run_simulation(**params)

            # ── Send the frame as JSON ───────────────────────────────────
            await websocket.send_json(result)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
