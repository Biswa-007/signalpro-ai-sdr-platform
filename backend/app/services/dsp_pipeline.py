"""
DSP Pipeline Service
====================
Orchestrates the full signal processing pipeline from generation to classification.

This service is called by both the REST and WebSocket routes.
It runs the complete chain:
    1. Signal generation (message + carrier)
    2. FM modulation
    3. Multi-signal interference mixing (optional)
    4. Additive Gaussian noise
    5. Digital filtering (optional: low-pass / high-pass)  ← BEFORE demodulation
    6. Noise gate (post-filter cleanup)
    7. FFT computation
    8. Waterfall row generation
    9. FM demodulation (on filtered signal)
    10. Post-demodulation normalization
    11. ML classification
    12. SNR calculation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dsp.signal_generation import generate_time_array, generate_message_signal, generate_carrier_signal
from dsp.modulation import fm_modulate
from dsp.noise import add_gaussian_noise, calculate_snr
from dsp.fft_analysis import compute_fft, compute_waterfall_row
from dsp.demodulation import fm_demodulate
from dsp.filters import apply_lowpass_filter, apply_highpass_filter
from dsp.interference import generate_interference, mix_signals
from app.ml.classifier import classifier

# ─── Constants ───────────────────────────────────────────────────────────────
SAMPLE_RATE = 10000
DURATION = 0.05
DOWNSAMPLE = 2
WATERFALL_BINS = 128
NOISE_GATE_THRESHOLD = 0.02  # Noise gate threshold


def downsample_array(arr: np.ndarray, factor: int = DOWNSAMPLE) -> list:
    """Reduce array size by taking every Nth sample, then convert to list."""
    return arr[::factor].tolist()


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1.0, 1.0] to avoid clipping."""
    peak = np.max(np.abs(signal))
    if peak > 1e-10:
        return signal / peak
    return signal.copy()


def apply_noise_gate(signal: np.ndarray, threshold: float = NOISE_GATE_THRESHOLD) -> np.ndarray:
    """
    Apply a soft noise gate to suppress low-amplitude noise.

    Samples below the threshold are smoothly attenuated rather than
    hard-zeroed to avoid clicks and artifacts.
    """
    abs_signal = np.abs(signal)
    # Soft gate: smooth transition near threshold
    gain = np.where(
        abs_signal > threshold,
        1.0,
        (abs_signal / threshold) ** 2  # quadratic taper
    )
    return signal * gain


def run_simulation(
    carrier_freq: float = 200.0,
    message_freq: float = 10.0,
    noise_level: float = 0.1,
    interference_freq: float = 0.0,
    interference_amp: float = 0.0,
    filter_type: str = "none",
) -> dict:
    """
    Execute the complete DSP pipeline and return all signal data.

    Args:
        carrier_freq: Carrier frequency in Hz (50–2000).
        message_freq: Message frequency in Hz (1–100).
        noise_level: Noise standard deviation (0–2).
        interference_freq: Interferer frequency in Hz (0 = disabled).
        interference_amp: Interferer amplitude (0 = disabled).
        filter_type: "none", "lowpass", or "highpass".

    Returns:
        Dict containing all signal arrays, FFT data, waterfall row,
        demodulated signal, classification result, and metadata.
    """
    # ── Step 1: Generate time base ──────────────────────────────────────
    t = generate_time_array(duration=DURATION, sample_rate=SAMPLE_RATE)

    # ── Step 2: Generate message signal ─────────────────────────────────
    message = generate_message_signal(t, frequency=message_freq, amplitude=1.0)

    # ── Step 3: Generate carrier signal (for reference) ─────────────────
    carrier = generate_carrier_signal(t, frequency=carrier_freq, amplitude=1.0)

    # ── Step 4: Apply FM modulation ─────────────────────────────────────
    fm_signal = fm_modulate(
        t, message,
        carrier_freq=carrier_freq,
        carrier_amplitude=1.0,
        freq_deviation=carrier_freq * 0.5,
    )

    # ── Step 5: Add interference (if enabled) ───────────────────────────
    if interference_amp > 0.01 and interference_freq > 0:
        interf = generate_interference(t, frequency=interference_freq, amplitude=interference_amp)
        signal_with_interf = mix_signals(fm_signal, interf)
    else:
        signal_with_interf = fm_signal.copy()

    # ── Step 6: Add noise ───────────────────────────────────────────────
    noisy_signal = add_gaussian_noise(signal_with_interf, noise_level=noise_level)

    # ── Step 7: Apply digital filter BEFORE demodulation ────────────────
    filtered_signal = noisy_signal.copy()
    if filter_type == "lowpass":
        filtered_signal = apply_lowpass_filter(
            noisy_signal, cutoff=carrier_freq * 1.5, sample_rate=SAMPLE_RATE
        )
    elif filter_type == "highpass":
        filtered_signal = apply_highpass_filter(
            noisy_signal, cutoff=carrier_freq * 0.3, sample_rate=SAMPLE_RATE
        )

    # Normalize filtered signal to prevent clipping
    filtered_signal = normalize_signal(filtered_signal)

    # ── Step 7.5: Apply noise gate AFTER filtering ──────────────────────
    gate_threshold = max(0.01, noise_level * 0.05)
    filtered_signal = apply_noise_gate(filtered_signal, threshold=gate_threshold)

    # Re-normalize after noise gate
    filtered_signal = normalize_signal(filtered_signal)

    # ── Step 8: Compute FFT (on filtered signal) ────────────────────────
    fft_freqs, fft_magnitudes = compute_fft(filtered_signal, sample_rate=SAMPLE_RATE)

    # ── Step 9: Compute waterfall row ───────────────────────────────────
    waterfall_freqs, waterfall_mags = compute_waterfall_row(
        filtered_signal, sample_rate=SAMPLE_RATE, n_bins=WATERFALL_BINS
    )

    # ── Step 10: Demodulate (on filtered signal) ────────────────────────
    demodulated = fm_demodulate(filtered_signal, sample_rate=SAMPLE_RATE)

    # Normalize demodulated signal to avoid distortion
    demodulated = normalize_signal(demodulated)

    # ── Step 11: Calculate SNR ──────────────────────────────────────────
    snr = calculate_snr(fm_signal, noise_level) if noise_level > 0 else float("inf")

    # ── Step 12: ML Classification ──────────────────────────────────────
    classification = classifier.predict(filtered_signal, fft_magnitudes)

    # ── Step 13: Prepare response ───────────────────────────────────────
    return {
        "time": downsample_array(t),
        "message_signal": downsample_array(message),
        "carrier_signal": downsample_array(carrier),
        "fm_signal": downsample_array(fm_signal),
        "noisy_signal": downsample_array(noisy_signal),
        "filtered_signal": downsample_array(filtered_signal),
        "fft_frequencies": downsample_array(fft_freqs),
        "fft_magnitudes": downsample_array(fft_magnitudes),
        "waterfall_frequencies": waterfall_freqs.tolist(),
        "waterfall_magnitudes": waterfall_mags.tolist(),
        "demodulated": downsample_array(demodulated),
        "snr_db": round(float(snr), 2) if not np.isinf(snr) else None,
        "classification": classification,
        "params": {
            "carrier_freq": carrier_freq,
            "message_freq": message_freq,
            "noise_level": noise_level,
            "interference_freq": interference_freq,
            "interference_amp": interference_amp,
            "filter_type": filter_type,
            "sample_rate": SAMPLE_RATE,
            "duration": DURATION,
        },
    }
