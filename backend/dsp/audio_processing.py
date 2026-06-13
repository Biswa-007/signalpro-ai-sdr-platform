"""
Audio Processing Module
=======================
Handles loading, normalization, noise injection, and denoising of audio signals.

This module extends the SDR platform with real audio processing capabilities:
  - WAV file loading + normalization
  - Gaussian noise injection at controllable SNR
  - Multi-stage denoising pipeline:
      1. Spectral gating (frequency-domain noise floor removal)
      2. Butterworth low-pass filtering (remove HF noise)
  - WAV export with proper clipping protection

Uses the existing filters from dsp.filters for consistency.
"""

import numpy as np
import io
import struct
import wave
from scipy.signal import butter, filtfilt, wiener


def load_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Load a WAV file from raw bytes into a numpy array.

    Supports 16-bit and 8-bit PCM WAV files, mono or stereo (stereo → mono).

    Args:
        wav_bytes: Raw bytes of a WAV file.

    Returns:
        Tuple of (audio_signal, sample_rate):
            - audio_signal: Normalized float64 array in [-1.0, 1.0]
            - sample_rate: Sample rate in Hz
    """
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # Convert raw bytes to numpy
    if sample_width == 2:
        # 16-bit signed PCM
        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        audio /= 32768.0  # Normalize to [-1, 1]
    elif sample_width == 1:
        # 8-bit unsigned PCM
        audio = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64)
        audio = (audio - 128.0) / 128.0  # Normalize to [-1, 1]
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    # Convert stereo to mono by averaging channels
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    elif n_channels > 2:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sample_rate


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalize a signal to [-1.0, 1.0] range.

    Prevents clipping when mixing with noise or after filtering.

    Args:
        signal: Input signal array.

    Returns:
        Normalized signal.
    """
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal.copy()


def add_noise_to_audio(
    signal: np.ndarray,
    noise_level: float = 0.1,
) -> np.ndarray:
    """
    Add Gaussian noise to an audio signal.

    Args:
        signal: Clean audio signal (normalized to [-1, 1]).
        noise_level: Standard deviation of noise (0.0 = clean, 0.5 = heavy).

    Returns:
        Noisy signal (may exceed [-1, 1], clip before export).
    """
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise


def denoise_spectral_gating(
    noisy: np.ndarray,
    sample_rate: int,
    noise_threshold_db: float = -30.0,
) -> np.ndarray:
    """
    Apply spectral gating noise reduction.

    Zeroes out frequency bins below a power threshold,
    then inverse FFTs back to time domain.

    Simple but effective for stationary broadband noise.

    Args:
        noisy: Noisy signal.
        sample_rate: Sample rate in Hz.
        noise_threshold_db: Bins below this power (dB) are zeroed.

    Returns:
        Denoised signal.
    """
    # Apply FFT
    fft_data = np.fft.rfft(noisy)
    magnitudes = np.abs(fft_data)
    phases = np.angle(fft_data)

    # Convert to dB
    mag_db = 20 * np.log10(magnitudes + 1e-12)

    # Find noise floor as the median magnitude
    noise_floor = np.median(mag_db)

    # Gate: zero out bins close to or below the noise floor
    threshold = noise_floor + abs(noise_threshold_db) * 0.3
    gate_mask = mag_db > threshold
    gated_magnitudes = magnitudes * gate_mask

    # Reconstruct
    fft_cleaned = gated_magnitudes * np.exp(1j * phases)
    cleaned = np.fft.irfft(fft_cleaned, n=len(noisy))

    return cleaned


def denoise_lowpass(
    noisy: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 4000.0,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter for denoising.

    Removes high-frequency noise above the cutoff frequency.
    Good for audio where most content is below 4-5 kHz.

    Args:
        noisy: Noisy signal.
        sample_rate: Sample rate in Hz.
        cutoff_hz: Cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Filtered signal.
    """
    nyquist = sample_rate / 2.0
    normalized = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(order, normalized, btype="low")
    return filtfilt(b, a, noisy).astype(np.float64)


def denoise_wiener(noisy: np.ndarray) -> np.ndarray:
    """
    Apply Wiener filter for noise reduction.

    The Wiener filter minimizes the mean square error between
    the estimated and true signal. Good for stationary noise.

    Args:
        noisy: Noisy signal.

    Returns:
        Wiener-filtered signal.
    """
    return wiener(noisy).astype(np.float64)


def denoise_audio(
    noisy: np.ndarray,
    sample_rate: int,
    method: str = "combined",
) -> np.ndarray:
    """
    Full denoising pipeline.

    Methods:
        - "lowpass": Butterworth low-pass filter only
        - "spectral": Spectral gating only
        - "wiener": Wiener filter only
        - "combined" (default): Spectral gating → Wiener → gentle low-pass

    Args:
        noisy: Noisy signal.
        sample_rate: Sample rate in Hz.
        method: Denoising method to use.

    Returns:
        Denoised and normalized signal.
    """
    if method == "lowpass":
        cleaned = denoise_lowpass(noisy, sample_rate, cutoff_hz=4500.0)
    elif method == "spectral":
        cleaned = denoise_spectral_gating(noisy, sample_rate)
    elif method == "wiener":
        cleaned = denoise_wiener(noisy)
    elif method == "combined":
        # Stage 1: Spectral gating to remove broadband noise
        cleaned = denoise_spectral_gating(noisy, sample_rate)
        # Stage 2: Wiener filter to further smooth
        cleaned = denoise_wiener(cleaned)
        # Stage 3: Gentle low-pass to clean up residual HF artifacts
        cleaned = denoise_lowpass(cleaned, sample_rate, cutoff_hz=5000.0, order=3)
    else:
        cleaned = noisy.copy()

    return normalize_signal(cleaned)


def signal_to_wav_bytes(
    signal: np.ndarray,
    sample_rate: int,
) -> bytes:
    """
    Convert a numpy signal array to WAV file bytes (16-bit PCM).

    Clips to [-1, 1] before conversion to prevent distortion.

    Args:
        signal: Audio signal array.
        sample_rate: Sample rate in Hz.

    Returns:
        Raw bytes of a valid WAV file.
    """
    # Clip to prevent overflow
    clipped = np.clip(signal, -1.0, 1.0)

    # Convert to 16-bit PCM
    pcm = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


def downsample_for_display(
    signal: np.ndarray,
    max_points: int = 2000,
) -> list:
    """
    Downsample a signal for frontend display.

    Audio signals can be hundreds of thousands of samples.
    We reduce to max_points for efficient chart rendering.

    Args:
        signal: Input signal array.
        max_points: Maximum number of points to return.

    Returns:
        Python list of downsampled values.
    """
    if len(signal) <= max_points:
        return signal.tolist()

    factor = len(signal) // max_points
    return signal[::factor][:max_points].tolist()
