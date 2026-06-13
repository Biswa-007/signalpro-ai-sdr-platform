"""
Audio Processing Service — Production Grade
=============================================
Multi-format audio loading via librosa, noise reduction pipeline,
spectrogram computation, and WAV export with proper normalization.

Supports: WAV, MP3, FLAC, OGG, M4A, AAC, WMA, AIFF
"""

import numpy as np
import io
import wave
import base64
import logging

from scipy.signal import butter, filtfilt, wiener

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
TARGET_SR = 22050          # Resample to 22.05 kHz for consistency
MAX_DURATION_SEC = 60      # Limit audio to 60 seconds
MAX_DISPLAY_POINTS = 2000  # Chart rendering limit
SPECTROGRAM_N_FFT = 1024
SPECTROGRAM_HOP = 256


# ═══════════════════════════════════════════════════════════════════════════
# AUDIO LOADING (multi-format via librosa)
# ═══════════════════════════════════════════════════════════════════════════

def load_audio_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> tuple[np.ndarray, int]:
    """
    Load any audio format from raw bytes using librosa.
    Returns (signal, sample_rate) with mono, float64, normalized to [-1, 1].
    """
    import librosa
    import soundfile as sf

    buf = io.BytesIO(audio_bytes)

    try:
        # Try soundfile first (fast, supports WAV/FLAC/OGG)
        signal, sr = sf.read(buf, dtype='float64')
        if signal.ndim > 1:
            signal = signal.mean(axis=1)  # stereo → mono
    except Exception:
        # Fallback to librosa for MP3 and other formats
        buf.seek(0)
        signal, sr = librosa.load(buf, sr=None, mono=True)
        signal = signal.astype(np.float64)

    # Resample to target sample rate for consistency
    if sr != TARGET_SR:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # Limit duration
    max_samples = sr * MAX_DURATION_SEC
    if len(signal) > max_samples:
        signal = signal[:max_samples]
        logger.info(f"Audio trimmed to {MAX_DURATION_SEC}s")

    # Normalize
    signal = _normalize(signal)

    logger.info(f"Loaded audio: {len(signal)} samples, {sr} Hz, {len(signal)/sr:.2f}s")
    return signal, sr


# ═══════════════════════════════════════════════════════════════════════════
# NOISE REDUCTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def apply_noise_reduction(
    signal: np.ndarray,
    sample_rate: int,
    method: str = "combined",
) -> np.ndarray:
    """
    Apply noise reduction. Methods:
      - lowpass:  Butterworth LPF at 4000 Hz
      - spectral: Spectral gating (zero bins below noise floor)
      - wiener:   Wiener filter (statistical MSE minimization)
      - combined: Spectral → Wiener → gentle LPF (best quality)
    """
    if len(signal) < 64:
        return signal.copy()

    try:
        if method == "lowpass":
            cleaned = _lowpass(signal, sample_rate, cutoff_hz=4000.0, order=5)
        elif method == "spectral":
            cleaned = _spectral_gate(signal, sample_rate)
        elif method == "wiener":
            cleaned = _wiener_filter(signal)
        elif method == "combined":
            # Multi-stage pipeline for best results
            cleaned = _spectral_gate(signal, sample_rate)
            cleaned = _wiener_filter(cleaned)
            cleaned = _lowpass(cleaned, sample_rate, cutoff_hz=6000.0, order=3)
        else:
            cleaned = signal.copy()

        cleaned = _normalize(cleaned)

        # Prevent the cleaned signal from being worse than original
        # (compare RMS energy — cleaned should not be mostly silence)
        orig_rms = np.sqrt(np.mean(signal ** 2))
        clean_rms = np.sqrt(np.mean(cleaned ** 2))
        if clean_rms < orig_rms * 0.05:
            # Cleaning was too aggressive, fall back to gentle LPF
            logger.warning("Noise reduction too aggressive, falling back to gentle LPF")
            cleaned = _lowpass(signal, sample_rate, cutoff_hz=5000.0, order=3)
            cleaned = _normalize(cleaned)

        return cleaned

    except Exception as e:
        logger.error(f"Noise reduction failed: {e}")
        return signal.copy()


def _normalize(signal: np.ndarray) -> np.ndarray:
    """Peak-normalize to [-1.0, 1.0]."""
    peak = np.max(np.abs(signal))
    if peak > 1e-10:
        return signal / peak
    return signal.copy()


def _lowpass(signal, sr, cutoff_hz=4000.0, order=5):
    """Butterworth low-pass filter."""
    nyquist = sr / 2.0
    norm_cutoff = min(cutoff_hz / nyquist, 0.99)
    b, a = butter(order, norm_cutoff, btype='low')
    filtered = filtfilt(b, a, signal)
    return filtered.astype(np.float64)


def _spectral_gate(signal, sr, threshold_factor=1.5):
    """
    Spectral gating: compute FFT, estimate noise floor from
    the median magnitude, zero out bins near/below it.
    """
    fft_data = np.fft.rfft(signal)
    magnitudes = np.abs(fft_data)
    phases = np.angle(fft_data)

    # Noise floor estimation (median of magnitudes in dB)
    mag_db = 20 * np.log10(magnitudes + 1e-12)
    noise_floor_db = np.median(mag_db)

    # Soft gating threshold
    threshold_db = noise_floor_db + threshold_factor * 10.0

    # Apply gate
    mask = mag_db > threshold_db
    gated = magnitudes * mask

    # Reconstruct
    result = np.fft.irfft(gated * np.exp(1j * phases), n=len(signal))
    return result.astype(np.float64)


def _wiener_filter(signal):
    """Wiener filter for noise reduction."""
    # Wiener filter with auto-estimated noise
    return wiener(signal).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# SPECTROGRAM
# ═══════════════════════════════════════════════════════════════════════════

def compute_spectrogram(
    signal: np.ndarray,
    sample_rate: int,
    n_fft: int = SPECTROGRAM_N_FFT,
    hop_length: int = SPECTROGRAM_HOP,
    max_freq_bins: int = 128,
    max_time_bins: int = 256,
) -> dict:
    """
    Compute a mel spectrogram (dB-scaled) for visualization.
    Returns a 2D array [time_bins x freq_bins] of dB values.
    """
    import librosa

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(
        y=signal.astype(np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=max_freq_bins,
        fmax=sample_rate // 2,
    )

    # Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max)

    # Transpose to [time x freq] for frontend
    S_db = S_db.T

    # Downsample time axis if needed
    if S_db.shape[0] > max_time_bins:
        step = S_db.shape[0] // max_time_bins
        S_db = S_db[::step][:max_time_bins]

    # Convert to list and clamp
    spectrogram_data = np.clip(S_db, -80, 0).tolist()

    duration = len(signal) / sample_rate
    time_labels = [round(i * duration / len(spectrogram_data), 3) for i in range(len(spectrogram_data))]
    freq_labels = librosa.mel_frequencies(n_mels=max_freq_bins, fmax=sample_rate // 2).tolist()

    return {
        "data": spectrogram_data,
        "time_labels": time_labels,
        "freq_labels": freq_labels[:max_freq_bins],
        "n_time": len(spectrogram_data),
        "n_freq": max_freq_bins,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FFT
# ═══════════════════════════════════════════════════════════════════════════

def compute_fft(signal: np.ndarray, sample_rate: int, max_points: int = 1000) -> dict:
    """Compute single-sided FFT magnitude spectrum."""
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    magnitudes = np.abs(fft_vals) * 2.0 / n  # proper scaling
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Downsample for display
    if len(frequencies) > max_points:
        step = len(frequencies) // max_points
        frequencies = frequencies[::step][:max_points]
        magnitudes = magnitudes[::step][:max_points]

    return {
        "frequencies": frequencies.tolist(),
        "magnitudes": magnitudes.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# WAV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def signal_to_wav_bytes(signal: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy signal → 16-bit PCM WAV bytes with clipping protection."""
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue()


def signal_to_b64(signal: np.ndarray, sample_rate: int) -> str:
    """Convert signal to base64-encoded WAV string for inline playback."""
    wav_bytes = signal_to_wav_bytes(signal, sample_rate)
    return base64.b64encode(wav_bytes).decode('ascii')


# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def downsample_for_display(signal: np.ndarray, max_points: int = MAX_DISPLAY_POINTS) -> list:
    """Downsample signal for chart rendering using peak-preserving method."""
    if len(signal) <= max_points:
        return [round(float(v), 6) for v in signal]

    # Use min-max envelope for better waveform representation
    chunk_size = len(signal) // (max_points // 2)
    result = []
    for i in range(0, len(signal) - chunk_size + 1, chunk_size):
        chunk = signal[i:i + chunk_size]
        result.append(round(float(np.min(chunk)), 6))
        result.append(round(float(np.max(chunk)), 6))
        if len(result) >= max_points:
            break

    return result[:max_points]


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_audio(audio_bytes: bytes, filename: str = "audio.wav", method: str = "combined") -> dict:
    """
    Complete audio analysis pipeline.

    Pipeline:
        1. Load audio (any format via librosa)
        2. Normalize to [-1, 1]
        3. Apply noise reduction
        4. Compute FFT (original signal)
        5. Compute spectrogram
        6. Generate base64 WAV for both original + cleaned
        7. Downsample waveforms for chart display

    Returns dict with all data needed for frontend rendering.
    """
    # Step 1-2: Load + normalize
    original, sr = load_audio_bytes(audio_bytes, filename)

    # Step 3: Noise reduction
    cleaned = apply_noise_reduction(original, sr, method=method)

    # Step 4: FFT of original
    fft_data = compute_fft(original, sr)

    # Step 5: Spectrogram of original
    spectrogram = compute_spectrogram(original, sr)

    # Step 6: Base64 audio for playback
    original_b64 = signal_to_b64(original, sr)
    cleaned_b64 = signal_to_b64(cleaned, sr)

    # Step 7: Downsampled waveforms for charts
    duration = len(original) / sr
    orig_display = downsample_for_display(original)
    clean_display = downsample_for_display(cleaned)

    display_len = len(orig_display)
    time_display = [round(i * duration / display_len, 6) for i in range(display_len)]

    return {
        "original_signal": orig_display,
        "cleaned_signal": clean_display,
        "time": time_display,
        "spectrogram": spectrogram,
        "fft": fft_data,
        "original_audio": original_b64,
        "cleaned_audio": cleaned_b64,
        "sample_rate": sr,
        "duration_seconds": round(duration, 3),
        "total_samples": len(original),
        "method": method,
    }
