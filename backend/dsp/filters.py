"""
Digital Filters Module
======================
Implements Butterworth IIR filters for signal conditioning.

Filter types:
- Low-pass: Removes high-frequency noise, keeps low-frequency content.
- High-pass: Removes low-frequency drift, keeps high-frequency detail.
- Band-pass: Keeps only a specific frequency range.

All filters use scipy.signal's Butterworth design + zero-phase filtering
(filtfilt) to avoid introducing group delay.
"""

import numpy as np
from scipy.signal import butter, filtfilt


def apply_lowpass_filter(
    signal: np.ndarray,
    cutoff: float = 500.0,
    sample_rate: int = 10000,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter.

    Passes frequencies below the cutoff and attenuates those above.
    Common use: remove high-frequency noise from a received signal.

    Args:
        signal: Input signal array.
        cutoff: Cutoff frequency in Hz.
        sample_rate: Sampling rate in Hz.
        order: Filter order (higher = sharper rolloff).

    Returns:
        Filtered signal with high-frequency components removed.
    """
    nyquist = sample_rate / 2.0
    normalized_cutoff = min(cutoff / nyquist, 0.99)

    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, signal).astype(np.float64)


def apply_highpass_filter(
    signal: np.ndarray,
    cutoff: float = 100.0,
    sample_rate: int = 10000,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth high-pass filter.

    Passes frequencies above the cutoff and attenuates those below.
    Common use: remove DC offset or low-frequency drift.

    Args:
        signal: Input signal array.
        cutoff: Cutoff frequency in Hz.
        sample_rate: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal with low-frequency components removed.
    """
    nyquist = sample_rate / 2.0
    normalized_cutoff = max(cutoff / nyquist, 0.01)
    normalized_cutoff = min(normalized_cutoff, 0.99)

    b, a = butter(order, normalized_cutoff, btype="high")
    return filtfilt(b, a, signal).astype(np.float64)


def apply_bandpass_filter(
    signal: np.ndarray,
    low_cutoff: float = 80.0,
    high_cutoff: float = 400.0,
    sample_rate: int = 10000,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter.

    Passes frequencies between low_cutoff and high_cutoff.
    Common use: isolate a specific signal in a crowded spectrum.

    Args:
        signal: Input signal array.
        low_cutoff: Lower cutoff frequency in Hz.
        high_cutoff: Upper cutoff frequency in Hz.
        sample_rate: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal keeping only the specified frequency band.
    """
    nyquist = sample_rate / 2.0
    low = max(low_cutoff / nyquist, 0.01)
    high = min(high_cutoff / nyquist, 0.99)

    if low >= high:
        return signal.copy()

    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal).astype(np.float64)
