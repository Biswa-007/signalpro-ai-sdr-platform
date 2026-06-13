"""
FM Demodulation Module
======================
This module recovers the original message from an FM-modulated signal.

How does FM Demodulation work?
- FM encodes information in the FREQUENCY variations of the carrier.
- To recover the message, we need to detect these frequency changes.
- One common method: differentiate the phase of the analytic signal.

Demodulation steps:
1. Compute the analytic signal using Hilbert transform.
2. Extract the instantaneous phase.
3. Differentiate the phase (frequency is the derivative of phase).
4. The result is proportional to the original message signal.

The Hilbert Transform creates an "analytic signal" which has no negative
frequency components, making phase extraction unambiguous.
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt


def fm_demodulate(
    signal: np.ndarray,
    sample_rate: int = 10000
) -> np.ndarray:
    """
    Demodulate an FM signal to recover the original message.

    This uses the analytic signal approach:
    1. Apply Hilbert transform to get the analytic signal.
    2. Extract instantaneous phase via arctan2.
    3. Unwrap the phase (remove 2π discontinuities).
    4. Differentiate to get instantaneous frequency.
    5. Low-pass filter to clean up the result.

    Args:
        signal: FM-modulated (possibly noisy) signal.
        sample_rate: Sampling rate in Hz.

    Returns:
        Demodulated signal (approximation of the original message).
    """
    # Step 1: Compute the analytic signal using Hilbert transform
    # The analytic signal z(t) = signal(t) + j * hilbert(signal(t))
    analytic_signal = hilbert(signal)

    # Step 2: Extract the instantaneous phase
    # phase(t) = arctan2(imag(z(t)), real(z(t)))
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))

    # Step 3: Compute instantaneous frequency by differentiating phase
    # frequency(t) = (1 / 2π) * d(phase) / dt
    # np.diff computes the discrete derivative
    instantaneous_freq = np.diff(instantaneous_phase) * sample_rate / (2 * np.pi)

    # Step 4: Pad to maintain original length (diff reduces length by 1)
    instantaneous_freq = np.append(instantaneous_freq, instantaneous_freq[-1])

    # Step 5: Remove DC offset (center around zero)
    instantaneous_freq -= np.mean(instantaneous_freq)

    # Step 6: Normalize the demodulated signal
    max_val = np.max(np.abs(instantaneous_freq))
    if max_val > 0:
        instantaneous_freq = instantaneous_freq / max_val

    # Step 7: Apply a low-pass filter to clean up the result
    demodulated = _lowpass_filter(instantaneous_freq, cutoff=200, sample_rate=sample_rate)

    return demodulated


def _lowpass_filter(
    signal: np.ndarray,
    cutoff: float = 200.0,
    sample_rate: int = 10000,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter.

    This removes high-frequency noise from the demodulated signal,
    leaving only the low-frequency message content.

    Args:
        signal: Input signal to filter.
        cutoff: Filter cutoff frequency in Hz.
        sample_rate: Sampling rate in Hz.
        order: Filter order (higher = sharper cutoff, but more ringing).

    Returns:
        Filtered signal.
    """
    # Nyquist frequency = half the sample rate
    nyquist = sample_rate / 2.0

    # Normalized cutoff frequency (0 to 1, where 1 = Nyquist)
    normalized_cutoff = cutoff / nyquist

    # Clamp to valid range
    normalized_cutoff = min(normalized_cutoff, 0.99)

    # Design the Butterworth filter
    # butter() returns filter coefficients (b, a)
    b, a = butter(order, normalized_cutoff, btype='low')

    # Apply the filter using zero-phase filtering (no delay)
    # filtfilt applies the filter forward and backward for zero phase distortion
    filtered = filtfilt(b, a, signal)

    return filtered
