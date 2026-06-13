"""
FFT Analysis Module
===================
This module performs frequency-domain analysis using the Fast Fourier Transform.

What is FFT?
- The Fast Fourier Transform converts a signal from time-domain to frequency-domain.
- Time-domain: shows how amplitude changes over TIME → "what does the signal look like?"
- Frequency-domain: shows what FREQUENCIES are present → "what is the signal made of?"

Why is FFT important in SDR?
- It lets us "see" all the radio stations/signals at different frequencies.
- Real SDR devices use FFT to create spectrum analyzers.
- It helps identify interference, measure bandwidth, and verify modulation.

Mathematical basis:
    X(f) = Σ x(t) * e^(-j*2π*f*t)
    This decomposes a signal into its constituent sinusoidal components.
"""

import numpy as np


def compute_fft(
    signal: np.ndarray,
    sample_rate: int = 10000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Fast Fourier Transform of a signal.

    We use the real-valued FFT (rfft) since our signals are real-valued.
    This is more efficient and only returns the positive frequency components.

    Args:
        signal: Time-domain signal to analyze.
        sample_rate: Sampling rate in Hz (determines frequency resolution).

    Returns:
        Tuple of (frequencies, magnitudes):
            - frequencies: Array of frequency values in Hz.
            - magnitudes: Normalized magnitude spectrum (0 to 1 scale).
    """
    n = len(signal)

    # Step 1: Compute the FFT
    # np.fft.rfft computes FFT for real-valued signals (only positive frequencies)
    fft_result = np.fft.rfft(signal)

    # Step 2: Compute the magnitude spectrum
    # |X(f)| gives us the amplitude at each frequency
    # We normalize by N to get the true amplitude
    magnitudes = np.abs(fft_result) / n

    # Step 3: Double the magnitudes (except DC and Nyquist)
    # Because rfft only gives positive frequencies, we multiply by 2
    # to account for the symmetric negative frequencies
    magnitudes[1:-1] *= 2

    # Step 4: Compute the corresponding frequency values
    # np.fft.rfftfreq returns the frequency bins for rfft output
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    return frequencies, magnitudes


def compute_power_spectral_density(
    signal: np.ndarray,
    sample_rate: int = 10000
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) of a signal.

    PSD shows the power distribution across frequencies.
    Useful for analyzing noise characteristics and signal bandwidth.

    Args:
        signal: Time-domain signal.
        sample_rate: Sampling rate in Hz.

    Returns:
        Tuple of (frequencies, psd_values) in dB scale.
    """
    n = len(signal)

    # Compute FFT
    fft_result = np.fft.rfft(signal)

    # PSD = |X(f)|^2 / N (power at each frequency)
    psd = (np.abs(fft_result) ** 2) / n

    # Convert to dB scale for better visualization
    # Add small epsilon to avoid log(0)
    psd_db = 10 * np.log10(psd + 1e-12)

    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    return frequencies, psd_db


def compute_waterfall_row(
    signal: np.ndarray,
    sample_rate: int = 10000,
    n_bins: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a single waterfall spectrogram row from a signal frame.

    Returns binned FFT magnitudes in dB scale, suitable for heatmap display.
    Each call produces ONE horizontal row of the waterfall.

    Args:
        signal: Time-domain signal for this frame.
        sample_rate: Sampling rate in Hz.
        n_bins: Number of frequency bins for the waterfall display.
                Fewer bins = faster rendering, more bins = finer detail.

    Returns:
        Tuple of (frequency_bins, magnitude_db):
            - frequency_bins: Center frequency of each bin (Hz).
            - magnitude_db: Magnitude in dB scale, clipped to [-80, 0] range.
    """
    n = len(signal)

    # Compute FFT
    fft_result = np.fft.rfft(signal)
    magnitudes = np.abs(fft_result) / n
    magnitudes[1:-1] *= 2

    # Frequencies
    frequencies = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    # Bin the FFT output into n_bins for display
    total_fft_bins = len(magnitudes)
    bin_size = max(1, total_fft_bins // n_bins)
    actual_bins = total_fft_bins // bin_size

    binned_mags = np.zeros(actual_bins)
    binned_freqs = np.zeros(actual_bins)

    for i in range(actual_bins):
        start = i * bin_size
        end = start + bin_size
        binned_mags[i] = np.max(magnitudes[start:end])
        binned_freqs[i] = np.mean(frequencies[start:end])

    # Convert to dB and clip
    binned_db = 20 * np.log10(binned_mags + 1e-12)
    binned_db = np.clip(binned_db, -80, 0)

    return binned_freqs, binned_db
