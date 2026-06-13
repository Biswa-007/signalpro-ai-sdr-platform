"""
Noise Module
============
This module adds realistic noise to signals.

What is Gaussian Noise?
- In real communication systems, signals are always corrupted by noise.
- The most common model is Additive White Gaussian Noise (AWGN).
- "Gaussian" = the noise amplitude follows a normal distribution (bell curve).
- "White" = the noise has equal power at all frequencies.
- "Additive" = noise is simply added to the signal.

The noise_level parameter controls the standard deviation (σ) of the noise.
- noise_level = 0.0 → no noise (perfect channel)
- noise_level = 0.1 → mild noise
- noise_level = 0.5 → moderate noise
- noise_level = 1.0+ → heavy noise (signal may be unrecoverable)
"""

import numpy as np


def add_gaussian_noise(
    signal: np.ndarray,
    noise_level: float = 0.1
) -> np.ndarray:
    """
    Add Additive White Gaussian Noise (AWGN) to a signal.

    This simulates real-world channel impairments where thermal noise,
    interference, and other random disturbances corrupt the signal.

    Args:
        signal: The clean input signal.
        noise_level: Standard deviation of the Gaussian noise.
                     Controls how strong the noise is relative to the signal.

    Returns:
        Noisy signal = original signal + random noise.
    """
    # Generate random noise with mean=0 and std=noise_level
    # The shape matches the input signal length
    noise = np.random.normal(
        loc=0.0,           # Mean of the distribution (centered at zero)
        scale=noise_level,  # Standard deviation (controls noise power)
        size=signal.shape   # Same number of samples as the signal
    )

    # Add noise to the signal (additive noise model)
    noisy_signal = signal + noise

    return noisy_signal


def calculate_snr(signal: np.ndarray, noise_level: float) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels.

    SNR indicates how much stronger the signal is compared to the noise.
    Higher SNR = cleaner signal.

    SNR (dB) = 10 * log10(P_signal / P_noise)

    Args:
        signal: The original clean signal.
        noise_level: Standard deviation of the noise.

    Returns:
        SNR value in decibels (dB).
    """
    # Signal power = mean of signal squared
    signal_power = np.mean(signal ** 2)

    # Noise power = variance of the noise = noise_level^2
    noise_power = noise_level ** 2

    if noise_power == 0:
        return float('inf')  # No noise → infinite SNR

    # SNR in decibels
    snr_db = 10 * np.log10(signal_power / noise_power)
    return round(snr_db, 2)
