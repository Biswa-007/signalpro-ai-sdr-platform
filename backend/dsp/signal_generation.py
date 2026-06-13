"""
Signal Generation Module
========================
This module handles the creation of base signals used in the SDR simulation.

Key Concepts:
- A "message signal" is the actual information we want to transmit (e.g., audio).
- A "carrier signal" is a high-frequency wave used to carry the message.
- In real radio, the carrier frequency determines the station (e.g., 98.1 FM).

We use sinusoidal waves: signal(t) = amplitude * sin(2 * pi * frequency * t)
"""

import numpy as np


def generate_time_array(duration: float = 0.05, sample_rate: int = 10000) -> np.ndarray:
    """
    Generate a time array for signal simulation.

    Args:
        duration: Length of the signal in seconds (default: 50ms).
                  Shorter durations keep the response size manageable.
        sample_rate: Number of samples per second (Nyquist: must be >= 2x max frequency).

    Returns:
        numpy array of evenly-spaced time values from 0 to duration.
    """
    # np.linspace creates evenly spaced values
    # endpoint=False avoids duplicating the last point (important for FFT)
    num_samples = int(duration * sample_rate)
    return np.linspace(0, duration, num_samples, endpoint=False)


def generate_message_signal(
    t: np.ndarray,
    frequency: float = 5.0,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    Generate the message (baseband) signal — the information to transmit.

    In real SDR applications, this would be audio, data, etc.
    Here we use a simple sine wave for clarity.

    Args:
        t: Time array (seconds).
        frequency: Message frequency in Hz (how fast the information oscillates).
        amplitude: Peak amplitude of the message signal.

    Returns:
        Message signal as a numpy array.
    """
    # The message signal: m(t) = A * sin(2π * f_m * t)
    return amplitude * np.sin(2 * np.pi * frequency * t)


def generate_carrier_signal(
    t: np.ndarray,
    frequency: float = 100.0,
    amplitude: float = 1.0
) -> np.ndarray:
    """
    Generate the carrier signal — the high-frequency wave that carries the message.

    In FM radio, carrier frequencies are in the MHz range (88-108 MHz).
    For simulation, we use much lower frequencies to keep things simple.

    Args:
        t: Time array (seconds).
        frequency: Carrier frequency in Hz (must be >> message frequency).
        amplitude: Peak amplitude of the carrier.

    Returns:
        Carrier signal as a numpy array.
    """
    # The carrier signal: c(t) = A * cos(2π * f_c * t)
    # We use cosine for the carrier by convention
    return amplitude * np.cos(2 * np.pi * frequency * t)
