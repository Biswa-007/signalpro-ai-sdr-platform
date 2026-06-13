"""
FM Modulation Module
====================
This module implements Frequency Modulation (FM).

What is FM Modulation?
- FM encodes information by varying the FREQUENCY of the carrier signal.
- When the message signal amplitude is high → carrier frequency increases.
- When the message signal amplitude is low → carrier frequency decreases.

The FM signal formula:
    s(t) = A_c * cos(2π * f_c * t + 2π * k_f * ∫m(τ)dτ)

Where:
    - A_c = carrier amplitude
    - f_c = carrier frequency
    - k_f = frequency deviation constant (sensitivity)
    - m(t) = message signal
    - ∫m(τ)dτ = cumulative integral of the message signal

The integral of the message creates the instantaneous phase deviation,
which causes the frequency to vary proportionally to the message amplitude.
"""

import numpy as np


def fm_modulate(
    t: np.ndarray,
    message_signal: np.ndarray,
    carrier_freq: float = 100.0,
    carrier_amplitude: float = 1.0,
    freq_deviation: float = 50.0
) -> np.ndarray:
    """
    Apply Frequency Modulation to the message signal.

    Args:
        t: Time array (seconds).
        message_signal: The baseband message signal m(t).
        carrier_freq: Carrier frequency in Hz (f_c).
        carrier_amplitude: Amplitude of the carrier (A_c).
        freq_deviation: Maximum frequency deviation in Hz (k_f).
                        Higher values = wider bandwidth = more robust signal.

    Returns:
        FM modulated signal as a numpy array.
    """
    # Step 1: Calculate the time step (dt) for numerical integration
    dt = t[1] - t[0]

    # Step 2: Compute cumulative integral of the message signal
    # np.cumsum performs numerical integration (trapezoidal approximation)
    # This gives us the instantaneous phase deviation
    cumulative_integral = np.cumsum(message_signal) * dt

    # Step 3: Compute the instantaneous phase of the FM signal
    # Phase = 2π * f_c * t + 2π * k_f * ∫m(τ)dτ
    instantaneous_phase = (
        2 * np.pi * carrier_freq * t
        + 2 * np.pi * freq_deviation * cumulative_integral
    )

    # Step 4: Generate the FM signal
    # s(t) = A_c * cos(instantaneous_phase)
    fm_signal = carrier_amplitude * np.cos(instantaneous_phase)

    return fm_signal
