"""
Interference Module
===================
Simulates multi-signal interference in a communication channel.

In real radio environments, multiple transmitters operate simultaneously.
When signals overlap in frequency, they interfere with each other.
This module generates interfering signals and mixes them with the primary.

Types of interference modeled:
- Co-channel: another signal at a similar frequency
- Adjacent-channel: a strong signal at a nearby frequency
- Narrowband: a single-tone interferer (like a jammer)
"""

import numpy as np


def generate_interference(
    t: np.ndarray,
    frequency: float = 300.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a single-tone interfering signal.

    This models narrowband interference — a common real-world impairment
    caused by other transmitters, oscillator leakage, or deliberate jamming.

    Args:
        t: Time array (seconds).
        frequency: Interference frequency in Hz.
        amplitude: Strength of the interferer (0 = none, 1 = same as signal).

    Returns:
        Interfering signal as a numpy array.
    """
    return amplitude * np.sin(2 * np.pi * frequency * t)


def generate_multi_tone_interference(
    t: np.ndarray,
    frequencies: list[float] | None = None,
    amplitudes: list[float] | None = None,
) -> np.ndarray:
    """
    Generate multi-tone interference (multiple interferers at once).

    Models a crowded RF environment with multiple overlapping signals.

    Args:
        t: Time array (seconds).
        frequencies: List of interferer frequencies in Hz.
        amplitudes: List of interferer amplitudes.

    Returns:
        Combined interference signal.
    """
    if frequencies is None:
        frequencies = [300.0, 450.0]
    if amplitudes is None:
        amplitudes = [0.3, 0.2]

    interference = np.zeros_like(t)
    for freq, amp in zip(frequencies, amplitudes):
        interference += amp * np.sin(2 * np.pi * freq * t)
    return interference


def mix_signals(
    primary: np.ndarray,
    interference: np.ndarray,
) -> np.ndarray:
    """
    Mix the primary signal with interference (additive model).

    In a real receiver, the antenna picks up ALL signals in its bandwidth.
    The received signal is the sum of the desired signal + all interferers.

    Args:
        primary: The desired signal.
        interference: The unwanted interfering signal.

    Returns:
        Combined signal (primary + interference).
    """
    return primary + interference
