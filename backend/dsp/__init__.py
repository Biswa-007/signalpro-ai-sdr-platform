# DSP Pipeline Package
# This package contains all Digital Signal Processing modules
# for the Software Defined Radio simulation.
#
# Modules:
#   signal_generation — Time arrays, message & carrier signals
#   modulation        — FM modulation
#   noise             — AWGN channel model
#   fft_analysis      — FFT & power spectral density
#   demodulation      — FM demodulation via Hilbert transform
#   filters           — Butterworth IIR filters (LP, HP, BP)
#   interference      — Multi-signal interference mixing

from .signal_generation import generate_time_array, generate_message_signal, generate_carrier_signal
from .modulation import fm_modulate
from .noise import add_gaussian_noise, calculate_snr
from .fft_analysis import compute_fft, compute_power_spectral_density, compute_waterfall_row
from .demodulation import fm_demodulate
from .filters import apply_lowpass_filter, apply_highpass_filter, apply_bandpass_filter
from .interference import generate_interference, generate_multi_tone_interference, mix_signals
