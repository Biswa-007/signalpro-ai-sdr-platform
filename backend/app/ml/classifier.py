"""
ML Signal Classifier
====================
A lightweight machine learning model that classifies signal conditions
based on spectral and statistical features extracted from the DSP pipeline.

Classification Labels:
    - "Clean Signal"         — Low noise, no interference
    - "Noisy Signal"         — High noise floor, degraded SNR
    - "Interference Present" — Spectral spikes from co-channel interferers

Architecture:
    - Feature extraction from FFT magnitudes + time-domain statistics
    - RandomForestClassifier (scikit-learn) for robust multi-class prediction
    - Trained on synthetic data generated at startup using the DSP pipeline
    - Outputs class label + per-class confidence probabilities
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import logging
import sys
import os

# Add parent directory to path so we can import dsp modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dsp.signal_generation import generate_time_array, generate_message_signal
from dsp.modulation import fm_modulate
from dsp.noise import add_gaussian_noise
from dsp.fft_analysis import compute_fft
from dsp.interference import generate_interference, mix_signals

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
LABELS = ["Clean Signal", "Noisy Signal", "Interference Present"]
SAMPLE_RATE = 10000
DURATION = 0.05


class SignalClassifier:
    """
    Trains a RandomForest on synthetic signal features and provides
    real-time classification of signal conditions.
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    # ─── Feature Extraction ──────────────────────────────────────────────

    @staticmethod
    def extract_features(
        signal: np.ndarray,
        fft_magnitudes: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """
        Extract a feature vector from a signal + its FFT.

        Features (8 total):
          1. Signal power (mean of squared amplitudes)
          2. Peak amplitude
          3. Crest factor (peak / RMS) — high for clean signals
          4. Signal variance
          5. Peak FFT magnitude
          6. Spectral mean — average energy across spectrum
          7. Spectral std — spread of energy (high = interference)
          8. Spectral flatness — how "white" the spectrum is (high = noise)

        Args:
            signal: Time-domain signal array.
            fft_magnitudes: FFT magnitude spectrum.
            sample_rate: Sampling rate (unused but kept for API consistency).

        Returns:
            1D numpy array of 8 features.
        """
        # Time-domain features
        signal_power = float(np.mean(signal ** 2))
        peak_amplitude = float(np.max(np.abs(signal)))
        rms = float(np.sqrt(signal_power))
        crest_factor = peak_amplitude / rms if rms > 0 else 0.0
        variance = float(np.var(signal))

        # Frequency-domain features
        peak_fft = float(np.max(fft_magnitudes))
        spectral_mean = float(np.mean(fft_magnitudes))
        spectral_std = float(np.std(fft_magnitudes))

        # Spectral flatness: geometric mean / arithmetic mean
        # Values close to 1 = flat (noise-like), close to 0 = tonal (signal-like)
        fft_positive = fft_magnitudes[fft_magnitudes > 0]
        if len(fft_positive) > 0:
            log_mean = np.mean(np.log(fft_positive + 1e-12))
            geometric_mean = np.exp(log_mean)
            arithmetic_mean = np.mean(fft_positive)
            spectral_flatness = float(geometric_mean / arithmetic_mean) if arithmetic_mean > 0 else 0.0
        else:
            spectral_flatness = 0.0

        return np.array([
            signal_power,
            peak_amplitude,
            crest_factor,
            variance,
            peak_fft,
            spectral_mean,
            spectral_std,
            spectral_flatness,
        ])

    # ─── Training ────────────────────────────────────────────────────────

    def train(self, n_samples_per_class: int = 200):
        """
        Train the classifier on synthetically generated signal data.

        Generates signals for each class using the DSP pipeline with
        varying parameters to create diversity in the training set.

        Args:
            n_samples_per_class: Number of training samples per class.
        """
        logger.info("Training signal classifier on synthetic data...")

        X = []
        y = []

        t = generate_time_array(duration=DURATION, sample_rate=SAMPLE_RATE)
        rng = np.random.RandomState(42)

        for i in range(n_samples_per_class):
            msg_freq = rng.uniform(5, 50)
            carrier_freq = rng.uniform(100, 1000)
            message = generate_message_signal(t, frequency=msg_freq)
            fm = fm_modulate(t, message, carrier_freq=carrier_freq, freq_deviation=carrier_freq * 0.5)

            # ── Class 0: Clean Signal ────────────────────────────────
            noise_level = rng.uniform(0.0, 0.08)
            clean = add_gaussian_noise(fm, noise_level=noise_level)
            _, fft_mag = compute_fft(clean, sample_rate=SAMPLE_RATE)
            features = self.extract_features(clean, fft_mag)
            X.append(features)
            y.append(0)

            # ── Class 1: Noisy Signal ────────────────────────────────
            noise_level = rng.uniform(0.4, 1.5)
            noisy = add_gaussian_noise(fm, noise_level=noise_level)
            _, fft_mag = compute_fft(noisy, sample_rate=SAMPLE_RATE)
            features = self.extract_features(noisy, fft_mag)
            X.append(features)
            y.append(1)

            # ── Class 2: Interference Present ────────────────────────
            noise_level = rng.uniform(0.0, 0.3)
            interf_freq = rng.uniform(100, 800)
            interf_amp = rng.uniform(0.3, 1.2)
            interf = generate_interference(t, frequency=interf_freq, amplitude=interf_amp)
            mixed = mix_signals(fm, interf)
            mixed = add_gaussian_noise(mixed, noise_level=noise_level)
            _, fft_mag = compute_fft(mixed, sample_rate=SAMPLE_RATE)
            features = self.extract_features(mixed, fft_mag)
            X.append(features)
            y.append(2)

        X = np.array(X)
        y = np.array(y)

        # Fit scaler and transform
        X_scaled = self.scaler.fit_transform(X)

        # Train the model
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Log accuracy on training data
        train_accuracy = self.model.score(X_scaled, y)
        logger.info(f"Classifier trained. Training accuracy: {train_accuracy:.2%}")

    # ─── Prediction ──────────────────────────────────────────────────────

    def predict(
        self,
        signal: np.ndarray,
        fft_magnitudes: np.ndarray,
    ) -> dict:
        """
        Classify the current signal condition.

        Args:
            signal: Time-domain signal array.
            fft_magnitudes: FFT magnitude spectrum.

        Returns:
            Dict with:
                - label: classification string
                - confidence: dict of class name → probability
        """
        if not self.is_trained:
            return {
                "label": "Model Not Trained",
                "confidence": {name: 0.0 for name in LABELS},
            }

        features = self.extract_features(signal, fft_magnitudes)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Get class probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        predicted_class = int(np.argmax(probabilities))

        confidence = {}
        for idx, label in enumerate(LABELS):
            confidence[label] = round(float(probabilities[idx]), 4)

        return {
            "label": LABELS[predicted_class],
            "confidence": confidence,
        }


# ─── Module-level singleton ─────────────────────────────────────────────────
# Created once, shared across all requests.
classifier = SignalClassifier()
