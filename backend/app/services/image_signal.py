"""
Image → Signal Conversion Service
==================================
Converts uploaded images to 1D signal representations.

Pipeline:
    1. Load image from bytes
    2. Convert to grayscale
    3. Flatten pixel intensities to 1D signal
    4. Normalize to [-1, 1]
    5. Compute FFT
    6. Downsample for display
"""

import numpy as np
import io

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def image_to_signal(image_bytes: bytes, max_points: int = 3000) -> dict:
    """
    Convert an image to a 1D signal by mapping pixel intensity → amplitude.

    Args:
        image_bytes: Raw bytes of an image file (PNG, JPG, etc.)
        max_points: Maximum number of signal points for display.

    Returns:
        dict with 'signal', 'row_signal', 'fft', 'width', 'height', 'total_pixels'
    """
    if not HAS_PIL:
        raise ImportError("Pillow is required for image processing. Install with: pip install Pillow")

    # Load image
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale
    gray = img.convert("L")

    # Get dimensions
    width, height = gray.size

    # Flatten to 1D array
    pixels = np.array(gray, dtype=np.float64).flatten()

    # Normalize to [-1, 1]
    signal = (pixels / 127.5) - 1.0

    # Downsample for display
    if len(signal) > max_points:
        step = len(signal) // max_points
        signal_display = signal[::step][:max_points].tolist()
    else:
        signal_display = signal.tolist()

    # Also compute a row-averaged signal (one value per row)
    gray_arr = np.array(gray, dtype=np.float64)
    row_signal = gray_arr.mean(axis=1)
    row_signal = (row_signal / 127.5) - 1.0
    row_signal_display = row_signal.tolist()

    # ── Compute FFT of the signal ──────────────────────────────────────
    fft_signal = signal
    if len(fft_signal) > 8192:
        fft_signal = fft_signal[:8192]  # Limit for performance

    n = len(fft_signal)
    fft_vals = np.fft.rfft(fft_signal)
    magnitudes = np.abs(fft_vals) * 2.0 / n
    frequencies = np.fft.rfftfreq(n, d=1.0)  # normalized frequency

    # Downsample FFT for display
    max_fft = 1000
    if len(frequencies) > max_fft:
        step = len(frequencies) // max_fft
        frequencies = frequencies[::step][:max_fft]
        magnitudes = magnitudes[::step][:max_fft]

    fft_data = {
        "frequencies": frequencies.tolist(),
        "magnitudes": magnitudes.tolist(),
    }

    return {
        "signal": signal_display,
        "row_signal": row_signal_display,
        "fft": fft_data,
        "width": width,
        "height": height,
        "total_pixels": width * height,
    }
