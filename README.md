# Real-Time AI-Powered Software Defined Radio (SDR) Simulation Platform

A full-stack engineering system that simulates communication signals in real-time, applies DSP operations, visualizes data across multiple domains, and uses AI to classify signal conditions.

Built with **FastAPI** (Python) + **Next.js** (React) + **scikit-learn** (ML).

---

## ⚡ Quick Start

### Backend (Python)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The backend will:
- Train the ML classifier on synthetic data (~1 second)
- Start serving on http://localhost:8000
- API docs available at http://localhost:8000/docs

### Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                        │
│                                                             │
│  ┌──────────┐  ┌───────────────────────────────────────┐    │
│  │ Controls │  │            Charts Grid                 │    │
│  │  Panel   │  │  ┌──────────┐  ┌──────────────────┐   │    │
│  │          │  │  │FM Signal │  │  FFT Spectrum     │   │    │
│  │ Sliders  │  │  │(Recharts)│  │  (Recharts)       │   │    │
│  │ Presets  │  │  └──────────┘  └──────────────────┘   │    │
│  │ Filters  │  │  ┌──────────┐  ┌──────────────────┐   │    │
│  │ Toggle   │  │  │Waterfall │  │  Demod Output    │   │    │
│  │ AI Panel │  │  │ (Canvas) │  │  (Recharts)       │   │    │
│  └──────────┘  │  └──────────┘  └──────────────────┘   │    │
│                └───────────────────────────────────────┘    │
└────────────────────┬──────────────────┬─────────────────────┘
                     │ POST /simulate   │ WS /ws/stream
                     ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                          │
│                                                             │
│  Routes ──► Services ──► DSP Pipeline ──► ML Classifier     │
│                          │                                  │
│              ┌───────────┼───────────────────┐              │
│              │ Signal Gen → FM Mod → + Noise │              │
│              │ → Interference → Filter       │              │
│              │ → FFT → Waterfall → Demod     │              │
│              └───────────────────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User** adjusts sliders (carrier freq, message freq, noise, interference, filter)
2. **Frontend** sends parameters via POST (one-shot) or WebSocket (real-time streaming)
3. **Backend** runs the DSP pipeline: Signal Gen → FM Mod → Interference → Noise → Filter → FFT → Waterfall → Demod → ML Classification
4. **Response** contains time-domain signals, FFT spectrum, waterfall row, demod output, SNR, and AI classification
5. **Frontend** renders 4 synchronized visualizations + AI confidence display

---

## 📁 Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI entry point + lifespan handler
│   ├── routes/
│   │   ├── simulate.py      # POST /simulate endpoint
│   │   └── websocket.py     # WS /ws/stream endpoint
│   ├── services/
│   │   └── dsp_pipeline.py  # Full DSP pipeline orchestration
│   └── ml/
│       └── classifier.py    # RandomForest signal classifier
├── dsp/
│   ├── signal_generation.py # Time arrays, message & carrier signals
│   ├── modulation.py        # FM modulation
│   ├── noise.py             # AWGN channel model
│   ├── fft_analysis.py      # FFT, PSD, waterfall row computation
│   ├── demodulation.py      # FM demodulation via Hilbert transform
│   ├── filters.py           # Butterworth IIR filters (LP, HP, BP)
│   └── interference.py      # Multi-signal interference mixing
└── requirements.txt

frontend/
├── app/
│   ├── page.js              # Main dashboard page
│   ├── layout.js            # Root layout with SEO metadata
│   ├── globals.css          # Complete glassmorphism design system
│   ├── api/
│   │   └── simulate.js      # REST + WebSocket API client
│   └── components/
│       ├── ControlsPanel.js  # Sliders, presets, toggle, export, AI panel
│       └── charts/
│           ├── SignalChart.js     # Recharts line chart
│           └── WaterfallChart.js  # Canvas-based waterfall spectrogram
└── package.json
```

---

## 🔧 API Endpoints

### POST /simulate
```json
{
  "carrier_freq": 200.0,
  "message_freq": 10.0,
  "noise_level": 0.1,
  "interference_freq": 0.0,
  "interference_amp": 0.0,
  "filter_type": "none"
}
```

Returns all signal arrays, FFT data, waterfall row, demodulated signal, SNR, and AI classification.

### WebSocket /ws/stream
- Sends signal frames at ~10 FPS continuously
- Client sends JSON to update parameters in real-time
- Each frame contains the same data as POST /simulate

---

## 🧠 ML Classification

The system uses a **RandomForestClassifier** (scikit-learn) trained on synthetic signal data at startup.

**Features extracted (8 total):**
1. Signal power
2. Peak amplitude
3. Crest factor (peak / RMS)
4. Signal variance
5. Peak FFT magnitude
6. Spectral mean
7. Spectral std
8. Spectral flatness

**Classes:**
- 🟢 **Clean Signal** — Low noise, no interference
- 🟡 **Noisy Signal** — High noise floor, degraded SNR
- 🔴 **Interference Present** — Spectral spikes from interferers

Training accuracy: **100%** on 600 synthetic samples (200 per class).

---

## 📄 Resume Description

> **Real-Time AI-Powered SDR Simulation Platform** — Designed and built a full-stack signal processing system using FastAPI and Next.js. Implemented a modular DSP pipeline (FM modulation, AWGN noise, Butterworth filtering, FFT analysis, waterfall spectrogram, Hilbert demodulation) with real-time WebSocket streaming at 10 FPS. Integrated a RandomForest classifier (scikit-learn) for automated signal condition detection (clean/noisy/interference) with 100% training accuracy. Frontend features a glassmorphism dashboard with Recharts line charts, a canvas-based scrolling waterfall spectrogram, and interactive preset modes.

---

## 🎓 Viva Questions & Answers

### 1. What is Frequency Modulation (FM)?
FM encodes information by varying the **frequency** of a carrier signal proportionally to the message signal amplitude. The FM equation is: `s(t) = A * cos(2π*fc*t + 2π*kf*∫m(τ)dτ)` where `kf` is the frequency deviation constant.

### 2. What is the FFT and why is it used in SDR?
The Fast Fourier Transform converts time-domain signals to frequency-domain, revealing which frequencies are present. In SDR, FFT enables spectrum analysis — identifying signals, measuring bandwidth, detecting interference, and creating waterfall displays.

### 3. How does the Hilbert Transform help in FM demodulation?
The Hilbert Transform creates an analytic signal (complex envelope), enabling extraction of instantaneous phase. Differentiating the unwrapped phase yields instantaneous frequency, which is proportional to the original message signal.

### 4. What is AWGN and why is it used?
Additive White Gaussian Noise models thermal noise in communication channels. It's "additive" (added to signal), "white" (equal power at all frequencies), and "Gaussian" (amplitude follows normal distribution). It's the standard channel model in communications.

### 5. Explain the Butterworth filter design.
Butterworth filters have maximally flat magnitude response in the passband. We use `scipy.signal.butter()` to compute filter coefficients and `filtfilt()` for zero-phase filtering (no group delay distortion), which applies the filter forward and backward.

### 6. How does the ML classifier work?
The RandomForest extracts 8 features (signal power, crest factor, spectral flatness, etc.) from each signal frame. It was trained on 600 synthetic samples generated through the DSP pipeline with varying noise/interference levels. It classifies signals into Clean/Noisy/Interference.

### 7. Why use WebSocket instead of polling?
WebSocket provides full-duplex, persistent connections. For real-time streaming at 10 FPS, polling would create excessive HTTP overhead (headers, TCP handshakes). WebSocket sends lightweight JSON frames over a single connection, enabling smooth real-time visualization.

### 8. What is a waterfall spectrogram?
A waterfall (or spectrogram) shows frequency content over time as a 2D heatmap. X-axis = frequency, Y-axis = time (scrolling), color = magnitude. Each row is one FFT frame. It reveals time-varying spectral characteristics of signals.

### 9. How is the Signal-to-Noise Ratio (SNR) calculated?
SNR (dB) = 10 * log10(P_signal / P_noise), where P_signal = mean(signal²) and P_noise = noise_level². Higher SNR means cleaner signal. Typical values: >20 dB = high quality, 10-20 dB = acceptable, <10 dB = poor.

### 10. What is spectral flatness and why is it useful?
Spectral flatness = geometric_mean(spectrum) / arithmetic_mean(spectrum). Values near 1.0 indicate white noise (flat spectrum), values near 0.0 indicate tonal signals (peaky spectrum). It distinguishes noise-dominated from signal-dominated conditions.

### 11. How does the interference simulation work?
Interference is modeled as additional sinusoidal signals at different frequencies. The primary FM signal and interference are summed (additive mixing), simulating how a receiver antenna captures all signals in its bandwidth simultaneously.

### 12. Why is the Nyquist criterion important?
The Nyquist theorem states that the sampling rate must be ≥ 2× the highest frequency in the signal. If violated, aliasing occurs — high frequencies fold down and corrupt lower frequencies. Our 10 kHz sample rate supports signals up to 5 kHz.

---

## 🚀 Future Scope

1. **RTL-SDR Hardware Integration** — Connect to real SDR hardware (RTL-SDR USB dongles) via `pyrtlsdr` for live RF signal capture and analysis.

2. **Edge Deployment** — Deploy the backend on a Raspberry Pi with an RTL-SDR dongle for portable spectrum monitoring.

3. **Advanced ML Models** — Replace RandomForest with CNN or Transformer models trained on real-world signal datasets (RadioML) for higher accuracy.

4. **Additional Modulation Schemes** — Add AM, PSK, QAM, OFDM modulation/demodulation.

5. **Multi-User Collaboration** — WebSocket broadcasting to multiple dashboards for shared spectrum monitoring.

6. **Signal Recording & Playback** — Record raw IQ samples for offline analysis and replay.
