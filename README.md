# AudioNoiseClassifier

Real-time audio noise regime classification and adaptive DSP filter routing for edge streaming pipelines.

The core ideation and problem formulation for this project was sieved and adapted from a hackathon problem statement centered on intelligent environmental noise suppression on resource-constrained microprocessors.

---

## The Problem

Traditional noise reduction pipelines apply a single static filter across all incoming audio. In dynamic real-world environments, this approach creates severe trade-offs:

1. Spectral Subtraction assumes noise is stationary. When an impulsive noise occurs (such as a door slam or keyboard click), the filter incorporates the high-energy transient into its background noise spectrum estimate. This causes audible musical noise and temporal smearing that distorts subsequent speech for 200 to 400 milliseconds.
2. Normalized Least Mean Squares (NLMS) adaptive filtering tracks smoothly changing signals like sirens or background voices, but transient spikes blow right through the error buffer before filter weights can adapt.
3. Median and MAD (Median Absolute Deviation) filters cleanly excise isolated clicks and pops, but cause non-linear harmonic distortion if applied continuously to sustained speech or background hum.

No single digital filter handles all three noise profiles effectively.

---

## System Design

AudioNoiseClassifier splits noise suppression into a two-stage hybrid system: a lightweight classifier that identifies the current noise regime, and a dynamic router that sends each audio chunk to a specialized filter.

```
 [ Microphone / Audio Stream @ 22.05 kHz ]
                     │
            (500ms chunk hops)
                     │
                     ▼
          1.0s Rolling Ring Buffer
                     │
                     ▼
          92 Acoustic Descriptors
        (Temporal kinetics, spectral flux,
         subband ratios, MFCC dynamics)
                     │
                     ▼
      Cost-Sensitive Decision Forest (C99 / HGB)
    (Bayes risk head penalizes transient leaks)
                     │
        ┌────────────┼────────────┐
        │ Class 0    │ Class 1    │ Class 2
        ▼            ▼            ▼
   [Spectral Sub]  [ NLMS ]    [Median/MAD]
    Stationary    Non-Stationary Impulsive
        │            │            │
        └────────────┼────────────┘
                     │ (quadratic crossfade on switch)
                     ▼
            [ Cleaned Output ]
```

### Noise Regimes and Filter Targets

1. Stationary Noise (wind, air conditioner hum, continuous rain, motor drone)
   - Filter: Overlap-Add (OLA) Spectral Subtraction with recursive noise power spectral density (PSD) tracking and a sine analysis/synthesis window satisfying the Constant Overlap-Add (COLA) condition.
2. Non-Stationary Noise (sirens, background chatter, passing vehicles)
   - Filter: Leaky Normalized Least Mean Squares (NLMS) filter with a decorrelation delay line and gradient clipping. Numba JIT accelerated in Python; pure C99 for embedded targets.
3. Impulsive Noise (door knocks, gunshots, keyboard clicks, microswitch clicks)
   - Filter: Sliding Median and Median Absolute Deviation (MAD) outlier excision filter that detects sharp energy spikes above a local noise floor and replaces outlier samples with the local median.

### Why Cost-Sensitive Learning?

Standard classifiers treat all misclassifications equally under symmetric loss. In an adaptive audio pipeline, misclassification costs are fundamentally asymmetric:
- Misclassifying stationary noise as impulsive triggers a brief median filter for 500ms, which is largely benign.
- Misclassifying an impulsive spike as stationary injects massive energy into the spectral subtraction PSD tracker, corrupting audio for hundreds of milliseconds.

To eliminate transient leakage, decisions use Bayes risk minimization instead of standard argmax:

$$\hat{y} = \arg\min_j \sum_i P(y=i \mid x) \cdot C_{ij}$$

using an asymmetric cost penalty matrix $C$:

| True \ Pred | Stationary | Non-Stationary | Impulsive |
| :--- | :---: | :---: | :---: |
| **Stationary** | 0.0 | 1.0 | 1.5 |
| **Non-Stationary** | 2.0 | 0.0 | 1.0 |
| **Impulsive** | **3.0** | 1.0 | 0.0 |

This heavily penalizes impulsive-to-stationary classification errors during inference without changing model weights.

---

## Acoustic Feature Extraction

Every 500ms audio chunk is evaluated within a 1.0-second sliding window at 22.05 kHz, extracting 92 acoustic descriptors:

- Time-Domain Dynamics: Crest factor, rolling RMS envelope moments (mean, standard deviation, coefficient of variation, peak-to-mean ratio, skewness, kurtosis, top-10% energy share), onset attack slope, exponential decay duration, and zero-crossing rate statistics.
- Spectral Moments: Spectral centroid, spread/bandwidth, spectral rolloff (85% and 95%), spectral flatness, and normalized frame-to-frame spectral flux.
- Subband Energy Distribution: Low (<500 Hz), mid (500–3500 Hz), and high (>3500 Hz) energy ratios.
- MFCC Dynamics: 20 Mel-Frequency Cepstral Coefficients (mean and standard deviation) plus first-order temporal delta dynamics.

Feature extraction is implemented identically in both Python (`src/features/`) and standalone C99 (`embedded/c_src/acoustic_features.c`) with a Pearson correlation exceeding 0.999 across all 92 dimensions.

---

## Repository Structure

```
AudioNoiseClassifier/
├── configs/
│   ├── config.yaml               # Audio parameters and benchmark thresholds
│   └── class_mapping.yaml        # ESC-50 taxonomy mapping (50 categories -> 3 regimes)
├── embedded/
│   ├── c_src/
│   │   ├── acoustic_features.c   # C99 92-feature extraction engine
│   │   ├── acoustic_features.h   # Feature engine headers
│   │   ├── acoustic_tables.h     # Precomputed twiddle factors and Mel filterbanks
│   │   ├── dsp_filters.c         # C99 streaming filter implementations
│   │   ├── dsp_filters.h         # Filter structures and prototypes
│   │   ├── embedded_router.c     # C99 chunk processing and filter routing
│   │   ├── embedded_router.h     # Router definitions
│   │   ├── hgb_forest.c          # Decision trees compiled to static C arrays
│   │   └── hgb_forest.h          # Tree inference prototypes
│   ├── lib/
│   │   └── libnoiserouter.dll    # Compiled shared library for C parity testing
│   ├── profiler/                 # Hardware cycle profiler source
│   └── tests/                    # Embedded pipeline unit tests (C)
├── experiments/                  # Validation experiments
│   ├── cost_sensitive_tuning.py
│   ├── multi_seed_leak_validation.py
│   ├── validate_adaptive_filter_on_impulses.py
│   └── verify_long_stream_latency.py
├── models/                       # Checkpoints
│   ├── best_baseline.pkl         # Trained cost-sensitive HGB model checkpoint
│   └── best_waveform_cnn.pt      # 1D waveform CNN weights
├── outputs/                      # Audio outputs generated by demo scripts
├── src/
│   ├── dataset/                  # Dataset downloader and taxonomy mapper
│   ├── evaluation/               # Metrics, segmental SNR, and LSD benchmarks
│   ├── export/                   # C-tree transpiler and ONNX exporter
│   ├── features/                 # Temporal and spectral acoustic feature extraction
│   ├── models/                   # Baselines, cost-sensitive wrapper, and 1D CNN
│   └── pipeline/                 # Streaming pipeline and DSP filter implementations
├── tests/                        # Automated test suite (pytest)
├── demo_pipeline.py              # Streaming demonstration script
├── pytest.ini                    # Test runner configuration
├── requirements.txt              # Python dependencies
└── train_baselines.py            # 5-fold cross-validation training script
```

---

## Setup and Installation

### Prerequisites
- Python 3.10 or later
- C compiler (`gcc`, `clang`, or MSVC) for the embedded C library

### 1. Python Environment

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# or on Windows:
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Compile Embedded C Shared Library (Optional for C Parity Testing)

```bash
# Windows (MinGW gcc):
gcc -O3 -shared -o embedded/lib/libnoiserouter.dll embedded/c_src/acoustic_features.c embedded/c_src/dsp_filters.c embedded/c_src/embedded_router.c embedded/c_src/hgb_forest.c

# Linux / macOS:
gcc -O3 -shared -fPIC -o embedded/lib/libnoiserouter.so embedded/c_src/acoustic_features.c embedded/c_src/dsp_filters.c embedded/c_src/embedded_router.c embedded/c_src/hgb_forest.c
```

---

## Running the Pipeline

### 1. End-to-End Streaming Demonstration
Processes concatenated environmental noise (wind -> siren -> door knocks) mixed with harmonic speech, performs real-time classification, routes audio chunks to active filters, and exports output WAV files:

```bash
python demo_pipeline.py
```

Outputs written to `outputs/`:
- `demo_noisy_input.wav`: Raw input stream with mixed noise.
- `demo_clean_suppressed.wav`: Cleaned audio stream after adaptive suppression.
- `demo_clean_reference.wav`: Clean speech reference.

### 2. Run Automated Tests
Verifies feature extraction parity, C engine parity, DSP continuity across chunk boundaries, Bayes decision rules, and filter stability:

```bash
pytest tests/ -v
```

### 3. Train Baseline Models
Runs 5-fold cross-validation across Logistic Regression, Random Forest, and HistGradientBoosting:

```bash
python train_baselines.py
```

### 4. Transpile Decision Forest to C99
Converts the trained HistGradientBoosting ensemble into static C99 lookup tables (`hgb_forest.c` and `hgb_forest.h`) with zero heap allocations:

```bash
python src/export/export_c_trees.py
```

### 5. Export Model to ONNX
Serializes the trained classifier to ONNX format and verifies numerical parity against Scikit-Learn:

```bash
python src/export/export_onnx.py
```

---

## Measured Performance

All measurements taken on standard x86 CPU hardware processing 500ms audio chunks (11,025 samples @ 22.05 kHz):

| Metric | Target Specification | Measured Result |
| :--- | :---: | :---: |
| **Python Chunk Turnaround** | < 100.0 ms | 41.9 ms (P50) / 44.1 ms (P95) |
| **Compiled C99 Turnaround** | < 5.0 ms | 0.82 ms (P50) / 1.15 ms (P95) |
| **Real-Time Processing Margin** | > 2x margin | > 11x margin (Python) / > 400x (C99) |
| **5-Fold Cross-Validation Macro F1** | > 80.0% | 86.8% +/- 2.1% |
| **Impulsive -> Stationary Leak Rate** | < 5.0% | 1.4% (with Bayes risk penalty) |
| **Filter Switch Discontinuity** | No audible clicks | Smooth C0 quadratic boundary crossfade |

---

## License

This project is licensed under the MIT License.
The ESC-50 dataset is authored by Karol J. Piczak and licensed under Creative Commons Attribution-NonCommercial (CC BY-NC 3.0).
