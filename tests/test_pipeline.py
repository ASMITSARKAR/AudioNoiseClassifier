from pathlib import Path
import numpy as np

from src.pipeline.suppression_filters import (
    SpectralSubtractionFilter,
    AdaptiveNLMSFilter,
    MedianMADSpikeFilter,
)
from src.pipeline.noise_suppression_pipeline import NoiseSuppressionPipeline

MODEL_PATH = Path('models/best_baseline.pkl')
CONFIG_PATH = Path('configs/config.yaml')


def test_spectral_subtraction_filter():
    flt = SpectralSubtractionFilter(sr=22050)
    noise = np.random.normal(0, 0.2, 22050).astype(np.float32)
    cleaned = flt.process(noise)

    assert len(cleaned) == len(noise)
    # output energy should decrease after spectral subtraction
    assert np.mean(cleaned ** 2) < np.mean(noise ** 2)


def test_median_mad_spike_filter():
    flt = MedianMADSpikeFilter(kernel_size=15, threshold_mad=3.5)
    t = np.linspace(0, 0.5, int(0.5 * 22050), dtype=np.float32)
    sine = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    noisy = sine.copy()
    spike_idx = 2000
    noisy[spike_idx] = 5.0  # inject large transient click

    cleaned = flt.process(noisy)
    assert len(cleaned) == len(noisy)
    assert np.abs(cleaned[spike_idx]) < 1.0
    assert np.allclose(cleaned[:1900], sine[:1900], atol=1e-3)


def test_adaptive_nlms_filter():
    flt = AdaptiveNLMSFilter(filter_order=32, mu=0.02)
    t = np.linspace(0, 0.5, int(0.5 * 22050), dtype=np.float32)
    sine = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    cleaned = flt.process(sine)
    assert len(cleaned) == len(sine)
    assert not np.isnan(cleaned).any()


def test_noise_suppression_pipeline_e2e():
    pipeline = NoiseSuppressionPipeline(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        sr=22050,
        chunk_sec=0.5
    )
    chunk_samples = int(0.5 * 22050)
    test_chunk = (np.random.randn(chunk_samples) * 0.1).astype(np.float32)

    clean_chunk, meta = pipeline.process_chunk(test_chunk)
    assert len(clean_chunk) == len(test_chunk)
    assert meta['class_id'] in [0, 1, 2]
    assert 'stationary' in meta['probabilities']
    assert meta['timings_ms']['end_to_end_ms'] < 250.0
