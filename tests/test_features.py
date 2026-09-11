from pathlib import Path
import numpy as np
import pytest

from src.features.extractor import AudioFeatureExtractor
from src.features.temporal import (
    compute_crest_factor,
    compute_attack_decay_slope,
    compute_zero_crossing_statistics,
)
from src.features.spectral import compute_spectral_flux


@pytest.fixture
def extractor():
    cfg_path = Path(__file__).resolve().parent.parent / 'configs' / 'config.yaml'
    return AudioFeatureExtractor(cfg_path)


def test_crest_factor_physics():
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # pure sine crest factor should be ~sqrt(2) = 1.414
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sine_cf = compute_crest_factor(sine)
    assert np.isclose(sine_cf, np.sqrt(2), atol=0.1)

    # gaussian noise crest factor ~3-4
    np.random.seed(42)
    noise = np.random.normal(0, 1, len(t)).astype(np.float32)
    noise_cf = compute_crest_factor(noise)
    assert 2.5 < noise_cf < 5.0

    # impulse spike should have very high crest factor
    impulse = np.zeros(len(t), dtype=np.float32)
    impulse[1000] = 1.0
    impulse_cf = compute_crest_factor(impulse)
    assert impulse_cf > 50.0


def test_zcr_matches_expected_rate():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    zcr_stats = compute_zero_crossing_statistics(sine, frame_length=1024, hop_length=512)
    expected_zcr = 2 * 440 / sr
    assert np.isclose(zcr_stats['zcr_mean'], expected_zcr, atol=0.005)


def test_attack_slope_constant_signal():
    sr = 22050
    constant = np.ones(sr, dtype=np.float32) * 0.5
    result = compute_attack_decay_slope(constant, sr=sr, frame_length=1024, hop_length=512)

    assert result['attack_slope'] == 0.0
    assert np.isfinite(result['attack_slope'])
    assert np.isfinite(result['decay_time_sec'])


def test_attack_slope_impulsive_signal():
    sr = 22050
    audio = np.zeros(sr, dtype=np.float32)
    audio[sr // 2] = 1.0
    result = compute_attack_decay_slope(audio, sr=sr, frame_length=1024, hop_length=512)

    assert np.isfinite(result['attack_slope'])
    assert np.isfinite(result['attack_time_sec'])


def test_spectral_flux_stationary_vs_dynamic(extractor):
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # constant tone vs chirp
    sine = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    sine_feats = extractor.extract_feature_dict(sine, sr)

    f_chirp = np.linspace(100, 5000, len(t))
    chirp = np.sin(2 * np.pi * f_chirp * t).astype(np.float32)
    chirp_feats = extractor.extract_feature_dict(chirp, sr)

    assert chirp_feats['spectral_flux_mean'] > sine_feats['spectral_flux_mean']
    assert chirp_feats['spectral_centroid_std'] > sine_feats['spectral_centroid_std']


def test_feature_vector_shape_and_latency(extractor):
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    vec = extractor.extract_feature_vector(audio, sr)
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert len(vec) == extractor.num_features
    assert np.all(np.isfinite(vec))

    lat = extractor.benchmark_latency(audio, sr, n_iterations=20, warmup=3)
    assert lat['p50_ms'] < 100.0
