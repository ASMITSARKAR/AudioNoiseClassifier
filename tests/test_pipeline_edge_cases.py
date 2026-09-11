import time
from pathlib import Path
import numpy as np
import pytest

from src.pipeline.suppression_filters import AdaptiveNLMSFilter
from src.pipeline.noise_suppression_pipeline import NoiseSuppressionPipeline
from src.dataset.loader import download_esc50, load_esc50_metadata, load_and_resample_audio
from src.dataset.mapper import ClassTaxonomyMapper

MODEL_PATH = Path('models/best_baseline.pkl')
CONFIG_PATH = Path('configs/config.yaml')


def test_numba_vs_python_nlms_numerical_equivalence():
    sr = 22050
    np.random.seed(42)
    t = np.linspace(0, 2.0, int(2.0 * sr), dtype=np.float32)
    sig = 0.3 * np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.1, len(t)).astype(np.float32)

    def run_ref_nlms(audio, order=16, mu=0.002, leak=0.999, delay=8, eps=0.001):
        buf_len = delay + order + 64
        buf = np.zeros(buf_len, dtype=np.float32)
        w = np.zeros(order, dtype=np.float32)
        chunk_size = 11025
        chunks = []

        for i in range(0, len(audio), chunk_size):
            chk = audio[i:i + chunk_size]
            N = len(chk)
            ext = np.concatenate([buf, chk])
            out = np.zeros(N, dtype=np.float32)

            for idx in range(N):
                n = buf_len + idx
                x_vec = ext[n - delay - order:n - delay][::-1]
                y_hat = float(np.dot(w, x_vec))
                d = float(ext[n])
                e = d - y_hat
                out[idx] = e
                norm = float(np.dot(x_vec, x_vec)) + eps
                grad = (mu / norm) * e * x_vec
                w = leak * w + np.clip(grad, -0.01, 0.01)

            buf = ext[-buf_len:]
            chunks.append(out)

        return np.concatenate(chunks), w

    ref_out, ref_w = run_ref_nlms(sig)

    numba_flt = AdaptiveNLMSFilter(filter_order=16, mu=0.002, leak=0.999, delay=8, eps=0.001)
    numba_chunks = []
    for i in range(0, len(sig), 11025):
        numba_chunks.append(numba_flt.process(sig[i:i + 11025]))
    numba_out = np.concatenate(numba_chunks)

    max_diff = np.max(np.abs(ref_out - numba_out))
    assert max_diff < 1e-4, f"Numba output mismatch: {max_diff}"
    assert np.allclose(numba_flt.weights, ref_w, atol=1e-4)


def test_isolated_single_chunk_impulsive_routing():
    sr = 22050
    raw_dir = Path('data/raw')
    meta_csv = raw_dir / 'ESC-50-master' / 'meta' / 'esc50.csv'

    if not meta_csv.exists():
        try:
            esc50_dir = download_esc50(raw_dir)
        except Exception as e:
            pytest.skip(f"ESC-50 offline: {e}")
    else:
        esc50_dir = raw_dir / 'ESC-50-master'

    mapper = ClassTaxonomyMapper('configs/class_mapping.yaml')
    meta_df = load_esc50_metadata(esc50_dir, mapper)

    stat_row = meta_df[meta_df['category'] == 'wind'].iloc[0]
    imp_row = meta_df[meta_df['category'] == 'door_wood_knock'].iloc[0]

    audio_stat, _ = load_and_resample_audio(stat_row['audio_path'], target_sr=sr)
    audio_imp, _ = load_and_resample_audio(imp_row['audio_path'], target_sr=sr)

    audio_stat = 0.3 * (audio_stat / (np.max(np.abs(audio_stat)) + 1e-6))
    audio_imp = 0.35 * (audio_imp / (np.max(np.abs(audio_imp)) + 1e-6))

    # stream: stationary -> stationary -> impulse -> stationary -> stationary
    stream = [
        audio_stat[:11025],
        audio_stat[11025:22050],
        audio_imp[:11025],
        audio_stat[22050:33075],
        audio_stat[33075:44100],
    ]

    pipeline = NoiseSuppressionPipeline(model_path=MODEL_PATH, config_path=CONFIG_PATH, sr=sr, chunk_sec=0.5)
    decisions = []
    for chk in stream:
        _, meta = pipeline.process_chunk(chk, apply_smoothing=True)
        decisions.append(meta['class_id'])

    assert decisions[0] == 0
    assert decisions[1] == 0
    assert decisions[2] == 2, f"Transient missed: got class {decisions[2]}"
    assert decisions[4] == 0


def test_pipeline_latency_breakdown():
    sr = 22050
    flt = AdaptiveNLMSFilter(filter_order=16, mu=0.002, leak=0.999, delay=8)
    chunk = np.random.randn(int(0.5 * sr)).astype(np.float32)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = flt.process(chunk)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    assert p50 < 5.0
