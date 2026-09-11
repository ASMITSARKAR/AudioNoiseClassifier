import ctypes
from pathlib import Path
import numpy as np
import pytest

from src.pipeline.suppression_filters import SpectralSubtractionFilter
from src.pipeline.noise_suppression_pipeline import NoiseSuppressionPipeline

DLL_PATH = Path('embedded/lib/libnoiserouter.dll')
MODEL_PATH = Path('models/best_baseline.pkl')
CONFIG_PATH = Path('configs/config.yaml')


def test_spectral_sub_streaming_continuity_python():
    sr = 22050
    chunk_size = 11025
    flt = SpectralSubtractionFilter(sr=sr)

    # 5 seconds of 440Hz sine + background noise
    t = np.linspace(0, 5.0, int(5.0 * sr), endpoint=False, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(t)).astype(np.float32)
    sig = tone + noise

    chunks = []
    for i in range(0, len(sig), chunk_size):
        chk = sig[i:i + chunk_size]
        chunks.append(flt.process(chk))

    streamed = np.concatenate(chunks)

    # verify boundary transitions don't collapse or explode
    for b in [11025, 22050, 33075, 44100]:
        pre_e = np.mean(streamed[b - 256:b] ** 2)
        post_e = np.mean(streamed[b:b + 256] ** 2)
        ratio = post_e / (pre_e + 1e-12)
        assert 0.3 < ratio < 3.0, f"Boundary energy jump at {b}: ratio={ratio:.3f}"
        assert not np.isnan(streamed[b - 50:b + 50]).any()


def test_pipeline_crossfade_no_raw_noise_bleed():
    sr = 22050
    pipeline = NoiseSuppressionPipeline(model_path=MODEL_PATH, config_path=CONFIG_PATH, sr=sr, chunk_sec=0.5)

    np.random.seed(42)
    noise = np.random.normal(0, 0.5, 11025).astype(np.float32)

    _, _ = pipeline.process_chunk(noise)
    pipeline.prev_class_id = 0
    out1, _ = pipeline.process_chunk(noise)

    # crossfade shouldn't amplify beyond raw noise peak
    raw_peak = np.max(np.abs(noise[:330]))
    cf_peak = np.max(np.abs(out1[:330]))
    assert cf_peak <= raw_peak
    assert not np.isnan(out1).any()


def test_c_spectral_sub_streaming_continuity():
    if not DLL_PATH.exists():
        pytest.skip(f"DLL not found: {DLL_PATH}")

    lib = ctypes.CDLL(str(DLL_PATH.resolve()))

    class SpectralSubFilterC(ctypes.Structure):
        _fields_ = [
            ("noise_psd", ctypes.c_float * 257),
            ("in_buffer", ctypes.c_float * 512),
            ("ola_tail", ctypes.c_float * 512),
            ("in_buf_len", ctypes.c_size_t),
            ("alpha", ctypes.c_float),
            ("beta", ctypes.c_float),
            ("smoothing", ctypes.c_float),
            ("is_initialized", ctypes.c_uint8)
        ]

    flt = SpectralSubFilterC()
    lib.spectral_sub_init(ctypes.byref(flt), ctypes.c_float(1.8), ctypes.c_float(0.02), ctypes.c_float(0.98))

    sr = 22050
    chunk_size = 11025
    t = np.linspace(0, 3.0, int(3.0 * sr), endpoint=False, dtype=np.float32)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, len(t)).astype(np.float32)
    sig = tone + noise

    streamed_output = []
    for i in range(0, len(sig), chunk_size):
        chk = sig[i:i + chunk_size].copy()
        out_buf = np.zeros(chunk_size, dtype=np.float32)
        in_p = chk.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out_p = out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.spectral_sub_process(ctypes.byref(flt), in_p, out_p, ctypes.c_size_t(chunk_size))
        streamed_output.append(out_buf)

    streamed = np.concatenate(streamed_output)

    for b in [11025, 22050]:
        pre_e = np.mean(streamed[b - 256:b] ** 2)
        post_e = np.mean(streamed[b:b + 256] ** 2)
        ratio = post_e / (pre_e + 1e-12)
        assert 0.3 < ratio < 3.0, f"C boundary collapse at {b}: ratio={ratio:.3f}"
        assert not np.isnan(streamed[b - 50:b + 50]).any()


def test_pipeline_multi_transition_no_state_corruption():
    sr = 22050
    pipeline = NoiseSuppressionPipeline(model_path=MODEL_PATH, config_path=CONFIG_PATH, sr=sr, chunk_sec=0.5)
    np.random.seed(42)

    outputs = []
    for i in range(10):
        t = np.linspace(0, 0.5, 11025)
        chunk = (0.2 * np.sin(2 * np.pi * (200 + i * 50) * t) + np.random.normal(0, 0.05, 11025)).astype(np.float32)
        forced_class = i % 3
        pipeline.prev_class_id = (forced_class - 1) % 3
        out_chunk, _ = pipeline.process_chunk(chunk)
        outputs.append(out_chunk)
        assert not np.isnan(out_chunk).any()
        assert np.max(np.abs(out_chunk)) < 10.0

    full = np.concatenate(outputs)
    assert len(full) == 10 * 11025
    assert not np.isnan(full).any()
