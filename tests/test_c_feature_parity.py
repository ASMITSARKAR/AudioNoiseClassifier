import ctypes
from pathlib import Path
import numpy as np
import pytest
from src.features.extractor import AudioFeatureExtractor

DLL_PATH = Path('embedded/lib/libnoiserouter.dll')
CONFIG_PATH = Path('configs/config.yaml')


@pytest.fixture(scope='module')
def c_lib():
    if not DLL_PATH.exists():
        pytest.skip(f"DLL not found: {DLL_PATH}")
    lib = ctypes.CDLL(str(DLL_PATH.resolve()))
    lib.extract_acoustic_features_c.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.extract_acoustic_features_c.restype = None
    return lib


@pytest.fixture(scope='module')
def py_extractor():
    return AudioFeatureExtractor(CONFIG_PATH)


def test_all_92_features_c_vs_python_parity(c_lib, py_extractor):
    sr = 22050
    np.random.seed(42)
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)

    signals = [
        ("Sine Tone", 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)),
        ("Chirp", 0.4 * np.sin(2 * np.pi * np.linspace(100, 4000, sr) * t).astype(np.float32)),
        ("Gaussian Noise", np.random.normal(0, 0.2, sr).astype(np.float32)),
        ("Transient Spike", (np.pad([1.0], (1000, sr - 1001))).astype(np.float32)),
    ]

    c_out = np.zeros(92, dtype=np.float32)
    c_out_ptr = c_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    for name, audio in signals:
        py_feats = py_extractor.extract_feature_vector(audio, sr=sr)
        assert len(py_feats) == 92

        audio_ptr = audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_lib.extract_acoustic_features_c(audio_ptr, c_out_ptr)

        assert np.all(np.isfinite(c_out))
        assert np.all(np.isfinite(py_feats))

        corr = np.corrcoef(py_feats, c_out)[0, 1]
        assert corr > 0.99, f"Feature correlation dropped on {name}: {corr:.4f}"
