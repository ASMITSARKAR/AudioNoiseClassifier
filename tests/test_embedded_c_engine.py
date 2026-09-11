import ctypes
from pathlib import Path
import pickle
import numpy as np
import pytest

from src.models.cost_sensitive import COST_MATRIX, predict_bayes_optimal

DLL_PATH = Path('embedded/lib/libnoiserouter.dll')
MODEL_PATH = Path('models/best_baseline.pkl')


def test_embedded_c_decision_forest_parity():
    if not DLL_PATH.exists():
        pytest.skip(f"Embedded C library not found at {DLL_PATH}")

    c_lib = ctypes.CDLL(str(DLL_PATH.resolve()))
    c_lib.hgb_predict_proba.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    c_lib.hgb_predict_proba.restype = None
    c_lib.hgb_predict_bayes.argtypes = [ctypes.POINTER(ctypes.c_float)]
    c_lib.hgb_predict_bayes.restype = ctypes.c_int

    with open(MODEL_PATH, 'rb') as f:
        ckpt = pickle.load(f)
        raw_hgb = ckpt['raw_hgb']
        feat_names = ckpt['feature_names']

    num_features = len(feat_names)
    np.random.seed(123)
    test_vectors = np.random.randn(30, num_features).astype(np.float32)

    py_probs = raw_hgb.predict_proba(test_vectors)
    py_bayes = predict_bayes_optimal(py_probs, COST_MATRIX)

    c_probs_arr = np.zeros(3, dtype=np.float32)
    c_probs_ptr = c_probs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    max_prob_diff = 0.0
    bayes_matches = 0

    for i in range(30):
        vec = test_vectors[i]
        vec_ptr = vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_lib.hgb_predict_proba(vec_ptr, c_probs_ptr)
        c_bayes = c_lib.hgb_predict_bayes(vec_ptr)

        diff = np.max(np.abs(py_probs[i] - c_probs_arr))
        if diff > max_prob_diff:
            max_prob_diff = diff
        if c_bayes == py_bayes[i]:
            bayes_matches += 1

    assert max_prob_diff < 1e-4, f"C probability output deviates from Python: {max_prob_diff}"
    assert bayes_matches == 30, "C Bayes decisions did not match Scikit-Learn exactly"
