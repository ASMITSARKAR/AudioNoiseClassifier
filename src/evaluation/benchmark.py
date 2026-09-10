import time
import pickle
import io
import numpy as np


def profile_model_inference(model, X_sample, n_iterations=1000, warmup=50):
    assert X_sample.ndim == 2, 'X_sample must be 2D: (1, n_features)'

    for _ in range(warmup):
        _ = model.predict(X_sample)

    latencies = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = model.predict(X_sample)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    lat = np.array(latencies)
    buf = io.BytesIO()
    pickle.dump(model, buf)
    model_kb = buf.tell() / 1024.0

    return {
        'p50_ms': float(np.percentile(lat, 50)),
        'p95_ms': float(np.percentile(lat, 95)),
        'p99_ms': float(np.percentile(lat, 99)),
        'mean_ms': float(np.mean(lat)),
        'std_ms': float(np.std(lat)),
        'model_kb': float(model_kb),
    }


def profile_end_to_end(feature_extractor, model, audio_clip, sr=22050, n_iterations=500, warmup=30):
    for _ in range(warmup):
        vec = feature_extractor.extract_feature_vector(audio_clip, sr)
        _ = model.predict(vec.reshape(1, -1))

    feat_lats = []
    model_lats = []
    total_lats = []

    for _ in range(n_iterations):
        t0 = time.perf_counter()
        vec = feature_extractor.extract_feature_vector(audio_clip, sr)
        t1 = time.perf_counter()
        _ = model.predict(vec.reshape(1, -1))
        t2 = time.perf_counter()

        feat_lats.append((t1 - t0) * 1000.0)
        model_lats.append((t2 - t1) * 1000.0)
        total_lats.append((t2 - t0) * 1000.0)

    def _calc_stats(arr):
        a = np.array(arr)
        return {
            'p50_ms': float(np.percentile(a, 50)),
            'p95_ms': float(np.percentile(a, 95)),
            'p99_ms': float(np.percentile(a, 99)),
            'mean_ms': float(np.mean(a)),
        }

    return {
        'feature_extraction': _calc_stats(feat_lats),
        'model_inference': _calc_stats(model_lats),
        'end_to_end': _calc_stats(total_lats),
    }


def format_latency_report(model_name, latency, budget_ms=10.0):
    e2e = latency['end_to_end']
    feat = latency['feature_extraction']
    inf = latency['model_inference']
    budget_ok = e2e['p99_ms'] <= budget_ms
    status = 'Within budget' if budget_ok else 'Exceeds budget'

    lines = [
        f"\nLatency Benchmark: {model_name}",
        f"  Feature Extraction : P50 = {feat['p50_ms']:.2f} ms | P99 = {feat['p99_ms']:.2f} ms",
        f"  Model Inference    : P50 = {inf['p50_ms']:.2f} ms | P99 = {inf['p99_ms']:.2f} ms",
        f"  End-to-End Total   : P50 = {e2e['p50_ms']:.2f} ms | P95 = {e2e['p95_ms']:.2f} ms | P99 = {e2e['p99_ms']:.2f} ms",
        f"  Target ({budget_ms:.0f} ms max)   : {status}",
    ]
    return '\n'.join(lines)
