import time
import numpy as np
import torch

from src.models.neural import WaveformTransientCNN, HighResLogMelTransform


def test_waveform_transient_cnn_architecture():
    model = WaveformTransientCNN(num_classes=3, base_channels=32)
    params = model.count_parameters()
    assert params < 250000

    x = torch.randn(4, 1, 22050)
    out = model(x)
    assert out.shape == (4, 3)
    assert not torch.isnan(out).any()


def test_high_res_log_mel_transform():
    transform = HighResLogMelTransform(sr=22050, n_fft=256, hop_length=64, n_mels=64)
    audio = torch.randn(2, 22050)
    mel = transform(audio)

    assert mel.shape[0] == 2
    assert mel.shape[1] == 1
    assert mel.shape[2] == 64
    assert 340 <= mel.shape[3] <= 350
    assert not torch.isnan(mel).any()


def test_waveform_cnn_cpu_latency():
    model = WaveformTransientCNN(num_classes=3, base_channels=32)
    model.eval()
    audio = torch.randn(1, 1, 22050)

    with torch.no_grad():
        for _ in range(10):
            _ = model(audio)

        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(audio)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    assert p50 < 15.0
