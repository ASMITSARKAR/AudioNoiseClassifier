from pathlib import Path
import time
import numpy as np
import yaml
import librosa

from src.features.temporal import (
    compute_crest_factor,
    compute_envelope_statistics,
    compute_attack_decay_slope,
    compute_zero_crossing_statistics,
)
from src.features.spectral import (
    compute_spectral_flux,
    compute_spectral_shape_statistics,
    compute_subband_energy_ratios,
    compute_mfcc_dynamics,
)


class AudioFeatureExtractor:
    def __init__(self, config_path=None):
        audio_cfg = {}
        if config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            audio_cfg = cfg.get('audio', {})

        self.sr = int(audio_cfg.get('sample_rate', 22050))
        self.n_fft = int(audio_cfg.get('n_fft', 1024))
        self.hop_length = int(audio_cfg.get('hop_length', 512))
        self.win_length = int(audio_cfg.get('win_length', 1024))
        self.window = audio_cfg.get('window', 'hann')
        self.n_mfcc = int(audio_cfg.get('n_mfcc', 20))

        dummy = np.zeros(self.sr, dtype=np.float32)
        sample_dict = self.extract_feature_dict(dummy, self.sr)
        self._feature_names = sorted(list(sample_dict.keys()))

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def num_features(self):
        return len(self._feature_names)

    def extract_feature_dict(self, audio, sr=None):
        target_sr = sr or self.sr

        stft = librosa.stft(
            y=audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        stft_mag = np.abs(stft)

        feats = {}
        feats['crest_factor'] = compute_crest_factor(audio)

        env_stats = compute_envelope_statistics(audio, frame_length=self.win_length, hop_length=self.hop_length)
        feats.update(env_stats)

        kinetics = compute_attack_decay_slope(audio, sr=target_sr, frame_length=self.win_length, hop_length=self.hop_length)
        feats.update(kinetics)

        zcr = compute_zero_crossing_statistics(audio, frame_length=self.win_length, hop_length=self.hop_length)
        feats.update(zcr)

        flux = compute_spectral_flux(stft_mag)
        feats.update(flux)

        shape_stats = compute_spectral_shape_statistics(stft_mag, sr=target_sr, n_fft=self.n_fft)
        feats.update(shape_stats)

        subbands = compute_subband_energy_ratios(stft_mag, sr=target_sr, n_fft=self.n_fft)
        feats.update(subbands)

        mfcc = compute_mfcc_dynamics(audio, sr=target_sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        feats.update(mfcc)

        return feats

    def extract_feature_vector(self, audio, sr=None):
        feat_dict = self.extract_feature_dict(audio, sr)
        vec = np.empty(len(self._feature_names), dtype=np.float32)
        for idx, key in enumerate(self._feature_names):
            vec[idx] = feat_dict[key]
        return vec

    def benchmark_latency(self, audio_clip, sr=None, n_iterations=100, warmup=10):
        target_sr = sr or self.sr
        for _ in range(warmup):
            _ = self.extract_feature_vector(audio_clip, target_sr)

        latencies = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            _ = self.extract_feature_vector(audio_clip, target_sr)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        lat_arr = np.array(latencies)
        return {
            'p50_ms': float(np.percentile(lat_arr, 50)),
            'p95_ms': float(np.percentile(lat_arr, 95)),
            'mean_ms': float(np.mean(lat_arr)),
            'std_ms': float(np.std(lat_arr)),
        }
