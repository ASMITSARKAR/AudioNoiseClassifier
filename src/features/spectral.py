import numpy as np
import librosa


def compute_spectral_flux(stft_mag, eps=1e-10):
    norms = np.linalg.norm(stft_mag, axis=0, keepdims=True) + eps
    norm_mag = stft_mag / norms
    diff = np.diff(norm_mag, axis=1)
    flux = np.sqrt(np.sum(diff ** 2, axis=0) + eps)

    return {
        'spectral_flux_mean': float(np.mean(flux)),
        'spectral_flux_std': float(np.std(flux)),
        'spectral_flux_max': float(np.max(flux)),
        'spectral_flux_p90': float(np.percentile(flux, 90)),
    }


def compute_spectral_shape_statistics(stft_mag, sr=22050, n_fft=1024):
    centroid = librosa.feature.spectral_centroid(S=stft_mag, sr=sr, n_fft=n_fft)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft_mag, sr=sr, n_fft=n_fft)[0]
    rolloff85 = librosa.feature.spectral_rolloff(S=stft_mag, sr=sr, n_fft=n_fft, roll_percent=0.85)[0]
    rolloff95 = librosa.feature.spectral_rolloff(S=stft_mag, sr=sr, n_fft=n_fft, roll_percent=0.95)[0]
    flatness = librosa.feature.spectral_flatness(S=stft_mag)[0]

    return {
        'spectral_centroid_mean': float(np.mean(centroid)),
        'spectral_centroid_std': float(np.std(centroid)),
        'spectral_bandwidth_mean': float(np.mean(bandwidth)),
        'spectral_bandwidth_std': float(np.std(bandwidth)),
        'spectral_rolloff_85_mean': float(np.mean(rolloff85)),
        'spectral_rolloff_85_std': float(np.std(rolloff85)),
        'spectral_rolloff_95_mean': float(np.mean(rolloff95)),
        'spectral_rolloff_95_std': float(np.std(rolloff95)),
        'spectral_flatness_mean': float(np.mean(flatness)),
        'spectral_flatness_std': float(np.std(flatness)),
        'spectral_flatness_max': float(np.max(flatness)),
    }


def compute_subband_energy_ratios(stft_mag, sr=22050, n_fft=1024, eps=1e-10):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power = np.mean(stft_mag ** 2, axis=1)
    total_power = np.sum(power) + eps

    low_idx = np.where((freqs >= 20.0) & (freqs < 500.0))[0]
    mid_idx = np.where((freqs >= 500.0) & (freqs < 3500.0))[0]
    high_idx = np.where(freqs >= 3500.0)[0]

    low_ratio = float(np.sum(power[low_idx]) / total_power)
    mid_ratio = float(np.sum(power[mid_idx]) / total_power)
    high_ratio = float(np.sum(power[high_idx]) / total_power)

    return {
        'energy_ratio_low': low_ratio,
        'energy_ratio_mid': mid_ratio,
        'energy_ratio_high': high_ratio,
    }


def compute_mfcc_dynamics(audio, sr=22050, n_mfcc=20, n_fft=1024, hop_length=512):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    mfcc_delta = librosa.feature.delta(mfcc)

    feats = {}
    for i in range(n_mfcc):
        feats[f'mfcc_{i + 1}_mean'] = float(np.mean(mfcc[i]))
        feats[f'mfcc_{i + 1}_std'] = float(np.std(mfcc[i]))
        feats[f'mfcc_delta_{i + 1}_std'] = float(np.std(mfcc_delta[i]))

    return feats
