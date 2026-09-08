import numpy as np
from scipy import stats


def compute_crest_factor(audio, eps=1e-10):
    rms = np.sqrt(np.mean(audio ** 2) + eps)
    peak = np.max(np.abs(audio))
    return float(peak / (rms + eps))


def compute_rms_envelope(audio, frame_length=512, hop_length=256, eps=1e-10):
    n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
    shape = (n_frames, frame_length)
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()
    return np.sqrt(np.mean(frames ** 2, axis=1) + eps)


def compute_envelope_statistics(audio, frame_length=512, hop_length=256, eps=1e-10):
    rms = compute_rms_envelope(audio, frame_length, hop_length, eps)
    mean_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))
    max_rms = float(np.max(rms))

    denom = mean_rms + eps
    cv_rms = std_rms / denom
    peak_to_mean = max_rms / denom

    if std_rms < 1e-6:
        kurt = 0.0
        skew = 0.0
    else:
        kurt = float(stats.kurtosis(rms))
        skew = float(stats.skew(rms))

    sorted_rms = np.sort(rms)
    cutoff = int(0.9 * len(sorted_rms))
    top_energy = np.sum(sorted_rms[cutoff:] ** 2)
    total_energy = np.sum(sorted_rms ** 2) + eps
    top_10_share = float(top_energy / total_energy)

    return {
        'rms_mean': mean_rms,
        'rms_std': std_rms,
        'rms_cov': cv_rms,
        'rms_peak_to_mean': peak_to_mean,
        'rms_kurtosis': kurt,
        'rms_skewness': skew,
        'rms_top10_share': top_10_share,
    }


def compute_attack_decay_slope(audio, sr=22050, frame_length=512, hop_length=256, eps=1e-10):
    rms = compute_rms_envelope(audio, frame_length, hop_length, eps)
    peak_idx = int(np.argmax(rms))
    peak_val = rms[peak_idx]

    if peak_val < eps:
        return {'attack_time_sec': 0.0, 'attack_slope': 0.0, 'decay_time_sec': 0.0}

    onset_thresh = 0.1 * peak_val
    pre_peak = rms[:peak_idx + 1]
    candidates = np.where(pre_peak <= onset_thresh)[0]
    onset_idx = candidates[-1] if len(candidates) > 0 else 0

    dt = hop_length / sr
    attack_frames = peak_idx - onset_idx
    attack_time = attack_frames * dt

    if attack_time < dt:
        attack_slope = 0.0
    else:
        attack_slope = (peak_val - onset_thresh) / attack_time

    decay_thresh = 0.3 * peak_val
    post_peak = rms[peak_idx:]
    decay_candidates = np.where(post_peak <= decay_thresh)[0]
    decay_idx = decay_candidates[0] if len(decay_candidates) > 0 else len(post_peak) - 1
    decay_time = decay_idx * dt

    return {
        'attack_time_sec': float(attack_time),
        'attack_slope': float(attack_slope),
        'decay_time_sec': float(decay_time),
    }


def compute_zero_crossing_statistics(audio, frame_length=512, hop_length=256):
    n_frames = max(1, 1 + (len(audio) - frame_length) // hop_length)
    shape = (n_frames, frame_length)
    strides = (audio.strides[0] * hop_length, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides).copy()

    signs = np.sign(frames)
    signs[signs == 0] = 1
    diffs = np.diff(signs, axis=1)
    zcr = np.mean(np.abs(diffs) > 0, axis=1)

    return {
        'zcr_mean': float(np.mean(zcr)),
        'zcr_std': float(np.std(zcr)),
        'zcr_max': float(np.max(zcr)),
    }
