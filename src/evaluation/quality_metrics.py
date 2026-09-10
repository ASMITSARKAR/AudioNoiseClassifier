import numpy as np
import scipy.signal as signal


def compute_segmental_snr(clean, processed, sr=22050, frame_ms=30.0, min_snr=-10.0, max_snr=35.0):
    frame_len = int(frame_ms / 1000.0 * sr)
    hop_len = frame_len // 2
    n = min(len(clean), len(processed))
    clean_clip = clean[:n]
    proc_clip = processed[:n]
    noise = proc_clip - clean_clip

    snr_list = []
    for start in range(0, n - frame_len, hop_len):
        c_chunk = clean_clip[start:start + frame_len]
        n_chunk = noise[start:start + frame_len]
        c_energy = np.sum(c_chunk ** 2)
        n_energy = np.sum(n_chunk ** 2)

        if c_energy > 1e-6:
            val = 10.0 * np.log10(c_energy / (n_energy + 1e-12))
            clamped = float(np.clip(val, min_snr, max_snr))
            snr_list.append(clamped)

    if not snr_list:
        return 0.0

    return float(np.mean(snr_list))


def compute_log_spectral_distance(clean, processed, sr=22050, n_fft=512, hop_length=128):
    n = min(len(clean), len(processed))
    c_sig = clean[:n]
    p_sig = processed[:n]

    _, _, z_clean = signal.stft(c_sig, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, z_proc = signal.stft(p_sig, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    p_clean = np.maximum(np.abs(z_clean) ** 2, 1e-10)
    p_proc = np.maximum(np.abs(z_proc) ** 2, 1e-10)

    log_c = 10.0 * np.log10(p_clean)
    log_p = 10.0 * np.log10(p_proc)

    diff_sq = (log_c - log_p) ** 2
    frame_lsd = np.sqrt(np.mean(diff_sq, axis=0))
    return float(np.mean(frame_lsd))


def compute_filter_switch_rate(decision_sequence):
    if len(decision_sequence) <= 1:
        return {'total_switches': 0, 'switch_rate': 0.0}

    switches = 0
    for idx in range(1, len(decision_sequence)):
        if decision_sequence[idx] != decision_sequence[idx - 1]:
            switches += 1

    rate = switches / (len(decision_sequence) - 1)
    return {
        'total_switches': switches,
        'switch_rate': float(rate),
    }
