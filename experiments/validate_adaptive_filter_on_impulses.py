import numpy as np
import scipy.signal as signal


def generate_test_speech_with_impulse(sr=22050, duration_sec=1.0, impulse_time_sec=0.5, impulse_amplitude=2.5):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    f0 = 200.0
    speech = np.zeros_like(t)
    for harmonic in [1, 2, 3, 4, 5]:
        speech += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)
    speech *= 0.5 * (1 + np.sin(2 * np.pi * 3 * t))

    np.random.seed(42)
    bg = np.random.normal(0, 0.03, len(t))

    impulse = np.zeros_like(t)
    imp_idx = int(impulse_time_sec * sr)
    decay_len = int(0.015 * sr)
    decay_curve = impulse_amplitude * np.exp(-np.linspace(0, 8, decay_len))
    impulse[imp_idx:imp_idx + decay_len] = decay_curve

    noisy_input = speech + bg + impulse
    return speech, noisy_input, impulse


def apply_spectral_subtraction(audio, sr=22050, n_fft=1024, hop_length=256, alpha=2.0):
    _, _, zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    mag = np.abs(zxx)
    phase = np.angle(zxx)

    noise_est = np.mean(mag[:, :5], axis=1, keepdims=True)
    out_mag = np.zeros_like(mag)

    for i in range(mag.shape[1]):
        frame_mag = mag[:, i:i + 1]
        noise_est = 0.95 * noise_est + 0.05 * frame_mag
        sub = np.maximum(frame_mag - alpha * noise_est, 0.05 * frame_mag)
        out_mag[:, i:i + 1] = sub

    _, out_audio = signal.istft(out_mag * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    return out_audio[:len(audio)]


def apply_normalized_lms_adaptive_filter(audio, filter_order=32, mu=0.01, eps=1e-4):
    N = len(audio)
    w = np.zeros(filter_order)
    out = np.zeros(N)
    delay = 4

    for n in range(delay + filter_order, N):
        x_vec = audio[n - delay - filter_order:n - delay][::-1]
        y_hat = np.dot(w, x_vec)
        e = audio[n] - y_hat
        out[n] = e
        norm = np.dot(x_vec, x_vec) + eps
        w += (mu / norm) * e * x_vec

    return out


def apply_median_spike_removal(audio, kernel_size=15, threshold_mad=3.5):
    med = signal.medfilt(audio, kernel_size=kernel_size)
    dev = np.abs(audio - med)
    mad = signal.medfilt(dev, kernel_size=kernel_size) + 1e-6
    outliers = dev > (threshold_mad * mad)

    cleaned = audio.copy()
    cleaned[outliers] = med[outliers]
    return cleaned


def run_dsp_validation():
    sr = 22050
    speech, noisy_input, _ = generate_test_speech_with_impulse(sr=sr)

    out_spec = apply_spectral_subtraction(noisy_input, sr=sr)
    out_nlms = apply_normalized_lms_adaptive_filter(noisy_input)
    out_med = apply_median_spike_removal(noisy_input)

    imp_idx = int(0.5 * sr)
    window = int(0.25 * sr)

    ref = speech[imp_idx:imp_idx + window]
    s_spec = out_spec[imp_idx:imp_idx + window]
    s_nlms = out_nlms[imp_idx:imp_idx + window]
    s_med = out_med[imp_idx:imp_idx + window]

    def sdr(clean, processed):
        err = processed - clean
        return 10.0 * np.log10(np.sum(clean ** 2) / (np.sum(err ** 2) + 1e-10))

    print("\nComparing filter responses to sudden transient impulses:")
    print(f"  1. Spectral Subtraction SDR: {sdr(ref, s_spec):6.2f} dB  (causes smeared temporal tail)")
    print(f"  2. Adaptive NLMS SDR       : {sdr(ref, s_nlms):6.2f} dB  (lets spike leak through)")
    print(f"  3. Median Filter SDR       : {sdr(ref, s_med):6.2f} dB  (clean excision without ringing)\n")


if __name__ == '__main__':
    run_dsp_validation()
