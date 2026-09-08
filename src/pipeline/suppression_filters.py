import numpy as np
import scipy.signal as signal


class SpectralSubtractionFilter:
    def __init__(self, sr=22050, n_fft=512, hop_length=256, alpha=1.8, beta=0.02, smoothing=0.98):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.beta = beta
        self.smoothing = smoothing

        n = np.arange(n_fft)
        self.window = np.sin(np.pi * (n + 0.5) / n_fft).astype(np.float32)
        cola_sum = n_fft / (2.0 * hop_length)
        self.norm_factor = float(1.0 / cola_sum)

        self.noise_psd = None
        self.in_buffer = np.zeros(n_fft - hop_length, dtype=np.float32)
        self.ola_tail = np.zeros(n_fft, dtype=np.float32)

    def reset(self):
        self.noise_psd = None
        self.in_buffer.fill(0)
        self.ola_tail.fill(0)

    def process(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        n_samples = len(audio)
        if n_samples == 0:
            return audio.copy()

        extended = np.concatenate([self.in_buffer, audio])
        output = np.zeros(n_samples + self.n_fft, dtype=np.float32)
        output[:len(self.ola_tail)] += self.ola_tail
        self.ola_tail.fill(0)

        n_frames = max(0, (len(extended) - self.n_fft) // self.hop_length + 1)
        for i in range(n_frames):
            start = i * self.hop_length
            frame = extended[start:start + self.n_fft]
            w_frame = frame * self.window
            spec = np.fft.rfft(w_frame)
            mag = np.abs(spec)
            phase = np.angle(spec)
            power = mag ** 2

            if self.noise_psd is None:
                self.noise_psd = power.copy()
            else:
                self.noise_psd = self.smoothing * self.noise_psd + (1.0 - self.smoothing) * power

            sub = 1.0 - (self.alpha * self.noise_psd) / (power + 1e-10)
            gain = np.clip(sub, self.beta, 1.0)

            clean_spec = (mag * gain) * np.exp(1j * phase)
            synth = np.fft.irfft(clean_spec, n=self.n_fft) * self.window * self.norm_factor
            output[start:start + self.n_fft] += synth

        consumed = n_frames * self.hop_length
        rem = extended[consumed:]
        buf_len = self.n_fft - self.hop_length

        if len(rem) >= buf_len:
            self.in_buffer = rem[-buf_len:].copy()
        else:
            self.in_buffer.fill(0)
            if len(rem) > 0:
                self.in_buffer[-len(rem):] = rem

        res = output[:n_samples].astype(np.float32)
        self.ola_tail = output[n_samples:n_samples + self.n_fft].copy()
        return res


try:
    import numba

    @numba.njit(fastmath=True)
    def _nlms_loop_fast(extended, buf_len, N, order, delay, mu, leak, eps, w):
        output = np.zeros(N, dtype=np.float32)
        for i in range(N):
            n = buf_len + i
            y_hat = 0.0
            norm = eps
            for k in range(order):
                sample = extended[n - delay - 1 - k]
                y_hat += w[k] * sample
                norm += sample * sample

            d = extended[n]
            e = d - y_hat
            output[i] = e

            step = (mu / norm) * e
            for k in range(order):
                grad = step * extended[n - delay - 1 - k]
                if grad > 0.01:
                    grad = 0.01
                elif grad < -0.01:
                    grad = -0.01
                w[k] = leak * w[k] + grad

        return output, w

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


class AdaptiveNLMSFilter:
    def __init__(self, filter_order=16, mu=0.002, leak=0.999, delay=8, eps=0.001):
        self.order = filter_order
        self.mu = mu
        self.leak = leak
        self.delay = delay
        self.eps = eps
        self.weights = np.zeros(filter_order, dtype=np.float32)
        self.buf_len = delay + filter_order + 64
        self.buffer = np.zeros(self.buf_len, dtype=np.float32)

        if HAS_NUMBA:
            dummy_ext = np.zeros(100, dtype=np.float32)
            dummy_w = np.zeros(filter_order, dtype=np.float32)
            _ = _nlms_loop_fast(dummy_ext, 50, 50, filter_order, delay, mu, leak, eps, dummy_w)

    def reset(self):
        self.weights = np.zeros(self.order, dtype=np.float32)
        self.buffer = np.zeros(self.buf_len, dtype=np.float32)

    def process(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        N = len(audio)
        buf_len = self.buf_len
        extended = np.concatenate([self.buffer, audio])

        if HAS_NUMBA:
            output, self.weights = _nlms_loop_fast(
                extended, buf_len, N, self.order, self.delay, self.mu, self.leak, self.eps, self.weights
            )
        else:
            output = np.zeros(N, dtype=np.float32)
            w = self.weights
            order = self.order
            delay = self.delay
            mu = self.mu
            leak = self.leak
            eps = self.eps

            for i in range(N):
                n = buf_len + i
                x_vec = extended[n - delay - order:n - delay][::-1]
                y_hat = float(np.dot(w, x_vec))
                d = float(extended[n])
                e = d - y_hat
                output[i] = e
                norm = float(np.dot(x_vec, x_vec)) + eps
                grad = (mu / norm) * e * x_vec
                w = leak * w + np.clip(grad, -0.01, 0.01)
            self.weights = w

        self.buffer = extended[-buf_len:]
        return output.astype(np.float32)


class MedianMADSpikeFilter:
    def __init__(self, kernel_size=15, threshold_mad=3.5):
        assert kernel_size % 2 == 1, 'kernel_size must be odd'
        self.kernel_size = kernel_size
        self.threshold_mad = threshold_mad

    def reset(self):
        pass

    def process(self, audio):
        if len(audio) < self.kernel_size:
            return audio.copy()

        med = signal.medfilt(audio, kernel_size=self.kernel_size)
        dev = np.abs(audio - med)
        mad = signal.medfilt(dev, kernel_size=self.kernel_size)
        mad_floor = 0.05 * (np.std(audio) + 1e-6)
        mad = np.maximum(mad, mad_floor)

        outliers = dev > (self.threshold_mad * mad)
        cleaned = audio.copy()
        cleaned[outliers] = med[outliers]
        return cleaned.astype(np.float32)
