/*
 * Embedded DSP Filter Implementations (Zero-allocation, streaming-safe C99).
 */

#include "dsp_filters.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ========================================================================= */
/* 1. Adaptive Normalized Least-Mean-Squares (NLMS) Filter                  */
/* ========================================================================= */

void nlms_filter_init(NLMSFilter* f, float mu, float leak, float eps) {
    f->mu = mu;
    f->leak = leak;
    f->eps = eps;
    nlms_filter_reset(f);
}

void nlms_filter_reset(NLMSFilter* f) {
    memset(f->weights, 0, sizeof(f->weights));
    memset(f->buffer, 0, sizeof(f->buffer));
}

void nlms_filter_process(NLMSFilter* f, const float* input, float* output, size_t n_samples) {
    const int order = NLMS_ORDER;
    const int delay = NLMS_DELAY;
    const int buf_len = NLMS_BUF_LEN;
    const float mu = f->mu;
    const float leak = f->leak;
    const float eps = f->eps;

    /* Cache raw input tail to support in-place processing where input == output */
    float tail_cache[NLMS_BUF_LEN] = {0};
    if (n_samples >= (size_t)buf_len) {
        memcpy(tail_cache, &input[n_samples - buf_len], buf_len * sizeof(float));
    } else {
        memcpy(tail_cache, input, n_samples * sizeof(float));
    }

    for (size_t i = 0; i < n_samples; ++i) {
        float d = input[i];
        float y_hat = 0.0f;
        float norm = eps;

        for (int k = 0; k < order; ++k) {
            int tap_idx = (int)i - delay - 1 - k;
            float sample = (tap_idx >= 0) ? input[tap_idx] : f->buffer[buf_len + tap_idx];
            y_hat += f->weights[k] * sample;
            norm += sample * sample;
        }

        float e = d - y_hat;  /* Error residual: signal minus predictable noise */

        /* Gradient descent weight update with leak and gradient clipping */
        float step = (mu / norm) * e;
        for (int k = 0; k < order; ++k) {
            int tap_idx = (int)i - delay - 1 - k;
            float sample = (tap_idx >= 0) ? input[tap_idx] : f->buffer[buf_len + tap_idx];
            float grad = step * sample;
            if (grad > 0.01f) grad = 0.01f;
            else if (grad < -0.01f) grad = -0.01f;
            f->weights[k] = leak * f->weights[k] + grad;
        }

        output[i] = e;
    }

    /* Update history buffer for next chunk */
    if (n_samples >= (size_t)buf_len) {
        memcpy(f->buffer, tail_cache, buf_len * sizeof(float));
    } else {
        memmove(f->buffer, &f->buffer[n_samples], (buf_len - n_samples) * sizeof(float));
        memcpy(&f->buffer[buf_len - n_samples], tail_cache, n_samples * sizeof(float));
    }
}

/* ========================================================================= */
/* 2. Median / MAD Outlier Spike Filter                                     */
/* ========================================================================= */

void median_mad_init(MedianMADFilter* f, float threshold_mad) {
    f->threshold_mad = threshold_mad;
}

void median_mad_reset(MedianMADFilter* f) {
    (void)f;
}

static float fast_median_15(float* arr) {
    float w[MEDIAN_KERNEL_SIZE];
    memcpy(w, arr, sizeof(w));
    for (int i = 1; i < MEDIAN_KERNEL_SIZE; ++i) {
        float key = w[i];
        int j = i - 1;
        while (j >= 0 && w[j] > key) {
            w[j + 1] = w[j];
            j--;
        }
        w[j + 1] = key;
    }
    return w[MEDIAN_KERNEL_SIZE / 2];
}

static float get_sample_median(const float* input, size_t n_samples, int center_idx) {
    const int half_k = MEDIAN_KERNEL_SIZE / 2;
    float w[MEDIAN_KERNEL_SIZE];
    for (int k = -half_k; k <= half_k; ++k) {
        int idx = center_idx + k;
        if (idx < 0) idx = 0;
        else if (idx >= (int)n_samples) idx = (int)n_samples - 1;
        w[k + half_k] = input[idx];
    }
    return fast_median_15(w);
}

void median_mad_process(MedianMADFilter* f, const float* input, float* output, size_t n_samples) {
    if (n_samples < MEDIAN_KERNEL_SIZE) {
        memcpy(output, input, n_samples * sizeof(float));
        return;
    }

    /* Compute chunk energy to establish minimum MAD floor */
    float sum = 0.0f, sum_sq = 0.0f;
    for (size_t i = 0; i < n_samples; ++i) {
        sum += input[i];
        sum_sq += input[i] * input[i];
    }
    float mean = sum / (float)n_samples;
    float variance = (sum_sq / (float)n_samples) - (mean * mean);
    float sig_std = (variance > 0.0f) ? sqrtf(variance) : 1e-4f;
    float mad_floor = 0.05f * (sig_std + 1e-6f);

    const int half_k = MEDIAN_KERNEL_SIZE / 2;
    float dev_buf[MEDIAN_KERNEL_SIZE];

    for (size_t i = 0; i < n_samples; ++i) {
        float center_med = get_sample_median(input, n_samples, (int)i);
        float center_dev = fabsf(input[i] - center_med);

        for (int k = -half_k; k <= half_k; ++k) {
            int idx = (int)i + k;
            if (idx < 0) idx = 0;
            else if (idx >= (int)n_samples) idx = (int)n_samples - 1;

            float neighbor_med = get_sample_median(input, n_samples, idx);
            dev_buf[k + half_k] = fabsf(input[idx] - neighbor_med);
        }

        float mad = fast_median_15(dev_buf);
        if (mad < mad_floor) mad = mad_floor;

        if (center_dev > (f->threshold_mad * mad)) {
            output[i] = center_med;
        } else {
            output[i] = input[i];
        }
    }
}

/* ========================================================================= */
/* 3. Spectral Subtraction Filter (512-point Radix-2 FFT, 50% Overlap-Add)   */
/* ========================================================================= */

static void bit_reverse_512(float* real, float* imag) {
    int j = 0;
    for (int i = 0; i < 512 - 1; ++i) {
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
        int k = 256;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
}

static void fft_512(float* real, float* imag, int inverse) {
    bit_reverse_512(real, imag);

    for (int len = 2; len <= 512; len <<= 1) {
        float angle = (inverse ? 2.0f : -2.0f) * (float)M_PI / (float)len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);

        for (int i = 0; i < 512; i += len) {
            float w_r = 1.0f;
            float w_i = 0.0f;
            int half_len = len >> 1;

            for (int j = 0; j < half_len; ++j) {
                int u_idx = i + j;
                int v_idx = i + j + half_len;

                float u_r = real[u_idx];
                float u_i = imag[u_idx];
                float v_r = real[v_idx] * w_r - imag[v_idx] * w_i;
                float v_i = real[v_idx] * w_i + imag[v_idx] * w_r;

                real[u_idx] = u_r + v_r;
                imag[u_idx] = u_i + v_i;
                real[v_idx] = u_r - v_r;
                imag[v_idx] = u_i - v_i;

                float next_w_r = w_r * wlen_r - w_i * wlen_i;
                w_i = w_r * wlen_i + w_i * wlen_r;
                w_r = next_w_r;
            }
        }
    }

    if (inverse) {
        const float inv_n = 1.0f / 512.0f;
        for (int i = 0; i < 512; ++i) {
            real[i] *= inv_n;
            imag[i] *= inv_n;
        }
    }
}

void spectral_sub_init(SpectralSubFilter* f, float alpha, float beta, float smoothing) {
    f->alpha = alpha;
    f->beta = beta;
    f->smoothing = smoothing;
    spectral_sub_reset(f);
}

void spectral_sub_reset(SpectralSubFilter* f) {
    memset(f->noise_psd, 0, sizeof(f->noise_psd));
    memset(f->in_buffer, 0, sizeof(f->in_buffer));
    memset(f->ola_tail, 0, sizeof(f->ola_tail));
    f->in_buf_len = 0;
    f->is_initialized = 0;
}

void spectral_sub_process(SpectralSubFilter* f, const float* input, float* output, size_t n_samples) {
    if (n_samples == 0) return;

    const size_t n_fft = SPEC_SUB_N_FFT;
    const size_t hop = SPEC_SUB_HOP;
    const size_t in_buf_len = f->in_buf_len;
    const size_t total_in_len = in_buf_len + n_samples;
    const float alpha = f->alpha;
    const float beta = f->beta;
    const float smoothing = f->smoothing;

    memset(output, 0, n_samples * sizeof(float));

    /* Fold in overlap-add tail from previous frame */
    for (size_t i = 0; i < n_fft && i < n_samples; ++i) {
        output[i] += f->ola_tail[i];
    }
    if (n_samples >= n_fft) {
        memset(f->ola_tail, 0, sizeof(f->ola_tail));
    } else {
        memmove(f->ola_tail, &f->ola_tail[n_samples], (n_fft - n_samples) * sizeof(float));
        memset(&f->ola_tail[n_fft - n_samples], 0, n_samples * sizeof(float));
    }

    float frame_re[SPEC_SUB_N_FFT];
    float frame_im[SPEC_SUB_N_FFT];

    size_t num_frames = (total_in_len >= n_fft) ? ((total_in_len - n_fft) / hop + 1) : 0;

    for (size_t f_idx = 0; f_idx < num_frames; ++f_idx) {
        size_t start = f_idx * hop;

        for (size_t k = 0; k < n_fft; ++k) {
            size_t m = start + k;
            float s = (m < in_buf_len) ? f->in_buffer[m] : input[m - in_buf_len];
            float win = sinf((float)M_PI * ((float)k + 0.5f) / (float)n_fft);
            frame_re[k] = s * win;
            frame_im[k] = 0.0f;
        }

        fft_512(frame_re, frame_im, 0);

        for (size_t k = 0; k < SPEC_SUB_BINS; ++k) {
            float pwr = frame_re[k] * frame_re[k] + frame_im[k] * frame_im[k];
            if (!f->is_initialized) {
                f->noise_psd[k] = pwr;
            } else {
                f->noise_psd[k] = smoothing * f->noise_psd[k] + (1.0f - smoothing) * pwr;
            }

            float gain = 1.0f - (alpha * f->noise_psd[k]) / (pwr + 1e-10f);
            if (gain < beta) gain = beta;
            else if (gain > 1.0f) gain = 1.0f;

            frame_re[k] *= gain;
            frame_im[k] *= gain;

            if (k > 0 && k < SPEC_SUB_BINS - 1) {
                size_t sym_k = n_fft - k;
                frame_re[sym_k] = frame_re[k];
                frame_im[sym_k] = -frame_im[k];
            }
        }
        f->is_initialized = 1;

        fft_512(frame_re, frame_im, 1);

        for (size_t k = 0; k < n_fft; ++k) {
            float win = sinf((float)M_PI * ((float)k + 0.5f) / (float)n_fft);
            float sample = frame_re[k] * win;
            size_t pos = start + k;

            if (pos < n_samples) {
                output[pos] += sample;
            } else {
                size_t tail_idx = pos - n_samples;
                if (tail_idx < n_fft) {
                    f->ola_tail[tail_idx] += sample;
                }
            }
        }
    }

    size_t consumed = num_frames * hop;
    size_t leftover = (total_in_len > consumed) ? (total_in_len - consumed) : 0;
    if (leftover > SPEC_SUB_IN_BUF) leftover = SPEC_SUB_IN_BUF;
    for (size_t k = 0; k < leftover; ++k) {
        size_t m = consumed + k;
        f->in_buffer[k] = (m < in_buf_len) ? f->in_buffer[m] : input[m - in_buf_len];
    }
    f->in_buf_len = leftover;
}

