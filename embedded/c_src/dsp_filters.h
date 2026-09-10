/*
 * Embedded Digital Signal Processing (DSP) Filters for Audio Noise Classifier.
 * Zero-allocation, streaming-safe C99 implementations of:
 * 1. Adaptive Normalized Least-Mean-Squares (NLMS) filter
 * 2. Sliding Median Absolute Deviation (MAD) spike filter
 * 3. Spectral Subtraction with 512-point FFT and Overlap-Add (OLA)
 */

#ifndef DSP_FILTERS_H
#define DSP_FILTERS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Adaptive NLMS Filter Parameters */
#define NLMS_ORDER         16
#define NLMS_DELAY         8
#define NLMS_BUF_LEN       (NLMS_DELAY + NLMS_ORDER + 64)

typedef struct {
    float   weights[NLMS_ORDER];
    float   buffer[NLMS_BUF_LEN];
    float   mu;
    float   leak;
    float   eps;
} NLMSFilter;

void nlms_filter_init(NLMSFilter* f, float mu, float leak, float eps);
void nlms_filter_reset(NLMSFilter* f);
void nlms_filter_process(NLMSFilter* f, const float* input, float* output, size_t n_samples);

/* Median/MAD Spike Filter Parameters */
#define MEDIAN_KERNEL_SIZE 15

typedef struct {
    float threshold_mad;
} MedianMADFilter;

void median_mad_init(MedianMADFilter* f, float threshold_mad);
void median_mad_reset(MedianMADFilter* f);
void median_mad_process(MedianMADFilter* f, const float* input, float* output, size_t n_samples);

/* Spectral Subtraction Parameters */
#define SPEC_SUB_N_FFT  512
#define SPEC_SUB_HOP    256
#define SPEC_SUB_BINS   (SPEC_SUB_N_FFT / 2 + 1)
#define SPEC_SUB_IN_BUF SPEC_SUB_N_FFT

typedef struct {
    float noise_psd[SPEC_SUB_BINS];
    float in_buffer[SPEC_SUB_IN_BUF];
    float ola_tail[SPEC_SUB_N_FFT];
    size_t in_buf_len;
    float alpha;
    float beta;
    float smoothing;
    uint8_t is_initialized;
} SpectralSubFilter;

void spectral_sub_init(SpectralSubFilter* f, float alpha, float beta, float smoothing);
void spectral_sub_reset(SpectralSubFilter* f);
void spectral_sub_process(SpectralSubFilter* f, const float* input, float* output, size_t n_samples);

#ifdef __cplusplus
}
#endif

#endif /* DSP_FILTERS_H */

