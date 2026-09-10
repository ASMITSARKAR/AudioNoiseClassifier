/*
 * 92-Dimensional Acoustic Feature Extraction Implementation (ANSI C99).
 * Zero-heap allocation, deterministic memory layout for embedded systems.
 */

#include "acoustic_features.h"
#include "acoustic_tables.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// 1024-point radix-2 complex FFT
static void bit_reverse_1024(float* real, float* imag) {
    int j = 0;
    for (int i = 0; i < 1024 - 1; ++i) {
        if (i < j) {
            float tr = real[i]; real[i] = real[j]; real[j] = tr;
            float ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
        int k = 512;
        while (k >= 1 && k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }
}

static void fft_1024(float* real, float* imag) {
    bit_reverse_1024(real, imag);

    for (int len = 2; len <= 1024; len <<= 1) {
        float angle = -2.0f * (float)M_PI / (float)len;
        float wlen_r = cosf(angle);
        float wlen_i = sinf(angle);

        for (int i = 0; i < 1024; i += len) {
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
}

// Float comparator for qsort
static int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

// Main re-entrant feature extraction entrypoint
void extract_acoustic_features_c_ws(
    const float* audio,
    float* out_feats,
    AcousticFeatureWorkspace* ws
) {
    memset(out_feats, 0, AF_NUM_FEATURES * sizeof(float));

    // -------------------------------------------------------------
    // 1. Time-Domain / Envelope Features (42 non-padded frames of 1024, hop 512)
    // -------------------------------------------------------------
    float sum_sq = 0.0f;
    float peak_val = 0.0f;
    for (int i = 0; i < AF_WINDOW_SAMPLES; ++i) {
        float abs_v = fabsf(audio[i]);
        if (abs_v > peak_val) peak_val = abs_v;
        sum_sq += audio[i] * audio[i];
    }
    float global_rms = sqrtf(sum_sq / (float)AF_WINDOW_SAMPLES + 1e-10f);
    float crest_factor = peak_val / (global_rms + 1e-10f);

    float frame_rms[AF_NUM_TIME_FRAMES];
    float sum_frame_rms = 0.0f;
    float max_frame_rms = 0.0f;
    int max_frame_idx = 0;

    for (int f = 0; f < AF_NUM_TIME_FRAMES; ++f) {
        int start = f * AF_HOP_SIZE;
        float f_sum_sq = 0.0f;
        for (int n = 0; n < AF_FRAME_SIZE; ++n) {
            float s = audio[start + n];
            f_sum_sq += s * s;
        }
        float r = sqrtf(f_sum_sq / (float)AF_FRAME_SIZE + 1e-10f);
        frame_rms[f] = r;
        sum_frame_rms += r;
        if (r > max_frame_rms) {
            max_frame_rms = r;
            max_frame_idx = f;
        }
    }

    float mean_rms = sum_frame_rms / (float)AF_NUM_TIME_FRAMES;
    float sum_rms_var = 0.0f;
    float sum_skew = 0.0f;
    float sum_kurt = 0.0f;

    for (int f = 0; f < AF_NUM_TIME_FRAMES; ++f) {
        float diff = frame_rms[f] - mean_rms;
        sum_rms_var += diff * diff;
        sum_skew += diff * diff * diff;
        sum_kurt += diff * diff * diff * diff;
    }
    float std_rms = sqrtf(sum_rms_var / (float)AF_NUM_TIME_FRAMES + 1e-10f);
    float cov_rms = std_rms / (mean_rms + 1e-10f);
    float peak_to_mean = max_frame_rms / (mean_rms + 1e-10f);
    float skewness = (std_rms < 1e-6f) ? 0.0f : (sum_skew / ((float)AF_NUM_TIME_FRAMES * std_rms * std_rms * std_rms));
    float kurtosis = (std_rms < 1e-6f) ? 0.0f : ((sum_kurt / ((float)AF_NUM_TIME_FRAMES * std_rms * std_rms * std_rms * std_rms)) - 3.0f);

    float sorted_rms[AF_NUM_TIME_FRAMES];
    memcpy(sorted_rms, frame_rms, sizeof(sorted_rms));
    qsort(sorted_rms, AF_NUM_TIME_FRAMES, sizeof(float), compare_floats);

    int top_idx = (int)(0.9f * (float)AF_NUM_TIME_FRAMES);
    float top_energy = 0.0f, total_energy = 0.0f;
    for (int f = 0; f < AF_NUM_TIME_FRAMES; ++f) {
        float e = sorted_rms[f] * sorted_rms[f];
        total_energy += e;
        if (f >= top_idx) top_energy += e;
    }
    float top10_share = top_energy / (total_energy + 1e-10f);

    // Attack / decay dynamics
    float onset_thresh = 0.1f * max_frame_rms;
    int onset_idx = 0;
    for (int f = max_frame_idx; f >= 0; --f) {
        if (frame_rms[f] <= onset_thresh) {
            onset_idx = f;
            break;
        }
    }
    float attack_time_sec = (float)(max_frame_idx - onset_idx) * (512.0f / 22050.0f);
    float attack_slope = (attack_time_sec < (512.0f / 22050.0f)) ? 0.0f : ((max_frame_rms - onset_thresh) / attack_time_sec);

    float decay_thresh = 0.3f * max_frame_rms;
    int decay_idx = AF_NUM_TIME_FRAMES - 1 - max_frame_idx;
    for (int f = max_frame_idx; f < AF_NUM_TIME_FRAMES; ++f) {
        if (frame_rms[f] <= decay_thresh) {
            decay_idx = f - max_frame_idx;
            break;
        }
    }
    float decay_time_sec = (float)decay_idx * (512.0f / 22050.0f);

    // Zero-crossing rate statistics (42 frames)
    float frame_zcr[AF_NUM_TIME_FRAMES];
    float sum_zcr = 0.0f;
    float max_zcr = 0.0f;

    for (int f = 0; f < AF_NUM_TIME_FRAMES; ++f) {
        int start = f * AF_HOP_SIZE;
        int z_count = 0;
        float prev_sign = (audio[start] >= 0.0f) ? 1.0f : -1.0f;
        for (int n = 1; n < AF_FRAME_SIZE; ++n) {
            float s = audio[start + n];
            float curr_sign = (s >= 0.0f) ? 1.0f : -1.0f;
            if (curr_sign != prev_sign) {
                z_count++;
            }
            prev_sign = curr_sign;
        }
        float z = (float)z_count / (float)(AF_FRAME_SIZE - 1);
        frame_zcr[f] = z;
        sum_zcr += z;
        if (z > max_zcr) max_zcr = z;
    }
    float mean_zcr = sum_zcr / (float)AF_NUM_TIME_FRAMES;
    float sum_zcr_var = 0.0f;
    for (int f = 0; f < AF_NUM_TIME_FRAMES; ++f) {
        float d = frame_zcr[f] - mean_zcr;
        sum_zcr_var += d * d;
    }
    float std_zcr = sqrtf(sum_zcr_var / (float)AF_NUM_TIME_FRAMES);

    // -------------------------------------------------------------
    // 2. Spectral Analysis (44 reflect-padded STFT frames)
    // -------------------------------------------------------------
    float (*stft_mag)[AF_NUM_STFT_FRAMES] = ws->stft_mag;
    float (*mel_power_matrix)[AF_NUM_STFT_FRAMES] = ws->mel_power_matrix;
    float (*log_mel_matrix)[AF_NUM_STFT_FRAMES] = ws->log_mel_matrix;
    float (*mfcc_matrix)[AF_NUM_STFT_FRAMES] = ws->mfcc_matrix;
    float frame_centroids[AF_NUM_STFT_FRAMES];
    float frame_bandwidths[AF_NUM_STFT_FRAMES];
    float frame_rolloff85[AF_NUM_STFT_FRAMES];
    float frame_rolloff95[AF_NUM_STFT_FRAMES];
    float frame_flatness[AF_NUM_STFT_FRAMES];

    // Subband power accumulators
    float mean_power_bins[AF_NUM_BINS];
    memset(mean_power_bins, 0, sizeof(mean_power_bins));

    for (int f = 0; f < AF_NUM_STFT_FRAMES; ++f) {
        int center_sample = f * AF_HOP_SIZE;
        float real[AF_N_FFT];
        float imag[AF_N_FFT];

        // Reflect padding: sample at index idx
        for (int n = 0; n < AF_N_FFT; ++n) {
            int idx = center_sample - 512 + n;
            if (idx < 0) idx = -idx;
            else if (idx >= AF_WINDOW_SAMPLES) idx = 2 * AF_WINDOW_SAMPLES - 2 - idx;
            if (idx < 0) idx = 0;
            if (idx >= AF_WINDOW_SAMPLES) idx = AF_WINDOW_SAMPLES - 1;

            real[n] = audio[idx] * AF_HANN_WINDOW[n];
            imag[n] = 0.0f;
        }

        fft_1024(real, imag);

        float total_mag = 1e-10f;
        float num_cent = 0.0f;
        float log_sum = 0.0f;
        float power_bins[AF_NUM_BINS];

        for (int k = 0; k < AF_NUM_BINS; ++k) {
            float mag_k = sqrtf(real[k] * real[k] + imag[k] * imag[k]);
            stft_mag[k][f] = mag_k;
            total_mag += mag_k;
            num_cent += AF_FFT_FREQS[k] * mag_k;
            log_sum += logf(mag_k + 1e-10f);

            float pwr = mag_k * mag_k;
            power_bins[k] = pwr;
            mean_power_bins[k] += pwr;
        }

        if (total_mag < 1e-6f) {
            frame_centroids[f] = 0.0f;
            frame_bandwidths[f] = 0.0f;
            frame_rolloff85[f] = 0.0f;
            frame_rolloff95[f] = 0.0f;
            frame_flatness[f] = 1.0f;
        } else {
            // Centroid & Bandwidth
            float cent = num_cent / total_mag;
            frame_centroids[f] = cent;

            float num_bw = 0.0f;
            for (int k = 0; k < AF_NUM_BINS; ++k) {
                float diff = AF_FFT_FREQS[k] - cent;
                num_bw += (diff * diff) * stft_mag[k][f];
            }
            frame_bandwidths[f] = sqrtf(num_bw / total_mag);

            // Rolloff 85% and 95%
            float cum_mag = 0.0f;
            float target_85 = 0.85f * total_mag;
            float target_95 = 0.95f * total_mag;
            float r85 = AF_FFT_FREQS[AF_NUM_BINS - 1];
            float r95 = AF_FFT_FREQS[AF_NUM_BINS - 1];
            int set_85 = 0, set_95 = 0;

            for (int k = 0; k < AF_NUM_BINS; ++k) {
                cum_mag += stft_mag[k][f];
                if (!set_85 && cum_mag >= target_85) {
                    r85 = AF_FFT_FREQS[k];
                    set_85 = 1;
                }
                if (!set_95 && cum_mag >= target_95) {
                    r95 = AF_FFT_FREQS[k];
                    set_95 = 1;
                }
            }
            frame_rolloff85[f] = r85;
            frame_rolloff95[f] = r95;
            // Spectral Flatness
            float geom_mean = expf(log_sum / (float)AF_NUM_BINS);
            float arith_mean = total_mag / (float)AF_NUM_BINS;
            frame_flatness[f] = geom_mean / (arith_mean + 1e-10f);
        }

        // Mel Filterbank
        for (int m = 0; m < AF_NUM_MELS; ++m) {
            uint16_t start = AF_MEL_RANGES[m].start_bin;
            uint16_t count = AF_MEL_RANGES[m].num_bins;
            uint16_t offset = AF_MEL_RANGES[m].weight_offset;
            float mel_energy = 0.0f;
            for (int k = 0; k < count; ++k) {
                mel_energy += AF_MEL_WEIGHTS[offset + k] * power_bins[start + k];
            }
            mel_power_matrix[m][f] = mel_energy;
        }
    }

    // Convert Mel Power to dB with global top_db = 80.0 clamping (matching librosa.power_to_db)
    float max_log_mel = -1000.0f;
    for (int m = 0; m < AF_NUM_MELS; ++m) {
        for (int f = 0; f < AF_NUM_STFT_FRAMES; ++f) {
            float e = mel_power_matrix[m][f];
            float val = (e <= 1e-10f) ? -100.0f : (10.0f * log10f(e));
            log_mel_matrix[m][f] = val;
            if (val > max_log_mel) max_log_mel = val;
        }
    }
    float min_floor = max_log_mel - 80.0f;
    for (int m = 0; m < AF_NUM_MELS; ++m) {
        for (int f = 0; f < AF_NUM_STFT_FRAMES; ++f) {
            if (log_mel_matrix[m][f] < min_floor) {
                log_mel_matrix[m][f] = min_floor;
            }
        }
    }

    // DCT-II for 20 MFCCs across all 44 frames
    for (int i = 0; i < AF_NUM_MFCC; ++i) {
        for (int f = 0; f < AF_NUM_STFT_FRAMES; ++f) {
            float coef = 0.0f;
            for (int m = 0; m < AF_NUM_MELS; ++m) {
                coef += AF_DCT_MATRIX[i][m] * log_mel_matrix[m][f];
            }
            ws->mfcc_matrix[i][f] = coef;
        }
    }

    // Subband Energy Ratios
    float total_subband_pwr = 1e-10f;
    float pwr_low = 0.0f, pwr_mid = 0.0f, pwr_high = 0.0f;
    for (int k = 0; k < AF_NUM_BINS; ++k) {
        float p = mean_power_bins[k] / (float)AF_NUM_STFT_FRAMES;
        total_subband_pwr += p;
        float freq = AF_FFT_FREQS[k];
        if (freq >= 20.0f && freq < 500.0f) pwr_low += p;
        else if (freq >= 500.0f && freq < 3500.0f) pwr_mid += p;
        else if (freq >= 3500.0f) pwr_high += p;
    }
    float energy_ratio_low = pwr_low / total_subband_pwr;
    float energy_ratio_mid = pwr_mid / total_subband_pwr;
    float energy_ratio_high = pwr_high / total_subband_pwr;

    // Spectral Flux across 44 STFT frames (43 diffs)
    float flux[AF_NUM_STFT_FRAMES - 1];
    float sum_flux = 0.0f;
    float max_flux = 0.0f;

    for (int f = 0; f < AF_NUM_STFT_FRAMES - 1; ++f) {
        float norm_curr = 1e-10f, norm_next = 1e-10f;
        for (int k = 0; k < AF_NUM_BINS; ++k) {
            norm_curr += stft_mag[k][f] * stft_mag[k][f];
            norm_next += stft_mag[k][f + 1] * stft_mag[k][f + 1];
        }
        norm_curr = sqrtf(norm_curr) + 1e-10f;
        norm_next = sqrtf(norm_next) + 1e-10f;

        float d_sum = 0.0f;
        for (int k = 0; k < AF_NUM_BINS; ++k) {
            float d = (stft_mag[k][f + 1] / norm_next) - (stft_mag[k][f] / norm_curr);
            d_sum += d * d;
        }
        float val = sqrtf(d_sum + 1e-10f);
        flux[f] = val;
        sum_flux += val;
        if (val > max_flux) max_flux = val;
    }
    float mean_flux = sum_flux / (float)(AF_NUM_STFT_FRAMES - 1);
    float sum_flux_var = 0.0f;
    for (int f = 0; f < AF_NUM_STFT_FRAMES - 1; ++f) {
        float d = flux[f] - mean_flux;
        sum_flux_var += d * d;
    }
    float std_flux = sqrtf(sum_flux_var / (float)(AF_NUM_STFT_FRAMES - 1));

    float sorted_flux[AF_NUM_STFT_FRAMES - 1];
    memcpy(sorted_flux, flux, sizeof(sorted_flux));
    qsort(sorted_flux, AF_NUM_STFT_FRAMES - 1, sizeof(float), compare_floats);
    int p90_idx = (int)(0.9f * (float)(AF_NUM_STFT_FRAMES - 1));
    if (p90_idx >= AF_NUM_STFT_FRAMES - 1) p90_idx = AF_NUM_STFT_FRAMES - 2;
    float p90_flux = sorted_flux[p90_idx];

    // Spectral Shape Summary Stats
    float sum_cent = 0.0f, sum_bw = 0.0f, sum_r85 = 0.0f, sum_r95 = 0.0f, sum_flat = 0.0f;
    float max_flat = 0.0f;
    for (int f = 0; f < AF_NUM_STFT_FRAMES; ++f) {
        sum_cent += frame_centroids[f];
        sum_bw += frame_bandwidths[f];
        sum_r85 += frame_rolloff85[f];
        sum_r95 += frame_rolloff95[f];
        sum_flat += frame_flatness[f];
        if (frame_flatness[f] > max_flat) max_flat = frame_flatness[f];
    }
    float mean_cent = sum_cent / (float)AF_NUM_STFT_FRAMES;
    float mean_bw = sum_bw / (float)AF_NUM_STFT_FRAMES;
    float mean_r85 = sum_r85 / (float)AF_NUM_STFT_FRAMES;
    float mean_r95 = sum_r95 / (float)AF_NUM_STFT_FRAMES;
    float mean_flat = sum_flat / (float)AF_NUM_STFT_FRAMES;

    float sum_cent_var = 0.0f, sum_bw_var = 0.0f, sum_r85_var = 0.0f, sum_r95_var = 0.0f, sum_flat_var = 0.0f;
    for (int f = 0; f < AF_NUM_STFT_FRAMES; ++f) {
        float dc = frame_centroids[f] - mean_cent; sum_cent_var += dc * dc;
        float db = frame_bandwidths[f] - mean_bw; sum_bw_var += db * db;
        float dr85 = frame_rolloff85[f] - mean_r85; sum_r85_var += dr85 * dr85;
        float dr95 = frame_rolloff95[f] - mean_r95; sum_r95_var += dr95 * dr95;
        float df = frame_flatness[f] - mean_flat; sum_flat_var += df * df;
    }
    float std_cent = sqrtf(sum_cent_var / (float)AF_NUM_STFT_FRAMES);
    float std_bw = sqrtf(sum_bw_var / (float)AF_NUM_STFT_FRAMES);
    float std_r85 = sqrtf(sum_r85_var / (float)AF_NUM_STFT_FRAMES);
    float std_r95 = sqrtf(sum_r95_var / (float)AF_NUM_STFT_FRAMES);
    float std_flat = sqrtf(sum_flat_var / (float)AF_NUM_STFT_FRAMES);

    // MFCC Statistics and Delta Dynamics (width 9 delta)
    float mfcc_mean[AF_NUM_MFCC];
    float mfcc_std[AF_NUM_MFCC];
    float mfcc_delta_std[AF_NUM_MFCC];

    for (int i = 0; i < AF_NUM_MFCC; ++i) {
        float sum_m = 0.0f;
        for (int f = 0; f < AF_NUM_STFT_FRAMES; ++f) {
            sum_m += mfcc_matrix[i][f];
        }
        float m_mean = sum_m / (float)AF_NUM_STFT_FRAMES;
        mfcc_mean[i] = m_mean;

        float sum_m_var = 0.0f;
        for (int f = 0; f < AF_NUM_STFT_FRAMES; ++f) {
            float d = mfcc_matrix[i][f] - m_mean;
            sum_m_var += d * d;
        }
        mfcc_std[i] = sqrtf(sum_m_var / (float)AF_NUM_STFT_FRAMES);

        // Compute librosa-compatible delta across time (order 1, width 9)
        // delta[t] = sum_{n=1..4} n * (val[t+n] - val[t-n]) / 60
        float sum_d = 0.0f;
        float delta_vals[AF_NUM_STFT_FRAMES];
        for (int t = 0; t < AF_NUM_STFT_FRAMES; ++t) {
            float num = 0.0f;
            for (int n = 1; n <= 4; ++n) {
                int t_plus = t + n;
                if (t_plus >= AF_NUM_STFT_FRAMES) t_plus = AF_NUM_STFT_FRAMES - 1;
                int t_minus = t - n;
                if (t_minus < 0) t_minus = 0;
                num += (float)n * (mfcc_matrix[i][t_plus] - mfcc_matrix[i][t_minus]);
            }
            float d = num / 60.0f;
            delta_vals[t] = d;
            sum_d += d;
        }
        float d_mean = sum_d / (float)AF_NUM_STFT_FRAMES;
        float sum_d_var = 0.0f;
        for (int t = 0; t < AF_NUM_STFT_FRAMES; ++t) {
            float d = delta_vals[t] - d_mean;
            sum_d_var += d * d;
        }
        mfcc_delta_std[i] = sqrtf(sum_d_var / (float)AF_NUM_STFT_FRAMES);
    }

    // -------------------------------------------------------------
    // 3. Assemble Output Vector in Exact Alphabetical Order (92 Features)
    // -------------------------------------------------------------
    out_feats[FEAT_ATTACK_SLOPE]            = attack_slope;
    out_feats[FEAT_ATTACK_TIME_SEC]        = attack_time_sec;
    out_feats[FEAT_CREST_FACTOR]            = crest_factor;
    out_feats[FEAT_DECAY_TIME_SEC]          = decay_time_sec;
    out_feats[FEAT_ENERGY_RATIO_HIGH]       = energy_ratio_high;
    out_feats[FEAT_ENERGY_RATIO_LOW]        = energy_ratio_low;
    out_feats[FEAT_ENERGY_RATIO_MID]        = energy_ratio_mid;

    // MFCC 10 to 19
    out_feats[FEAT_MFCC_10_MEAN]            = mfcc_mean[9];
    out_feats[FEAT_MFCC_10_STD]             = mfcc_std[9];
    out_feats[FEAT_MFCC_11_MEAN]            = mfcc_mean[10];
    out_feats[FEAT_MFCC_11_STD]             = mfcc_std[10];
    out_feats[FEAT_MFCC_12_MEAN]            = mfcc_mean[11];
    out_feats[FEAT_MFCC_12_STD]             = mfcc_std[11];
    out_feats[FEAT_MFCC_13_MEAN]            = mfcc_mean[12];
    out_feats[FEAT_MFCC_13_STD]             = mfcc_std[12];
    out_feats[FEAT_MFCC_14_MEAN]            = mfcc_mean[13];
    out_feats[FEAT_MFCC_14_STD]             = mfcc_std[13];
    out_feats[FEAT_MFCC_15_MEAN]            = mfcc_mean[14];
    out_feats[FEAT_MFCC_15_STD]             = mfcc_std[14];
    out_feats[FEAT_MFCC_16_MEAN]            = mfcc_mean[15];
    out_feats[FEAT_MFCC_16_STD]             = mfcc_std[15];
    out_feats[FEAT_MFCC_17_MEAN]            = mfcc_mean[16];
    out_feats[FEAT_MFCC_17_STD]             = mfcc_std[16];
    out_feats[FEAT_MFCC_18_MEAN]            = mfcc_mean[17];
    out_feats[FEAT_MFCC_18_STD]             = mfcc_std[17];
    out_feats[FEAT_MFCC_19_MEAN]            = mfcc_mean[18];
    out_feats[FEAT_MFCC_19_STD]             = mfcc_std[18];

    // MFCC 1
    out_feats[FEAT_MFCC_1_MEAN]             = mfcc_mean[0];
    out_feats[FEAT_MFCC_1_STD]              = mfcc_std[0];

    // MFCC 20
    out_feats[FEAT_MFCC_20_MEAN]            = mfcc_mean[19];
    out_feats[FEAT_MFCC_20_STD]             = mfcc_std[19];

    // MFCC 2 to 9
    out_feats[FEAT_MFCC_2_MEAN]             = mfcc_mean[1];
    out_feats[FEAT_MFCC_2_STD]              = mfcc_std[1];
    out_feats[FEAT_MFCC_3_MEAN]             = mfcc_mean[2];
    out_feats[FEAT_MFCC_3_STD]              = mfcc_std[2];
    out_feats[FEAT_MFCC_4_MEAN]             = mfcc_mean[3];
    out_feats[FEAT_MFCC_4_STD]              = mfcc_std[3];
    out_feats[FEAT_MFCC_5_MEAN]             = mfcc_mean[4];
    out_feats[FEAT_MFCC_5_STD]              = mfcc_std[4];
    out_feats[FEAT_MFCC_6_MEAN]             = mfcc_mean[5];
    out_feats[FEAT_MFCC_6_STD]              = mfcc_std[5];
    out_feats[FEAT_MFCC_7_MEAN]             = mfcc_mean[6];
    out_feats[FEAT_MFCC_7_STD]              = mfcc_std[6];
    out_feats[FEAT_MFCC_8_MEAN]             = mfcc_mean[7];
    out_feats[FEAT_MFCC_8_STD]              = mfcc_std[7];
    out_feats[FEAT_MFCC_9_MEAN]             = mfcc_mean[8];
    out_feats[FEAT_MFCC_9_STD]              = mfcc_std[8];

    // MFCC Delta 10 to 19
    out_feats[FEAT_MFCC_DELTA_10_STD]       = mfcc_delta_std[9];
    out_feats[FEAT_MFCC_DELTA_11_STD]       = mfcc_delta_std[10];
    out_feats[FEAT_MFCC_DELTA_12_STD]       = mfcc_delta_std[11];
    out_feats[FEAT_MFCC_DELTA_13_STD]       = mfcc_delta_std[12];
    out_feats[FEAT_MFCC_DELTA_14_STD]       = mfcc_delta_std[13];
    out_feats[FEAT_MFCC_DELTA_15_STD]       = mfcc_delta_std[14];
    out_feats[FEAT_MFCC_DELTA_16_STD]       = mfcc_delta_std[15];
    out_feats[FEAT_MFCC_DELTA_17_STD]       = mfcc_delta_std[16];
    out_feats[FEAT_MFCC_DELTA_18_STD]       = mfcc_delta_std[17];
    out_feats[FEAT_MFCC_DELTA_19_STD]       = mfcc_delta_std[18];

    // MFCC Delta 1 & 20
    out_feats[FEAT_MFCC_DELTA_1_STD]        = mfcc_delta_std[0];
    out_feats[FEAT_MFCC_DELTA_20_STD]       = mfcc_delta_std[19];

    // MFCC Delta 2 to 9
    out_feats[FEAT_MFCC_DELTA_2_STD]        = mfcc_delta_std[1];
    out_feats[FEAT_MFCC_DELTA_3_STD]        = mfcc_delta_std[2];
    out_feats[FEAT_MFCC_DELTA_4_STD]        = mfcc_delta_std[3];
    out_feats[FEAT_MFCC_DELTA_5_STD]        = mfcc_delta_std[4];
    out_feats[FEAT_MFCC_DELTA_6_STD]        = mfcc_delta_std[5];
    out_feats[FEAT_MFCC_DELTA_7_STD]        = mfcc_delta_std[6];
    out_feats[FEAT_MFCC_DELTA_8_STD]        = mfcc_delta_std[7];
    out_feats[FEAT_MFCC_DELTA_9_STD]        = mfcc_delta_std[8];

    // RMS Envelope Features
    out_feats[FEAT_RMS_COV]                 = cov_rms;
    out_feats[FEAT_RMS_KURTOSIS]            = kurtosis;
    out_feats[FEAT_RMS_MEAN]                = mean_rms;
    out_feats[FEAT_RMS_PEAK_TO_MEAN]        = peak_to_mean;
    out_feats[FEAT_RMS_SKEWNESS]            = skewness;
    out_feats[FEAT_RMS_STD]                 = std_rms;
    out_feats[FEAT_RMS_TOP10_SHARE]         = top10_share;

    // Spectral Shape Features
    out_feats[FEAT_SPECTRAL_BANDWIDTH_MEAN] = mean_bw;
    out_feats[FEAT_SPECTRAL_BANDWIDTH_STD]  = std_bw;
    out_feats[FEAT_SPECTRAL_CENTROID_MEAN]  = mean_cent;
    out_feats[FEAT_SPECTRAL_CENTROID_STD]   = std_cent;
    out_feats[FEAT_SPECTRAL_FLATNESS_MAX]   = max_flat;
    out_feats[FEAT_SPECTRAL_FLATNESS_MEAN]  = mean_flat;
    out_feats[FEAT_SPECTRAL_FLATNESS_STD]   = std_flat;
    out_feats[FEAT_SPECTRAL_FLUX_MAX]       = max_flux;
    out_feats[FEAT_SPECTRAL_FLUX_MEAN]      = mean_flux;
    out_feats[FEAT_SPECTRAL_FLUX_P90]       = p90_flux;
    out_feats[FEAT_SPECTRAL_FLUX_STD]       = std_flux;
    out_feats[FEAT_SPECTRAL_ROLLOFF_85_MEAN]= mean_r85;
    out_feats[FEAT_SPECTRAL_ROLLOFF_85_STD] = std_r85;
    out_feats[FEAT_SPECTRAL_ROLLOFF_95_MEAN]= mean_r95;
    out_feats[FEAT_SPECTRAL_ROLLOFF_95_STD] = std_r95;

    // Zero Crossing Rate Features
    out_feats[FEAT_ZCR_MAX]                 = max_zcr;
    out_feats[FEAT_ZCR_MEAN]                = mean_zcr;
    out_feats[FEAT_ZCR_STD]                 = std_zcr;
}

// Backward-compatible C feature extraction API using internal static workspace
void extract_acoustic_features_c(
    const float* audio,
    float* out_feats
) {
    static AcousticFeatureWorkspace default_ws;
    extract_acoustic_features_c_ws(audio, out_feats, &default_ws);
}
