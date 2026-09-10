/*
 * Embedded Noise Router Implementation (ANSI C99).
 * Zero dynamic memory allocation (no malloc/free).
 */

#include "embedded_router.h"
#include "acoustic_features.h"
#include <string.h>

void embedded_router_init(EmbeddedNoiseRouter* r) {
    spectral_sub_init(&r->spec_sub, 1.8f, 0.02f, 0.98f);
    nlms_filter_init(&r->nlms, 0.002f, 0.999f, 1e-3f);
    median_mad_init(&r->median_mad, 3.5f);
    embedded_router_reset(r);
}

void embedded_router_reset(EmbeddedNoiseRouter* r) {
    memset(r->circular_buffer, 0, sizeof(r->circular_buffer));
    r->is_primed = 0;
    r->prev_class_id = 0;
    r->last_decision = 0;
    r->last_probabilities[0] = 1.0f;
    r->last_probabilities[1] = 0.0f;
    r->last_probabilities[2] = 0.0f;
    r->prev_last_sample = 0.0f;
    r->has_prev_sample = 0;

    spectral_sub_reset(&r->spec_sub);
    nlms_filter_reset(&r->nlms);
    median_mad_reset(&r->median_mad);
}

int embedded_router_process_chunk(
    EmbeddedNoiseRouter* r,
    const float* in_chunk,
    const float* feature_vec,
    float* out_clean_chunk,
    size_t n_samples
) {
    if (n_samples == 0) {
        return r->last_decision;
    }

    /* Prime rolling buffer on first arrival */
    if (!r->is_primed) {
        size_t written = 0;
        while (written < ROUTER_WINDOW_SAMPLES) {
            size_t to_copy = (ROUTER_WINDOW_SAMPLES - written < n_samples) ? (ROUTER_WINDOW_SAMPLES - written) : n_samples;
            memcpy(&r->circular_buffer[written], in_chunk, to_copy * sizeof(float));
            written += to_copy;
        }
        r->is_primed = 1;
    } else {
        /* Shift rolling buffer left and append new chunk */
        if (n_samples >= ROUTER_WINDOW_SAMPLES) {
            memcpy(r->circular_buffer, &in_chunk[n_samples - ROUTER_WINDOW_SAMPLES], ROUTER_WINDOW_SAMPLES * sizeof(float));
        } else {
            memmove(r->circular_buffer, &r->circular_buffer[n_samples], (ROUTER_WINDOW_SAMPLES - n_samples) * sizeof(float));
            memcpy(&r->circular_buffer[ROUTER_WINDOW_SAMPLES - n_samples], in_chunk, n_samples * sizeof(float));
        }
    }

    /* Extract features from 1.0-second circular buffer if not precomputed by caller */
    float local_features[AF_NUM_FEATURES];
    const float* active_feats = feature_vec;
    if (active_feats == NULL) {
        extract_acoustic_features_c(r->circular_buffer, local_features);
        active_feats = local_features;
    }

    /* Compute Bayes risk-optimal decision from ensemble */
    hgb_predict_proba(active_feats, r->last_probabilities);
    int class_id = hgb_predict_bayes(active_feats);
    r->last_decision = class_id;

    /* Route chunk to the appropriate suppression filter */
    switch (class_id) {
        case 0:
            spectral_sub_process(&r->spec_sub, in_chunk, out_clean_chunk, n_samples);
            break;
        case 1:
            nlms_filter_process(&r->nlms, in_chunk, out_clean_chunk, n_samples);
            break;
        case 2:
        default:
            median_mad_process(&r->median_mad, in_chunk, out_clean_chunk, n_samples);
            break;
    }

    /* Apply quadratic boundary crossfade when switching filter regimes to avoid click artifacts */
    if (class_id != r->prev_class_id && r->has_prev_sample && n_samples >= ROUTER_CROSSFADE_SAMP) {
        const size_t cf_len = ROUTER_CROSSFADE_SAMP;
        float step = r->prev_last_sample - out_clean_chunk[0];
        for (size_t i = 0; i < cf_len; ++i) {
            float fade = 1.0f - ((float)i / (float)cf_len);
            out_clean_chunk[i] += step * (fade * fade);
        }
    }

    if (n_samples > 0) {
        r->prev_last_sample = out_clean_chunk[n_samples - 1];
        r->has_prev_sample = 1;
    }

    r->prev_class_id = class_id;
    return class_id;
}

int embedded_router_process_raw_chunk(
    EmbeddedNoiseRouter* r,
    const float* in_chunk,
    float* out_clean_chunk,
    size_t n_samples
) {
    return embedded_router_process_chunk(r, in_chunk, NULL, out_clean_chunk, n_samples);
}

