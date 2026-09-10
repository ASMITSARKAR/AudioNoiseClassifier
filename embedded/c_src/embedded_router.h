/*
 * Embedded Noise Router Header (ANSI C99).
 * Manages rolling audio analysis buffer, feature extraction, Bayes risk classification,
 * and adaptive filter switching with crossfade boundary smoothing.
 */

#ifndef EMBEDDED_ROUTER_H
#define EMBEDDED_ROUTER_H

#include <stdint.h>
#include <stddef.h>
#include "hgb_forest.h"
#include "dsp_filters.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROUTER_SR             22050
#define ROUTER_CHUNK_SAMPLES  11025
#define ROUTER_WINDOW_SAMPLES 22050
#define ROUTER_CROSSFADE_SAMP 330

typedef struct {
    float   circular_buffer[ROUTER_WINDOW_SAMPLES];
    uint8_t is_primed;
    int     prev_class_id;
    float   prev_last_sample;
    uint8_t has_prev_sample;

    SpectralSubFilter spec_sub;
    NLMSFilter        nlms;
    MedianMADFilter   median_mad;

    int   last_decision;
    float last_probabilities[3];
} EmbeddedNoiseRouter;

void embedded_router_init(EmbeddedNoiseRouter* r);
void embedded_router_reset(EmbeddedNoiseRouter* r);

/* Process chunk with caller-supplied feature vector (if NULL, extracts from circular buffer) */
int embedded_router_process_chunk(
    EmbeddedNoiseRouter* r,
    const float* in_chunk,
    const float* feature_vec,
    float* out_clean_chunk,
    size_t n_samples
);

/* Autonomous streaming entrypoint: updates buffer, extracts features, classifies, and filters chunk */
int embedded_router_process_raw_chunk(
    EmbeddedNoiseRouter* r,
    const float* in_chunk,
    float* out_clean_chunk,
    size_t n_samples
);

#ifdef __cplusplus
}
#endif

#endif /* EMBEDDED_ROUTER_H */
