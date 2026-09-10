/*
 * Embedded pipeline cycle profiler and host latency benchmark.
 * Measures wall-clock execution time on host and projects Cortex-M cycle requirements.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "../c_src/hgb_forest.h"
#include "../c_src/dsp_filters.h"
#include "../c_src/embedded_router.h"
#include "../c_src/acoustic_features.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int main(void) {
    printf("--- Embedded Pipeline Profiler & Benchmark ---\n\n");

    /* Generate synthetic test signal: 440 Hz fundamental + 1200 Hz harmonic */
    float audio_window[AF_WINDOW_SAMPLES];
    for (int i = 0; i < AF_WINDOW_SAMPLES; ++i) {
        audio_window[i] = (float)(0.2 * sin(2.0 * M_PI * 440.0 * (double)i / 22050.0) +
                                  0.05 * sin(2.0 * M_PI * 1200.0 * (double)i / 22050.0));
    }

    float feats[AF_NUM_FEATURES];
    const int N_REPS = 50;

    /* 1. Feature Extraction Benchmark */
    clock_t t0 = clock();
    for (int r = 0; r < N_REPS; ++r) {
        extract_acoustic_features_c(audio_window, feats);
    }
    clock_t t1 = clock();
    double feat_ms = (((double)(t1 - t0) / (double)CLOCKS_PER_SEC) * 1000.0) / (double)N_REPS;

    /* 2. Forest Classification Benchmark */
    clock_t t2 = clock();
    for (int r = 0; r < N_REPS * 10; ++r) {
        int bayes_dec = hgb_predict_bayes(feats);
        (void)bayes_dec;
    }
    clock_t t3 = clock();
    double tree_ms = (((double)(t3 - t2) / (double)CLOCKS_PER_SEC) * 1000.0) / (double)(N_REPS * 10);

    /* 3. Full End-to-End Pipeline */
    EmbeddedNoiseRouter router;
    embedded_router_init(&router);
    float chunk_out[ROUTER_CHUNK_SAMPLES];

    clock_t t4 = clock();
    for (int r = 0; r < N_REPS; ++r) {
        embedded_router_process_raw_chunk(&router, audio_window, chunk_out, ROUTER_CHUNK_SAMPLES);
    }
    clock_t t5 = clock();
    double total_e2e_ms = (((double)(t5 - t4) / (double)CLOCKS_PER_SEC) * 1000.0) / (double)N_REPS;

    printf("1. Measured Host Latency (Single-Threaded):\n");
    printf("   - 92-Feature Acoustic Extraction : %6.2f ms / 1.0s window\n", feat_ms);
    printf("   - 900-Tree Decision Inference    : %6.3f ms (%6.1f us)\n", tree_ms, tree_ms * 1000.0);
    printf("   - End-to-End Autonomous Pipeline : %6.2f ms / 500ms chunk (%.1f%% real-time headroom)\n",
           total_e2e_ms, ((500.0 - total_e2e_ms) / 500.0) * 100.0);
    printf("-----------------------------------------------------------------\n");

    /* 4. Tree Traversal Depth & Projected Microcontroller Timing */
    uint64_t m4_cycles = 0;
    uint64_t m7_cycles = 0;
    uint64_t nodes_visited = hgb_profile_traversal(feats, &m4_cycles, &m7_cycles);

    printf("2. Decision Tree Traversal Complexity (900 Trees):\n");
    printf("   - Total Nodes Visited in Chunk  : %llu nodes (avg %.2f nodes/tree)\n",
           (unsigned long long)nodes_visited, (double)nodes_visited / 900.0);
    printf("   - Projected Cortex-M4 Cycles    : %llu cycles (~18 cyc/node Flash access)\n", (unsigned long long)m4_cycles);
    printf("   - Projected Cortex-M7 Cycles    : %llu cycles (~5 cyc/node with I-Cache)\n", (unsigned long long)m7_cycles);
    printf("-----------------------------------------------------------------\n");

    double m4_latency_ms = ((double)m4_cycles / 168000000.0) * 1000.0;
    double m7_latency_ms = ((double)m7_cycles / 480000000.0) * 1000.0;

    printf("3. Projected Microcontroller Inference Latency:\n");
    printf("   - STM32F407 (Cortex-M4 @ 168 MHz) : %.4f ms (%6.2f us)\n", m4_latency_ms, m4_latency_ms * 1000.0);
    printf("   - STM32H743 (Cortex-M7 @ 480 MHz) : %.4f ms (%6.2f us)\n", m7_latency_ms, m7_latency_ms * 1000.0);
    printf("=================================================================\n");
    return 0;
}

