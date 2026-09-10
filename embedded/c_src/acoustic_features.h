/*
 * 92-Dimensional Acoustic Feature Extraction Pipeline (Embedded ANSI C99).
 * Computes exact mathematical parity with Librosa/SciPy feature extractors:
 * - STFT spectral moments (Centroid, Bandwidth, Rolloff, Flatness, Flux, Subbands)
 * - Mel filterbank and 20 MFCCs + first-order delta dynamics
 * - Time-domain RMS kinetic envelope moments and Zero Crossing Rate (ZCR)
 */

#ifndef ACOUSTIC_FEATURES_H
#define ACOUSTIC_FEATURES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AF_NUM_FEATURES    92
#define AF_WINDOW_SAMPLES  22050
#define AF_N_FFT           1024
#define AF_FRAME_SIZE      1024
#define AF_HOP_SIZE        512
#define AF_NUM_STFT_FRAMES 44
#define AF_NUM_TIME_FRAMES 42
#define AF_NUM_BINS        513
#define AF_NUM_MFCC        20
#define AF_NUM_MELS        128

/* Feature index mapping matching Python AudioFeatureExtractor ordering */
typedef enum {

    FEAT_ATTACK_SLOPE = 0,
    FEAT_ATTACK_TIME_SEC = 1,
    FEAT_CREST_FACTOR = 2,
    FEAT_DECAY_TIME_SEC = 3,
    FEAT_ENERGY_RATIO_HIGH = 4,
    FEAT_ENERGY_RATIO_LOW = 5,
    FEAT_ENERGY_RATIO_MID = 6,
    FEAT_MFCC_10_MEAN = 7,
    FEAT_MFCC_10_STD = 8,
    FEAT_MFCC_11_MEAN = 9,
    FEAT_MFCC_11_STD = 10,
    FEAT_MFCC_12_MEAN = 11,
    FEAT_MFCC_12_STD = 12,
    FEAT_MFCC_13_MEAN = 13,
    FEAT_MFCC_13_STD = 14,
    FEAT_MFCC_14_MEAN = 15,
    FEAT_MFCC_14_STD = 16,
    FEAT_MFCC_15_MEAN = 17,
    FEAT_MFCC_15_STD = 18,
    FEAT_MFCC_16_MEAN = 19,
    FEAT_MFCC_16_STD = 20,
    FEAT_MFCC_17_MEAN = 21,
    FEAT_MFCC_17_STD = 22,
    FEAT_MFCC_18_MEAN = 23,
    FEAT_MFCC_18_STD = 24,
    FEAT_MFCC_19_MEAN = 25,
    FEAT_MFCC_19_STD = 26,
    FEAT_MFCC_1_MEAN = 27,
    FEAT_MFCC_1_STD = 28,
    FEAT_MFCC_20_MEAN = 29,
    FEAT_MFCC_20_STD = 30,
    FEAT_MFCC_2_MEAN = 31,
    FEAT_MFCC_2_STD = 32,
    FEAT_MFCC_3_MEAN = 33,
    FEAT_MFCC_3_STD = 34,
    FEAT_MFCC_4_MEAN = 35,
    FEAT_MFCC_4_STD = 36,
    FEAT_MFCC_5_MEAN = 37,
    FEAT_MFCC_5_STD = 38,
    FEAT_MFCC_6_MEAN = 39,
    FEAT_MFCC_6_STD = 40,
    FEAT_MFCC_7_MEAN = 41,
    FEAT_MFCC_7_STD = 42,
    FEAT_MFCC_8_MEAN = 43,
    FEAT_MFCC_8_STD = 44,
    FEAT_MFCC_9_MEAN = 45,
    FEAT_MFCC_9_STD = 46,
    FEAT_MFCC_DELTA_10_STD = 47,
    FEAT_MFCC_DELTA_11_STD = 48,
    FEAT_MFCC_DELTA_12_STD = 49,
    FEAT_MFCC_DELTA_13_STD = 50,
    FEAT_MFCC_DELTA_14_STD = 51,
    FEAT_MFCC_DELTA_15_STD = 52,
    FEAT_MFCC_DELTA_16_STD = 53,
    FEAT_MFCC_DELTA_17_STD = 54,
    FEAT_MFCC_DELTA_18_STD = 55,
    FEAT_MFCC_DELTA_19_STD = 56,
    FEAT_MFCC_DELTA_1_STD = 57,
    FEAT_MFCC_DELTA_20_STD = 58,
    FEAT_MFCC_DELTA_2_STD = 59,
    FEAT_MFCC_DELTA_3_STD = 60,
    FEAT_MFCC_DELTA_4_STD = 61,
    FEAT_MFCC_DELTA_5_STD = 62,
    FEAT_MFCC_DELTA_6_STD = 63,
    FEAT_MFCC_DELTA_7_STD = 64,
    FEAT_MFCC_DELTA_8_STD = 65,
    FEAT_MFCC_DELTA_9_STD = 66,
    FEAT_RMS_COV = 67,
    FEAT_RMS_KURTOSIS = 68,
    FEAT_RMS_MEAN = 69,
    FEAT_RMS_PEAK_TO_MEAN = 70,
    FEAT_RMS_SKEWNESS = 71,
    FEAT_RMS_STD = 72,
    FEAT_RMS_TOP10_SHARE = 73,
    FEAT_SPECTRAL_BANDWIDTH_MEAN = 74,
    FEAT_SPECTRAL_BANDWIDTH_STD = 75,
    FEAT_SPECTRAL_CENTROID_MEAN = 76,
    FEAT_SPECTRAL_CENTROID_STD = 77,
    FEAT_SPECTRAL_FLATNESS_MAX = 78,
    FEAT_SPECTRAL_FLATNESS_MEAN = 79,
    FEAT_SPECTRAL_FLATNESS_STD = 80,
    FEAT_SPECTRAL_FLUX_MAX = 81,
    FEAT_SPECTRAL_FLUX_MEAN = 82,
    FEAT_SPECTRAL_FLUX_P90 = 83,
    FEAT_SPECTRAL_FLUX_STD = 84,
    FEAT_SPECTRAL_ROLLOFF_85_MEAN = 85,
    FEAT_SPECTRAL_ROLLOFF_85_STD = 86,
    FEAT_SPECTRAL_ROLLOFF_95_MEAN = 87,
    FEAT_SPECTRAL_ROLLOFF_95_STD = 88,
    FEAT_ZCR_MAX = 89,
    FEAT_ZCR_MEAN = 90,
    FEAT_ZCR_STD = 91
} AcousticFeatureIndex;

/* Re-entrant workspace for intermediate STFT and Mel matrices (~138 KB) */
typedef struct {
    float stft_mag[AF_NUM_BINS][AF_NUM_STFT_FRAMES];
    float mel_power_matrix[AF_NUM_MELS][AF_NUM_STFT_FRAMES];
    float log_mel_matrix[AF_NUM_MELS][AF_NUM_STFT_FRAMES];
    float mfcc_matrix[AF_NUM_MFCC][AF_NUM_STFT_FRAMES];
} AcousticFeatureWorkspace;

/* Thread-safe extraction API with caller-provided workspace */
void extract_acoustic_features_c_ws(
    const float* audio_window_22050,
    float* out_features_92,
    AcousticFeatureWorkspace* ws
);

/* Primary feature extraction entrypoint using static/stack buffer */
void extract_acoustic_features_c(
    const float* audio_window_22050,
    float* out_features_92
);

#ifdef __cplusplus
}
#endif

#endif /* ACOUSTIC_FEATURES_H */

