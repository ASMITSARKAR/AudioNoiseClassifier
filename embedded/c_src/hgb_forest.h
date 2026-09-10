/*
 * Auto-generated C99 Decision Forest Header for Audio Noise Classifier.
 * Generated from trained scikit-learn HistGradientBoostingClassifier.
 */

#ifndef HGB_FOREST_H
#define HGB_FOREST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HGB_NUM_FEATURES   92
#define HGB_NUM_CLASSES    3
#define HGB_NUM_ITERATIONS 300

/* Single node in decision forest */
typedef struct {
    int16_t  feature_idx;
    float    threshold;
    uint16_t left;
    uint16_t right;
    float    value;
    uint8_t  is_leaf;
} HGBNode;

/* Inference routines */
void hgb_predict_raw(const float* features, float* raw_margins);
void hgb_predict_proba(const float* features, float* probabilities);
int  hgb_predict_bayes(const float* features);
uint64_t hgb_profile_traversal(const float* features, uint64_t* m4_cycles, uint64_t* m7_cycles);

#ifdef __cplusplus
}
#endif

#endif /* HGB_FOREST_H */
