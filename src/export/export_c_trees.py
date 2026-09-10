import pickle
import sys
from pathlib import Path

sys.path.insert(0, '.')
from src.models.cost_sensitive import COST_MATRIX

MODEL_PKL_PATH = Path('models/best_baseline.pkl')
C_SRC_DIR = Path('embedded/c_src')


def transpile_hgb_to_c():
    print(f"Loading {MODEL_PKL_PATH}...")
    with open(MODEL_PKL_PATH, 'rb') as f:
        ckpt = pickle.load(f)
        raw_hgb = ckpt['raw_hgb']
        feat_names = ckpt['feature_names']

    C_SRC_DIR.mkdir(parents=True, exist_ok=True)
    n_features = len(feat_names)
    n_iterations = len(raw_hgb._predictors)
    n_classes = len(raw_hgb._predictors[0])
    base_pred = raw_hgb._baseline_prediction[0]

    total_nodes = sum(len(tree.nodes) for iter_trees in raw_hgb._predictors for tree in iter_trees)
    print(f"Exporting C forest: {n_iterations} iters, {n_classes} classes, {n_features} feats ({total_nodes:,} nodes)")

    # Header
    header_content = f"""#ifndef HGB_FOREST_H
#define HGB_FOREST_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define HGB_NUM_FEATURES   {n_features}
#define HGB_NUM_CLASSES    {n_classes}
#define HGB_NUM_ITERATIONS {n_iterations}

typedef struct {{
    int16_t  feature_idx;
    float    threshold;
    uint16_t left;
    uint16_t right;
    float    value;
    uint8_t  is_leaf;
}} HGBNode;

void hgb_predict_raw(const float* features, float* raw_margins);
void hgb_predict_proba(const float* features, float* probabilities);
int  hgb_predict_bayes(const float* features);
uint64_t hgb_profile_traversal(const float* features, uint64_t* m4_cycles, uint64_t* m7_cycles);

#ifdef __cplusplus
}}
#endif

#endif
"""
    with open(C_SRC_DIR / 'hgb_forest.h', 'w', encoding='utf-8') as f:
        f.write(header_content)

    # C implementation
    c_lines = [
        '#include "hgb_forest.h"',
        '#include <math.h>',
        '',
        f'static const float BASELINE_PREDICTION[{n_classes}] = {{ {base_pred[0]:.8f}f, {base_pred[1]:.8f}f, {base_pred[2]:.8f}f }};',
        '',
        '/* Bayes risk cost matrix */',
        'static const float COST_MATRIX[3][3] = {',
        f'    {{ {COST_MATRIX[0][0]:.2f}f, {COST_MATRIX[0][1]:.2f}f, {COST_MATRIX[0][2]:.2f}f }},',
        f'    {{ {COST_MATRIX[1][0]:.2f}f, {COST_MATRIX[1][1]:.2f}f, {COST_MATRIX[1][2]:.2f}f }},',
        f'    {{ {COST_MATRIX[2][0]:.2f}f, {COST_MATRIX[2][1]:.2f}f, {COST_MATRIX[2][2]:.2f}f }}',
        '};',
        '',
        f'static const HGBNode ALL_NODES[{total_nodes}] = {{'
    ]

    node_idx = 0
    tree_offsets = []
    for iter_idx, iter_trees in enumerate(raw_hgb._predictors):
        for class_idx, tree in enumerate(iter_trees):
            tree_offsets.append(node_idx)
            for n in tree.nodes:
                c_lines.append(f"    {{ {int(n['feature_idx'])}, {float(n['num_threshold']):.8f}f, {int(n['left'])}, {int(n['right'])}, {float(n['value']):.8f}f, {int(n['is_leaf'])} }},")
                node_idx += 1
    c_lines.append('};')
    c_lines.append('')
    c_lines.append(f'static const uint32_t TREE_OFFSETS[{len(tree_offsets)}] = {{')
    c_lines.append('    ' + ', '.join(str(o) for o in tree_offsets))
    c_lines.append('};')
    c_lines.append('')

    c_lines.extend([
        'void hgb_predict_raw(const float* features, float* raw_margins) {',
        '    raw_margins[0] = BASELINE_PREDICTION[0];',
        '    raw_margins[1] = BASELINE_PREDICTION[1];',
        '    raw_margins[2] = BASELINE_PREDICTION[2];',
        '',
        '    for (int iter = 0; iter < HGB_NUM_ITERATIONS; ++iter) {',
        '        for (int c = 0; c < HGB_NUM_CLASSES; ++c) {',
        '            uint32_t tree_idx = iter * HGB_NUM_CLASSES + c;',
        '            uint32_t curr_node = TREE_OFFSETS[tree_idx];',
        '            while (!ALL_NODES[curr_node].is_leaf) {',
        '                int16_t f_idx = ALL_NODES[curr_node].feature_idx;',
        '                if (features[f_idx] <= ALL_NODES[curr_node].threshold) {',
        '                    curr_node = TREE_OFFSETS[tree_idx] + ALL_NODES[curr_node].left;',
        '                } else {',
        '                    curr_node = TREE_OFFSETS[tree_idx] + ALL_NODES[curr_node].right;',
        '                }',
        '            }',
        '            raw_margins[c] += ALL_NODES[curr_node].value;',
        '        }',
        '    }',
        '}',
        '',
        'void hgb_predict_proba(const float* features, float* probabilities) {',
        '    float raw[3];',
        '    hgb_predict_raw(features, raw);',
        '',
        '    float max_m = raw[0];',
        '    if (raw[1] > max_m) max_m = raw[1];',
        '    if (raw[2] > max_m) max_m = raw[2];',
        '',
        '    float exp0 = expf(raw[0] - max_m);',
        '    float exp1 = expf(raw[1] - max_m);',
        '    float exp2 = expf(raw[2] - max_m);',
        '    float sum = exp0 + exp1 + exp2;',
        '',
        '    probabilities[0] = exp0 / sum;',
        '    probabilities[1] = exp1 / sum;',
        '    probabilities[2] = exp2 / sum;',
        '}',
        '',
        'int hgb_predict_bayes(const float* features) {',
        '    float probs[3];',
        '    hgb_predict_proba(features, probs);',
        '',
        '    float cost0 = probs[0]*COST_MATRIX[0][0] + probs[1]*COST_MATRIX[1][0] + probs[2]*COST_MATRIX[2][0];',
        '    float cost1 = probs[0]*COST_MATRIX[0][1] + probs[1]*COST_MATRIX[1][1] + probs[2]*COST_MATRIX[2][1];',
        '    float cost2 = probs[0]*COST_MATRIX[0][2] + probs[1]*COST_MATRIX[1][2] + probs[2]*COST_MATRIX[2][2];',
        '',
        '    if (cost0 <= cost1 && cost0 <= cost2) return 0;',
        '    if (cost1 <= cost0 && cost1 <= cost2) return 1;',
        '    return 2;',
        '}',
        '',
        'uint64_t hgb_profile_traversal(const float* features, uint64_t* m4_cycles, uint64_t* m7_cycles) {',
        '    uint64_t total_nodes_visited = 0;',
        '    for (int iter = 0; iter < HGB_NUM_ITERATIONS; ++iter) {',
        '        for (int c = 0; c < HGB_NUM_CLASSES; ++c) {',
        '            uint32_t tree_idx = iter * HGB_NUM_CLASSES + c;',
        '            uint32_t curr_node = TREE_OFFSETS[tree_idx];',
        '            while (!ALL_NODES[curr_node].is_leaf) {',
        '                total_nodes_visited++;',
        '                int16_t f_idx = ALL_NODES[curr_node].feature_idx;',
        '                if (features[f_idx] <= ALL_NODES[curr_node].threshold) {',
        '                    curr_node = TREE_OFFSETS[tree_idx] + ALL_NODES[curr_node].left;',
        '                } else {',
        '                    curr_node = TREE_OFFSETS[tree_idx] + ALL_NODES[curr_node].right;',
        '                }',
        '            }',
        '            total_nodes_visited++;',
        '        }',
        '    }',
        '    if (m4_cycles) *m4_cycles = total_nodes_visited * 18ULL;',
        '    if (m7_cycles) *m7_cycles = total_nodes_visited * 5ULL;',
        '    return total_nodes_visited;',
        '}'
    ])

    with open(C_SRC_DIR / 'hgb_forest.c', 'w', encoding='utf-8') as f:
        f.write('\n'.join(c_lines))
    print(f"Generated C files -> {C_SRC_DIR / 'hgb_forest.c'}")


if __name__ == '__main__':
    transpile_hgb_to_c()
