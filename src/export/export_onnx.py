import pickle
from pathlib import Path
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from src.models.cost_sensitive import COST_MATRIX, predict_bayes_optimal

MODEL_PKL_PATH = Path('models/best_baseline.pkl')
ONNX_OUTPUT_PATH = Path('models/noise_classifier.onnx')


def export_hgb_to_onnx():
    print(f"Loading {MODEL_PKL_PATH}...")
    with open(MODEL_PKL_PATH, 'rb') as f:
        ckpt = pickle.load(f)
        raw_hgb = ckpt['raw_hgb']
        feat_names = ckpt['feature_names']

    n_features = len(feat_names)
    print(f"Converting HGB model ({n_features} features) to ONNX...")
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    options = {id(raw_hgb): {'zipmap': False}}
    onnx_model = convert_sklearn(raw_hgb, initial_types=initial_type, options=options, target_opset=17)

    ONNX_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ONNX_OUTPUT_PATH, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    size_kb = ONNX_OUTPUT_PATH.stat().st_size / 1024.0
    print(f"Saved ONNX model -> {ONNX_OUTPUT_PATH} ({size_kb:.1f} KB)")

    print("Testing ONNX Runtime output parity against sklearn...")
    sess = ort.InferenceSession(str(ONNX_OUTPUT_PATH))
    input_name = sess.get_inputs()[0].name

    np.random.seed(42)
    test_vecs = np.random.randn(50, n_features).astype(np.float32)

    skl_probs = raw_hgb.predict_proba(test_vecs)
    skl_preds = predict_bayes_optimal(skl_probs, COST_MATRIX)

    onnx_outs = sess.run(None, {input_name: test_vecs})
    onnx_probs = onnx_outs[1]
    onnx_preds = predict_bayes_optimal(onnx_probs, COST_MATRIX)

    diff = np.max(np.abs(skl_probs - onnx_probs))
    matches = int(np.sum(skl_preds == onnx_preds))

    print(f"  Max prob delta: {diff:.2e}")
    print(f"  Matches: {matches}/50 ({matches / 50 * 100:.1f}%)")

    assert diff < 1e-4, f"Probabilities diverge: {diff}"
    assert matches == 50, "Predictions diverge between ONNX and scikit-learn"
    print("Parity OK.")


if __name__ == '__main__':
    export_hgb_to_onnx()
