import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features.extractor import AudioFeatureExtractor
from src.evaluation.metrics import COST_MATRIX, compute_all_metrics
from src.models.cost_sensitive import predict_bayes_optimal

CONFIG_PATH = Path('configs/config.yaml')
FEATURES_PARQUET = Path('data/processed/features_esc50.parquet')


def evaluate_cost_sensitive_hgb(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    sample_weights_by_class: dict | None = None,
    use_bayes_risk: bool = False,
    hgb_params: dict | None = None
) -> dict:
    params = {
        'max_iter': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'max_leaf_nodes': 31,
        'min_samples_leaf': 20,
        'l2_regularization': 0.1,
        'early_stopping': True,
        'n_iter_no_change': 15,
        'random_state': 42,
    }
    if hgb_params:
        params.update(hgb_params)

    unique_folds = sorted(np.unique(folds))
    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for test_fold in unique_folds:
        train_mask = folds != test_fold
        test_mask = folds == test_fold
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if sample_weights_by_class:
            sw_train = np.array([sample_weights_by_class[lbl] for lbl in y_train], dtype=np.float32)
        else:
            sw_train = None

        clf = HistGradientBoostingClassifier(**params)
        clf.fit(X_train, y_train, sample_weight=sw_train)

        if use_bayes_risk:
            probs = clf.predict_proba(X_test)
            y_pred = predict_bayes_optimal(probs, COST_MATRIX)
        else:
            y_pred = clf.predict(X_test)

        m = compute_all_metrics(y_test, y_pred, model_name='HGB_CostSensitive')
        fold_metrics.append(m)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
    imp_total = np.sum(cm[2])
    imp_stat = cm[2, 0]
    imp_non_stat = cm[2, 1]
    imp_correct = cm[2, 2]

    macro_f1s = [m['macro_f1'] for m in fold_metrics]
    cost_errors = [m['cost_weighted_error'] for m in fold_metrics]
    imp_recalls = [m['impulsive_recall'] for m in fold_metrics]

    return {
        'macro_f1_mean': float(np.mean(macro_f1s)),
        'macro_f1_std': float(np.std(macro_f1s)),
        'cost_error_mean': float(np.mean(cost_errors)),
        'cost_error_std': float(np.std(cost_errors)),
        'impulsive_recall_mean': float(np.mean(imp_recalls)),
        'impulsive_recall_std': float(np.std(imp_recalls)),
        'imp_total': int(imp_total),
        'imp_as_stat': int(imp_stat),
        'imp_as_stat_pct': float(imp_stat / imp_total * 100.0),
        'imp_as_non_stat': int(imp_non_stat),
        'imp_correct': int(imp_correct),
        'confusion_matrix': cm.tolist(),
    }


def run_cost_sensitive_experiments():
    print(f"Reading {FEATURES_PARQUET}...")
    df = pd.read_parquet(FEATURES_PARQUET)

    extractor = AudioFeatureExtractor(CONFIG_PATH)
    cols = [c for c in extractor.feature_names if c in df.columns]
    X = df[cols].values.astype(np.float32)
    y = df['target_class_id'].values.astype(int)
    folds = df['fold'].values.astype(int)

    rows = []

    # 1. Plain HGB
    print("\n--- 1. Baseline HGB (Argmax) ---")
    res = evaluate_cost_sensitive_hgb(X, y, folds, sample_weights_by_class=None, use_bayes_risk=False)
    rows.append({
        'Config': 'Baseline HGB',
        'Macro F1': f"{res['macro_f1_mean']:.4f}",
        'Cost Error': f"{res['cost_error_mean']:.4f}",
        'Imp Recall': f"{res['impulsive_recall_mean'] * 100:.1f}%",
        'Imp->Stat': f"{res['imp_as_stat']} ({res['imp_as_stat_pct']:.1f}%)",
    })

    # 2. HGB + Bayes Risk
    print("\n--- 2. HGB + Bayes Risk Head ---")
    b_res = evaluate_cost_sensitive_hgb(X, y, folds, sample_weights_by_class=None, use_bayes_risk=True)
    rows.append({
        'Config': 'HGB + Bayes Risk',
        'Macro F1': f"{b_res['macro_f1_mean']:.4f}",
        'Cost Error': f"{b_res['cost_error_mean']:.4f}",
        'Imp Recall': f"{b_res['impulsive_recall_mean'] * 100:.1f}%",
        'Imp->Stat': f"{b_res['imp_as_stat']} ({b_res['imp_as_stat_pct']:.1f}%)",
    })

    # 3. Sample weight configs
    weight_configs = [
        ('W_Imp=1.5', {0: 1.0, 1: 1.0, 2: 1.5}),
        ('W_Imp=2.0', {0: 1.0, 1: 1.0, 2: 2.0}),
        ('W_Imp=3.0', {0: 1.0, 1: 1.0, 2: 3.0}),
        ('W_Imp=3.0, W_NonStat=1.5', {0: 1.0, 1: 1.5, 2: 3.0}),
    ]
    for name, w in weight_configs:
        w_res = evaluate_cost_sensitive_hgb(X, y, folds, sample_weights_by_class=w, use_bayes_risk=False)
        rows.append({
            'Config': name,
            'Macro F1': f"{w_res['macro_f1_mean']:.4f}",
            'Cost Error': f"{w_res['cost_error_mean']:.4f}",
            'Imp Recall': f"{w_res['impulsive_recall_mean'] * 100:.1f}%",
            'Imp->Stat': f"{w_res['imp_as_stat']} ({w_res['imp_as_stat_pct']:.1f}%)",
        })

    # 4. Combined weights + Bayes risk
    for name, w in [('W_Imp=2.0 + Bayes', {0: 1.0, 1: 1.0, 2: 2.0}), ('W_Imp=3.0, W_NonStat=1.5 + Bayes', {0: 1.0, 1: 1.5, 2: 3.0})]:
        c_res = evaluate_cost_sensitive_hgb(X, y, folds, sample_weights_by_class=w, use_bayes_risk=True)
        rows.append({
            'Config': name,
            'Macro F1': f"{c_res['macro_f1_mean']:.4f}",
            'Cost Error': f"{c_res['cost_error_mean']:.4f}",
            'Imp Recall': f"{c_res['impulsive_recall_mean'] * 100:.1f}%",
            'Imp->Stat': f"{c_res['imp_as_stat']} ({c_res['imp_as_stat_pct']:.1f}%)",
        })

    summary_df = pd.DataFrame(rows)
    print("\nSummary Results:")
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == '__main__':
    run_cost_sensitive_experiments()
