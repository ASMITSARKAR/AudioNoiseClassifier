import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix

CLASS_NAMES = ['stationary', 'non_stationary', 'impulsive']
CLASS_IDS = [0, 1, 2]

COST_MATRIX = np.array([
    [0.0, 1.0, 1.5],
    [2.0, 0.0, 1.0],
    [3.0, 1.0, 0.0]
], dtype=np.float32)


def compute_cost_weighted_error(y_true, y_pred, cost_matrix=COST_MATRIX):
    costs = cost_matrix[y_true, y_pred]
    return float(np.mean(costs))


def compute_per_class_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    p, r, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        zero_division=0
    )
    results = {}
    for idx, name in enumerate(class_names):
        results[name] = {
            'precision': float(p[idx]),
            'recall': float(r[idx]),
            'f1': float(f1[idx]),
            'support': int(sup[idx]),
        }
    return results


def compute_all_metrics(y_true, y_pred, model_name='model'):
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cost_err = compute_cost_weighted_error(y_true, y_pred)
    per_class = compute_per_class_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    imp_recall = per_class['impulsive']['recall']

    return {
        'model': model_name,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'cost_weighted_error': cost_err,
        'impulsive_recall': imp_recall,
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
    }


def format_results_table(results):
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  Model: {results['model']}")
    lines.append("=" * 60)
    lines.append(f"  Macro F1            : {results['macro_f1']:.4f}")
    lines.append(f"  Weighted F1         : {results['weighted_f1']:.4f}")
    lines.append(f"  Cost-Weighted Error : {results['cost_weighted_error']:.4f}")
    lines.append(f"  Impulsive Recall    : {results['impulsive_recall']:.4f}")
    lines.append("")
    lines.append("  Per-Class Breakdown:")

    for cls_name, m in results['per_class'].items():
        p_str = f"P={m['precision']:.3f}"
        r_str = f"R={m['recall']:.3f}"
        f1_str = f"F1={m['f1']:.3f}"
        lines.append(f"    {cls_name:<18}  {p_str}  {r_str}  {f1_str}  n={m['support']}")

    cm = np.array(results['confusion_matrix'])
    lines.append("")
    lines.append("  Confusion Matrix (rows=true, cols=pred):")
    lines.append("              stat  non_stat  impulse")
    for idx, row_name in enumerate(CLASS_NAMES):
        lines.append(f"    {row_name:<14}  {cm[idx, 0]:5d}  {cm[idx, 1]:8d}  {cm[idx, 2]:7d}")

    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)


def aggregate_fold_results(fold_results):
    metric_keys = ['macro_f1', 'weighted_f1', 'cost_weighted_error', 'impulsive_recall']
    aggregated = {
        'model': fold_results[0]['model'],
        'n_folds': len(fold_results),
    }

    for key in metric_keys:
        vals = [r[key] for r in fold_results]
        aggregated[f"{key}_mean"] = float(np.mean(vals))
        aggregated[f"{key}_std"] = float(np.std(vals))

    return aggregated


def format_cv_summary(agg):
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  CV Results ({agg['n_folds']}-Fold): {agg['model']}")
    lines.append("=" * 60)
    lines.append(f"  Macro F1            : {agg['macro_f1_mean']:.4f} +/- {agg['macro_f1_std']:.4f}")
    lines.append(f"  Weighted F1         : {agg['weighted_f1_mean']:.4f} +/- {agg['weighted_f1_std']:.4f}")
    lines.append(f"  Cost-Weighted Error : {agg['cost_weighted_error_mean']:.4f} +/- {agg['cost_weighted_error_std']:.4f}")
    lines.append(f"  Impulsive Recall    : {agg['impulsive_recall_mean']:.4f} +/- {agg['impulsive_recall_std']:.4f}")
    lines.append("=" * 60)
    lines.append("")
    return "\n".join(lines)
