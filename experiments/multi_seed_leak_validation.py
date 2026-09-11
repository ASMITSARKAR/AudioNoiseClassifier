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
SEEDS = [42, 101, 202, 303, 404]


def run_5fold_cv_for_seed(X, y, folds, seed, config_type):
    unique_folds = sorted(np.unique(folds))
    all_y_true = []
    all_y_pred = []
    fold_f1s = []
    fold_costs = []

    for test_fold in unique_folds:
        train_mask = folds != test_fold
        test_mask = folds == test_fold
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if config_type == 'config_1_baseline':
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6, l2_regularization=0.1, random_state=seed
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        elif config_type == 'config_4_cost_sensitive':
            class_weights = {0: 1.0, 1: 1.5, 2: 3.0}
            sw = np.array([class_weights[lbl] for lbl in y_train], dtype=np.float32)
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=6, l2_regularization=0.1, random_state=seed
            )
            clf.fit(X_train, y_train, sample_weight=sw)
            probs = clf.predict_proba(X_test)
            y_pred = predict_bayes_optimal(probs, COST_MATRIX)
        elif config_type == 'config_5_tuned':
            class_weights = {0: 1.0, 1: 1.0, 2: 2.0}
            sw = np.array([class_weights[lbl] for lbl in y_train], dtype=np.float32)
            clf = HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, max_depth=8, max_leaf_nodes=63,
                min_samples_leaf=15, l2_regularization=0.1, random_state=seed
            )
            clf.fit(X_train, y_train, sample_weight=sw)
            probs = clf.predict_proba(X_test)
            y_pred = predict_bayes_optimal(probs, COST_MATRIX)

        m = compute_all_metrics(y_test, y_pred)
        fold_f1s.append(m['macro_f1'])
        fold_costs.append(m['cost_weighted_error'])
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
    imp_total = np.sum(cm[2])
    imp_as_stat = cm[2, 0]

    return {
        'macro_f1': float(np.mean(fold_f1s)),
        'cost_error': float(np.mean(fold_costs)),
        'imp_total': int(imp_total),
        'imp_as_stat': int(imp_as_stat),
        'imp_as_stat_pct': float(imp_as_stat / imp_total * 100.0),
    }


def main():
    print(f"Loading {FEATURES_PARQUET}...")
    df = pd.read_parquet(FEATURES_PARQUET)

    extractor = AudioFeatureExtractor(CONFIG_PATH)
    cols = [c for c in extractor.feature_names if c in df.columns]
    X = df[cols].values.astype(np.float32)
    y = df['target_class_id'].values.astype(int)
    folds = df['fold'].values.astype(int)

    configs = {
        'Baseline (Argmax)': 'config_1_baseline',
        'Cost-Sensitive (Bayes Head)': 'config_4_cost_sensitive',
        'Tuned Model': 'config_5_tuned',
    }

    print(f"Validating across seeds: {SEEDS}\n")
    rows = []

    for name, cfg_key in configs.items():
        print(f"Testing {name}...")
        seed_runs = []
        for s in SEEDS:
            res = run_5fold_cv_for_seed(X, y, folds, seed=s, config_type=cfg_key)
            seed_runs.append(res)
            print(f"  Seed {s}: Cost={res['cost_error']:.4f} | Imp->Stat={res['imp_as_stat']} ({res['imp_as_stat_pct']:.1f}%) | F1={res['macro_f1']:.4f}")

        leaks = [r['imp_as_stat'] for r in seed_runs]
        leak_pcts = [r['imp_as_stat_pct'] for r in seed_runs]
        costs = [r['cost_error'] for r in seed_runs]
        f1s = [r['macro_f1'] for r in seed_runs]

        rows.append({
            'Configuration': name,
            'Imp->Stat Count': f"{np.mean(leaks):.1f} +/- {np.std(leaks):.2f}",
            'Leak %': f"{np.mean(leak_pcts):.2f}% +/- {np.std(leak_pcts):.2f}%",
            'Cost Error': f"{np.mean(costs):.4f} +/- {np.std(costs):.4f}",
            'Macro F1': f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
        })

    summary_df = pd.DataFrame(rows)
    print("\nStability Summary:")
    print(summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
