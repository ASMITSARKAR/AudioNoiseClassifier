import argparse
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.features.dataset_processor import build_feature_dataset
from src.features.extractor import AudioFeatureExtractor
from src.models.baselines import get_all_baselines
from src.models.cost_sensitive import CostSensitiveHGBClassifier, COST_MATRIX
from src.evaluation.metrics import (
    compute_all_metrics,
    aggregate_fold_results,
    format_cv_summary,
)
from src.evaluation.benchmark import profile_end_to_end, format_latency_report

CONFIG_PATH = Path('configs/config.yaml')
MAPPING_PATH = Path('configs/class_mapping.yaml')
FEATURES_PARQUET = Path('data/processed/features_esc50.parquet')
FEATURES_CSV = Path('data/processed/features_esc50.csv')
MODELS_DIR = Path('models')


def load_or_build_features(skip_extraction=False):
    if FEATURES_PARQUET.exists():
        print(f"Loading cached features: {FEATURES_PARQUET}")
        return pd.read_parquet(FEATURES_PARQUET)
    if FEATURES_CSV.exists():
        print(f"Loading cached features: {FEATURES_CSV}")
        return pd.read_csv(FEATURES_CSV)
    if skip_extraction:
        raise FileNotFoundError("Feature cache missing and --skip-extraction was passed.")

    print("No feature cache found. Running extraction pipeline...")
    return build_feature_dataset(config_path=CONFIG_PATH, mapping_path=MAPPING_PATH, save_output=True)


def prepare_arrays(df, extractor):
    feature_cols = extractor.feature_names
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        print(f"[warning] {len(missing)} features missing from dataset: {missing}")

    X = df[available].values.astype(np.float32)
    y = df['target_class_id'].values.astype(int)
    folds = df['fold'].values.astype(int)
    return X, y, folds, available


def run_predefined_fold_cv(model, X, y, folds, model_name):
    unique_folds = sorted(np.unique(folds))
    fold_results = []

    for test_fold in unique_folds:
        train_mask = folds != test_fold
        test_mask = folds == test_fold
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time_ms = (time.perf_counter() - t0) * 1000.0

        y_pred = model.predict(X_test)
        metrics = compute_all_metrics(y_test, y_pred, model_name=model_name)
        metrics['train_time_ms'] = round(fit_time_ms, 1)
        metrics['test_fold'] = int(test_fold)
        fold_results.append(metrics)

        print(f"  Fold {test_fold}: macro_f1={metrics['macro_f1']:.3f} | cost={metrics['cost_weighted_error']:.3f} | imp_recall={metrics['impulsive_recall']:.3f} | {fit_time_ms:.0f}ms")

    return fold_results


def print_feature_importance(model, feature_names, top_n=15):
    estimator = model
    if hasattr(model, 'named_steps'):
        for step in model.named_steps.values():
            if hasattr(step, 'feature_importances_'):
                estimator = step
                break

    if not hasattr(estimator, 'feature_importances_'):
        return

    importances = estimator.feature_importances_
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Top {top_n} features:")
    for rank, (name, imp) in enumerate(ranked[:top_n], 1):
        bar = '#' * int(imp * 300)
        print(f"    {rank:2d}. {name:<30} {imp:.4f}  {bar}")


def main(skip_extraction=False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_or_build_features(skip_extraction)
    print(f"Total samples: {len(df)}")
    print("Class breakdown:\n", df['target_class_name'].value_counts(), "\n")

    extractor = AudioFeatureExtractor(CONFIG_PATH)
    X, y, folds, feature_names = prepare_arrays(df, extractor)
    print(f"X: {X.shape}, folds: {np.unique(folds).tolist()}")

    models = get_all_baselines()
    cv_summaries = {}
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name} across folds...")
        fold_res = run_predefined_fold_cv(model, X, y, folds, name)
        agg = aggregate_fold_results(fold_res)
        cv_summaries[name] = agg
        print(format_cv_summary(agg))

        model.fit(X, y)
        fitted_models[name] = model
        print_feature_importance(model, feature_names)

    print("\nMeasuring end-to-end CPU inference speed...")
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    bench_cfg = cfg.get('benchmark', {})
    budget_ms = bench_cfg.get('target_latency_budget_ms', 10.0)
    n_iter = bench_cfg.get('n_iterations', 300)
    warmup = bench_cfg.get('n_warmup', 20)

    dummy_sig = np.random.randn(extractor.sr).astype(np.float32)
    for name, model in fitted_models.items():
        lat = profile_end_to_end(extractor, model, dummy_sig, sr=extractor.sr, n_iterations=n_iter, warmup=warmup)
        print(format_latency_report(name, lat, budget_ms=budget_ms))

    best_name = min(cv_summaries, key=lambda n: cv_summaries[n]['cost_weighted_error_mean'])
    best_model = fitted_models[best_name]
    best_path = MODELS_DIR / 'best_baseline.pkl'

    save_dict = {
        'feature_names': feature_names,
        'model_name': best_name,
    }

    if best_name == 'hist_gradient_boosting':
        wrapper = CostSensitiveHGBClassifier(best_model, COST_MATRIX)
        save_dict['model'] = wrapper
        save_dict['raw_hgb'] = best_model
        save_dict['cost_matrix'] = COST_MATRIX
    else:
        save_dict['model'] = best_model

    with open(best_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f"\nBest baseline: '{best_name}'")
    print(f"  Cost error : {cv_summaries[best_name]['cost_weighted_error_mean']:.4f}")
    print(f"  Macro F1   : {cv_summaries[best_name]['macro_f1_mean']:.4f}")
    print(f"Saved checkpoint -> {best_path}\n")

    print(f"{'Model':<28} {'Macro F1':>10} {'Cost Err':>10} {'Imp Recall':>12}")
    print("-" * 62)
    for name, agg in sorted(cv_summaries.items(), key=lambda x: x[1]['cost_weighted_error_mean']):
        print(f"{name:<28} {agg['macro_f1_mean']:>10.4f} {agg['cost_weighted_error_mean']:>10.4f} {agg['impulsive_recall_mean']:>12.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-extraction', action='store_true', help='Skip feature extraction')
    args = parser.parse_args()
    main(skip_extraction=args.skip_extraction)
