import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features.extractor import AudioFeatureExtractor
from src.models.cost_sensitive import CostSensitiveHGBClassifier
from src.evaluation.metrics import COST_MATRIX

CONFIG_PATH = Path('configs/config.yaml')
FEATURES_PARQUET = Path('data/processed/features_esc50.parquet')
MODELS_DIR = Path('models')


def train_and_save_model():
    print(f"Reading {FEATURES_PARQUET}...")
    df = pd.read_parquet(FEATURES_PARQUET)

    extractor = AudioFeatureExtractor(CONFIG_PATH)
    cols = [c for c in extractor.feature_names if c in df.columns]
    X = df[cols].values.astype(np.float32)
    y = df['target_class_id'].values.astype(int)

    # weight impulsive samples heavier so model doesn't miss transients
    class_weights = {0: 1.0, 1: 1.5, 2: 3.0}
    weights = np.array([class_weights[label] for label in y], dtype=np.float32)

    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.1,
        random_state=42
    )

    print(f"Training HGB on {len(X)} samples x {len(cols)} features...")
    hgb.fit(X, y, sample_weight=weights)

    wrapper = CostSensitiveHGBClassifier(hgb, COST_MATRIX)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / 'best_baseline.pkl'

    with open(out_path, 'wb') as f:
        pickle.dump({
            'model': wrapper,
            'raw_hgb': hgb,
            'feature_names': cols,
            'model_name': 'CostSensitive_HistGBM',
            'cost_matrix': COST_MATRIX
        }, f)

    print(f"Saved model checkpoint -> {out_path}")


if __name__ == '__main__':
    train_and_save_model()
