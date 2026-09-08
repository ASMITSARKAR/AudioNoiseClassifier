from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.dataset.mapper import ClassTaxonomyMapper
from src.dataset.loader import (
    download_esc50,
    load_esc50_metadata,
    load_and_resample_audio,
    slice_audio_windows,
)
from src.features.extractor import AudioFeatureExtractor


def build_feature_dataset(config_path='configs/config.yaml', mapping_path='configs/class_mapping.yaml', save_output=True):
    config_path = Path(config_path)
    mapping_path = Path(mapping_path)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    audio_cfg = config.get('audio', {})
    ds_cfg = config.get('dataset', {})

    target_sr = audio_cfg.get('sample_rate', 22050)
    win_sec = audio_cfg.get('window_size_sec', 1.0)
    hop_sec = audio_cfg.get('hop_size_sec', 0.5)

    raw_dir = Path(ds_cfg.get('raw_dir', 'data/raw'))
    processed_dir = Path(ds_cfg.get('processed_dir', 'data/processed'))
    processed_dir.mkdir(parents=True, exist_ok=True)

    esc50_dir = download_esc50(raw_dir)
    mapper = ClassTaxonomyMapper(mapping_path)
    meta_df = load_esc50_metadata(esc50_dir, mapper)

    print(f"Loaded {len(meta_df)} clips across {len(meta_df['category'].unique())} categories.")
    print("Class breakdown:\n", meta_df['target_class_name'].value_counts())

    extractor = AudioFeatureExtractor(config_path)
    records = []

    print(f"Computing features ({win_sec}s window, {hop_sec}s hop @ {target_sr} Hz)...")
    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc='Feature extraction'):
        try:
            audio, sr = load_and_resample_audio(row['audio_path'], target_sr=target_sr)
            windows = slice_audio_windows(audio, sr=sr, window_sec=win_sec, hop_sec=hop_sec)
            file_peak = np.max(np.abs(audio)) + 1e-8

            for chunk_idx, chunk in enumerate(windows):
                chunk_peak = np.max(np.abs(chunk))
                if chunk_peak < 1e-5:
                    continue

                if int(row['target_class_id']) == 2 and chunk_peak < 0.15 * file_peak:
                    continue

                feat_dict = extractor.extract_feature_dict(chunk, sr=sr)
                feat_dict['orig_filename'] = row['filename']
                feat_dict['fold'] = int(row['fold'])
                feat_dict['src_id'] = int(row['src_id'])
                feat_dict['category'] = row['category']
                feat_dict['chunk_idx'] = chunk_idx
                feat_dict['target_class_id'] = int(row['target_class_id'])
                feat_dict['target_class_name'] = row['target_class_name']
                records.append(feat_dict)
        except Exception as e:
            print(f"[warning] failed to process {row['filename']}: {e}")

    feature_df = pd.DataFrame(records)
    print(f"Extracted {len(feature_df)} total feature windows.")
    print("Distribution by class:\n", feature_df['target_class_name'].value_counts())

    if save_output:
        parquet_path = processed_dir / 'features_esc50.parquet'
        csv_path = processed_dir / 'features_esc50.csv'
        try:
            feature_df.to_parquet(parquet_path, index=False)
            print(f"Cached to {parquet_path}")
        except Exception:
            feature_df.to_csv(csv_path, index=False)
            print(f"Cached to {csv_path}")

    return feature_df
