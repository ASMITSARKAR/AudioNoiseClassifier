import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.loader import download_esc50, load_esc50_metadata, load_and_resample_audio
from src.dataset.mapper import ClassTaxonomyMapper
from src.pipeline.noise_suppression_pipeline import NoiseSuppressionPipeline
from src.evaluation.quality_metrics import compute_filter_switch_rate


def assemble_30s_stream(sr=22050):
    raw_dir = Path('data/raw')
    esc50_dir = download_esc50(raw_dir)
    mapper = ClassTaxonomyMapper('configs/class_mapping.yaml')
    meta_df = load_esc50_metadata(esc50_dir, mapper)

    stat_row = meta_df[meta_df['category'] == 'wind'].iloc[0]
    nonstat_row = meta_df[meta_df['category'] == 'siren'].iloc[0]
    imp_row = meta_df[meta_df['category'] == 'door_wood_knock'].iloc[0]

    audio_stat, _ = load_and_resample_audio(stat_row['audio_path'], target_sr=sr)
    audio_nonstat, _ = load_and_resample_audio(nonstat_row['audio_path'], target_sr=sr)
    audio_imp, _ = load_and_resample_audio(imp_row['audio_path'], target_sr=sr)

    def repeat_to_len(arr, target_len):
        reps = int(np.ceil(target_len / len(arr)))
        return np.tile(arr, reps)[:target_len]

    seg_samples = int(10.0 * sr)
    s_stat = repeat_to_len(audio_stat, seg_samples)
    s_nonstat = repeat_to_len(audio_nonstat, seg_samples)
    s_imp = repeat_to_len(audio_imp, seg_samples)

    s_stat = 0.3 * (s_stat / (np.max(np.abs(s_stat)) + 1e-6))
    s_nonstat = 0.3 * (s_nonstat / (np.max(np.abs(s_nonstat)) + 1e-6))
    s_imp = 0.35 * (s_imp / (np.max(np.abs(s_imp)) + 1e-6))

    return np.concatenate([s_stat, s_nonstat, s_imp]).astype(np.float32)


def main():
    sr = 22050
    print("\nStreaming Stress Test (30s continuous audio)...")

    stream_30s = assemble_30s_stream(sr=sr)
    pipeline = NoiseSuppressionPipeline(
        model_path='models/best_baseline.pkl',
        config_path='configs/config.yaml',
        sr=sr,
        chunk_sec=0.5
    )
    chunk_size = pipeline.chunk_size

    latencies = []
    decisions = []

    for i in range(0, len(stream_30s), chunk_size):
        chunk = stream_30s[i:i + chunk_size]
        if len(chunk) < 64:
            continue

        _, meta = pipeline.process_chunk(chunk, apply_smoothing=True)
        latencies.append(meta['timings_ms']['end_to_end_ms'])
        decisions.append(meta['class_id'])

    lats = np.array(latencies)
    max_lat = np.max(lats)
    p50_lat = np.percentile(lats, 50)
    p95_lat = np.percentile(lats, 95)
    margin = 500.0 - max_lat

    print(f"  Processed {len(decisions)} chunks (500ms each)")
    print(f"  Turnaround latency: P50={p50_lat:.1f}ms, P95={p95_lat:.1f}ms, Max={max_lat:.1f}ms")
    print(f"  Budget headroom   : {margin:.1f}ms ({margin / 500.0 * 100:.1f}%)")

    switches = compute_filter_switch_rate(decisions)
    print(f"  Filter transitions: {switches['total_switches']} switches ({switches['switch_rate'] * 100:.1f}%)\n")


if __name__ == '__main__':
    main()
