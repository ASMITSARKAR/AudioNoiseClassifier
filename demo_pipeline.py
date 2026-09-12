import sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.dataset.loader import download_esc50, load_esc50_metadata, load_and_resample_audio
from src.dataset.mapper import ClassTaxonomyMapper
from src.pipeline.noise_suppression_pipeline import NoiseSuppressionPipeline
from src.evaluation.quality_metrics import (
    compute_segmental_snr,
    compute_log_spectral_distance,
    compute_filter_switch_rate,
)

OUTPUTS_DIR = Path('outputs')


def load_environmental_noise_tracks(sr: int = 22050) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_dir = Path('data/raw')
    esc50_dir = download_esc50(raw_dir)
    mapper = ClassTaxonomyMapper('configs/class_mapping.yaml')
    meta_df = load_esc50_metadata(esc50_dir, mapper)

    # pick 3 representative clips: wind (stationary), siren (nonstationary), door knock (impulsive)
    stat_row = meta_df[meta_df['category'] == 'wind'].iloc[0]
    nonstat_row = meta_df[meta_df['category'] == 'siren'].iloc[0]
    imp_row = meta_df[meta_df['category'] == 'door_wood_knock'].iloc[0]

    audio_stat, _ = load_and_resample_audio(stat_row['audio_path'], target_sr=sr)
    audio_nonstat, _ = load_and_resample_audio(nonstat_row['audio_path'], target_sr=sr)
    audio_imp, _ = load_and_resample_audio(imp_row['audio_path'], target_sr=sr)

    # 2 seconds per regime
    dur = int(2.0 * sr)
    n_stat = 0.3 * (audio_stat[:dur] / (np.max(np.abs(audio_stat[:dur])) + 1e-6))
    n_nonstat = 0.3 * (audio_nonstat[:dur] / (np.max(np.abs(audio_nonstat[:dur])) + 1e-6))
    n_imp = 0.35 * (audio_imp[:dur] / (np.max(np.abs(audio_imp[:dur])) + 1e-6))

    return (
        n_stat.astype(np.float32),
        n_nonstat.astype(np.float32),
        n_imp.astype(np.float32),
    )


def run_pipeline_demo():
    sr = 22050
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- Audio Noise Classifier & Adaptive Suppression Pipeline ---")

    n_stat, n_nonstat, n_imp = load_environmental_noise_tracks(sr=sr)
    pure_noise = np.concatenate([n_stat, n_nonstat, n_imp])

    print("\n[1] Dynamic Noise Regime Routing")
    print("  0.0s - 2.0s: Wind (Stationary)       -> Spectral Subtraction")
    print("  2.0s - 4.0s: Siren (Non-Stationary)  -> Adaptive NLMS")
    print("  4.0s - 6.0s: Door Knock (Impulsive)  -> Median / MAD Filter")
    print("-" * 75)
    print(f"{'Time':<12} {'Regime':<16} {'Probabilities [S, NS, I]':<26} {'Latency':>8}  {'Filter'}")
    print("-" * 75)

    pipeline = NoiseSuppressionPipeline(
        model_path='models/best_baseline.pkl',
        config_path='configs/config.yaml',
        sr=sr,
        chunk_sec=0.5
    )
    chunk_size = pipeline.chunk_size
    filter_names = {0: 'Spectral Subtraction', 1: 'Adaptive NLMS', 2: 'Median/MAD'}

    for i in range(0, len(pure_noise), chunk_size):
        chunk = pure_noise[i:i + chunk_size]
        _, meta = pipeline.process_chunk(chunk, apply_smoothing=False)
        probs = meta['probabilities']
        lat = meta['timings_ms']['end_to_end_ms']
        prob_str = f"[{probs['stationary']:.2f}, {probs['non_stationary']:.2f}, {probs['impulsive']:.2f}]"
        t0, t1 = i / sr, (i + len(chunk)) / sr
        print(f"[{t0:3.1f}s - {t1:3.1f}s]  {meta['class_name']:<16} {prob_str:<26} {lat:>6.1f}ms  -> {filter_names[meta['class_id']]}")

    # synthesize harmonic voice carrier over concatenated noise
    print("\n[2] Processing Mixed Speech + Noise Audio Stream...")
    total_len = len(pure_noise)
    t = np.linspace(0, total_len / sr, total_len, endpoint=False)
    carrier = np.zeros_like(t)
    for h in [1, 2, 3, 4, 5]:
        carrier += (0.8 / h) * np.sin(2 * np.pi * 220 * h * t)

    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 2.0 * t)) * (np.sin(2 * np.pi * 0.5 * t) > -0.3)
    clean_speech = (carrier * envelope * 0.35).astype(np.float32)
    mixed_audio = (clean_speech + pure_noise).astype(np.float32)

    in_wav = OUTPUTS_DIR / 'demo_noisy_input.wav'
    out_wav = OUTPUTS_DIR / 'demo_clean_suppressed.wav'
    ref_wav = OUTPUTS_DIR / 'demo_clean_reference.wav'

    sf.write(str(in_wav), mixed_audio, sr)
    sf.write(str(ref_wav), clean_speech, sr)

    pipeline.reset()
    cleaned_chunks = []
    latencies = []
    decisions = []

    for i in range(0, len(mixed_audio), chunk_size):
        chunk = mixed_audio[i:i + chunk_size]
        clean_chunk, meta = pipeline.process_chunk(chunk, apply_smoothing=True)
        cleaned_chunks.append(clean_chunk)
        latencies.append(meta['timings_ms']['end_to_end_ms'])
        decisions.append(meta['class_id'])

    final_audio = np.concatenate(cleaned_chunks)
    sf.write(str(out_wav), final_audio, sr)

    def calc_snr(sig, noisy):
        err = noisy - sig
        return 10.0 * np.log10(np.sum(sig ** 2) / (np.sum(err ** 2) + 1e-12))

    in_snr = calc_snr(clean_speech, mixed_audio)
    out_snr = calc_snr(clean_speech, final_audio)
    gain = out_snr - in_snr

    in_segsnr = compute_segmental_snr(clean_speech, mixed_audio, sr=sr)
    out_segsnr = compute_segmental_snr(clean_speech, final_audio, sr=sr)
    seg_gain = out_segsnr - in_segsnr

    in_lsd = compute_log_spectral_distance(clean_speech, mixed_audio, sr=sr)
    out_lsd = compute_log_spectral_distance(clean_speech, final_audio, sr=sr)

    switches = compute_filter_switch_rate(decisions)
    total_boundaries = len(decisions) - 1
    switch_pct = (switches['total_switches'] / max(total_boundaries, 1)) * 100.0

    print(f"  SNR Improvement     : {in_snr:5.2f} dB -> {out_snr:5.2f} dB ({gain:+5.2f} dB)")
    print(f"  Segmental SNR       : {in_segsnr:5.2f} dB -> {out_segsnr:5.2f} dB ({seg_gain:+5.2f} dB)")
    print(f"  Log Spectral Dist.  : {in_lsd:5.2f} dB -> {out_lsd:5.2f} dB")
    print(f"  Filter Transitions  : {switches['total_switches']} / {total_boundaries} ({switch_pct:.1f}%)")

    lats = np.array(latencies)
    print(f"  Latency (500ms chunk): P50 = {np.percentile(lats, 50):.1f}ms | P95 = {np.percentile(lats, 95):.1f}ms | Max = {np.max(lats):.1f}ms")
    print(f"  Output saved to {out_wav}\n")


if __name__ == '__main__':
    run_pipeline_demo()
