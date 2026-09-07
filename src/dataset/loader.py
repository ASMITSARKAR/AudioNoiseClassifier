from pathlib import Path
import urllib.request
import zipfile
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

from src.dataset.mapper import ClassTaxonomyMapper

ESC50_ZIP_URL = 'https://github.com/karolpiczak/ESC-50/archive/master.zip'


def download_esc50(raw_dir):
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    target_folder = raw_path / 'ESC-50-master'
    meta_csv = target_folder / 'meta' / 'esc50.csv'

    if meta_csv.exists():
        return target_folder

    print(f"Downloading ESC-50 archive from {ESC50_ZIP_URL}...")
    zip_path = raw_path / 'ESC-50-master.zip'

    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = int(count * block_size * 100 / total_size)
            if count % 500 == 0 or pct >= 100:
                print(f"\rDownloading ESC-50: {pct}%", end='', flush=True)

    urllib.request.urlretrieve(ESC50_ZIP_URL, zip_path, reporthook=_progress)
    print("\nExtracting ESC-50...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(raw_path)

    if zip_path.exists():
        zip_path.unlink()

    print(f"Dataset ready at {target_folder}")
    return target_folder


def load_esc50_metadata(esc50_dir, mapper):
    esc50_path = Path(esc50_dir)
    meta_csv = esc50_path / 'meta' / 'esc50.csv'
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing ESC-50 metadata CSV: {meta_csv}")

    df = pd.read_csv(meta_csv)
    df['target_class_id'] = df['category'].apply(mapper.map_category)
    df['target_class_name'] = df['target_class_id'].apply(
        lambda tid: mapper.get_target_name(tid) if pd.notnull(tid) else 'excluded'
    )

    filtered_df = df[df['target_class_id'].notnull()].copy()
    filtered_df['target_class_id'] = filtered_df['target_class_id'].astype(int)
    filtered_df['audio_path'] = filtered_df['filename'].apply(
        lambda fn: str(esc50_path / 'audio' / fn)
    )

    if 'src_file' in filtered_df.columns and 'src_id' not in filtered_df.columns:
        filtered_df = filtered_df.rename(columns={'src_file': 'src_id'})

    return filtered_df.reset_index(drop=True)


def load_and_resample_audio(audio_path, target_sr=22050):
    audio, sr = sf.read(str(audio_path), dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type='soxr_hq')

    return audio, target_sr


def slice_audio_windows(audio, sr, window_sec=1.0, hop_sec=0.5):
    win_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    if len(audio) < win_len:
        padded = np.zeros(win_len, dtype=audio.dtype)
        padded[:len(audio)] = audio
        return [padded]

    windows = []
    for start in range(0, len(audio) - win_len + 1, hop_len):
        windows.append(audio[start:start + win_len])

    return windows
