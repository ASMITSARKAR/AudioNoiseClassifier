import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.dataset.mapper import ClassTaxonomyMapper
from src.dataset.loader import download_esc50, load_esc50_metadata, load_and_resample_audio, slice_audio_windows
from src.models.neural import WaveformTransientCNN
from src.evaluation.metrics import COST_MATRIX, compute_all_metrics, aggregate_fold_results, format_cv_summary
from src.models.cost_sensitive import predict_bayes_optimal

CONFIG_PATH = Path('configs/config.yaml')
MAPPING_PATH = Path('configs/class_mapping.yaml')
CACHE_AUDIO_PATH = Path('data/processed/raw_audio_windows_esc50.pt')
MODELS_DIR = Path('models')


class RawAudioDataset(Dataset):
    def __init__(self, waveforms: torch.Tensor, labels: torch.Tensor, folds: torch.Tensor, augment: bool = False):
        self.waveforms = waveforms
        self.labels = labels
        self.folds = folds
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        audio = self.waveforms[idx].clone()
        if self.augment:
            # mild gain jitter + low level gaussian noise
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain
            if np.random.rand() < 0.3:
                audio = audio + torch.randn_like(audio) * 0.005

        return audio.unsqueeze(0), int(self.labels[idx])


def build_or_load_raw_audio_dataset(
    config_path: Path = CONFIG_PATH,
    mapping_path: Path = MAPPING_PATH,
    cache_path: Path = CACHE_AUDIO_PATH
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cache_path.exists():
        print(f"Loading cached audio -> {cache_path}")
        data = torch.load(cache_path, weights_only=True)
        return data['waveforms'], data['labels'], data['folds']

    print("Generating raw audio windows from ESC-50...")
    esc50_dir = download_esc50(Path('data/raw'))
    mapper = ClassTaxonomyMapper(mapping_path)
    meta_df = load_esc50_metadata(esc50_dir, mapper)

    all_waveforms = []
    all_labels = []
    all_folds = []

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc='Slicing audio'):
        try:
            audio, sr = load_and_resample_audio(row['audio_path'], target_sr=22050)
            windows = slice_audio_windows(audio, sr=sr, window_sec=1.0, hop_sec=0.5)
            file_peak = np.max(np.abs(audio)) + 1e-8

            for chunk in windows:
                chunk_peak = np.max(np.abs(chunk))
                if chunk_peak < 1e-5:
                    continue
                # skip dead-air frames for impulsive class
                if int(row['target_class_id']) == 2 and chunk_peak < 0.15 * file_peak:
                    continue

                all_waveforms.append(torch.from_numpy(chunk).float())
                all_labels.append(int(row['target_class_id']))
                all_folds.append(int(row['fold']))
        except Exception as e:
            print(f"[warning] skipping {row['filename']}: {e}")

    waveforms_tensor = torch.stack(all_waveforms).float()
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    folds_tensor = torch.tensor(all_folds, dtype=torch.long)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'waveforms': waveforms_tensor, 'labels': labels_tensor, 'folds': folds_tensor}, cache_path)
    print(f"Saved {len(waveforms_tensor)} windows to {cache_path}")
    return waveforms_tensor, labels_tensor, folds_tensor


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for audios, targets in loader:
        audios, targets = audios.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(audios)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / len(loader.dataset)


def evaluate_bayes(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for audios, targets in loader:
            audios = audios.to(device)
            logits = model(audios)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(targets.numpy())

    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)
    preds = predict_bayes_optimal(probs_arr, COST_MATRIX)
    return targets_arr, preds


def benchmark_waveform_cnn_latency(model: nn.Module, sr: int = 22050, n_iterations: int = 500, warmup: int = 30) -> dict:
    model.eval()
    dummy = torch.randn(1, 1, sr).float()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    latencies = []
    with torch.no_grad():
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    arr = np.array(latencies)
    return {
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'mean_ms': float(np.mean(arr)),
    }


def main():
    parser = argparse.ArgumentParser(description='Train WaveformTransientCNN on raw ESC-50 audio')
    parser.add_argument('--epochs', type=int, default=20, help='Epochs per fold')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    waveforms, labels, folds = build_or_load_raw_audio_dataset()
    print(f"Loaded {waveforms.shape[0]} windows ({waveforms.shape[1]} samples each)")

    unique_folds = sorted(torch.unique(folds).tolist())
    fold_results = []
    # heavier weight on impulsive class
    class_weights = torch.tensor([1.0, 1.5, 3.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\nRunning 5-fold CV...")
    total_imp_leaks = 0

    for test_fold in unique_folds:
        train_pool = [f for f in unique_folds if f != test_fold]
        val_fold = train_pool[0]
        train_inner = [f for f in train_pool if f != val_fold]

        train_mask = torch.isin(folds, torch.tensor(train_inner))
        val_mask = folds == val_fold
        test_mask = folds == test_fold

        train_ds = RawAudioDataset(waveforms[train_mask], labels[train_mask], folds[train_mask], augment=True)
        val_ds = RawAudioDataset(waveforms[val_mask], labels[val_mask], folds[val_mask], augment=False)
        test_ds = RawAudioDataset(waveforms[test_mask], labels[test_mask], folds[test_mask], augment=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = WaveformTransientCNN(num_classes=3, base_channels=32).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_val_f1 = -1.0
        best_state = copy.deepcopy(model.state_dict())

        for epoch in range(1, args.epochs + 1):
            _ = train_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            val_true, val_pred = evaluate_bayes(model, val_loader, device)
            val_m = compute_all_metrics(val_true, val_pred, model_name=f'Val_Fold{test_fold}')
            if val_m['macro_f1'] > best_val_f1:
                best_val_f1 = val_m['macro_f1']
                best_state = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_state)
        y_true, y_pred = evaluate_bayes(model, test_loader, device)
        test_m = compute_all_metrics(y_true, y_pred, model_name=f'Fold_{test_fold}')
        fold_results.append(test_m)

        cm = np.array(test_m['confusion_matrix'])
        total_imp_leaks += int(cm[2, 0])
        print(f"  Fold {test_fold}: val_f1={best_val_f1:.3f} | test_f1={test_m['macro_f1']:.3f} | cost={test_m['cost_weighted_error']:.3f} | imp_rec={test_m['impulsive_recall']:.3f}")

    agg = aggregate_fold_results(fold_results)
    print(format_cv_summary(agg))

    n_imp = int((labels == 2).sum().item())
    print(f"Impulsive -> Stationary leaks: {total_imp_leaks}/{n_imp} ({total_imp_leaks / max(n_imp, 1) * 100:.1f}%)")

    # benchmark CPU latency
    print("\nBenchmarking CPU inference latency...")
    cpu_model = WaveformTransientCNN(num_classes=3, base_channels=32).to('cpu')
    lat = benchmark_waveform_cnn_latency(cpu_model, sr=22050, n_iterations=500, warmup=30)
    print(f"Latency: P50={lat['p50_ms']:.2f}ms, P95={lat['p95_ms']:.2f}ms, P99={lat['p99_ms']:.2f}ms")
    print(f"Params: {cpu_model.count_parameters():,} ({cpu_model.count_parameters() * 4 / 1024:.1f} KB)")

    # train final model on all data
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / 'best_waveform_cnn.pt'

    full_ds = RawAudioDataset(waveforms, labels, folds, augment=True)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    final_model = WaveformTransientCNN(num_classes=3, base_channels=32).to(device)
    final_opt = optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=1e-2)
    final_sched = optim.lr_scheduler.CosineAnnealingLR(final_opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_epoch(final_model, full_loader, final_opt, criterion, device)
        final_sched.step()

    torch.save(final_model.state_dict(), out_path)
    print(f"Saved final weights to {out_path}")


if __name__ == '__main__':
    main()
