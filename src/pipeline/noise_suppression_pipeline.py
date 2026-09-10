import pickle
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

from src.features.extractor import AudioFeatureExtractor
from src.pipeline.suppression_filters import (
    SpectralSubtractionFilter,
    AdaptiveNLMSFilter,
    MedianMADSpikeFilter,
)

CONFIG_PATH = Path('configs/config.yaml')
MODEL_PATH = Path('models/best_baseline.pkl')
CLASS_NAMES = ['stationary', 'non_stationary', 'impulsive']


class NoiseSuppressionPipeline:

    def __init__(
        self,
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        sr=22050,
        chunk_sec=0.5,
        analysis_window_sec=1.0,
        crossfade_ms=15.0
    ):
        self.sr = sr
        self.chunk_size = int(chunk_sec * sr)
        self.window_size = int(analysis_window_sec * sr)
        self.crossfade_samples = int(crossfade_ms / 1000.0 * sr)
        self.extractor = AudioFeatureExtractor(config_path)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
            self.classifier = checkpoint['model']
            self.feature_names = checkpoint['feature_names']
            self.model_name = checkpoint.get('model_name', 'CostSensitive_HGB')

        self.filters = {
            0: SpectralSubtractionFilter(sr=sr, alpha=1.8, beta=0.02, smoothing=0.98),
            1: AdaptiveNLMSFilter(filter_order=32, mu=0.02, delay=4),
            2: MedianMADSpikeFilter(kernel_size=15, threshold_mad=3.5),
        }

        self.circular_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.is_buffer_primed = False
        self.prev_class_id = 0
        self.prev_last_sample = None
        self.decision_history = []

    def reset(self):
        self.circular_buffer = np.zeros(self.window_size, dtype=np.float32)
        self.is_buffer_primed = False
        self.prev_class_id = 0
        self.prev_last_sample = None
        self.decision_history.clear()
        for flt in self.filters.values():
            flt.reset()

    def classify_window(self, window_audio):
        t0 = time.perf_counter()
        feat_vec = self.extractor.extract_feature_vector(window_audio, self.sr)
        t1 = time.perf_counter()

        feat_2d = feat_vec.reshape(1, -1)
        probs = self.classifier.predict_proba(feat_2d)[0]
        class_id = int(self.classifier.predict(feat_2d)[0])
        t2 = time.perf_counter()

        timings = {
            'feature_extraction_ms': (t1 - t0) * 1000.0,
            'inference_ms': (t2 - t1) * 1000.0,
            'total_classifier_ms': (t2 - t0) * 1000.0,
        }
        return class_id, CLASS_NAMES[class_id], probs, timings

    def process_chunk(self, chunk, apply_smoothing=False):
        chunk = np.asarray(chunk, dtype=np.float32)
        chunk_len = len(chunk)

        if not self.is_buffer_primed:
            if chunk_len >= self.window_size:
                self.circular_buffer = chunk[-self.window_size:].copy()
            else:
                repeats = int(np.ceil(self.window_size / chunk_len))
                self.circular_buffer = np.tile(chunk, repeats)[:self.window_size].copy()
            self.is_buffer_primed = True
        elif chunk_len >= self.window_size:
            self.circular_buffer = chunk[-self.window_size:].copy()
        else:
            self.circular_buffer = np.roll(self.circular_buffer, -chunk_len)
            self.circular_buffer[-chunk_len:] = chunk

        raw_class_id, class_name, probs, timings = self.classify_window(self.circular_buffer)
        self.decision_history.append(raw_class_id)

        if raw_class_id == 2:
            class_id = 2
        elif apply_smoothing and len(self.decision_history) >= 3:
            prev_two = self.decision_history[-3:-1]
            if raw_class_id in (0, 1) and prev_two[0] in (0, 1) and (prev_two[1] in (0, 1)):
                if prev_two[0] == prev_two[1] and prev_two[1] != raw_class_id:
                    class_id = prev_two[1]
                else:
                    class_id = raw_class_id
            else:
                class_id = raw_class_id
        else:
            class_id = raw_class_id

        t_filt_start = time.perf_counter()
        active_filter = self.filters[class_id]
        processed_chunk = active_filter.process(chunk)

        if class_id != self.prev_class_id and self.prev_last_sample is not None and len(processed_chunk) >= self.crossfade_samples:
            cf_len = min(self.crossfade_samples, len(processed_chunk))
            fade = np.linspace(1.0, 0.0, cf_len, endpoint=False, dtype=np.float32)
            step = self.prev_last_sample - processed_chunk[0]
            processed_chunk[:cf_len] += step * (fade ** 2)

        if len(processed_chunk) > 0:
            self.prev_last_sample = float(processed_chunk[-1])

        t_filt_end = time.perf_counter()
        timings['filter_ms'] = (t_filt_end - t_filt_start) * 1000.0
        timings['end_to_end_ms'] = timings['total_classifier_ms'] + timings['filter_ms']
        self.prev_class_id = class_id

        meta = {
            'class_id': class_id,
            'class_name': class_name,
            'probabilities': {
                'stationary': float(probs[0]),
                'non_stationary': float(probs[1]),
                'impulsive': float(probs[2]),
            },
            'timings_ms': timings,
        }
        return processed_chunk, meta

    def process_file(self, input_wav_path, output_wav_path=None):
        audio, sr = sf.read(str(input_wav_path))
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        self.reset()
        clean_chunks = []
        meta_log = []

        for i in range(0, len(audio), self.chunk_size):
            chunk = audio[i:i + self.chunk_size]
            if len(chunk) < 64:
                clean_chunks.append(chunk)
                continue
            clean_chunk, meta = self.process_chunk(chunk)
            clean_chunks.append(clean_chunk)
            meta_log.append(meta)

        clean_audio = np.concatenate(clean_chunks)
        if output_wav_path:
            out_p = Path(output_wav_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_p), clean_audio, self.sr)
            print(f"Saved processed audio -> {out_p}")

        return clean_audio, meta_log
