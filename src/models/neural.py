import torch
import torch.nn as nn
import librosa


class WaveformTransientCNN(nn.Module):

    def __init__(self, num_classes=3, in_channels=1, base_channels=32, dropout=0.25):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=64, stride=16, padding=32, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.act1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(base_channels, base_channels * 2, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(base_channels * 2)
        self.act2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(base_channels * 2, base_channels * 4, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn3 = nn.BatchNorm1d(base_channels * 4)
        self.act3 = nn.SiLU()

        self.conv4 = nn.Conv1d(base_channels * 4, base_channels * 4, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn4 = nn.BatchNorm1d(base_channels * 4)
        self.act4 = nn.SiLU()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(base_channels * 4 * 2, 64),
            nn.SiLU(),
            nn.Dropout(p=dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + 1e-6
        x = (x - mean) / std

        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))

        feat_avg = torch.flatten(self.avg_pool(x), 1)
        feat_max = torch.flatten(self.max_pool(x), 1)
        pooled = torch.cat([feat_avg, feat_max], dim=1)

        return self.classifier(pooled)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class HighResLogMelTransform(nn.Module):

    def __init__(self, sr=22050, n_fft=256, hop_length=64, n_mels=64, f_min=20.0, f_max=11025.0, eps=1e-6):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.eps = eps

        fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=f_min, fmax=f_max)
        self.register_buffer('mel_filterbank', torch.from_numpy(fb).float())
        self.register_buffer('window', torch.hann_window(n_fft))

    def forward(self, audio):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(audio.device),
            center=True,
            pad_mode='reflect',
            return_complex=True
        )
        mag = torch.abs(stft)
        mel = torch.matmul(self.mel_filterbank.to(audio.device), mag)
        log_mel = torch.log(mel + self.eps)

        mean = log_mel.mean(dim=(-2, -1), keepdim=True)
        std = log_mel.std(dim=(-2, -1), keepdim=True) + self.eps
        norm_mel = (log_mel - mean) / std

        return norm_mel.unsqueeze(1)
