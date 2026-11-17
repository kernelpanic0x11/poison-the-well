"""Surrogate model: Audio Spectrogram Transformer trained on AudioSet.

The stock Hugging Face feature extractor runs outside autograd, so the front end
(resample, Kaldi filterbank, pad, normalise) is rebuilt here in differentiable
torch ops. That is what lets gradients flow from the model's opinion of the
audio all the way back to individual samples.
"""

from __future__ import annotations

import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio.functional as AF
from transformers import ASTForAudioClassification

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
MODEL_SR = 16_000
MAX_FRAMES = 1024  # 10.24 s
MAX_SECONDS = 10.0
FBANK_MEAN = -4.2677393
FBANK_STD = 4.5689974


class Surrogate:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = ASTForAudioClassification.from_pretrained(MODEL_ID).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.labels = self.model.config.id2label

    def features(self, wave: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Mono waveform in [-1, 1] at any sample rate -> (1, 1024, 128) model input."""
        wave = wave.cpu()  # the Kaldi front end is cheap and safest on CPU
        if sample_rate != MODEL_SR:
            wave = AF.resample(wave, sample_rate, MODEL_SR)
        fbank = kaldi.fbank(
            wave.unsqueeze(0),
            sample_frequency=MODEL_SR,
            htk_compat=True,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        frames = fbank.shape[0]
        if frames < MAX_FRAMES:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, MAX_FRAMES - frames))
        else:
            fbank = fbank[:MAX_FRAMES]
        fbank = (fbank - FBANK_MEAN) / (FBANK_STD * 2)
        return fbank.unsqueeze(0).to(self.device)

    def forward(self, wave: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (logits over 527 AudioSet labels, pooled embedding)."""
        out = self.model(input_values=self.features(wave, sample_rate), output_hidden_states=True)
        embedding = out.hidden_states[-1][0, :2].mean(dim=0)  # CLS + distillation tokens
        return out.logits[0], embedding

    @torch.no_grad()
    def describe(self, wave: torch.Tensor, sample_rate: int, k: int = 5) -> list[tuple[str, float]]:
        logits, _ = self.forward(wave, sample_rate)
        probs = torch.sigmoid(logits).cpu()
        top = torch.topk(probs, k)
        return [(self.labels[int(i)], float(p)) for p, i in zip(top.values, top.indices)]

    def label_id(self, name: str) -> int:
        for idx, label in self.labels.items():
            if label.lower() == name.lower():
                return int(idx)
        raise ValueError(f"'{name}' is not an AudioSet label. Try e.g. Speech, Dog, Rain, Silence.")
