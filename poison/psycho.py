"""Psychoacoustic masking threshold (MPEG-1 model 1 style).

Follows the formulation used in Qin et al. 2019, "Imperceptible, Robust, and
Targeted Adversarial Examples for Automatic Speech Recognition", which is built
on Lin & Abdulla's audio watermarking model. For every STFT frame of the clean
audio we find the tonal maskers, spread them across the Bark scale, and combine
them with the absolute threshold of hearing. A perturbation whose power stays
under that curve is hidden by the music itself.
"""

from __future__ import annotations

import numpy as np
import torch

N_FFT = 2048
HOP = 512


def bark(freq_hz: np.ndarray) -> np.ndarray:
    return 13 * np.arctan(0.00076 * freq_hz) + 3.5 * np.arctan((freq_hz / 7500.0) ** 2)


def absolute_threshold_db(freq_hz: np.ndarray) -> np.ndarray:
    """Threshold in quiet (dB SPL). Clipped where the formula blows up."""
    f = np.clip(freq_hz, 20.0, None) / 1000.0
    ath = 3.64 * f**-0.8 - 6.5 * np.exp(-0.6 * (f - 3.3) ** 2) + 1e-3 * f**4
    return np.clip(ath, None, 96.0)


def power_spectrum(wave: torch.Tensor) -> torch.Tensor:
    """Linear PSD, shape (bins, frames). Differentiable."""
    window = torch.hann_window(N_FFT, device=wave.device, dtype=wave.dtype)
    spec = torch.stft(wave, N_FFT, HOP, window=window, center=True, return_complex=True)
    return (8.0 / 3.0) * (spec.abs() / N_FFT) ** 2


def to_db_spl(psd: torch.Tensor, ref_max: float) -> torch.Tensor:
    """Map linear PSD to dB with the clean signal's peak pinned at 96 dB SPL."""
    return 96.0 - 10.0 * np.log10(ref_max) + 10.0 * torch.log10(psd + 1e-20)


def _frame_threshold(psd_db: np.ndarray, barks: np.ndarray, ath: np.ndarray) -> np.ndarray:
    # Tonal maskers: local maxima of the spectrum, excluding the edges.
    inner = psd_db[1:-1]
    peaks = np.where((inner > psd_db[:-2]) & (inner >= psd_db[2:]))[0] + 1
    if peaks.size == 0:
        return ath.copy()
    power = 10 * np.log10(
        10 ** (psd_db[peaks - 1] / 10) + 10 ** (psd_db[peaks] / 10) + 10 ** (psd_db[peaks + 1] / 10)
    )
    audible = power > ath[peaks]
    peaks, power = peaks[audible], power[audible]

    # Within 0.5 Bark only the strongest masker survives.
    keep: list[int] = []
    for i in range(peaks.size):
        if keep and barks[peaks[i]] - barks[peaks[keep[-1]]] < 0.5:
            if power[i] > power[keep[-1]]:
                keep[-1] = i
        else:
            keep.append(i)
    peaks, power = peaks[keep], power[keep]
    if peaks.size == 0:
        return ath.copy()

    # Spreading function: steep below the masker, level dependent above it.
    dz = barks[None, :] - barks[peaks][:, None]
    upper = -27.0 + 0.37 * np.maximum(power - 40.0, 0.0)
    spread = np.where(dz < 0, 27.0 * dz, upper[:, None] * dz)
    offset = -6.025 - 0.275 * barks[peaks]
    individual = power[:, None] + offset[:, None] + spread
    return 10 * np.log10(np.sum(10 ** (individual / 10), axis=0) + 10 ** (ath / 10))


def masking_threshold(clean: torch.Tensor, sample_rate: int) -> tuple[torch.Tensor, float]:
    """Return (threshold in dB SPL with shape (bins, frames), clean PSD peak)."""
    psd = power_spectrum(clean.detach().cpu().double())
    ref_max = max(float(psd.max()), 1e-20)
    psd_db = to_db_spl(psd, ref_max).numpy()
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sample_rate)
    barks, ath = bark(freqs), absolute_threshold_db(freqs)
    theta = np.stack(
        [_frame_threshold(psd_db[:, t], barks, ath) for t in range(psd_db.shape[1])], axis=1
    )
    return torch.from_numpy(theta).float(), ref_max


def masking_loss(delta: torch.Tensor, theta_db: torch.Tensor, ref_max: float, margin_db: float = 0.0):
    """Mean dB by which the perturbation pokes above the masking threshold.

    Returns (loss, fraction of time-frequency bins that are fully masked).
    """
    excess = to_db_spl(power_spectrum(delta), ref_max) - theta_db + margin_db
    masked = (excess <= margin_db).float().mean()
    return torch.relu(excess).mean(), masked


def project_under_mask(delta: torch.Tensor, theta_db: torch.Tensor, ref_max: float,
                       margin_db: float = 1.0, iters: int = 10) -> torch.Tensor:
    """Hard clip: attenuate every time-frequency bin that pokes above the threshold.

    Overlapping STFT frames leak into each other, so one pass is not exact; a few
    passes converge.
    """
    delta = delta.detach().clone()
    window = torch.hann_window(N_FFT, dtype=delta.dtype)
    for _ in range(iters):
        spec = torch.stft(delta, N_FFT, HOP, window=window, center=True, return_complex=True)
        level = to_db_spl((8.0 / 3.0) * (spec.abs() / N_FFT) ** 2, ref_max)
        excess = torch.relu(level - theta_db + margin_db)
        spec = spec * 10 ** (-excess / 20)
        delta = torch.istft(spec, N_FFT, HOP, window=window, center=True, length=len(delta))
    return delta
