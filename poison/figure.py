"""One 1080x1920 figure for the reel: waveform, spectrum under the mask, model verdict."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .psycho import N_FFT, masking_threshold, power_spectrum, to_db_spl

SURFACE, INK, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
CLEAN, POISON = "#3987e5", "#d95926"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=11, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _bars(ax, rows, color, title):
    names = [n for n, _ in rows][::-1]
    probs = [p for _, p in rows][::-1]
    ax.barh(names, probs, color=color, height=0.55)
    for y, p in enumerate(probs):
        ax.text(p + 0.02, y, f"{p:.0%}", va="center", color=INK, fontsize=12)
    ax.set_xlim(0, 1.15)
    ax.set_xticks([])
    ax.set_title(title, color=INK, fontsize=14, loc="left", pad=10)
    _style(ax)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(labelsize=12, colors=INK)


def render(path, clean, delta, sample_rate, labels_clean, labels_poisoned, snr_db, masked):
    clean_t, delta_t = torch.as_tensor(clean).float(), torch.as_tensor(delta).float()
    theta, ref = masking_threshold(clean_t, sample_rate)
    freqs = np.fft.rfftfreq(N_FFT, 1 / sample_rate)

    def avg_db(psd_db):  # average over time in the power domain
        return 10 * np.log10(np.mean(10 ** (psd_db.numpy() / 10), axis=1))

    spec_clean = avg_db(to_db_spl(power_spectrum(clean_t), ref))
    spec_delta = avg_db(to_db_spl(power_spectrum(delta_t), ref))
    spec_theta = avg_db(theta)

    fig = plt.figure(figsize=(9, 16), dpi=120, facecolor=SURFACE)
    grid = fig.add_gridspec(4, 1, height_ratios=[1, 1.5, 1, 1], hspace=0.45,
                            left=0.2, right=0.94, top=0.9, bottom=0.05)
    fig.text(0.06, 0.955, "Poison the well", color=INK, fontsize=26, weight="bold")
    fig.text(0.06, 0.93, f"Perturbation sits {snr_db:.0f} dB under the music, "
             f"{masked:.0%} of it below the masking threshold", color=MUTED, fontsize=13)

    ax = fig.add_subplot(grid[0])
    t = np.arange(len(clean)) / sample_rate
    ax.plot(t, clean, color=CLEAN, linewidth=0.6, label="Original audio")
    ax.plot(t, delta, color=POISON, linewidth=0.6, label="Added perturbation (same scale)")
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("seconds", color=MUTED, fontsize=11)
    ax.set_title("Waveform", color=INK, fontsize=14, loc="left", pad=10)
    _style(ax)
    ax.legend(frameon=False, labelcolor=INK, fontsize=11, loc="upper right", ncols=2,
              bbox_to_anchor=(1, 1.22))

    ax = fig.add_subplot(grid[1])
    keep = freqs >= 30
    ax.plot(freqs[keep], spec_clean[keep], color=CLEAN, linewidth=2, label="Original audio")
    ax.plot(freqs[keep], spec_theta[keep], color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)),
            label="Masking threshold (what ears can detect)")
    ax.plot(freqs[keep], spec_delta[keep], color=POISON, linewidth=2, label="Perturbation")
    ax.set_xscale("log")
    ax.set_xlim(30, sample_rate / 2)
    ax.set_ylim(-20, 100)
    ax.set_xticks([100, 1000, 10000], ["100 Hz", "1 kHz", "10 kHz"])
    ax.set_ylabel("dB SPL", color=MUTED, fontsize=11)
    ax.set_title("Spectrum, averaged over time", color=INK, fontsize=14, loc="left", pad=10)
    _style(ax)
    ax.legend(frameon=False, labelcolor=INK, fontsize=11, loc="lower left")

    _bars(fig.add_subplot(grid[2]), labels_clean, CLEAN, "What the model hears: original")
    _bars(fig.add_subplot(grid[3]), labels_poisoned, POISON, "What the model hears: poisoned")

    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
